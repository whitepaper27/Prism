"""
PRISM Conflict Resolver — deterministic, not LLM-decided.

Trust ordering: T1 (1.00) > T3 (active, >= 0.70) > T2 (0.70)

The algorithm:
  1. Check proposed action against T1 Policy rules (regex) → BLOCKED if match
  2. Filter T3 Crystal rules by activation threshold (>= 0.70)
  3. Check proposed action against active T3 rules (regex) → MODIFIED if match
  4. No conflict → ENRICHED with T2 episodic context for LLM reasoning

The LLM never decides which memory wins. The algorithm does.
This is the core of PRISM contribution 3 (deterministic conflict resolution).
"""

from __future__ import annotations

import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .storage import PRISMStorage

from .models import (
    BlockedResult,
    ConflictResult,
    CrystalRule,
    EpisodicEntry,
    EnrichedResult,
    ModifiedResult,
    PolicyRule,
)

logger = logging.getLogger(__name__)

# Crystal rules below this threshold are logged but do NOT block.
# However, inactive rules that MATCH still earn evidence (see resolve()).
# This avoids cold-start deadlock: rules at trust=0.50 can accumulate
# evidence through "shadow matching" until they cross 0.70 and activate.
CRYSTAL_ACTIVATION_THRESHOLD = 0.70

# Match modes for the resolver
MATCH_REGEX = "regex"    # Original: deterministic regex patterns only
MATCH_FTS5 = "fts5"      # FTS5 keyword overlap on constraint text


class ConflictResolver:
    def __init__(self, match_mode: str = MATCH_REGEX, storage: Optional["PRISMStorage"] = None):
        """
        match_mode: "regex" (deterministic pattern matching) or
                    "fts5" (keyword overlap on constraint text via SQLite FTS5)
        storage: required for FTS5 mode — provides the FTS5 query interface
        """
        self.match_mode = match_mode
        self._storage = storage
        if match_mode == MATCH_FTS5 and storage is None:
            raise ValueError("FTS5 match mode requires a PRISMStorage instance")

    def _rule_matches(self, rule, text: str, rule_type: str = "policy") -> bool:
        """Check if a rule matches the text using the configured match mode."""
        if self.match_mode == MATCH_REGEX:
            if rule_type == "policy":
                return rule.blocks(text)
            else:
                return rule.conflicts_with(text)
        else:
            # FTS5 mode: check keyword overlap between constraint and action
            if rule_type == "policy":
                matching_ids = self._storage.fts5_match_policy(text)
            else:
                matching_ids = self._storage.fts5_match_crystal(text)
            return rule.rule_id in matching_ids

    def resolve(
        self,
        proposed_action: str,
        t1_rules: List[PolicyRule],
        t3_rules: List[CrystalRule],
        t2_entries: List[EpisodicEntry],
        task_context: str = "",
    ) -> ConflictResult:
        """
        Deterministic trust-ordered conflict resolution.
        No LLM calls. No probabilistic decisions.

        Matching surface is proposed_action ONLY.
        Match mode is "regex" (pattern matching) or "fts5" (keyword overlap).
        """
        full_text = proposed_action.strip()

        # Step 1: T1 Policy check — highest trust, always active
        for rule in t1_rules:
            if self._rule_matches(rule, full_text, "policy"):
                return BlockedResult(
                    proposed_action=proposed_action,
                    reason_rule_id=rule.rule_id,
                    reason_constraint=rule.constraint,
                    tier="T1_POLICY",
                    trust_score=1.0,
                )

        # Step 2: T3 Crystal check — only ACTIVE rules (trust >= threshold)
        active_t3 = [r for r in t3_rules if r.trust_score >= CRYSTAL_ACTIVATION_THRESHOLD]
        inactive_t3 = [r for r in t3_rules if r.trust_score < CRYSTAL_ACTIVATION_THRESHOLD]

        # Shadow matching: inactive rules that match earn evidence without blocking.
        self._shadow_matched_rules: list[CrystalRule] = []
        for rule in inactive_t3:
            if self._rule_matches(rule, full_text, "crystal"):
                self._shadow_matched_rules.append(rule)
                logger.info(
                    "Inactive Crystal rule %s (trust=%.3f) shadow-matched "
                    "(below threshold %.2f) — earning evidence, not blocking",
                    rule.rule_id[:8], rule.trust_score, CRYSTAL_ACTIVATION_THRESHOLD,
                )

        # Sort active rules: highest trust > highest evidence > most restrictive
        active_t3.sort(
            key=lambda r: (r.trust_score, r.evidence_count, r.last_triggered_session or ""),
            reverse=True,
        )

        for rule in active_t3:
            if self._rule_matches(rule, full_text, "crystal"):
                return ModifiedResult(
                    original=proposed_action,
                    modification=rule.suggested_alternative,
                    reason_rule_id=rule.rule_id,
                    reason_constraint=rule.constraint,
                    tier="T3_CRYSTAL",
                    trust_score=rule.trust_score,
                )

        # Step 3: No conflict — enrich with T2 episodic context
        return EnrichedResult(
            action=proposed_action,
            episodic_context=t2_entries,
            tier="T2_EPISODIC",
        )

    def get_shadow_matched_rules(self) -> list[CrystalRule]:
        """Return inactive Crystal rules that matched on the last resolve() call.

        These rules matched the proposed action but lack the trust to block.
        Callers should update their trust (evidence event) so they can
        eventually cross the activation threshold. This breaks the cold-start
        deadlock where rules at trust=0.50 could never earn evidence.
        """
        return getattr(self, "_shadow_matched_rules", [])

    def explain(self, result: ConflictResult) -> str:
        if isinstance(result, BlockedResult):
            return (
                f"BLOCKED [{result.tier}] trust={result.trust_score:.2f}\n"
                f"  Rule [{result.reason_rule_id}]: {result.reason_constraint}"
            )
        elif isinstance(result, ModifiedResult):
            return (
                f"MODIFIED [{result.tier}] trust={result.trust_score:.2f}\n"
                f"  Rule [{result.reason_rule_id[:8]}]: {result.reason_constraint}\n"
                f"  -> {result.modification}"
            )
        else:
            n = len(result.episodic_context)
            return f"ENRICHED [{result.tier}] with {n} episodic entr{'y' if n == 1 else 'ies'}"


def should_promote_to_crystal(rejection_source: str) -> bool:
    """Crystal cascade prevention guard.

    Only promote from upstream rejections (safety mesh, judge, human, T1 policy).
    Never promote from Crystal-caused blocks — prevents false positive cascades.
    """
    ALLOWED_SOURCES = {"safety_mesh", "judge", "human", "t1_policy",
                       "ablation_ground_truth_checker"}

    if rejection_source in ALLOWED_SOURCES:
        return True

    if rejection_source in {"t3_crystal", "crystal_rule"}:
        logger.warning(
            "Cascade prevention: rejecting Crystal promotion from source '%s'",
            rejection_source,
        )
        return False

    # Unknown source — conservative: do not promote
    logger.warning(
        "Unknown rejection source '%s' — not promoting to Crystal",
        rejection_source,
    )
    return False


def count_crystal_stats(rules: List[CrystalRule]) -> dict:
    """Extract Crystal statistics for paper Table 8."""
    active = [r for r in rules if r.trust_score >= CRYSTAL_ACTIVATION_THRESHOLD]
    inactive = [r for r in rules if r.trust_score < CRYSTAL_ACTIVATION_THRESHOLD]

    return {
        "total_rules": len(rules),
        "active_rules": len(active),
        "inactive_rules": len(inactive),
        "mean_trust": sum(r.trust_score for r in rules) / max(len(rules), 1),
        "mean_evidence": sum(r.evidence_count for r in rules) / max(len(rules), 1),
        "trust_distribution": [r.trust_score for r in rules],
    }
