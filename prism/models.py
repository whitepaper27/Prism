"""
PRISM data models — all tiers, conflict results, and events.
Trust ordering: T1 (1.00) > T3 (0.85) > T2 (0.70)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union


def _now() -> str:
    return datetime.now().isoformat()


# ── T1 Policy Memory ──────────────────────────────────────────────────────────

@dataclass
class PolicyRule:
    """Human-curated invariant safety rule. Trust = 1.00. Never changes."""
    rule_id: str
    domain: str
    constraint: str
    description: str
    blocked_patterns: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def blocks(self, action: str) -> bool:
        """Deterministic regex check — no LLM involved."""
        for pattern in self.blocked_patterns:
            if re.search(pattern, action, re.IGNORECASE):
                return True
        return False


# ── T3 Crystal Memory ─────────────────────────────────────────────────────────

@dataclass
class CrystalRule:
    """
    Failure-derived rule promoted from rejected agent actions.
    Trust starts at 0.5, grows with evidence up to ~0.95.
    Formally ranks above Episodic (T2) because it is domain-specific
    and evidence-backed, not general-corpus retrieved.
    """
    rule_id: str
    domain: str
    constraint: str
    origin_rejection_id: str
    blocked_patterns: List[str] = field(default_factory=list)
    evidence_count: int = 0
    trust_score: float = 0.5
    created_session: str = ""
    last_triggered_session: str = ""
    promoted_to_t1: bool = False
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def conflicts_with(self, action: str) -> bool:
        """Deterministic regex check — no LLM involved."""
        for pattern in self.blocked_patterns:
            if re.search(pattern, action, re.IGNORECASE):
                return True
        return False

    @property
    def suggested_alternative(self) -> str:
        return f"Constraint from failure history: {self.constraint}"


# ── T2 Episodic Memory ────────────────────────────────────────────────────────

@dataclass
class EpisodicEntry:
    """
    Past incident, retrieved example, or document-mined knowledge.
    Trust = 0.70. Retrieved at query time — not always in context.
    """
    entry_id: str
    domain: str
    content: str
    source: str
    created_at: str = field(default_factory=_now)


# ── Rejection Event ───────────────────────────────────────────────────────────

@dataclass
class RejectionEvent:
    """
    Records a rejected agent action. Feeds the Crystal promotion pipeline.
    Every CrystalRule must trace back to a RejectionEvent (provenance rule).
    """
    rejection_id: str
    session_id: str
    domain: str
    proposed_action: str
    rejection_reason: str
    rejector: str          # "safety_mesh" | "human" | "policy" | "crystal"
    context: str
    created_at: str = field(default_factory=_now)


# ── Conflict Resolution Results ───────────────────────────────────────────────

@dataclass
class BlockedResult:
    """T1 or T3 rule deterministically blocks the proposed action."""
    proposed_action: str
    reason_rule_id: str
    reason_constraint: str
    tier: str              # "T1_POLICY" | "T3_CRYSTAL"
    trust_score: float = 1.0


@dataclass
class ModifiedResult:
    """T3 Crystal rule conflicts — action must be modified."""
    original: str
    modification: str
    reason_rule_id: str
    reason_constraint: str
    tier: str = "T3_CRYSTAL"
    trust_score: float = 0.85


@dataclass
class EnrichedResult:
    """No conflict — action proceeds, enriched with T2 episodic context."""
    action: str
    episodic_context: List[EpisodicEntry]
    tier: str = "T2_EPISODIC"


ConflictResult = Union[BlockedResult, ModifiedResult, EnrichedResult]
