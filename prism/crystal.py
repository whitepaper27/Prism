"""
T3 Crystal Memory -- failure-derived rules, trust starts at 0.50, grows to ~0.95.

Promotion pipeline:
  1. An agent action is rejected (RejectionEvent logged)
  2. Gemini analyzes the rejection and generates a structured CrystalRule
  3. The rule is stored in SQLite with trust_score = 0.5
  4. Each time the rule correctly blocks a future action, trust grows
  5. At T1_PROMOTION_THRESHOLD, the rule is flagged for human review -> T1

Trust formula (PRISM spec s12):
  correctly_blocked: trust = min(0.95, 0.5 + 0.1 * log(1 + evidence_count))
  false_positive:    trust = max(0.30, trust - 0.15)
  human_override:    trust = max(0.20, trust - 0.25)
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from math import log
from typing import List

from google import genai

from .models import CrystalRule, RejectionEvent
from .storage import PRISMStorage

TRUST_INITIAL = 0.50
TRUST_MATURE = 0.85
T1_PROMOTION_THRESHOLD = 0.90


class CrystalMemory:
    def __init__(self, storage: PRISMStorage, client: genai.Client, model_name: str):
        self.storage = storage
        self._client = client
        self._model_name = model_name

    def get_rules(self, domain: str) -> List[CrystalRule]:
        return self.storage.get_crystal_rules(domain)

    # -- Failure-to-rule promotion --------------------------------------------

    def promote_from_rejection(self, event: RejectionEvent) -> CrystalRule:
        """
        PRISM contribution 2: convert rejection event into a structured,
        policy-ranked Crystal rule. Every rule has provenance back to its
        origin rejection event.
        """
        self.storage.save_rejection_event(event)

        prompt = (
            f"You are a safety analysis system for an AI agent in domain: {event.domain}\n\n"
            f"An agent action was REJECTED. Generate a generalizable safety constraint rule.\n\n"
            f"REJECTION EVENT:\n"
            f"- Proposed action: {event.proposed_action}\n"
            f"- Rejection reason: {event.rejection_reason}\n"
            f"- Rejected by: {event.rejector}\n"
            f"- Context: {event.context}\n\n"
            f"Generate a Crystal rule as JSON. CRITICAL REQUIREMENTS:\n"
            f"1. constraint: Describe the GENERAL class of violation, not just this instance\n"
            f"2. blocked_patterns: Make patterns match SIMILAR future violations, not just this exact case\n"
            f"   - Cover the action verb AND the violation condition\n"
            f"   - Use alternation (a|b) to cover multiple phrasings\n"
            f"   - For numeric thresholds: match a RANGE, not just one value\n"
            f"   - Example for 'applied 30% discount when max is 10%':\n"
            f'     ["apply_discount.*(1[1-9]|[2-9]\\\\d|100)\\\\s*%", "(1[1-9]|[2-9]\\\\d|100)\\\\s*(percent|%).*discount"]\n'
            f"   - Example for 'modified shipped order item type':\n"
            f'     ["modify.*item.*(type|product).*(shipped|in.?transit|out.?for.?delivery)", "change.*(item|product).*(shipped|dispatched)"]\n'
            f"3. blocked_patterns must be valid Python regex strings (double-escape backslashes)\n\n"
            f"Output ONLY valid JSON (no markdown):\n"
            f'{{\n'
            f'  "constraint": "General constraint that prevents this class of failure",\n'
            f'  "blocked_patterns": ["pattern1", "pattern2", "pattern3"],\n'
            f'  "domain_context": "Why this matters in this domain"\n'
            f'}}'
        )

        for attempt in range(6):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                )
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 20 * (2 ** attempt)
                    print(f"      [crystal rate limit] waiting {wait}s...")
                    import time as _t; _t.sleep(wait)
                    if attempt == 5:
                        raise
                else:
                    raise
        raw = response.text.strip()

        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\n?```$", "", raw, flags=re.IGNORECASE)

        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            data = {
                "constraint": f"Avoid: {event.proposed_action[:200]}",
                "blocked_patterns": [],
                "domain_context": event.rejection_reason,
            }

        rule = CrystalRule(
            rule_id=str(uuid.uuid4()),
            domain=event.domain,
            constraint=data.get("constraint", ""),
            origin_rejection_id=event.rejection_id,
            blocked_patterns=data.get("blocked_patterns", []),
            evidence_count=0,
            trust_score=TRUST_INITIAL,
            created_session=event.session_id,
        )
        self.storage.save_crystal_rule(rule)
        # Index constraint text for FTS5 keyword matching
        self.storage.index_crystal_rule_fts(rule.rule_id, rule.constraint)
        return rule

    # -- Trust update ---------------------------------------------------------

    def update_trust(
        self, rule: CrystalRule, event_type: str, session_id: str
    ) -> CrystalRule:
        if event_type == "correctly_blocked":
            rule.evidence_count += 1
            rule.trust_score = min(0.95, 0.5 + 0.1 * log(1 + rule.evidence_count))
        elif event_type == "false_positive":
            rule.trust_score = max(0.30, rule.trust_score - 0.15)
        elif event_type == "human_override":
            rule.trust_score = max(0.20, rule.trust_score - 0.25)

        rule.last_triggered_session = session_id
        rule.updated_at = datetime.now().isoformat()

        if rule.trust_score >= T1_PROMOTION_THRESHOLD and not rule.promoted_to_t1:
            print(
                f"  [PRISM] Crystal rule {rule.rule_id[:8]}... nominated for T1 promotion "
                f"(trust={rule.trust_score:.3f}, evidence={rule.evidence_count})"
            )

        self.storage.update_crystal_rule(rule)
        return rule

    # -- Context formatting ---------------------------------------------------

    def format_for_context(self, rules: List[CrystalRule]) -> str:
        if not rules:
            return ""
        lines = [
            f"=== T3 CRYSTAL MEMORY (Trust: {TRUST_MATURE:.2f} mature -- FAILURE-DERIVED) ===",
        ]
        for r in rules:
            lines.append(
                f"[{r.rule_id[:8]}] (trust={r.trust_score:.2f}, "
                f"evidence={r.evidence_count}) {r.constraint}"
            )
        lines.append(
            "These rules were promoted from past failure events. "
            "They override Episodic suggestions on conflict."
        )
        return "\n".join(lines)
