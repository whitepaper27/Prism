"""
PRISM Agent -- orchestrates all three memory tiers and the conflict resolver.

Context injection order (from PRISM spec s12):
  1. Base system prompt (agent identity + output format)
  2. T1 Policy rules  (always loaded -- zero retrieval cost)
  3. T3 Crystal rules (always loaded after promotion -- zero retrieval cost)
  4. T2 Episodic results (retrieved at query time)
  5. Current task

Public API:
  agent.run(task, session_id)         -> process task through full PRISM pipeline
  agent.log_rejection(...)            -> promote failure to Crystal rule
  agent.add_episodic(content, source) -> seed T2 episodic memory
  agent.update_crystal_trust(...)     -> feedback on Crystal rule quality
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Optional

from google import genai

from .crystal import CrystalMemory
from .episodic import EpisodicMemory
from .models import (
    BlockedResult,
    EpisodicEntry,
    ModifiedResult,
    RejectionEvent,
)
from .policy import PolicyMemory
from .resolver import ConflictResolver, should_promote_to_crystal
from .storage import PRISMStorage

_BASE_SYSTEM_PROMPT = """\
You are a safe, production-aware AI agent operating under the PRISM memory architecture.

Your memory has three tiers:
  T1 POLICY   (trust=1.00) -- hard constraints; you CANNOT violate these
  T3 CRYSTAL  (trust=0.85) -- learned from past failures; treat as strong warnings
  T2 EPISODIC (trust=0.70) -- retrieved context; helpful but lower authority

Respond with a JSON object only (no markdown, no extra text):
{
  "analysis": "Your reasoning about the task and relevant memory",
  "proposed_action": "The specific, concrete action you recommend",
  "action_type": "query|ddl|dml|report|analysis|tool_call",
  "confidence": 0.0,
  "safety_notes": "Any safety considerations you identified"
}"""


class PRISMAgent:
    def __init__(
        self,
        api_key: str,
        domain: str,
        policy_file: str,
        db_path: str = "data/prism.db",
        model_name: str = "gemini-2.0-flash",
    ):
        self.domain = domain
        self._model_name = model_name
        self._client = genai.Client(api_key=api_key)

        self._storage = PRISMStorage(db_path)
        self._policy = PolicyMemory(policy_file)
        self._episodic = EpisodicMemory(self._storage)
        self._crystal = CrystalMemory(self._storage, self._client, model_name)
        self._resolver = ConflictResolver()

    # -- Public API -----------------------------------------------------------

    def run(self, task: str, session_id: Optional[str] = None) -> dict:
        """
        Full PRISM pipeline:
          load tiers -> build context -> call Gemini -> resolve conflict -> return
        """
        session_id = session_id or str(uuid.uuid4())

        t1_rules = self._policy.get_rules()
        t3_rules = self._crystal.get_rules(self.domain)
        t2_entries = self._episodic.retrieve(task, self.domain)

        context = self._build_context(t1_rules, t3_rules, t2_entries)
        full_prompt = f"{context}\n\n=== CURRENT TASK ===\n{task}"
        token_estimate = len(full_prompt.split())

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=full_prompt,
        )
        raw_text = response.text.strip()
        proposed_action, parsed = self._parse_response(raw_text)

        resolution = self._resolver.resolve(
            proposed_action, t1_rules, t3_rules, t2_entries, task_context=task
        )
        explanation = self._resolver.explain(resolution)

        if isinstance(resolution, ModifiedResult):
            matching = [r for r in t3_rules if r.rule_id == resolution.reason_rule_id]
            if matching:
                self._crystal.update_trust(matching[0], "correctly_blocked", session_id)

        return {
            "session_id": session_id,
            "task": task,
            "gemini_raw": raw_text,
            "parsed_action": parsed,
            "proposed_action": proposed_action,
            "resolution": resolution,
            "resolution_explanation": explanation,
            "t1_rules_loaded": len(t1_rules),
            "t3_rules_loaded": len(t3_rules),
            "t2_entries_retrieved": len(t2_entries),
            "token_estimate": token_estimate,
        }

    def log_rejection(
        self,
        proposed_action: str,
        rejection_reason: str,
        rejector: str,
        context: str,
        session_id: Optional[str] = None,
    ) -> dict:
        """Log rejection and promote to T3 Crystal rule via Gemini.

        Cascade prevention: only promotes from allowed sources (safety_mesh,
        judge, human, t1_policy, ablation_ground_truth_checker). Rejects
        Crystal-caused blocks to prevent false positive cascades.
        """
        if not should_promote_to_crystal(rejector):
            return {
                "rejection_id": None,
                "crystal_rule_id": None,
                "constraint": None,
                "blocked_patterns": [],
                "trust_score": 0.0,
                "origin_rejection_id": None,
                "cascade_prevented": True,
            }

        session_id = session_id or str(uuid.uuid4())
        event = RejectionEvent(
            rejection_id=str(uuid.uuid4()),
            session_id=session_id,
            domain=self.domain,
            proposed_action=proposed_action,
            rejection_reason=rejection_reason,
            rejector=rejector,
            context=context,
        )
        rule = self._crystal.promote_from_rejection(event)
        return {
            "rejection_id": event.rejection_id,
            "crystal_rule_id": rule.rule_id,
            "constraint": rule.constraint,
            "blocked_patterns": rule.blocked_patterns,
            "trust_score": rule.trust_score,
            "origin_rejection_id": rule.origin_rejection_id,
        }

    def add_episodic(self, content: str, source: str) -> None:
        """Seed T2 Episodic Memory."""
        entry = EpisodicEntry(
            entry_id=str(uuid.uuid4()),
            domain=self.domain,
            content=content,
            source=source,
        )
        self._episodic.add(entry)

    def update_crystal_trust(
        self, rule_id: str, event_type: str, session_id: Optional[str] = None
    ) -> Optional[dict]:
        session_id = session_id or str(uuid.uuid4())
        rules = self._storage.get_crystal_rules(self.domain)
        matching = [r for r in rules if r.rule_id == rule_id]
        if not matching:
            return None
        updated = self._crystal.update_trust(matching[0], event_type, session_id)
        return {
            "rule_id": updated.rule_id,
            "trust_score": updated.trust_score,
            "evidence_count": updated.evidence_count,
            "promoted_to_t1": updated.promoted_to_t1,
        }

    # -- Private helpers ------------------------------------------------------

    def _build_context(self, t1_rules, t3_rules, t2_entries) -> str:
        parts = [_BASE_SYSTEM_PROMPT]
        t1_ctx = self._policy.format_for_context()
        if t1_ctx:
            parts.append(t1_ctx)
        t3_ctx = self._crystal.format_for_context(t3_rules)
        if t3_ctx:
            parts.append(t3_ctx)
        t2_ctx = self._episodic.format_for_context(t2_entries)
        if t2_ctx:
            parts.append(t2_ctx)
        return "\n\n".join(parts)

    def _parse_response(self, raw: str) -> tuple:
        clean = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
        clean = re.sub(r"\n?```$", "", clean, flags=re.IGNORECASE)
        try:
            parsed = json.loads(clean.strip())
            proposed = parsed.get("proposed_action", raw)
        except json.JSONDecodeError:
            parsed = {"raw_response": raw}
            proposed = raw
        return proposed, parsed
