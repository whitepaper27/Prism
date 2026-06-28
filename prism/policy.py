"""
T1 Policy Memory — human-curated, always loaded, trust = 1.00.

Loaded from a YAML file at agent startup. Never changes during a session
(only a human can edit the YAML). Policy rules block actions deterministically
via regex patterns — no LLM involvement in blocking decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from .models import PolicyRule

TRUST = 1.00


class PolicyMemory:
    def __init__(self, policy_file: str):
        self.policy_file = Path(policy_file)
        self.rules: List[PolicyRule] = []
        self._load()

    def _load(self) -> None:
        if not self.policy_file.exists():
            self.rules = []
            return
        with open(self.policy_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        domain = data.get("domain", "unknown")
        self.rules = [
            PolicyRule(
                rule_id=r["rule_id"],
                domain=domain,
                constraint=r["constraint"],
                description=r.get("description", ""),
                blocked_patterns=r.get("blocked_patterns", []),
            )
            for r in data.get("rules", [])
        ]

    def get_rules(self) -> List[PolicyRule]:
        return self.rules

    def format_for_context(self) -> str:
        if not self.rules:
            return ""
        lines = [
            f"=== T1 POLICY MEMORY (Trust: {TRUST:.2f} — DETERMINISTIC OVERRIDE) ===",
        ]
        for r in self.rules:
            lines.append(f"[{r.rule_id}] {r.constraint}")
        lines.append(
            "These rules BLOCK actions deterministically before LLM output is used."
        )
        return "\n".join(lines)
