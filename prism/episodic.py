"""
T2 Episodic Memory — retrieved at query time, trust = 0.70.

Stores past incidents, RAG-retrieved examples, and document-mined knowledge.
Retrieved via keyword search against the SQLite store. Lower trust than
Policy (T1) and Crystal (T3) — can be overridden by either.
"""

from __future__ import annotations

from typing import List

from .models import EpisodicEntry
from .storage import PRISMStorage

TRUST = 0.70


class EpisodicMemory:
    def __init__(self, storage: PRISMStorage):
        self.storage = storage

    def add(self, entry: EpisodicEntry) -> None:
        self.storage.add_episodic_entry(entry)

    def retrieve(self, query: str, domain: str, limit: int = 5) -> List[EpisodicEntry]:
        return self.storage.search_episodic(query, domain, limit)

    def format_for_context(self, entries: List[EpisodicEntry]) -> str:
        if not entries:
            return ""
        lines = [
            f"=== T2 EPISODIC MEMORY (Trust: {TRUST:.2f} — RAG-RETRIEVED) ===",
        ]
        for e in entries:
            lines.append(f"[{e.source}] {e.content}")
        lines.append(
            "Use this as context. Episodic memory has lower authority than Policy and Crystal rules."
        )
        return "\n".join(lines)
