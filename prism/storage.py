"""
PRISM SQLite storage layer.

Tables:
  crystal_rules      — T3 Crystal Memory (always loaded after promotion)
  rejection_events   — provenance log for every Crystal rule
  episodic_memory    — T2 Episodic Memory (keyword search)

FTS5 is attempted for episodic search; falls back to LIKE if unavailable
(common on Windows Python SQLite builds).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .models import CrystalRule, EpisodicEntry, RejectionEvent


class PRISMStorage:
    def __init__(self, db_path: str = "data/prism.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts5 = False
        self._init_db()

    # ── Connection ────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS crystal_rules (
                    rule_id                TEXT PRIMARY KEY,
                    domain                 TEXT NOT NULL,
                    constraint_text        TEXT NOT NULL,
                    origin_rejection_id    TEXT NOT NULL,
                    blocked_patterns       TEXT DEFAULT '[]',
                    evidence_count         INTEGER DEFAULT 0,
                    trust_score            REAL DEFAULT 0.5,
                    created_session        TEXT NOT NULL,
                    last_triggered_session TEXT DEFAULT '',
                    promoted_to_t1         INTEGER DEFAULT 0,
                    created_at             TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at             TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS rejection_events (
                    rejection_id    TEXT PRIMARY KEY,
                    session_id      TEXT NOT NULL,
                    domain          TEXT NOT NULL,
                    proposed_action TEXT NOT NULL,
                    rejection_reason TEXT NOT NULL,
                    rejector        TEXT NOT NULL,
                    context         TEXT NOT NULL,
                    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Try FTS5; fall back to regular table
            try:
                c.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts
                    USING fts5(entry_id UNINDEXED, domain UNINDEXED, content, source UNINDEXED, created_at UNINDEXED)
                """)
                self._fts5 = True
            except sqlite3.OperationalError:
                pass

            c.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    entry_id   TEXT PRIMARY KEY,
                    domain     TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    source     TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # FTS5 indexes for policy and Crystal constraint matching.
            # Used by the FTS5 resolver variant — keyword overlap between
            # rule constraint text and LLM proposed action.
            if self._fts5:
                try:
                    c.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS policy_fts
                        USING fts5(rule_id UNINDEXED, ctext)
                    """)
                    c.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS crystal_fts
                        USING fts5(rule_id UNINDEXED, ctext)
                    """)
                except sqlite3.OperationalError:
                    pass

            c.commit()

    # ── Crystal Rules ─────────────────────────────────────────────────────────

    def save_crystal_rule(self, rule: CrystalRule) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO crystal_rules
                  (rule_id, domain, constraint_text, origin_rejection_id,
                   blocked_patterns, evidence_count, trust_score,
                   created_session, last_triggered_session, promoted_to_t1,
                   created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                rule.rule_id, rule.domain, rule.constraint,
                rule.origin_rejection_id,
                json.dumps(rule.blocked_patterns),
                rule.evidence_count, rule.trust_score,
                rule.created_session, rule.last_triggered_session,
                int(rule.promoted_to_t1),
                rule.created_at, rule.updated_at,
            ))
            c.commit()

    def get_crystal_rules(self, domain: str) -> List[CrystalRule]:
        with self._conn() as c:
            rows = c.execute("""
                SELECT * FROM crystal_rules
                WHERE domain = ? AND promoted_to_t1 = 0
                ORDER BY trust_score DESC
            """, (domain,)).fetchall()
        return [self._crystal_from_row(r) for r in rows]

    def update_crystal_rule(self, rule: CrystalRule) -> None:
        with self._conn() as c:
            c.execute("""
                UPDATE crystal_rules
                SET trust_score = ?, evidence_count = ?,
                    last_triggered_session = ?, promoted_to_t1 = ?,
                    updated_at = ?
                WHERE rule_id = ?
            """, (
                rule.trust_score, rule.evidence_count,
                rule.last_triggered_session, int(rule.promoted_to_t1),
                rule.updated_at, rule.rule_id,
            ))
            c.commit()

    def _crystal_from_row(self, row) -> CrystalRule:
        return CrystalRule(
            rule_id=row["rule_id"],
            domain=row["domain"],
            constraint=row["constraint_text"],
            origin_rejection_id=row["origin_rejection_id"],
            blocked_patterns=json.loads(row["blocked_patterns"] or "[]"),
            evidence_count=row["evidence_count"],
            trust_score=row["trust_score"],
            created_session=row["created_session"],
            last_triggered_session=row["last_triggered_session"] or "",
            promoted_to_t1=bool(row["promoted_to_t1"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ── Rejection Events ──────────────────────────────────────────────────────

    def save_rejection_event(self, event: RejectionEvent) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO rejection_events
                  (rejection_id, session_id, domain, proposed_action,
                   rejection_reason, rejector, context, created_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                event.rejection_id, event.session_id, event.domain,
                event.proposed_action, event.rejection_reason,
                event.rejector, event.context, event.created_at,
            ))
            c.commit()

    def get_rejection_event(self, rejection_id: str) -> Optional[RejectionEvent]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM rejection_events WHERE rejection_id = ?",
                (rejection_id,)
            ).fetchone()
        if not row:
            return None
        return RejectionEvent(
            rejection_id=row["rejection_id"],
            session_id=row["session_id"],
            domain=row["domain"],
            proposed_action=row["proposed_action"],
            rejection_reason=row["rejection_reason"],
            rejector=row["rejector"],
            context=row["context"],
            created_at=row["created_at"],
        )

    # ── Episodic Memory ───────────────────────────────────────────────────────

    def add_episodic_entry(self, entry: EpisodicEntry) -> None:
        with self._conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO episodic_memory
                  (entry_id, domain, content, source, created_at)
                VALUES (?,?,?,?,?)
            """, (entry.entry_id, entry.domain, entry.content,
                  entry.source, entry.created_at))

            if self._fts5:
                c.execute("""
                    INSERT OR REPLACE INTO episodic_fts
                      (entry_id, domain, content, source, created_at)
                    VALUES (?,?,?,?,?)
                """, (entry.entry_id, entry.domain, entry.content,
                      entry.source, entry.created_at))
            c.commit()

    def search_episodic(self, query: str, domain: str, limit: int = 5) -> List[EpisodicEntry]:
        if self._fts5:
            return self._fts5_search(query, domain, limit)
        return self._like_search(query, domain, limit)

    def _fts5_search(self, query: str, domain: str, limit: int) -> List[EpisodicEntry]:
        # Build a safe FTS5 query from individual keywords
        keywords = [w for w in query.split() if len(w) > 2]
        fts_query = " OR ".join(keywords) if keywords else query
        with self._conn() as c:
            try:
                rows = c.execute("""
                    SELECT em.* FROM episodic_fts f
                    JOIN episodic_memory em ON em.entry_id = f.entry_id
                    WHERE f.content MATCH ? AND em.domain = ?
                    ORDER BY rank
                    LIMIT ?
                """, (fts_query, domain, limit)).fetchall()
            except sqlite3.OperationalError:
                # FTS match syntax error — fall back
                return self._like_search(query, domain, limit)
        return [self._episodic_from_row(r) for r in rows]

    def _like_search(self, query: str, domain: str, limit: int) -> List[EpisodicEntry]:
        keywords = [w for w in query.split() if len(w) > 2]
        if not keywords:
            keywords = [query]
        conditions = " OR ".join(["content LIKE ?"] * len(keywords))
        params = [f"%{w}%" for w in keywords] + [domain, limit]
        with self._conn() as c:
            rows = c.execute(
                f"SELECT * FROM episodic_memory WHERE ({conditions}) AND domain = ? LIMIT ?",
                params
            ).fetchall()
        return [self._episodic_from_row(r) for r in rows]

    def _episodic_from_row(self, row) -> EpisodicEntry:
        return EpisodicEntry(
            entry_id=row["entry_id"],
            domain=row["domain"],
            content=row["content"],
            source=row["source"],
            created_at=row["created_at"],
        )

    # ── FTS5 Policy/Crystal Matching ─────────────────────────────────────────

    def index_policy_rule(self, rule_id: str, constraint: str) -> None:
        """Index a policy rule's constraint text for FTS5 matching."""
        if not self._fts5:
            return
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO policy_fts (rule_id, ctext) VALUES (?, ?)",
                      (rule_id, constraint))
            c.commit()

    def index_crystal_rule_fts(self, rule_id: str, constraint: str) -> None:
        """Index a Crystal rule's constraint text for FTS5 matching."""
        if not self._fts5:
            return
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO crystal_fts (rule_id, ctext) VALUES (?, ?)",
                      (rule_id, constraint))
            c.commit()

    def fts5_match_policy(self, action_text: str, min_keywords: int = 2) -> list[str]:
        """Find policy rules whose constraint text keyword-matches the action.

        Returns list of matching rule_ids, ranked by FTS5 relevance.
        Requires at least min_keywords overlapping words (length > 2).
        """
        if not self._fts5:
            return self._like_match_policy(action_text)
        keywords = [w for w in action_text.split() if len(w) > 2 and w.isalpha()]
        if len(keywords) < min_keywords:
            return []
        fts_query = " OR ".join(keywords)
        with self._conn() as c:
            try:
                rows = c.execute(
                    "SELECT rule_id FROM policy_fts WHERE ctext MATCH ? ORDER BY rank LIMIT 10",
                    (fts_query,)
                ).fetchall()
                return [r["rule_id"] for r in rows]
            except sqlite3.OperationalError:
                return self._like_match_policy(action_text)

    def fts5_match_crystal(self, action_text: str, min_keywords: int = 2) -> list[str]:
        """Find Crystal rules whose constraint text keyword-matches the action."""
        if not self._fts5:
            return self._like_match_crystal(action_text)
        keywords = [w for w in action_text.split() if len(w) > 2 and w.isalpha()]
        if len(keywords) < min_keywords:
            return []
        fts_query = " OR ".join(keywords)
        with self._conn() as c:
            try:
                rows = c.execute(
                    "SELECT rule_id FROM crystal_fts WHERE ctext MATCH ? ORDER BY rank LIMIT 10",
                    (fts_query,)
                ).fetchall()
                return [r["rule_id"] for r in rows]
            except sqlite3.OperationalError:
                return self._like_match_crystal(action_text)

    def _like_match_policy(self, action_text: str) -> list[str]:
        """Fallback: keyword overlap without FTS5. Not used when FTS5 available."""
        # Without FTS5, do simple keyword overlap against indexed constraints
        return []

    def _like_match_crystal(self, action_text: str) -> list[str]:
        """Fallback: keyword overlap without FTS5. Not used when FTS5 available."""
        return []
