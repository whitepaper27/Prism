#!/usr/bin/env python3
"""
PRISM Demo — SQL Tuning Domain
Walks through all five PRISM contributions in sequence.

Usage:
  1. Copy .env.example to .env and add your GEMINI_API_KEY
  2. pip install -r requirements.txt
  3. python demo_sql.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from prism import PRISMAgent
from prism.models import BlockedResult, EnrichedResult, ModifiedResult


# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    sys.exit(
        "ERROR: GEMINI_API_KEY not set.\n"
        "  1. Copy .env.example to .env\n"
        "  2. Add your key: GEMINI_API_KEY=your_key_here\n"
        "  3. Re-run: python demo_sql.py"
    )

DOMAIN = "sql_tuning"
POLICY_FILE = "data/policy_sql_tuning.yaml"
DB_PATH = "data/prism.db"


# ── Helpers ───────────────────────────────────────────────────────────────────

def sep(title: str, width: int = 64) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def show_resolution(result) -> None:
    if isinstance(result, BlockedResult):
        print(f"  ► RESOLUTION : BLOCKED [{result.tier}] trust={result.trust_score:.2f}")
        print(f"  ► Rule       : {result.reason_constraint.strip()}")
    elif isinstance(result, ModifiedResult):
        print(f"  ► RESOLUTION : MODIFIED [{result.tier}] trust={result.trust_score:.2f}")
        print(f"  ► Rule       : {result.reason_constraint.strip()}")
        print(f"  ► Suggested  : {result.modification}")
    else:
        print(f"  ► RESOLUTION : ENRICHED [{result.tier}]")
        print(f"  ► Episodic entries used: {len(result.episodic_context)}")
        for e in result.episodic_context:
            print(f"      [{e.source}] {e.content[:80]}...")


# ── Demo ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 64)
    print("  PRISM — Policy-Ranked Injection with Stratified Memory")
    print("  SQL Tuning Domain Demo")
    print("═" * 64)

    agent = PRISMAgent(
        api_key=GEMINI_API_KEY,
        domain=DOMAIN,
        policy_file=POLICY_FILE,
        db_path=DB_PATH,
    )

    # ─────────────────────────────────────────────────────────────────────────
    sep("STEP 1 — Contribution 1: T2 Episodic Memory (Architecture)")
    print("  Seeding T2 Episodic Memory with domain incident knowledge...\n")

    agent.add_episodic(
        content=(
            "Composite index on (customer_id, order_date) reduced ORDERS "
            "query latency by 40% in incident INC-2024-081."
        ),
        source="incident_report_INC-2024-081",
    )
    agent.add_episodic(
        content=(
            "ORDERS table receives ~50k inserts/hour during batch windows "
            "(00:00–06:00). Adding indexes during batch caused 3x latency spikes "
            "in sessions 12, 17, and 23."
        ),
        source="ops_runbook_orders_v3",
    )
    agent.add_episodic(
        content=(
            "PostgreSQL: always use CREATE INDEX CONCURRENTLY on tables >1M rows "
            "to avoid exclusive table locks. MySQL: use ALGORITHM=ONLINE."
        ),
        source="dba_guidelines_v3",
    )
    agent.add_episodic(
        content=(
            "IDX_ACCT_STATUS on ACCOUNTS is referenced by FK_ORDERS_ACCT. "
            "Dropping it would cascade-lock 12 tables."
        ),
        source="schema_fk_map_2024",
    )
    print("  Added 4 episodic entries (incident reports, runbooks, DBA guidelines).")

    # ─────────────────────────────────────────────────────────────────────────
    sep("STEP 2 — Contribution 3: T1 Policy Blocks Dangerous DDL (Deterministic Resolver)")
    task_blocked = (
        "The ORDERS table is slow on customer queries. "
        "Optimize it: drop the old IDX_ORDERS_STATUS index and "
        "create a new composite index on (customer_id, order_date)."
    )
    print(f"  Task: {task_blocked}\n")

    r1 = agent.run(task_blocked, session_id="session-001")
    print(f"  T1 rules loaded    : {r1['t1_rules_loaded']}")
    print(f"  T3 rules loaded    : {r1['t3_rules_loaded']}")
    print(f"  T2 entries retrieved: {r1['t2_entries_retrieved']}")
    print(f"  Context tokens ~   : {r1['token_estimate']}")
    show_resolution(r1["resolution"])
    print(f"\n  Gemini proposed    : {str(r1['parsed_action'].get('proposed_action', r1['gemini_raw']))[:120]}")
    print(f"\n  Full explanation:\n  {r1['resolution_explanation']}")

    # ─────────────────────────────────────────────────────────────────────────
    sep("STEP 3 — Contribution 2: Failure → Crystal Rule Promotion")
    print("  Simulating: Safety Mesh rejects agent's FK-index drop suggestion.\n")

    rej = agent.log_rejection(
        proposed_action="DROP INDEX IDX_ACCT_STATUS ON ACCOUNTS",
        rejection_reason=(
            "IDX_ACCT_STATUS is referenced by FK_ORDERS_ACCT foreign key. "
            "Dropping it would cascade-lock 12 tables and break referential integrity."
        ),
        rejector="safety_mesh_blast_radius_classifier",
        context=(
            "Agent was optimizing ACCOUNTS after a slow account-lookup query. "
            "It identified IDX_ACCT_STATUS as low-cardinality and proposed dropping it."
        ),
        session_id="session-001",
    )
    print(f"  Rejection logged   : {rej['rejection_id']}")
    print(f"  Crystal rule ID    : {rej['crystal_rule_id'][:8]}...")
    print(f"  Constraint         : {rej['constraint']}")
    print(f"  Blocked patterns   : {rej['blocked_patterns']}")
    print(f"  Initial trust score: {rej['trust_score']}")
    print(f"  Provenance link    : origin_rejection_id = {rej['origin_rejection_id']}")

    # ─────────────────────────────────────────────────────────────────────────
    sep("STEP 4 — Contribution 4: Crystal Rule Prevents Repeat Failure (Ablation Effect)")
    print("  Next session — same dangerous action attempted again...\n")

    task_repeat = (
        "The ACCOUNTS table queries are slow. IDX_ACCT_STATUS has very low "
        "cardinality. Should we drop it to reduce index maintenance overhead?"
    )
    print(f"  Task: {task_repeat}\n")

    r2 = agent.run(task_repeat, session_id="session-002")
    print(f"  T3 rules loaded    : {r2['t3_rules_loaded']}  ← Crystal rule from Step 3 is now active")
    show_resolution(r2["resolution"])
    print(f"\n  Full explanation:\n  {r2['resolution_explanation']}")

    # Show trust growth after the correct block
    print("\n  Trust score after correctly blocking repeat failure:")
    res = r2["resolution"]
    if isinstance(res, ModifiedResult):
        rules = agent._storage.get_crystal_rules(DOMAIN)
        for rule in rules:
            if rule.rule_id == res.reason_rule_id:
                print(f"    Rule {rule.rule_id[:8]}: trust={rule.trust_score:.3f}, evidence={rule.evidence_count}")

    # ─────────────────────────────────────────────────────────────────────────
    sep("STEP 5 — Contribution 1: Safe Task Enriched by T2 Episodic Context")
    task_safe = (
        "The ORDERS table has a slow query that filters by customer_id "
        "and order_date range. What index would help, and how should I create it safely?"
    )
    print(f"  Task: {task_safe}\n")

    r3 = agent.run(task_safe, session_id="session-003")
    show_resolution(r3["resolution"])
    print(f"\n  Agent recommendation:")
    action_text = r3["parsed_action"].get("proposed_action", r3["gemini_raw"])
    print(f"  {str(action_text)[:300]}")
    if r3["parsed_action"].get("safety_notes"):
        print(f"\n  Safety notes: {r3['parsed_action']['safety_notes']}")

    # ─────────────────────────────────────────────────────────────────────────
    sep("STEP 6 — Contribution 5: Token Efficiency vs Prompt Stuffing")
    print(f"  PRISM context tokens (Step 5)    : ~{r3['token_estimate']:,} words")
    print(f"  Without PRISM (all incidents stuffed): ~10,000+ words")
    print()
    print(f"  T1 policy rules (always loaded)  : {r3['t1_rules_loaded']} rules, compact, high-density")
    print(f"  T3 crystal rules (always loaded)  : {r3['t3_rules_loaded']} rules, evidence-backed, compact")
    print(f"  T2 episodic entries retrieved     : {r3['t2_entries_retrieved']} (query-time, relevant only)")
    print()
    print("  PRISM retrieves only what is relevant for this query.")
    print("  Policy and Crystal rules are compact by design — no retrieval overhead.")

    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  DEMO COMPLETE — Five PRISM Contributions Demonstrated")
    print("═" * 64)
    print("  1. Policy-ranked memory architecture (T1/T2/T3 tiers)    ✓  Step 1, 5")
    print("  2. Failure-to-rule promotion (Crystal Memory)             ✓  Step 3")
    print("  3. Deterministic conflict resolution                      ✓  Step 2, 4")
    print("  4. Crystal memory reduces repeat failures                 ✓  Step 4")
    print("  5. Token efficiency vs prompt stuffing                    ✓  Step 6")
    print()


if __name__ == "__main__":
    main()
