#!/usr/bin/env python3
"""
PRISM C0-C4 Ablation Study -- THREADED (8 workers)

Same experiment as run_ablation.py but parallelized:
  - C0-C3 run in parallel across configs AND domains (up to 8 threads)
  - C4 trials run in parallel (each trial is independent, fresh Crystal DB)
  - Within each C4 trial, tasks run sequentially (Crystal accumulates)

Usage:
  python run_ablation_parallel.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from prism.ablation import (
    AblationRunner, AblationResults, CheckpointManager,
    Task, CONFIGS, CONFIG_LABELS,
)
from prism.models import BlockedResult, ModifiedResult
from prism.crystal import CrystalMemory
from prism.storage import PRISMStorage

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    sys.exit("ERROR: GEMINI_API_KEY not set in .env")

N_TRIALS = 5
MAX_WORKERS = 8

# Thread-safe print
_print_lock = threading.Lock()
def tprint(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def load_tasks(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        Task(
            task_id=t["task_id"],
            group=t["group"],
            violation_type=t["violation_type"],
            user_request=t["user_request"],
            ground_truth=t["ground_truth"],
            expected_rule=t.get("expected_rule"),
            violation_keywords=t.get("violation_keywords", []),
            correct_response_keywords=t.get("correct_response_keywords", []),
        )
        for t in data["tasks"]
    ]


def make_runner(domain: str, policy_file: str, db_prefix: str,
                match_mode: str = "regex") -> AblationRunner:
    """Create a fresh AblationRunner for a domain (no checkpoint for parallel)."""
    return AblationRunner(
        api_key=GEMINI_API_KEY,
        domain=domain,
        policy_file=policy_file,
        db_path=f"data/ablation_{db_prefix}.db",
        model_name="gemini-2.5-flash",
        call_delay=0.5,  # reduced — rate limiter handles throttling
        checkpoint_manager=None,
        match_mode=match_mode,
    )


def run_config_tasks(domain: str, policy_file: str, db_prefix: str,
                     tasks: list[Task], config: str,
                     episodic_seeds: list[tuple[str, str]],
                     match_mode: str = "regex") -> list[dict]:
    """Run one config across all tasks. Returns list of result dicts."""
    runner = make_runner(domain, policy_file, db_prefix, match_mode=match_mode)
    for content, source in episodic_seeds:
        runner.seed_episodic([(content, source)])

    results = []
    for task in tasks:
        r = runner._run_one(task, config, trial=0)
        results.append(AblationRunner._result_to_dict(r))
        tprint(f"    [{domain[:3]}] {config} {task.task_id} "
               f"{'[OK]' if r.correct_outcome else '[XX]'}")
        time.sleep(0.5)
    return results


def run_c4_trial(domain: str, policy_file: str, db_prefix: str,
                 tasks: list[Task], trial: int,
                 episodic_seeds: list[tuple[str, str]],
                 match_mode: str = "regex") -> list[dict]:
    """Run one C4 trial across all tasks sequentially (Crystal accumulates)."""
    runner = make_runner(domain, policy_file, db_prefix, match_mode=match_mode)
    for content, source in episodic_seeds:
        runner.seed_episodic([(content, source)])

    trial_db = f"data/prism_trial_{trial}_{domain}.db"
    trial_storage = PRISMStorage(trial_db)
    trial_crystal = CrystalMemory(trial_storage, runner._client, runner.model_name)

    # Copy episodic entries
    for e in runner._all_episodic:
        trial_storage.add_episodic_entry(e)

    ablation_results = AblationResults()
    results = []
    for task in tasks:
        r = runner._run_c4(task, trial, trial_storage, trial_crystal, ablation_results)
        results.append(AblationRunner._result_to_dict(r))
        tprint(f"    [{domain[:3]}] C4-t{trial} {task.task_id} "
               f"{'[OK]' if r.correct_outcome else '[XX]'}")
        time.sleep(0.5)
    return results


# -- Domain configs --

DOMAINS = {
    "tau_retail": {
        "tasks_file": "data/tau_tasks.json",
        "policy_file": "data/policy_tau_retail.yaml",
        "db_prefix": "retail",
        "seeds": [
            ("Customer returned defective headphones within 7 days of delivery. "
             "Refund issued to original credit card. Resolution time: 2 mins.",
             "tau_retail_session_041"),
            ("Gold-tier customer exchanged blue jacket (outerwear) for red jacket "
             "(outerwear, same category confirmed). 10% loyalty discount applied.",
             "tau_retail_session_078"),
            ("Pending order modification (size S to M) processed successfully "
             "before shipment using modify_order(field=size).",
             "tau_retail_session_103"),
            ("Agent attempted to cancel shipped order -- rejected. "
             "Offered customer return-after-delivery as alternative.",
             "tau_retail_session_055"),
            ("Agent proposed 30% discount for silver-tier -- rejected by supervisor. "
             "Policy maximum is 10% for any tier. Agent offered 10% instead.",
             "tau_retail_session_201"),
            ("Agent tried to modify item type on shipped order -- rejected. "
             "Shipped orders cannot have item type changed. Agent offered exchange instead.",
             "tau_retail_session_202"),
            ("Customer requested store credit instead of refund to original credit card. "
             "Rejected -- refunds must go to original payment method.",
             "tau_retail_session_203"),
            ("Silver-tier customer attempted to use Gold-tier discount rate. "
             "Rejected -- discount rate must match customer's actual tier.",
             "tau_retail_session_204"),
        ],
    },
    "tau_airline": {
        "tasks_file": "data/tau_airline_tasks.json",
        "policy_file": "data/policy_tau_airline.yaml",
        "db_prefix": "airline",
        "seeds": [
            ("Economy passenger attempted to rebook 6 hours before departure. "
             "Request rejected -- economy tickets require 24+ hours notice for changes.",
             "tau_airline_session_101"),
            ("Customer requested cash refund for non-refundable ticket cancelled 48 hours before departure. "
             "Issued travel credit instead. Cash refunds require 72+ hours notice.",
             "tau_airline_session_102"),
            ("Customer attempted to transfer economy ticket to spouse. "
             "Rejected -- tickets are non-transferable. New ticket purchase required.",
             "tau_airline_session_103"),
            ("Business class passenger rebooked same-day without issue. "
             "Business and First class allow same-day changes.",
             "tau_airline_session_104"),
            ("Gold Elite member had first checked bag waived successfully. "
             "Gold and Platinum Elite receive one free checked bag.",
             "tau_airline_session_105"),
            ("Economy passenger requested free Business class upgrade. "
             "Rejected -- upgrades must be purchased or redeemed with miles.",
             "tau_airline_session_106"),
            ("Agent attempted courtesy upgrade during service recovery. "
             "Rejected -- no free upgrades policy applies regardless of reason.",
             "tau_airline_session_107"),
            ("Customer attempted to use companion's Gold Elite membership for baggage waiver. "
             "Rejected -- membership benefits apply to the account holder only.",
             "tau_airline_session_108"),
        ],
    },
}


def run_phase(domain_tasks, match_mode, label):
    """Run C0-C3 + C4 trials for both domains with given match_mode."""
    tprint(f"\n{'='*60}")
    tprint(f"  {label} (match_mode={match_mode})")
    tprint(f"{'='*60}")

    all_results: dict[str, list[dict]] = {}

    # C0-C3: no resolver for C0/C1/C2, so match_mode only affects C3
    tprint(f"\n  Phase 1: C0-C3 (parallel, {MAX_WORKERS} workers)")
    p1_start = time.time()

    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for domain, cfg in DOMAINS.items():
            for config in ["C0", "C1", "C2", "C3"]:
                key = f"{domain}_{config}"
                fut = pool.submit(
                    run_config_tasks,
                    domain, cfg["policy_file"], cfg["db_prefix"],
                    domain_tasks[domain], config, cfg["seeds"],
                    match_mode=match_mode,
                )
                futures[fut] = key

        for fut in as_completed(futures):
            key = futures[fut]
            try:
                results = fut.result()
                all_results[key] = results
                tprint(f"    [done] {key}: {len(results)} results")
            except Exception as e:
                tprint(f"    [ERROR] {key}: {e}")

    tprint(f"    Phase 1 done in {time.time() - p1_start:.0f}s")

    # C4 trials
    tprint(f"\n  Phase 2: C4 trials (parallel, {MAX_WORKERS} workers)")
    p2_start = time.time()

    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for domain, cfg in DOMAINS.items():
            for trial in range(N_TRIALS):
                key = f"{domain}_C4_trial{trial}"
                fut = pool.submit(
                    run_c4_trial,
                    domain, cfg["policy_file"], cfg["db_prefix"],
                    domain_tasks[domain], trial, cfg["seeds"],
                    match_mode=match_mode,
                )
                futures[fut] = key

        for fut in as_completed(futures):
            key = futures[fut]
            try:
                results = fut.result()
                all_results[key] = results
                tprint(f"    [done] {key}: {len(results)} results")
            except Exception as e:
                tprint(f"    [ERROR] {key}: {e}")

    tprint(f"    Phase 2 done in {time.time() - p2_start:.0f}s")

    return all_results


def save_results(all_results, suffix=""):
    """Save aggregated results per domain."""
    for domain, cfg in DOMAINS.items():
        db_prefix = cfg["db_prefix"]
        domain_results = []

        for config in ["C0", "C1", "C2", "C3"]:
            key = f"{domain}_{config}"
            if key in all_results:
                domain_results.extend(all_results[key])

        for trial in range(N_TRIALS):
            key = f"{domain}_C4_trial{trial}"
            if key in all_results:
                domain_results.extend(all_results[key])

        fname = f"data/ablation_results_{db_prefix}{suffix}.json"
        raw_path = Path(fname)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(domain_results, f, indent=2)
        tprint(f"  [{domain}] {len(domain_results)} results -> {raw_path}")


def main():
    t0 = time.time()

    # Load tasks
    domain_tasks = {}
    for domain, cfg in DOMAINS.items():
        tasks = load_tasks(cfg["tasks_file"])
        domain_tasks[domain] = tasks
        tprint(f"  [{domain}] {len(tasks)} tasks loaded")

    # ---- RUN 1: PRISM-regex (deterministic pattern matching) ----
    regex_results = run_phase(domain_tasks, "regex", "PRISM-regex")
    save_results(regex_results, "_regex")

    # ---- RUN 2: PRISM-FTS5 (keyword overlap matching) ----
    fts5_results = run_phase(domain_tasks, "fts5", "PRISM-FTS5")
    save_results(fts5_results, "_fts5")

    # Also save regex as the default results file (backwards compat)
    save_results(regex_results, "")

    total_calls = sum(len(v) for v in regex_results.values()) + sum(len(v) for v in fts5_results.values())
    elapsed = time.time() - t0
    tprint(f"\n== COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min) ==")
    tprint(f"  Total API calls: ~{total_calls}")
    tprint(f"  Workers: {MAX_WORKERS}")
    tprint(f"  Results:")
    tprint(f"    PRISM-regex: data/ablation_results_*_regex.json")
    tprint(f"    PRISM-FTS5:  data/ablation_results_*_fts5.json")
    tprint(f"\n  Next: python agents/run_analysis.py")


if __name__ == "__main__":
    main()
