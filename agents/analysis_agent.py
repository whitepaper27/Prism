"""
AnalysisAgent -- programmatic analysis of PRISM's 720-run ablation results.

Uses stdlib (json, sqlite3, math) for computation, Gemini only for
natural language interpretation. Does NOT import from prism/ -- this
is tooling, not the research artifact.
"""

from __future__ import annotations

import json
import sqlite3
from math import log, sqrt
from pathlib import Path
from typing import Optional

from .base import GeminiResearchAgent
from .prompts.analysis_prompts import INTERPRET_RESULTS, CRYSTAL_ANALYSIS, PER_GROUP_ANALYSIS


class AnalysisAgent(GeminiResearchAgent):
    """Analyzes PRISM ablation results and Crystal DB statistics."""

    def __init__(self, results_dir: str = "data", model_name: str = "gemini-2.5-flash"):
        super().__init__(model_name=model_name)
        self.results_dir = Path(results_dir)

    # -- Data loading ---------------------------------------------------------

    def load_results(self, domain: str) -> list[dict]:
        """Load ablation_results_{domain}.json."""
        path = self.results_dir / f"ablation_results_{domain}.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_tasks(self, domain: str) -> list[dict]:
        """Load task definitions."""
        name = "tau_tasks.json" if domain == "retail" else f"tau_{domain}_tasks.json"
        path = self.results_dir / name
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -- Crystal statistics (fills Table 3) -----------------------------------

    def compute_crystal_stats(self, domain: str, n_trials: int = 5) -> dict:
        """Query Crystal SQLite DBs across trials.

        Returns:
            rules_generated: total Crystal rules across trials
            active_rules: rules with trust >= 0.70
            trust_scores: list of all trust scores
            evidence_counts: list of all evidence counts
            precision: active rules that correctly blocked / total active
        """
        db_prefix = "tau_retail" if domain == "retail" else "tau_airline"
        all_rules = []

        for trial in range(n_trials):
            db_path = self.results_dir / f"prism_trial_{trial}_{db_prefix}.db"
            if not db_path.exists():
                continue
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    "SELECT rule_id, constraint_text, evidence_count, "
                    "trust_score, created_session, promoted_to_t1 "
                    "FROM crystal_rules"
                ).fetchall()
                for row in rows:
                    all_rules.append({
                        "trial": trial,
                        "rule_id": row["rule_id"],
                        "constraint": row["constraint_text"],
                        "evidence_count": row["evidence_count"],
                        "trust_score": row["trust_score"],
                        "created_session": row["created_session"],
                        "promoted_to_t1": bool(row["promoted_to_t1"]),
                    })
            except sqlite3.OperationalError:
                pass  # table might not exist
            finally:
                conn.close()

        active = [r for r in all_rules if r["trust_score"] >= 0.70]
        trust_scores = [r["trust_score"] for r in all_rules]
        evidence_counts = [r["evidence_count"] for r in all_rules]

        # Precision: rules with evidence > 0 (they actually blocked something)
        active_with_evidence = [r for r in active if r["evidence_count"] > 0]
        precision = (len(active_with_evidence) / len(active)) if active else 0.0

        return {
            "domain": domain,
            "n_trials": n_trials,
            "rules_generated": len(all_rules),
            "rules_per_trial": len(all_rules) / max(n_trials, 1),
            "active_rules": len(active),
            "active_per_trial": len(active) / max(n_trials, 1),
            "precision": precision,
            "trust_mean": sum(trust_scores) / len(trust_scores) if trust_scores else 0,
            "trust_std": _std(trust_scores),
            "evidence_mean": sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0,
            "evidence_max": max(evidence_counts) if evidence_counts else 0,
            "all_rules": all_rules,
        }

    # -- Per-group breakdown --------------------------------------------------

    def per_group_breakdown(self, domain: str) -> dict:
        """Break down success rate by task group and config."""
        results = self.load_results(domain)
        groups = {}

        for r in results:
            group = r.get("group", "unknown")
            config = r.get("config", "unknown")
            key = (group, config)

            if key not in groups:
                groups[key] = {"total": 0, "correct": 0}
            groups[key]["total"] += 1
            if r.get("correct_outcome", False):
                groups[key]["correct"] += 1

        # Restructure as group -> config -> accuracy
        breakdown = {}
        for (group, config), counts in groups.items():
            if group not in breakdown:
                breakdown[group] = {}
            acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            breakdown[group][config] = {
                "accuracy": round(acc * 100, 1),
                "correct": counts["correct"],
                "total": counts["total"],
            }

        return breakdown

    # -- Figure data generation -----------------------------------------------

    def generate_figure_data(self, figure_name: str, domain: str = "retail") -> dict:
        """Generate data for the 4 missing paper figures."""
        if figure_name == "trust_growth":
            return self._trust_growth_data()
        elif figure_name == "accuracy_bars":
            return self._accuracy_bars_data()
        elif figure_name == "repeat_trajectory":
            return self._repeat_trajectory_data(domain)
        elif figure_name == "self_compliance":
            return self._self_compliance_data()
        else:
            raise ValueError(f"Unknown figure: {figure_name}")

    def _trust_growth_data(self) -> dict:
        """Trust score vs evidence count (theoretical curve + actual data)."""
        # Theoretical curve from formula
        theoretical = []
        for n in range(101):
            trust = min(0.95, 0.5 + 0.1 * log(1 + n))
            theoretical.append({"evidence": n, "trust": round(trust, 4)})

        # Actual data from Crystal DBs
        actual_retail = self.compute_crystal_stats("retail")
        actual_airline = self.compute_crystal_stats("airline")

        return {
            "theoretical_curve": theoretical,
            "actual_retail": [
                {"evidence": r["evidence_count"], "trust": r["trust_score"]}
                for r in actual_retail["all_rules"]
            ],
            "actual_airline": [
                {"evidence": r["evidence_count"], "trust": r["trust_score"]}
                for r in actual_airline["all_rules"]
            ],
            "activation_threshold": 0.70,
            "cap": 0.95,
        }

    def _accuracy_bars_data(self) -> dict:
        """Per-config accuracy for both domains."""
        data = {}
        for domain in ["retail", "airline"]:
            results = self.load_results(domain)
            configs = {}
            for r in results:
                cfg = r.get("config", "unknown")
                trial = r.get("trial", 0)
                key = (cfg, trial)
                if key not in configs:
                    configs[key] = {"total": 0, "correct": 0}
                configs[key]["total"] += 1
                if r.get("correct_outcome", False):
                    configs[key]["correct"] += 1

            # Aggregate by config
            config_stats = {}
            for (cfg, trial), counts in configs.items():
                if cfg not in config_stats:
                    config_stats[cfg] = []
                acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
                config_stats[cfg].append(acc * 100)

            data[domain] = {
                cfg: {
                    "mean": round(sum(vals) / len(vals), 1),
                    "std": round(_std(vals), 1),
                    "trials": vals,
                }
                for cfg, vals in config_stats.items()
            }

        return data

    def _repeat_trajectory_data(self, domain: str) -> dict:
        """Repeat violation rate per config."""
        results = self.load_results(domain)
        configs = {}

        for r in results:
            cfg = r.get("config", "unknown")
            group = r.get("group", "")
            if "repeat" not in group:
                continue
            if cfg not in configs:
                configs[cfg] = {"total": 0, "violations": 0}
            configs[cfg]["total"] += 1
            if r.get("policy_violated", False):
                configs[cfg]["violations"] += 1

        return {
            cfg: {
                "repeat_rate": round(
                    counts["violations"] / counts["total"] * 100, 1
                ) if counts["total"] > 0 else 0,
                "violations": counts["violations"],
                "total": counts["total"],
            }
            for cfg, counts in configs.items()
        }

    def _self_compliance_data(self) -> dict:
        """LLM self-compliance rates by config and domain."""
        data = {}
        for domain in ["retail", "airline"]:
            results = self.load_results(domain)
            configs = {}
            for r in results:
                cfg = r.get("config", "unknown")
                if cfg not in configs:
                    configs[cfg] = {"total": 0, "compliant": 0}
                configs[cfg]["total"] += 1
                if r.get("llm_self_compliant", False):
                    configs[cfg]["compliant"] += 1

            data[domain] = {
                cfg: round(
                    counts["compliant"] / counts["total"] * 100, 1
                ) if counts["total"] > 0 else 0
                for cfg, counts in configs.items()
            }

        return data

    # -- LLM interpretation ---------------------------------------------------

    def interpret_results(self, domain: str = "retail") -> str:
        """Use Gemini to generate natural-language interpretation."""
        breakdown = self.per_group_breakdown(domain)
        table_str = json.dumps(breakdown, indent=2)
        prompt = INTERPRET_RESULTS.format(results_table=table_str)
        return self._call(prompt)

    def interpret_crystal_stats(self, domain: str = "retail") -> str:
        """Use Gemini to analyze Crystal memory statistics."""
        stats = self.compute_crystal_stats(domain)
        prompt = CRYSTAL_ANALYSIS.format(
            rules_per_trial=f"{stats['rules_per_trial']:.1f}",
            trust_distribution=f"mean={stats['trust_mean']:.3f}, std={stats['trust_std']:.3f}",
            active_count=f"{stats['active_per_trial']:.1f} per trial ({stats['active_rules']} total)",
            evidence_distribution=f"mean={stats['evidence_mean']:.1f}, max={stats['evidence_max']}",
        )
        return self._call(prompt)


def _std(values: list[float]) -> float:
    """Population standard deviation (no numpy dependency)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return sqrt(variance)
