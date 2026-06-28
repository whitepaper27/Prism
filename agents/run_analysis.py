#!/usr/bin/env python3
"""
PRISM Research Analysis Runner

Generates all missing paper artifacts:
  1. Crystal statistics for Table 3
  2. Figure 2: Trust growth curve
  3. Figure 3: Per-config accuracy bars
  4. Figure 4: Repeat violation trajectory
  5. Figure 5: Self-compliance comparison

Usage:
  python agents/run_analysis.py              # all artifacts
  python agents/run_analysis.py --stats      # Crystal stats only
  python agents/run_analysis.py --figures    # figures only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from agents.analysis_agent import AnalysisAgent
from agents.figure_agent import FigureAgent


def print_crystal_stats(agent: AnalysisAgent) -> None:
    """Print Crystal statistics for Table 3."""
    print("=" * 60)
    print("TABLE 3: Crystal Memory Statistics")
    print("=" * 60)

    for domain in ["retail", "airline"]:
        stats = agent.compute_crystal_stats(domain)
        print(f"\n  {domain.upper()} (across {stats['n_trials']} trials):")
        print(f"    Rules generated:   {stats['rules_generated']} total "
              f"({stats['rules_per_trial']:.1f} per trial)")
        print(f"    Active rules:      {stats['active_rules']} total "
              f"({stats['active_per_trial']:.1f} per trial)")
        print(f"    Precision:         {stats['precision']:.1%}")
        print(f"    Trust mean +/- std: {stats['trust_mean']:.3f} +/- {stats['trust_std']:.3f}")
        print(f"    Evidence mean:     {stats['evidence_mean']:.1f}")
        print(f"    Evidence max:      {stats['evidence_max']}")


def print_per_group(agent: AnalysisAgent) -> None:
    """Print per-group breakdown."""
    print("\n" + "=" * 60)
    print("PER-GROUP BREAKDOWN")
    print("=" * 60)

    for domain in ["retail", "airline"]:
        breakdown = agent.per_group_breakdown(domain)
        print(f"\n  {domain.upper()}:")
        for group in sorted(breakdown.keys()):
            configs = breakdown[group]
            print(f"\n    {group}:")
            for cfg in sorted(configs.keys()):
                data = configs[cfg]
                print(f"      {cfg:4s}: {data['accuracy']:5.1f}% "
                      f"({data['correct']}/{data['total']})")


def generate_figures(fig_agent: FigureAgent) -> None:
    """Generate all 4 missing paper figures."""
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    paths = fig_agent.generate_all()
    print("\n  Generated:")
    for name, path in paths.items():
        print(f"    {name}: {path}")


def main():
    args = set(sys.argv[1:])
    do_all = not args or "--all" in args
    do_stats = do_all or "--stats" in args
    do_figures = do_all or "--figures" in args

    analysis = AnalysisAgent(results_dir="data")

    if do_stats:
        print_crystal_stats(analysis)
        print_per_group(analysis)

    if do_figures:
        fig = FigureAgent(output_dir="paper/figures", results_dir="data")
        generate_figures(fig)

    print("\n[done]")


if __name__ == "__main__":
    main()
