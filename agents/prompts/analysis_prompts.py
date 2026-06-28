"""Prompt templates for experiment analysis tasks."""

INTERPRET_RESULTS = """\
Analyze these PRISM ablation results for the paper Discussion section.

Results:
{results_table}

Key context:
- 720 total runs across retail (N=50) and airline (N=30)
- No pairwise accuracy differences are statistically significant (Fisher's exact, all p > 0.05)
- PRISM's contribution is enforcement semantics, not raw accuracy
- C2 (Reflexion) achieves 100% airline success -- must acknowledge honestly
- C4 is the ONLY config with 0% violation AND 0% repeat on retail

Write 2-3 paragraphs suitable for an academic paper Discussion section.
Be honest about limitations. Do not overclaim. ASCII only (no Unicode)."""

CRYSTAL_ANALYSIS = """\
Analyze these Crystal memory statistics from PRISM's C4 trials.

Crystal rules generated per trial: {rules_per_trial}
Trust score distribution: {trust_distribution}
Active rules (trust >= 0.70): {active_count}
Evidence counts: {evidence_distribution}

Context:
- Trust formula: min(0.95, 0.5 + 0.1 * ln(1 + evidence_count))
- Activation threshold: 0.70 (requires ~6 evidence events to cross)
- Initial trust: 0.50 (below Episodic's 0.70 -- must earn authority)
- Cap: 0.95 (Crystal NEVER reaches T1's 1.00)

Provide 1-2 paragraphs of analysis suitable for the Crystal Memory Statistics
subsection of the paper. Report numbers precisely. ASCII only."""

PER_GROUP_ANALYSIS = """\
Analyze this per-group breakdown of PRISM ablation results.

{group_table}

Task groups:
- t1_violation: direct policy violations (should be blocked by T1)
- subtle_violation: violations not covered by T1 regex (Crystal's target)
- safe: safe actions that should NOT be blocked
- repeat_t1: repeated T1 violations (should be caught every time)
- repeat_subtle: repeated subtle violations (Crystal should learn to catch)

Key question: does each config perform as expected on each group?
- C0/C1/C2 should miss some t1_violations (no deterministic enforcement)
- C3 should catch t1_violations but miss subtle_violations
- C4 should catch both (T1 + Crystal)
- All configs should allow safe actions (low over-block)

Provide analysis in 1-2 paragraphs. ASCII only."""
