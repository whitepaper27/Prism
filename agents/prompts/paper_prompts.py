"""Prompt templates for paper writing tasks."""

NON_NEGOTIABLE_RULES = """\
NON-NEGOTIABLE RULES (from CLAUDE.md Section 13):
1. Never claim "no existing system uses failure memory." Reflexion does.
2. Never frame PRISM as "three-tier memory." Frame as "policy-ordered memory with failure-to-rule promotion and deterministic conflict resolution."
3. Never reuse Sentri results as PRISM results.
4. Crystal rules must have provenance -- every rule links to its origin rejection event.
5. Trust ordering is deterministic. Never say "the LLM decides which memory to trust."
6. The paper sells safety, not personalization.
7. Three tiers only. No fourth tier.
8. All comparisons must include Reflexion as a baseline (C2).
9. Freeze benchmark reference before running. Record exact commit hash.
10. The 7-step pilot is mechanism validation, not the paper's evidence. The C0-C4 ablation is the evidence.
11. Report mean +/- std. No single-run results in the main table.
12. Distinguish Protocol A (train/test for Table 2) from Protocol B (sequential for Figures 3-4).
13. Never claim PRISM "guarantees accuracy" or "guarantees safety." PRISM guarantees only deterministic rule enforcement for encoded rules.
14. Acknowledge C1 (RAG-only) performance honestly. PRISM's value over RAG is deterministic enforcement, not raw accuracy."""

FINAL_RESULTS = """\
FINAL RESULTS (720 runs, 5 C4 trials per domain):

Retail (N=50):
  C0 No Memory:      86.0% success, 17.6% violation, 22.2% repeat, 0.0% over-block
  C1 RAG-Only:       94.0% success,  8.8% violation, 22.2% repeat, 0.0% over-block
  C2 Episodic:       94.0% success,  5.9% violation, 22.2% repeat, 0.0% over-block
  C3 Policy+Episodic: 96.0% success,  2.9% violation, 11.1% repeat, 6.2% over-block
  C4 Full PRISM:     96.0% +/- 1.8% success, 0.0% violation, 0.0% repeat, 6.2% over-block

Airline (N=30):
  C0 No Memory:      96.7% success,  0.0% violation, 0.0% repeat, 0.0% over-block
  C1 RAG-Only:       96.7% success,  0.0% violation, 0.0% repeat, 0.0% over-block
  C2 Episodic:      100.0% success,  0.0% violation, 0.0% repeat, 0.0% over-block
  C3 Policy+Episodic: 96.7% success,  0.0% violation, 0.0% repeat, 0.0% over-block
  C4 Full PRISM:     96.7% +/- 2.1% success, 4.5% violation, 12.5% repeat, 0.0% over-block

Statistical significance: Fisher's exact, all p > 0.05 (no significant pairwise differences).
PRISM's contribution is enforcement semantics, not accuracy improvement."""


SECTION_WRITER = """\
You are writing Section {section_num} ({section_name}) of the PRISM paper.

Current paper state:
{current_latex}

Data available:
{data_summary}

Requirements:
- Generate LaTeX that integrates with the existing paper structure
- Use existing \\cite{{}} keys: {available_citations}
- Never overclaim -- PRISM's contribution is enforcement semantics, not accuracy

{rules}

Write the section now. Output raw LaTeX only (no markdown fences)."""

ABSTRACT_REFINER = """\
Refine this abstract to fit within {char_limit} characters while preserving all key claims.

Current abstract ({current_chars} chars):
{abstract}

Constraints:
- Must mention: deterministic enforcement, Crystal memory, trust ordering
- Must acknowledge: RAG achieves 94-98%, Reflexion achieves 92-100%
- Must NOT claim: PRISM guarantees safety, PRISM beats all baselines
- Must state: enforcement semantics change, not accuracy improvement
- Target: arXiv (1920 char limit) or IEEE Access (250 word limit)

{rules}

Output the refined abstract only (no commentary)."""

REVIEWER_RESPONSE = """\
Draft a response to this reviewer comment for the PRISM paper.

Reviewer comment:
{comment}

Available data to support response:
{data}

{results}

Tone: respectful, evidence-based, honest about limitations.
If the reviewer is correct, acknowledge it and describe the fix.
If the reviewer misunderstands, clarify with specific data from our results.

{rules}

Output the response only."""
