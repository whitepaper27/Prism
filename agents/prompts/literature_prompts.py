"""Prompt templates for literature review and positioning tasks."""

COMPARE_PAPER = """\
Compare this paper to PRISM's contributions.

Paper: {title} ({authors}, {year})
Abstract: {abstract}

PRISM's four distinguishing claims:
1. TRUST ORDERING: Deterministic T1 (1.00) > T3 (0.50-0.95) > T2 (0.70) ordering
   where higher-trust memory blocks lower-trust suggestions before tool execution.
2. FAILURE-TO-RULE PROMOTION: Rejected actions are converted into structured
   blocking rules (Crystal memory) with evidence-based trust scores -- not just
   reflective text storage (which Reflexion already does).
3. DETERMINISTIC CONFLICT RESOLUTION: The algorithm decides which memory wins,
   not the LLM. No probabilistic reasoning in the enforcement loop.
4. PRE-ACTION BLOCKING: Rules fire before tool execution, not as advisory context
   that the LLM may choose to ignore.

For each claim, assess: does this paper provide the same capability?

Output JSON only:
{{
  "paper_title": "{title}",
  "overlap": "none|partial|significant",
  "claim_assessment": {{
    "trust_ordering": {{"status": "present|absent|partial", "evidence": "..."}},
    "failure_promotion": {{"status": "present|absent|partial", "evidence": "..."}},
    "deterministic_resolution": {{"status": "present|absent|partial", "evidence": "..."}},
    "pre_action_blocking": {{"status": "present|absent|partial", "evidence": "..."}}
  }},
  "recommendation": "cite_and_differentiate|acknowledge_concurrent|revise_claim",
  "rationale": "Why this recommendation",
  "latex_paragraph": "LaTeX paragraph for Related Work section"
}}"""

NOVELTY_CHECK = """\
Quick novelty threat assessment for PRISM.

Paper abstract:
{abstract}

Does this paper threaten any of PRISM's four claims?
1. Deterministic trust ordering across memory tiers
2. Failure-to-rule promotion (not just reflective text)
3. Deterministic conflict resolution (algorithm, not LLM)
4. Pre-action blocking before tool execution

Answer briefly:
- Threat level: none / low / medium / high
- Which claims affected (if any)
- One-sentence recommendation"""

RELATED_WORK_UPDATE = """\
Generate an updated Related Work paragraph incorporating these new papers.

Existing Related Work (from paper/prism.tex Section 2):
{existing_related_work}

New papers to incorporate:
{new_papers}

PRISM's positioning sentence: "Where existing agent memory systems optimize
what to remember and when to retrieve, PRISM addresses which memory is
allowed to win."

Requirements:
- Output raw LaTeX (no markdown fences)
- Use \\cite{{}} for existing keys, \\cite{{new_key}} for new papers
- Maintain honest tone -- acknowledge overlap where it exists
- Keep the paragraph focused on differentiation, not exhaustive survey"""
