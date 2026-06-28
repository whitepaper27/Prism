"""Prompt templates for figure caption generation."""

FIGURE_CAPTION = """\
Generate a LaTeX figure caption for this PRISM paper figure.

Figure type: {figure_type}
Data shown: {data_description}
Key takeaway: {takeaway}

Requirements:
- Caption should be 2-3 sentences
- First sentence: what the figure shows
- Second sentence: the key finding
- Third sentence (optional): important caveat or context
- Do not overclaim. State findings precisely.
- Use existing notation: C0-C4, T1/T2/T3, PRISM-Bench
- Output raw LaTeX (no markdown fences)

Example:
\\caption{{Crystal rule trust scores grow logarithmically with evidence count,
reaching the activation threshold (0.70) after approximately 6 confirmed blocks.
The 0.95 cap ensures Crystal rules never override human-curated Policy (T1) memory.}}"""
