"""
LiteratureAgent -- compares papers against PRISM's contributions.

Uses Gemini to analyze paper abstracts/content and assess whether
they overlap with PRISM's four distinguishing claims.
"""

from __future__ import annotations

from .base import GeminiResearchAgent
from .prompts.literature_prompts import COMPARE_PAPER, NOVELTY_CHECK, RELATED_WORK_UPDATE


class LiteratureAgent(GeminiResearchAgent):
    """Compares papers to PRISM and generates Related Work content."""

    PRISM_CLAIMS = {
        "trust_ordering": "Deterministic T1 > T3 > T2 trust ordering",
        "failure_promotion": "Failure-to-rule promotion (Crystal memory)",
        "deterministic_resolution": "Deterministic conflict resolution (algorithm, not LLM)",
        "pre_action_blocking": "Pre-action blocking before tool execution",
    }

    KNOWN_SYSTEMS = {
        "reflexion": {
            "citation": "shinn2023reflexion",
            "has": [],
            "missing": list(PRISM_CLAIMS.keys()),
            "note": "C2 in ablation is the Reflexion baseline",
        },
        "memgpt": {
            "citation": "packer2023memgpt",
            "has": [],
            "missing": list(PRISM_CLAIMS.keys()),
            "note": "Virtual context management, no trust ordering",
        },
        "a-mem": {
            "citation": None,
            "has": [],
            "missing": list(PRISM_CLAIMS.keys()),
            "note": "Dynamic memory network, no policy authority",
        },
        "zep": {
            "citation": None,
            "has": [],
            "missing": list(PRISM_CLAIMS.keys()),
            "note": "Temporal knowledge graphs, no deterministic override",
        },
    }

    def compare_paper(
        self,
        title: str,
        authors: str,
        year: str,
        abstract: str,
    ) -> dict:
        """Analyze a paper against PRISM's 4 claims. Returns structured assessment."""
        prompt = COMPARE_PAPER.format(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
        )
        return self._call_json(prompt)

    def check_novelty(self, abstract: str) -> str:
        """Quick threat assessment -- does this paper overlap with PRISM?"""
        prompt = NOVELTY_CHECK.format(abstract=abstract)
        return self._call(prompt)

    def update_related_work(
        self,
        existing_related_work: str,
        new_papers: list[dict],
    ) -> str:
        """Generate updated Related Work LaTeX incorporating new papers."""
        papers_str = "\n\n".join(
            f"Title: {p['title']}\nAuthors: {p['authors']}\n"
            f"Year: {p['year']}\nAbstract: {p['abstract']}"
            for p in new_papers
        )
        prompt = RELATED_WORK_UPDATE.format(
            existing_related_work=existing_related_work,
            new_papers=papers_str,
        )
        return self._call(prompt)

    def batch_compare(self, papers: list[dict]) -> list[dict]:
        """Compare multiple papers sequentially. Returns list of assessments."""
        results = []
        for p in papers:
            print(f"  [lit] Comparing: {p['title'][:60]}...")
            result = self.compare_paper(
                title=p["title"],
                authors=p.get("authors", ""),
                year=p.get("year", ""),
                abstract=p.get("abstract", ""),
            )
            results.append(result)
        return results
