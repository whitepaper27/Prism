"""
FigureAgent -- generates publication-quality figures for the PRISM paper.

Uses matplotlib for rendering. Also generates pgfplots LaTeX code
for IEEE Access submission.

Figures:
  Figure 2: Crystal trust growth curve (trust vs evidence count)
  Figure 3: Per-config accuracy bar chart (both domains)
  Figure 4: Repeat violation trajectory (C2 vs C4)
  Figure 5: LLM self-compliance vs resolver enforcement
"""

from __future__ import annotations

from pathlib import Path
from math import log

from .base import GeminiResearchAgent
from .analysis_agent import AnalysisAgent
from .prompts.figure_prompts import FIGURE_CAPTION

# matplotlib import deferred to method calls so the module loads
# even if matplotlib is not installed yet


class FigureAgent(GeminiResearchAgent):
    """Generates publication-quality figures for the PRISM paper."""

    def __init__(
        self,
        output_dir: str = "paper/figures",
        results_dir: str = "data",
        model_name: str = "gemini-2.5-flash",
    ):
        super().__init__(model_name=model_name)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis = AnalysisAgent(results_dir=results_dir, model_name=model_name)

    def generate_all(self) -> dict[str, Path]:
        """Generate all 4 missing figures. Returns {name: path}."""
        return {
            "trust_growth": self.trust_growth_curve(),
            "accuracy_bars": self.accuracy_bars(),
            "repeat_trajectory": self.repeat_trajectory(),
            "self_compliance": self.self_compliance(),
        }

    # -- Figure 2: Trust Growth Curve -----------------------------------------

    def trust_growth_curve(self) -> Path:
        """Crystal trust score vs evidence count with activation threshold."""
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")

        data = self.analysis.generate_figure_data("trust_growth")

        fig, ax = plt.subplots(figsize=(8, 5))

        # Theoretical curve
        xs = [p["evidence"] for p in data["theoretical_curve"]]
        ys = [p["trust"] for p in data["theoretical_curve"]]
        ax.plot(xs, ys, "b-", linewidth=2, label="Theoretical: min(0.95, 0.5 + 0.1 ln(1+n))")

        # Actual data points
        if data["actual_retail"]:
            rx = [p["evidence"] for p in data["actual_retail"]]
            ry = [p["trust"] for p in data["actual_retail"]]
            ax.scatter(rx, ry, c="green", marker="o", s=40, alpha=0.7, label="Retail Crystal rules")

        if data["actual_airline"]:
            ax_x = [p["evidence"] for p in data["actual_airline"]]
            ax_y = [p["trust"] for p in data["actual_airline"]]
            ax.scatter(ax_x, ax_y, c="orange", marker="^", s=40, alpha=0.7, label="Airline Crystal rules")

        # Reference lines
        ax.axhline(y=0.70, color="red", linestyle="--", alpha=0.7, label="Activation threshold (0.70)")
        ax.axhline(y=0.95, color="gray", linestyle=":", alpha=0.7, label="Cap (0.95)")
        ax.axhline(y=0.50, color="gray", linestyle=":", alpha=0.4, label="Initial trust (0.50)")

        ax.set_xlabel("Evidence Count (confirmed blocks)", fontsize=12)
        ax.set_ylabel("Trust Score", fontsize=12)
        ax.set_title("Crystal Rule Trust Growth", fontsize=14)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim(-1, 50)
        ax.set_ylim(0.35, 1.0)
        ax.grid(True, alpha=0.3)

        path = self.output_dir / "fig2_trust_growth.pdf"
        fig.savefig(str(path), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  [figure] Saved {path}")
        return path

    # -- Figure 3: Accuracy Bars ----------------------------------------------

    def accuracy_bars(self) -> Path:
        """Per-config task success rate for both domains."""
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")

        data = self.analysis.generate_figure_data("accuracy_bars")

        config_order = ["c0", "c1", "c2", "c3", "c4"]
        config_labels = ["C0\nNo Memory", "C1\nRAG-Only", "C2\nEpisodic", "C3\nPolicy+Ep", "C4\nFull PRISM"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        for ax, domain, title in [(ax1, "retail", "Retail (N=50)"), (ax2, "airline", "Airline (N=30)")]:
            means = []
            stds = []
            colors = []
            for cfg in config_order:
                if cfg in data.get(domain, {}):
                    stats = data[domain][cfg]
                    means.append(stats["mean"])
                    stds.append(stats["std"])
                else:
                    means.append(0)
                    stds.append(0)
                colors.append("#2196F3" if cfg != "c4" else "#4CAF50")

            bars = ax.bar(config_labels, means, yerr=stds, capsize=4,
                         color=colors, edgecolor="black", linewidth=0.5)
            ax.set_title(title, fontsize=13)
            ax.set_ylabel("Task Success Rate (%)" if domain == "retail" else "", fontsize=11)
            ax.set_ylim(75, 105)
            ax.grid(True, axis="y", alpha=0.3)

            # Value labels
            for bar, m, s in zip(bars, means, stds):
                label = f"{m:.1f}%"
                if s > 0:
                    label += f"\n+/-{s:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        label, ha="center", va="bottom", fontsize=8)

        fig.suptitle("PRISM C0-C4 Ablation: Task Success Rate", fontsize=14)
        fig.subplots_adjust(top=0.88, wspace=0.1)

        path = self.output_dir / "fig3_accuracy_bars.pdf"
        fig.savefig(str(path), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  [figure] Saved {path}")
        return path

    # -- Figure 4: Repeat Violation Trajectory --------------------------------

    def repeat_trajectory(self) -> Path:
        """Repeat violation rate comparison across configs."""
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        config_order = ["c0", "c1", "c2", "c3", "c4"]
        config_labels = ["C0", "C1", "C2", "C3", "C4"]

        for ax, domain, title in [(ax1, "retail", "Retail"), (ax2, "airline", "Airline")]:
            data = self.analysis.generate_figure_data("repeat_trajectory", domain)
            rates = [data.get(cfg, {}).get("repeat_rate", 0) for cfg in config_order]
            colors = ["#f44336" if r > 0 else "#4CAF50" for r in rates]

            bars = ax.bar(config_labels, rates, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_title(f"{title} -- Repeat Violation Rate", fontsize=13)
            ax.set_ylabel("Repeat Violation Rate (%)", fontsize=11)
            ax.set_ylim(0, max(30, max(rates) + 5))
            ax.grid(True, axis="y", alpha=0.3)

            for bar, r in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{r:.1f}%", ha="center", va="bottom", fontsize=10)

        fig.suptitle("Repeat Violation Rate by Configuration", fontsize=14)
        fig.subplots_adjust(top=0.88, wspace=0.3)

        path = self.output_dir / "fig4_repeat_trajectory.pdf"
        fig.savefig(str(path), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  [figure] Saved {path}")
        return path

    # -- Figure 5: Self-Compliance vs Resolver --------------------------------

    def self_compliance(self) -> Path:
        """LLM self-compliance rates showing advisory vs deterministic enforcement."""
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")

        data = self.analysis.generate_figure_data("self_compliance")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        config_order = ["c0", "c1", "c2", "c3", "c4"]
        config_labels = ["C0", "C1", "C2", "C3", "C4"]

        for ax, domain, title in [(ax1, "retail", "Retail"), (ax2, "airline", "Airline")]:
            rates = [data.get(domain, {}).get(cfg, 0) for cfg in config_order]
            colors = []
            for cfg in config_order:
                if cfg in ("c0", "c1", "c2"):
                    colors.append("#FF9800")  # advisory (orange)
                else:
                    colors.append("#2196F3")  # deterministic (blue)

            bars = ax.bar(config_labels, rates, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_title(f"{title} -- LLM Self-Compliance", fontsize=13)
            ax.set_ylabel("Self-Compliance Rate (%)" if domain == "retail" else "", fontsize=11)
            ax.set_ylim(70, 105)
            ax.grid(True, axis="y", alpha=0.3)

            for bar, r in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{r:.1f}%", ha="center", va="bottom", fontsize=10)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#FF9800", edgecolor="black", label="Advisory (C0-C2)"),
            Patch(facecolor="#2196F3", edgecolor="black", label="Deterministic (C3-C4)"),
        ]
        fig.legend(handles=legend_elements, loc="upper center", ncol=2,
                   fontsize=10, bbox_to_anchor=(0.5, 1.08))
        fig.suptitle("LLM Self-Compliance vs Deterministic Enforcement", fontsize=14)
        fig.subplots_adjust(top=0.82, wspace=0.1)

        path = self.output_dir / "fig5_self_compliance.pdf"
        fig.savefig(str(path), bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  [figure] Saved {path}")
        return path

    # -- Caption generation ---------------------------------------------------

    def generate_caption(self, figure_type: str, data_desc: str, takeaway: str) -> str:
        """Use Gemini to generate a LaTeX figure caption."""
        prompt = FIGURE_CAPTION.format(
            figure_type=figure_type,
            data_description=data_desc,
            takeaway=takeaway,
        )
        return self._call(prompt)
