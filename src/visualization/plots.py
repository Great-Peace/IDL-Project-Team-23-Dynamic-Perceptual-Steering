"""
plots.py
=========
All visualization functions for the Dynamic Perceptual Steering project.

Generates publication-quality figures following the style of
Gavrikov et al. (2025) and Geirhos et al. (2019):

1. Shape bias scatter plot (like Figure 2 in Gavrikov 2025)
2. Confidence distribution plots (like Figure 3 in Gavrikov 2025)
3. Accuracy vs. texture recovery tradeoff curve
4. Per-category heatmap
5. Famous vs. everyday split bar chart
6. APO optimization progress curve
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

logger = logging.getLogger(__name__)


# ── Style Configuration ──────────────────────────────────────────────────────

# Paper-quality styling
plt.rcParams.update({
    "figure.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.constrained_layout.use": True,
})

# Color palette for conditions
CONDITION_COLORS = {
    "neutral":          "#7f7f7f",   # grey
    "structural":       "#1f77b4",   # blue
    "cultural":         "#d62728",   # red
    "sequential":       "#2ca02c",   # green
    "apo_best":         "#ff7f0e",   # orange
    "clip_baseline":    "#9467bd",   # purple
    "humans":           "#8c564b",   # brown
}

# Marker shapes per condition (for accessibility)
CONDITION_MARKERS = {
    "neutral":      "o",
    "structural":   "s",
    "cultural":     "^",
    "sequential":   "D",
    "apo_best":     "*",
    "clip_baseline":"P",
    "humans":       "H",
}


class ResultsVisualizer:
    """
    Generates all plots for the Dynamic Perceptual Steering project.
    Saves figures to the configured figures directory.
    """

    def __init__(self, config: dict):
        self.config = config
        self.figures_dir = Path(config["paths"]["figures"])
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Figures will be saved to {self.figures_dir}")

    # ─────────────────────────────────────────────────────────────
    # Figure 1: Shape Bias Scatter Plot
    # ─────────────────────────────────────────────────────────────

    def plot_shape_bias_scatter(self, metrics_by_condition: Dict[str, Dict],
                                 title: str = "Shape vs Texture Bias Across Conditions",
                                 filename: str = "shape_bias_scatter.pdf"):
        """
        Scatter plot of shape bias per condition.

        X-axis: fraction of texture decisions
        Y-axis: fraction of shape decisions
        Each point = one experimental condition

        Replicates the visualization style of Figure 2 (Gavrikov 2025)
        but for African cultural context.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        for condition, metrics in metrics_by_condition.items():
            shape_acc = metrics.get("shape_accuracy", 0)
            texture_acc = metrics.get("texture_accuracy", 0)
            color = CONDITION_COLORS.get(condition, "#333333")
            marker = CONDITION_MARKERS.get(condition, "o")

            ax.scatter(
                texture_acc, shape_acc,
                color=color, marker=marker, s=150,
                label=condition, zorder=5,
                edgecolors="black", linewidths=0.5
            )

        # Diagonal line: equal shape and texture
        lims = [0, 1]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1,
                label="Equal bias")

        # Annotations
        ax.set_xlabel("Fraction of 'texture/cultural' decisions", fontsize=12)
        ax.set_ylabel("Fraction of 'shape' decisions", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

        # Region labels
        ax.text(0.02, 0.92, "Shape Bias", transform=ax.transAxes,
                fontsize=10, color="#444444", style="italic")
        ax.text(0.75, 0.08, "Texture Bias", transform=ax.transAxes,
                fontsize=10, color="#444444", style="italic")

        output_path = self.figures_dir / filename
        fig.savefig(str(output_path), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved shape bias scatter: {output_path}")

    # ─────────────────────────────────────────────────────────────
    # Figure 2: Confidence Distribution (Mechanistic Probing)
    # ─────────────────────────────────────────────────────────────

    def plot_confidence_distributions(self,
                                       probing_results: Dict[str, List[Dict]],
                                       filename: str = "confidence_distributions.pdf"):
        """
        Plot confidence score distributions for shape vs texture tokens
        across prompt conditions.

        Replicates Figure 3 of Gavrikov et al. (2025) for African context.

        Key finding to visualize:
        - Under neutral prompt: shape confidence ≈ 1.0, texture ≈ 0.0
          → Cultural information is suppressed
        - Under cultural prompt: texture confidence RISES
          → Cultural knowledge was always there, just needed activation
        """
        conditions = list(probing_results.keys())
        n_conditions = len(conditions)

        if n_conditions == 0:
            logger.warning("No probing results to plot.")
            return

        fig, axes = plt.subplots(1, n_conditions,
                                  figsize=(5 * n_conditions, 4),
                                  sharey=True)

        if n_conditions == 1:
            axes = [axes]

        for ax, condition in zip(axes, conditions):
            results = probing_results[condition]

            shape_confs = [r["shape_confidence"] for r in results]
            texture_confs = [r["texture_confidence"] for r in results]

            # Plot distributions
            ax.hist(shape_confs, bins=30, alpha=0.7, color="#1f77b4",
                    label="Shape token", density=True)
            ax.hist(texture_confs, bins=30, alpha=0.7, color="#d62728",
                    label="Texture/Cultural token", density=True)

            ax.set_title(f"{condition}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Token Confidence", fontsize=10)
            if ax == axes[0]:
                ax.set_ylabel("Density", fontsize=10)
            ax.legend(fontsize=8)

            # Add mean lines
            ax.axvline(np.mean(shape_confs), color="#1f77b4",
                       linestyle="--", linewidth=1.5, alpha=0.8)
            ax.axvline(np.mean(texture_confs), color="#d62728",
                       linestyle="--", linewidth=1.5, alpha=0.8)

        fig.suptitle("Token Confidence: Shape vs Cultural Label\n"
                     "(Near-zero texture confidence = perceptual erasure)",
                     fontsize=12, fontweight="bold")

        output_path = self.figures_dir / filename
        fig.savefig(str(output_path), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved confidence distributions: {output_path}")

    # ─────────────────────────────────────────────────────────────
    # Figure 3: Accuracy vs Cultural Recovery Tradeoff
    # ─────────────────────────────────────────────────────────────

    def plot_accuracy_tradeoff_curve(self,
                                      metrics_by_condition: Dict[str, Dict],
                                      filename: str = "accuracy_tradeoff.pdf"):
        """
        Plot the tradeoff between functional accuracy (shape recognition)
        and cultural recovery (texture/cultural recognition).

        This is your "perceptual over-steering" analysis.

        X-axis: shape/functional accuracy
        Y-axis: texture/cultural accuracy
        Each point = one experimental condition

        A good steering prompt should move UP and RIGHT:
        more cultural recognition WITHOUT losing functional accuracy.
        A bad one moves up but far LEFT: cultural gain at cost of function.
        """
        fig, ax = plt.subplots(figsize=(7, 6))

        for condition, metrics in metrics_by_condition.items():
            shape_acc = metrics.get("shape_accuracy", 0)
            texture_acc = metrics.get("texture_accuracy", 0)
            color = CONDITION_COLORS.get(condition, "#333333")
            marker = CONDITION_MARKERS.get(condition, "o")

            ax.scatter(
                shape_acc, texture_acc,
                color=color, marker=marker, s=200, zorder=5,
                edgecolors="black", linewidths=0.7
            )
            # Label each point
            ax.annotate(
                condition,
                (shape_acc, texture_acc),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=8, color=color
            )

        # Draw arrows showing direction of improvement
        ax.annotate("", xy=(0.9, 0.8), xytext=(0.5, 0.3),
                    arrowprops=dict(arrowstyle="->", color="green",
                                   lw=2, alpha=0.4))
        ax.text(0.65, 0.58, "Ideal direction", fontsize=9,
                color="green", alpha=0.6, rotation=42)

        # Minimum acceptable functional accuracy line
        min_acc = self.config["apo"].get("min_functional_accuracy", 0.75)
        ax.axvline(min_acc, color="orange", linestyle="--",
                   alpha=0.6, linewidth=1.5)
        ax.text(min_acc + 0.01, 0.05,
                f"Min acceptable\nfunctional acc ({min_acc})",
                fontsize=8, color="orange")

        ax.set_xlabel("Functional / Shape Accuracy", fontsize=12)
        ax.set_ylabel("Cultural / Texture Accuracy", fontsize=12)
        ax.set_title("Accuracy vs Cultural Recovery Tradeoff\n"
                     "(Perceptual Over-steering Analysis)",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        output_path = self.figures_dir / filename
        fig.savefig(str(output_path), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved tradeoff curve: {output_path}")

    # ─────────────────────────────────────────────────────────────
    # Figure 4: Famous vs Everyday Split (Insight 1)
    # ─────────────────────────────────────────────────────────────

    def plot_famous_vs_everyday(self,
                                 metrics_by_condition: Dict[str, Dict],
                                 filename: str = "famous_vs_everyday.pdf"):
        """
        Bar chart comparing texture accuracy for famous vs everyday
        African artifacts across all conditions.

        This directly visualizes your Insight 1 finding.
        """
        conditions = list(metrics_by_condition.keys())
        famous_texture = []
        everyday_texture = []

        for cond in conditions:
            m = metrics_by_condition[cond]
            famous_texture.append(
                m.get("famous_items", {}).get("texture_accuracy", 0)
            )
            everyday_texture.append(
                m.get("everyday_items", {}).get("texture_accuracy", 0)
            )

        x = np.arange(len(conditions))
        width = 0.35

        fig, ax = plt.subplots(figsize=(9, 5))

        bars1 = ax.bar(x - width/2, famous_texture, width,
                       label="Famous/Iconic items",
                       color="#2ca02c", alpha=0.85, edgecolor="black")
        bars2 = ax.bar(x + width/2, everyday_texture, width,
                       label="Everyday items",
                       color="#d62728", alpha=0.85, edgecolor="black")

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel("Prompt Condition", fontsize=12)
        ax.set_ylabel("Cultural/Texture Recognition Rate", fontsize=12)
        ax.set_title("Insight 1: Famous vs Everyday African Artifacts\n"
                     "Cultural Recognition by Prompt Condition",
                     fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=15)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.2)

        output_path = self.figures_dir / filename
        fig.savefig(str(output_path), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved famous vs everyday plot: {output_path}")

    # ─────────────────────────────────────────────────────────────
    # Figure 5: Per-Category Heatmap
    # ─────────────────────────────────────────────────────────────

    def plot_category_heatmap(self, metrics_by_condition: Dict[str, Dict],
                               metric: str = "texture_accuracy",
                               filename: str = "category_heatmap.pdf"):
        """
        Heatmap of texture accuracy per artifact category per condition.

        Shows which categories benefit most from cultural steering.
        """
        conditions = list(metrics_by_condition.keys())

        # Collect all categories
        all_categories = set()
        for m in metrics_by_condition.values():
            by_cat = m.get("by_category", {})
            all_categories.update(by_cat.keys())
        categories = sorted(all_categories)

        if not categories:
            logger.warning("No category breakdown data found.")
            return

        # Build matrix
        matrix = np.zeros((len(categories), len(conditions)))
        for j, cond in enumerate(conditions):
            by_cat = metrics_by_condition[cond].get("by_category", {})
            for i, cat in enumerate(categories):
                if cat in by_cat:
                    matrix[i, j] = by_cat[cat].get(metric, 0)

        fig, ax = plt.subplots(figsize=(max(8, len(conditions)*1.5),
                                        max(5, len(categories)*0.8)))

        sns.heatmap(
            matrix,
            xticklabels=conditions,
            yticklabels=categories,
            annot=True, fmt=".2f",
            cmap="RdYlGn",
            vmin=0, vmax=1,
            ax=ax,
            cbar_kws={"label": metric.replace("_", " ").title()}
        )

        ax.set_title(f"{metric.replace('_', ' ').title()} by Category and Condition",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Prompt Condition", fontsize=11)
        ax.set_ylabel("Artifact Category", fontsize=11)
        plt.xticks(rotation=20)

        output_path = self.figures_dir / filename
        fig.savefig(str(output_path), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved category heatmap: {output_path}")

    # ─────────────────────────────────────────────────────────────
    # Figure 6: APO Progress Curve
    # ─────────────────────────────────────────────────────────────

    def plot_apo_progress(self, apo_history: List[Dict],
                           filename: str = "apo_progress.pdf"):
        """
        Line plot showing APO optimization progress over iterations.

        X-axis: iteration number
        Y-axis: texture_accuracy and shape_accuracy

        Shows the best prompt discovered at each iteration.
        """
        if not apo_history:
            logger.warning("Empty APO history.")
            return

        df = pd.DataFrame(apo_history)
        df = df[df["texture_accuracy"].notna()]

        # Track best per iteration
        best_per_iter = []
        best_texture = -1
        for iteration in sorted(df["iteration"].unique()):
            iter_df = df[df["iteration"] == iteration]
            best_in_iter = iter_df.loc[iter_df["texture_accuracy"].idxmax()]
            if best_in_iter["texture_accuracy"] > best_texture:
                best_texture = best_in_iter["texture_accuracy"]
            best_per_iter.append({
                "iteration": iteration,
                "best_texture_acc": best_texture,
                "current_shape_acc": best_in_iter["shape_accuracy"],
            })

        best_df = pd.DataFrame(best_per_iter)

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(best_df["iteration"], best_df["best_texture_acc"],
                color="#d62728", linewidth=2.5, marker="o",
                label="Best texture/cultural accuracy")
        ax.plot(best_df["iteration"], best_df["current_shape_acc"],
                color="#1f77b4", linewidth=2, marker="s", linestyle="--",
                label="Shape/functional accuracy")

        # Shade the region where functional accuracy is acceptable
        min_acc = self.config["apo"].get("min_functional_accuracy", 0.75)
        ax.axhline(min_acc, color="orange", linestyle=":", alpha=0.7,
                   label=f"Min functional threshold ({min_acc})")

        # All evaluated points (grey scatter)
        ax.scatter(df["iteration"], df["texture_accuracy"],
                   alpha=0.2, s=20, color="grey", zorder=1)

        ax.set_xlabel("APO Iteration", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("APO Optimization Progress\n"
                     "(Maximizing Cultural Recognition)",
                     fontsize=12, fontweight="bold")
        ax.legend()
        ax.set_ylim(0, 1.05)

        output_path = self.figures_dir / filename
        fig.savefig(str(output_path), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved APO progress plot: {output_path}")

    def generate_all_figures(self, metrics_by_condition: Dict,
                              probing_results: Optional[Dict] = None,
                              apo_history: Optional[List] = None):
        """
        Generate all figures in one call.
        Call this at the end of all experiments.
        """
        logger.info("\nGenerating all figures...")

        self.plot_shape_bias_scatter(metrics_by_condition)
        self.plot_accuracy_tradeoff_curve(metrics_by_condition)
        self.plot_famous_vs_everyday(metrics_by_condition)
        self.plot_category_heatmap(metrics_by_condition)

        if probing_results:
            self.plot_confidence_distributions(probing_results)

        if apo_history:
            self.plot_apo_progress(apo_history)

        logger.info(f"All figures saved to {self.figures_dir}")
