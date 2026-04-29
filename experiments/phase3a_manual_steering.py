"""
phase3a_manual_steering.py
===========================
Phase 3A — Manual Dynamic Steering Evaluation

Runs the full Sequential Dual-Lens pipeline and all hand-crafted
prompt conditions across the entire dataset.

Conditions evaluated:
  1. structural  — shape/function steering
  2. cultural    — texture/culture steering
  3. sequential  — two-stage Dual-Lens
  4. cultural_geometric — alternative cultural prompt
  5. cultural_expert    — expert persona prompt

Computes improvement over neutral baseline for each condition.
Checks for perceptual over-steering.

Run on PSC:
    sbatch scripts/run_steering.slurm

Or locally:
    python experiments/phase3a_manual_steering.py --config configs/config.yaml
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.data.dataset_loader import load_all_datasets
from src.models.internvl_wrapper import InternVLWrapper
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import compute_all_metrics, compare_conditions
from src.steering.prompts import PromptLibrary
from src.steering.dual_lens import DualLensSteering
from src.visualization.plots import ResultsVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/phase3a_steering.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("="*60)
    logger.info("PHASE 3A — MANUAL DYNAMIC STEERING")
    logger.info("="*60)

    # ── Load Dataset ──────────────────────────────────────────
    dataset = load_all_datasets(config)
    if len(dataset) == 0:
        logger.error("Empty dataset. Run phase2_baseline.py first.")
        sys.exit(1)

    # ── Load Model & Prompts ──────────────────────────────────
    model = InternVLWrapper(config)
    prompts = PromptLibrary(config)
    evaluator = Evaluator(model, config)

    # ── Define All Conditions ─────────────────────────────────
    # Run all prompt conditions in one pass
    conditions = {
        "neutral":           prompts.get_text("neutral"),
        "structural":        prompts.get_text("structural"),
        "cultural":          prompts.get_text("cultural"),
        "cultural_geometric":prompts.get_text("cultural_geometric"),
        "cultural_expert":   prompts.get_text("cultural_expert"),
    }

    # Run all conditions
    all_results = evaluator.run_all_conditions(
        dataset=dataset,
        prompts=conditions
    )

    # ── Sequential Dual-Lens (separate pipeline) ──────────────
    logger.info("\nRunning Sequential Dual-Lens pipeline...")
    dual_lens = DualLensSteering(model, prompts, config)
    sequential_results = dual_lens.run_batch(dataset)

    # Add to all_results
    all_results["sequential"] = sequential_results

    # ── Compute All Metrics ───────────────────────────────────
    metrics_by_condition = {}
    for condition, results in all_results.items():
        metrics_by_condition[condition] = compute_all_metrics(results)

    # ── Statistical Comparisons ───────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("STATISTICAL COMPARISONS (vs neutral baseline)")
    logger.info("="*60)

    if "neutral" in all_results:
        for condition in ["cultural", "sequential", "cultural_expert"]:
            if condition in all_results:
                comparison = compare_conditions(
                    all_results["neutral"],
                    all_results[condition],
                    condition_a="neutral",
                    condition_b=condition
                )
                logger.info(f"\nNeutral vs {condition}:")
                logger.info(f"  {comparison['interpretation']}")
                logger.info(f"  shape_bias change: "
                            f"{comparison['difference']:+.4f}")

    # ── Perceptual Over-Steering Analysis ─────────────────────
    logger.info("\n" + "="*60)
    logger.info("PERCEPTUAL OVER-STEERING ANALYSIS")
    logger.info("="*60)

    neutral_shape_acc = metrics_by_condition.get("neutral", {}).get("shape_accuracy", 0)

    for condition, metrics in metrics_by_condition.items():
        if condition == "neutral":
            continue
        shape_drop = neutral_shape_acc - metrics.get("shape_accuracy", 0)
        texture_gain = (metrics.get("texture_accuracy", 0) -
                        metrics_by_condition["neutral"].get("texture_accuracy", 0))

        over_steering = shape_drop > 0.10  # >10% drop = over-steering risk

        logger.info(
            f"  {condition:<25} "
            f"shape_drop={shape_drop:+.3f}  "
            f"texture_gain={texture_gain:+.3f}  "
            f"{'⚠ OVER-STEERING RISK' if over_steering else '✓ OK'}"
        )

    # ── Save Summary ──────────────────────────────────────────
    summary = {
        "phase": "Phase 3A — Manual Steering",
        "conditions_evaluated": list(metrics_by_condition.keys()),
        "metrics": {
            cond: {k: v for k, v in m.items()
                   if k in ["shape_accuracy", "texture_accuracy",
                             "cue_accuracy", "shape_bias", "n_images"]}
            for cond, m in metrics_by_condition.items()
        }
    }

    summary_path = Path(config["paths"]["results"]) / "phase3a_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nPhase 3A summary saved to {summary_path}")

    # ── Generate All Figures ──────────────────────────────────
    logger.info("\nGenerating Phase 3A figures...")
    viz = ResultsVisualizer(config)
    viz.plot_shape_bias_scatter(metrics_by_condition)
    viz.plot_accuracy_tradeoff_curve(metrics_by_condition)
    viz.plot_famous_vs_everyday(metrics_by_condition)
    viz.plot_category_heatmap(metrics_by_condition)

    logger.info("\n✓ Phase 3A complete!")
    return all_results, metrics_by_condition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3A — Manual Steering")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
