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

To run only specific (missing) conditions without re-running completed ones:
    python experiments/phase3a_manual_steering.py --config configs/config.yaml \
        --conditions cultural cultural_geometric cultural_expert sequential
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

ALL_CONDITIONS = ["neutral", "structural", "cultural", "cultural_geometric",
                  "cultural_expert", "sequential"]


def load_existing_results(condition: str, results_dir: Path):
    """
    Load the most recent complete (no-error) result file for a condition.
    Returns the list of result dicts, or None if nothing usable is found.
    """
    # Phase 2 neutral lives under a different prefix
    prefixes = ["phase2_"] if condition == "neutral" else [""]

    for prefix in prefixes:
        candidates = sorted(
            results_dir.glob(f"{prefix}{condition}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            try:
                with open(path) as f:
                    data = json.load(f)
                if not isinstance(data, list) or not data:
                    continue
                real = [r for r in data if r.get("response", "").strip()]
                if len(real) == len(data):  # all records have real responses
                    logger.info(
                        f"  Loaded existing {condition} results from {path.name} "
                        f"({len(data)} records)"
                    )
                    return data
            except Exception:
                continue
    return None


def main(config_path: str, run_conditions: list):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config["paths"]["results"])

    logger.info("=" * 60)
    logger.info("PHASE 3A — MANUAL DYNAMIC STEERING")
    logger.info("=" * 60)

    # ── Load Dataset ──────────────────────────────────────────
    dataset = load_all_datasets(config)
    if len(dataset) == 0:
        logger.error("Empty dataset. Run phase2_baseline.py first.")
        sys.exit(1)

    # ── Determine which conditions need inference ─────────────
    need_inference = [c for c in run_conditions if c != "sequential"]
    need_sequential = "sequential" in run_conditions

    all_results = {}

    # Load already-complete conditions from disk (skip re-running them)
    for condition in ALL_CONDITIONS:
        if condition in run_conditions:
            continue  # will run below
        existing = load_existing_results(condition, results_dir)
        if existing:
            all_results[condition] = existing

    # ── Load Model only if inference is actually needed ───────
    model = None
    if need_inference or need_sequential:
        model = InternVLWrapper(config)

    prompts = PromptLibrary(config)

    # ── Run inference for requested non-sequential conditions ─
    if need_inference:
        evaluator = Evaluator(model, config)
        conditions_to_run = {
            c: prompts.get_text(c)
            for c in need_inference
            if c in ["neutral", "structural", "cultural",
                     "cultural_geometric", "cultural_expert"]
        }
        new_results = evaluator.run_all_conditions(
            dataset=dataset,
            prompts=conditions_to_run,
        )
        all_results.update(new_results)

    # ── Sequential Dual-Lens ──────────────────────────────────
    if need_sequential:
        logger.info("\nRunning Sequential Dual-Lens pipeline...")
        dual_lens = DualLensSteering(model, prompts, config)
        all_results["sequential"] = dual_lens.run_batch(dataset)

    # ── Compute All Metrics ───────────────────────────────────
    metrics_by_condition = {
        condition: compute_all_metrics(results)
        for condition, results in all_results.items()
    }

    # ── Statistical Comparisons ───────────────────────────────
    if "neutral" in all_results:
        logger.info("\n" + "=" * 60)
        logger.info("STATISTICAL COMPARISONS (vs neutral baseline)")
        logger.info("=" * 60)
        for condition in ["cultural", "sequential", "cultural_expert"]:
            if condition not in all_results:
                continue
            comparison = compare_conditions(
                all_results["neutral"],
                all_results[condition],
                condition_a="neutral",
                condition_b=condition,
            )
            logger.info(f"\nNeutral vs {condition}:")
            logger.info(f"  {comparison['interpretation']}")
            logger.info(f"  shape_bias change: {comparison['difference']:+.4f}")

    # ── Perceptual Over-Steering Analysis ─────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PERCEPTUAL OVER-STEERING ANALYSIS")
    logger.info("=" * 60)

    neutral_shape_acc = metrics_by_condition.get("neutral", {}).get("shape_accuracy", 0)
    neutral_texture_acc = metrics_by_condition.get("neutral", {}).get("texture_accuracy", 0)

    for condition, metrics in metrics_by_condition.items():
        if condition == "neutral":
            continue
        shape_drop = neutral_shape_acc - metrics.get("shape_accuracy", 0)
        texture_gain = metrics.get("texture_accuracy", 0) - neutral_texture_acc
        over_steering = shape_drop > 0.10

        logger.info(
            f"  {condition:<25} "
            f"shape_drop={shape_drop:+.3f}  "
            f"texture_gain={texture_gain:+.3f}  "
            f"{'⚠ OVER-STEERING RISK' if over_steering else '✓ OK'}"
        )

    # ── Save Summary (all conditions, including pre-loaded) ───
    summary = {
        "phase": "Phase 3A — Manual Steering",
        "conditions_evaluated": list(metrics_by_condition.keys()),
        "metrics": {
            cond: {k: v for k, v in m.items()
                   if k in ["shape_accuracy", "texture_accuracy",
                             "cue_accuracy", "shape_bias", "n_images"]}
            for cond, m in metrics_by_condition.items()
        },
    }

    summary_path = results_dir / "phase3a_summary.json"
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
    parser.add_argument(
        "--conditions", nargs="+", metavar="COND",
        default=ALL_CONDITIONS,
        help=(
            "Which conditions to run inference for. "
            "Already-complete conditions are loaded from disk automatically. "
            f"All: {ALL_CONDITIONS}"
        ),
    )
    args = parser.parse_args()

    unknown = [c for c in args.conditions if c not in ALL_CONDITIONS]
    if unknown:
        parser.error(f"Unknown conditions: {unknown}. Valid: {ALL_CONDITIONS}")

    main(args.config, args.conditions)
