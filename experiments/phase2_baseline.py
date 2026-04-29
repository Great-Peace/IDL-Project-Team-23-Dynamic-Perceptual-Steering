"""
phase2_baseline.py
==================
Phase 2 — Full Automated Baseline Evaluation

Runs InternVL-3 and CLIP baseline on the full 400-500 image dataset
using the neutral prompt. Records Shape Accuracy, Texture Accuracy,
Cue Accuracy, and Shape Bias.

This establishes the "perceptual erasure" baseline — how bad is
the model without any steering.

Run on PSC:
    sbatch scripts/run_baseline.slurm

Or locally:
    python experiments/phase2_baseline.py --config configs/config.yaml
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.data.dataset_loader import load_all_datasets
from src.models.internvl_wrapper import InternVLWrapper
from src.models.clip_baseline import CLIPBaseline
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import compute_all_metrics
from src.steering.prompts import PromptLibrary
from src.visualization.plots import ResultsVisualizer

# ── Logging Setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/phase2_baseline.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    # ── Load Configuration ────────────────────────────────────
    with open(config_path) as f:
        config = yaml.safe_load(f)

    os.makedirs(config["paths"]["results"], exist_ok=True)
    os.makedirs(config["paths"]["figures"], exist_ok=True)

    logger.info("="*60)
    logger.info("PHASE 2 — BASELINE EVALUATION")
    logger.info("="*60)

    # ── Load Dataset ──────────────────────────────────────────
    logger.info("\nStep 1: Loading dataset...")
    dataset = load_all_datasets(config)

    if len(dataset) == 0:
        logger.error(
            "Dataset is empty. Please download datasets first.\n"
            "See README.md for instructions."
        )
        sys.exit(1)

    dataset.print_summary()

    # ── Load InternVL-3 ───────────────────────────────────────
    logger.info("\nStep 2: Loading InternVL-3 (8B)...")
    model = InternVLWrapper(config)
    model_info = model.get_model_info()
    logger.info(f"Model info: {model_info}")

    # ── Load Prompts ──────────────────────────────────────────
    prompts = PromptLibrary(config)

    # ── Run Neutral Baseline ──────────────────────────────────
    logger.info("\nStep 3: Running Neutral Baseline (InternVL-3)...")
    evaluator = Evaluator(model, config)

    neutral_results = evaluator.run(
        dataset=dataset,
        prompt=prompts.get_text("neutral"),
        prompt_type="neutral",
        output_prefix="phase2_"
    )

    # Compute and display metrics
    neutral_metrics = compute_all_metrics(neutral_results)

    logger.info("\n" + "="*50)
    logger.info("NEUTRAL BASELINE RESULTS")
    logger.info("="*50)
    logger.info(f"Shape Accuracy   : {neutral_metrics['shape_accuracy']:.4f}")
    logger.info(f"Texture Accuracy : {neutral_metrics['texture_accuracy']:.4f}")
    logger.info(f"Cue Accuracy     : {neutral_metrics['cue_accuracy']:.4f}")
    logger.info(f"Shape Bias       : {neutral_metrics['shape_bias']:.4f}")

    # ── Insight 1: Famous vs Everyday ────────────────────────
    if "famous_items" in neutral_metrics and "everyday_items" in neutral_metrics:
        logger.info("\nINSIGHT 1 — Famous vs Everyday:")
        f = neutral_metrics["famous_items"]
        e = neutral_metrics["everyday_items"]
        logger.info(f"  Famous items  → texture_acc: {f['texture_accuracy']:.4f}")
        logger.info(f"  Everyday items→ texture_acc: {e['texture_accuracy']:.4f}")
        logger.info(f"  → Gap: {f['texture_accuracy'] - e['texture_accuracy']:.4f}")

    # ── CLIP Vision-Only Baseline ─────────────────────────────
    logger.info("\nStep 4: Running CLIP Vision-Only Baseline...")
    try:
        clip = CLIPBaseline(config)
        clip_metrics = clip.compute_shape_bias(dataset)
        logger.info(f"\nCLIP BASELINE RESULTS")
        logger.info(f"Shape Bias     : {clip_metrics['shape_bias']:.4f}")
        logger.info(f"Cue Accuracy   : {clip_metrics['cue_accuracy']:.4f}")
        logger.info(
            f"\nComparison (CLIP vs InternVL-3 neutral):\n"
            f"  CLIP shape_bias     = {clip_metrics['shape_bias']:.4f}\n"
            f"  InternVL shape_bias = {neutral_metrics['shape_bias']:.4f}\n"
            f"  → {'InternVL more shape-biased' if neutral_metrics['shape_bias'] > clip_metrics['shape_bias'] else 'CLIP more shape-biased'}"
        )
    except ImportError:
        logger.warning("CLIP not installed. Skipping CLIP baseline.")
        logger.warning("Install: pip install git+https://github.com/openai/CLIP.git")
        clip_metrics = {}

    # ── Save Summary ──────────────────────────────────────────
    summary = {
        "phase": "Phase 2 — Baseline",
        "dataset_size": len(dataset),
        "internvl_neutral": neutral_metrics,
        "clip_baseline": clip_metrics,
    }

    summary_path = Path(config["paths"]["results"]) / "phase2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nPhase 2 summary saved to {summary_path}")

    # ── Generate Figures ──────────────────────────────────────
    logger.info("\nGenerating baseline figures...")
    conditions_data = {"neutral": neutral_metrics}
    if clip_metrics:
        conditions_data["clip_baseline"] = clip_metrics

    viz = ResultsVisualizer(config)
    viz.plot_shape_bias_scatter(conditions_data,
                                title="Phase 2: Baseline Shape Bias")

    logger.info("\n✓ Phase 2 complete!")
    return neutral_results, neutral_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 — Baseline Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)
