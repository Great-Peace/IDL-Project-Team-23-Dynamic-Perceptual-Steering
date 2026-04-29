"""
phase3b_apo.py
===============
Phase 3B — Automated Prompt Optimization (APO)

Uses Mistral-7B-Instruct as optimizer to automatically discover
better cultural steering prompts than hand-crafted ones.

Run: sbatch scripts/run_apo.slurm
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
from src.evaluation.metrics import compute_all_metrics
from src.steering.apo import AutomatedPromptOptimizer
from src.visualization.plots import ResultsVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/phase3b_apo.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("="*60)
    logger.info("PHASE 3B — AUTOMATED PROMPT OPTIMIZATION (APO)")
    logger.info("="*60)

    # ── Load Dataset ──────────────────────────────────────────
    dataset = load_all_datasets(config)
    if len(dataset) == 0:
        logger.error("Empty dataset.")
        sys.exit(1)

    # ── Load InternVL-3 ───────────────────────────────────────
    vlm = InternVLWrapper(config)

    # ── Run APO ───────────────────────────────────────────────
    logger.info("\nStarting APO loop...")
    apo = AutomatedPromptOptimizer(vlm, config)

    best_prompt, history = apo.optimize(dataset)

    logger.info(f"\n{'='*60}")
    logger.info("APO COMPLETE — BEST PROMPT DISCOVERED:")
    logger.info(f"{'='*60}")
    logger.info(best_prompt)

    # ── Evaluate Best APO Prompt ──────────────────────────────
    logger.info("\nEvaluating best APO prompt on full dataset...")
    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator(vlm, config)

    apo_results = evaluator.run(
        dataset=dataset,
        prompt=best_prompt,
        prompt_type="apo_best",
        output_prefix="phase3b_"
    )

    apo_metrics = compute_all_metrics(apo_results)

    logger.info("\nAPO BEST PROMPT — FINAL METRICS:")
    logger.info(f"  texture_accuracy : {apo_metrics['texture_accuracy']:.4f}")
    logger.info(f"  shape_accuracy   : {apo_metrics['shape_accuracy']:.4f}")
    logger.info(f"  shape_bias       : {apo_metrics['shape_bias']:.4f}")

    # ── Save Results ──────────────────────────────────────────
    summary = {
        "phase": "Phase 3B — APO",
        "best_prompt": best_prompt,
        "best_metrics": {
            k: v for k, v in apo_metrics.items()
            if k in ["shape_accuracy", "texture_accuracy",
                     "cue_accuracy", "shape_bias"]
        },
        "top_5_prompts": [c.to_dict() for c in apo.get_top_prompts(5)],
    }

    summary_path = Path(config["paths"]["results"]) / "phase3b_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nPhase 3B summary saved to {summary_path}")

    # ── Plot APO Progress ─────────────────────────────────────
    viz = ResultsVisualizer(config)
    viz.plot_apo_progress(history)

    # ── Print Top 5 Discovered Prompts ────────────────────────
    logger.info("\nTOP 5 APO-DISCOVERED PROMPTS:")
    for i, candidate in enumerate(apo.get_top_prompts(5), 1):
        logger.info(f"\n  #{i} (texture_acc={candidate.texture_accuracy:.4f}, "
                    f"shape_acc={candidate.shape_accuracy:.4f}):")
        logger.info(f"  {candidate.text}")

    logger.info("\n✓ Phase 3B complete!")
    return best_prompt, history, apo_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3B — APO")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
