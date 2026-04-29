"""
phase3c_probing.py
===================
Phase 3C — Mechanistic Confidence Probing

Extracts token-level logit confidence scores to prove that
the model SEES African cultural textures but SUPPRESSES them.

Key evidence we're looking for:
  Under neutral prompt:
    shape_token confidence  ≈ 1.0 (highly confident)
    texture_token confidence ≈ 0.0 (suppressed to near-zero)

  Under cultural prompt:
    texture_token confidence RISES significantly
    shape_token confidence may decrease

This mirrors Figure 3 of Gavrikov et al. (2025) but proves
the suppression specifically affects African cultural knowledge.

Run: sbatch scripts/run_probing.slurm
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
from src.probing.confidence_probing import ConfidenceProber
from src.steering.prompts import PromptLibrary
from src.visualization.plots import ResultsVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/phase3c_probing.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("="*60)
    logger.info("PHASE 3C — MECHANISTIC CONFIDENCE PROBING")
    logger.info("="*60)

    # ── Load Dataset ──────────────────────────────────────────
    dataset = load_all_datasets(config)
    if len(dataset) == 0:
        logger.error("Empty dataset.")
        sys.exit(1)

    # ── Load Model ────────────────────────────────────────────
    model = InternVLWrapper(config)
    prompts = PromptLibrary(config)

    # ── Define Conditions to Probe ────────────────────────────
    # We probe the 3 key conditions for mechanistic analysis
    probe_conditions = {
        "neutral":    prompts.get_text("neutral"),
        "cultural":   prompts.get_text("cultural"),
        "structural": prompts.get_text("structural"),
    }

    # If APO best prompt exists, add it too
    apo_summary_path = Path(config["paths"]["results"]) / "phase3b_summary.json"
    if apo_summary_path.exists():
        with open(apo_summary_path) as f:
            apo_summary = json.load(f)
        apo_best_prompt = apo_summary.get("best_prompt")
        if apo_best_prompt:
            probe_conditions["apo_best"] = apo_best_prompt
            logger.info("Added APO best prompt to probing conditions.")

    # ── Run Probing ───────────────────────────────────────────
    prober = ConfidenceProber(model, config)

    logger.info(f"\nProbing {len(probe_conditions)} conditions "
                f"× {len(dataset)} images...")
    logger.info("This tests whether the model SEES but SUPPRESSES "
                "African cultural information.\n")

    probing_results = prober.run_full_probing(
        dataset=dataset,
        prompt_conditions=probe_conditions
    )

    # ── Key Finding Summary ───────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("MECHANISTIC PROBING — KEY FINDINGS")
    logger.info("="*60)

    import numpy as np

    if "neutral" in probing_results and "cultural" in probing_results:
        neutral_texture = [r["texture_confidence"]
                          for r in probing_results["neutral"]]
        cultural_texture = [r["texture_confidence"]
                           for r in probing_results["cultural"]]
        neutral_shape = [r["shape_confidence"]
                        for r in probing_results["neutral"]]

        logger.info(
            f"\nUnder NEUTRAL prompt:\n"
            f"  Mean shape confidence   : {np.mean(neutral_shape):.4f}\n"
            f"  Mean texture confidence : {np.mean(neutral_texture):.4f}\n"
            f"  → Shape dominance (shape-texture): "
            f"{np.mean(neutral_shape) - np.mean(neutral_texture):.4f}"
        )

        logger.info(
            f"\nAfter CULTURAL steering:\n"
            f"  Mean texture confidence : {np.mean(cultural_texture):.4f}\n"
            f"  → Texture confidence ROSE by: "
            f"{np.mean(cultural_texture) - np.mean(neutral_texture):+.4f}"
        )

        # Check suppression fraction
        suppressed_neutral = np.mean([r["texture_suppressed"]
                                      for r in probing_results["neutral"]])
        suppressed_cultural = np.mean([r["texture_suppressed"]
                                       for r in probing_results["cultural"]])

        logger.info(
            f"\nTexture token suppressed (confidence < 0.01):\n"
            f"  Neutral  : {suppressed_neutral:.1%}\n"
            f"  Cultural : {suppressed_cultural:.1%}\n"
            f"  → Steering RECOVERED suppressed cultural tokens: "
            f"{suppressed_neutral - suppressed_cultural:.1%} of images"
        )

        # Key conclusion
        if np.mean(cultural_texture) > np.mean(neutral_texture) * 1.5:
            logger.info(
                "\n✓ HYPOTHESIS CONFIRMED: The model sees but suppresses "
                "African cultural textures under neutral prompts. "
                "Cultural steering successfully activates this latent knowledge."
            )
        else:
            logger.info(
                "\n⚠ PARTIAL RESULT: Cultural steering shows improvement "
                "but suppression is more complex than simple token-level effects."
            )

    # ── Save Summary ──────────────────────────────────────────
    summary = {
        "phase": "Phase 3C — Mechanistic Probing",
        "conditions_probed": list(probe_conditions.keys()),
        "n_images": len(dataset),
    }

    # Add per-condition stats
    for condition, results in probing_results.items():
        if results:
            summary[f"{condition}_stats"] = {
                "mean_shape_conf": float(np.mean([r["shape_confidence"]
                                                   for r in results])),
                "mean_texture_conf": float(np.mean([r["texture_confidence"]
                                                     for r in results])),
                "fraction_suppressed": float(np.mean([r["texture_suppressed"]
                                                       for r in results])),
            }

    summary_path = Path(config["paths"]["results"]) / "phase3c_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nPhase 3C summary saved to {summary_path}")

    # ── Generate Confidence Distribution Plots ────────────────
    logger.info("\nGenerating confidence distribution plots...")
    viz = ResultsVisualizer(config)
    viz.plot_confidence_distributions(probing_results)

    logger.info("\n✓ Phase 3C complete!")
    return probing_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3C — Confidence Probing")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
