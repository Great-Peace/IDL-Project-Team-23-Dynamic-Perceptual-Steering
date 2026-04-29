"""
phase4_adversarial.py
======================
Phase 4 — Adversarial Texture Steering (Extension Experiment)

Tests "Cultural Stubbornness": Can steering force the model to
prioritize African textures even when applied to ICONIC WESTERN SHAPES?

Examples:
  - Zulu beadwork on a British red phone booth
  - Kente cloth pattern on an Eiffel Tower silhouette
  - Mudcloth on a classic American diner

If the model STILL defaults to the Western shape even after steering,
that's evidence of deep-rooted cultural stubbornness — the Western
shape bias is so strong it overrides even explicit cultural prompts.

This quantifies the LIMITS of runtime alignment.

Run: sbatch scripts/run_adversarial.slurm
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from PIL import Image

from src.data.cue_conflict_synthesizer import CueConflictSynthesizer
from src.models.internvl_wrapper import InternVLWrapper
from src.evaluation.metrics import compute_all_metrics, parse_decision
from src.steering.prompts import PromptLibrary
from src.visualization.plots import ResultsVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/phase4_adversarial.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AdversarialPair:
    """
    An adversarial pair: African texture applied to iconic Western shape.
    The model SHOULD recognize the African texture after steering.
    """
    western_shape_path: str    # Image of the iconic Western object
    african_texture_path: str  # Image of the African cultural artifact
    conflict_path: str         # Synthesized conflict image
    western_shape_label: str   # e.g. "phone booth", "eiffel tower"
    african_texture_label: str # e.g. "zulu beadwork", "kente cloth"
    iconicity_level: str       # "high" / "medium" / "low"


# ── Hard-coded adversarial test cases ──────────────────────────────────────
# You need to provide these images in data/adversarial/western_shapes/
# and data/adversarial/african_textures/
ADVERSARIAL_CASES = [
    {
        "western_shape": "phone_booth.jpg",
        "african_texture": "zulu_beadwork.jpg",
        "western_label": "phone booth",
        "african_label": "zulu beadwork",
        "iconicity": "high"
    },
    {
        "western_shape": "eiffel_tower.jpg",
        "african_texture": "kente_cloth.jpg",
        "western_label": "eiffel tower",
        "african_label": "kente cloth",
        "iconicity": "high"
    },
    {
        "western_shape": "london_bus.jpg",
        "african_texture": "mudcloth.jpg",
        "western_label": "london bus",
        "african_label": "mudcloth",
        "iconicity": "high"
    },
    {
        "western_shape": "american_diner.jpg",
        "african_texture": "adire.jpg",
        "western_label": "american diner",
        "african_label": "adire cloth",
        "iconicity": "medium"
    },
    {
        "western_shape": "yellow_taxi.jpg",
        "african_texture": "ankara.jpg",
        "western_label": "yellow taxi",
        "african_label": "ankara fabric",
        "iconicity": "medium"
    },
    {
        "western_shape": "mailbox.jpg",
        "african_texture": "adinkra.jpg",
        "western_label": "mailbox",
        "african_label": "adinkra symbols",
        "iconicity": "low"
    },
]


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("="*60)
    logger.info("PHASE 4 — ADVERSARIAL TEXTURE STEERING")
    logger.info("="*60)
    logger.info(
        "\nHypothesis: The model shows 'cultural stubbornness' — "
        "even after cultural steering, highly iconic Western shapes "
        "resist African texture recognition more than generic shapes."
    )

    # ── Check for adversarial image data ─────────────────────
    western_dir = Path(config["paths"]["data_raw"]) / "adversarial" / "western_shapes"
    african_dir = Path(config["paths"]["data_raw"]) / "adversarial" / "african_textures"

    if not western_dir.exists() or not african_dir.exists():
        logger.warning(
            f"\nAdversarial image directories not found:\n"
            f"  {western_dir}\n"
            f"  {african_dir}\n"
            f"\nPlease create these directories and add images.\n"
            f"See README.md for instructions on collecting adversarial images."
        )
        logger.info("Creating directory structure...")
        western_dir.mkdir(parents=True, exist_ok=True)
        african_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Directories created. Add images and rerun.")
        return

    # ── Synthesize Adversarial Conflict Images ────────────────
    logger.info("\nSynthesizing adversarial cue-conflict images...")
    synthesizer = CueConflictSynthesizer(config)

    adversarial_pairs = []
    conflict_dir = Path(config["paths"]["data_cue_conflict"]) / "adversarial"
    conflict_dir.mkdir(parents=True, exist_ok=True)

    for case in ADVERSARIAL_CASES:
        western_path = western_dir / case["western_shape"]
        african_path = african_dir / case["african_texture"]

        if not western_path.exists():
            logger.warning(f"Missing western shape image: {western_path}")
            continue
        if not african_path.exists():
            logger.warning(f"Missing African texture image: {african_path}")
            continue

        western_img = Image.open(western_path).convert("RGB")
        african_img = Image.open(african_path).convert("RGB")

        pair_id = f"adv_{case['western_label'].replace(' ', '_')}"
        conflict_pair = synthesizer.synthesize(
            content_image=western_img,
            style_image=african_img,
            shape_label=case["western_label"],
            texture_label=case["african_label"],
            category="adversarial",
            region="mixed",
            pair_id=pair_id
        )

        if conflict_pair:
            adversarial_pairs.append(AdversarialPair(
                western_shape_path=str(western_path),
                african_texture_path=str(african_path),
                conflict_path=conflict_pair.conflict_path,
                western_shape_label=case["western_label"],
                african_texture_label=case["african_label"],
                iconicity_level=case["iconicity"]
            ))

    if not adversarial_pairs:
        logger.error("No adversarial pairs created. Check image files.")
        return

    logger.info(f"Created {len(adversarial_pairs)} adversarial pairs.")

    # ── Load Model & Prompts ──────────────────────────────────
    model = InternVLWrapper(config)
    prompts = PromptLibrary(config)

    # ── Evaluate Under Each Condition ─────────────────────────
    conditions = {
        "neutral":              prompts.get_text("neutral"),
        "cultural":             prompts.get_text("cultural"),
        "adversarial_texture":  prompts.get_text("adversarial_texture"),
    }

    results_by_condition = {}

    for condition_name, prompt_text in conditions.items():
        logger.info(f"\nEvaluating adversarial pairs — condition: {condition_name}")
        condition_results = []

        for pair in adversarial_pairs:
            try:
                conflict_img = Image.open(pair.conflict_path).convert("RGB")
                response = model.generate(conflict_img, prompt_text)

                shape_hit, texture_hit = parse_decision(
                    response,
                    pair.western_shape_label,
                    pair.african_texture_label
                )

                condition_results.append({
                    "conflict_path": pair.conflict_path,
                    "western_label": pair.western_shape_label,
                    "african_label": pair.african_texture_label,
                    "iconicity_level": pair.iconicity_level,
                    "shape_label": pair.western_shape_label,
                    "texture_label": pair.african_texture_label,
                    "prompt_type": condition_name,
                    "response": response,
                    "shape_hit": shape_hit,
                    "texture_hit": texture_hit,
                })

                logger.info(
                    f"  {pair.western_shape_label} + {pair.african_texture_label} "
                    f"→ shape={shape_hit}, texture={texture_hit} | "
                    f"{response[:60]}..."
                )

            except Exception as e:
                logger.error(f"Failed for {pair.conflict_path}: {e}")

        results_by_condition[condition_name] = condition_results

    # ── Stubbornness Analysis ─────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("CULTURAL STUBBORNNESS ANALYSIS")
    logger.info("="*60)

    # Group by iconicity level
    iconicity_levels = ["high", "medium", "low"]

    for condition_name, results in results_by_condition.items():
        logger.info(f"\n[{condition_name}]")

        for level in iconicity_levels:
            level_results = [r for r in results if r["iconicity_level"] == level]
            if not level_results:
                continue

            texture_recognition = sum(r["texture_hit"] for r in level_results)
            total = len(level_results)
            rate = texture_recognition / total if total > 0 else 0

            logger.info(
                f"  Iconicity={level:<6}: "
                f"African texture recognized {texture_recognition}/{total} "
                f"({rate:.0%})"
            )

    # Key finding
    if "neutral" in results_by_condition and "cultural" in results_by_condition:
        # For high iconicity items
        neutral_high = [r for r in results_by_condition["neutral"]
                       if r["iconicity_level"] == "high"]
        cultural_high = [r for r in results_by_condition["cultural"]
                        if r["iconicity_level"] == "high"]

        if neutral_high and cultural_high:
            neutral_texture_high = sum(r["texture_hit"] for r in neutral_high) / len(neutral_high)
            cultural_texture_high = sum(r["texture_hit"] for r in cultural_high) / len(cultural_high)

            logger.info(
                f"\nFor HIGH ICONICITY Western shapes:\n"
                f"  Neutral prompt : {neutral_texture_high:.0%} African texture recognition\n"
                f"  Cultural prompt: {cultural_texture_high:.0%} African texture recognition\n"
                f"  → Stubbornness resistance: "
                f"{'HIGH — steering barely helps' if cultural_texture_high < 0.5 else 'MODERATE — steering partially helps'}"
            )

    # ── Save Results ──────────────────────────────────────────
    summary = {
        "phase": "Phase 4 — Adversarial",
        "n_pairs": len(adversarial_pairs),
        "results_by_condition": {
            cond: [
                {k: v for k, v in r.items() if k != "response"}
                for r in results
            ]
            for cond, results in results_by_condition.items()
        }
    }

    summary_path = Path(config["paths"]["results"]) / "phase4_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nPhase 4 summary saved to {summary_path}")

    logger.info("\n✓ Phase 4 complete!")
    return results_by_condition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 — Adversarial Steering")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
