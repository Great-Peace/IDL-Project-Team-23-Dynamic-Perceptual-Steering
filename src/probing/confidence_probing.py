"""
confidence_probing.py
======================
Phase 3C — Mechanistic Confidence Probing.

Core hypothesis (from Gavrikov et al. 2025, extended to cultural domain):
  When shown a cue-conflict image with an African texture,
  the model SEES the cultural information in its encoder features,
  but the LLM component SUPPRESSES it in favor of the shape.

Evidence for this:
  - Token logit for shape label = very high confidence (near 1.0)
  - Token logit for cultural/texture label = near zero
  → Even though the visual encoder likely encoded both

After cultural steering:
  - Token logit for cultural label RISES significantly
  - Token logit for shape label may FALL slightly

This proves the cultural knowledge was always there — 
just suppressed by the LLM's default bias.

Usage:
    prober = ConfidenceProber(model, config)
    result = prober.probe(image, shape_label, texture_label, prompt)
    prober.run_full_probing(dataset, conditions)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from ..models.internvl_wrapper import InternVLWrapper

logger = logging.getLogger(__name__)


class ConfidenceProber:
    """
    Extracts and analyzes token-level confidence scores from InternVL-3.

    For each cue-conflict image, under each prompt condition,
    we measure how confident the model is in its shape token
    vs its cultural texture token.

    This is the mechanistic proof that "perceptual erasure" is a
    LLM suppression phenomenon, not an encoder encoding failure.
    """

    def __init__(self, model: InternVLWrapper, config: dict):
        self.model = model
        self.config = config
        self.results_dir = Path(config["paths"]["results"]) / "probing"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def probe(self, image: Union[str, Image.Image],
               shape_label: str,
               texture_label: str,
               prompt: str,
               prompt_type: str = "neutral") -> Dict:
        """
        Probe the confidence scores for shape and texture tokens
        for a single image under a given prompt.

        Args:
            image         : PIL image or path
            shape_label   : functional label (e.g. "bowl")
            texture_label : cultural label (e.g. "kente")
            prompt        : text prompt to use
            prompt_type   : label for this condition

        Returns:
            Dict with confidence scores and derived analysis
        """
        # Target tokens to probe: shape, texture, and alternatives
        target_tokens = [
            shape_label,
            texture_label,
            # Also probe "African" as a generic positive signal
            "African",
            "traditional",
            "cultural",
        ]

        # Add texture synonyms (first synonym for each)
        from ..evaluation.metrics import _get_texture_synonyms
        synonyms = _get_texture_synonyms(texture_label.lower())
        target_tokens.extend(synonyms[:2])  # Add first 2 synonyms

        # Get token logits from model
        confidence_scores = self.model.get_token_logits(
            image=image,
            prompt=prompt,
            target_tokens=target_tokens
        )

        # Determine if shape or texture "wins"
        shape_conf = confidence_scores.get(shape_label, 0.0)
        texture_conf = confidence_scores.get(texture_label, 0.0)

        # Check if texture is essentially suppressed (near-zero)
        texture_suppressed = texture_conf < 0.01

        # Shape dominance: how much more confident shape is vs texture
        shape_dominance = shape_conf - texture_conf

        return {
            "prompt_type": prompt_type,
            "prompt": prompt[:100],
            "shape_label": shape_label,
            "texture_label": texture_label,
            "shape_confidence": round(shape_conf, 6),
            "texture_confidence": round(texture_conf, 6),
            "african_confidence": round(confidence_scores.get("African", 0.0), 6),
            "traditional_confidence": round(confidence_scores.get("traditional", 0.0), 6),
            "shape_dominance": round(shape_dominance, 6),
            "texture_suppressed": texture_suppressed,
            "all_scores": confidence_scores,
        }

    def run_full_probing(self, dataset,
                          prompt_conditions: Dict[str, str]) -> Dict[str, List[Dict]]:
        """
        Run confidence probing across all images and all prompt conditions.

        This generates the data for the confidence distribution plots
        (like Figure 3 in Gavrikov et al. 2025).

        Args:
            dataset           : AfricanCulturalDataset
            prompt_conditions : Dict mapping condition name to prompt text

        Returns:
            Dict mapping condition name to list of probe results
        """
        all_results = {}

        for condition_name, prompt_text in prompt_conditions.items():
            logger.info(f"\nProbing condition: {condition_name}")
            condition_results = []

            for record in tqdm(dataset,
                                desc=f"Probing [{condition_name}]"):
                try:
                    result = self.probe(
                        image=record.load_image(),
                        shape_label=record.shape_label,
                        texture_label=record.texture_label,
                        prompt=prompt_text,
                        prompt_type=condition_name
                    )
                    # Add record metadata
                    result.update({
                        "image_path": record.image_path,
                        "category": record.category,
                        "region": record.region,
                        "is_famous": record.is_famous,
                    })
                    condition_results.append(result)

                except Exception as e:
                    logger.debug(f"Probing failed for {record.image_path}: {e}")

            all_results[condition_name] = condition_results
            logger.info(f"  Probed {len(condition_results)} images.")

            # Compute and log aggregate confidence stats
            self._log_aggregate_stats(condition_results, condition_name)

        # Save all probing results
        self._save_probing_results(all_results)

        # Compute cross-condition comparison
        self._compute_cross_condition_analysis(all_results)

        return all_results

    def _log_aggregate_stats(self, results: List[Dict], condition: str):
        """Log aggregate confidence statistics for a condition."""
        if not results:
            return

        shape_confs = [r["shape_confidence"] for r in results]
        texture_confs = [r["texture_confidence"] for r in results]
        suppressed_frac = np.mean([r["texture_suppressed"] for r in results])

        logger.info(f"\n  [{condition}] Confidence Statistics:")
        logger.info(f"    Mean shape confidence   : {np.mean(shape_confs):.4f} "
                    f"± {np.std(shape_confs):.4f}")
        logger.info(f"    Mean texture confidence : {np.mean(texture_confs):.4f} "
                    f"± {np.std(texture_confs):.4f}")
        logger.info(f"    Texture suppressed (<0.01): {suppressed_frac:.1%}")

    def _compute_cross_condition_analysis(self, all_results: Dict[str, List[Dict]]):
        """
        Compare texture confidence across conditions.

        Key questions:
        1. Does texture confidence rise after cultural steering?
        2. Is the rise statistically significant?
        3. Which categories show the biggest suppression?
        """
        if "neutral" not in all_results:
            return

        neutral_texture = [r["texture_confidence"]
                           for r in all_results["neutral"]]

        analysis = {"conditions": {}}

        for condition, results in all_results.items():
            condition_texture = [r["texture_confidence"] for r in results]
            mean_change = np.mean(condition_texture) - np.mean(neutral_texture)

            # Paired comparison
            from scipy import stats
            if len(condition_texture) == len(neutral_texture):
                t_stat, p_val = stats.ttest_rel(neutral_texture, condition_texture)
            else:
                t_stat, p_val = stats.ttest_ind(neutral_texture, condition_texture)

            analysis["conditions"][condition] = {
                "mean_texture_confidence": round(np.mean(condition_texture), 6),
                "std_texture_confidence": round(np.std(condition_texture), 6),
                "change_from_neutral": round(float(mean_change), 6),
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_val), 6),
                "significant": bool(p_val < 0.05),
            }

        # Log the key finding
        logger.info("\n" + "="*60)
        logger.info("MECHANISTIC PROBING — CROSS CONDITION ANALYSIS")
        logger.info("="*60)
        for cond, stats_dict in analysis["conditions"].items():
            sig = "✓ SIGNIFICANT" if stats_dict["significant"] else "✗ not significant"
            logger.info(
                f"  {cond:<20} "
                f"texture_conf={stats_dict['mean_texture_confidence']:.4f}  "
                f"Δ={stats_dict['change_from_neutral']:+.4f}  "
                f"{sig}"
            )

        # Save analysis
        analysis_file = self.results_dir / "cross_condition_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"\nCross-condition analysis saved to {analysis_file}")

    def _save_probing_results(self, all_results: Dict[str, List[Dict]]):
        """Save all probing results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"probing_results_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Probing results saved to {output_file}")
