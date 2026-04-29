"""
evaluator.py
=============
Orchestrates evaluation runs across the full dataset for
each prompt condition (neutral, structural, cultural, sequential).

Saves per-image results to JSON and aggregate metrics to CSV.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from ..models.internvl_wrapper import InternVLWrapper
from .metrics import compute_all_metrics, compare_conditions

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Runs the full evaluation pipeline for one prompt condition.

    For each image in the dataset:
      1. Sends the image + prompt to InternVL-3
      2. Records the response
      3. Parses shape/texture decision
      4. Saves results to JSON

    Then computes aggregate metrics and saves to CSV.
    """

    def __init__(self, model: InternVLWrapper, config: dict):
        self.model = model
        self.config = config
        self.results_dir = Path(config["paths"]["results"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.save_per_image = config["logging"].get("save_per_image", True)

    def run(self, dataset,
            prompt: str,
            prompt_type: str,
            output_prefix: str = "") -> List[Dict]:
        """
        Run evaluation over the full dataset with a single prompt.

        Args:
            dataset      : AfricanCulturalDataset
            prompt       : The text prompt to use
            prompt_type  : Label for this condition (e.g. "neutral", "cultural")
            output_prefix: Optional prefix for output filenames

        Returns:
            List of per-image result dicts
        """
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"{output_prefix}{prompt_type}_{timestamp}.json"

        logger.info(f"\n{'='*60}")
        logger.info(f"Running evaluation: {prompt_type}")
        logger.info(f"Prompt: {prompt[:80]}...")
        logger.info(f"Dataset size: {len(dataset)}")
        logger.info(f"{'='*60}")

        for idx, record in enumerate(tqdm(dataset, desc=f"Eval [{prompt_type}]")):
            try:
                # Generate response from InternVL-3
                response = self.model.generate(
                    image=record.load_image(),
                    prompt=prompt
                )

                result = {
                    # Identification
                    "idx": idx,
                    "image_path": record.image_path,
                    "source": record.source,
                    # Labels
                    "shape_label": record.shape_label,
                    "texture_label": record.texture_label,
                    "category": record.category,
                    "region": record.region,
                    "is_famous": record.is_famous,
                    # Experiment condition
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                    # Model output
                    "response": response,
                }
                results.append(result)

                # Auto-save every 50 images to protect against crashes
                if self.save_per_image and (idx + 1) % 50 == 0:
                    self._save_results(results, output_file)
                    logger.debug(f"Auto-saved {idx + 1} results.")

            except Exception as e:
                logger.error(f"Evaluation failed for {record.image_path}: {e}")
                results.append({
                    "idx": idx,
                    "image_path": record.image_path,
                    "shape_label": record.shape_label,
                    "texture_label": record.texture_label,
                    "category": record.category,
                    "region": record.region,
                    "is_famous": record.is_famous,
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                    "response": "",
                    "error": str(e)
                })

        # Final save
        self._save_results(results, output_file)
        logger.info(f"Results saved to {output_file}")

        # Compute and log aggregate metrics
        metrics = compute_all_metrics(results)
        self._log_metrics(metrics, prompt_type)

        # Save aggregate metrics
        metrics_file = self.results_dir / f"{output_prefix}metrics_{prompt_type}_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        return results

    def run_all_conditions(self, dataset,
                            prompts: Dict[str, str]) -> Dict[str, List[Dict]]:
        """
        Run evaluation for all prompt conditions in one go.

        Args:
            dataset : AfricanCulturalDataset
            prompts : Dict mapping condition name to prompt string
                     e.g. {"neutral": "What is in this image?",
                            "cultural": "Analyze the cultural textures..."}

        Returns:
            Dict mapping condition name to list of results
        """
        all_results = {}

        for prompt_type, prompt in prompts.items():
            results = self.run(dataset, prompt, prompt_type)
            all_results[prompt_type] = results

        # Compute pairwise comparisons
        if "neutral" in all_results:
            for condition, results in all_results.items():
                if condition == "neutral":
                    continue
                comparison = compare_conditions(
                    all_results["neutral"],
                    results,
                    condition_a="neutral",
                    condition_b=condition
                )
                logger.info(f"\nStatistical comparison (neutral vs {condition}):")
                logger.info(f"  {comparison['interpretation']}")
                logger.info(f"  Δ shape_bias = {comparison['difference']:.4f}")

        # Save comparison summary CSV
        self._save_comparison_csv(all_results)

        return all_results

    def _save_results(self, results: List[Dict], output_file: Path):
        """Save results list to JSON."""
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    def _log_metrics(self, metrics: Dict, prompt_type: str):
        """Log aggregate metrics to console."""
        logger.info(f"\n--- Metrics: {prompt_type} ---")
        logger.info(f"  N images         : {metrics.get('n_images', 0)}")
        logger.info(f"  Shape Accuracy   : {metrics.get('shape_accuracy', 0):.4f}")
        logger.info(f"  Texture Accuracy : {metrics.get('texture_accuracy', 0):.4f}")
        logger.info(f"  Cue Accuracy     : {metrics.get('cue_accuracy', 0):.4f}")
        logger.info(f"  Shape Bias       : {metrics.get('shape_bias', 0):.4f}")

        # Log insight 1 if available
        if "famous_items" in metrics and "everyday_items" in metrics:
            logger.info(f"\n  INSIGHT 1 — Famous vs Everyday:")
            famous = metrics["famous_items"]
            everyday = metrics["everyday_items"]
            logger.info(f"  Famous   texture_acc : {famous.get('texture_accuracy', 0):.4f}")
            logger.info(f"  Everyday texture_acc : {everyday.get('texture_accuracy', 0):.4f}")

    def _save_comparison_csv(self, all_results: Dict[str, List[Dict]]):
        """
        Save a summary CSV comparing all conditions side by side.
        This is the main table for your final report.
        """
        rows = []
        for condition, results in all_results.items():
            metrics = compute_all_metrics(results)
            row = {
                "condition": condition,
                "n_images": metrics.get("n_images", 0),
                "shape_accuracy": metrics.get("shape_accuracy", 0),
                "texture_accuracy": metrics.get("texture_accuracy", 0),
                "cue_accuracy": metrics.get("cue_accuracy", 0),
                "shape_bias": metrics.get("shape_bias", 0),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = self.results_dir / "comparison_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"\nComparison summary saved to {csv_path}")
        print("\n" + df.to_string(index=False))
