"""
dual_lens.py
=============
Implements the Sequential Dual-Lens steering pipeline.

The core contribution of this project:
  Stage 1 (Structural Lens): identify WHAT the object IS (shape)
  Stage 2 (Cultural Lens):   identify WHERE it comes FROM (texture/culture)

This two-stage approach mirrors how a culturally-informed human
might examine an unfamiliar object: first understand its function,
then investigate its origin.

Usage:
    dual_lens = DualLensSteering(model, prompts, config)
    result = dual_lens.analyze(image, shape_label, texture_label)
"""

import logging
from typing import Dict, Optional, Union

from PIL import Image

from ..models.internvl_wrapper import InternVLWrapper
from .prompts import PromptLibrary

logger = logging.getLogger(__name__)


class DualLensSteering:
    """
    Implements the Sequential Dual-Lens prompting pipeline.

    The pipeline:
    ┌─────────────────────────────────────────────────┐
    │  Input: cue-conflict image                      │
    │                                                 │
    │  Stage 1 — STRUCTURAL LENS                      │
    │  Prompt: "Based on shape alone, what is this?"  │
    │  Output: shape_response (e.g. "bowl")           │
    │                                                 │
    │  Stage 2 — CULTURAL LENS                        │
    │  Prompt: "Based on texture/patterns, where      │
    │          is this from?" + context from Stage 1  │
    │  Output: cultural_response (e.g. "kente/ghana") │
    │                                                 │
    │  Final: Combined identity + cultural origin     │
    └─────────────────────────────────────────────────┘
    """

    def __init__(self, model: InternVLWrapper,
                 prompts: PromptLibrary,
                 config: dict):
        self.model = model
        self.prompts = prompts
        self.config = config

    def analyze(self, image: Union[str, Image.Image],
                shape_label: str,
                texture_label: str,
                record_metadata: Optional[Dict] = None) -> Dict:
        """
        Run the full Sequential Dual-Lens analysis on a single image.

        Args:
            image            : PIL image or path
            shape_label      : ground truth shape label (for evaluation)
            texture_label    : ground truth texture label (for evaluation)
            record_metadata  : optional metadata from the dataset record

        Returns:
            Dict with responses from both stages and combined output
        """
        # ── Stage 1: Structural Lens ──────────────────────────────
        stage1_prompt = self.prompts.get_text("sequential_stage1")
        stage1_response = self.model.generate(image, stage1_prompt)

        # ── Stage 2: Cultural Lens ────────────────────────────────
        # Build stage 2 prompt that incorporates stage 1 knowledge
        # This creates a coherent two-turn dialogue
        stage2_prompt = self._build_stage2_prompt(
            stage1_response, self.prompts.get_text("sequential_stage2")
        )
        stage2_response = self.model.generate(image, stage2_prompt)

        # ── Combined Response ─────────────────────────────────────
        combined = self._combine_responses(stage1_response, stage2_response)

        result = {
            "image_path": image if isinstance(image, str) else "",
            "shape_label": shape_label,
            "texture_label": texture_label,
            "prompt_type": "sequential",
            # Stage outputs
            "stage1_prompt": stage1_prompt,
            "stage1_response": stage1_response,
            "stage2_prompt": stage2_prompt,
            "stage2_response": stage2_response,
            # Combined for metric evaluation
            "response": combined,
            # Metadata
            **(record_metadata or {})
        }

        return result

    def _build_stage2_prompt(self, stage1_response: str,
                              base_stage2_prompt: str) -> str:
        """
        Build Stage 2 prompt incorporating Stage 1 findings.

        By telling Stage 2 what Stage 1 found, we prevent the model
        from simply repeating the shape answer and force it to
        reason specifically about cultural texture.
        """
        # Clean up stage 1 response
        stage1_clean = stage1_response.strip().rstrip(".")

        contextual_prompt = (
            f"You already identified this object as: {stage1_clean}. "
            f"\n\n"
            f"{base_stage2_prompt}"
        )
        return contextual_prompt

    def _combine_responses(self, stage1: str, stage2: str) -> str:
        """
        Combine Stage 1 and Stage 2 responses for metric evaluation.
        The combined response contains both the shape and cultural info.
        """
        return f"{stage1} | {stage2}"

    def run_batch(self, dataset) -> list:
        """
        Run Sequential Dual-Lens on all records in the dataset.

        Returns:
            List of result dicts
        """
        from tqdm import tqdm

        results = []
        logger.info(f"Running Sequential Dual-Lens on {len(dataset)} images...")

        for record in tqdm(dataset, desc="Dual-Lens Steering"):
            try:
                result = self.analyze(
                    image=record.load_image(),
                    shape_label=record.shape_label,
                    texture_label=record.texture_label,
                    record_metadata={
                        "category": record.category,
                        "region": record.region,
                        "is_famous": record.is_famous,
                        "source": record.source,
                        "image_path": record.image_path
                    }
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Dual-Lens failed for {record.image_path}: {e}")

        logger.info(f"Dual-Lens complete. {len(results)} records processed.")
        return results
