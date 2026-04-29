"""
clip_baseline.py
=================
CLIP ViT-L/14 vision-only baseline for Phase 2.

Purpose: Isolate whether perceptual erasure of African textures
comes from the VISION ENCODER (CLIP) or the LLM component.

If CLIP also shows erasure → the problem is in visual encoding
If CLIP shows it but LLM suppresses it → problem is in language fusion
(this is what Gavrikov et al. found for texture/shape bias)

Usage:
    baseline = CLIPBaseline(config)
    decision = baseline.classify(image, shape_label, texture_label)
    shape_bias = baseline.compute_shape_bias(records)
"""

import logging
from typing import List, Dict, Tuple, Union

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# CLIP is installed from OpenAI's repo
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning(
        "CLIP not installed. Install with: "
        "pip install git+https://github.com/openai/CLIP.git"
    )


class CLIPBaseline:
    """
    CLIP ViT-L/14 vision-only baseline.

    Given a cue-conflict image and both a shape label and texture label,
    CLIP decides which one is more similar to the image in its
    embedding space. This reveals the encoder's inherent bias.

    Zero-shot classification: embed the image and both text labels,
    then assign the image to the closer label.
    """

    def __init__(self, config: dict):
        self.config = config
        clip_config = config.get("clip", {})
        self.model_name = clip_config.get("model_name", "ViT-L/14")
        self.device = clip_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        if not CLIP_AVAILABLE:
            raise ImportError(
                "Please install CLIP: "
                "pip install git+https://github.com/openai/CLIP.git"
            )

        logger.info(f"Loading CLIP {self.model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.model.eval()
        logger.info("CLIP loaded.")

    @torch.no_grad()
    def classify(self, image: Union[str, Image.Image],
                 shape_label: str,
                 texture_label: str) -> Dict:
        """
        Zero-shot classify a cue-conflict image as either
        shape_label or texture_label.

        Returns:
            Dict with:
              decision       : "shape" or "texture"
              shape_prob     : probability for shape label
              texture_prob   : probability for texture label
              shape_label    : the shape label string
              texture_label  : the texture label string
        """
        # Load and preprocess image
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        # Encode text labels
        # Use simple templates like "a photo of a {label}"
        text_inputs = clip.tokenize([
            f"a photo of a {shape_label}",
            f"a photo of {texture_label}"
        ]).to(self.device)

        # Get embeddings
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_inputs)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        shape_prob = float(similarities[0, 0].cpu())
        texture_prob = float(similarities[0, 1].cpu())

        decision = "shape" if shape_prob > texture_prob else "texture"

        return {
            "decision": decision,
            "shape_prob": shape_prob,
            "texture_prob": texture_prob,
            "shape_label": shape_label,
            "texture_label": texture_label
        }

    @torch.no_grad()
    def compute_shape_bias(self, records) -> Dict:
        """
        Compute Shape Bias over the full dataset using CLIP.

        Shape Bias = Shape Accuracy / (Shape Accuracy + Texture Accuracy)

        Returns:
            Dict with shape_bias, cue_accuracy, shape_accuracy, texture_accuracy
        """
        from tqdm import tqdm

        shape_correct = 0
        texture_correct = 0
        total_cue_accurate = 0

        for record in tqdm(records, desc="CLIP baseline"):
            try:
                result = self.classify(
                    record.load_image(),
                    record.shape_label,
                    record.texture_label
                )
                if result["decision"] == "shape":
                    shape_correct += 1
                    total_cue_accurate += 1
                elif result["decision"] == "texture":
                    texture_correct += 1
                    total_cue_accurate += 1
            except Exception as e:
                logger.debug(f"CLIP classification failed: {e}")
                continue

        if total_cue_accurate == 0:
            return {"shape_bias": 0.0, "cue_accuracy": 0.0,
                    "shape_accuracy": 0.0, "texture_accuracy": 0.0}

        n = len(list(records))
        cue_accuracy = total_cue_accurate / n
        shape_accuracy = shape_correct / n
        texture_accuracy = texture_correct / n
        shape_bias = shape_correct / total_cue_accurate

        logger.info(f"CLIP Shape Bias: {shape_bias:.3f}")
        logger.info(f"CLIP Cue Accuracy: {cue_accuracy:.3f}")

        return {
            "shape_bias": round(shape_bias, 4),
            "cue_accuracy": round(cue_accuracy, 4),
            "shape_accuracy": round(shape_accuracy, 4),
            "texture_accuracy": round(texture_accuracy, 4),
            "n_shape_decisions": shape_correct,
            "n_texture_decisions": texture_correct,
            "n_total": n
        }
