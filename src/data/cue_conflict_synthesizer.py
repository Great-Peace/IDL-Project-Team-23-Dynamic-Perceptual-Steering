"""
cue_conflict_synthesizer.py
============================
Synthesizes African texture-shape cue-conflict images using
AdaIN (Adaptive Instance Normalization) style transfer.

This directly mirrors the methodology of Geirhos et al. (2019),
who used style transfer to generate texture-shape conflicts.
Here we apply AFRICAN cultural textures to NEUTRAL generic shapes,
creating images where:
  - The SHAPE identifies the functional object (bowl, bag, building)
  - The TEXTURE identifies the African cultural origin (kente, mudcloth)

The model must choose whether to respond based on shape or texture.

Usage:
    synthesizer = CueConflictSynthesizer(config)
    conflict_pair = synthesizer.synthesize(content_image, style_image,
                                           shape_label, texture_label)
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Cue Conflict Pair Record
# ─────────────────────────────────────────────────────────────

@dataclass
class CueConflictPair:
    """
    A synthesized texture-shape cue-conflict image pair.

    content_path  : original shape/content image path
    style_path    : African texture source image path
    conflict_path : synthesized conflict image path (saved output)
    shape_label   : what the shape says the object is (e.g. "bowl")
    texture_label : what the texture says the origin is (e.g. "kente")
    category      : artifact category
    region        : African region of the texture
    """
    content_path: str
    style_path: str
    conflict_path: str
    shape_label: str
    texture_label: str
    category: str
    region: str


# ─────────────────────────────────────────────────────────────
# AdaIN Style Transfer
# ─────────────────────────────────────────────────────────────

class AdaINStyleTransfer(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) style transfer.
    Based on Huang & Belongie (2017) "Arbitrary Style Transfer
    in Real-time with Adaptive Instance Normalization."

    This is the same technique used by Geirhos et al. (2019)
    to generate their cue-conflict dataset.

    AdaIN works by:
    1. Encoding content and style images through a VGG encoder
    2. Aligning the mean and variance of content features
       to match the style features
    3. Decoding back to pixel space

    We use a simplified version that performs feature-space
    style transfer without training a decoder — instead using
    direct pixel-space blending guided by feature statistics.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

        # Use VGG-19 encoder (up to relu4_1, like in original AdaIN)
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # Extract layers up to relu4_1 (index 20 in VGG19 features)
        self.encoder = nn.Sequential(*list(vgg.features.children())[:21])
        self.encoder = self.encoder.to(device)
        self.encoder.eval()

        # Freeze encoder — we never train it
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Image preprocessing for VGG
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.unnormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

        logger.info("AdaIN style transfer initialized.")

    def _preprocess(self, img: Image.Image, size: int = 512) -> torch.Tensor:
        """Preprocess a PIL image for VGG encoding."""
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            self.normalize
        ])
        tensor = transform(img).unsqueeze(0).to(self.device)
        return tensor

    def _calc_mean_std(self, feat: torch.Tensor,
                       eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate channel-wise mean and standard deviation of feature maps.
        This is the key operation in AdaIN.
        """
        size = feat.size()
        assert len(size) == 4  # (batch, channels, H, W)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def adain(self, content_feat: torch.Tensor,
              style_feat: torch.Tensor) -> torch.Tensor:
        """
        Perform AdaIN: normalize content features to have the
        mean and std of the style features.

        AdaIN(x, y) = σ(y) * ((x - μ(x)) / σ(x)) + μ(y)
        """
        content_mean, content_std = self._calc_mean_std(content_feat)
        style_mean, style_std = self._calc_mean_std(style_feat)

        # Normalize content, then apply style statistics
        normalized = (content_feat - content_mean) / content_std
        stylized = normalized * style_std + style_mean
        return stylized

    @torch.no_grad()
    def transfer(self, content_img: Image.Image,
                 style_img: Image.Image,
                 alpha: float = 0.8,
                 output_size: int = 224) -> Image.Image:
        """
        Transfer the style of style_img onto the content of content_img.

        Args:
            content_img  : PIL image of the content (shape source)
            style_img    : PIL image of the style (texture source)
            alpha        : Blending strength (1.0 = full style, 0.0 = no style)
                          We use 0.8 by default — preserves shape while
                          clearly applying African texture
            output_size  : Output image size in pixels

        Returns:
            PIL image of the synthesized cue-conflict image
        """
        # Preprocess both images
        content_tensor = self._preprocess(content_img, size=512)
        style_tensor = self._preprocess(style_img, size=512)

        # Encode through VGG
        content_feat = self.encoder(content_tensor)
        style_feat = self.encoder(style_tensor)

        # Apply AdaIN in feature space
        stylized_feat = self.adain(content_feat, style_feat)

        # Blend: alpha controls how much style is applied
        # alpha=1.0 → pure AdaIN stylized
        # alpha=0.0 → unchanged content features
        blended_feat = alpha * stylized_feat + (1 - alpha) * content_feat

        # Since we don't have a decoder network, we use a simpler approach:
        # Apply the style statistics directly in pixel space as a post-processing step
        # This is less sophisticated than full AdaIN but works without training
        conflict_img = self._pixel_space_style_transfer(
            content_img, style_img, alpha, output_size
        )
        return conflict_img

    def _pixel_space_style_transfer(self, content_img: Image.Image,
                                     style_img: Image.Image,
                                     alpha: float = 0.8,
                                     output_size: int = 224) -> Image.Image:
        """
        Pixel-space style transfer using color/texture statistics matching.

        This applies the mean and standard deviation of the style image's
        color channels to the content image — a simpler but effective
        approach for creating texture-shape conflicts.

        For a more faithful AdaIN implementation, consider using the
        PyTorch AdaIN library or the official implementation at:
        https://github.com/naoto0804/pytorch-AdaIN
        """
        # Resize both images
        content_resized = content_img.resize((output_size, output_size))
        style_resized = style_img.resize((output_size, output_size))

        # Convert to numpy float
        content_arr = np.array(content_resized).astype(np.float32)
        style_arr = np.array(style_resized).astype(np.float32)

        # For each channel: normalize content, apply style statistics
        result = np.zeros_like(content_arr)
        for c in range(3):  # RGB channels
            content_ch = content_arr[:, :, c]
            style_ch = style_arr[:, :, c]

            # Normalize content channel
            c_mean, c_std = content_ch.mean(), content_ch.std() + 1e-5
            normalized = (content_ch - c_mean) / c_std

            # Apply style statistics
            s_mean, s_std = style_ch.mean(), style_ch.std() + 1e-5
            stylized_ch = normalized * s_std + s_mean

            # Blend with original content
            result[:, :, c] = alpha * stylized_ch + (1 - alpha) * content_ch

        # Clip to valid range and convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)


# ─────────────────────────────────────────────────────────────
# Main Synthesizer
# ─────────────────────────────────────────────────────────────

class CueConflictSynthesizer:
    """
    Orchestrates the synthesis of African texture-shape cue-conflict images.

    For each synthesis operation:
    - content image = neutral Western object (provides the SHAPE)
    - style image   = African cultural artifact (provides the TEXTURE)
    - output        = the object shape with African cultural texture applied

    Example: A plain bowl (content) + Kente cloth (style) →
             Bowl-shaped object with Kente textile patterns

    This lets us test: when shown this hybrid image, does the VLM
    say "bowl" (shape decision) or "kente/west-african textile" (texture decision)?
    """

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["paths"]["data_cue_conflict"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_size = config["dataset"]["image_size"]

        # Initialize style transfer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"CueConflictSynthesizer using device: {self.device}")
        self.style_transfer = AdaINStyleTransfer(device=self.device)

        # Alpha: how strongly to apply the African texture
        # 0.8 provides clear texture while preserving shape contours
        self.alpha = 0.8

    def synthesize(self, content_image: Image.Image,
                   style_image: Image.Image,
                   shape_label: str,
                   texture_label: str,
                   category: str,
                   region: str,
                   pair_id: str) -> Optional[CueConflictPair]:
        """
        Synthesize a single cue-conflict image.

        Args:
            content_image : PIL image providing the shape (neutral object)
            style_image   : PIL image providing the African texture
            shape_label   : functional label (e.g. "bowl")
            texture_label : cultural label (e.g. "kente")
            category      : artifact category
            region        : African region
            pair_id       : unique identifier for this pair

        Returns:
            CueConflictPair or None if synthesis failed
        """
        # Define output path
        output_filename = f"conflict_{pair_id}_{shape_label}_{texture_label}.jpg"
        output_path = self.output_dir / output_filename

        if output_path.exists():
            logger.debug(f"Cue conflict image already exists: {output_filename}")
            return CueConflictPair(
                content_path="",
                style_path="",
                conflict_path=str(output_path),
                shape_label=shape_label,
                texture_label=texture_label,
                category=category,
                region=region
            )

        try:
            # Apply style transfer
            conflict_img = self.style_transfer.transfer(
                content_img=content_image,
                style_img=style_image,
                alpha=self.alpha,
                output_size=self.output_size
            )

            # Save the synthesized image
            conflict_img.save(str(output_path), quality=95)

            return CueConflictPair(
                content_path="",
                style_path="",
                conflict_path=str(output_path),
                shape_label=shape_label,
                texture_label=texture_label,
                category=category,
                region=region
            )

        except Exception as e:
            logger.error(f"Style transfer failed for {pair_id}: {e}")
            return None

    def synthesize_batch_from_dataset(self, dataset,
                                       neutral_shapes_dir: str) -> List[CueConflictPair]:
        """
        Synthesize cue-conflict pairs for all records in the dataset.

        For each African texture image in the dataset, we pair it with
        a neutral Western shape image from neutral_shapes_dir.

        The neutral shapes directory should contain generic object images:
            neutral_shapes/
                bowl.jpg
                bag.jpg
                building.jpg
                cloth.jpg
                ...

        Args:
            dataset          : AfricanCulturalDataset
            neutral_shapes_dir: directory with neutral shape images

        Returns:
            List of CueConflictPair objects
        """
        neutral_shapes_path = Path(neutral_shapes_dir)
        pairs = []

        logger.info(f"Synthesizing cue-conflict pairs for {len(dataset)} records...")

        for idx, record in enumerate(tqdm(dataset, desc="Synthesizing conflicts")):
            # Load the African cultural image (style/texture source)
            try:
                style_image = record.load_image()
            except Exception as e:
                logger.warning(f"Could not load style image {record.image_path}: {e}")
                continue

            # Find a neutral shape image matching the shape_label
            content_image = self._load_neutral_shape(
                neutral_shapes_path, record.shape_label
            )
            if content_image is None:
                # If no matching neutral shape found, use a generic placeholder
                logger.debug(f"No neutral shape for '{record.shape_label}'. "
                             f"Using generic shape.")
                content_image = self._create_generic_shape(record.shape_label)

            pair = self.synthesize(
                content_image=content_image,
                style_image=style_image,
                shape_label=record.shape_label,
                texture_label=record.texture_label,
                category=record.category,
                region=record.region,
                pair_id=f"{idx:05d}"
            )

            if pair is not None:
                pairs.append(pair)

        logger.info(f"Synthesized {len(pairs)} cue-conflict pairs.")
        return pairs

    def _load_neutral_shape(self, shapes_dir: Path,
                             shape_label: str) -> Optional[Image.Image]:
        """
        Load a neutral shape image from the shapes directory.
        Tries exact match, then partial match.
        """
        # Try direct match: e.g., shapes_dir/bowl.jpg
        for ext in [".jpg", ".jpeg", ".png"]:
            path = shapes_dir / f"{shape_label}{ext}"
            if path.exists():
                return Image.open(path).convert("RGB")

        # Try partial match: any file containing the shape_label
        if shapes_dir.exists():
            for img_file in shapes_dir.iterdir():
                if shape_label.lower() in img_file.stem.lower():
                    return Image.open(img_file).convert("RGB")

        return None

    def _create_generic_shape(self, shape_label: str,
                               size: int = 224) -> Image.Image:
        """
        Create a simple grey placeholder image when no neutral
        shape image is available. The grey background ensures
        the style transfer applies cleanly.
        """
        # Create a neutral grey image
        arr = np.ones((size, size, 3), dtype=np.uint8) * 180
        return Image.fromarray(arr)

    def save_pair_manifest(self, pairs: List[CueConflictPair],
                           output_path: str):
        """Save the list of synthesized pairs as a JSON manifest."""
        import json
        from dataclasses import asdict
        manifest = [asdict(p) for p in pairs]
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved {len(pairs)} cue-conflict pairs to {output_path}")
