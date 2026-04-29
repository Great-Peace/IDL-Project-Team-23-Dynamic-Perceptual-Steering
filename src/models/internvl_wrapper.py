"""
internvl_wrapper.py
====================
Wrapper around InternVL-3 (8B) for the Dynamic Perceptual Steering project.

InternVL-3 is a state-of-the-art open-source Vision-Language Model.
HuggingFace: OpenGVLab/InternVL3-8B

This wrapper provides:
  1. Simple inference with any prompt
  2. Token logit extraction for confidence probing (Phase 3C)
  3. Batch inference support
  4. Prompt formatting per InternVL's expected input format

Usage:
    model = InternVLWrapper(config)
    response = model.generate(image, prompt)
    logits = model.get_token_logits(image, prompt, target_tokens)
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
    BitsAndBytesConfig,
)
import torchvision.transforms as T
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Image Preprocessing for InternVL-3
# ─────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size: int = 448) -> T.Compose:
    """Build the image preprocessing transform for InternVL-3."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_image(image_or_path: Union[str, Image.Image],
               input_size: int = 448) -> torch.Tensor:
    """
    Load and preprocess a single image for InternVL-3.

    Args:
        image_or_path : PIL Image or path to image file
        input_size    : Image size expected by InternVL-3 (default 448)

    Returns:
        Preprocessed tensor of shape (1, 3, H, W)
    """
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path).convert("RGB")
    else:
        image = image_or_path.convert("RGB")

    transform = build_transform(input_size)
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values


# ─────────────────────────────────────────────────────────────
# InternVL Wrapper
# ─────────────────────────────────────────────────────────────

class InternVLWrapper:
    """
    Wrapper for InternVL-3 (8B) Vision-Language Model.

    Handles:
    - Loading the model with 4-bit quantization (for PSC GPU memory)
    - Image preprocessing
    - Single and batch inference
    - Token logit extraction for confidence probing
    """

    MODEL_ID = "OpenGVLab/InternVL3-8B"

    def __init__(self, config: dict):
        self.config = config
        model_config = config.get("model", {})

        self.model_name = model_config.get("name", self.MODEL_ID)
        self.max_new_tokens = model_config.get("max_new_tokens", 512)
        self.temperature = model_config.get("temperature", 0.0)
        self.use_4bit = model_config.get("load_in_4bit", True)
        self.cache_dir = self._resolve_cache_dir(
            config["paths"].get("model_cache", None)
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading InternVL-3 on {self.device}...")

        self.model, self.tokenizer = self._load_model()
        logger.info("InternVL-3 loaded successfully.")

    def _resolve_cache_dir(self, cache_dir: Optional[str]) -> Optional[str]:
        """Resolve environment variables in cache paths and ignore empty placeholders."""
        if not cache_dir:
            return None
        resolved = os.path.expandvars(os.path.expanduser(str(cache_dir))).strip()
        if not resolved or resolved == "${HF_HOME}":
            return os.environ.get("HF_HOME") or None
        return resolved

    def _load_model(self):
        """Load InternVL-3 with optional 4-bit quantization."""
        # Configure 4-bit quantization for memory efficiency on PSC
        if self.use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Using 4-bit quantization (BitsAndBytes NF4).")
        else:
            bnb_config = None
            if not torch.cuda.is_available():
                logger.warning("CUDA not available. Running on CPU — this will be slow.")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir
        )

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "cache_dir": self.cache_dir,
        }

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
            model_kwargs["device_map"] = "auto"

        model = AutoModel.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        model.eval()

        return model, tokenizer

    def _format_prompt(self, prompt: str) -> str:
        """
        Format the prompt using InternVL-3's expected conversation format.
        InternVL-3 uses a specific template for visual question answering.
        """
        # InternVL-3 uses <image> token to indicate image position
        # The conversation format wraps the prompt in a human turn
        formatted = f"<image>\n{prompt}"
        return formatted

    @torch.no_grad()
    def generate(self, image: Union[str, Image.Image],
                 prompt: str,
                 return_full_output: bool = False) -> str:
        """
        Generate a text response for a given image and prompt.

        Args:
            image             : PIL Image or image path
            prompt            : Text prompt
            return_full_output: If True, return full generation output dict

        Returns:
            Generated text response as string
        """
        # Preprocess image
        pixel_values = load_image(image).to(self.device)
        if self.use_4bit:
            pixel_values = pixel_values.to(torch.bfloat16)

        # Format prompt
        formatted_prompt = self._format_prompt(prompt)

        # Generation configuration
        gen_config = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_config["temperature"] = self.temperature

        # Generate response
        try:
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=gen_config
            )
        except AttributeError:
            # Fallback if .chat() not available — use direct generation
            response = self._generate_direct(pixel_values, formatted_prompt, gen_config)

        return response

    def _generate_direct(self, pixel_values: torch.Tensor,
                          prompt: str,
                          gen_config: dict) -> str:
        """
        Direct generation fallback using raw tokenizer + model.forward().
        Used when the .chat() interface is not available.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            pixel_values=pixel_values,
            **gen_config
        )

        # Decode only the new tokens (skip input prompt tokens)
        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    @torch.no_grad()
    def get_token_logits(self, image: Union[str, Image.Image],
                          prompt: str,
                          target_tokens: List[str]) -> Dict[str, float]:
        """
        Extract logit confidence scores for specific target tokens.

        This is the core of Phase 3C — Mechanistic Confidence Probing.

        We ask: when shown a cue-conflict image, how confident is the model
        in predicting a shape token (e.g. "bowl") vs a texture/cultural
        token (e.g. "kente")? If the shape token has near-certainty
        confidence and the cultural token near-zero, the model is
        suppressing the cultural information.

        Args:
            image         : PIL Image or image path
            prompt        : Text prompt
            target_tokens : List of tokens to get confidence for
                           e.g. ["bowl", "kente", "building", "mudcloth"]

        Returns:
            Dict mapping each target token to its probability [0, 1]
        """
        # Preprocess image
        pixel_values = load_image(image).to(self.device)
        if self.use_4bit:
            pixel_values = pixel_values.to(torch.bfloat16)

        # Tokenize prompt
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Forward pass to get logits
        try:
            outputs = self.model(
                **inputs,
                pixel_values=pixel_values,
                return_dict=True,
                output_hidden_states=False
            )
            # logits shape: (batch, seq_len, vocab_size)
            next_token_logits = outputs.logits[:, -1, :]  # last position

            # Convert to probabilities via softmax
            probs = torch.softmax(next_token_logits, dim=-1)

            # Get probability for each target token
            result = {}
            for token_str in target_tokens:
                # Tokenize the target token
                token_ids = self.tokenizer.encode(
                    token_str, add_special_tokens=False
                )
                if len(token_ids) == 0:
                    result[token_str] = 0.0
                    continue
                # Use the first subtoken (main token)
                token_id = token_ids[0]
                if token_id < probs.shape[-1]:
                    result[token_str] = float(probs[0, token_id].cpu())
                else:
                    result[token_str] = 0.0

            return result

        except Exception as e:
            logger.error(f"Logit extraction failed: {e}")
            # Return zeros if extraction fails
            return {token: 0.0 for token in target_tokens}

    @torch.no_grad()
    def generate_batch(self, images: List[Union[str, Image.Image]],
                       prompts: List[str],
                       show_progress: bool = True) -> List[str]:
        """
        Generate responses for a batch of image-prompt pairs.

        Note: InternVL-3 with 4-bit quantization typically runs
        one image at a time. We process sequentially but with
        progress tracking.

        Args:
            images       : List of PIL images or paths
            prompts      : Corresponding list of prompts
            show_progress: Show tqdm progress bar

        Returns:
            List of text responses
        """
        assert len(images) == len(prompts), \
            f"Images ({len(images)}) and prompts ({len(prompts)}) must match."

        responses = []
        iterator = zip(images, prompts)

        if show_progress:
            iterator = tqdm(iterator, total=len(images), desc="Generating responses")

        for image, prompt in iterator:
            try:
                response = self.generate(image, prompt)
                responses.append(response)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                responses.append("")

        return responses

    def get_model_info(self) -> Dict:
        """Return information about the loaded model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": self.model_name,
            "device": self.device,
            "quantization": "4bit" if self.use_4bit else "none",
            "total_parameters": f"{total_params / 1e9:.2f}B",
        }
