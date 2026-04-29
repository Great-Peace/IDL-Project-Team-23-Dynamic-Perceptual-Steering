"""
apo.py
=======
Automated Prompt Optimization (APO) loop for Phase 3B.

Mirrors the approach in Gavrikov et al. (2025) Section 3.1.1,
adapted for African cultural context using an open-source
optimizer (Mistral-7B-Instruct).

The APO loop:
  1. Start with hand-crafted cultural prompts as seed
  2. Evaluate each prompt on the full dataset
  3. Report shape_bias and texture_accuracy to the optimizer LLM
  4. Ask optimizer to generate better prompts
  5. Repeat until convergence or max iterations

Optimization objective:
  MAXIMIZE texture/cultural recognition (decrease shape bias)
  SUBJECT TO: functional accuracy >= 75%

This is directly analogous to Gavrikov's texture-bias optimization
but in the cultural domain: we want the model to prioritize
African cultural cues over shape cues.

Usage:
    apo = AutomatedPromptOptimizer(model, config)
    best_prompt, history = apo.optimize(dataset)
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

from ..models.internvl_wrapper import InternVLWrapper
from ..evaluation.metrics import compute_all_metrics, parse_decision

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Prompt Candidate
# ─────────────────────────────────────────────────────────────

class PromptCandidate:
    """Represents a candidate prompt with its evaluation scores."""

    def __init__(self, prompt_text: str, iteration: int):
        self.text = prompt_text
        self.iteration = iteration
        self.shape_bias: Optional[float] = None
        self.texture_accuracy: Optional[float] = None
        self.shape_accuracy: Optional[float] = None
        self.cue_accuracy: Optional[float] = None
        self.evaluated = False

    def to_dict(self) -> Dict:
        return {
            "iteration": self.iteration,
            "prompt": self.text,
            "shape_bias": self.shape_bias,
            "texture_accuracy": self.texture_accuracy,
            "shape_accuracy": self.shape_accuracy,
            "cue_accuracy": self.cue_accuracy,
        }


# ─────────────────────────────────────────────────────────────
# APO Optimizer
# ─────────────────────────────────────────────────────────────

class AutomatedPromptOptimizer:
    """
    Uses an open-source LLM (Mistral-7B-Instruct) as an optimizer
    to automatically discover better cultural steering prompts.

    The optimizer receives feedback about each prompt's performance
    and generates improved prompts in a closed loop.

    Based on Yang et al. (2024) "Large Language Models as Optimizers"
    as used by Gavrikov et al. (2025).
    """

    OPTIMIZER_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

    # Instruction given to the optimizer LLM
    OPTIMIZER_SYSTEM_PROMPT = """You are an expert in African cultural heritage and AI prompt engineering.
Your task is to help design prompts that maximize a Vision-Language Model's ability to 
recognize African cultural textures, patterns, and artistic traditions in images.

The model tends to focus on object shapes and ignore cultural texture details (this is called "perceptual erasure").
You need to design prompts that force the model to notice:
- Traditional weaving patterns (kente, mudcloth, adire, raffia, etc.)
- Regional artistic motifs and symbols (adinkra, geometric patterns, etc.)  
- Material culture (mud-brick, organic dyes, hand-woven techniques)
- Geographic and ethnic origins of visual styles

CONSTRAINTS:
- The prompt should NOT compromise the model's ability to identify what the object IS
- Keep prompts under 100 words
- Be specific about what visual features to examine
- Do NOT use generic phrases like "pay attention to" — be precise about what to look for

When I tell you a prompt's performance, generate an improved version.
Output ONLY the new prompt text on a single line starting with: PROMPT: """

    def __init__(self, vlm: InternVLWrapper, config: dict):
        self.vlm = vlm
        self.config = config
        apo_config = config.get("apo", {})

        self.optimizer_model_name = apo_config.get(
            "optimizer_model", self.OPTIMIZER_MODEL
        )
        self.max_iterations = apo_config.get("max_iterations", 20)
        self.candidates_per_iter = apo_config.get("candidates_per_iteration", 5)
        self.convergence_threshold = apo_config.get("convergence_threshold", 0.005)
        self.min_functional_accuracy = apo_config.get("min_functional_accuracy", 0.75)
        self.use_4bit = apo_config.get("load_in_4bit", True)
        self.cache_dir = self._resolve_cache_dir(
            config["paths"].get("model_cache", None)
        )

        # Output directory for APO results
        self.output_dir = Path(config["paths"]["apo_prompts"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # History of all evaluated prompts
        self.history: List[PromptCandidate] = []

        # Load optimizer model
        logger.info(f"Loading APO optimizer: {self.optimizer_model_name}")
        self.optimizer_tokenizer, self.optimizer_model = self._load_optimizer()
        logger.info("APO optimizer loaded.")

    def _resolve_cache_dir(self, cache_dir: Optional[str]) -> Optional[str]:
        """Resolve environment variables in cache paths and ignore empty placeholders."""
        if not cache_dir:
            return None
        resolved = os.path.expandvars(os.path.expanduser(str(cache_dir))).strip()
        if not resolved or resolved == "${HF_HOME}":
            return os.environ.get("HF_HOME") or None
        return resolved

    def _load_optimizer(self):
        """Load the Mistral-7B optimizer with 4-bit quantization."""
        if self.use_4bit and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
        else:
            bnb_config = None

        tokenizer = AutoTokenizer.from_pretrained(
            self.optimizer_model_name,
            cache_dir=self.cache_dir
        )

        kwargs = {
            "device_map": "auto",
            "cache_dir": self.cache_dir
        }
        if bnb_config:
            kwargs["quantization_config"] = bnb_config
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(
            self.optimizer_model_name, **kwargs
        )
        model.eval()
        return tokenizer, model

    def evaluate_prompt(self, prompt_text: str,
                         dataset,
                         candidate: PromptCandidate) -> PromptCandidate:
        """
        Evaluate a single candidate prompt on the full dataset.

        Runs inference on all images and computes:
        - texture_accuracy (what we want to maximize)
        - shape_accuracy (what we must not drop too much)
        - shape_bias, cue_accuracy
        """
        results = []

        for record in tqdm(dataset,
                           desc=f"APO eval iter {candidate.iteration}",
                           leave=False):
            try:
                response = self.vlm.generate(record.load_image(), prompt_text)
                results.append({
                    "response": response,
                    "shape_label": record.shape_label,
                    "texture_label": record.texture_label,
                    "category": record.category,
                    "region": record.region,
                    "is_famous": record.is_famous,
                    "prompt_type": "apo_candidate",
                })
            except Exception as e:
                logger.debug(f"APO inference error: {e}")

        if not results:
            logger.warning("No results from APO candidate evaluation.")
            return candidate

        # Compute metrics
        metrics = compute_all_metrics(results)

        candidate.shape_bias = metrics.get("shape_bias", 0.5)
        candidate.texture_accuracy = metrics.get("texture_accuracy", 0.0)
        candidate.shape_accuracy = metrics.get("shape_accuracy", 0.0)
        candidate.cue_accuracy = metrics.get("cue_accuracy", 0.0)
        candidate.evaluated = True

        logger.info(
            f"  APO iter {candidate.iteration}: "
            f"shape_bias={candidate.shape_bias:.3f}, "
            f"texture_acc={candidate.texture_accuracy:.3f}, "
            f"shape_acc={candidate.shape_accuracy:.3f}"
        )

        return candidate

    @torch.no_grad()
    def _generate_new_prompt(self, conversation_history: str,
                              best_so_far: Optional[PromptCandidate]) -> List[str]:
        """
        Ask the optimizer LLM to generate improved prompts.

        Formats the conversation history as feedback and extracts
        new PROMPT: lines from the optimizer's response.
        """
        # Build the optimizer input
        instruction = (
            f"{self.OPTIMIZER_SYSTEM_PROMPT}\n\n"
            f"Here are the prompts tested so far and their results:\n"
            f"{conversation_history}\n\n"
        )

        if best_so_far:
            instruction += (
                f"The best prompt so far achieves:\n"
                f"  texture_accuracy = {best_so_far.texture_accuracy:.3f}\n"
                f"  shape_accuracy   = {best_so_far.shape_accuracy:.3f}\n"
                f"  shape_bias       = {best_so_far.shape_bias:.3f}\n\n"
            )

        instruction += (
            f"Generate {self.candidates_per_iter} improved prompts. "
            f"Each must start with 'PROMPT: ' on its own line. "
            f"Focus on MAXIMIZING texture/cultural recognition while "
            f"keeping shape/functional accuracy above {self.min_functional_accuracy}."
        )

        # Tokenize and generate
        messages = [
            {"role": "user", "content": instruction}
        ]

        # Mistral instruction format
        input_text = self.optimizer_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.optimizer_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.optimizer_model.device)

        outputs = self.optimizer_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][input_length:]
        response_text = self.optimizer_tokenizer.decode(
            new_tokens, skip_special_tokens=True
        )

        # Extract all PROMPT: lines
        prompts = []
        for line in response_text.split("\n"):
            line = line.strip()
            if line.startswith("PROMPT:"):
                prompt_text = line[len("PROMPT:"):].strip()
                if prompt_text and len(prompt_text) > 10:
                    prompts.append(prompt_text)

        logger.info(f"Optimizer generated {len(prompts)} candidate prompts.")
        return prompts

    def optimize(self, dataset,
                  seed_prompts: Optional[List[str]] = None) -> Tuple[str, List[Dict]]:
        """
        Run the full APO optimization loop.

        Args:
            dataset      : AfricanCulturalDataset to evaluate on
            seed_prompts : Initial prompts to start from.
                          If None, uses the default cultural prompts.

        Returns:
            (best_prompt_text, history_list)
        """
        logger.info("="*60)
        logger.info("Starting Automated Prompt Optimization (APO)")
        logger.info(f"Max iterations: {self.max_iterations}")
        logger.info(f"Convergence threshold: {self.convergence_threshold}")
        logger.info(f"Min functional accuracy: {self.min_functional_accuracy}")
        logger.info("="*60)

        # Seed prompts
        if seed_prompts is None:
            seed_prompts = [
                "Analyze the specific cultural textures, materials, patterns, "
                "and regional artistic style. What African culture does this come from?",

                "Focus on the surface patterns, weaving techniques, and decorative "
                "motifs. Which African cultural tradition produced these visual elements?",

                "Examine the repetitive geometric patterns, color palette, and "
                "textile construction. Name the specific African ethnic group or "
                "geographic region this artwork represents.",
            ]

        best_candidate: Optional[PromptCandidate] = None
        conversation_history = ""
        iteration = 0

        # ── Evaluate seed prompts ─────────────────────────────────
        logger.info("\nPhase APO-0: Evaluating seed prompts...")
        for seed_text in seed_prompts:
            candidate = PromptCandidate(seed_text, iteration=0)
            candidate = self.evaluate_prompt(seed_text, dataset, candidate)

            # Only keep candidates that maintain functional accuracy
            if candidate.shape_accuracy >= self.min_functional_accuracy:
                self.history.append(candidate)
                conversation_history += self._format_candidate_for_history(candidate)

                if (best_candidate is None or
                        candidate.texture_accuracy > best_candidate.texture_accuracy):
                    best_candidate = candidate

        if best_candidate is None:
            logger.warning(
                "All seed prompts failed the functional accuracy constraint. "
                "Lowering constraint to proceed."
            )
            best_candidate = max(self.history,
                                  key=lambda c: c.texture_accuracy,
                                  default=None)

        # ── Main APO Loop ─────────────────────────────────────────
        prev_best_texture = best_candidate.texture_accuracy if best_candidate else 0.0

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\nAPO Iteration {iteration}/{self.max_iterations}")
            logger.info(f"  Current best texture_accuracy: "
                        f"{best_candidate.texture_accuracy:.4f}")

            # Generate new candidate prompts
            new_prompt_texts = self._generate_new_prompt(
                conversation_history, best_candidate
            )

            if not new_prompt_texts:
                logger.warning("Optimizer produced no valid prompts. Stopping.")
                break

            # Evaluate each candidate
            for prompt_text in new_prompt_texts[:self.candidates_per_iter]:
                candidate = PromptCandidate(prompt_text, iteration=iteration)
                candidate = self.evaluate_prompt(prompt_text, dataset, candidate)
                self.history.append(candidate)
                conversation_history += self._format_candidate_for_history(candidate)

                # Update best if this candidate is better AND meets constraint
                if (candidate.evaluated and
                        candidate.shape_accuracy >= self.min_functional_accuracy and
                        candidate.texture_accuracy > best_candidate.texture_accuracy):
                    best_candidate = candidate
                    logger.info(f"  ✓ New best! texture_acc = "
                                f"{candidate.texture_accuracy:.4f}")

            # Check convergence
            improvement = best_candidate.texture_accuracy - prev_best_texture
            if improvement < self.convergence_threshold and iteration > 3:
                logger.info(
                    f"Converged at iteration {iteration} "
                    f"(improvement {improvement:.4f} < threshold "
                    f"{self.convergence_threshold})"
                )
                break

            prev_best_texture = best_candidate.texture_accuracy

            # Save intermediate results
            self._save_history()

        # ── Final Results ─────────────────────────────────────────
        logger.info("\n" + "="*60)
        logger.info("APO COMPLETE")
        if best_candidate:
            logger.info(f"Best prompt found:")
            logger.info(f"  {best_candidate.text}")
            logger.info(f"  texture_accuracy = {best_candidate.texture_accuracy:.4f}")
            logger.info(f"  shape_accuracy   = {best_candidate.shape_accuracy:.4f}")
            logger.info(f"  shape_bias       = {best_candidate.shape_bias:.4f}")
        logger.info("="*60)

        # Final save
        self._save_history()

        history_dicts = [c.to_dict() for c in self.history]
        best_text = best_candidate.text if best_candidate else seed_prompts[0]

        return best_text, history_dicts

    def _format_candidate_for_history(self, candidate: PromptCandidate) -> str:
        """Format a candidate as a string for the optimizer's conversation history."""
        return (
            f"\nPrompt (iter {candidate.iteration}): {candidate.text}\n"
            f"  → texture_accuracy={candidate.texture_accuracy:.3f}, "
            f"shape_accuracy={candidate.shape_accuracy:.3f}, "
            f"shape_bias={candidate.shape_bias:.3f}\n"
        )

    def _save_history(self):
        """Save APO history to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"apo_history_{timestamp}.json"
        history_dicts = [c.to_dict() for c in self.history]
        with open(output_file, "w") as f:
            json.dump(history_dicts, f, indent=2)
        logger.debug(f"APO history saved to {output_file}")

    def get_top_prompts(self, n: int = 5) -> List[PromptCandidate]:
        """Return the top N prompts ranked by texture_accuracy."""
        evaluated = [c for c in self.history if c.evaluated]
        sorted_candidates = sorted(
            evaluated,
            key=lambda c: c.texture_accuracy,
            reverse=True
        )
        return sorted_candidates[:n]
