"""
prompts.py
===========
Central library of all prompts used in the experiments.

Prompts are organized by type:
  - Neutral baseline
  - Structural Lens (shape-steering)
  - Cultural Lens (texture/culture-steering)
  - Sequential (two-stage Dual-Lens)
  - APO-discovered prompts (populated after running APO)

Each prompt has a name, text, and description explaining its purpose.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Prompt:
    """A named, documented prompt."""
    name: str
    text: str
    description: str
    prompt_type: str  # neutral / structural / cultural / sequential / apo


class PromptLibrary:
    """
    Central registry of all prompts for the Dynamic Perceptual
    Steering experiments.

    Designed following the prompt structure from Gavrikov et al. (2025)
    and adapted for African cultural context.
    """

    def __init__(self, config: Optional[dict] = None):
        self._prompts: Dict[str, Prompt] = {}
        self._register_all_prompts(config)

    def _register_all_prompts(self, config: Optional[dict]):
        """Register all prompts from config and hardcoded defaults."""

        # ── Neutral Baseline ─────────────────────────────────────
        self.register(Prompt(
            name="neutral",
            text="What is in this image?",
            description=(
                "Neutral baseline. No steering toward shape or texture. "
                "Represents default VLM behavior. "
                "Expected to show high shape bias / perceptual erasure."
            ),
            prompt_type="neutral"
        ))

        # ── Structural Lens (Shape Steering) ─────────────────────
        self.register(Prompt(
            name="structural",
            text=(
                "Identify the primary object in this image based on its "
                "shape and physical structure alone. "
                "What is the functional category of this object? "
                "Answer with just the object type (e.g. 'bowl', 'bag', 'building')."
            ),
            description=(
                "Steers model toward shape/functional identity. "
                "Stage 1 of the Sequential Dual-Lens pipeline. "
                "Should maximize shape bias."
            ),
            prompt_type="structural"
        ))

        # ── Cultural Lens (Texture/Culture Steering) ──────────────
        self.register(Prompt(
            name="cultural",
            text=(
                "Analyze the specific cultural textures, materials, weaving "
                "patterns, decorative motifs, and regional artistic style "
                "visible in this image. "
                "What African culture, ethnic group, or geographic region "
                "does this object most likely originate from? "
                "Describe the specific cultural markers, traditional techniques, "
                "and artistic traditions you observe."
            ),
            description=(
                "Steers model toward texture/cultural recognition. "
                "Stage 2 of the Sequential Dual-Lens pipeline. "
                "Should recover African cultural knowledge erased by neutral prompt."
            ),
            prompt_type="cultural"
        ))

        # ── Sequential Stage 1 ────────────────────────────────────
        self.register(Prompt(
            name="sequential_stage1",
            text=(
                "Look at the shape and structure of the object in this image. "
                "Ignoring all surface patterns and colors, what type of object "
                "is this based on its form alone? "
                "Just name the object type."
            ),
            description=(
                "First stage of Sequential Dual-Lens. "
                "Explicitly directs model to ignore texture and focus on shape."
            ),
            prompt_type="sequential"
        ))

        # ── Sequential Stage 2 ────────────────────────────────────
        self.register(Prompt(
            name="sequential_stage2",
            text=(
                "Now, focusing entirely on the surface texture, patterns, "
                "weaving style, materials, and decorative elements — "
                "completely ignoring the shape of the object — "
                "what specific African cultural tradition, ethnic group, "
                "or geographic region produced these visual patterns?"
            ),
            description=(
                "Second stage of Sequential Dual-Lens. "
                "Explicitly directs model to ignore shape and focus on cultural texture."
            ),
            prompt_type="sequential"
        ))

        # ── Adversarial Texture Steering ──────────────────────────
        self.register(Prompt(
            name="adversarial_texture",
            text=(
                "This image shows an everyday Western object that has been "
                "decorated with traditional African patterns. "
                "Identify: (1) what the object is, and (2) which specific "
                "African textile tradition, cultural practice, or ethnic group "
                "the decorative patterns belong to."
            ),
            description=(
                "Phase 4 adversarial experiment. "
                "Applied to images of Western shapes with African textures. "
                "Tests whether model can recognize African culture even on "
                "strongly Western-encoded shapes."
            ),
            prompt_type="adversarial"
        ))

        # ── Alternative Cultural Prompts for APO Starting Points ─
        self.register(Prompt(
            name="cultural_geometric",
            text=(
                "Analyze the repetitive geometric motifs, weave patterns, "
                "color combinations, and symbolic designs in this image. "
                "Which African artistic tradition uses these specific "
                "visual elements?"
            ),
            description=(
                "Alternative cultural steering prompt focusing on geometry. "
                "May be more effective for textile patterns."
            ),
            prompt_type="cultural"
        ))

        self.register(Prompt(
            name="cultural_material",
            text=(
                "Examine the materials, construction techniques, and surface "
                "treatment visible in this image. "
                "What traditional African craft or manufacturing technique "
                "produced this? Name the specific cultural tradition."
            ),
            description=(
                "Alternative cultural steering prompt focusing on materials."
            ),
            prompt_type="cultural"
        ))

        self.register(Prompt(
            name="cultural_expert",
            text=(
                "As an expert in African art history and material culture, "
                "identify the cultural origin of the visual elements in this "
                "image. Reference specific ethnic groups, geographic regions, "
                "historical periods, and the names of traditional techniques "
                "or textile types you recognize."
            ),
            description=(
                "Expert persona prompt. May unlock deeper latent knowledge "
                "by assigning the model a cultural expert role."
            ),
            prompt_type="cultural"
        ))

        # Override with config prompts if provided
        if config and "prompts" in config:
            cfg_prompts = config["prompts"]
            for name, text in cfg_prompts.items():
                if name in self._prompts:
                    self._prompts[name].text = text

    def register(self, prompt: Prompt):
        """Register a prompt in the library."""
        self._prompts[prompt.name] = prompt

    def get(self, name: str) -> Prompt:
        """Retrieve a prompt by name."""
        if name not in self._prompts:
            raise KeyError(f"Prompt '{name}' not found. "
                           f"Available: {list(self._prompts.keys())}")
        return self._prompts[name]

    def get_text(self, name: str) -> str:
        """Get just the text of a prompt by name."""
        return self.get(name).text

    def get_all_cultural_prompts(self) -> List[Prompt]:
        """Return all cultural-type prompts (for APO initialization)."""
        return [p for p in self._prompts.values()
                if p.prompt_type == "cultural"]

    def get_experiment_prompts(self) -> Dict[str, str]:
        """
        Return the standard experiment prompts as a dict.
        This is what gets passed to Evaluator.run_all_conditions().
        """
        return {
            "neutral": self.get_text("neutral"),
            "structural": self.get_text("structural"),
            "cultural": self.get_text("cultural"),
            "sequential_stage1": self.get_text("sequential_stage1"),
        }

    def add_apo_prompt(self, name: str, text: str,
                        shape_bias: float, texture_accuracy: float):
        """
        Register a prompt discovered by the APO loop.
        Includes performance metadata.
        """
        self.register(Prompt(
            name=f"apo_{name}",
            text=text,
            description=(
                f"APO-discovered prompt. "
                f"Shape bias: {shape_bias:.3f}, "
                f"Texture accuracy: {texture_accuracy:.3f}"
            ),
            prompt_type="apo"
        ))

    def list_prompts(self):
        """Print all registered prompts."""
        print(f"\n{'='*60}")
        print(f"Registered Prompts ({len(self._prompts)} total)")
        print(f"{'='*60}")
        for name, prompt in self._prompts.items():
            print(f"\n[{name}] ({prompt.prompt_type})")
            print(f"  {prompt.text[:100]}...")
            print(f"  → {prompt.description[:80]}")
