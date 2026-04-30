"""Generates dynamic_perceptual_steering.ipynb from the project source files."""
import json
from pathlib import Path

ROOT = Path(__file__).parent


def read(p):
    return (ROOT / p).read_text(encoding="utf-8")


def strip_relative_imports(src: str) -> str:
    """Remove 'from . ...' and 'from .. ...' lines; all deps are defined globally above."""
    lines = []
    for line in src.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("from .") or stripped.startswith("from .."):
            lines.append("# " + line.rstrip() + "  # (defined globally in notebook)\n")
        else:
            lines.append(line)
    return "".join(lines)



# ── Load every source file ────────────────────────────────────────────────────
dataset_loader     = read("src/data/dataset_loader.py")
cue_conflict_synth = read("src/data/cue_conflict_synthesizer.py")
internvl_wrapper   = read("src/models/internvl_wrapper.py")
clip_baseline      = read("src/models/clip_baseline.py")
metrics_py         = read("src/evaluation/metrics.py")
evaluator_src      = read("src/evaluation/evaluator.py")
prompts_py         = read("src/steering/prompts.py")
dual_lens_py       = read("src/steering/dual_lens.py")
apo_py             = read("src/steering/apo.py")
probing_py         = read("src/probing/confidence_probing.py")
plots_py           = read("src/visualization/plots.py")
bootstrap_py       = read("scripts/bootstrap_data_layout.py")


# ── Cell builders ─────────────────────────────────────────────────────────────
def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
# Dynamic Perceptual Steering for African Cultural Competency in Vision-Language Models

**CMU Africa — 11-785 Introduction to Deep Learning | Team 21/23**
Boniface Godwin · Jean De Dieu Iradukunda · Oyindamola Olatunji · Peace Bakare

---

## Research Question

> *Can a Vision-Language Model be talked into **seeing** African cultural artifacts
> differently at runtime — without any retraining?*

Standard VLMs exhibit **perceptual erasure**: they identify the generic functional shape
of an African cultural object (e.g. "bowl", "bag", "building") but suppress the culturally
specific texture information (e.g. "kente cloth", "mudcloth", "Sudano-Sahelian
architecture"). This notebook implements the full experimental pipeline that:

1. **Measures** the erasure quantitatively using cue-conflict imagery (Geirhos 2019).
2. **Steers** the model at runtime with hand-crafted prompts and a Sequential Dual-Lens pipeline.
3. **Automates** prompt discovery with an APO loop (Gavrikov 2025 / Yang 2024).
4. **Probes** the mechanism — is cultural knowledge absent or merely suppressed?
5. **Tests** adversarial cultural stubbornness on iconic Western shapes.
6. **Synthesises** all findings into a unified hypothesis-testing report.

### Key References
- Geirhos et al. (2019). *ImageNet-trained CNNs are biased towards textures.*
- Gavrikov et al. (2025). *Can LLMs be steered towards texture bias?*
- Yang et al. (2024). *Large Language Models as Optimizers.*\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## Table of Contents

| # | Section | Description |
|---|---------|-------------|
| 0 | [Environment Setup](#part-0) | Install dependencies & global imports |
| 1 | [Configuration](#part-1) | Master config dict + directory initialisation |
| 2 | [Data Infrastructure](#part-2) | Five-source African cultural dataset loaders |
| 3 | [Cue-Conflict Synthesis](#part-3) | AdaIN style transfer (Geirhos 2019 methodology) |
| 4 | [Model Wrappers](#part-4) | InternVL-3 8B VLM + CLIP ViT-L/14 baseline |
| 5 | [Evaluation Pipeline](#part-5) | Shape-bias metrics + evaluation orchestrator |
| 6 | [Prompt Steering](#part-6) | Prompt library · Dual-Lens · APO loop |
| 7 | [Mechanistic Probing](#part-7) | Token-level confidence extraction |
| 8 | [Visualization](#part-8) | Publication-quality figure generation |
| 9 | [Data Bootstrap Script](#part-9) | Folder structure + annotation templates |
| 10 | [Phase 2 — Baseline](#phase-2) | Neutral evaluation of InternVL-3 & CLIP |
| 11 | [Phase 3A — Manual Steering](#phase-3a) | All hand-crafted prompt conditions |
| 12 | [Phase 3B — APO](#phase-3b) | Automated Prompt Optimization loop |
| 13 | [Phase 3C — Probing](#phase-3c) | Mechanistic confidence probing |
| 14 | [Phase 4 — Adversarial](#phase-4) | Cultural stubbornness on iconic Western shapes |
| 15 | [Phase 5 — Final Analysis](#phase-5) | Hypothesis testing & unified report |
| 16 | [Execution Guide](#execution-guide) | End-to-end usage instructions |\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 0: SETUP
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='part-0'></a>
## Part 0 — Environment Setup

Install all required packages. On PSC Bridges-2, activate the conda environment
created by `scripts/setup_environment.sh` instead of running pip directly.\
"""))

cells.append(code("""\
# Install all required packages.
# Run once; comment out after first successful installation.
# On PSC: skip this cell and use: conda activate dps_env

%pip install torch>=2.1.0 torchvision>=0.16.0 transformers>=4.40.0 accelerate>=0.27.0 \\
            einops>=0.7.0 datasets>=2.18.0 huggingface_hub>=0.21.0 tokenizers>=0.15.0 \\
            sentencepiece>=0.1.99 timm>=0.9.12 Pillow>=10.0.0 opencv-python>=4.8.0 \\
            scikit-image>=0.22.0 imageio>=2.33.0 scipy>=1.11.0 numpy>=1.24.0 \\
            pandas>=2.0.0 scikit-learn>=1.3.0 matplotlib>=3.7.0 seaborn>=0.12.0 \\
            ftfy>=6.1.1 regex>=2023.0.0 tqdm>=4.66.0 wandb>=0.16.0 pyyaml>=6.0.0 \\
            omegaconf>=2.3.0 jsonlines>=3.1.0 requests>=2.31.0 statsmodels>=0.14.0 \\
            bitsandbytes>=0.42.0 --quiet

# CLIP (OpenAI — not on PyPI)
%pip install git+https://github.com/openai/CLIP.git --quiet\
"""))

cells.append(code("""\
# ── Standard library ─────────────────────────────────────────────────────────
import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict

# ── Numerical / data science ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats

# ── Deep learning ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

# ── HuggingFace ───────────────────────────────────────────────────────────────
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    AutoProcessor, BitsAndBytesConfig,
)

# ── Visualization ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── CLIP (optional) ───────────────────────────────────────────────────────────
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    warnings.warn("CLIP not installed — CLIPBaseline will be unavailable.")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("dps")

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()} "
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")
print(f"CLIP     : {CLIP_AVAILABLE}")\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='part-1'></a>
## Part 1 — Configuration

The configuration mirrors `configs/config.yaml`.
Edit `CONFIG["paths"]` to match your local or PSC environment before running.\
"""))

cells.append(code(r"""# ── Master Configuration ─────────────────────────────────────────────────────
CONFIG = {
    "project": {
        "name": "dynamic_perceptual_steering",
        "team": "Team 21/23 - CMU Africa",
        "version": "1.0.0",
    },
    "paths": {
        "data_raw":          "data/raw",
        "data_processed":    "data/processed",
        "data_cue_conflict": "data/cue_conflict",
        "results":           "results",
        "figures":           "results/figures",
        "apo_prompts":       "results/apo_prompts",
        "model_cache":       os.environ.get("HF_HOME", None),
    },
    # ── Primary VLM ──────────────────────────────────────────────────────────
    "model": {
        "name":           "OpenGVLab/InternVL3-8B",
        "load_in_4bit":   True,    # required for V100-32 GB on PSC
        "torch_dtype":    "bfloat16",
        "max_new_tokens": 512,
        "temperature":    0.0,     # greedy / deterministic decoding
        "device_map":     "auto",
    },
    # ── CLIP baseline ─────────────────────────────────────────────────────────
    "clip": {
        "model_name": "ViT-L/14",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    # ── APO optimizer ─────────────────────────────────────────────────────────
    "apo": {
        "optimizer_model":          "mistralai/Mistral-7B-Instruct-v0.3",
        "load_in_4bit":             False,
        "torch_dtype":              "float16",
        "candidates_per_iteration": 5,
        "max_iterations":           20,
        "convergence_threshold":    0.005,
        "min_functional_accuracy":  0.75,
        "optimization_target":      "texture_bias",
    },
    # ── Dataset ───────────────────────────────────────────────────────────────
    "dataset": {
        "target_size": 10,
        "categories": [
            "textiles", "architecture", "everyday_objects",
            "food_and_drink", "ritual_items", "musical_instruments",
        ],
        "regions": [
            "West Africa", "East Africa", "Southern Africa",
            "Central Africa", "North Africa",
        ],
        "image_size": 224,
        "cue_conflict_per_category": 80,
    },
    # ── Evaluation ────────────────────────────────────────────────────────────
    "evaluation": {
        "batch_size": 1,
        "run_stats":  True,
        "alpha":      0.05,
    },
    # ── Prompts (overrides defaults in PromptLibrary) ─────────────────────────
    "prompts": {
        "neutral":   "What is in this image?",
        "structural": (
            "Identify the primary object in this image based on its shape and structure. "
            "What is the functional category of this object? "
            "Answer with just the object type."
        ),
        "cultural": (
            "Analyze the specific cultural textures, materials, patterns, and regional "
            "artistic style visible in this image. What African culture, ethnic group, "
            "or geographic region does this object most likely originate from? "
            "Describe the specific cultural markers you observe."
        ),
        "sequential_stage1": (
            "First, identify the basic functional type of the object in this image "
            "based on its shape alone. What type of object is this?"
        ),
        "sequential_stage2": (
            "Now, focusing entirely on the surface texture, patterns, weaving style, "
            "materials, colors, and decorative elements — not the shape — what "
            "specific African cultural tradition, ethnic group, or region produced this?"
        ),
    },
    # ── Logging ───────────────────────────────────────────────────────────────
    "logging": {
        "use_wandb":      False,
        "wandb_project":  "dynamic_perceptual_steering",
        "level":          "INFO",
        "save_per_image": True,
    },
}

# ── Create all output directories ─────────────────────────────────────────────
for _key in ["data_raw", "data_processed", "data_cue_conflict",
             "results", "figures", "apo_prompts"]:
    Path(CONFIG["paths"][_key]).mkdir(parents=True, exist_ok=True)

print("Configuration loaded. Directories initialised.")
print(f"Results root: {Path(CONFIG['paths']['results']).resolve()}")\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: DATA
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='part-2'></a>
## Part 2 — Data Infrastructure

Five African cultural image sources are unified into a single `AfricanCulturalDataset`.

| Source | Acquisition | Notes |
|--------|-------------|-------|
| **SURA Benchmark** | Local `data/raw/SURA/` | Primary source; needs `annotations.json` |
| **Africa-500** | Local `data/raw/Africa500/` | Team-curated; needs `annotations.json` |
| **Afri-MCQA** | HuggingFace `Atnafu/Afri-MCQA` | Auto-downloaded; max 400 samples |
| **CulturalVQA** | HuggingFace `vcr-org/CulturalVQA` | African countries only; max 200 |
| **Afri-Aya** | HuggingFace `CohereLabsCommunity/afri-aya` | Community-curated; max 200 |

Every record carries:
- `shape_label` — functional type ("bowl", "bag", "building")
- `texture_label` — cultural origin ("kente", "mudcloth", "sudano-sahelian")
- `category`, `region`, `source`, `is_famous`

These two labels enable the **cue-conflict evaluation**: the model must choose
whether to respond based on shape or cultural texture.\
"""))

# Strip the module-level docstring from dataset_loader to avoid redundancy,
# but keep the full code.
cells.append(code(dataset_loader))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: CUE-CONFLICT SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='part-3'></a>
## Part 3 — Cue-Conflict Synthesis

Following **Geirhos et al. (2019)**, we synthesise images where:
- The **shape** (content) comes from a neutral Western object.
- The **texture** (style) comes from an African cultural artifact.

**AdaIN** (Adaptive Instance Normalization, Huang & Belongie 2017):
1. Encode both images through VGG-19 up to `relu4_1`.
2. Align content feature statistics (μ, σ) to match style statistics.
3. Blend back to pixel space: `α=0.8` preserves shape while clearly applying the texture.

The simplified pixel-space implementation avoids training a decoder while
still producing perceptually convincing texture-shape conflicts.\
"""))

cells.append(code(cue_conflict_synth))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: MODELS
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='part-4'></a>
## Part 4 — Model Wrappers

### 4.1  InternVL-3 (8B) — Primary VLM

`InternVLWrapper` loads **OpenGVLab/InternVL3-8B** via HuggingFace.
Key design choices:
- **4-bit NF4 quantisation** (BitsAndBytes) to fit within V100-32 GB VRAM.
- **Temperature 0.0** for deterministic, fully reproducible outputs.
- **`get_token_logits()`** exposes next-token probabilities for Phase 3C probing.\
"""))

cells.append(code(internvl_wrapper))

cells.append(md("""\
### 4.2  CLIP ViT-L/14 — Vision-Only Baseline

`CLIPBaseline` answers the attribution question:
*Does perceptual erasure originate in the visual encoder (CLIP) or in the LLM?*

- If CLIP also misses cultural textures → problem is in visual encoding.
- If CLIP sees them but InternVL-3 suppresses them → problem is in language-vision fusion.

This is the diagnostic proposed in Gavrikov et al. (2025).\
"""))

cells.append(code(clip_baseline))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='part-5'></a>
## Part 5 — Evaluation Pipeline

### 5.1  Metrics

Core metric definitions following **Geirhos et al. (2019)** Equations 1–3,
extended with per-category, per-region, and Famous/Everyday breakdowns.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Shape Accuracy | fraction of responses mentioning shape label | Functional recognition rate |
| Texture Accuracy | fraction mentioning texture/culture label | Cultural recognition rate |
| Cue Accuracy | max(shape, texture) per image | Model got *at least one* correct |
| **Shape Bias β** | Shape Acc / Cue Acc | β→1 = pure shape; β→0 = pure texture/culture |
| Cultural Recovery Rate | texture_steered − texture_neutral | Improvement from steering |

**Synonym expansion** ensures that e.g. "bogolanfini" or "Bamana" counts
as a correct recognition of the "mudcloth" label.\
"""))

cells.append(code(metrics_py))

cells.append(md("""\
### 5.2  Evaluator

`Evaluator` orchestrates a full dataset pass for a single prompt condition.

Design features:
- Auto-saves every 50 images (crash protection on PSC).
- Logs per-condition metrics and Insight 1 (Famous vs Everyday) inline.
- `run_all_conditions()` runs multiple prompts sequentially and produces a
  side-by-side comparison CSV.\
"""))

# Inline evaluator with resolved imports (no relative paths needed in notebook)
cells.append(code("""\
# ── evaluator.py — inline (relative imports resolved by global definitions above) ─

class Evaluator:
    \"\"\"
    Orchestrates evaluation over the full dataset for a single prompt condition.

    Saves per-image results to JSON (auto-save every 50 images for crash recovery)
    and aggregate metrics to a separate JSON file.
    \"\"\"

    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.results_dir = Path(config["paths"]["results"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.save_per_image = config["logging"].get("save_per_image", True)

    def run(self, dataset, prompt: str, prompt_type: str,
            output_prefix: str = "") -> List[Dict]:
        \"\"\"Run evaluation over the full dataset with a single prompt.\"\"\"
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"{output_prefix}{prompt_type}_{timestamp}.json"

        logger.info(f"Evaluating [{prompt_type}] on {len(dataset)} images ...")

        for idx, record in enumerate(tqdm(dataset, desc=f"[{prompt_type}]")):
            try:
                response = self.model.generate(
                    image=record.load_image(), prompt=prompt
                )
                result = {
                    "idx": idx,
                    "image_path": record.image_path,
                    "source": record.source,
                    "shape_label": record.shape_label,
                    "texture_label": record.texture_label,
                    "category": record.category,
                    "region": record.region,
                    "is_famous": record.is_famous,
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                    "response": response,
                }
                results.append(result)
                if self.save_per_image and (idx + 1) % 50 == 0:
                    self._save_results(results, output_file)
            except Exception as e:
                logger.error(f"Eval failed [{record.image_path}]: {e}")
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
                    "error": str(e),
                })

        self._save_results(results, output_file)
        metrics = compute_all_metrics(results)
        self._log_metrics(metrics, prompt_type)
        metrics_file = (self.results_dir /
                        f"{output_prefix}metrics_{prompt_type}_{timestamp}.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        return results

    def run_all_conditions(self, dataset,
                           prompts: Dict[str, str]) -> Dict[str, List[Dict]]:
        \"\"\"Run evaluation for all prompt conditions and compute pairwise comparisons.\"\"\"
        all_results = {}
        for prompt_type, prompt in prompts.items():
            all_results[prompt_type] = self.run(dataset, prompt, prompt_type)

        if "neutral" in all_results:
            for condition, results in all_results.items():
                if condition == "neutral":
                    continue
                cmp = compare_conditions(
                    all_results["neutral"], results,
                    condition_a="neutral", condition_b=condition,
                )
                logger.info(f"neutral vs {condition}: {cmp['interpretation']}  "
                            f"Δshape_bias={cmp['difference']:.4f}")

        self._save_comparison_csv(all_results)
        return all_results

    def _save_results(self, results: List[Dict], output_file: Path):
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    def _log_metrics(self, metrics: Dict, prompt_type: str):
        logger.info(
            f"[{prompt_type}] "
            f"shape={metrics.get('shape_accuracy', 0):.4f}  "
            f"texture={metrics.get('texture_accuracy', 0):.4f}  "
            f"cue={metrics.get('cue_accuracy', 0):.4f}  "
            f"bias={metrics.get('shape_bias', 0):.4f}"
        )
        if "famous_items" in metrics and "everyday_items" in metrics:
            f_, e_ = metrics["famous_items"], metrics["everyday_items"]
            logger.info(
                f"  Insight 1 — Famous texture: {f_.get('texture_accuracy', 0):.4f}  "
                f"Everyday texture: {e_.get('texture_accuracy', 0):.4f}"
            )

    def _save_comparison_csv(self, all_results: Dict[str, List[Dict]]):
        rows = []
        for condition, results in all_results.items():
            m = compute_all_metrics(results)
            rows.append({
                "condition": condition,
                "n_images": m.get("n_images", 0),
                "shape_accuracy": m.get("shape_accuracy", 0),
                "texture_accuracy": m.get("texture_accuracy", 0),
                "cue_accuracy": m.get("cue_accuracy", 0),
                "shape_bias": m.get("shape_bias", 0),
            })
        df = pd.DataFrame(rows)
        csv_path = self.results_dir / "comparison_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Comparison CSV saved to {csv_path}")
        print("\\n" + df.to_string(index=False))\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: STEERING
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='part-6'></a>
## Part 6 — Prompt Steering

Three steering mechanisms, in increasing sophistication:

| Mechanism | Description |
|-----------|-------------|
| **Manual prompts** | Hand-crafted variants targeting shape, culture, geometry, expert persona |
| **Sequential Dual-Lens** | Stage 1 names the shape; Stage 2 identifies cultural origin given that context |
| **APO loop** | Mistral-7B-Instruct iteratively proposes and evaluates prompts |

### 6.1  Prompt Library\
"""))

cells.append(code(prompts_py))

cells.append(md("""\
### 6.2  Sequential Dual-Lens Pipeline

The two-stage pipeline separates the competing visual cues:

```
Input image
    │
    ▼  Stage 1 — STRUCTURAL LENS
    │  "Based on shape alone, what is this?"
    │  → shape_response  e.g. "bowl"
    │
    ▼  Stage 2 — CULTURAL LENS
    │  "You identified this as {shape}. Now focus entirely on texture …"
    │  → cultural_response  e.g. "this is kente cloth from Ghana"
    │
    ▼  Combined response  (pipe-separated, used for metric evaluation)
```

By feeding Stage 1's output into Stage 2, the model cannot simply repeat the
shape answer — it must reason specifically about cultural texture.\
"""))

cells.append(code(strip_relative_imports(dual_lens_py)))

cells.append(md("""\
### 6.3  Automated Prompt Optimization (APO)

**Objective**: maximise `texture_accuracy` subject to `shape_accuracy ≥ 0.75`
**Optimizer**: Mistral-7B-Instruct generates candidate prompts from structured feedback
**Convergence**: improvement < `0.005` for ≥ 3 iterations, or `max_iterations = 20`

The optimizer receives a structured history of every prompt tried, along with its
`texture_accuracy`, `shape_accuracy`, and `shape_bias` scores. It outputs new
candidate prompts (each starting with `PROMPT:`) for evaluation.\
"""))

cells.append(code(strip_relative_imports(apo_py)))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: PROBING
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='part-7'></a>
## Part 7 — Mechanistic Confidence Probing

**Core hypothesis** (extending Gavrikov et al. 2025 to the cultural domain):
InternVL-3 *encodes* African cultural texture information in its visual features
but the LLM component *suppresses* it in favour of the shape label.

**Expected evidence:**

| Condition | Shape token confidence | Texture token confidence |
|-----------|----------------------|--------------------------|
| Neutral prompt | ≈ 1.0 | ≈ 0.0 (suppressed) |
| Cultural prompt | slightly lower | significantly **higher** |

If texture confidence rises substantially after cultural steering, the knowledge
was always encoded — it just needed activation. This rules out the alternative
explanation that the model simply lacks African cultural knowledge.\
"""))

cells.append(code(strip_relative_imports(probing_py)))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 8: VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='part-8'></a>
## Part 8 — Visualization

Publication-quality figures following the style of Gavrikov et al. (2025)
and Geirhos et al. (2019). All outputs are saved as PDF to `results/figures/`.

| Figure | File | Description |
|--------|------|-------------|
| Shape Bias Scatter | `shape_bias_scatter.pdf` | X=texture acc, Y=shape acc per condition |
| Confidence Distributions | `confidence_distributions.pdf` | Token confidence histograms |
| Accuracy Tradeoff | `accuracy_tradeoff.pdf` | Functional vs cultural accuracy frontier |
| Famous vs Everyday | `famous_vs_everyday.pdf` | Insight 1 bar chart |
| Category Heatmap | `category_heatmap.pdf` | Texture acc per category per condition |
| APO Progress | `apo_progress.pdf` | Best texture accuracy over APO iterations |\
"""))

cells.append(code(plots_py))

# ═══════════════════════════════════════════════════════════════════════════════
# PART 9: BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='part-9'></a>
## Part 9 — Data Bootstrap Script

Creates the local folder structure and drops annotation template files.
Run this before placing real dataset images in `data/raw/`.
Existing files are not overwritten.\
"""))

cells.append(code(bootstrap_py.replace(
    "ROOT = Path(__file__).resolve().parent.parent",
    "ROOT = Path.cwd()  # notebook runs from project root",
)))

cells.append(code("""\
# Run the bootstrap
# (creates all directories and annotation template files)
main()  # main() defined in the cell above\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='phase-2'></a>
## Part 10 — Phase 2: Neutral Baseline Evaluation

Runs InternVL-3 (8B) and CLIP ViT-L/14 on the evaluation set with the
**neutral prompt** `"What is in this image?"`.

This establishes the **perceptual erasure baseline**: how much cultural texture
information does the model suppress with no steering?

**Key outputs:**
- Shape Accuracy, Texture Accuracy, Cue Accuracy, Shape Bias
- Insight 1: Famous vs Everyday cultural recognition gap
- CLIP vs InternVL-3 comparison (encoder vs LLM attribution)
- `results/phase2_summary.json`, `results/figures/shape_bias_scatter.pdf`

**Expected runtime**: ~4 h on PSC V100-32 GB (4-bit quantised InternVL-3).\
"""))

cells.append(code("""\
def run_phase2(config=CONFIG):
    \"\"\"Phase 2 — Neutral Baseline Evaluation.\"\"\"

    logger.info("=" * 60)
    logger.info("PHASE 2 — BASELINE EVALUATION")
    logger.info("=" * 60)

    # ── Load Dataset ─────────────────────────────────────────────────────────
    logger.info("Step 1: Loading dataset ...")
    dataset = load_all_datasets(config)
    if len(dataset) == 0:
        logger.error("Dataset is empty. Download datasets first (see README).")
        return None, None
    dataset.print_summary()

    # ── Load InternVL-3 ──────────────────────────────────────────────────────
    logger.info("Step 2: Loading InternVL-3 (8B) ...")
    model = InternVLWrapper(config)
    logger.info(f"Model info: {model.get_model_info()}")

    # ── Neutral Baseline ─────────────────────────────────────────────────────
    prompts_lib    = PromptLibrary(config)
    evaluator      = Evaluator(model, config)
    neutral_results = evaluator.run(
        dataset=dataset,
        prompt=prompts_lib.get_text("neutral"),
        prompt_type="neutral",
        output_prefix="phase2_",
    )
    neutral_metrics = compute_all_metrics(neutral_results)

    print("\\n" + "=" * 50)
    print("NEUTRAL BASELINE RESULTS")
    print("=" * 50)
    print(f"  Shape Accuracy   : {neutral_metrics['shape_accuracy']:.4f}")
    print(f"  Texture Accuracy : {neutral_metrics['texture_accuracy']:.4f}")
    print(f"  Cue Accuracy     : {neutral_metrics['cue_accuracy']:.4f}")
    print(f"  Shape Bias       : {neutral_metrics['shape_bias']:.4f}")

    # ── Insight 1: Famous vs Everyday ────────────────────────────────────────
    if "famous_items" in neutral_metrics and "everyday_items" in neutral_metrics:
        fam = neutral_metrics["famous_items"]
        eve = neutral_metrics["everyday_items"]
        print("\\nINSIGHT 1 — Famous vs Everyday:")
        print(f"  Famous   → texture_acc : {fam['texture_accuracy']:.4f}")
        print(f"  Everyday → texture_acc : {eve['texture_accuracy']:.4f}")
        print(f"  Gap                    : {fam['texture_accuracy'] - eve['texture_accuracy']:+.4f}")

    # ── CLIP Vision-Only Baseline ────────────────────────────────────────────
    clip_metrics = {}
    if CLIP_AVAILABLE:
        logger.info("Step 3: Running CLIP Vision-Only Baseline ...")
        try:
            clip_model   = CLIPBaseline(config)
            clip_metrics = clip_model.compute_shape_bias(dataset)
            print("\\nCLIP BASELINE:")
            print(f"  Shape Bias   : {clip_metrics['shape_bias']:.4f}")
            print(f"  Cue Accuracy : {clip_metrics['cue_accuracy']:.4f}")
            verdict = ("InternVL-3 more shape-biased"
                       if neutral_metrics["shape_bias"] > clip_metrics["shape_bias"]
                       else "CLIP more shape-biased")
            print(f"  Comparison   : {verdict}")
        except Exception as e:
            logger.warning(f"CLIP baseline failed: {e}")
    else:
        logger.warning("CLIP not installed — skipping CLIP baseline.")

    # ── Save Summary ─────────────────────────────────────────────────────────
    summary = {
        "phase": "Phase 2 — Baseline",
        "dataset_size": len(dataset),
        "internvl_neutral": neutral_metrics,
        "clip_baseline": clip_metrics,
    }
    summary_path = Path(config["paths"]["results"]) / "phase2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Phase 2 summary saved to {summary_path}")

    # ── Figure ────────────────────────────────────────────────────────────────
    viz = ResultsVisualizer(config)
    cdata = {"neutral": neutral_metrics}
    if clip_metrics:
        cdata["clip_baseline"] = clip_metrics
    viz.plot_shape_bias_scatter(cdata, title="Phase 2: Baseline Shape Bias")

    logger.info("Phase 2 complete.")
    return neutral_results, neutral_metrics


# Uncomment to execute:
# neutral_results, neutral_metrics = run_phase2(CONFIG)\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3A
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='phase-3a'></a>
## Part 11 — Phase 3A: Manual Dynamic Steering

Evaluates all hand-crafted prompt conditions plus the Sequential Dual-Lens pipeline.

**Conditions evaluated:**
- `neutral` — no steering (comparison baseline)
- `structural` — explicit shape/function steering
- `cultural` — explicit texture/culture steering
- `cultural_geometric` — geometric motif focus
- `cultural_expert` — expert persona

**Perceptual Over-Steering** check: flag conditions where shape accuracy drops
more than 10 percentage points below neutral (cultural gain at too high a cost).

**Expected runtime**: ~6 h on PSC V100-32 GB.\
"""))

cells.append(code("""\
def run_phase3a(config=CONFIG):
    \"\"\"Phase 3A — Manual Dynamic Steering.\"\"\"

    logger.info("=" * 60)
    logger.info("PHASE 3A — MANUAL DYNAMIC STEERING")
    logger.info("=" * 60)

    dataset = load_all_datasets(config)
    if len(dataset) == 0:
        logger.error("Empty dataset."); return None, None

    model       = InternVLWrapper(config)
    prompts_lib = PromptLibrary(config)
    evaluator   = Evaluator(model, config)

    # ── All single-pass conditions ────────────────────────────────────────────
    conditions = {
        "neutral":            prompts_lib.get_text("neutral"),
        "structural":         prompts_lib.get_text("structural"),
        "cultural":           prompts_lib.get_text("cultural"),
        "cultural_geometric": prompts_lib.get_text("cultural_geometric"),
        "cultural_expert":    prompts_lib.get_text("cultural_expert"),
    }
    all_results = evaluator.run_all_conditions(dataset=dataset, prompts=conditions)

    # ── Sequential Dual-Lens (two-stage pipeline) ─────────────────────────────
    logger.info("\\nRunning Sequential Dual-Lens ...")
    dual_lens = DualLensSteering(model, prompts_lib, config)
    all_results["sequential"] = dual_lens.run_batch(dataset)

    # ── Metrics for every condition ───────────────────────────────────────────
    metrics_by_condition = {c: compute_all_metrics(r) for c, r in all_results.items()}

    # ── Statistical comparisons ───────────────────────────────────────────────
    print("\\n" + "=" * 60)
    print("STATISTICAL COMPARISONS (vs neutral baseline)")
    print("=" * 60)
    for cond in ["cultural", "sequential", "cultural_expert"]:
        if cond in all_results:
            cmp = compare_conditions(
                all_results["neutral"], all_results[cond],
                condition_a="neutral", condition_b=cond,
            )
            print(f"\\nneutral vs {cond}:")
            print(f"  {cmp['interpretation']}")
            print(f"  shape_bias change: {cmp['difference']:+.4f}")

    # ── Perceptual Over-Steering Analysis ─────────────────────────────────────
    print("\\n" + "=" * 60)
    print("PERCEPTUAL OVER-STEERING ANALYSIS")
    print("=" * 60)
    neutral_shape = metrics_by_condition.get("neutral", {}).get("shape_accuracy", 0)
    for cond, m in metrics_by_condition.items():
        if cond == "neutral":
            continue
        shape_drop   = neutral_shape - m.get("shape_accuracy", 0)
        texture_gain = (m.get("texture_accuracy", 0) -
                        metrics_by_condition["neutral"].get("texture_accuracy", 0))
        flag = "⚠ OVER-STEERING RISK" if shape_drop > 0.10 else "✓ OK"
        print(f"  {cond:<25}  shape_drop={shape_drop:+.3f}  "
              f"texture_gain={texture_gain:+.3f}  {flag}")

    # ── Save Summary ─────────────────────────────────────────────────────────
    summary = {
        "phase": "Phase 3A — Manual Steering",
        "conditions_evaluated": list(metrics_by_condition.keys()),
        "metrics": {
            c: {k: v for k, v in m.items()
                if k in ["shape_accuracy", "texture_accuracy",
                          "cue_accuracy", "shape_bias", "n_images",
                          "famous_items", "everyday_items"]}
            for c, m in metrics_by_condition.items()
        },
    }
    summary_path = Path(config["paths"]["results"]) / "phase3a_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Phase 3A summary saved to {summary_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    viz = ResultsVisualizer(config)
    viz.plot_shape_bias_scatter(metrics_by_condition)
    viz.plot_accuracy_tradeoff_curve(metrics_by_condition)
    viz.plot_famous_vs_everyday(metrics_by_condition)
    viz.plot_category_heatmap(metrics_by_condition)

    logger.info("Phase 3A complete.")
    return all_results, metrics_by_condition


# Uncomment to execute:
# all_results_3a, metrics_3a = run_phase3a(CONFIG)\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3B
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='phase-3b'></a>
## Part 12 — Phase 3B: Automated Prompt Optimization (APO)

Uses **Mistral-7B-Instruct** as an automated prompt engineer in a closed loop.
The optimizer receives a structured history of every prompt and its scores,
then generates 5 new candidate prompts per iteration.

**Objective**: maximise `texture_accuracy` subject to `shape_accuracy ≥ 0.75`
**Convergence**: improvement < `0.005` for 3+ consecutive iterations
**Output**: best discovered prompt + top-5 candidates in `phase3b_summary.json`

**Expected runtime**: ~12 h on PSC V100-32 GB (both InternVL-3 and Mistral loaded).\
"""))

cells.append(code("""\
def run_phase3b(config=CONFIG):
    \"\"\"Phase 3B — Automated Prompt Optimization (APO).\"\"\"

    logger.info("=" * 60)
    logger.info("PHASE 3B — AUTOMATED PROMPT OPTIMIZATION (APO)")
    logger.info("=" * 60)

    dataset = load_all_datasets(config)
    if len(dataset) == 0:
        logger.error("Empty dataset."); return None, None, None

    vlm = InternVLWrapper(config)

    logger.info("Initialising APO optimizer (Mistral-7B-Instruct) ...")
    apo = AutomatedPromptOptimizer(vlm, config)

    best_prompt, history = apo.optimize(dataset)

    print("\\n" + "=" * 60)
    print("APO COMPLETE — BEST PROMPT DISCOVERED:")
    print("=" * 60)
    print(best_prompt)

    # ── Evaluate best APO prompt on full dataset ──────────────────────────────
    logger.info("\\nEvaluating best APO prompt on full dataset ...")
    evaluator   = Evaluator(vlm, config)
    apo_results = evaluator.run(
        dataset=dataset,
        prompt=best_prompt,
        prompt_type="apo_best",
        output_prefix="phase3b_",
    )
    apo_metrics = compute_all_metrics(apo_results)

    print("\\nAPO BEST PROMPT — FINAL METRICS:")
    print(f"  texture_accuracy : {apo_metrics['texture_accuracy']:.4f}")
    print(f"  shape_accuracy   : {apo_metrics['shape_accuracy']:.4f}")
    print(f"  shape_bias       : {apo_metrics['shape_bias']:.4f}")

    # ── Save Summary ─────────────────────────────────────────────────────────
    top5    = apo.get_top_prompts(5)
    summary = {
        "phase": "Phase 3B — APO",
        "best_prompt": best_prompt,
        "best_metrics": {k: v for k, v in apo_metrics.items()
                         if k in ["shape_accuracy", "texture_accuracy",
                                   "cue_accuracy", "shape_bias"]},
        "top_5_prompts": [c.to_dict() for c in top5],
    }
    summary_path = Path(config["paths"]["results"]) / "phase3b_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Phase 3B summary saved to {summary_path}")

    # ── Plot APO progress ─────────────────────────────────────────────────────
    viz = ResultsVisualizer(config)
    viz.plot_apo_progress(history)

    print("\\nTOP 5 APO-DISCOVERED PROMPTS:")
    for i, c in enumerate(top5, 1):
        print(f"  #{i} (texture_acc={c.texture_accuracy:.4f}, "
              f"shape_acc={c.shape_accuracy:.4f}):")
        print(f"  {c.text}\\n")

    logger.info("Phase 3B complete.")
    return best_prompt, history, apo_metrics


# Uncomment to execute:
# best_prompt, apo_history, apo_metrics = run_phase3b(CONFIG)\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3C
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='phase-3c'></a>
## Part 13 — Phase 3C: Mechanistic Confidence Probing

Extracts token-level logit probabilities to test the suppression hypothesis.

**Expected pattern** (mirroring Figure 3, Gavrikov et al. 2025):
- Under neutral prompt: shape confidence ≈ 1.0, texture confidence ≈ 0.0
- After cultural steering: texture confidence rises significantly

If `phase3b_summary.json` is present, the best APO prompt is automatically
added as a fourth probing condition for comparison.

**Expected runtime**: ~8 h on PSC V100-32 GB.\
"""))

cells.append(code("""\
def run_phase3c(config=CONFIG):
    \"\"\"Phase 3C — Mechanistic Confidence Probing.\"\"\"

    logger.info("=" * 60)
    logger.info("PHASE 3C — MECHANISTIC CONFIDENCE PROBING")
    logger.info("=" * 60)

    dataset = load_all_datasets(config)
    if len(dataset) == 0:
        logger.error("Empty dataset."); return None

    model       = InternVLWrapper(config)
    prompts_lib = PromptLibrary(config)

    probe_conditions = {
        "neutral":    prompts_lib.get_text("neutral"),
        "cultural":   prompts_lib.get_text("cultural"),
        "structural": prompts_lib.get_text("structural"),
    }

    # Optionally add APO best prompt
    apo_path = Path(config["paths"]["results"]) / "phase3b_summary.json"
    if apo_path.exists():
        with open(apo_path) as f:
            apo_data = json.load(f)
        if apo_data.get("best_prompt"):
            probe_conditions["apo_best"] = apo_data["best_prompt"]
            logger.info("Added APO best prompt to probing conditions.")

    # ── Run probing ───────────────────────────────────────────────────────────
    prober = ConfidenceProber(model, config)
    logger.info(f"Probing {len(probe_conditions)} conditions × {len(dataset)} images ...")
    probing_results = prober.run_full_probing(
        dataset=dataset,
        prompt_conditions=probe_conditions,
    )

    # ── Print key findings ────────────────────────────────────────────────────
    print("\\n" + "=" * 60)
    print("MECHANISTIC PROBING — KEY FINDINGS")
    print("=" * 60)

    if "neutral" in probing_results and "cultural" in probing_results:
        n_tex = [r["texture_confidence"] for r in probing_results["neutral"]]
        c_tex = [r["texture_confidence"] for r in probing_results["cultural"]]
        n_shp = [r["shape_confidence"]   for r in probing_results["neutral"]]

        print(f"\\nUnder NEUTRAL prompt:")
        print(f"  Mean shape confidence   : {np.mean(n_shp):.4f}")
        print(f"  Mean texture confidence : {np.mean(n_tex):.4f}")
        print(f"  Shape dominance         : {np.mean(n_shp) - np.mean(n_tex):.4f}")

        print(f"\\nAfter CULTURAL steering:")
        print(f"  Mean texture confidence : {np.mean(c_tex):.4f}")
        print(f"  Rise from neutral       : {np.mean(c_tex) - np.mean(n_tex):+.4f}")

        sup_n = np.mean([r["texture_suppressed"] for r in probing_results["neutral"]])
        sup_c = np.mean([r["texture_suppressed"] for r in probing_results["cultural"]])
        print(f"\\nTexture token suppressed (conf < 0.01):")
        print(f"  Neutral  : {sup_n:.1%}")
        print(f"  Cultural : {sup_c:.1%}")
        print(f"  Recovered: {sup_n - sup_c:.1%} of images")

        if np.mean(c_tex) > np.mean(n_tex) * 1.5:
            print("\\n✓ HYPOTHESIS CONFIRMED: Model suppresses African cultural textures "
                  "under neutral prompts; cultural steering activates latent knowledge.")
        else:
            print("\\n⚠ PARTIAL: Improvement observed but suppression is more complex.")

    # ── Save Summary ─────────────────────────────────────────────────────────
    summary = {
        "phase": "Phase 3C — Mechanistic Probing",
        "conditions_probed": list(probe_conditions.keys()),
        "n_images": len(dataset),
    }
    for cond, results in probing_results.items():
        if results:
            summary[f"{cond}_stats"] = {
                "mean_shape_conf":     float(np.mean([r["shape_confidence"]    for r in results])),
                "mean_texture_conf":   float(np.mean([r["texture_confidence"]  for r in results])),
                "fraction_suppressed": float(np.mean([r["texture_suppressed"]  for r in results])),
            }
    summary_path = Path(config["paths"]["results"]) / "phase3c_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # ── Plot confidence distributions ─────────────────────────────────────────
    viz = ResultsVisualizer(config)
    viz.plot_confidence_distributions(probing_results)

    logger.info("Phase 3C complete.")
    return probing_results


# Uncomment to execute:
# probing_results = run_phase3c(CONFIG)\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='phase-4'></a>
## Part 14 — Phase 4: Adversarial Texture Steering

Tests **cultural stubbornness**: can steering force the model to prioritise
African textures even when applied to *highly iconic Western shapes*?

If the model still defaults to the Western shape identity even after explicit
cultural steering, that is evidence of deep-rooted cultural stubbornness —
the Western shape bias is so strong it overrides runtime prompting.

**Required images** (place in `data/raw/adversarial/`):

| `western_shapes/` | `african_textures/` |
|-------------------|---------------------|
| phone_booth.jpg | zulu_beadwork.jpg |
| eiffel_tower.jpg | kente_cloth.jpg |
| london_bus.jpg | mudcloth.jpg |
| american_diner.jpg | adire.jpg |
| yellow_taxi.jpg | ankara.jpg |
| mailbox.jpg | adinkra.jpg |\
"""))

cells.append(code("""\
from dataclasses import dataclass as _dc


@_dc
class AdversarialPair:
    \"\"\"An adversarial pair: African texture applied to an iconic Western shape.\"\"\"
    western_shape_path: str
    african_texture_path: str
    conflict_path: str
    western_shape_label: str
    african_texture_label: str
    iconicity_level: str   # 'high' | 'medium' | 'low'


ADVERSARIAL_CASES = [
    {"western_shape": "phone_booth.jpg",    "african_texture": "zulu_beadwork.jpg",
     "western_label": "phone booth",        "african_label": "zulu beadwork",   "iconicity": "high"},
    {"western_shape": "eiffel_tower.jpg",   "african_texture": "kente_cloth.jpg",
     "western_label": "eiffel tower",       "african_label": "kente cloth",     "iconicity": "high"},
    {"western_shape": "london_bus.jpg",     "african_texture": "mudcloth.jpg",
     "western_label": "london bus",         "african_label": "mudcloth",        "iconicity": "high"},
    {"western_shape": "american_diner.jpg", "african_texture": "adire.jpg",
     "western_label": "american diner",     "african_label": "adire cloth",     "iconicity": "medium"},
    {"western_shape": "yellow_taxi.jpg",    "african_texture": "ankara.jpg",
     "western_label": "yellow taxi",        "african_label": "ankara fabric",   "iconicity": "medium"},
    {"western_shape": "mailbox.jpg",        "african_texture": "adinkra.jpg",
     "western_label": "mailbox",            "african_label": "adinkra symbols", "iconicity": "low"},
]


def run_phase4(config=CONFIG):
    \"\"\"Phase 4 — Adversarial Texture Steering.\"\"\"

    logger.info("=" * 60)
    logger.info("PHASE 4 — ADVERSARIAL TEXTURE STEERING")
    logger.info("=" * 60)

    western_dir = Path(config["paths"]["data_raw"]) / "adversarial" / "western_shapes"
    african_dir = Path(config["paths"]["data_raw"]) / "adversarial" / "african_textures"

    if not western_dir.exists() or not african_dir.exists():
        logger.warning(
            f"Adversarial image directories missing:\\n"
            f"  {western_dir}\\n  {african_dir}\\n"
            "Create them and add images, then rerun."
        )
        western_dir.mkdir(parents=True, exist_ok=True)
        african_dir.mkdir(parents=True, exist_ok=True)
        return None

    synthesizer  = CueConflictSynthesizer(config)
    conflict_dir = Path(config["paths"]["data_cue_conflict"]) / "adversarial"
    conflict_dir.mkdir(parents=True, exist_ok=True)

    adversarial_pairs = []
    for case in ADVERSARIAL_CASES:
        w_path = western_dir / case["western_shape"]
        a_path = african_dir / case["african_texture"]
        if not w_path.exists() or not a_path.exists():
            logger.warning(f"Missing: {case['western_shape']} or {case['african_texture']}")
            continue

        pair_id = f"adv_{case['western_label'].replace(' ', '_')}"
        cp = synthesizer.synthesize(
            content_image=Image.open(w_path).convert("RGB"),
            style_image=Image.open(a_path).convert("RGB"),
            shape_label=case["western_label"],
            texture_label=case["african_label"],
            category="adversarial", region="mixed", pair_id=pair_id,
        )
        if cp:
            adversarial_pairs.append(AdversarialPair(
                western_shape_path=str(w_path),
                african_texture_path=str(a_path),
                conflict_path=cp.conflict_path,
                western_shape_label=case["western_label"],
                african_texture_label=case["african_label"],
                iconicity_level=case["iconicity"],
            ))

    if not adversarial_pairs:
        logger.error("No adversarial pairs created."); return None

    logger.info(f"Created {len(adversarial_pairs)} adversarial conflict images.")

    model       = InternVLWrapper(config)
    prompts_lib = PromptLibrary(config)
    conditions  = {
        "neutral":             prompts_lib.get_text("neutral"),
        "cultural":            prompts_lib.get_text("cultural"),
        "adversarial_texture": prompts_lib.get_text("adversarial_texture"),
    }

    results_by_condition = {}
    for cname, ptext in conditions.items():
        logger.info(f"\\nEvaluating [{cname}] ...")
        cond_results = []
        for pair in adversarial_pairs:
            try:
                img = Image.open(pair.conflict_path).convert("RGB")
                response = model.generate(img, ptext)
                shp_hit, tex_hit = parse_decision(
                    response, pair.western_shape_label, pair.african_texture_label
                )
                cond_results.append({
                    "conflict_path":    pair.conflict_path,
                    "shape_label":      pair.western_shape_label,
                    "texture_label":    pair.african_texture_label,
                    "iconicity_level":  pair.iconicity_level,
                    "prompt_type":      cname,
                    "response":         response,
                    "shape_hit":        shp_hit,
                    "texture_hit":      tex_hit,
                })
                logger.info(f"  {pair.western_shape_label} + {pair.african_texture_label} "
                            f"→ shape={shp_hit}, texture={tex_hit}")
            except Exception as e:
                logger.error(f"Failed: {e}")
        results_by_condition[cname] = cond_results

    # ── Stubbornness Analysis ─────────────────────────────────────────────────
    print("\\n" + "=" * 60)
    print("CULTURAL STUBBORNNESS ANALYSIS")
    print("=" * 60)
    for cname, results in results_by_condition.items():
        print(f"\\n[{cname}]")
        for level in ["high", "medium", "low"]:
            lvl = [r for r in results if r["iconicity_level"] == level]
            if not lvl: continue
            n_tex = sum(r["texture_hit"] for r in lvl)
            rate  = n_tex / len(lvl)
            print(f"  Iconicity={level:<6}: {n_tex}/{len(lvl)} ({rate:.0%}) "
                  "African texture recognised")

    # ── Save Summary ─────────────────────────────────────────────────────────
    summary = {
        "phase": "Phase 4 — Adversarial",
        "n_pairs": len(adversarial_pairs),
        "results_by_condition": {
            c: [{k: v for k, v in r.items() if k != "response"} for r in res]
            for c, res in results_by_condition.items()
        },
    }
    summary_path = Path(config["paths"]["results"]) / "phase4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info("Phase 4 complete.")
    return results_by_condition


# Uncomment to execute:
# adv_results = run_phase4(CONFIG)\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='phase-5'></a>
## Part 15 — Phase 5: Final Analysis & Hypothesis Testing

Loads all phase summaries and produces:
1. Unified results table (Paper Table 1)
2. Statistical hypothesis tests for H1, H2, H3
3. Insight 1 summary (Famous vs Everyday recognition gap)
4. All final publication-quality figures

**Run after phases 2, 3A, and 3B have completed.**

### Hypotheses

| Hypothesis | Claim |
|-----------|-------|
| **H1** | Perceptual erasure is real — neutral texture accuracy is very low |
| **H2** | Latent cultural knowledge exists — cultural prompt substantially improves texture accuracy |
| **H3** | APO discovers better prompts — apo_best > cultural prompt texture accuracy |\
"""))

cells.append(code("""\
def load_phase_results(results_dir: Path) -> dict:
    \"\"\"Load all phase summary JSON files into a unified results dict.\"\"\"
    all_results = {}

    p2 = results_dir / "phase2_summary.json"
    if p2.exists():
        with open(p2) as f: d = json.load(f)
        all_results["neutral"] = d.get("internvl_neutral", {})
        if d.get("clip_baseline"):
            all_results["clip_baseline"] = d["clip_baseline"]
        logger.info("Loaded Phase 2 results.")
    else:
        logger.warning("Phase 2 not found — run run_phase2() first.")

    p3a = results_dir / "phase3a_summary.json"
    if p3a.exists():
        with open(p3a) as f: d = json.load(f)
        for cond, m in d.get("metrics", {}).items():
            if cond not in all_results:
                all_results[cond] = m
        logger.info("Loaded Phase 3A results.")
    else:
        logger.warning("Phase 3A not found — run run_phase3a() first.")

    p3b = results_dir / "phase3b_summary.json"
    if p3b.exists():
        with open(p3b) as f: d = json.load(f)
        all_results["apo_best"] = d.get("best_metrics", {})
        logger.info("Loaded Phase 3B results.")

    return all_results


def print_results_table(all_results: dict):
    \"\"\"Print the unified results table (Paper Table 1).\"\"\"
    print("\\n" + "=" * 82)
    print("TABLE 1: DYNAMIC PERCEPTUAL STEERING — MAIN RESULTS")
    print("=" * 82)
    print(f"  {'Condition':<23}  {'Shape Acc':>10}  {'Texture Acc':>12}  "
          f"{'Cue Acc':>9}  {'Shape Bias':>10}")
    print("-" * 82)
    order = ["clip_baseline", "neutral", "structural", "cultural",
             "cultural_geometric", "cultural_expert", "sequential", "apo_best"]
    for cond in order:
        if cond not in all_results:
            continue
        m = all_results[cond]
        marker = " ★" if cond == "apo_best" else "  "
        print(f"  {cond:<23}  "
              f"{m.get('shape_accuracy', 0):>10.4f}  "
              f"{m.get('texture_accuracy', 0):>12.4f}  "
              f"{m.get('cue_accuracy', 0):>9.4f}  "
              f"{m.get('shape_bias', 0):>10.4f}{marker}")
    print("=" * 82)
    print("★ = APO-discovered best prompt")
    print("Shape Bias = Shape Accuracy / Cue Accuracy  (1.0 = pure shape; 0.0 = pure texture)")


def test_hypotheses(all_results: dict, alpha: float = 0.05):
    \"\"\"Test the three main project hypotheses.\"\"\"
    neutral  = all_results.get("neutral",  {})
    cultural = all_results.get("cultural", {})
    apo_best = all_results.get("apo_best", {})

    print("\\n" + "=" * 82)
    print("HYPOTHESIS TESTING")
    print("=" * 82)

    # H1 — Perceptual Erasure
    n_tex = neutral.get("texture_accuracy", 0)
    print(f"\\nH1: Perceptual erasure is real and measurable")
    print(f"    neutral texture_accuracy = {n_tex:.4f}")
    print("    VERDICT:", end=" ")
    if n_tex < 0.30:
        print("✓ CONFIRMED — low texture accuracy demonstrates perceptual erasure")
    elif n_tex < 0.50:
        print("⚠ PARTIAL — erasure present but category-dependent")
    else:
        print("✗ WEAK — texture accuracy is not low; "
              "check dataset balance or label parsing")

    # H2 — Latent Knowledge
    c_tex  = cultural.get("texture_accuracy", 0)
    delta2 = c_tex - n_tex
    print(f"\\nH2: Latent cultural knowledge exists and can be activated")
    print(f"    neutral texture_acc  = {n_tex:.4f}")
    print(f"    cultural texture_acc = {c_tex:.4f}  (Δ = {delta2:+.4f})")
    print("    VERDICT:", end=" ")
    if delta2 > 0.10:
        print("✓ CONFIRMED — >10% cultural recovery from steering")
    elif delta2 > 0.05:
        print("✓ PARTIAL — moderate improvement")
    else:
        print("✗ INCONCLUSIVE — <5% improvement; revisit prompt design or dataset")

    # H3 — APO
    a_tex  = apo_best.get("texture_accuracy", 0)
    a_shp  = apo_best.get("shape_accuracy", 0)
    delta3 = a_tex - c_tex
    print(f"\\nH3: APO discovers better prompts than hand-crafted cultural prompts")
    print(f"    cultural texture_acc = {c_tex:.4f}")
    print(f"    APO best texture_acc = {a_tex:.4f}  (Δ = {delta3:+.4f})")
    print(f"    APO best shape_acc   = {a_shp:.4f}  (threshold ≥ 0.75)")
    print("    VERDICT:", end=" ")
    if a_tex > c_tex and a_shp >= 0.75:
        print(f"✓ CONFIRMED (+{delta3:.4f} texture accuracy within constraint)")
    elif a_tex > c_tex:
        print(f"⚠ PARTIAL — APO improved texture but shape acc {a_shp:.4f} < 0.75")
    else:
        print("✗ NOT CONFIRMED — hand-crafted prompt performs comparably or better")

    print("=" * 82)


def run_phase5(config=CONFIG):
    \"\"\"Phase 5 — Final Analysis and Synthesis.\"\"\"

    logger.info("=" * 60)
    logger.info("PHASE 5 — FINAL ANALYSIS AND SYNTHESIS")
    logger.info("=" * 60)

    results_dir = Path(config["paths"]["results"])
    all_results = load_phase_results(results_dir)

    if not all_results:
        logger.error("No results found. Run phases 2, 3A, 3B first."); return

    print_results_table(all_results)
    test_hypotheses(all_results, alpha=config["evaluation"]["alpha"])

    # ── Optional: load probing results ────────────────────────────────────────
    probing_results = None
    probe_dir = results_dir / "probing"
    if probe_dir.exists():
        probe_files = sorted(probe_dir.glob("probing_results_*.json"),
                             key=lambda p: p.stat().st_mtime)
        if probe_files:
            with open(probe_files[-1]) as f:
                probing_results = json.load(f)

    # ── Optional: load APO history ────────────────────────────────────────────
    apo_history = None
    apo_dir = Path(config["paths"]["apo_prompts"])
    if apo_dir.exists():
        apo_files = sorted(apo_dir.glob("apo_history_*.json"),
                           key=lambda p: p.stat().st_mtime)
        if apo_files:
            with open(apo_files[-1]) as f:
                apo_history = json.load(f)

    # ── Generate all final figures ────────────────────────────────────────────
    logger.info("Generating all final figures ...")
    viz = ResultsVisualizer(config)
    viz.generate_all_figures(
        metrics_by_condition=all_results,
        probing_results=probing_results,
        apo_history=apo_history,
    )

    # ── Save master summary ───────────────────────────────────────────────────
    master = {
        "project": "Dynamic Perceptual Steering for African Cultural Competency",
        "team": "CMU Africa — Team 21/23",
        "results": all_results,
    }
    (results_dir / "MASTER_SUMMARY.json").write_text(json.dumps(master, indent=2))
    logger.info("Master summary saved. Phase 5 complete — ready to write final report.")


# Uncomment to execute:
# run_phase5(CONFIG)\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION GUIDE
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
<a id='execution-guide'></a>
## Part 16 — Execution Guide

### Running the Full Pipeline (notebook)

Execute cells in order, then call the phase functions:

```python
# Step 1: Bootstrap folder structure
main()

# Step 2: Place images in data/raw/ and populate annotations.json files.
#         SURA and Africa-500 require local annotation.
#         Afri-MCQA, CulturalVQA, Afri-Aya auto-download on first call.

# Step 3: Run experiments in sequence
neutral_results, neutral_metrics  = run_phase2(CONFIG)   # ~4 h on V100
all_results_3a, metrics_3a        = run_phase3a(CONFIG)  # ~6 h
best_prompt, apo_history, apo_m   = run_phase3b(CONFIG)  # ~12 h
probing_results                   = run_phase3c(CONFIG)  # ~8 h
adv_results                       = run_phase4(CONFIG)   # ~2 h (images needed)
run_phase5(CONFIG)                                        # minutes
```

### Running on PSC Bridges-2 (SLURM)

```bash
# One-time setup
bash scripts/setup_environment.sh

# Submit jobs (each waits for data to be in place)
sbatch scripts/run_baseline.slurm
sbatch scripts/run_steering.slurm
sbatch scripts/run_apo.slurm
sbatch scripts/run_probing.slurm
sbatch scripts/run_adversarial.slurm

# Final analysis (run interactively after all jobs complete)
python experiments/phase5_final_analysis.py --config configs/config.yaml
```

### Dataset Acquisition Summary

| Source | Action |
|--------|--------|
| **SURA Benchmark** | Request from research authors; place in `data/raw/SURA/` |
| **Africa-500** | Team-curated images; place in `data/raw/Africa500/` |
| **Afri-MCQA** | Auto-downloaded from HuggingFace on first `load_all_datasets()` call |
| **CulturalVQA** | Auto-downloaded from HuggingFace on first `load_all_datasets()` call |
| **Afri-Aya** | Auto-downloaded from HuggingFace on first `load_all_datasets()` call |
| **Adversarial images** | Collect manually; place in `data/raw/adversarial/` |

### Key Output Files

| Path | Description |
|------|-------------|
| `results/phase2_summary.json` | Neutral baseline metrics (H1 evidence) |
| `results/phase3a_summary.json` | Manual steering metrics for all conditions |
| `results/phase3b_summary.json` | APO best prompt + top-5 discovered prompts |
| `results/phase3c_summary.json` | Mechanistic probing statistics |
| `results/MASTER_SUMMARY.json` | Unified summary for final report |
| `results/comparison_summary.csv` | Side-by-side metric table for all conditions |
| `results/figures/shape_bias_scatter.pdf` | Figure 1 — Shape vs texture decisions |
| `results/figures/confidence_distributions.pdf` | Figure 2 — Token confidence (H2/H3) |
| `results/figures/accuracy_tradeoff.pdf` | Figure 3 — Over-steering frontier |
| `results/figures/famous_vs_everyday.pdf` | Figure 4 — Insight 1 |
| `results/figures/category_heatmap.pdf` | Figure 5 — Per-category texture accuracy |
| `results/figures/apo_progress.pdf` | Figure 6 — APO convergence curve |\
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE AND WRITE
# ═══════════════════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out = ROOT / "dynamic_perceptual_steering.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

nb_size = out.stat().st_size / 1024
print(f"Notebook written : {out.resolve()}")
print(f"File size        : {nb_size:.1f} KB")
print(f"Total cells      : {len(cells)}")
print(f"  Markdown cells : {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
print(f"  Code cells     : {sum(1 for c in cells if c['cell_type'] == 'code')}")
