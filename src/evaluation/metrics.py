"""
metrics.py
===========
Evaluation metrics for the Dynamic Perceptual Steering project.

Core metrics (following Geirhos et al. 2019 and Gavrikov et al. 2025):

1. Shape Accuracy   : fraction of images where model correctly identified
                      the shape/functional category
2. Texture Accuracy : fraction of images where model correctly identified
                      the cultural texture/origin
3. Cue Accuracy     : Shape Accuracy + Texture Accuracy
                      (fraction of images where model got EITHER label right)
4. Shape Bias       : Shape Accuracy / Cue Accuracy
                      (how often shape wins when model gets SOMETHING right)
                      Shape Bias = 1.0 → always answers shape
                      Shape Bias = 0.0 → always answers texture/culture
5. Cultural Recovery Rate : improvement in texture recognition from
                             neutral to steered prompt

For each metric, we also compute:
- Per-category breakdown (textiles vs architecture vs food, etc.)
- Per-region breakdown (West Africa vs East Africa, etc.)
- Famous vs everyday split (your Insight 1)
- Statistical significance (two-sided t-test)
"""

import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Response Parsing
# ─────────────────────────────────────────────────────────────

def parse_decision(response: str,
                   shape_label: str,
                   texture_label: str) -> Tuple[bool, bool]:
    """
    Parse a VLM text response to determine if it mentions
    the shape label, texture label, or both.

    This is more flexible than exact matching — we check if
    the label keywords appear anywhere in the response.

    Args:
        response      : VLM's text output
        shape_label   : the functional label (e.g. "bowl")
        texture_label : the cultural label (e.g. "kente")

    Returns:
        (shape_mentioned, texture_mentioned) : tuple of booleans
    """
    response_lower = response.lower()
    shape_lower = shape_label.lower()
    texture_lower = texture_label.lower()

    # Check if shape label is mentioned
    shape_mentioned = shape_lower in response_lower

    # Check if texture label is mentioned
    # Also check common related terms for the texture
    texture_mentioned = texture_lower in response_lower
    if not texture_mentioned:
        # Expand texture check with common synonyms/related terms
        texture_synonyms = _get_texture_synonyms(texture_lower)
        texture_mentioned = any(t in response_lower for t in texture_synonyms)

    return shape_mentioned, texture_mentioned


def _get_texture_synonyms(texture_label: str) -> List[str]:
    """
    Return common synonyms and related terms for African texture labels.
    This ensures we don't miss correct responses due to phrasing variation.
    """
    synonyms_map = {
        "kente": ["kente", "akan", "ghanaian cloth", "kente cloth", "ashanti"],
        "mudcloth": ["mudcloth", "mud cloth", "bogolanfini", "bogolan",
                     "malian", "bamana"],
        "adire": ["adire", "yoruba", "resist-dyed", "indigo cloth", "batik"],
        "ankara": ["ankara", "wax print", "african print", "dutch wax",
                   "african fabric"],
        "kanga": ["kanga", "kangas", "east african cloth", "swahili"],
        "shweshwe": ["shweshwe", "three cats", "south african",
                     "xhosa cloth", "sotho"],
        "raffia": ["raffia", "palm leaf", "woven grass"],
        "adinkra": ["adinkra", "akan symbols", "ghanaian symbols"],
        "zulu": ["zulu", "beadwork", "nguni", "south african bead"],
        "maasai": ["maasai", "masai", "east african beads", "shuka"],
        "sudano-sahelian": ["sudano", "sahelian", "mali architecture",
                            "mud brick", "timbuktu", "djenne"],
        "swahili": ["swahili", "zanzibar", "coastal east african"],
    }
    return synonyms_map.get(texture_label, [texture_label])


# ─────────────────────────────────────────────────────────────
# Core Metric Functions
# ─────────────────────────────────────────────────────────────

def compute_shape_bias(shape_accuracies: List[float],
                       texture_accuracies: List[float]) -> float:
    """
    Compute Shape Bias = Shape Accuracy / Cue Accuracy
    following Geirhos et al. (2019) Equation 2.

    Args:
        shape_accuracies   : list of 1/0 for each image (1=shape mentioned)
        texture_accuracies : list of 1/0 for each image (1=texture mentioned)

    Returns:
        Shape Bias score in [0, 1]
    """
    shape_acc = np.mean(shape_accuracies)
    texture_acc = np.mean(texture_accuracies)
    cue_acc = shape_acc + texture_acc

    if cue_acc == 0:
        logger.warning("Cue accuracy is 0 — model got nothing right. "
                       "Check label parsing.")
        return 0.5  # undefined, return midpoint

    return shape_acc / cue_acc


def compute_cue_accuracy(shape_accuracies: List[float],
                          texture_accuracies: List[float]) -> float:
    """
    Cue Accuracy = fraction of images where model identified
    EITHER the shape OR the texture correctly.
    """
    both = [max(s, t) for s, t in zip(shape_accuracies, texture_accuracies)]
    return float(np.mean(both))


def compute_cultural_recovery_rate(baseline_texture_acc: float,
                                    steered_texture_acc: float) -> float:
    """
    Cultural Recovery Rate = improvement in texture/cultural recognition
    after applying cultural steering prompt.

    Positive value = steering helped recover cultural knowledge.
    """
    return steered_texture_acc - baseline_texture_acc


def compute_all_metrics(results: List[Dict]) -> Dict:
    """
    Compute all metrics from a list of per-image result dicts.

    Each result dict should have:
        shape_label      : str
        texture_label    : str
        response         : str (model's text output)
        prompt_type      : str (neutral/structural/cultural/sequential)
        category         : str
        region           : str
        is_famous        : bool

    Returns:
        Dict with all computed metrics
    """
    if not results:
        logger.warning("Empty results list passed to compute_all_metrics.")
        return {}

    # Parse each response
    shape_hits = []
    texture_hits = []

    for r in results:
        shape_hit, texture_hit = parse_decision(
            r["response"], r["shape_label"], r["texture_label"]
        )
        shape_hits.append(int(shape_hit))
        texture_hits.append(int(texture_hit))

    # Core metrics
    shape_accuracy = float(np.mean(shape_hits))
    texture_accuracy = float(np.mean(texture_hits))
    cue_accuracy = compute_cue_accuracy(shape_hits, texture_hits)
    shape_bias = compute_shape_bias(shape_hits, texture_hits)

    metrics = {
        "n_images": len(results),
        "shape_accuracy": round(shape_accuracy, 4),
        "texture_accuracy": round(texture_accuracy, 4),
        "cue_accuracy": round(cue_accuracy, 4),
        "shape_bias": round(shape_bias, 4),
        "n_shape_decisions": sum(shape_hits),
        "n_texture_decisions": sum(texture_hits),
        "prompt_type": results[0].get("prompt_type", "unknown"),
    }

    # Breakdown by category
    metrics["by_category"] = _breakdown_by_field(
        results, shape_hits, texture_hits, field="category"
    )

    # Breakdown by region
    metrics["by_region"] = _breakdown_by_field(
        results, shape_hits, texture_hits, field="region"
    )

    # Famous vs everyday (Insight 1)
    famous_mask = [r.get("is_famous", False) for r in results]
    everyday_mask = [not f for f in famous_mask]

    if any(famous_mask):
        metrics["famous_items"] = _compute_subset_metrics(
            shape_hits, texture_hits, famous_mask
        )
    if any(everyday_mask):
        metrics["everyday_items"] = _compute_subset_metrics(
            shape_hits, texture_hits, everyday_mask
        )

    return metrics


def _breakdown_by_field(results: List[Dict],
                         shape_hits: List[int],
                         texture_hits: List[int],
                         field: str) -> Dict:
    """Compute metrics broken down by a categorical field."""
    groups = defaultdict(lambda: {"shape": [], "texture": []})

    for i, r in enumerate(results):
        key = r.get(field, "unknown")
        groups[key]["shape"].append(shape_hits[i])
        groups[key]["texture"].append(texture_hits[i])

    breakdown = {}
    for key, data in groups.items():
        s_acc = float(np.mean(data["shape"]))
        t_acc = float(np.mean(data["texture"]))
        c_acc = compute_cue_accuracy(data["shape"], data["texture"])
        sb = compute_shape_bias(data["shape"], data["texture"])
        breakdown[key] = {
            "n": len(data["shape"]),
            "shape_accuracy": round(s_acc, 4),
            "texture_accuracy": round(t_acc, 4),
            "cue_accuracy": round(c_acc, 4),
            "shape_bias": round(sb, 4),
        }

    return breakdown


def _compute_subset_metrics(shape_hits: List[int],
                              texture_hits: List[int],
                              mask: List[bool]) -> Dict:
    """Compute metrics for a subset defined by a boolean mask."""
    s = [sh for sh, m in zip(shape_hits, mask) if m]
    t = [th for th, m in zip(texture_hits, mask) if m]

    if not s:
        return {}

    return {
        "n": len(s),
        "shape_accuracy": round(float(np.mean(s)), 4),
        "texture_accuracy": round(float(np.mean(t)), 4),
        "cue_accuracy": round(compute_cue_accuracy(s, t), 4),
        "shape_bias": round(compute_shape_bias(s, t), 4),
    }


# ─────────────────────────────────────────────────────────────
# Statistical Testing
# ─────────────────────────────────────────────────────────────

def compare_conditions(results_a: List[Dict],
                        results_b: List[Dict],
                        condition_a: str = "Neutral",
                        condition_b: str = "Steered",
                        alpha: float = 0.05) -> Dict:
    """
    Compare two experimental conditions using a two-sided t-test
    on shape bias scores, following Gavrikov et al. methodology.

    Args:
        results_a   : per-image results for condition A (e.g. neutral)
        results_b   : per-image results for condition B (e.g. steered)
        condition_a : name for condition A
        condition_b : name for condition B
        alpha       : significance threshold

    Returns:
        Dict with t-statistic, p-value, and significance verdict
    """
    # Per-image shape decisions (binary: 1=shape, 0=texture/other)
    def get_shape_decisions(results):
        decisions = []
        for r in results:
            shape_hit, texture_hit = parse_decision(
                r["response"], r["shape_label"], r["texture_label"]
            )
            # 1 if shape, 0 if texture (for paired comparison)
            decisions.append(int(shape_hit))
        return np.array(decisions)

    decisions_a = get_shape_decisions(results_a)
    decisions_b = get_shape_decisions(results_b)

    # Paired t-test (same images, different prompts)
    if len(decisions_a) == len(decisions_b):
        t_stat, p_value = stats.ttest_rel(decisions_a, decisions_b)
        test_type = "paired t-test"
    else:
        t_stat, p_value = stats.ttest_ind(decisions_a, decisions_b)
        test_type = "independent t-test"

    significant = p_value < alpha

    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "mean_a": float(np.mean(decisions_a)),
        "mean_b": float(np.mean(decisions_b)),
        "difference": float(np.mean(decisions_b) - np.mean(decisions_a)),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": significant,
        "alpha": alpha,
        "test_type": test_type,
        "interpretation": (
            f"{'Significant' if significant else 'Not significant'} difference "
            f"(p={p_value:.4f} {'<' if significant else '≥'} {alpha})"
        )
    }
