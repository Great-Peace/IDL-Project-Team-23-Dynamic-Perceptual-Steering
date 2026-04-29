"""
dataset_loader.py
=================
Loads and unifies African cultural image datasets from multiple sources:
  1. SURA Benchmark (local files)
  2. Africa-500 (local files)
  3. Afri-MCQA (HuggingFace)
  4. CulturalVQA — African subset (HuggingFace)
  5. Afri-Aya (HuggingFace)

Each image in the final dataset has:
  - image          : PIL.Image
  - image_path     : str
  - shape_label    : str  — functional object type (e.g. "bag", "bowl", "building")
  - texture_label  : str  — cultural/material label (e.g. "kente", "mudcloth", "adire")
  - category       : str  — artifact category (textiles, architecture, etc.)
  - region         : str  — African region (West Africa, East Africa, etc.)
  - source         : str  — which dataset this came from
  - is_famous      : bool — True if globally prominent (for Insight 1 analysis)

Usage:
    from src.data.dataset_loader import load_all_datasets
    dataset = load_all_datasets(config)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

from PIL import Image
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Data Record
# ─────────────────────────────────────────────────────────────

@dataclass
class CulturalImageRecord:
    """
    A single record in the African cultural image dataset.
    Every image must have both a shape_label (what it IS functionally)
    and a texture_label (what cultural origin it represents).
    These two labels are what enable cue-conflict evaluation.
    """
    image_path: str
    shape_label: str          # e.g. "bag", "cloth", "building", "bowl"
    texture_label: str        # e.g. "kente", "mudcloth", "sudano-sahelian"
    category: str             # e.g. "textiles", "architecture"
    region: str               # e.g. "West Africa"
    source: str               # e.g. "SURA", "Afri-MCQA"
    is_famous: bool = False   # True for globally iconic items (Djenné, etc.)
    metadata: Dict = field(default_factory=dict)

    def load_image(self) -> Image.Image:
        """Load and return the PIL image."""
        return Image.open(self.image_path).convert("RGB")

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


# ─────────────────────────────────────────────────────────────
# SURA Benchmark Loader
# ─────────────────────────────────────────────────────────────

class SURALoader:
    """
    Loads the SURA Benchmark.
    SURA is designed to evaluate African cultural/geographical knowledge.

    Expected local directory structure:
        data/raw/SURA/
            images/
                textiles/
                    image_001.jpg  ...
                architecture/
                    ...
            annotations.json   ← must contain shape_label, texture_label, region, is_famous
    """

    def __init__(self, sura_dir: str):
        self.sura_dir = Path(sura_dir)
        self.annotations_path = self.sura_dir / "annotations.json"

    def load(self) -> List[CulturalImageRecord]:
        records = []

        if not self.sura_dir.exists():
            logger.warning(f"SURA directory not found at {self.sura_dir}. Skipping.")
            return records

        if not self.annotations_path.exists():
            logger.warning(f"SURA annotations.json not found. Attempting image-only load.")
            return self._load_images_only()

        with open(self.annotations_path) as f:
            annotations = json.load(f)

        for ann in tqdm(annotations, desc="Loading SURA"):
            img_path = self.sura_dir / "images" / ann["image_file"]
            if not img_path.exists():
                logger.debug(f"Image not found: {img_path}")
                continue

            record = CulturalImageRecord(
                image_path=str(img_path),
                shape_label=ann.get("shape_label", "object"),
                texture_label=ann.get("texture_label", "african"),
                category=ann.get("category", "unknown"),
                region=ann.get("region", "Africa"),
                source="SURA",
                is_famous=ann.get("is_famous", False),
                metadata=ann.get("metadata", {})
            )
            records.append(record)

        logger.info(f"SURA: loaded {len(records)} records.")
        return records

    def _load_images_only(self) -> List[CulturalImageRecord]:
        """
        Fallback: if no annotations.json, scan image folders and infer
        category from subfolder name. Shape/texture labels will need
        manual review.
        """
        records = []
        images_dir = self.sura_dir / "images"
        if not images_dir.exists():
            return records

        for category_dir in images_dir.iterdir():
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            for img_file in category_dir.glob("*.jpg"):
                record = CulturalImageRecord(
                    image_path=str(img_file),
                    shape_label="object",          # needs manual annotation
                    texture_label="african",        # needs manual annotation
                    category=category,
                    region="Africa",               # needs manual annotation
                    source="SURA",
                    is_famous=False,
                    metadata={"needs_annotation": True}
                )
                records.append(record)

        logger.info(f"SURA (image-only): loaded {len(records)} records. "
                    f"Manual annotation of shape_label/texture_label required.")
        return records


# ─────────────────────────────────────────────────────────────
# Africa-500 Loader
# ─────────────────────────────────────────────────────────────

class Africa500Loader:
    """
    Loads Africa-500.
    Expected structure:
        data/raw/Africa500/
            images/
            annotations.json
    """

    def __init__(self, africa500_dir: str):
        self.africa500_dir = Path(africa500_dir)

    def load(self) -> List[CulturalImageRecord]:
        records = []

        if not self.africa500_dir.exists():
            logger.warning(f"Africa-500 directory not found at {self.africa500_dir}. Skipping.")
            return records

        annotations_path = self.africa500_dir / "annotations.json"
        images_dir = self.africa500_dir / "images"

        if not annotations_path.exists():
            logger.warning("Africa-500 annotations.json not found.")
            return records

        with open(annotations_path) as f:
            annotations = json.load(f)

        for ann in tqdm(annotations, desc="Loading Africa-500"):
            img_path = images_dir / ann["image_file"]
            if not img_path.exists():
                continue

            record = CulturalImageRecord(
                image_path=str(img_path),
                shape_label=ann.get("shape_label", "object"),
                texture_label=ann.get("texture_label", "african"),
                category=ann.get("category", "unknown"),
                region=ann.get("region", "Africa"),
                source="Africa500",
                is_famous=ann.get("is_famous", False),
            )
            records.append(record)

        logger.info(f"Africa-500: loaded {len(records)} records.")
        return records


# ─────────────────────────────────────────────────────────────
# Afri-MCQA Loader (HuggingFace)
# ─────────────────────────────────────────────────────────────

class AfriMCQALoader:
    """
    Loads the Afri-MCQA dataset from HuggingFace.
    https://huggingface.co/datasets/Atnafu/Afri-MCQA

    Afri-MCQA provides culturally-grounded image QA pairs.
    We extract the image + cultural answer as the texture_label
    and infer a shape_label from the question context.

    Note: Only pulls the VISUAL (image-grounded) subset.
    """

    HF_DATASET_ID = "Atnafu/Afri-MCQA"

    def __init__(self, save_dir: str, max_samples: int = 500):
        self.save_dir = Path(save_dir) / "Afri-MCQA"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples

    def load(self) -> List[CulturalImageRecord]:
        records = []

        try:
            logger.info(f"Downloading Afri-MCQA from HuggingFace: {self.HF_DATASET_ID}")
            ds = load_dataset(self.HF_DATASET_ID, split="test", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load Afri-MCQA: {e}")
            logger.info("Please manually download from: "
                        "https://huggingface.co/datasets/Atnafu/Afri-MCQA")
            return records

        count = 0
        for idx, item in enumerate(tqdm(ds, desc="Processing Afri-MCQA")):
            if count >= self.max_samples:
                break

            # Only process items that have an image
            if item.get("image") is None:
                continue

            # Save image locally
            img_filename = f"afrimcqa_{idx:05d}.jpg"
            img_path = self.save_dir / img_filename
            if not img_path.exists():
                try:
                    item["image"].save(str(img_path))
                except Exception:
                    continue

            # Extract labels from the QA pair
            # The question typically asks about cultural object
            question = item.get("question", "")
            answer = item.get("answer_text", item.get("answer", "african"))

            # Use the correct answer as texture label
            # and infer shape from question (simple heuristic)
            shape_label = self._infer_shape_from_question(question)
            texture_label = str(answer).lower().strip()

            record = CulturalImageRecord(
                image_path=str(img_path),
                shape_label=shape_label,
                texture_label=texture_label,
                category=self._infer_category(question),
                region=item.get("country", "Africa"),
                source="Afri-MCQA",
                is_famous=False,
                metadata={"question": question, "language": item.get("language", "en")}
            )
            records.append(record)
            count += 1

        logger.info(f"Afri-MCQA: loaded {len(records)} records.")
        return records

    def _infer_shape_from_question(self, question: str) -> str:
        """
        Simple keyword-based shape label inference.
        Maps question keywords to functional object types.
        """
        question = question.lower()
        shape_map = {
            "cloth": "cloth", "fabric": "cloth", "textile": "cloth",
            "dress": "garment", "wear": "garment", "outfit": "garment",
            "building": "building", "structure": "building", "mosque": "building",
            "house": "building", "church": "building", "palace": "building",
            "bowl": "bowl", "pot": "pot", "vessel": "vessel",
            "basket": "basket", "bag": "bag", "container": "container",
            "drum": "drum", "instrument": "instrument",
            "jewelry": "jewelry", "necklace": "jewelry", "bracelet": "jewelry",
            "mask": "mask", "statue": "statue", "sculpture": "sculpture",
            "food": "food", "dish": "food",
        }
        for keyword, shape in shape_map.items():
            if keyword in question:
                return shape
        return "object"

    def _infer_category(self, question: str) -> str:
        """Infer artifact category from question."""
        question = question.lower()
        if any(k in question for k in ["cloth", "fabric", "dress", "textile", "wear"]):
            return "textiles"
        if any(k in question for k in ["building", "mosque", "church", "structure", "house"]):
            return "architecture"
        if any(k in question for k in ["food", "dish", "meal", "drink"]):
            return "food_and_drink"
        if any(k in question for k in ["ritual", "ceremony", "dance", "festival"]):
            return "ritual_items"
        if any(k in question for k in ["drum", "instrument", "music"]):
            return "musical_instruments"
        return "everyday_objects"


# ─────────────────────────────────────────────────────────────
# CulturalVQA Loader (HuggingFace) — African Subset
# ─────────────────────────────────────────────────────────────

class CulturalVQALoader:
    """
    Loads African subset from CulturalVQA.
    https://huggingface.co/datasets/Atnafu/Afri-MCQA (culturalvqa.org)

    CulturalVQA has 2,378 image-question pairs from 11 countries.
    We filter for African countries: Ethiopia, Nigeria, Rwanda, etc.
    """

    # List of African countries in CulturalVQA
    AFRICAN_COUNTRIES = {
        "Ethiopia", "Nigeria", "Rwanda", "Kenya", "Ghana",
        "South Africa", "Egypt", "Morocco", "Senegal", "Tanzania",
        "Uganda", "Cameroon", "Ivory Coast", "Mali", "Zimbabwe"
    }

    def __init__(self, save_dir: str, max_samples: int = 300):
        self.save_dir = Path(save_dir) / "CulturalVQA"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples

    def load(self) -> List[CulturalImageRecord]:
        records = []

        try:
            logger.info("Downloading CulturalVQA from HuggingFace...")
            # Try the culturalvqa dataset
            ds = load_dataset("vcr-org/CulturalVQA", split="test",
                              trust_remote_code=True)
        except Exception as e:
            logger.warning(f"CulturalVQA load failed: {e}. Trying alternative...")
            try:
                ds = load_dataset("Otter-Research/CulturalVQA",
                                  split="test", trust_remote_code=True)
            except Exception as e2:
                logger.error(f"CulturalVQA unavailable: {e2}")
                logger.info("Download manually from: https://culturalvqa.org")
                return records

        count = 0
        for idx, item in enumerate(tqdm(ds, desc="Processing CulturalVQA")):
            if count >= self.max_samples:
                break

            # Filter for African countries
            country = item.get("country", "")
            if country not in self.AFRICAN_COUNTRIES:
                continue

            if item.get("image") is None:
                continue

            # Save image
            img_filename = f"culturalvqa_{idx:05d}.jpg"
            img_path = self.save_dir / img_filename
            if not img_path.exists():
                try:
                    item["image"].save(str(img_path))
                except Exception:
                    continue

            # CulturalVQA has facet: clothing, food, rituals, traditions, drinks
            facet = item.get("facet", "unknown")
            category = self._facet_to_category(facet)

            record = CulturalImageRecord(
                image_path=str(img_path),
                shape_label=self._facet_to_shape(facet),
                texture_label=item.get("answer", "african"),
                category=category,
                region=self._country_to_region(country),
                source="CulturalVQA",
                is_famous=False,
                metadata={"country": country, "facet": facet,
                          "question": item.get("question", "")}
            )
            records.append(record)
            count += 1

        logger.info(f"CulturalVQA: loaded {len(records)} African records.")
        return records

    def _facet_to_category(self, facet: str) -> str:
        mapping = {
            "clothing": "textiles", "food": "food_and_drink",
            "drinks": "food_and_drink", "rituals": "ritual_items",
            "traditions": "ritual_items"
        }
        return mapping.get(facet.lower(), "everyday_objects")

    def _facet_to_shape(self, facet: str) -> str:
        mapping = {
            "clothing": "garment", "food": "food",
            "drinks": "beverage", "rituals": "artifact",
            "traditions": "artifact"
        }
        return mapping.get(facet.lower(), "object")

    def _country_to_region(self, country: str) -> str:
        west = {"Nigeria", "Ghana", "Senegal", "Ivory Coast", "Mali", "Cameroon"}
        east = {"Ethiopia", "Kenya", "Tanzania", "Uganda", "Rwanda"}
        south = {"South Africa", "Zimbabwe"}
        north = {"Egypt", "Morocco"}
        if country in west:
            return "West Africa"
        if country in east:
            return "East Africa"
        if country in south:
            return "Southern Africa"
        if country in north:
            return "North Africa"
        return "Central Africa"


class AfriAyaLoader:
    """
    Loads the Afri-Aya dataset from HuggingFace.

    Afri-Aya is a community-curated African multimodal cultural dataset with
    captions and QA pairs. Since it is not organized around the exact
    shape/texture schema used in this repo, we apply lightweight heuristics to
    infer a functional label and use the collection context as the cultural cue.
    """

    HF_DATASET_ID = "CohereLabsCommunity/afri-aya"

    def __init__(self, save_dir: str, max_samples: int = 200):
        self.save_dir = Path(save_dir) / "Afri-Aya"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples

    def load(self) -> List[CulturalImageRecord]:
        records = []

        try:
            logger.info(f"Downloading Afri-Aya from HuggingFace: {self.HF_DATASET_ID}")
            ds = load_dataset(self.HF_DATASET_ID, split="train", trust_remote_code=True)
        except Exception as e:
            logger.error(f"Failed to load Afri-Aya: {e}")
            logger.info("Please manually download from: "
                        "https://huggingface.co/datasets/CohereLabsCommunity/afri-aya")
            return records

        count = 0
        for idx, item in enumerate(tqdm(ds, desc="Processing Afri-Aya")):
            if count >= self.max_samples:
                break

            if item.get("image") is None:
                continue

            img_filename = f"afriaya_{idx:05d}.jpg"
            img_path = self.save_dir / img_filename
            if not img_path.exists():
                try:
                    item["image"].save(str(img_path))
                except Exception:
                    continue

            category = str(item.get("category", "everyday_objects"))
            caption = str(item.get("caption_en", "") or "")
            original_query = str(item.get("original_query", "") or "")
            language = str(item.get("language", "") or "")

            record = CulturalImageRecord(
                image_path=str(img_path),
                shape_label=self._infer_shape(category, caption, original_query),
                texture_label=self._infer_texture_label(item),
                category=self._normalize_category(category, caption),
                region=self._infer_region(original_query, caption),
                source="Afri-Aya",
                is_famous=self._infer_is_famous(caption, original_query),
                metadata={
                    "language": language,
                    "category_raw": category,
                    "original_query": original_query,
                    "caption_en": caption,
                    "image_filename": item.get("image_filename", ""),
                    "source_url": item.get("source_url", ""),
                }
            )
            records.append(record)
            count += 1

        logger.info(f"Afri-Aya: loaded {len(records)} records.")
        return records

    def _normalize_category(self, category: str, caption: str) -> str:
        value = (category or "").strip().lower()
        if value in {"fashion", "clothing", "textile", "textiles"}:
            return "textiles"
        if value in {"architecture", "buildings", "religion", "landmark"}:
            return "architecture"
        if value in {"food", "drink", "cuisine"}:
            return "food_and_drink"
        if value in {"ritual", "ceremony", "festival", "tradition"}:
            return "ritual_items"
        if value in {"music", "instrument", "musical instruments"}:
            return "musical_instruments"

        caption_lower = caption.lower()
        if any(k in caption_lower for k in ["cloth", "fabric", "garment", "woven"]):
            return "textiles"
        if any(k in caption_lower for k in ["building", "church", "mosque", "house"]):
            return "architecture"
        if any(k in caption_lower for k in ["food", "dish", "meal", "drink"]):
            return "food_and_drink"
        if any(k in caption_lower for k in ["drum", "music", "instrument"]):
            return "musical_instruments"
        return "everyday_objects"

    def _infer_shape(self, category: str, caption: str, query: str) -> str:
        text = " ".join([category, caption, query]).lower()
        shape_map = {
            "cloth": "cloth",
            "fabric": "cloth",
            "garment": "garment",
            "dress": "garment",
            "basket": "basket",
            "bag": "bag",
            "pot": "pot",
            "bowl": "bowl",
            "building": "building",
            "church": "building",
            "mosque": "building",
            "house": "building",
            "mask": "mask",
            "drum": "drum",
            "instrument": "instrument",
            "food": "food",
        }
        for keyword, shape in shape_map.items():
            if keyword in text:
                return shape
        return "object"

    def _infer_texture_label(self, item: Dict) -> str:
        query = str(item.get("original_query", "") or "").strip()
        category = str(item.get("category", "") or "").strip()
        if query:
            return query.lower()
        if category:
            return category.lower()
        return "african"

    def _infer_region(self, query: str, caption: str) -> str:
        text = " ".join([query, caption]).lower()
        region_keywords = {
            "West Africa": ["ghana", "nigeria", "senegal", "mali", "ivory coast"],
            "East Africa": ["kenya", "uganda", "tanzania", "rwanda", "ethiopia"],
            "Southern Africa": ["south africa", "zimbabwe", "zulu", "xhosa"],
            "North Africa": ["morocco", "egypt", "sahel", "sahara"],
            "Central Africa": ["cameroon", "congo", "gabon", "central africa"],
        }
        for region, keywords in region_keywords.items():
            if any(keyword in text for keyword in keywords):
                return region
        return "Africa"

    def _infer_is_famous(self, caption: str, query: str) -> bool:
        text = " ".join([caption, query]).lower()
        famous_markers = [
            "landmark", "cathedral", "mosque", "heritage", "famous", "monument"
        ]
        return any(marker in text for marker in famous_markers)


# ─────────────────────────────────────────────────────────────
# Main Dataset Class
# ─────────────────────────────────────────────────────────────

class AfricanCulturalDataset:
    """
    Unified dataset of African cultural images for the
    Dynamic Perceptual Steering experiments.

    Holds a list of CulturalImageRecord objects.
    Can be filtered by category, region, source, etc.
    """

    def __init__(self, records: List[CulturalImageRecord]):
        self.records = records
        self._df = None  # lazy-loaded pandas DataFrame

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> CulturalImageRecord:
        return self.records[idx]

    def __iter__(self):
        return iter(self.records)

    @property
    def df(self) -> pd.DataFrame:
        """Return dataset as a pandas DataFrame (lazy-loaded)."""
        if self._df is None:
            self._df = pd.DataFrame([r.to_dict() for r in self.records])
        return self._df

    def filter_by_category(self, category: str) -> "AfricanCulturalDataset":
        """Return a subset filtered by artifact category."""
        filtered = [r for r in self.records if r.category == category]
        return AfricanCulturalDataset(filtered)

    def filter_by_region(self, region: str) -> "AfricanCulturalDataset":
        """Return a subset filtered by African region."""
        filtered = [r for r in self.records if r.region == region]
        return AfricanCulturalDataset(filtered)

    def filter_famous(self, is_famous: bool = True) -> "AfricanCulturalDataset":
        """Return famous (is_famous=True) or everyday (is_famous=False) items."""
        filtered = [r for r in self.records if r.is_famous == is_famous]
        return AfricanCulturalDataset(filtered)

    def get_category_distribution(self) -> Dict[str, int]:
        """Return count of records per category."""
        dist = {}
        for r in self.records:
            dist[r.category] = dist.get(r.category, 0) + 1
        return dist

    def get_region_distribution(self) -> Dict[str, int]:
        """Return count of records per region."""
        dist = {}
        for r in self.records:
            dist[r.region] = dist.get(r.region, 0) + 1
        return dist

    def save(self, output_path: str):
        """Save dataset manifest as JSON."""
        manifest = [r.to_dict() for r in self.records]
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Dataset manifest saved to {output_path} ({len(self.records)} records).")

    @classmethod
    def load_from_manifest(cls, manifest_path: str) -> "AfricanCulturalDataset":
        """Reload a previously saved dataset manifest."""
        with open(manifest_path) as f:
            manifest = json.load(f)
        records = [CulturalImageRecord(**item) for item in manifest]
        return cls(records)

    def print_summary(self):
        """Print a summary of the dataset."""
        print(f"\n{'='*50}")
        print(f"African Cultural Dataset Summary")
        print(f"{'='*50}")
        print(f"Total records     : {len(self.records)}")
        print(f"\nBy Category:")
        for cat, count in sorted(self.get_category_distribution().items()):
            print(f"  {cat:<25} {count}")
        print(f"\nBy Region:")
        for reg, count in sorted(self.get_region_distribution().items()):
            print(f"  {reg:<25} {count}")
        famous = sum(1 for r in self.records if r.is_famous)
        print(f"\nFamous landmarks  : {famous}")
        print(f"Everyday objects  : {len(self.records) - famous}")
        sources = {}
        for r in self.records:
            sources[r.source] = sources.get(r.source, 0) + 1
        print(f"\nBy Source:")
        for src, count in sorted(sources.items()):
            print(f"  {src:<25} {count}")
        print(f"{'='*50}\n")


# ─────────────────────────────────────────────────────────────
# Master Load Function
# ─────────────────────────────────────────────────────────────

def load_all_datasets(config: dict) -> AfricanCulturalDataset:
    """
    Load all configured datasets and merge into a single
    AfricanCulturalDataset. Saves a manifest for reproducibility.

    Args:
        config: Loaded from configs/config.yaml

    Returns:
        AfricanCulturalDataset with all records merged
    """
    raw_dir = config["paths"]["data_raw"]
    processed_dir = config["paths"]["data_processed"]
    os.makedirs(processed_dir, exist_ok=True)

    manifest_path = os.path.join(processed_dir, "dataset_manifest.json")

    # If manifest already exists, reload it (saves time)
    if os.path.exists(manifest_path):
        logger.info(f"Reloading dataset from manifest: {manifest_path}")
        dataset = AfricanCulturalDataset.load_from_manifest(manifest_path)
        dataset.print_summary()
        return dataset

    all_records = []

    # 1. SURA Benchmark
    sura_loader = SURALoader(os.path.join(raw_dir, "SURA"))
    all_records.extend(sura_loader.load())

    # 2. Africa-500
    africa500_loader = Africa500Loader(os.path.join(raw_dir, "Africa500"))
    all_records.extend(africa500_loader.load())

    # 3. Afri-MCQA (HuggingFace)
    afrimcqa_loader = AfriMCQALoader(raw_dir, max_samples=400)
    all_records.extend(afrimcqa_loader.load())

    # 4. CulturalVQA African subset (HuggingFace)
    culturalvqa_loader = CulturalVQALoader(raw_dir, max_samples=200)
    all_records.extend(culturalvqa_loader.load())

    # 5. Afri-Aya (HuggingFace)
    afriaya_loader = AfriAyaLoader(raw_dir, max_samples=200)
    all_records.extend(afriaya_loader.load())

    if len(all_records) == 0:
        logger.error(
            "No records loaded! Please download at least one dataset.\n"
            "See README.md for dataset download instructions."
        )
        return AfricanCulturalDataset([])

    dataset = AfricanCulturalDataset(all_records)
    dataset.print_summary()

    # Save manifest for reproducibility
    dataset.save(manifest_path)

    return dataset
