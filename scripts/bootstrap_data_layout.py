"""
bootstrap_data_layout.py
========================

Creates the local folder structure expected by the dataset loaders and
drops small annotation templates into the raw dataset folders.

Usage:
    python scripts/bootstrap_data_layout.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    print(f"[ok] {path}")


def write_json_if_missing(path: Path, payload) -> None:
    if path.exists():
        print(f"[skip] {path}")
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[new] {path}")


def write_text_if_missing(path: Path, text: str) -> None:
    if path.exists():
        print(f"[skip] {path}")
        return
    path.write_text(text, encoding="utf-8")
    print(f"[new] {path}")


def main() -> None:
    data_raw = ROOT / "data" / "raw"
    data_processed = ROOT / "data" / "processed"
    data_cue_conflict = ROOT / "data" / "cue_conflict"
    results = ROOT / "results"

    for path in [
        data_raw,
        data_processed,
        data_cue_conflict,
        results / "baseline",
        results / "steering",
        results / "apo",
        results / "probing",
        results / "adversarial",
        results / "figures",
    ]:
        ensure_dir(path)

    sura_dir = data_raw / "SURA"
    africa500_dir = data_raw / "Africa500"
    afrimcqa_dir = data_raw / "Afri-MCQA"
    culturalvqa_dir = data_raw / "CulturalVQA"
    afriaya_dir = data_raw / "Afri-Aya"
    adversarial_dir = data_raw / "adversarial"

    sura_categories = [
        "textiles",
        "architecture",
        "everyday_objects",
        "food_and_drink",
        "ritual_items",
        "musical_instruments",
    ]

    ensure_dir(sura_dir / "images")
    for category in sura_categories:
        ensure_dir(sura_dir / "images" / category)

    ensure_dir(africa500_dir / "images")
    ensure_dir(afrimcqa_dir)
    ensure_dir(culturalvqa_dir)
    ensure_dir(afriaya_dir)
    ensure_dir(adversarial_dir / "western_shapes")
    ensure_dir(adversarial_dir / "african_textures")

    sura_template = [
        {
            "image_file": "textiles/kente_001.jpg",
            "shape_label": "cloth",
            "texture_label": "kente",
            "category": "textiles",
            "region": "West Africa",
            "is_famous": False,
            "metadata": {
                "country": "Ghana",
                "notes": "Replace with real annotation values.",
            },
        }
    ]
    africa500_template = [
        {
            "image_file": "basket_001.jpg",
            "shape_label": "basket",
            "texture_label": "raffia",
            "category": "everyday_objects",
            "region": "Central Africa",
            "is_famous": False,
        }
    ]

    write_json_if_missing(sura_dir / "annotations.template.json", sura_template)
    write_json_if_missing(
        africa500_dir / "annotations.template.json", africa500_template
    )

    write_text_if_missing(
        data_raw / "README_LOCAL_DATA.txt",
        "\n".join(
            [
                "This folder is intentionally gitignored.",
                "Put real datasets here before running the experiments.",
                "",
                "Expected high-priority sources:",
                "- data/raw/SURA/",
                "- data/raw/Africa500/",
                "- data/raw/Afri-MCQA/",
                "- data/raw/CulturalVQA/",
                "",
                "Copy annotations.template.json to annotations.json after replacing",
                "the example rows with real labels and file paths.",
            ]
        )
        + "\n",
    )

    print("\nDone. You can now place real data into data/raw/ and prepare annotations.json files.")


if __name__ == "__main__":
    main()
