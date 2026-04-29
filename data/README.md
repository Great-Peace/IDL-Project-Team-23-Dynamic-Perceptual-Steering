# Local Data Layout

Use `python scripts/bootstrap_data_layout.py` to create the local folder structure expected by the loaders.

Important notes:

- `data/raw/`, `data/processed/`, and `data/cue_conflict/` are gitignored for size reasons.
- Real datasets should live under `data/raw/`.
- The loaders expect `annotations.json` for `SURA` and `Africa500`.
- Template files are created locally as `annotations.template.json`.

Recommended first-run path:

1. Run the bootstrap script.
2. Place a very small number of real images into `data/raw/SURA/images/...` or `data/raw/Africa500/images/...`.
3. Create `annotations.json` from the local template.
4. Lower `dataset.target_size` to `10`.
5. Run `python experiments/phase2_baseline.py --config configs/config.yaml`.
