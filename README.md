# Dynamic Perceptual Steering

Orchestrating shape and texture biases in vision-language models for African cultural competency.

CMU Africa Deep Learning Project | Team 21/23  
Boniface Godwin, Jean De Dieu Iradukunda, Oyindamola Olatunji, Peace Bakare

## Overview

This repository studies whether language can steer a vision-language model away from default, Western-centric visual interpretation and toward African cultural recognition.

The project is grounded in three connected ideas:

1. Geirhos et al. (2019) showed that standard ImageNet-trained CNNs are strongly texture-biased under cue conflict.
2. Gavrikov et al. (2025) showed that VLM cue preference can be steered with language, but only partially.
3. Our extension asks whether that same steering mechanism can recover African cultural textures, materials, and regional visual identity that are otherwise flattened into generic object descriptions.

We refer to this failure mode as perceptual erasure: the model identifies what an object is at a generic functional level while suppressing the cultural information that makes it locally meaningful.

## Current Research Position

This repo should be read as a midterm-to-final research codebase, not a finished benchmark release.

What is already supported in code:

- Neutral baseline evaluation
- Manual prompt steering
- Sequential Dual-Lens prompting
- Automated Prompt Optimization (APO) with an open-source optimizer
- Token-confidence probing
- Final aggregation and figure generation

What is still project-dependent:

- Populating the dataset
- Verifying annotation quality and balance
- Running experiments end to end on PSC
- Interpreting results carefully enough to support stronger cultural claims

## Main Research Question

Can a VLM be talked into seeing African cultural artifacts differently at runtime, without retraining?

More specifically:

- Under a neutral prompt, does the model default to generic shape or function?
- Under cultural steering, does it recover regionally meaningful texture or material identity?
- Is that knowledge absent, or merely suppressed?
- How much cultural recovery is possible before the model over-steers and loses functional accuracy?

## Midterm Direction

The proposal and midterm reports in `papers/` point to the same central direction:

- Everyday African artifacts are more likely to be erased than globally famous landmarks.
- Steering may surface latent knowledge that the model does not reveal under neutral prompting.
- The central technical tension is not just recovery, but the tradeoff between cultural sensitivity and functional accuracy.

That tradeoff is the reason this repo emphasizes both cue recovery and over-steering analysis.

## Repository Layout

```text
dynamic_perceptual_steering/
|-- configs/
|   |-- config.yaml
|   |-- dataset_config.yaml
|   |-- experiment_config.yaml
|   `-- model_config.yaml
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- cue_conflict/
|-- experiments/
|   |-- phase2_baseline.py
|   |-- phase3a_manual_steering.py
|   |-- phase3b_apo.py
|   |-- phase3c_probing.py
|   |-- phase4_adversarial.py
|   `-- phase5_final_analysis.py
|-- papers/
|-- results/
|   |-- baseline/
|   |-- steering/
|   |-- apo/
|   |-- probing/
|   |-- adversarial/
|   `-- figures/
|-- scripts/
|   |-- setup_environment.sh
|   |-- run_baseline.slurm
|   |-- run_steering.slurm
|   |-- run_apo.slurm
|   |-- run_probing.slurm
|   `-- run_adversarial.slurm
|-- src/
|   |-- data/
|   |-- evaluation/
|   |-- models/
|   |-- probing/
|   |-- steering/
|   `-- visualization/
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Source Modules

- `src/data/dataset_loader.py`: unified loader for local and HuggingFace-backed sources
- `src/data/cue_conflict_synthesizer.py`: style-transfer-based cue-conflict image synthesis
- `src/models/internvl_wrapper.py`: InternVL-3 loading, generation, and token confidence extraction
- `src/models/clip_baseline.py`: CLIP vision-only comparison baseline
- `src/evaluation/metrics.py`: shape accuracy, texture accuracy, cue accuracy, shape bias, comparisons
- `src/evaluation/evaluator.py`: dataset-wide evaluation runner
- `src/steering/prompts.py`: prompt registry for baseline, cultural, sequential, and adversarial settings
- `src/steering/dual_lens.py`: two-stage structural-then-cultural prompting
- `src/steering/apo.py`: prompt search loop with an open-source optimizer model
- `src/probing/confidence_probing.py`: confidence probing over shape versus texture tokens
- `src/visualization/plots.py`: publication-oriented plots and final figure generation

## Experiment Phases

### Phase 2: Neutral Baseline

Run the model with a neutral prompt and measure:

- Shape Accuracy
- Texture Accuracy
- Cue Accuracy
- Shape Bias

This is the baseline test for perceptual erasure.

### Phase 3A: Manual Steering

Evaluate hand-crafted prompts:

- `neutral`
- `structural`
- `cultural`
- `cultural_geometric`
- `cultural_expert`
- `sequential`

This phase measures whether carefully designed prompts can recover cultural recognition and whether they damage functional recognition.

### Phase 3B: Automated Prompt Optimization

Use an open-source optimizer model to search for prompts that improve cultural recovery while preserving minimum functional accuracy.

### Phase 3C: Confidence Probing

Probe shape versus texture token confidence to test whether cultural information is:

- encoded and suppressed
- partially accessible
- or genuinely absent

This phase supports the mechanistic claim behind the project.

### Phase 4: Adversarial Texture Steering

Apply African texture to iconic Western shapes and test the limits of steering.

This phase is optional, but useful if the goal is to quantify cultural stubbornness rather than only recovery.

### Phase 5: Final Analysis

Aggregate metrics, compare conditions, generate plots, and produce final summaries for reporting.

## Setup on PSC

### 1. Clone the repository

```bash
cd /ocean/projects/YOUR_PROJECT_ID/$USER
git clone https://github.com/JeanIrad/dl_project23.git dynamic_perceptual_steering
cd dynamic_perceptual_steering
```

### 2. Create the environment

```bash
bash scripts/setup_environment.sh
```

### 3. Replace placeholders

Before running anything, confirm:

- `HF_HOME` points to the cache location you want on PSC
- `PROJECT_DIR` is set if you do not want the scripts to use `SLURM_SUBMIT_DIR`
- `your_email@andrew.cmu.edu` in every `scripts/*.slurm` if you want job email notifications

### 4. Populate the dataset

Expected local structure:

```text
data/raw/
|-- SURA/
|   |-- images/
|   `-- annotations.json
|-- Africa500/
|   |-- images/
|   `-- annotations.json
|-- Afri-MCQA/
`-- CulturalVQA/
```

The code can auto-download some HuggingFace sources when network and permissions allow, but local dataset preparation should still be treated as a required setup step.

### 5. Start with a small smoke test

Reduce the target dataset size in `configs/config.yaml` before the first run:

```yaml
dataset:
  target_size: 10
```

Then run:

```bash
python experiments/phase2_baseline.py --config configs/config.yaml
```

## Recommended Run Order

Use this order if starting from scratch:

1. Populate and validate the dataset
2. Run Phase 2 baseline
3. Run Phase 3A manual steering
4. Run Phase 3C probing
5. Run Phase 3B APO
6. Run Phase 4 adversarial
7. Run Phase 5 final analysis

This order keeps the science stable: baseline and dataset quality come before optimization.

## PSC Job Scripts

```bash
sbatch scripts/run_baseline.slurm
sbatch scripts/run_steering.slurm
sbatch scripts/run_apo.slurm
sbatch scripts/run_probing.slurm
sbatch scripts/run_adversarial.slurm
```

Check jobs with:

```bash
squeue -u $USER
tail -f logs/baseline_JOBID.out
```

## Core Metrics

| Metric | Formula | Purpose |
|---|---|---|
| Shape Accuracy | correct shape mentions / total | Measures functional identification |
| Texture Accuracy | correct texture or cultural mentions / total | Measures cultural recognition |
| Cue Accuracy | fraction with either cue recovered | Measures overall usable recognition |
| Shape Bias | shape accuracy / (shape accuracy + texture accuracy) | Measures which cue dominates |
| Cultural Recovery Rate | steered texture accuracy - neutral texture accuracy | Measures prompt benefit |

Interpretation:

- high shape bias means the model defaults toward generic object identity
- lower shape bias under valid steering can indicate recovered cultural recognition
- a drop in functional accuracy can indicate over-steering

## Prompt Families

| Prompt | Role |
|---|---|
| `neutral` | default description |
| `structural` | emphasizes shape and function |
| `cultural` | emphasizes materials, patterns, and cultural origin |
| `sequential_stage1` | stage 1 of Dual-Lens |
| `sequential_stage2` | stage 2 of Dual-Lens |
| `cultural_geometric` | cultural prompt focused on motifs and repeated structure |
| `cultural_expert` | expert-role cultural prompt |
| `adversarial_texture` | used for Western-shape/African-texture mixtures |

## Claims This Repo Can Support

With a sufficiently strong dataset and successful runs, this repo is well positioned to support claims like:

- neutral prompting suppresses African cultural detail
- targeted prompting can recover some of that detail
- the recovery/accuracy tradeoff can be quantified
- some cultural knowledge appears latent rather than completely absent

It is not, by itself, enough to prove broad cultural competence or fairness in deployment settings. Those stronger claims require stronger benchmarks, broader evaluation, and much tighter dataset controls.

## Dataset Build Plan

The dataset is the most important research dependency in this project. The code supports multiple sources, but they do not all play the same role.

### Core datasets for this repo

| Dataset | Role in project | Expected use |
|---|---|---|
| SURA Benchmark | Primary African cultural benchmark | Best source for culturally grounded evaluation if access is available |
| Africa-500 | Local or team-curated African image source | Good for filling category gaps and supporting curated experiments |
| Afri-MCQA | African cultural QA benchmark | Useful for expanding culturally grounded samples and metadata |
| CulturalVQA | Cross-cultural VQA benchmark | Useful for African subset extraction and comparison against broader cultural benchmarks |

### Supporting datasets

| Dataset | Role in project | Expected use |
|---|---|---|
| Afri-Aya | Additional African cultural content | Useful as a supplementary image and metadata source |
| SAFARI | African stereotype and cultural-context benchmark | Useful for broader contextual discussion, but not a direct cue-conflict source |

### Recommended usage by phase

| Phase | Dataset priority |
|---|---|
| Phase 2 baseline | SURA, Africa-500, curated CulturalVQA African subset |
| Phase 3 manual steering | Same evaluation set as Phase 2 |
| Phase 3B APO | Same fixed evaluation set to avoid moving the target |
| Phase 3C probing | Same fixed evaluation set, preferably with high-confidence labels |
| Phase 4 adversarial | Team-curated Western shape images plus African texture sources from SURA, Africa-500, or Afri-Aya |

### Minimum dataset structure per sample

Each usable sample should provide:

- image path
- shape label
- texture or cultural label
- category
- region
- source
- famous versus everyday tag when possible

### Suggested category coverage

To support the midterm claim about perceptual erasure beyond anecdotal examples, the dataset should aim to cover:

- textiles
- architecture
- everyday objects
- food and drink
- ritual items
- musical instruments

### Practical recommendation

If starting from scratch, the strongest path for this repo is:

1. Use SURA and Africa-500 as the highest-priority culturally grounded sources.
2. Use Afri-MCQA and CulturalVQA to expand coverage where the local datasets are thin.
3. Use cue-conflict synthesis only after the base labels are trustworthy.
4. Keep one fixed evaluation set for baseline, steering, APO, and probing so results stay comparable.

### Dataset Sources

| Dataset | Notes |
|---|---|
| SURA Benchmark | Primary African cultural evaluation source if access is available |
| Africa-500 | Local or manually curated source used in the project plan |
| Afri-MCQA | Multilingual African cultural QA benchmark |
| CulturalVQA | Culture-focused VQA benchmark with African subset potential |
| Afri-Aya | Additional culturally grounded source |
| SAFARI | Useful contextual benchmark, though not a direct cue-conflict source |

## References

1. Geirhos et al. (2019). ImageNet-trained CNNs are biased towards texture.
2. Gavrikov et al. (2025). Can We Talk Models Into Seeing the World Differently?
3. The project proposal and midterm report in `papers/`.
4. Related cultural-competence work including CulturalVQA and the MBZUAI NAACL 2025 framing around cross-cultural gaps and meta-cultural competence.

## Notes

- `data/`, `results/`, and `logs/` are intentionally gitignored because they can become large.
- The repo has been cleaned to use a single active probing path and a single active CLIP baseline path.
- The code is intended for research iteration, so careful result validation still matters more than automation alone.
