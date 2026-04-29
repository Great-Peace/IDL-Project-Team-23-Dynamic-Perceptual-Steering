# Dataset Card

## Purpose

This project studies perceptual erasure in vision-language models on African cultural artifacts. The dataset is intended to support cue-conflict evaluation, prompt steering, and confidence probing.

The goal is not just to classify objects, but to measure when a model:

- recovers only generic function or shape
- suppresses local cultural texture or material identity
- or can be steered into recovering that cultural information

## Recommended Dataset Composition

The working target in this repo is a balanced evaluation set of roughly 400 to 500 images, with enough coverage to support per-category and per-region breakdowns.

Recommended categories:

- textiles
- architecture
- everyday objects
- food and drink
- ritual items
- musical instruments

Recommended regional coverage:

- West Africa
- East Africa
- Southern Africa
- Central Africa
- North Africa

## Priority Data Sources

### 1. SURA Benchmark

Best candidate for a culturally grounded primary evaluation set if access is available.

Use it for:

- core evaluation
- famous versus everyday analysis
- region-aware artifact coverage

### 2. Africa-500

Useful as a local or curated image source when you need to fill coverage gaps.

Use it for:

- category balancing
- additional curated artifacts
- team-reviewed examples

### 3. Afri-MCQA

Useful for expanding the pool of African cultural examples and associated metadata.

Use it for:

- culturally grounded sample discovery
- additional labels and QA context
- expansion beyond manually collected samples

### 4. CulturalVQA

Useful for African subset extraction and for contextual comparison against broader cultural benchmarks.

Use it for:

- African subset extraction
- broader cross-cultural positioning
- secondary evaluation and sanity checks

## Supporting Sources

### Afri-Aya

Can provide extra culturally relevant content when the main benchmark sources are thin.

### SAFARI

More useful for contextual discussion of African bias and representation than for direct cue-conflict evaluation.

## Unit of Annotation

Each usable sample should include:

- `image_path`
- `shape_label`
- `texture_label`
- `category`
- `region`
- `source`
- `is_famous`

Optional but useful metadata:

- local object name
- country
- materials
- style keywords
- annotation confidence
- notes about ambiguity

## Labeling Guidance

### Shape label

The shape label should describe the generic functional identity of the object.

Examples:

- bowl
- bag
- cloth
- building
- basket
- garment
- mask

### Texture or cultural label

The texture label should capture the culturally meaningful visual identity.

Examples:

- kente
- mudcloth
- adire
- sudano-sahelian
- zulu beadwork

These labels should be visually grounded whenever possible, not just historically associated.

## Inclusion Criteria

Prefer samples where:

- the object is visually clear
- the cultural texture or material identity is visible
- the shape label and texture label are both meaningful
- the artifact supports one or more of the project hypotheses

## Exclusion Criteria

Exclude samples where:

- the image is too low quality to support visual reasoning
- the cultural signal is only inferable from text context rather than the image
- labels are too ambiguous to support reliable evaluation
- the same artifact appears repeatedly with little variation

## Famous Versus Everyday Split

This repo explicitly tracks the difference between:

- famous or globally recognizable landmarks or artifacts
- everyday, locally meaningful artifacts that are more likely to be erased

Examples:

- famous: Great Mosque of Djenne
- everyday: mudcloth bag, woven basket, clay pot

This split matters because one of the project’s strongest hypotheses is that training data preserves globally famous African visuals while flattening or suppressing everyday African material culture.

## Cue-Conflict Construction

After the base dataset is trustworthy, cue-conflict images can be synthesized by combining:

- a content image that supplies shape
- a style image that supplies African texture or material cues

Use synthesized pairs only after:

- the source labels are reviewed
- the resulting image remains interpretable
- the shape and texture cues are both still visible

## Suggested Workflow

1. Collect candidate images from SURA and Africa-500.
2. Expand with Afri-MCQA and CulturalVQA where needed.
3. Annotate or normalize labels into the repo schema.
4. Review image quality and ambiguity.
5. Build a fixed evaluation set.
6. Only then create cue-conflict images and adversarial mixtures.

## Risks

The largest risks are:

- weak or noisy labels
- over-reliance on QA datasets whose answers are not always visually grounded
- category imbalance
- too few everyday artifacts
- mixing evaluation data and prompt-development data too loosely

## What This Dataset Can Support

With careful curation, the dataset can support:

- neutral baseline evaluation
- manual and automated prompt steering
- probing for shape versus texture confidence
- famous versus everyday analysis
- adversarial texture steering

It does not, by itself, establish general cultural competence across all African contexts.
