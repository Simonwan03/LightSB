# Compositional LightSB Guide

This document explains how to run `scripts/run_compositional_lightsb.py`,
including metadata preparation, metadata checks, bridge fitting, bridge
refitting, inference, latent-only runs, the one-step baseline, and the two
compositional modes:

- `global` / unconditional: each bridge is trained on broad source and target
  subsets for the attribute currently being edited.
- `local` / conditional: each bridge is trained on stricter source and target
  subsets that also condition on the other attributes that should stay fixed.

## Overview

Compositional LightSB decomposes a complex edit into several semantic bridges.
The default `configs/compositional_lightsb.yaml` experiment targets:

```text
male adult neutral
  -> female adult neutral
  -> female age 30-40 neutral
  -> female age 18-30 neutral
  -> female child neutral
  -> female child smiling
```

The pipeline fits one independent `src.light_sb.LightSB` object per step:

```text
z0 = E(x_source)
z1 = T_gender(z0)
z2 = T_age_adult_to_young_adult(z1)
z3 = T_age_young_adult_to_teen(z2)
z4 = T_age_teen_to_child(z3)
z5 = T_expression(z4)
x_out = D(z5)
```

Each bridge uses the original LightSB objective:

```python
loss = (-bridge.get_log_potential(target_batch) + bridge.get_log_C(source_batch)).mean()
```

## Files

- `scripts/run_compositional_lightsb.py`: main entry point. It fits or loads
  bridges, then runs inference unless `--fit-only` is used.
- `configs/compositional_lightsb.yaml`: default gender + age + expression
  multi-step edit config.
- `configs/compositional_lightsb_ffhq_gender_age.yaml`: simpler gender + age
  config.
- `src/compositional_lightsb.py`: bridge steps, caching, fitting, and
  composition logic.
- `src/compositional_metadata.py`: metadata loading, latent loading, and filter
  parsing.
- `src/compositional_visualization.py`: latent sequence, stage image, and panel
  output helpers.

## Data Preparation

The default config expects:

```text
data/latents.npy
data/ffhq_metadata.csv
```

`data/latents.npy` should be an ALAE latent matrix with shape
`(n_samples, 512)`. `data/ffhq_metadata.csv` must have the same row order as
`latents.npy`.

The recommended workflow is to build a unified metadata file from the DCGM FFHQ
feature JSON files:

```bash
git clone https://github.com/DCGM/ffhq-features-dataset data/ffhq-features-dataset

python scripts/build_ffhq_metadata.py \
  --config configs/ffhq_metadata_integration.yaml
```

This reads:

```text
data/latents.npy
data/gender.npy
data/age.npy
data/ffhq-features-dataset/json/00000.json
...
data/ffhq-features-dataset/json/69999.json
```

and writes:

```text
data/ffhq_metadata.csv
data/ffhq_metadata.pkl
data/ffhq_metadata_report.json
```

The JSON filename stem is treated as the FFHQ / latent row index. For example,
`00000.json` maps to `latents.npy[0]`.

If you do not have the DCGM JSON files, you can build a simpler metadata file
from existing arrays:

```bash
python scripts/prepare_ffhq_metadata.py \
  --latents data/latents.npy \
  --gender data/gender.npy \
  --age data/age.npy \
  --expression data/expression.npy \
  --output data/ffhq_attributes.csv
```

If you only need gender + age, use
`configs/compositional_lightsb_ffhq_gender_age.yaml` and point
`data.metadata_path` to either `data/ffhq_attributes.csv` or
`data/ffhq_metadata.csv`.

The metadata must include every column referenced by the config filters. The
default full config uses:

```text
gender: male / female
age_group: adult / child
age: numeric age
expression: neutral / smiling
image_path: optional, used for source image panels
latent_path: optional, only needed when data.latents_path is not used
```

Filters support case-insensitive exact string matching, list matching, and
numeric ranges:

```yaml
gender: male
expression: [neutral, smiling]
age:
  min: 18
  max: 30
```

Numeric ranges are left-closed and right-open: `min <= value < max`.

## Check Metadata

Before fitting bridges, inspect the attribute counts and key intersections:

```bash
python scripts/analyze_ffhq_metadata.py \
  --metadata data/ffhq_metadata.csv
```

If a source or target subset is empty, the training script exits early and
prints the failing filter. A common cause is a metadata value mismatch. For
example, `Female` and `female` match, but `woman` and `female` do not.

## Global / Unconditional

`global` mode trains broad, unconditional bridges. Each step only filters by
the attribute currently being edited:

```yaml
pipeline:
  subset_mode: global
```

`configs/compositional_lightsb.yaml` currently defaults to `global`:

```yaml
pipeline:
  subset_mode: global
  steps:
    - name: gender
      source_filter_global:
        gender: male
      target_filter_global:
        gender: female
```

This means the gender bridge learns:

```text
all male -> all female
```

The age bridges learn:

```text
all adult -> all age 30-40
all age 30-40 -> all age 18-30
all age 18-30 -> all child
```

The expression bridge learns:

```text
all neutral -> all smiling
```

Run the full global compositional pipeline:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional
```

Fit or load the global bridges only, without inference:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --fit-only
```

Force refitting of the global bridges and ignore cached bridges:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --force-refit
```

Force refitting of the global bridges only, without inference or decoding:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --force-refit \
  --fit-only
```

Global bridge caches are written to:

```text
outputs/compositional_lightsb/bridges/global/<step>.pt
```

## Local / Conditional

`local` mode trains conditional bridges. Each step filters by the edited
attribute and also fixes other attributes that should be preserved:

```yaml
pipeline:
  subset_mode: local
```

For example, the default config's local gender step is:

```yaml
- name: gender
  source_filter_local:
    gender: male
    age_group: adult
    expression: neutral
  target_filter_local:
    gender: female
    age_group: adult
    expression: neutral
```

This means the gender bridge learns:

```text
male adult neutral -> female adult neutral
```

The later age and expression bridges are also trained under more specific
conditions:

```text
female adult neutral -> female age 30-40 neutral
female age 30-40 neutral -> female age 18-30 neutral
female age 18-30 neutral -> female child neutral
female child neutral -> female child smiling
```

The script does not provide a CLI flag to override `subset_mode`, so the
recommended workflow is to copy the config:

```bash
cp configs/compositional_lightsb.yaml configs/compositional_lightsb_local.yaml
```

Then edit `configs/compositional_lightsb_local.yaml`:

```yaml
output_dir: outputs/compositional_lightsb_local

pipeline:
  subset_mode: local
```

Run the full local / conditional pipeline:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb_local.yaml \
  --mode compositional
```

Fit or load the local bridges only:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb_local.yaml \
  --mode compositional \
  --fit-only
```

Force refitting of the local bridges:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb_local.yaml \
  --mode compositional \
  --force-refit
```

Force refitting of the local bridges only, without inference or decoding:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb_local.yaml \
  --mode compositional \
  --force-refit \
  --fit-only
```

With the recommended local config, local bridge caches are written to:

```text
outputs/compositional_lightsb_local/bridges/local/<step>.pt
```

If you do not set a separate `output_dir`, the local and global
`runs/compositional` outputs and `compositional_summary.json` will write to the
same directory and can overwrite each other. Use separate output directories
when comparing the two modes.

## Run Partial Compositions

Apply only the first `N` compositional steps:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --num-steps 2
```

`--num-steps 2` applies the first two bridges and saves:

```text
z0
z1 after gender
z2 after age_adult_to_young_adult
```

You can also set `inference.num_steps` in the YAML. The CLI argument
`--num-steps` takes precedence.

## One-Step Baseline

The one-step baseline uses `baselines.one_step.source_filter` and
`baselines.one_step.target_filter` to fit one direct bridge instead of a
sequence of bridges.

The default full config defines:

```yaml
baselines:
  one_step:
    name: one_step_all_attributes
    source_filter:
      gender: male
      age_group: adult
      expression: neutral
    target_filter:
      gender: female
      age_group: child
      expression: smiling
```

Run the one-step baseline:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode one_step
```

Force refitting of the one-step bridge:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode one_step \
  --force-refit
```

Fit only the one-step bridge:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode one_step \
  --force-refit \
  --fit-only
```

The one-step cache is written to:

```text
outputs/compositional_lightsb/bridges/one_step/<name>.pt
```

## Gender + Age Experiment

For the original FFHQ ALAE gender + age experiment, use the simpler config:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb_ffhq_gender_age.yaml \
  --mode compositional
```

This config has two bridges:

```text
gender: male -> female
age: adult -> child
```

Running the first two steps is equivalent to the full run because the config
only contains two steps:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb_ffhq_gender_age.yaml \
  --mode compositional \
  --num-steps 2
```

Run the gender + age one-step baseline:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb_ffhq_gender_age.yaml \
  --mode one_step
```

## Select Inference Inputs

By default, the script randomly samples source latents that match
`inference.source_filter`:

```yaml
inference:
  source_filter:
    gender: male
    age_group: adult
    expression: neutral
  max_inputs: 8
```

To force a fixed set of metadata row indices, set:

```yaml
inference:
  input_indices: [0, 10, 25]
```

Or read indices from a file:

```yaml
inference:
  input_indices_path: data/my_input_indices.npy
```

Plain text files are also supported, with one integer index per line:

```yaml
inference:
  input_indices_path: data/my_input_indices.txt
```

By default, `validate_source_filter: true`, so the selected indices must match
`inference.source_filter`. To skip this check:

```yaml
inference:
  validate_source_filter: false
```

To provide existing latents directly:

```yaml
inference:
  input_latents_path: data/my_latents.npy
```

For `.npz` inputs, specify the array key if needed:

```yaml
inference:
  input_latents_path: data/my_latents.npz
  input_latent_key: latents
```

To start from raw images, set:

```yaml
inference:
  input_image_paths:
    - data/my_images/source_01.png
    - data/my_images/source_02.png
  input_image_size: 1024
```

This loads the ALAE encoder, encodes the images to 512D latents, and applies
the same bridge sequence.

## Decoding and Latent-Only Runs

The default config enables decoding:

```yaml
inference:
  decode:
    enabled: true
```

If the ALAE checkpoint is available, the script saves decoded images for each
stage and a panel image.

Skip ALAE decoding and save only the latent trajectory:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --no-decode
```

If the ALAE checkpoint is missing but you only need bridge fitting or latent
trajectories, use `--fit-only` or `--no-decode`.

Download ALAE checkpoints:

```bash
cd ALAE
python training_artifacts/download_all.py
```

## Outputs

The default full config writes to:

```text
outputs/compositional_lightsb
```

Common outputs:

```text
bridges/<subset_mode>/<step>.pt
runs/<mode>/latent_sequence.npz
runs/<mode>/source_indices.npy
runs/<mode>/selected_metadata.csv
runs/<mode>/stages/<stage_name>/*.png
runs/<mode>/source_images/*
runs/<mode>/panel.png
<mode>_summary.json
```

Output meanings:

- `bridges/<subset_mode>/<step>.pt`: cached LightSB bridge parameters.
- `latent_sequence.npz`: intermediate latents `z0`, `z1`, ...
- `selected_metadata.csv`: metadata rows for the selected source samples.
- `stages/`: decoded image for each stage.
- `panel.png`: source and stage image grid.
- `<mode>_summary.json`: filters, cache paths, and training metrics.

## Command Reference

Default global / unconditional full run:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional
```

Refit default global bridges:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --force-refit
```

Refit only default global bridges:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --force-refit \
  --fit-only
```

Local / conditional full run:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb_local.yaml \
  --mode compositional
```

Latent-only run without decoding:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --no-decode
```

Run only the first two compositional steps:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --num-steps 2
```

One-step baseline:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode one_step
```

Metadata check:

```bash
python scripts/analyze_ffhq_metadata.py \
  --metadata data/ffhq_metadata.csv
```

## FAQ

### How do I recompute bridges?

Use `--force-refit`. For the default global config:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --force-refit
```

If you only want to refit bridges and skip inference:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --force-refit \
  --fit-only
```

### Why did changing a filter not use the old cache?

The script stores each bridge's source and target filters inside the cache. If
the cached filters differ from the current YAML filters, the bridge is refit
automatically. If the filters are the same but you still want to retrain, use
`--force-refit`.

### What if local mode has too few samples?

Local / conditional filters are stricter, so a source or target subset can be
small or empty. First inspect metadata:

```bash
python scripts/analyze_ffhq_metadata.py \
  --metadata data/ffhq_metadata.csv
```

Then consider:

- Relaxing `source_filter_local` / `target_filter_local`.
- Using `global` mode.
- Lowering `bridge.n_potentials`. The target subset size must be at least
  `n_potentials`.

### Where are caches stored?

Global compositional:

```text
outputs/compositional_lightsb/bridges/global/
```

Local compositional with the recommended local config:

```text
outputs/compositional_lightsb_local/bridges/local/
```

One-step baseline:

```text
outputs/compositional_lightsb/bridges/one_step/
```

### Can I delete caches manually?

Yes, but normally you do not need to. Prefer `--force-refit` so other run
outputs are not accidentally removed.
