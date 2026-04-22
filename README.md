# Light Schrödinger Bridge

This repository contains the `PyTorch` code to reproduce the experiments from work **Light Schrödinger Bridge** (LightSB paper on [arxiv](https://arxiv.org/abs/2310.01174) and [OpenReview](https://openreview.net/forum?id=WhZoCLRWYJ)) by  [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ&hl=en), [Nikita Gushchin](https://scholar.google.com/citations?user=UaRTbNoAAAAJ&hl=en&oi=ao) and [Evgeny Burnaev](https://scholar.google.ru/citations?user=pCRdcOwAAAAJ&hl=ru).

**An example:** Unpaired *Male* -> *Female* translation by our LightSB solver applied in the latent space of ALAE for 1024x1024 FFHQ images. *Our LightSB converges on 4 cpu cores in less than 1 minute.*

<p align="center"><img src="teaser/teaser.png" width="800" /></p>

## Repository structure:
All the experiments are issued in the form of pretty self-explanatory jupyter notebooks (`notebooks/`). Auxiliary source code is moved to `.py` modules (`src/`). 

Note that we use `wandb` ([link](https://wandb.ai/site)) dashboard system when launching our experiments. The practitioners are expected to use `wandb` too. 

```notebooks/Toy_experiments.ipynb``` - Toy experiments.

```ALAE``` - Code for the ALAE model.

```src``` - LightSB implementation and auxiliary code for plotting.

```notebooks/LightSB_swiss_roll.ipynb``` - Code for swiss roll experiments.

```notebooks/swiss_roll_plot.ipynb``` - Code for plotting the reported image for swiss roll.

```notebooks/LightSB_EOT_benchmark.ipynb``` - Code for benchmark experiments.

```notebooks/LightSB_single_cell.ipynb``` - Code for single cell experiments.

```notebooks/LightSB_alae.ipynb``` - Code for image experiments with ALAE.

```scripts/run_compositional_lightsb.py``` - Experimental multi-step compositional LightSB pipeline in ALAE latent space.

```scripts/build_ffhq_metadata.py``` - Builds a richer FFHQ metadata table from
local ALAE latents, existing age/gender arrays, and the DCGM FFHQ feature JSON
files.

```docs/compositional_lightsb.md``` - Full guide for compositional LightSB metadata preparation, config options, bridge caching, inference, and baselines.

## Compositional LightSB

The repository includes an experimental compositional extension for applying several LightSB bridges sequentially in ALAE latent space. Instead of fitting one bridge for a complex edit, the pipeline decomposes the edit into semantic steps, for example:

```text
male adult neutral
  -> female adult neutral
  -> female age 30-40 neutral
  -> female age 18-30 neutral
  -> female child neutral
  -> female child smiling
```

The main entry point is:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional
```

The default config expects:

```text
data/latents.npy
data/ffhq_metadata.csv
```

`data/latents.npy` should contain ALAE latents with shape `(n_samples, 512)`, and the metadata rows must be in the same order as the latent rows. To build the richer FFHQ metadata used by the default gender + age + expression config, place the DCGM FFHQ feature JSON files under `data/ffhq-features-dataset/json/` and run:

```bash
python scripts/build_ffhq_metadata.py \
  --config configs/ffhq_metadata_integration.yaml
```

If only age and gender arrays are available, build simpler metadata and use the gender + age config:

```bash
python scripts/prepare_ffhq_metadata.py \
  --latents data/latents.npy \
  --gender data/gender.npy \
  --age data/age.npy \
  --output data/ffhq_metadata.csv

python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb_ffhq_gender_age.yaml \
  --mode compositional
```

Useful run modes:

```bash
# Fit or load bridges, then skip ALAE decoding.
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --no-decode

# Refit cached bridges.
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --force-refit

# Apply only the first N compositional steps during inference.
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional \
  --num-steps 2

# Train the one-step baseline from the same config.
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode one_step
```

The config supports `pipeline.subset_mode: global` for broad unconditional bridges and `pipeline.subset_mode: local` for stricter conditional bridges that also fix attributes intended to remain unchanged. Outputs are written under `outputs/compositional_lightsb/` by default, including cached bridges, intermediate latent sequences, selected metadata, decoded stage images, and `panel.png` when ALAE decoding is enabled. See `docs/compositional_lightsb.md` for the complete metadata format, filter syntax, local-mode setup, and troubleshooting notes.

## Citation
```
@inproceedings{
korotin2024light,
title={Light Schr\"odinger Bridge},
author={Alexander Korotin and Nikita Gushchin and Evgeny Burnaev},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=WhZoCLRWYJ}
}
```
