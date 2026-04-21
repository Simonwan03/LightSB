# Compositional LightSB in ALAE Latent Space

This experimental pipeline adds sequential attribute translation on top of the
original one-step LightSB implementation. It keeps the original GMM-based
`LightSB` class unchanged and fits one independent LightSB bridge per semantic
edit.

Example composition:

```text
z0 = E(x_source)
z1 = T_gender(z0)
z2 = T_age(z1)
z3 = T_glasses(z2)
x_out = D(z3)
```

Each `T_*` is a standard `src.light_sb.LightSB` object trained in ALAE latent
space with the same objective used in the notebooks:

```python
loss = (-bridge.get_log_potential(target_batch) + bridge.get_log_C(source_batch)).mean()
```

## Files

- `configs/compositional_lightsb.yaml`: example config for global and local
  multi-step bridges.
- `src/compositional_metadata.py`: metadata loading, latent loading, and
  attribute filtering.
- `src/compositional_lightsb.py`: `AttributeBridgeStep`,
  `CompositionalLightSB`, bridge fitting, caching, and sequential transforms.
- `src/compositional_visualization.py`: saves intermediate latents, decoded
  stage images, and panel grids.
- `scripts/run_compositional_lightsb.py`: runnable CLI for fitting and
  inference.

## Metadata Format

Use either CSV or JSON. The recommended current path is to build a unified CSV
from the DCGM FFHQ feature JSON files:

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

The JSON file stem is treated as the FFHQ / latent row index: `00000.json`
maps to `latents.npy[0]`, `00001.json` maps to `latents.npy[1]`, and so on.

The metadata CSV can also be created manually. A minimal CSV can look like:

```csv
image_path,latent_path,gender,age_group,glasses
images/000001.png,latents/000001.npy,male,adult,no_glasses
images/000002.png,latents/000002.npy,female,child,reading_glasses
```

If all latents are stored in one matrix, set `data.latents_path` in the config.
The row order must match the metadata row order, and `latent_path` may be left
empty. If `data.latents_path` is null, every metadata row must provide a
`latent_path` pointing to one latent vector.

For the original FFHQ ALAE arrays used by `notebooks/LightSB_alae.ipynb`, first
download or place these files in `data/`:

```text
data/latents.npy
data/gender.npy
data/age.npy
```

If you do not have the DCGM JSON files yet, you can still build the simpler
gender+age metadata CSV:

```bash
python scripts/prepare_ffhq_metadata.py \
  --latents data/latents.npy \
  --gender data/gender.npy \
  --age data/age.npy \
  --output data/ffhq_attributes.csv
```

Without glasses labels, use `configs/compositional_lightsb_ffhq_gender_age.yaml`
and set its `data.metadata_path` back to `data/ffhq_attributes.csv`. For
glasses-aware experiments, prefer `scripts/build_ffhq_metadata.py`, which reads
the DCGM JSON `faceAttributes.glasses` field and normalizes values such as
`NoGlasses`, `ReadingGlasses`, and `Sunglasses`.

Expected attribute values in the example config:

- `gender`: `male` / `female`
- `age_group`: `adult` / `child`
- `glasses`: `yes` / `no`

The filtering code uses case-insensitive exact matching.

## Global vs Local Bridges

The config supports two subset modes:

- `global`: train broad bridges such as `gender=male -> gender=female`.
- `local`: train stricter bridges such as
  `male adult no_glasses -> female adult no_glasses`.

Choose with:

```yaml
pipeline:
  subset_mode: local
```

Each step can define both `source_filter_global` / `target_filter_global` and
`source_filter_local` / `target_filter_local`.

## Filtering API

The reusable filtering helpers live in `src/ffhq_metadata.py`:

```python
from src.ffhq_metadata import (
    build_source_target_subsets,
    read_metadata_table,
    select_subset,
)

df = read_metadata_table("data/ffhq_metadata.csv")
male_adults = select_subset(df, gender="male", age_group="adult")
source, target = build_source_target_subsets(
    df,
    source_filter={"gender": "male", "age_group": "adult", "glasses": "no_glasses"},
    target_filter={"gender": "female", "age_group": "adult", "glasses": "no_glasses"},
)
```

Constraint values may be exact strings, lists of accepted strings, or numeric
ranges such as `{"min": 18, "max": 30}`.

## Running

From the repository root:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode compositional
```

For the original FFHQ gender + age arrays, run:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb_ffhq_gender_age.yaml \
  --num-steps 2
```

Print metadata statistics before fitting bridges:

```bash
python scripts/analyze_ffhq_metadata.py \
  --metadata data/ffhq_metadata.csv
```

This reports counts by gender, age group, glasses category, and important local
bridge intersections such as `male adult no_glasses`.

Fit only, without inference:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --fit-only
```

Run only the first two steps:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --num-steps 2
```

Run the one-step baseline defined in `baselines.one_step`:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --mode one_step
```

Skip ALAE decoding and save only latent trajectories:

```bash
python scripts/run_compositional_lightsb.py \
  --config configs/compositional_lightsb.yaml \
  --no-decode
```

To start from raw source images instead of metadata-selected or precomputed
latents, set `inference.input_image_paths` in the config. The script will load
the ALAE encoder, resize images to `inference.input_image_size`, encode them to
512D latents, and then apply the same bridge sequence.

To force a specific, manually checked set of source latents, set either
`inference.input_indices` in the YAML or `inference.input_indices_path` to a
`.npy` / text file of integer indices. By default these indices are validated
against `inference.source_filter`, so a gender+age run configured with
`gender: male` will fail early if any selected `z0` index is not labeled male.

## Outputs

By default outputs are written to `outputs/compositional_lightsb`:

- `bridges/<subset_mode>/<step>.pt`: cached LightSB bridge parameters.
- `runs/<mode>/latent_sequence.npz`: `z0`, `z1`, ... intermediate latents.
- `runs/<mode>/selected_metadata.csv`: metadata rows for the selected `z0`
  latents, including the original row index.
- `runs/<mode>/stages/<stage_name>/*.png`: decoded stage images.
- `runs/<mode>/panel.png`: side-by-side panel for source and stages.
- `<mode>_summary.json`: filters, cache paths, and training metrics.

## ALAE Caveats

Decoding requires downloaded ALAE checkpoints. The original repository expects
these files under `ALAE/training_artifacts/<dataset>` and points
`last_checkpoint` to a `.pth` file. If those files are missing, run:

```bash
cd ALAE
python training_artifacts/download_all.py
```

The pipeline can still train and save latent trajectories without ALAE weights
by using `--no-decode`.

## Notes

- This feature is additive. It does not modify `src/light_sb.py` or the original
  notebooks.
- The bridge cache stores trained `LightSB` state dicts, not a separate neural
  residual model.
- If an attribute subset is empty, the script raises a clear error showing the
  failed filter.
