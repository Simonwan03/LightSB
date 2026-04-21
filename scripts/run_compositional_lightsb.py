import argparse
import csv
import contextlib
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.compositional_lightsb import (  # noqa: E402
    CompositionalLightSB,
    build_steps_from_config,
    fit_bridges,
    fit_one_step_bridge_from_filters,
)
from src.compositional_metadata import (  # noqa: E402
    LatentMetadataDataset,
    filter_metadata_indices,
    load_latent_metadata_dataset,
)
from src.compositional_visualization import (  # noqa: E402
    copy_source_images,
    decode_latent_sequence,
    save_latent_sequence,
    save_panel,
    save_stage_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compositional LightSB in ALAE latent space.")
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join("configs", "compositional_lightsb.yaml"),
        help="Path to compositional LightSB YAML config.",
    )
    parser.add_argument(
        "--mode",
        choices=("compositional", "one_step"),
        default="compositional",
        help="Run sequential composition or the configured one-step baseline.",
    )
    parser.add_argument("--num-steps", type=int, default=None, help="Use only the first N compositional steps.")
    parser.add_argument("--fit-only", action="store_true", help="Fit/load bridges and exit before inference.")
    parser.add_argument("--force-refit", action="store_true", help="Ignore cached bridges and retrain.")
    parser.add_argument("--device", default=None, help="Override config device.")
    parser.add_argument("--no-decode", action="store_true", help="Skip ALAE decoding even if enabled in config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(resolve_path(args.config))
    seed = int(config.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = args.device or config.get("device", "cpu")
    output_dir = resolve_path(config.get("output_dir", "outputs/compositional_lightsb"))
    os.makedirs(output_dir, exist_ok=True)

    data_config = dict(config["data"])
    data_config["root"] = resolve_path(data_config.get("root", "."))
    dataset = load_latent_metadata_dataset(data_config)
    bridge_config = dict(config.get("bridge", {}))
    force_refit = args.force_refit or bool(bridge_config.get("force_refit", False))

    if args.mode == "one_step":
        steps = build_one_step_baseline(config, dataset, bridge_config, output_dir, device, force_refit)
        num_steps = 1
    else:
        subset_mode = config.get("pipeline", {}).get("subset_mode", "local")
        raw_steps = config["pipeline"]["steps"]
        steps = build_steps_from_config(raw_steps, subset_mode=subset_mode)
        cache_dir = os.path.join(output_dir, "bridges", subset_mode)
        steps = fit_bridges(
            steps,
            dataset,
            bridge_config,
            cache_dir=cache_dir,
            device=device,
            force_refit=force_refit,
        )
        num_steps = args.num_steps or config.get("inference", {}).get("num_steps")

    write_run_summary(output_dir, args.mode, steps)
    if args.fit_only:
        print("Fit-only mode complete.")
        return

    latents, source_indices, source_image_paths = select_input_latents(config, dataset, device)
    z0 = torch.tensor(latents, dtype=torch.float32, device=device)
    pipeline = CompositionalLightSB(steps, device=device)
    latent_sequence = pipeline.transform_with_intermediates(z0, num_steps=num_steps)

    run_dir = os.path.join(output_dir, "runs", args.mode)
    os.makedirs(run_dir, exist_ok=True)
    if config.get("inference", {}).get("save_intermediate_latents", True):
        save_latent_sequence(os.path.join(run_dir, "latent_sequence.npz"), latent_sequence)
        np.save(os.path.join(run_dir, "source_indices.npy"), source_indices)
    save_selected_metadata(os.path.join(run_dir, "selected_metadata.csv"), dataset, source_indices)
    print_selected_metadata_summary(dataset, source_indices)

    decode_config = config.get("inference", {}).get("decode", {})
    if decode_config.get("enabled", False) and not args.no_decode:
        decoded = decode_with_alae(decode_config, latent_sequence, device)
        stage_names = make_stage_names(steps, len(decoded), num_steps)
        if decode_config.get("save_stage_images", True):
            save_stage_images(os.path.join(run_dir, "stages"), decoded, stage_names=stage_names)
        if decode_config.get("include_source_images", True):
            copy_source_images(os.path.join(run_dir, "source_images"), source_image_paths)
        if decode_config.get("save_panel", True):
            save_panel(
                os.path.join(run_dir, "panel.png"),
                decoded,
                source_image_paths=source_image_paths if decode_config.get("include_source_images", True) else None,
                stage_labels=stage_names,
            )

    print(f"Saved compositional LightSB outputs to {run_dir}")


def build_one_step_baseline(
    config: Dict[str, Any],
    dataset: LatentMetadataDataset,
    bridge_config: Dict[str, Any],
    output_dir: str,
    device: str,
    force_refit: bool,
):
    baseline = config.get("baselines", {}).get("one_step")
    if not baseline:
        raise ValueError("Config does not define baselines.one_step.")
    step = fit_one_step_bridge_from_filters(
        name=baseline.get("name", "one_step"),
        source_filter=baseline.get("source_filter", {}),
        target_filter=baseline.get("target_filter", {}),
        dataset=dataset,
        bridge_config=bridge_config,
        cache_dir=os.path.join(output_dir, "bridges", "one_step"),
        device=device,
        force_refit=force_refit,
    )
    return [step]


def select_input_latents(
    config: Dict[str, Any],
    dataset: LatentMetadataDataset,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[Optional[str]]]:
    inference = config.get("inference", {})
    if inference.get("input_image_paths"):
        image_paths = [resolve_path(path) for path in inference["input_image_paths"]]
        decode_config = inference.get("decode", {})
        latents = encode_images_with_alae(
            image_paths,
            decode_config,
            device=device,
            image_size=int(inference.get("input_image_size", 1024)),
        )
        indices = np.arange(latents.shape[0], dtype=np.int64)
        return latents, indices, image_paths

    if inference.get("input_latents_path"):
        path = resolve_path(inference["input_latents_path"])
        latents = np.load(path)
        if isinstance(latents, np.lib.npyio.NpzFile):
            key = inference.get("input_latent_key") or latents.files[0]
            latents = latents[key]
        latents = np.asarray(latents, dtype=np.float32)
        indices = np.arange(latents.shape[0], dtype=np.int64)
        return latents, indices, [None] * latents.shape[0]

    if inference.get("input_indices") is not None or inference.get("input_indices_path"):
        indices = load_input_indices(inference)
        if inference.get("validate_source_filter", True):
            validate_indices_match_filter(indices, dataset, inference.get("source_filter"))
    else:
        source_filter = inference.get("source_filter")
        indices = filter_metadata_indices(dataset.metadata, source_filter)
        if len(indices) == 0:
            raise ValueError(
                f"No inference inputs match source_filter={dict(source_filter or {})}."
            )
        max_inputs = int(inference.get("max_inputs", 8))
        if len(indices) > max_inputs:
            indices = np.random.choice(indices, size=max_inputs, replace=False)

    latents = dataset.latents[indices]
    return latents, indices, dataset.image_paths(indices)


def load_input_indices(inference: Dict[str, Any]) -> np.ndarray:
    if inference.get("input_indices") is not None:
        return np.asarray(inference["input_indices"], dtype=np.int64)

    path = resolve_path(inference["input_indices_path"])
    if path.endswith(".npy"):
        return np.load(path).astype(np.int64).reshape(-1)
    with open(path) as handle:
        return np.asarray([int(line.strip()) for line in handle if line.strip()], dtype=np.int64)


def validate_indices_match_filter(
    indices: np.ndarray,
    dataset: LatentMetadataDataset,
    source_filter: Optional[Dict[str, Any]],
) -> None:
    if not source_filter:
        return
    allowed = set(filter_metadata_indices(dataset.metadata, source_filter).tolist())
    bad = [int(index) for index in indices if int(index) not in allowed]
    if bad:
        examples = bad[:10]
        raise ValueError(
            "Some inference input indices do not match source_filter="
            f"{dict(source_filter)}. Bad examples: {examples}"
        )


def save_selected_metadata(
    path: str,
    dataset: LatentMetadataDataset,
    indices: np.ndarray,
) -> None:
    if len(indices) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [dict(dataset.metadata[int(index)], selected_index=int(index)) for index in indices]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_selected_metadata_summary(
    dataset: LatentMetadataDataset,
    indices: np.ndarray,
) -> None:
    if len(indices) == 0:
        return
    rows = [dataset.metadata[int(index)] for index in indices]
    for column in ("gender", "age_group", "expression", "glasses"):
        if column in rows[0]:
            counts = Counter(row.get(column, "") for row in rows)
            print(f"selected z0 {column}: {dict(counts)}")


def decode_with_alae(
    decode_config: Dict[str, Any],
    latent_sequence: Sequence[torch.Tensor],
    device: str,
):
    alae_config = resolve_path(decode_config.get("alae_config", "ALAE/configs/ffhq.yaml"))
    artifacts_dir = decode_config.get("training_artifacts_dir", "ALAE/training_artifacts/ffhq")
    with alae_working_directory():
        from alae_ffhq_inference import decode, load_model

        model = load_model(
            alae_config,
            training_artifacts_dir=path_for_alae_checkpointer(artifacts_dir),
        )
    return decode_latent_sequence(model, latent_sequence, decode, device=device)


def encode_images_with_alae(
    image_paths: Sequence[str],
    decode_config: Dict[str, Any],
    device: str,
    image_size: int = 1024,
) -> np.ndarray:
    from PIL import Image

    alae_config = resolve_path(decode_config.get("alae_config", "ALAE/configs/ffhq.yaml"))
    artifacts_dir = decode_config.get("training_artifacts_dir", "ALAE/training_artifacts/ffhq")
    with alae_working_directory():
        from alae_ffhq_inference import encode, load_model

        model = load_model(
            alae_config,
            training_artifacts_dir=path_for_alae_checkpointer(artifacts_dir),
        )
    model.to(device)
    model.eval()

    images = []
    for path in image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input image not found: {path}")
        image = Image.open(path).convert("RGB").resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        images.append(torch.tensor(array / 127.5 - 1.0, dtype=torch.float32))

    batch = torch.stack(images, dim=0).to(device)
    with torch.no_grad():
        encoded = encode(model, batch)
    if encoded.ndim == 3:
        encoded = encoded[:, 0, :]
    return encoded.detach().cpu().numpy().astype(np.float32)


@contextlib.contextmanager
def alae_working_directory():
    alae_root = os.path.join(REPO_ROOT, "ALAE")
    if alae_root not in sys.path:
        sys.path.insert(0, alae_root)
    old_cwd = os.getcwd()
    os.chdir(alae_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def path_for_alae_checkpointer(path: str) -> str:
    path = resolve_path(path)
    alae_root = os.path.join(REPO_ROOT, "ALAE")
    try:
        return os.path.relpath(path, alae_root)
    except ValueError:
        return path


def make_stage_names(steps, decoded_count: int, num_steps: Optional[int]) -> List[str]:
    active_steps = steps if num_steps is None else steps[:num_steps]
    names = ["z0"] + [step.name for step in active_steps]
    if len(names) < decoded_count:
        names.extend([f"stage_{i}" for i in range(len(names), decoded_count)])
    return names[:decoded_count]


def write_run_summary(output_dir: str, mode: str, steps) -> None:
    payload = {
        "mode": mode,
        "steps": [
            {
                "name": step.name,
                "source_filter": step.source_filter,
                "target_filter": step.target_filter,
                "cache_path": step.cache_path,
                "metrics": step.metrics,
            }
            for step in steps
        ],
    }
    with open(os.path.join(output_dir, f"{mode}_summary.json"), "w") as handle:
        json.dump(payload, handle, indent=2)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path) as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(REPO_ROOT, path)


if __name__ == "__main__":
    main()
