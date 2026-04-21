import os
import shutil
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image


def save_latent_sequence(path: str, latent_sequence: Sequence[torch.Tensor]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {f"z{i}": latent.detach().cpu().numpy() for i, latent in enumerate(latent_sequence)}
    np.savez(path, **payload)


def decode_latent_sequence(
    alae_model,
    latent_sequence: Sequence[torch.Tensor],
    decode_fn,
    device: str = "cpu",
) -> List[np.ndarray]:
    decoded = []
    alae_model.to(device)
    alae_model.eval()
    with torch.no_grad():
        for latents in latent_sequence:
            images = decode_fn(alae_model, latents.to(device).float())
            decoded.append(tensor_to_uint8_images(images))
    return decoded


def tensor_to_uint8_images(images: torch.Tensor) -> np.ndarray:
    images = ((images.detach().cpu() * 0.5 + 0.5) * 255.0)
    images = images.clamp(0, 255).to(torch.uint8)
    return images.permute(0, 2, 3, 1).numpy()


def save_stage_images(
    output_dir: str,
    decoded_stages: Sequence[np.ndarray],
    stage_names: Optional[Sequence[str]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    names = list(stage_names or [f"stage_{i}" for i in range(len(decoded_stages))])
    for stage_index, images in enumerate(decoded_stages):
        stage_dir = os.path.join(output_dir, names[stage_index])
        os.makedirs(stage_dir, exist_ok=True)
        for image_index, image in enumerate(images):
            Image.fromarray(image).save(os.path.join(stage_dir, f"{image_index:04d}.png"))


def save_panel(
    path: str,
    decoded_stages: Sequence[np.ndarray],
    source_image_paths: Optional[Sequence[Optional[str]]] = None,
    stage_labels: Optional[Sequence[str]] = None,
) -> None:
    if not decoded_stages:
        raise ValueError("decoded_stages cannot be empty.")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    columns = [stage for stage in decoded_stages]
    labels = list(stage_labels or [f"z{i}" for i in range(len(columns))])
    if source_image_paths:
        source_images = [_load_image_or_blank(p, decoded_stages[0][0].shape) for p in source_image_paths]
        columns = [np.stack(source_images, axis=0)] + columns
        labels = ["source"] + labels

    rows = int(columns[0].shape[0])
    height, width = columns[0].shape[1], columns[0].shape[2]
    label_height = 22
    panel = Image.new("RGB", (len(columns) * width, rows * height + label_height), "white")

    for col_index, label in enumerate(labels):
        _draw_label(panel, label, col_index * width + 4, 4)

    for col_index, images in enumerate(columns):
        for row_index in range(rows):
            image = Image.fromarray(images[row_index]).convert("RGB")
            panel.paste(image, (col_index * width, label_height + row_index * height))

    panel.save(path)


def copy_source_images(
    output_dir: str,
    source_image_paths: Sequence[Optional[str]],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for index, source_path in enumerate(source_image_paths):
        if source_path and os.path.exists(source_path):
            ext = os.path.splitext(source_path)[1] or ".png"
            shutil.copyfile(source_path, os.path.join(output_dir, f"{index:04d}{ext}"))


def _load_image_or_blank(path: Optional[str], shape: Sequence[int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    if path and os.path.exists(path):
        return np.asarray(Image.open(path).convert("RGB").resize((width, height)))
    return np.full((height, width, 3), 255, dtype=np.uint8)


def _draw_label(panel: Image.Image, label: str, x: int, y: int) -> None:
    try:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(panel)
        draw.text((x, y), label, fill=(0, 0, 0))
    except Exception:
        return
