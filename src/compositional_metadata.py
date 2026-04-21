import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MetadataRow = Dict[str, Any]
AttributeFilter = Mapping[str, Any]


@dataclass
class LatentMetadataDataset:
    latents: np.ndarray
    metadata: List[MetadataRow]
    image_path_column: str = "image_path"
    image_root: Optional[str] = None

    def __post_init__(self) -> None:
        if self.latents.ndim != 2:
            raise ValueError(
                "Expected a 2D latent matrix with shape (n_samples, dim); "
                f"got shape {self.latents.shape}."
            )
        if len(self.metadata) != self.latents.shape[0]:
            raise ValueError(
                "Metadata and latent count mismatch: "
                f"{len(self.metadata)} metadata rows vs {self.latents.shape[0]} latents."
            )

    @property
    def dim(self) -> int:
        return int(self.latents.shape[1])

    def filter_indices(self, constraints: Optional[AttributeFilter]) -> np.ndarray:
        return filter_metadata_indices(self.metadata, constraints)

    def subset(self, constraints: Optional[AttributeFilter]) -> np.ndarray:
        return self.latents[self.filter_indices(constraints)]

    def image_paths(self, indices: Sequence[int]) -> List[Optional[str]]:
        paths = []
        for index in indices:
            value = self.metadata[int(index)].get(self.image_path_column)
            if value in (None, ""):
                paths.append(None)
            else:
                paths.append(_resolve_path(str(value), self.image_root))
        return paths


def load_metadata(path: str) -> List[MetadataRow]:
    if not path:
        raise ValueError("metadata path is required.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if ext == ".json":
        with open(path) as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("samples", payload.get("metadata"))
        if not isinstance(payload, list):
            raise ValueError(
                "JSON metadata must be a list of rows or a dict with a "
                "'samples' or 'metadata' list."
            )
        return [dict(row) for row in payload]

    raise ValueError(f"Unsupported metadata extension '{ext}'. Use CSV or JSON.")


def load_latent_matrix(
    latents_path: Optional[str],
    metadata: Sequence[MetadataRow],
    latent_key: Optional[str] = None,
    latent_path_column: str = "latent_path",
    root: Optional[str] = None,
) -> np.ndarray:
    if latents_path:
        path = _resolve_path(latents_path, root)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Latent matrix not found: {path}")
        if path.endswith(".npz"):
            payload = np.load(path)
            key = latent_key or (payload.files[0] if len(payload.files) == 1 else None)
            if key is None:
                raise ValueError(
                    f"{path} contains multiple arrays {payload.files}; set data.latent_key."
                )
            latents = payload[key]
        else:
            latents = np.load(path)
        return np.asarray(latents, dtype=np.float32)

    rows_with_paths = [row for row in metadata if row.get(latent_path_column)]
    if len(rows_with_paths) != len(metadata):
        raise ValueError(
            "No shared latent matrix was provided, so every metadata row must "
            f"define '{latent_path_column}'."
        )

    latents = []
    for row in metadata:
        latent_path = _resolve_path(str(row[latent_path_column]), root)
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        latents.append(np.load(latent_path).reshape(-1))
    return np.asarray(latents, dtype=np.float32)


def load_latent_metadata_dataset(config: Mapping[str, Any]) -> LatentMetadataDataset:
    root = config.get("root")
    metadata_path = _resolve_path(str(config["metadata_path"]), root)
    metadata = load_metadata(metadata_path)
    latents = load_latent_matrix(
        config.get("latents_path"),
        metadata,
        latent_key=config.get("latent_key"),
        latent_path_column=config.get("latent_path_column", "latent_path"),
        root=root,
    )
    return LatentMetadataDataset(
        latents=latents,
        metadata=metadata,
        image_path_column=config.get("image_path_column", "image_path"),
        image_root=config.get("image_root", root),
    )


def filter_metadata_indices(
    metadata: Sequence[MetadataRow],
    constraints: Optional[AttributeFilter],
) -> np.ndarray:
    if not constraints:
        return np.arange(len(metadata), dtype=np.int64)
    matches = [
        index
        for index, row in enumerate(metadata)
        if _row_matches(row, constraints)
    ]
    return np.asarray(matches, dtype=np.int64)


def require_non_empty_subset(
    name: str,
    indices: np.ndarray,
    constraints: Optional[AttributeFilter],
) -> None:
    if len(indices) == 0:
        raise ValueError(
            f"Subset '{name}' is empty for constraints {dict(constraints or {})}. "
            "Check metadata values or relax the filter."
        )


def resolve_step_filters(
    step_config: Mapping[str, Any],
    subset_mode: str = "local",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if subset_mode not in {"global", "local"}:
        raise ValueError("subset_mode must be either 'global' or 'local'.")

    source_key = f"source_filter_{subset_mode}"
    target_key = f"target_filter_{subset_mode}"
    source_filter = step_config.get(source_key, step_config.get("source_filter", {}))
    target_filter = step_config.get(target_key, step_config.get("target_filter", {}))
    return dict(source_filter or {}), dict(target_filter or {})


def _row_matches(row: MetadataRow, constraints: AttributeFilter) -> bool:
    for key, expected in constraints.items():
        if expected is None:
            continue
        if key not in row:
            return False
        actual = row[key]
        if isinstance(expected, (list, tuple, set)):
            if not any(_same_value(actual, item) for item in expected):
                return False
        elif not _same_value(actual, expected):
            return False
    return True


def _same_value(actual: Any, expected: Any) -> bool:
    return str(actual).strip().lower() == str(expected).strip().lower()


def _resolve_path(path: str, root: Optional[str]) -> str:
    if os.path.isabs(path) or not root:
        return path
    return os.path.join(root, path)
