import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from src.compositional_metadata import (
    LatentMetadataDataset,
    require_non_empty_subset,
    resolve_step_filters,
)
from src.distributions import TensorSampler
from src.light_sb import LightSB


@dataclass
class AttributeBridgeStep:
    name: str
    source_filter: Dict[str, Any]
    target_filter: Dict[str, Any]
    bridge_object: Optional[LightSB] = None
    source_indices: Optional[np.ndarray] = None
    target_indices: Optional[np.ndarray] = None
    cache_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        return self.bridge_object is not None


class CompositionalLightSB:
    def __init__(
        self,
        steps: Sequence[AttributeBridgeStep],
        device: str = "cpu",
    ) -> None:
        self.steps = list(steps)
        self.device = device

    def transform(self, z: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        return self.transform_with_intermediates(z, num_steps=num_steps)[-1]

    def transform_with_intermediates(
        self,
        z: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> List[torch.Tensor]:
        active_steps = self.steps if num_steps is None else self.steps[:num_steps]
        current = z.to(self.device).float()
        intermediates = [current.detach().cpu()]
        for step in active_steps:
            current = apply_bridge(step, current)
            intermediates.append(current.detach().cpu())
        return intermediates


def build_steps_from_config(
    step_configs: Sequence[Mapping[str, Any]],
    subset_mode: str = "local",
) -> List[AttributeBridgeStep]:
    steps = []
    for raw_step in step_configs:
        source_filter, target_filter = resolve_step_filters(raw_step, subset_mode=subset_mode)
        steps.append(
            AttributeBridgeStep(
                name=str(raw_step["name"]),
                source_filter=source_filter,
                target_filter=target_filter,
            )
        )
    return steps


def fit_attribute_bridge(
    step: AttributeBridgeStep,
    dataset: LatentMetadataDataset,
    bridge_config: Mapping[str, Any],
    cache_dir: str,
    device: str = "cpu",
    force_refit: bool = False,
) -> AttributeBridgeStep:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{step.name}.pt")
    step.cache_path = cache_path

    dim = int(bridge_config.get("dim") or dataset.dim)
    n_potentials = int(bridge_config.get("n_potentials", 10))
    epsilon = float(bridge_config.get("epsilon", 0.1))
    is_diagonal = bool(bridge_config.get("is_diagonal", True))
    sampling_batch_size = int(bridge_config.get("sampling_batch_size", 128))
    s_diagonal_init = float(bridge_config.get("S_diagonal_init", 0.1))

    if os.path.exists(cache_path) and not force_refit:
        bridge = _make_bridge(dim, n_potentials, epsilon, is_diagonal, sampling_batch_size, s_diagonal_init)
        payload = torch.load(cache_path, map_location=device)
        bridge.load_state_dict(payload["state_dict"])
        bridge.to(device)
        step.bridge_object = bridge
        step.source_indices = _as_index_array(payload.get("source_indices"))
        step.target_indices = _as_index_array(payload.get("target_indices"))
        step.metrics = payload.get("metrics", {})
        print(f"[{step.name}] loaded cached bridge from {cache_path}")
        return step

    source_indices = dataset.filter_indices(step.source_filter)
    target_indices = dataset.filter_indices(step.target_filter)
    require_non_empty_subset(f"{step.name}:source", source_indices, step.source_filter)
    require_non_empty_subset(f"{step.name}:target", target_indices, step.target_filter)
    if len(target_indices) < n_potentials:
        raise ValueError(
            f"[{step.name}] target subset has {len(target_indices)} samples, "
            f"but n_potentials={n_potentials}. Lower n_potentials or add data."
        )
    if dataset.dim != dim:
        raise ValueError(f"[{step.name}] config dim={dim}, but latent dim={dataset.dim}.")

    source_latents = torch.tensor(dataset.latents[source_indices], dtype=torch.float32, device=device)
    target_latents = torch.tensor(dataset.latents[target_indices], dtype=torch.float32, device=device)
    bridge = _make_bridge(dim, n_potentials, epsilon, is_diagonal, sampling_batch_size, s_diagonal_init).to(device)

    if bool(bridge_config.get("init_by_samples", True)):
        perm = torch.randperm(target_latents.shape[0], device=device)[:n_potentials]
        bridge.init_r_by_samples(target_latents[perm])

    optimizer = torch.optim.Adam(bridge.parameters(), lr=float(bridge_config.get("lr", 1e-3)))
    source_sampler = TensorSampler(source_latents, device=device)
    target_sampler = TensorSampler(target_latents, device=device)

    requested_batch_size = int(bridge_config.get("batch_size", 128))
    batch_size = min(requested_batch_size, int(source_latents.shape[0]), int(target_latents.shape[0]))
    max_steps = int(bridge_config.get("max_steps", 10000))
    grad_max_norm = float(bridge_config.get("gradient_max_norm", float("inf")))
    log_every = int(bridge_config.get("log_every", 100))
    losses = []

    print(
        f"[{step.name}] fitting LightSB with "
        f"{len(source_indices)} source / {len(target_indices)} target latents"
    )
    for train_step in tqdm(range(max_steps), desc=f"fit:{step.name}"):
        optimizer.zero_grad()
        x0 = source_sampler.sample(batch_size)
        x1 = target_sampler.sample(batch_size)
        loss = (-bridge.get_log_potential(x1) + bridge.get_log_C(x0)).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), max_norm=grad_max_norm)
        optimizer.step()

        if train_step % log_every == 0 or train_step == max_steps - 1:
            losses.append(float(loss.detach().cpu()))
            print(
                f"[{step.name}] step={train_step} "
                f"loss={float(loss.detach().cpu()):.6f} grad_norm={float(grad_norm):.6f}"
            )

    step.bridge_object = bridge
    step.source_indices = source_indices
    step.target_indices = target_indices
    step.metrics = {
        "losses": losses,
        "source_count": int(len(source_indices)),
        "target_count": int(len(target_indices)),
        "epsilon": epsilon,
        "n_potentials": n_potentials,
        "is_diagonal": is_diagonal,
    }
    torch.save(
        {
            "name": step.name,
            "source_filter": step.source_filter,
            "target_filter": step.target_filter,
            "source_indices": source_indices.tolist(),
            "target_indices": target_indices.tolist(),
            "metrics": step.metrics,
            "state_dict": bridge.state_dict(),
        },
        cache_path,
    )
    print(f"[{step.name}] saved bridge to {cache_path}")
    return step


def fit_bridges(
    steps: Sequence[AttributeBridgeStep],
    dataset: LatentMetadataDataset,
    bridge_config: Mapping[str, Any],
    cache_dir: str,
    device: str = "cpu",
    force_refit: bool = False,
) -> List[AttributeBridgeStep]:
    fitted = []
    for step in steps:
        fitted.append(
            fit_attribute_bridge(
                step,
                dataset,
                bridge_config,
                cache_dir=cache_dir,
                device=device,
                force_refit=force_refit,
            )
        )
    return fitted


def apply_bridge(step: AttributeBridgeStep, z: torch.Tensor) -> torch.Tensor:
    if step.bridge_object is None:
        raise RuntimeError(f"Bridge step '{step.name}' is not fitted or loaded.")
    bridge = step.bridge_object
    bridge.eval()
    with torch.no_grad():
        return bridge(z.to(next(bridge.parameters()).device).float())


def compose_bridges(
    steps: Sequence[AttributeBridgeStep],
    z: torch.Tensor,
    num_steps: Optional[int] = None,
) -> List[torch.Tensor]:
    pipeline = CompositionalLightSB(steps, device=str(z.device))
    return pipeline.transform_with_intermediates(z, num_steps=num_steps)


def fit_one_step_bridge_from_filters(
    name: str,
    source_filter: Mapping[str, Any],
    target_filter: Mapping[str, Any],
    dataset: LatentMetadataDataset,
    bridge_config: Mapping[str, Any],
    cache_dir: str,
    device: str = "cpu",
    force_refit: bool = False,
) -> AttributeBridgeStep:
    step = AttributeBridgeStep(
        name=name,
        source_filter=dict(source_filter),
        target_filter=dict(target_filter),
    )
    return fit_attribute_bridge(
        step,
        dataset,
        bridge_config,
        cache_dir=cache_dir,
        device=device,
        force_refit=force_refit,
    )


def _make_bridge(
    dim: int,
    n_potentials: int,
    epsilon: float,
    is_diagonal: bool,
    sampling_batch_size: int,
    s_diagonal_init: float,
) -> LightSB:
    return LightSB(
        dim=dim,
        n_potentials=n_potentials,
        epsilon=epsilon,
        is_diagonal=is_diagonal,
        sampling_batch_size=sampling_batch_size,
        S_diagonal_init=s_diagonal_init,
    )


def _as_index_array(indices: Any) -> Optional[np.ndarray]:
    if indices is None:
        return None
    return np.asarray(indices, dtype=np.int64)
