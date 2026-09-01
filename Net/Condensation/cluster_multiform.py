"""Cluster innovations layered on strict IDM.

The CACDM path uses resized raw pixels, PCA, and class-wise K-means.
An ablation may explicitly switch the descriptor to ResNet-18 or DINOv2;
the mode is embedded in cache metadata so variants can never reuse each
other's cluster assignments.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from Core.io_utils import lock_is_abandoned
from Net.Condensation.idm_official import partition_and_expand


PRETRAINED_ROOT = Path(__file__).resolve().parents[3] / "pretrained"
RESNET18_WEIGHT = PRETRAINED_ROOT / "resnet18-f37072fd.pth"
DINOV2_WEIGHT = (
    PRETRAINED_ROOT / "vit_small_patch14_dinov2_lvd142m.safetensors"
)
PRETRAINED_SHA256 = {
    RESNET18_WEIGHT.name: (
        "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    ),
    DINOV2_WEIGHT.name: (
        "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
    ),
}


def _verified_pretrained_path(path: Path) -> Path:
    """Require packaged ablation weights; default CACDM needs no weights."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Offline descriptor weight is missing: {path}. "
            "Place the verified weight in the project pretrained directory."
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    expected = PRETRAINED_SHA256[path.name]
    actual = digest.hexdigest().lower()
    if actual != expected:
        raise RuntimeError(
            f"Descriptor weight checksum mismatch: {path}; "
            f"expected={expected}, actual={actual}"
        )
    return path


@dataclass(frozen=True)
class PixelClusterIndex:
    """Deterministic per-class K-means membership and radial ordering."""

    members: dict[int, list[list[int]]]
    radial_members: dict[int, list[list[int]]]
    sizes: dict[int, list[int]]
    clusters_by_class: dict[int, int]
    descriptor_size: tuple[int, int]
    pca_components: int
    seed: int
    descriptor_mode: str = "pixel_pca"
    allocations: dict[int, list[int]] | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 4,
            "members": self.members,
            "radial_members": self.radial_members,
            "sizes": self.sizes,
            "clusters_by_class": {
                int(class_id): int(count)
                for class_id, count in self.clusters_by_class.items()
            },
            "descriptor_size": list(self.descriptor_size),
            "pca_components": int(self.pca_components),
            "seed": int(self.seed),
            "descriptor_mode": str(self.descriptor_mode),
            "allocations": self.allocations,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PixelClusterIndex":
        version = int(state.get("version", 0))
        if version not in {1, 2, 3, 4}:
            raise ValueError("Unsupported pixel-cluster cache version")
        members = {
            int(class_id): [list(map(int, group)) for group in groups]
            for class_id, groups in dict(state["members"]).items()
        }
        if version >= 4:
            clusters_by_class = {
                int(class_id): int(count)
                for class_id, count in dict(
                    state["clusters_by_class"]
                ).items()
            }
        else:
            legacy_count = int(state["clusters_per_class"])
            clusters_by_class = {
                int(class_id): legacy_count for class_id in members
            }
        return cls(
            members=members,
            radial_members={
                int(class_id): [list(map(int, group)) for group in groups]
                for class_id, groups in dict(state["radial_members"]).items()
            },
            sizes={
                int(class_id): list(map(int, values))
                for class_id, values in dict(state["sizes"]).items()
            },
            clusters_by_class=clusters_by_class,
            descriptor_size=tuple(map(int, state["descriptor_size"])),
            pca_components=int(state["pca_components"]),
            seed=int(state["seed"]),
            descriptor_mode=str(state.get("descriptor_mode", "pixel_pca")),
            allocations=(
                {
                    int(class_id): list(map(int, values))
                    for class_id, values in dict(state["allocations"]).items()
                }
                if state.get("allocations") is not None
                else None
            ),
        )

    def cluster_count(self, class_id: int) -> int:
        count = int(self.clusters_by_class[int(class_id)])
        if count <= 0:
            raise ValueError("Class cluster count must be positive")
        return count

    def representatives(
        self, class_id: int, cluster_id: int, count: int
    ) -> list[int]:
        """Select samples at evenly spaced centre-to-edge quantiles."""

        ordered = self.radial_members[int(class_id)][int(cluster_id)]
        if not ordered:
            raise ValueError("A K-means cluster cannot be empty")
        requested = max(1, int(count))
        if len(ordered) == 1:
            return [ordered[0]] * requested
        positions = torch.linspace(
            0, len(ordered) - 1, steps=requested
        ).round().to(torch.long).tolist()
        return [ordered[int(position)] for position in positions]

    def class_allocations(self, class_id: int, ipc: int) -> list[int]:
        """Return the exact canvas quota of every cluster in one class."""

        if self.allocations is None:
            cluster_count = self.cluster_count(int(class_id))
            if int(ipc) != cluster_count:
                raise ValueError("High IPC requires adaptive cluster capacity")
            return [1] * cluster_count
        values = list(map(int, self.allocations[int(class_id)]))
        if len(values) != self.cluster_count(int(class_id)):
            raise ValueError("Allocation count does not match cluster count")
        if min(values) <= 0 or sum(values) != int(ipc):
            raise ValueError("Allocations must be positive and sum to IPC")
        return values

    def canvas_representatives(
        self, class_id: int, ipc: int, sources_per_canvas: int
    ) -> tuple[list[list[int]], list[int]]:
        """Create one centre-to-edge source group for every allocated canvas."""

        source_groups: list[list[int]] = []
        group_ids: list[int] = []
        for cluster_id, canvas_count in enumerate(
            self.class_allocations(int(class_id), int(ipc))
        ):
            ordered = self.radial_members[int(class_id)][cluster_id]
            for canvas_id in range(int(canvas_count)):
                start = (canvas_id * len(ordered)) // int(canvas_count)
                stop = ((canvas_id + 1) * len(ordered)) // int(canvas_count)
                if stop <= start:
                    position = min(
                        len(ordered) - 1,
                        ((2 * canvas_id + 1) * len(ordered))
                        // (2 * int(canvas_count)),
                    )
                    cell = [ordered[position]]
                else:
                    cell = ordered[start:stop]
                positions = torch.linspace(
                    0,
                    len(cell) - 1,
                    steps=max(1, int(sources_per_canvas)),
                ).round().to(torch.long).tolist()
                source_groups.append([cell[int(position)] for position in positions])
                group_ids.append(int(cluster_id))
        if len(source_groups) != int(ipc):
            raise RuntimeError("Adaptive canvas allocation does not sum to IPC")
        return source_groups, group_ids


def size_allocations(sizes: Sequence[float], ipc: int) -> list[int]:
    """Lower-bounded Hamilton apportionment of the full IPC quota."""

    values = np.asarray(sizes, dtype=np.float64)
    cluster_count = int(values.size)
    if (
        cluster_count <= 0
        or int(ipc) < cluster_count
        or bool((values <= 0).any())
    ):
        raise ValueError("IPC must cover every non-empty cluster")
    quota = values / values.sum() * float(ipc)
    allocation = np.maximum(1, np.floor(quota).astype(np.int64))
    while int(allocation.sum()) < int(ipc):
        allocation[int(np.argmax(quota - allocation))] += 1
    while int(allocation.sum()) > int(ipc):
        eligible = allocation > 1
        if not bool(eligible.any()):
            raise RuntimeError("Minimum cluster capacity exceeds IPC")
        residual = np.where(eligible, quota - allocation, np.inf)
        allocation[int(np.argmin(residual))] -= 1
    return allocation.astype(int).tolist()


def attach_size_capacity(
    index: PixelClusterIndex, ipc: int
) -> PixelClusterIndex:
    return replace(
        index,
        allocations={
            int(class_id): size_allocations(sizes, int(ipc))
            for class_id, sizes in index.sizes.items()
        },
    )

def _raw_image(
    normalized: torch.Tensor,
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
) -> torch.Tensor:
    mean = torch.tensor(normalization_mean, dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(normalization_std, dtype=torch.float32).view(-1, 1, 1)
    return normalized.detach().float().cpu() * std + mean


def _descriptor_encoder(mode: str, device: torch.device):
    """Create an optional frozen descriptor encoder on demand."""

    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "resnet18":
        from torchvision.models import resnet18

        weight_path = _verified_pretrained_path(RESNET18_WEIGHT)
        model = resnet18(weights=None)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.fc = torch.nn.Identity()
    elif normalized_mode == "dinov2":
        try:
            import timm
        except ImportError as error:
            raise RuntimeError(
                "The DINOv2 descriptor ablation requires timm"
            ) from error
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise RuntimeError(
                "The DINOv2 descriptor ablation requires safetensors"
            ) from error
        weight_path = _verified_pretrained_path(DINOV2_WEIGHT)
        model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            # The published checkpoint defaults to 518px. Dynamic positional
            # interpolation keeps its weights frozen while accepting the
            # common 224px descriptor input used by both deep backends.
            dynamic_img_size=True,
        )
        model.load_state_dict(
            load_file(str(weight_path), device="cpu"), strict=True
        )
    else:
        raise ValueError(f"Unsupported descriptor mode: {mode}")
    model.eval().requires_grad_(False).to(device)
    return model


@torch.inference_mode()
def _deep_descriptors(
    dataset,
    indices: Sequence[int],
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    mean = torch.tensor(
        [0.485, 0.456, 0.406], device=device
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.229, 0.224, 0.225], device=device
    ).view(1, 3, 1, 1)
    parts: list[torch.Tensor] = []
    requested = max(1, int(batch_size))
    for start in range(0, len(indices), requested):
        images = torch.stack(
            [
                _raw_image(
                    dataset[index]["image"],
                    normalization_mean,
                    normalization_std,
                )
                for index in indices[start : start + requested]
            ]
        ).to(device, non_blocking=True)
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        images = F.interpolate(
            images,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        features = model((images - mean) / std)
        if isinstance(features, Mapping):
            features = features.get("x_norm_clstoken", next(iter(features.values())))
        if isinstance(features, (tuple, list)):
            features = features[0]
        parts.append(features.float().flatten(1).cpu())
        del images, features
    return torch.cat(parts, dim=0).numpy().astype(np.float32, copy=False)


def _deep_descriptor_cache(
    cache_path: Path,
    dataset,
    class_indices: Mapping[int, Sequence[int]],
    maximum_samples_per_class: int | None,
    minimum_samples_per_class: int,
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
    mode: str,
    device: torch.device,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """Build frozen deep features once; K-means seeds still remain independent."""

    selected_indices = {
        int(class_id): list(map(int, indices))[
            : max(
                int(minimum_samples_per_class),
                int(maximum_samples_per_class),
            )
            if maximum_samples_per_class is not None
            else None
        ]
        for class_id, indices in class_indices.items()
    }

    def load() -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        with np.load(cache_path, allow_pickle=False) as payload:
            for class_id, indices in selected_indices.items():
                stored_indices = payload[f"indices_{class_id}"]
                expected = np.asarray(indices, dtype=np.int64)
                if not np.array_equal(stored_indices, expected):
                    raise ValueError("Deep descriptor cache indices changed")
                features = payload[f"features_{class_id}"]
                if features.ndim != 2 or features.shape[0] != expected.size:
                    raise ValueError("Deep descriptor cache shape is invalid")
                result[class_id] = features.astype(np.float32, copy=False)
        return result

    if cache_path.is_file():
        try:
            return load()
        except (OSError, ValueError, KeyError):
            pass
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    descriptor: int | None = None
    while descriptor is None:
        if cache_path.is_file():
            try:
                return load()
            except (OSError, ValueError, KeyError):
                pass
        try:
            descriptor = os.open(
                lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            if lock_is_abandoned(lock_path, 7200.0):
                lock_path.unlink(missing_ok=True)
                continue
            time.sleep(0.25)
    try:
        # Only the lock owner may discard a damaged cache.
        if cache_path.is_file():
            try:
                return load()
            except (OSError, ValueError, KeyError):
                cache_path.unlink(missing_ok=True)
        model = _descriptor_encoder(mode, device)
        arrays: dict[str, np.ndarray] = {}
        result: dict[int, np.ndarray] = {}
        try:
            for class_id, indices in sorted(selected_indices.items()):
                features = _deep_descriptors(
                    dataset,
                    indices,
                    normalization_mean,
                    normalization_std,
                    model,
                    device,
                    int(batch_size),
                )
                arrays[f"indices_{class_id}"] = np.asarray(
                    indices, dtype=np.int64
                )
                arrays[f"features_{class_id}"] = features
                result[class_id] = features
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        temporary = cache_path.with_suffix(
            cache_path.suffix + f".{os.getpid()}.tmp"
        )
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        return result
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def build_pixel_cluster_index(
    dataset,
    class_indices: Mapping[int, Sequence[int]],
    clusters_per_class: int | Mapping[int, int],
    descriptor_size: Sequence[int],
    pca_components: int,
    kmeans_n_init: int,
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
    seed: int,
    maximum_samples_per_class: int | None = None,
    descriptor_mode: str = "pixel_pca",
    descriptor_device: torch.device | str = "cpu",
    descriptor_batch_size: int = 128,
    descriptor_cache_path: str | Path | None = None,
) -> PixelClusterIndex:
    """Build a class-wise descriptor/PCA/K-means index on training only."""

    if isinstance(clusters_per_class, Mapping):
        cluster_counts = {
            int(class_id): int(count)
            for class_id, count in clusters_per_class.items()
        }
    else:
        cluster_counts = {
            int(class_id): int(clusters_per_class)
            for class_id in class_indices
        }
    expected_classes = set(map(int, class_indices))
    if set(cluster_counts) != expected_classes:
        raise ValueError("clusters_per_class must cover every class exactly")
    if min(cluster_counts.values(), default=0) <= 0:
        raise ValueError("clusters_per_class must be positive")
    descriptor_hw = tuple(map(int, descriptor_size))
    if len(descriptor_hw) != 2 or min(descriptor_hw) <= 0:
        raise ValueError("descriptor_size must contain two positive integers")

    members: dict[int, list[list[int]]] = {}
    radial_members: dict[int, list[list[int]]] = {}
    sizes: dict[int, list[int]] = {}
    mode = str(descriptor_mode).strip().lower()
    if mode not in {"pixel_pca", "resnet18", "dinov2"}:
        raise ValueError(f"Unsupported descriptor mode: {descriptor_mode}")
    deep_descriptor_map = (
        _deep_descriptor_cache(
            Path(descriptor_cache_path),
            dataset,
            class_indices,
            maximum_samples_per_class,
            max(cluster_counts.values()),
            normalization_mean,
            normalization_std,
            mode,
            torch.device(descriptor_device),
            int(descriptor_batch_size),
        )
        if mode != "pixel_pca" and descriptor_cache_path is not None
        else None
    )
    deep_model = (
        _descriptor_encoder(mode, torch.device(descriptor_device))
        if mode != "pixel_pca" and deep_descriptor_map is None
        else None
    )
    for class_id in sorted(map(int, class_indices)):
        cluster_count = int(cluster_counts[class_id])
        indices = list(map(int, class_indices[class_id]))
        if maximum_samples_per_class is not None:
            indices = indices[: max(cluster_count, int(maximum_samples_per_class))]
        if len(indices) < cluster_count:
            raise ValueError(
                f"Class {class_id} has {len(indices)} samples for "
                f"{cluster_count} clusters"
            )

        if mode == "pixel_pca":
            descriptor_parts: list[torch.Tensor] = []
            for start in range(0, len(indices), 256):
                images = torch.stack(
                    [
                        _raw_image(
                            dataset[index]["image"],
                            normalization_mean,
                            normalization_std,
                        )
                        for index in indices[start : start + 256]
                    ]
                )
                resized = F.interpolate(
                    images,
                    size=descriptor_hw,
                    mode="bilinear",
                    align_corners=False,
                )
                descriptor_parts.append(resized.flatten(1))
            descriptors = torch.cat(descriptor_parts, dim=0).numpy().astype(
                np.float32, copy=False
            )
        elif deep_descriptor_map is not None:
            descriptors = deep_descriptor_map[class_id]
        else:
            descriptors = _deep_descriptors(
                dataset,
                indices,
                normalization_mean,
                normalization_std,
                deep_model,
                torch.device(descriptor_device),
                int(descriptor_batch_size),
            )
        components = min(
            int(pca_components),
            int(descriptors.shape[0] - 1),
            int(descriptors.shape[1]),
        )
        if components <= 0:
            raise ValueError("PCA requires at least two samples")
        reduced = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=int(seed) + class_id,
        ).fit_transform(descriptors)
        kmeans = KMeans(
            n_clusters=cluster_count,
            init="k-means++",
            n_init=int(kmeans_n_init),
            random_state=int(seed) + 1009 * class_id,
        ).fit(reduced)

        class_members: list[list[int]] = []
        class_radial: list[list[int]] = []
        for cluster_id in range(cluster_count):
            local = np.flatnonzero(kmeans.labels_ == cluster_id)
            if local.size == 0:
                raise RuntimeError("K-means returned an empty cluster")
            distances = np.linalg.norm(
                reduced[local] - kmeans.cluster_centers_[cluster_id], axis=1
            )
            radial_order = local[np.argsort(distances, kind="stable")]
            class_members.append([indices[int(position)] for position in local])
            class_radial.append(
                [indices[int(position)] for position in radial_order]
            )
        members[class_id] = class_members
        radial_members[class_id] = class_radial
        sizes[class_id] = [len(group) for group in class_members]

    if deep_model is not None:
        del deep_model
        if torch.device(descriptor_device).type == "cuda":
            torch.cuda.empty_cache()
    return PixelClusterIndex(
        members=members,
        radial_members=radial_members,
        sizes=sizes,
        clusters_by_class=cluster_counts,
        descriptor_size=descriptor_hw,
        pca_components=int(pca_components),
        seed=int(seed),
        descriptor_mode=str(descriptor_mode),
    )


class ClusterSampler:
    """No-replacement sampler with an independent stream per class/cluster."""

    def __init__(self, index: PixelClusterIndex, seed: int):
        self.index = index
        self.generator = torch.Generator().manual_seed(int(seed))
        self.orders: dict[tuple[int, int], list[int]] = {}
        self.positions: dict[tuple[int, int], int] = {}

    def take(self, class_id: int, cluster_id: int, count: int) -> list[int]:
        key = (int(class_id), int(cluster_id))
        candidates = self.index.members[key[0]][key[1]]
        result: list[int] = []
        while len(result) < int(count):
            order = self.orders.get(key, [])
            position = self.positions.get(key, 0)
            if position >= len(order):
                permutation = torch.randperm(
                    len(candidates), generator=self.generator
                ).tolist()
                order = [candidates[item] for item in permutation]
                self.orders[key] = order
                position = 0
            amount = min(int(count) - len(result), len(order) - position)
            result.extend(order[position : position + amount])
            self.positions[key] = position + amount
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator_state": self.generator.get_state(),
            "orders": {
                f"{class_id}:{cluster_id}": values
                for (class_id, cluster_id), values in self.orders.items()
            },
            "positions": {
                f"{class_id}:{cluster_id}": int(value)
                for (class_id, cluster_id), value in self.positions.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        if torch.is_tensor(state.get("generator_state")):
            self.generator.set_state(state["generator_state"].cpu())
        self.orders = {
            tuple(map(int, key.split(":"))): list(map(int, values))
            for key, values in dict(state.get("orders", {})).items()
        }
        self.positions = {
            tuple(map(int, key.split(":"))): int(value)
            for key, value in dict(state.get("positions", {})).items()
        }


def compose_storage_canvas(
    source_images: torch.Tensor,
    image_size: Sequence[int],
    partition_factor: int,
) -> torch.Tensor:
    """Pack representative real images into one learnable stored canvas."""

    height, width = map(int, image_size)
    channels = int(source_images.shape[1])
    factor = int(partition_factor)
    if factor not in {1, 2}:
        raise ValueError("partition_factor must be 1 or 2")
    spatial_views = factor**2
    expected = spatial_views
    if int(source_images.shape[0]) != expected:
        raise ValueError(
            f"Expected {expected} source images, got {source_images.shape[0]}"
        )
    canvas = torch.empty(
        (channels, height, width), dtype=source_images.dtype
    )
    positions = (
        ((0, 0),)
        if factor == 1
        else ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    for patch_index, (row, column) in enumerate(positions):
        source = source_images[patch_index : patch_index + 1]
        patch = F.interpolate(
            source,
            size=(height // factor, width // factor),
            mode="bilinear",
            align_corners=False,
        )[0]
        canvas[
            :,
            row * height // factor : (row + 1) * height // factor,
            column * width // factor : (column + 1) * width // factor,
        ] = patch
    return canvas


def partition_training_images(
    images: torch.Tensor,
    labels: torch.Tensor,
    partition_factor: int,
    group_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply official 2x2 P&E and preserve optional cluster identifiers."""

    factor = int(partition_factor)
    expanded, expanded_labels = partition_and_expand(
        images, labels, factor
    )
    return (
        expanded,
        expanded_labels,
        group_ids.repeat(factor**2) if group_ids is not None else None,
    )


def cluster_distribution_losses(
    real_features: torch.Tensor,
    real_groups: torch.Tensor,
    synthetic_features: torch.Tensor,
    synthetic_groups: torch.Tensor,
    cluster_count: int,
    cluster_weights: Sequence[float] | torch.Tensor,
    radial_weight: float,
    standard_deviation_weight: float,
    smooth_l1_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match cluster centres and bounded centre-to-edge feature coverage."""

    real = real_features.float().flatten(1)
    synthetic = synthetic_features.float().flatten(1)
    spread_loss = synthetic.sum() * 0.0
    compute_spread = (
        float(radial_weight) > 0.0
        or float(standard_deviation_weight) > 0.0
    )
    weights = torch.as_tensor(
        cluster_weights, dtype=synthetic.dtype, device=synthetic.device
    ).flatten()
    if int(weights.numel()) != int(cluster_count):
        raise ValueError("cluster_weights must contain one value per cluster")
    if bool((weights <= 0).any()):
        raise ValueError("cluster_weights must be positive")
    weights = weights / weights.sum()
    real_group_ids = real_groups.to(
        device=real.device, dtype=torch.long
    ).flatten()
    synthetic_group_ids = synthetic_groups.to(
        device=synthetic.device, dtype=torch.long
    ).flatten()
    if int(real_group_ids.numel()) != int(real.shape[0]):
        raise ValueError("real_groups must contain one id per real feature")
    if int(synthetic_group_ids.numel()) != int(synthetic.shape[0]):
        raise ValueError(
            "synthetic_groups must contain one id per synthetic feature"
        )

    # Idea 2 原实现逐簇做布尔索引和 mean，每个类别/指导网络都会启动
    # 2*K 个小 GPU kernel。index_add 一次聚合全部簇，目标函数和梯度不变。
    real_counts = torch.bincount(
        real_group_ids, minlength=int(cluster_count)
    )
    synthetic_counts = torch.bincount(
        synthetic_group_ids, minlength=int(cluster_count)
    )
    if int(real_counts.numel()) != int(cluster_count):
        raise ValueError("real_groups contains an out-of-range cluster id")
    if int(synthetic_counts.numel()) != int(cluster_count):
        raise ValueError("synthetic_groups contains an out-of-range cluster id")
    if bool((real_counts < 2).any()):
        raise ValueError("Each real cluster target needs at least two samples")
    if bool((synthetic_counts < 2).any()):
        raise ValueError("Each synthetic cluster needs at least two views")
    real_sums = real.new_zeros((int(cluster_count), real.shape[1])).index_add(
        0, real_group_ids, real
    )
    synthetic_sums = synthetic.new_zeros(
        (int(cluster_count), synthetic.shape[1])
    ).index_add(0, synthetic_group_ids, synthetic)
    real_means = (
        real_sums / real_counts.to(real.dtype).unsqueeze(1)
    ).detach()
    synthetic_means = synthetic_sums / synthetic_counts.to(
        synthetic.dtype
    ).unsqueeze(1)
    mean_loss = (
        (synthetic_means - real_means).square().sum(dim=1) * weights
    ).sum()

    # Idea 1+2 到这里直接返回，不再进入任何逐簇 Python 循环。
    if not compute_spread:
        return mean_loss, spread_loss

    # Idea 3 先一次性计算全部样本相对各自簇中心的半径。原实现每簇对
    # [N,D] 特征做两次布尔切片；现在循环里只处理很小的 1-D 半径。
    dimension_scale = math.sqrt(float(real.shape[1]))
    if float(radial_weight) > 0.0:
        real_radii = (
            (
                real
                - real_means.index_select(0, real_group_ids)
            ).norm(dim=1)
            / dimension_scale
        ).detach()
        synthetic_radii = (
            synthetic
            - real_means.index_select(0, synthetic_group_ids)
        ).norm(dim=1) / dimension_scale
        radial_loss_sum = spread_loss
        for cluster_id in range(int(cluster_count)):
            real_radius = real_radii[
                real_group_ids == int(cluster_id)
            ]
            synthetic_radius = synthetic_radii[
                synthetic_group_ids == int(cluster_id)
            ]
            count = int(synthetic_radius.numel())
            quantiles = (
                torch.arange(
                    count,
                    device=synthetic.device,
                    dtype=torch.float32,
                )
                + 0.5
            ) / float(count)
            radial_target = torch.quantile(
                real_radius, quantiles
            ).detach()
            radial_loss = F.smooth_l1_loss(
                synthetic_radius.sort().values,
                radial_target,
                beta=float(smooth_l1_beta),
            )
            radial_loss_sum = radial_loss_sum + (
                weights[cluster_id]
                * float(radial_weight)
                * radial_loss
            )
        spread_loss = radial_loss_sum

    # 对角标准差使用 E[x^2]-E[x]^2 和 index_add 一次聚合全部簇，
    # 避免完整 CACDM 再对每簇切片完整特征矩阵。
    if float(standard_deviation_weight) > 0.0:
        real_square_sums = real.new_zeros(
            (int(cluster_count), real.shape[1])
        ).index_add(0, real_group_ids, real.square())
        synthetic_square_sums = synthetic.new_zeros(
            (int(cluster_count), synthetic.shape[1])
        ).index_add(0, synthetic_group_ids, synthetic.square())
        real_second_moments = real_square_sums / real_counts.to(
            real.dtype
        ).unsqueeze(1)
        synthetic_second_moments = (
            synthetic_square_sums
            / synthetic_counts.to(synthetic.dtype).unsqueeze(1)
        )
        real_stds = (
            real_second_moments - real_means.square()
        ).clamp_min(0.0).sqrt().detach()
        synthetic_stds = (
            synthetic_second_moments - synthetic_means.square()
        ).clamp_min(0.0).sqrt()
        std_losses = F.smooth_l1_loss(
            synthetic_stds,
            real_stds,
            beta=float(smooth_l1_beta),
            reduction="none",
        ).mean(dim=1)
        spread_loss = spread_loss + (
            weights
            * float(standard_deviation_weight)
            * std_losses
        ).sum()
    return mean_loss, spread_loss
