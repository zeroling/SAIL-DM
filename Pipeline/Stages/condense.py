"""Pixel IDM with cluster-conditioned, size-weighted distribution matching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from Core.checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
)
from Core.config import output_root
from Core.data import build_loader, unpack_batch
from Core.io_utils import atomic_write_json, lock_is_abandoned, read_json
from Core.logging_utils import get_stage_logger
from Core.results_report import refresh_results_report
from Core.run_context import autocast_context, resolve_device
from Core.seed import seed_everything
from Pipeline.data import TensorImageDataset, experiment_bundle
from Net.Condensation.idm_official import (
    IDMConvNet,
    IDMFeatures,
    ParamDiffAug,
    build_idm_convnet,
    diff_augment,
    initialize_partitioned_pixels,
)
from Core.experiment_runtime import (
    cleanup_memory,
    cuda_peak_megabytes,
)
from Net.Condensation.cluster_multiform import (
    ClusterSampler,
    PixelClusterIndex,
    attach_size_capacity,
    build_pixel_cluster_index,
    cluster_distribution_losses,
    compose_storage_canvas,
    partition_training_images,
)

ARCHITECTURE_VERSION = 11


def _algorithm_version(condensation: Mapping[str, Any]) -> int:
    """Return the explicit checkpoint version for the active IDM pipeline."""

    del condensation
    return ARCHITECTURE_VERSION


def _ipc(config: Mapping[str, Any], override: int | None = None) -> int:
    value = int(
        override
        if override is not None
        else config["condensation"]["idm"]["ipc"]
    )
    if value <= 0:
        raise ValueError("IPC 必须为正整数")
    return value


def _idm_value(
    settings: Mapping[str, Any], key: str, ipc: int
) -> float | int:
    """Resolve an IDM scalar, optionally from its per-IPC schedule."""

    schedule = settings.get(f"{key}_by_ipc")
    if not schedule:
        if key not in settings:
            raise KeyError(f"Missing IDM setting: {key}")
        return settings[key]
    direct = schedule.get(int(ipc), schedule.get(str(int(ipc))))
    if direct is not None:
        return direct
    known = sorted((int(point), value) for point, value in schedule.items())
    lower = [item for item in known if item[0] <= int(ipc)]
    return (lower[-1] if lower else known[0])[1]


def _ce_weight(config: Mapping[str, Any], ipc: int) -> float:
    return float(
        _idm_value(config["condensation"]["idm"], "ce_weight", ipc)
    )


def _partition_factor(
    settings: Mapping[str, Any], ipc: int
) -> int:
    factor = int(
        _idm_value(settings["idm"], "partition_expansion", int(ipc))
    )
    if factor not in {1, 2}:
        raise ValueError("partition expansion 只支持 1 或 2")
    return factor


class IndexedImagePool:
    """不放回循环采样真实图；图像首次读取后缓存为 CPU float16。"""

    def __init__(
        self,
        dataset,
        num_classes: int,
        seed: int,
        cache_images: bool = True,
    ):
        self.dataset = dataset
        self.num_classes = int(num_classes)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.class_indices: dict[int, list[int]] = {
            class_id: [] for class_id in range(self.num_classes)
        }
        for index, label in enumerate(dataset.targets):
            self.class_indices[int(label)].append(int(index))
        self.targets = torch.as_tensor(dataset.targets, dtype=torch.long)
        self.class_orders: dict[int, list[int]] = {}
        self.class_positions = {class_id: 0 for class_id in self.class_indices}
        self.all_indices = list(range(len(dataset)))
        self.all_order: list[int] = []
        self.all_position = 0
        # 定长列表比逐样本字典查找更轻；缓存内容和原实现完全相同。
        self.cache_images = bool(cache_images)
        self.cache: list[torch.Tensor | None] = (
            [None] * len(dataset) if self.cache_images else []
        )

    def _reshuffle(self, candidates: list[int]) -> list[int]:
        order = torch.randperm(
            len(candidates), generator=self.generator
        ).tolist()
        return [candidates[position] for position in order]

    def _take_class(self, class_id: int, count: int) -> list[int]:
        result: list[int] = []
        candidates = self.class_indices[int(class_id)]
        while len(result) < int(count):
            order = self.class_orders.get(int(class_id), [])
            position = self.class_positions[int(class_id)]
            if position >= len(order):
                order = self._reshuffle(candidates)
                self.class_orders[int(class_id)] = order
                position = 0
            amount = min(int(count) - len(result), len(order) - position)
            result.extend(order[position : position + amount])
            self.class_positions[int(class_id)] = position + amount
        return result

    def _take_all(self, count: int) -> list[int]:
        result: list[int] = []
        while len(result) < int(count):
            if self.all_position >= len(self.all_order):
                self.all_order = self._reshuffle(self.all_indices)
                self.all_position = 0
            amount = min(
                int(count) - len(result),
                len(self.all_order) - self.all_position,
            )
            result.extend(
                self.all_order[
                    self.all_position : self.all_position + amount
                ]
            )
            self.all_position += amount
        return result

    def _cached_half(self, index: int) -> torch.Tensor:
        normalized_index = int(index)
        if not self.cache_images:
            return (
                self.dataset[normalized_index]["image"]
                .detach()
                .to(dtype=torch.float16)
                .cpu()
                .contiguous()
            )
        cached = self.cache[normalized_index]
        if cached is None:
            image = self.dataset[normalized_index]["image"]
            cached = (
                image.detach()
                .to(dtype=torch.float16)
                .cpu()
                .contiguous()
            )
            self.cache[normalized_index] = cached
        return cached

    def _batch(self, indices: list[int]) -> torch.Tensor:
        # 先堆叠 half，再用一次向量化转换得到 float32。逐张 half.float()
        # 会为每个样本启动一个 CPU kernel，在较慢的服务器 CPU 上代价很高。
        return torch.stack(
            [self._cached_half(index) for index in indices]
        ).float()

    def sample(self, class_id: int, count: int) -> torch.Tensor:
        return self._batch(self._take_class(class_id, count))

    def images(self, indices: list[int]) -> torch.Tensor:
        return self._batch(indices)

    def sample_general(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self._take_all(count)
        images = self._batch(indices)
        labels = self.targets.index_select(
            0, torch.as_tensor(indices, dtype=torch.long)
        )
        return images, labels

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator_state": self.generator.get_state(),
            "class_orders": self.class_orders,
            "class_positions": self.class_positions,
            "all_order": self.all_order,
            "all_position": int(self.all_position),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        if torch.is_tensor(state.get("generator_state")):
            self.generator.set_state(state["generator_state"].cpu())
        self.class_orders = {
            int(key): list(map(int, value))
            for key, value in dict(state.get("class_orders", {})).items()
        }
        self.class_positions = {
            int(key): int(value)
            for key, value in dict(
                state.get("class_positions", self.class_positions)
            ).items()
        }
        self.all_order = list(map(int, state.get("all_order", [])))
        self.all_position = int(state.get("all_position", 0))


def _rank_assigned_members(
    assignments: torch.Tensor,
    centers: torch.Tensor,
    features: torch.Tensor,
    cluster_count: int,
) -> list[list[int]]:
    """Rank members only against their assigned centre with one CPU transfer."""

    assigned_centers = centers.index_select(0, assignments)
    assigned_distances = (
        features.float() - assigned_centers.float()
    ).square().sum(dim=1)
    global_order = torch.argsort(assigned_distances)
    ordered = torch.stack(
        (assignments.index_select(0, global_order), global_order), dim=0
    ).detach().cpu().to(torch.long)
    ranked: list[list[int]] = [[] for _ in range(int(cluster_count))]
    for cluster_id, member in zip(
        ordered[0].tolist(), ordered[1].tolist(), strict=True
    ):
        ranked[int(cluster_id)].append(int(member))
    empty_clusters = [
        cluster_id
        for cluster_id, members in enumerate(ranked)
        if not members
    ]
    if empty_clusters:
        empty_centers = centers[empty_clusters].float()
        fallback = torch.cdist(empty_centers, features.float()).argmin(1)
        fallback_members = fallback.detach().cpu().tolist()
        for cluster_id, member in zip(
            empty_clusters, fallback_members, strict=True
        ):
            ranked[cluster_id].append(int(member))
    return ranked


class DreamRepresentativeSampler:
    """DREAM-style representative sampling from current network embeddings.

    The selected indices are cached per guidance model/class and refreshed at
    the configured interval. K-means runs on the GPU through the same package
    used by the official DREAM implementation.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        pool: IndexedImagePool,
        device: torch.device,
        seed: int,
    ) -> None:
        self.config = config
        self.pool = pool
        self.device = device
        self.seed = int(seed)
        settings = config["condensation"].get(
            "representative_sampling", {}
        )
        self.interval = max(
            1, int(settings.get("refresh_interval_iterations", 10))
        )
        self.maximum_candidates = max(
            1, int(settings.get("maximum_candidates_per_class", 8192))
        )
        self.maximum_clusters = max(
            1, int(settings.get("maximum_clusters_per_class", 256))
        )
        self.cache: dict[tuple[int, int], tuple[int, list[int]]] = {}

    @torch.inference_mode()
    def _embedding(self, model: IDMConvNet, indices: list[int]) -> torch.Tensor:
        requested = max(
            1,
            int(
                self.config["condensation"]["memory"].get(
                    "real_feature_microbatch", 256
                )
            ),
        )
        parts: list[torch.Tensor] = []
        for start in range(0, len(indices), requested):
            images = self.pool.images(indices[start : start + requested]).to(
                self.device, non_blocking=True
            )
            with autocast_context(self.config, self.device):
                features = model.forward_idm(images).embedding
            parts.append(features.float().flatten(1))
            del images, features
        return torch.cat(parts, dim=0)

    @torch.inference_mode()
    def _select(
        self,
        model: IDMConvNet,
        class_id: int,
        count: int,
        iteration: int,
    ) -> list[int]:
        all_indices = list(map(int, self.pool.class_indices[int(class_id)]))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            self.seed
            + int(iteration) * 1009
            + int(class_id) * 9176
            + int(count)
        )
        if len(all_indices) > self.maximum_candidates:
            order = torch.randperm(len(all_indices), generator=generator)
            candidates = [
                all_indices[int(position)]
                for position in order[: self.maximum_candidates]
            ]
        else:
            candidates = all_indices
        features = self._embedding(model, candidates)
        cluster_count = min(
            int(count), len(candidates), self.maximum_clusters
        )
        if cluster_count <= 0:
            raise RuntimeError("DREAM representative candidate pool is empty")
        try:
            from fast_pytorch_kmeans import KMeans as TorchKMeans
        except ImportError as error:
            raise RuntimeError(
                "DREAM-DM requires fast-pytorch-kmeans"
            ) from error
        kmeans = TorchKMeans(
            n_clusters=int(cluster_count), mode="euclidean"
        )
        assignments = kmeans.fit_predict(features)
        centers = kmeans.centroids
        # We only need each sample's distance to its assigned final centre.
        # The previous full [K,N] cdist repeated K times more arithmetic and
        # then synchronized GPU->CPU once per cluster.  One assigned-distance
        # vector and one bulk transfer preserve the same nearest-member rule.
        ranked = _rank_assigned_members(
            assignments, centers, features, cluster_count
        )
        # Round-robin nearest members across clusters. At high batch_real this
        # keeps far more distinct real images than repeating only centroids.
        selected_local: list[int] = []
        depth = 0
        while len(selected_local) < min(int(count), len(candidates)):
            changed = False
            for members in ranked:
                if depth < len(members):
                    selected_local.append(int(members[depth]))
                    changed = True
                    if len(selected_local) >= min(int(count), len(candidates)):
                        break
            if not changed:
                break
            depth += 1
        selected = [candidates[position] for position in selected_local]
        while len(selected) < int(count):
            position = int(
                torch.randint(
                    len(selected), (1,), generator=generator
                ).item()
            )
            selected.append(selected[position])
        del (
            features,
            centers,
            assignments,
            kmeans,
        )
        return selected[: int(count)]

    def sample(
        self,
        model: IDMConvNet,
        member_identifier: int,
        class_id: int,
        count: int,
        iteration: int,
    ) -> torch.Tensor:
        key = (int(member_identifier), int(class_id))
        cached = self.cache.get(key)
        refresh_bucket = int(iteration) // self.interval
        if cached is None or int(cached[0]) != refresh_bucket:
            indices = self._select(
                model, int(class_id), int(count), int(iteration)
            )
            self.cache[key] = (refresh_bucket, indices)
        return self.pool.images(self.cache[key][1])

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "entries": [
                {
                    "member_identifier": int(member_identifier),
                    "class_id": int(class_id),
                    "refresh_bucket": int(refresh_bucket),
                    "indices": list(map(int, indices)),
                }
                for (
                    member_identifier,
                    class_id,
                ), (
                    refresh_bucket,
                    indices,
                ) in sorted(self.cache.items())
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        self.cache = {}
        if not state:
            return
        if int(state.get("version", -1)) != 1:
            raise ValueError("DREAM representative cache version mismatch")
        for item in state.get("entries", []):
            key = (
                int(item["member_identifier"]),
                int(item["class_id"]),
            )
            self.cache[key] = (
                int(item["refresh_bucket"]),
                list(map(int, item["indices"])),
            )


def _cluster_cache_path(
    config: Mapping[str, Any],
    ipc: int,
    cluster_counts: Mapping[int, int],
    seed: int,
    dataset_size: int,
) -> Path:
    settings = config["condensation"]["cluster_matching"]
    root = Path(str(settings["cache_root"]))
    if not root.is_absolute():
        root = Path(config["_runtime"]["project_root"]) / root
    descriptor = "x".join(map(str, settings["descriptor_size"]))
    descriptor_mode = str(settings.get("descriptor_mode", "pixel_pca"))
    layout_token = "-".join(
        str(int(cluster_counts[class_id]))
        for class_id in sorted(cluster_counts)
    )
    support = int(settings.get("images_per_cluster", 100))
    mode = str(settings.get("cluster_count_mode", "fixed"))
    cluster_suffix = f"_k{layout_token}_{mode}_support{support}"
    name = (
        f"{config['_runtime']['dataset']}_n{int(dataset_size)}_ipc{int(ipc)}"
        f"{cluster_suffix}_"
        f"mode{descriptor_mode}_d{descriptor}_"
        f"pca{int(settings['pca_components'])}_seed{int(seed)}.pt"
    )
    candidate = root.resolve() / name
    # Windows still applies a short path limit in tempfile.mkstemp on some
    # installations.  Native 224x224 cache names can cross that limit because
    # the project root is intentionally descriptive.  Keep the readable name
    # whenever possible; otherwise replace only the filename with a stable
    # digest of the complete cache identity.  The clustering inputs and cache
    # contents are unchanged.
    if os.name == "nt" and len(str(candidate)) > 230:
        identity = hashlib.sha256(name.encode("utf-8")).hexdigest()[:20]
        compact = (
            f"{config['_runtime']['dataset']}_ipc{int(ipc)}_{identity}.pt"
        )
        candidate = root.resolve() / compact
    return candidate


def _load_or_build_cluster_index(
    config: Mapping[str, Any],
    pool: IndexedImagePool,
    ipc: int,
    cluster_counts: Mapping[int, int],
    seed: int,
    device: torch.device,
) -> tuple[PixelClusterIndex, Path]:
    cache_path = _cluster_cache_path(
        config,
        ipc,
        cluster_counts,
        seed,
        len(pool.dataset),
    )
    if bool(config.get("_smoke", False)):
        cache_path = cache_path.with_name(
            f"{cache_path.stem}_smoke{cache_path.suffix}"
        )
    def for_requested_ipc(index: PixelClusterIndex) -> PixelClusterIndex:
        return attach_size_capacity(index, int(ipc))

    if cache_path.is_file():
        payload = load_checkpoint(cache_path, "cpu")
        return for_requested_ipc(PixelClusterIndex.from_state_dict(payload)), cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    owns_lock = False
    while not owns_lock:
        if cache_path.is_file():
            payload = load_checkpoint(cache_path, "cpu")
            return for_requested_ipc(PixelClusterIndex.from_state_dict(payload)), cache_path
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
            os.close(descriptor)
            owns_lock = True
        except FileExistsError:
            try:
                stale = lock_is_abandoned(lock_path, 3600.0)
            except FileNotFoundError:
                stale = False
            if stale:
                lock_path.unlink(missing_ok=True)
                continue
            time.sleep(0.25)
    try:
        if cache_path.is_file():
            payload = load_checkpoint(cache_path, "cpu")
            return for_requested_ipc(PixelClusterIndex.from_state_dict(payload)), cache_path
        settings = config["condensation"]["cluster_matching"]
        normalization = config["data"]["image"]["normalization"]
        maximum = 64 if bool(config.get("_smoke", False)) else None
        descriptor_mode = str(
            settings.get("descriptor_mode", "pixel_pca")
        ).strip().lower()
        normalization_token = hashlib.sha256(
            repr(
                (
                    tuple(map(float, normalization["mean"])),
                    tuple(map(float, normalization["std"])),
                )
            ).encode("ascii")
        ).hexdigest()[:12]
        descriptor_cache_path = (
            cache_path.parent
            / (
                f"{config['_runtime']['dataset']}_n{len(pool.dataset)}_"
                f"mode{descriptor_mode}_norm{normalization_token}_"
                f"max{maximum if maximum is not None else 'all'}_"
                "deep_descriptors_v1.npz"
            )
            if descriptor_mode != "pixel_pca"
            else None
        )
        index = build_pixel_cluster_index(
            pool.dataset,
            pool.class_indices,
            clusters_per_class=cluster_counts,
            descriptor_size=settings["descriptor_size"],
            pca_components=int(settings["pca_components"]),
            kmeans_n_init=int(settings["kmeans_n_init"]),
            normalization_mean=normalization["mean"],
            normalization_std=normalization["std"],
            seed=int(seed),
            maximum_samples_per_class=maximum,
            descriptor_mode=descriptor_mode,
            descriptor_device=device,
            descriptor_batch_size=int(
                settings.get("descriptor_batch_size", 128)
            ),
            descriptor_cache_path=descriptor_cache_path,
        )
        atomic_torch_save(index.state_dict(), cache_path)
        return for_requested_ipc(index), cache_path
    finally:
        if owns_lock:
            lock_path.unlink(missing_ok=True)


def _cluster_layouts(
    cluster_settings: Mapping[str, Any],
    ipc: int,
    class_indices: Mapping[int, Sequence[int]],
) -> dict[int, int]:
    """Resolve train-only per-class K while preserving exact IPC per class."""

    maximum = min(
        int(ipc),
        int(cluster_settings.get("max_clusters_per_class", int(ipc))),
    )
    if maximum <= 0:
        raise ValueError("max_clusters_per_class must be positive")
    mode = str(cluster_settings.get("cluster_count_mode", "fixed")).strip().lower()
    if mode == "class_support_round":
        support = int(cluster_settings.get("images_per_cluster", 100))
        if support <= 0:
            raise ValueError("images_per_cluster must be positive")
        # Explicit half-up rounding avoids Python's bankers-rounding ambiguity.
        layouts = {
            int(class_id): min(
                maximum,
                max(1, int(math.floor(len(indices) / support + 0.5))),
            )
            for class_id, indices in class_indices.items()
        }
    elif mode == "fixed":
        schedule = cluster_settings.get("layout_by_ipc", {})
        configured = schedule.get(int(ipc), schedule.get(str(int(ipc))))
        fixed = (
            int(configured["clusters"])
            if configured is not None
            else maximum
        )
        layouts = {int(class_id): fixed for class_id in class_indices}
    else:
        raise ValueError(f"Unsupported cluster_count_mode: {mode}")
    if any(count <= 0 or count > int(ipc) for count in layouts.values()):
        raise ValueError("Every class cluster count must be in [1, IPC]")
    return layouts


def _stored_cluster_groups(
    allocations: list[int], device: torch.device
) -> torch.Tensor:
    counts = torch.tensor(allocations, device=device, dtype=torch.long)
    return torch.arange(
        len(allocations), device=device, dtype=torch.long
    ).repeat_interleave(counts)


def _initialize_synthetic_pixels(
    config: Mapping[str, Any],
    pool: IndexedImagePool,
    cluster_index: PixelClusterIndex | None,
    num_classes: int,
    ipc: int,
    partition_factor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cluster_settings = config["condensation"]["cluster_matching"]
    cluster_initialization = bool(
        cluster_settings.get("initialization_enabled", False)
    )
    if not cluster_initialization:
        return initialize_partitioned_pixels(
            pool,
            int(num_classes),
            config["data"]["image"]["size"],
            ipc=int(ipc),
            factor=int(partition_factor),
        )
    if cluster_initialization and cluster_index is None:
        raise ValueError("Cluster initialization requires a cluster index")

    sources_per_canvas = int(partition_factor) ** 2
    canvases: list[torch.Tensor] = []
    labels: list[int] = []
    for class_id in range(int(num_classes)):
        if cluster_initialization:
            assert cluster_index is not None
            source_groups, _ = cluster_index.canvas_representatives(
                class_id, int(ipc), sources_per_canvas
            )
            if len(source_groups) != int(ipc):
                raise RuntimeError("The cluster layout does not match IPC")
        else:
            random_images = pool.sample(
                class_id, int(ipc) * sources_per_canvas
            )
            source_groups = [
                random_images[
                    index * sources_per_canvas : (index + 1) * sources_per_canvas
                ]
                for index in range(int(ipc))
            ]
        for source_group in source_groups:
            sources = (
                pool.images(source_group)
                if isinstance(source_group, list)
                else source_group
            )
            canvases.append(
                compose_storage_canvas(
                    sources,
                    config["data"]["image"]["size"],
                    int(partition_factor),
                )
            )
            labels.append(int(class_id))
    return torch.stack(canvases), torch.tensor(labels, dtype=torch.long)


def _sample_clustered_real_batch(
    pool: IndexedImagePool,
    sampler: ClusterSampler,
    class_id: int,
    cluster_count: int,
    total_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    minimum_per_cluster = 2
    required = int(cluster_count) * minimum_per_cluster
    if int(total_count) < required:
        raise ValueError(
            "batch_real must provide at least two samples per IPC cluster"
        )
    sizes = torch.tensor(
        sampler.index.sizes[int(class_id)], dtype=torch.float64
    )
    remaining = int(total_count) - required
    ideal = sizes / sizes.sum() * float(remaining)
    extras = ideal.floor().to(torch.long)
    leftover = remaining - int(extras.sum().item())
    if leftover > 0:
        order = (ideal - extras).argsort(descending=True)
        extras[order[:leftover]] += 1
    counts = extras + minimum_per_cluster
    indices: list[int] = []
    groups: list[int] = []
    for cluster_id in range(int(cluster_count)):
        count = int(counts[cluster_id].item())
        selected = sampler.take(class_id, cluster_id, count)
        indices.extend(selected)
        groups.extend([cluster_id] * count)
    return pool.images(indices), torch.tensor(groups, dtype=torch.long)


def _set_model_trainable(model: nn.Module, enabled: bool) -> None:
    model.train(bool(enabled))
    for parameter in model.parameters():
        parameter.requires_grad_(bool(enabled))


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _build_synthetic_optimizer(
    parameter: nn.Parameter,
    settings: Mapping[str, Any],
    ipc: int,
) -> tuple[torch.optim.Optimizer, str, float]:
    learning_rate = float(
        _idm_value(settings["idm"], "image_learning_rate", int(ipc))
    )
    optimizer = torch.optim.SGD(
        [parameter],
        lr=learning_rate,
        momentum=float(settings["idm"]["image_momentum"]),
    )
    return optimizer, "sgd", learning_rate


def _normalization_tensors(
    config: Mapping[str, Any],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalization = config["data"]["image"]["normalization"]
    mean = torch.tensor(
        normalization["mean"], dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)
    std = torch.tensor(
        normalization["std"], dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)
    return mean, std


def _display_images(
    config: Mapping[str, Any], images: torch.Tensor
) -> torch.Tensor:
    mean, std = _normalization_tensors(config, images.device)
    return (images.detach().float() * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def _pixel_statistics(
    images: torch.Tensor,
    config: Mapping[str, Any],
) -> dict[str, float]:
    values = _display_images(config, images)
    keys = (
        "pixel/minimum",
        "pixel/maximum",
        "pixel/mean",
        "pixel/std",
        "pixel/below_zero_fraction",
        "pixel/above_one_fraction",
        "pixel/at_zero_fraction",
        "pixel/at_one_fraction",
    )
    statistics_tensor = torch.stack(
        (
            values.min(),
            values.max(),
            values.mean(),
            values.std(unbiased=False),
            (values < 0.0).float().mean(),
            (values > 1.0).float().mean(),
            (values == 0.0).float().mean(),
            (values == 1.0).float().mean(),
        )
    )
    return dict(
        zip(keys, statistics_tensor.detach().cpu().tolist(), strict=True)
    )


@torch.no_grad()
def _project_synthetic_pixels(
    parameter: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    collect_metrics: bool = True,
) -> dict[str, float] | None:
    """投影到真实像素域，并清除仍会把边界像素向外推的 SGD 动量。"""

    settings = config["condensation"]["pixel_constraints"]
    before = parameter.detach()
    mean, std = _normalization_tensors(config, before.device)
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std
    below = before < lower
    above = before > upper
    projected = below | above
    result = None
    if collect_metrics:
        pre_minimum, pre_maximum, projected_fraction = torch.stack(
            (before.min(), before.max(), projected.float().mean())
        ).detach().cpu().tolist()
        result = {
            "pixel/pre_projection_minimum": pre_minimum,
            "pixel/pre_projection_maximum": pre_maximum,
            "pixel/projected_fraction": projected_fraction,
            "pixel/momentum_cleared_fraction": 0.0,
        }
    parameter.copy_(torch.maximum(torch.minimum(parameter, upper), lower))

    if bool(settings.get("clear_outward_momentum", True)):
        state = optimizer.state.get(parameter, {})
        momentum = state.get("momentum_buffer")
        if torch.is_tensor(momentum):
            outward = ((parameter <= lower) & (momentum > 0.0)) | (
                (parameter >= upper) & (momentum < 0.0)
            )
            if result is not None:
                result["pixel/momentum_cleared_fraction"] = float(
                    outward.float().mean().item()
                )
            momentum.masked_fill_(outward, 0.0)

    if result is not None:
        result.update(_pixel_statistics(parameter, config))
    return result


@dataclass
class QueueMember:
    identifier: int
    model: IDMConvNet
    optimizer: torch.optim.Optimizer
    birth_iteration: int
    update_steps: int = 0
    correct: torch.Tensor | None = None
    count: torch.Tensor | None = None

    def reliability_percent_tensor(self) -> torch.Tensor:
        """Official cumulative accuracy in percent without a device sync."""

        if self.correct is None or self.count is None:
            return torch.zeros((), device=next(self.model.parameters()).device)
        return torch.where(
            self.count > 0,
            self.correct.double() / self.count.double() * 100.0,
            torch.zeros((), dtype=torch.float64, device=self.correct.device),
        )


class OfficialIDMQueue:
    """官方 ImageNet IPC=1 的 3→50 动态同构模型池。"""

    VERSION = 1

    def __init__(
        self,
        config: Mapping[str, Any],
        num_classes: int,
        device: torch.device,
        seed: int,
    ):
        self.config = config
        self.settings = config["condensation"]
        self.num_classes = int(num_classes)
        self.device = device
        self.random = random.Random(int(seed))
        self.members: list[QueueMember] = []
        self.next_identifier = 0
        for _ in range(3):
            self.members.append(self._new_member(0))

    def _new_member(self, birth_iteration: int) -> QueueMember:
        model = build_idm_convnet(
            int(self.config["data"]["image"]["channels"]),
            self.num_classes,
            self.config["data"]["image"]["size"],
            depth=int(self.settings["idm"].get("network_depth", 6)),
        ).to(self.device)
        settings = self.settings["idm"]
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(settings["lr_net"]),
            momentum=float(settings["net_momentum"]),
            weight_decay=float(settings["net_weight_decay"]),
        )
        member = QueueMember(
            identifier=int(self.next_identifier),
            model=model,
            optimizer=optimizer,
            birth_iteration=int(birth_iteration),
            correct=torch.zeros((), dtype=torch.long, device=self.device),
            count=torch.zeros((), dtype=torch.long, device=self.device),
        )
        self.next_identifier += 1
        _set_model_trainable(model, False)
        return member

    def _value(self, key: str) -> float | int:
        settings = self.settings["idm"]
        return _idm_value(settings, key, int(settings["ipc"]))

    def grow(self, iteration: int) -> None:
        interval = int(self._value("net_generate_interval"))
        if int(iteration) % interval != 0:
            return
        maximum = int(self._value("net_num"))
        if len(self.members) == maximum:
            # 引用释放后 PyTorch 缓存分配器会把显存直接复用于新模型。
            # 此处强制 synchronize/empty_cache 会每 30 轮让 GPU 停一次。
            del self.members[0]
        self.members.append(self._new_member(int(iteration)))

    def reset_reliability_if_due(self, iteration: int) -> bool:
        """Mirror the official accuracy-meter reset at evaluation intervals."""

        interval = int(self._value("reliability_reset_interval"))
        if int(iteration) <= 0 or int(iteration) % interval != 0:
            return False
        for member in self.members:
            assert member.correct is not None and member.count is not None
            member.correct.zero_()
            member.count.zero_()
        return True

    def guidance_members(self) -> list[QueueMember]:
        settings = self.settings["idm"]
        count = min(int(settings["train_net_num"]), len(self.members))
        indices = list(range(len(self.members)))
        self.random.shuffle(indices)
        selected = [self.members[index] for index in indices[:count]]
        for member in selected:
            _set_model_trainable(member.model, False)
        return selected

    def training_members(self) -> list[QueueMember]:
        indices = list(range(len(self.members)))
        self.random.shuffle(indices)
        count = min(int(self.settings["idm"]["fetch_net_num"]), len(indices))
        return [self.members[index] for index in indices[:count]]

    def _train_member(
        self,
        member: QueueMember,
        pool: IndexedImagePool,
        collect_metrics: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        settings = self.settings["idm"]
        target_batch = int(settings["batch_train"])
        microbatch = int(self.settings["memory"]["real_train_microbatch"])
        minimum = int(self.settings["memory"].get("retry_minimum", 1))
        loss_total_tensor = (
            torch.zeros((), device=self.device) if collect_metrics else None
        )
        correct_total_tensor = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        count_total = 0
        _set_model_trainable(member.model, True)
        for _ in range(int(settings["model_train_steps"])):
            images_cpu, labels_cpu = pool.sample_general(target_batch)
            while True:
                member.optimizer.zero_grad(set_to_none=True)
                batch_loss = (
                    torch.zeros((), device=self.device)
                    if collect_metrics else None
                )
                batch_correct = torch.zeros(
                    (), dtype=torch.long, device=self.device
                )
                try:
                    for start in range(0, target_batch, microbatch):
                        images = images_cpu[start : start + microbatch].to(
                            self.device, non_blocking=True
                        )
                        labels = labels_cpu[start : start + microbatch].to(
                            self.device, non_blocking=True
                        )
                        logits = member.model(images)
                        # 累加后等价于一次有效 batch=target_batch 的平均交叉熵。
                        loss = F.cross_entropy(
                            logits, labels, reduction="sum"
                        ) / float(target_batch)
                        loss.backward()
                        if batch_loss is not None:
                            batch_loss += loss.detach()
                        batch_correct += (
                            logits.detach().argmax(1) == labels
                        ).sum()
                        del images, labels, logits, loss
                    member.optimizer.step()
                    break
                except torch.OutOfMemoryError:
                    member.optimizer.zero_grad(set_to_none=True)
                    cleanup_memory()
                    if microbatch <= minimum:
                        raise
                    microbatch = max(minimum, microbatch // 2)
            member.update_steps += 1
            if loss_total_tensor is not None and batch_loss is not None:
                loss_total_tensor += batch_loss
            correct_total_tensor += batch_correct
            count_total += int(target_batch)
        assert member.correct is not None and member.count is not None
        member.correct.add_(correct_total_tensor)
        member.count.add_(int(count_total))
        _set_model_trainable(member.model, False)
        if loss_total_tensor is None:
            return None
        return (
            loss_total_tensor / max(1, int(settings["model_train_steps"])),
            correct_total_tensor.double() / max(1, count_total),
            microbatch,
        )

    def train_independent_members(
        self,
        pool: IndexedImagePool,
        collect_metrics: bool,
    ) -> dict[str, float]:
        losses: list[torch.Tensor] = []
        accuracies: list[torch.Tensor] = []
        microbatches: list[int] = []
        selected = self.training_members()
        for member in selected:
            result = self._train_member(member, pool, collect_metrics)
            if result is None:
                continue
            loss, accuracy, microbatch = result
            losses.append(loss)
            accuracies.append(accuracy)
            microbatches.append(microbatch)
        if not collect_metrics:
            return {}
        reliabilities = torch.stack(
            [member.reliability_percent_tensor() for member in self.members]
        )
        values = torch.stack(
            (
                torch.stack(losses).double().mean()
                if losses else torch.zeros((), device=self.device, dtype=torch.double),
                torch.stack(accuracies).double().mean()
                if accuracies else torch.zeros((), device=self.device, dtype=torch.double),
                reliabilities.double().mean(),
            )
        ).detach().cpu().tolist()
        return {
            "queue/train_loss": float(values[0]),
            "queue/train_accuracy": float(values[1]),
            "queue/train_microbatch": float(min(microbatches))
            if microbatches
            else 0.0,
            "queue/size": float(len(self.members)),
            "queue/mean_updates": statistics.fmean(
                [member.update_steps for member in self.members]
            ),
            "queue/mean_reliability_percent": float(values[2]),
        }

    def state_dict(self) -> dict[str, Any]:
        counter_values = torch.stack(
            [
                torch.stack(
                    (
                        member.correct
                        if member.correct is not None
                        else torch.zeros((), dtype=torch.long, device=self.device),
                        member.count
                        if member.count is not None
                        else torch.zeros((), dtype=torch.long, device=self.device),
                    )
                )
                for member in self.members
            ]
        ).detach().cpu().tolist()
        return {
            "version": self.VERSION,
            "next_identifier": int(self.next_identifier),
            "random_state": self.random.getstate(),
            "members": [
                {
                    "identifier": int(member.identifier),
                    "model": member.model.state_dict(),
                    "optimizer": member.optimizer.state_dict(),
                    "birth_iteration": int(member.birth_iteration),
                    "update_steps": int(member.update_steps),
                    "correct": int(counters[0]),
                    "count": int(counters[1]),
                }
                for member, counters in zip(
                    self.members, counter_values, strict=True
                )
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        if int(state.get("version", -1)) != self.VERSION:
            raise ValueError("IDM 队列断点版本不兼容")
        restored: list[QueueMember] = []
        for item in state["members"]:
            member = self._new_member(int(item["birth_iteration"]))
            member.identifier = int(item["identifier"])
            member.model.load_state_dict(item["model"], strict=True)
            member.optimizer.load_state_dict(item["optimizer"])
            _optimizer_to_device(member.optimizer, self.device)
            settings = self.settings["idm"]
            for group in member.optimizer.param_groups:
                # 断点不校验配置哈希；恢复状态后仍以当前 YAML 的超参数为准。
                group["lr"] = float(settings["lr_net"])
                group["momentum"] = float(settings["net_momentum"])
                group["weight_decay"] = float(settings["net_weight_decay"])
            member.update_steps = int(item.get("update_steps", 0))
            member.correct = torch.tensor(
                int(item.get("correct", 0)),
                dtype=torch.long,
                device=self.device,
            )
            member.count = torch.tensor(
                int(item.get("count", 0)),
                dtype=torch.long,
                device=self.device,
            )
            _set_model_trainable(member.model, False)
            restored.append(member)
        self.members = restored
        self.next_identifier = max(
            int(state.get("next_identifier", 0)),
            max(member.identifier for member in restored) + 1,
        )
        if state.get("random_state") is not None:
            self.random.setstate(state["random_state"])


class SyntheticParameterization(nn.Module):
    """直接优化像素的每类渲染接口。"""

    def __init__(
        self,
        initial_pixels: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ):
        super().__init__()
        self.mode = "pixel"
        self.num_classes = int(labels.max().item()) + 1
        self.register_buffer("labels", labels.to(device))
        self.variable = nn.Parameter(
            initial_pixels.to(device).detach().float()
        )
        self._class_ranges: list[tuple[int, int]] = []
        cpu_labels = labels.detach().long().cpu()
        for class_id in range(self.num_classes):
            positions = torch.nonzero(
                cpu_labels == class_id, as_tuple=False
            ).flatten()
            if positions.numel() == 0:
                raise ValueError(f"Synthetic labels miss class {class_id}")
            start = int(positions[0])
            stop = int(positions[-1]) + 1
            if stop - start != int(positions.numel()):
                raise ValueError("Synthetic labels must be contiguous by class")
            self._class_ranges.append((start, stop))

    def class_slice(self, class_id: int) -> slice:
        start, stop = self._class_ranges[int(class_id)]
        return slice(start, stop)

    def render_class(self, class_id: int) -> torch.Tensor:
        return self.variable[self.class_slice(class_id)]

    @torch.no_grad()
    def render_all(self) -> torch.Tensor:
        return torch.cat(
            [self.render_class(class_id) for class_id in range(self.num_classes)]
        ).float()

    def prior_loss(self, class_id: int) -> torch.Tensor:
        del class_id
        return self.variable.sum() * 0.0


class AuxiliaryGradientMixer:
    """Budget an auxiliary gradient and remove components opposing the base."""

    _HISTORY_KEYS = (
        "base_norms",
        "auxiliary_norms",
        "gradient_scales",
        "achieved_fractions",
        "gradient_cosines",
        "conflict_projections",
    )

    def __init__(self, settings: Mapping[str, Any]):
        self.settings = dict(settings)
        self.base_norms: list[float] = []
        self.auxiliary_norms: list[float] = []
        self.gradient_scales: list[float] = []
        self.achieved_fractions: list[float] = []
        self.gradient_cosines: list[float] = []
        self.conflict_projections: list[float] = []
        self.skipped_gradients = 0
        self.mixed_gradients = 0
        self._pending_skipped: torch.Tensor | None = None
        self._pending_mixed: torch.Tensor | None = None

    def target_fraction(self) -> float:
        """Return the fixed auxiliary gradient budget."""

        return float(self.settings["target_gradient_fraction"])

    def _append(self, name: str, value: float | torch.Tensor) -> None:
        values = getattr(self, name)
        values.append(
            value.detach().reshape(())
            if torch.is_tensor(value)
            else float(value)
        )
        history_size = max(1, int(self.settings.get("history_size", 256)))
        if len(values) > history_size:
            del values[:-history_size]

    def _pending_increment(self, name: str, value: torch.Tensor) -> None:
        attribute = f"_pending_{name}"
        current = getattr(self, attribute)
        detached = value.detach().to(dtype=torch.int64).reshape(())
        setattr(
            self,
            attribute,
            detached if current is None else current + detached,
        )

    def materialize_history(self) -> None:
        """把延迟的 GPU 标量一次性转到 CPU，避免每簇同步。"""

        tensors: list[torch.Tensor] = []
        slots: list[tuple[str, str, int | None]] = []
        for key in self._HISTORY_KEYS:
            values = getattr(self, key)
            for index, value in enumerate(values):
                if torch.is_tensor(value):
                    tensors.append(value.detach().double())
                    slots.append(("history", key, index))
        for key in ("skipped", "mixed"):
            value = getattr(self, f"_pending_{key}")
            if value is not None:
                tensors.append(value.detach().double())
                slots.append(("counter", key, None))
        if not tensors:
            return
        transferred = torch.stack(tensors).cpu().tolist()
        for (kind, key, index), value in zip(slots, transferred, strict=True):
            if kind == "history":
                getattr(self, key)[int(index)] = float(value)
            elif key == "skipped":
                self.skipped_gradients += int(round(float(value)))
            else:
                self.mixed_gradients += int(round(float(value)))
        self._pending_skipped = None
        self._pending_mixed = None

    def weight_against_base(
        self,
        base_gradient: torch.Tensor,
        auxiliary_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, float | torch.Tensor]:
        """Combine the auxiliary spread gradient with the main gradient."""

        base = base_gradient.float()
        auxiliary = auxiliary_gradient.float()
        base_norm_tensor = base.norm()
        auxiliary_norm_tensor = auxiliary.norm()
        finite_pair = torch.isfinite(base).all() & torch.isfinite(auxiliary).all()
        dot_tensor = (base.detach() * auxiliary.detach()).sum()
        self._append("base_norms", base_norm_tensor)
        self._append("auxiliary_norms", auxiliary_norm_tensor)
        epsilon = float(self.settings.get("epsilon", 1.0e-12))
        denominator = (
            base_norm_tensor.double() * auxiliary_norm_tensor.double()
        ).clamp_min(epsilon)
        cosine_tensor = dot_tensor.double() / denominator
        self._append("gradient_cosines", cosine_tensor)
        mode = str(self.settings.get("mixer_mode", "full")).strip().lower()
        if mode not in {"naive", "projection", "full"}:
            raise ValueError(f"Unsupported cluster spread mixer mode: {mode}")
        projection_enabled = mode in {"projection", "full"}
        projection_mask = (
            (cosine_tensor < 0.0)
            & (base_norm_tensor.double() > epsilon)
            & finite_pair
            & torch.tensor(
                projection_enabled, device=base.device, dtype=torch.bool
            )
        )
        projection = (auxiliary * base).sum() / base.square().sum().clamp_min(
            epsilon
        )
        projected = auxiliary - projection * base
        auxiliary = torch.where(projection_mask, projected, auxiliary)
        auxiliary_norm_tensor = auxiliary.norm()
        self._append("conflict_projections", projection_mask.float())
        if mode in {"naive", "projection"}:
            usable = finite_pair & torch.isfinite(auxiliary_norm_tensor)
            scale = usable.to(dtype=torch.float64)
            self._append("gradient_scales", scale)
            achieved = auxiliary_norm_tensor.double() / (
                base_norm_tensor.double() + auxiliary_norm_tensor.double()
            ).clamp_min(epsilon)
            self._append(
                "achieved_fractions",
                torch.where(usable, achieved, torch.zeros_like(achieved)),
            )
            self._pending_increment("mixed", usable)
            self._pending_increment("skipped", ~usable)
            weighted = torch.where(
                usable,
                auxiliary,
                torch.zeros_like(auxiliary),
            )
            return weighted.to(auxiliary_gradient.dtype), scale.detach()
        target = self.target_fraction()
        base_fraction = 1.0 - target
        relative_norm = auxiliary_norm_tensor.double() / (
            base_norm_tensor.double().clamp_min(epsilon)
        )
        usable = (
            torch.tensor(
                target > 0.0 and float(base_fraction) > epsilon,
                device=base.device,
                dtype=torch.bool,
            )
            & torch.isfinite(base_norm_tensor)
            & torch.isfinite(auxiliary_norm_tensor)
            & finite_pair
            & (base_norm_tensor.double() > epsilon)
            & (auxiliary_norm_tensor.double() > epsilon)
            & (relative_norm >= float(
                self.settings.get(
                    "minimum_relative_gradient_norm", 0.0
                )
            ))
        )
        raw_scale = (
            target
            / float(base_fraction)
            * base_norm_tensor.double()
            / auxiliary_norm_tensor.double().clamp_min(epsilon)
        )
        scale = torch.where(usable, raw_scale, torch.zeros_like(raw_scale))
        weighted = torch.where(
            usable,
            auxiliary.to(auxiliary_gradient.dtype)
            * scale.to(auxiliary_gradient.dtype),
            torch.zeros_like(auxiliary_gradient),
        )
        self._append("gradient_scales", scale)
        self._append(
            "achieved_fractions",
            torch.where(
                usable,
                torch.full_like(scale, target),
                torch.zeros_like(scale),
            ),
        )
        self._pending_increment("mixed", usable)
        self._pending_increment("skipped", ~usable)
        return weighted, scale.detach()

    def recent_mean(self, name: str, count: int) -> float:
        self.materialize_history()
        values = getattr(self, name)
        recent = values[-max(1, int(count)) :]
        return statistics.fmean(recent) if recent else 0.0

    def state_dict(self) -> dict[str, Any]:
        self.materialize_history()
        return {
            **{key: list(getattr(self, key)) for key in self._HISTORY_KEYS},
            "skipped_gradients": int(self.skipped_gradients),
            "mixed_gradients": int(self.mixed_gradients),
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        for key in self._HISTORY_KEYS:
            setattr(
                self,
                key,
                [float(value) for value in state.get(key, [])],
            )
        self.skipped_gradients = int(state.get("skipped_gradients", 0))
        self.mixed_gradients = int(state.get("mixed_gradients", 0))
        self._pending_skipped = None
        self._pending_mixed = None


def _real_features(
    config: Mapping[str, Any],
    model: IDMConvNet,
    images_cpu: torch.Tensor,
    dsa_seed: int,
    device: torch.device,
    strategy: str,
    augmentation: ParamDiffAug,
) -> tuple[torch.Tensor, int]:
    """Extract all real embeddings with OOM-safe microbatching."""

    condensation = config["condensation"]
    requested = int(condensation["memory"]["real_feature_microbatch"])
    minimum = int(condensation["memory"].get("retry_minimum", 1))
    while True:
        embedding_parts: list[torch.Tensor] = []
        try:
            with torch.no_grad():
                for start in range(0, images_cpu.shape[0], requested):
                    images = images_cpu[start : start + requested].to(
                        device, non_blocking=True
                    )
                    images = diff_augment(
                        images,
                        strategy,
                        seed=int(dsa_seed),
                        param=augmentation,
                    )
                    output = model.forward_idm(images)
                    embedding_parts.append(output.embedding.float().detach())
                    del images, output
            break
        except torch.OutOfMemoryError:
            cleanup_memory()
            if requested <= minimum:
                raise
            requested = max(minimum, requested // 2)
    if not embedding_parts:
        raise ValueError("Real feature batch is empty")
    return torch.cat(embedding_parts, dim=0), requested


def _synthetic_features(
    model: IDMConvNet,
    images: torch.Tensor,
    microbatch: int,
) -> IDMFeatures:
    """Run the exact full synthetic objective with bounded activation memory.

    P&E and DiffAug are applied to the complete class batch before this helper.
    Only the IDM network forward is split. Checkpointing recomputes one chunk
    during autograd, while concatenating every logit and embedding preserves the
    original full-batch CE, feature means, cluster means, and spread losses.
    """

    requested = int(microbatch)
    if requested <= 0:
        raise ValueError("synthetic_feature_microbatch 必须为正整数")
    if int(images.shape[0]) <= requested:
        return model.forward_idm(images)

    logits_parts: list[torch.Tensor] = []
    embedding_parts: list[torch.Tensor] = []

    def forward_chunk(chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = model.forward_idm(chunk)
        return output.logits, output.embedding

    for chunk in images.split(requested, dim=0):
        logits, embedding = checkpoint(
            forward_chunk,
            chunk,
            use_reentrant=False,
            preserve_rng_state=True,
        )
        logits_parts.append(logits)
        embedding_parts.append(embedding)
    return IDMFeatures(
        logits=torch.cat(logits_parts, dim=0),
        embedding=torch.cat(embedding_parts, dim=0),
    )


def _online_classifier_step(
    config: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    microbatch: int,
    device: torch.device,
) -> None:
    """Accumulate one logical batch without retaining all image activations.

    Augmentation must already have run once on the complete logical batch.
    The online IDM ConvNet uses GroupNorm, not batch-dependent BatchNorm.
    Keep one optimizer update and weight a short last chunk by its sample count.
    """

    count = int(labels.numel())
    if count < 1 or int(microbatch) < 1:
        raise ValueError("Online evaluation requires a positive batch/microbatch")
    optimizer.zero_grad(set_to_none=True)
    for start in range(0, count, int(microbatch)):
        stop = min(start + int(microbatch), count)
        with autocast_context(config, device):
            logits = model(images[start:stop])
            loss = F.cross_entropy(logits.float(), labels[start:stop])
            loss = loss * ((stop - start) / count)
        loss.backward()
        del logits, loss
    optimizer.step()


def _quick_evaluate(
    config: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    evaluation_dataset,
    num_classes: int,
    iteration: int,
    seed: int,
    ipc: int,
    partition_factor: int,
    device: torch.device,
    logger=None,
) -> dict[str, float]:
    """Train a fresh ConvNet for full epochs and score the validation split."""

    settings = config["condensation"]["online_evaluation"]
    rng_state = capture_rng_state()
    started = time.perf_counter()
    model = None
    optimizer = None
    loader = None
    train_loader = None
    try:
        seed_everything(
            int(seed), bool(config["project"].get("deterministic", False))
        )
        expanded_images, expanded_labels, _ = partition_training_images(
            images.detach().cpu(),
            labels.detach().cpu(),
            partition_factor=int(partition_factor),
        )
        model = build_idm_convnet(
            int(config["data"]["image"]["channels"]),
            int(num_classes),
            config["data"]["image"]["size"],
            depth=int(config["condensation"]["idm"]["network_depth"]),
        ).to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(settings["learning_rate"]),
            momentum=float(settings.get("momentum", 0.9)),
            weight_decay=float(settings.get("weight_decay", 0.0005)),
        )
        batch_size = min(
            int(settings["batch_size"]), int(expanded_labels.numel())
        )
        microbatch = min(batch_size, int(settings.get("train_microbatch", batch_size)))
        if microbatch < 1:
            raise ValueError("online_evaluation.train_microbatch must be positive")
        train_dataset = TensorImageDataset(expanded_images, expanded_labels)
        train_loader = build_loader(
            train_dataset,
            config,
            train=True,
            batch_size=batch_size,
            num_workers=0,
        )
        epochs = int(_idm_value(settings, "epochs", int(ipc)))
        training_updates = epochs + int(
            bool(settings.get("include_epoch_zero_update", True))
        )
        initial_learning_rate = float(settings["learning_rate"])
        augmentation = ParamDiffAug()
        optimizer_steps = 0
        log_interval = max(1, int(settings.get("log_interval_epochs", 100)))
        if logger is not None:
            logger.info(
                "online evaluation start: iteration=%d images=%d logical_batch=%d "
                "train_microbatch=%d inference_batch=%d epoch_passes=%d",
                iteration, len(train_dataset), batch_size, microbatch,
                int(settings["evaluation_batch_size"]), training_updates,
            )
        for update_index in range(training_updates):
            model.train()
            for batch in train_loader:
                batch_images, batch_labels = unpack_batch(batch)
                batch_images = batch_images.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                batch_images = diff_augment(
                    batch_images,
                    str(config["condensation"]["idm"]["dsa_strategy"]),
                    param=augmentation,
                )
                _online_classifier_step(
                    config, model, optimizer, batch_images, batch_labels,
                    microbatch, device,
                )
                optimizer_steps += 1
                del batch_images, batch_labels
            if logger is not None and (
                update_index == 0
                or (update_index + 1) % log_interval == 0
                or update_index + 1 == training_updates
            ):
                logger.info(
                    "online evaluation progress: iteration=%d epoch_pass=%d/%d "
                    "optimizer_steps=%d peak=%.0fMiB elapsed=%.1fs",
                    iteration, update_index + 1, training_updates, optimizer_steps,
                    cuda_peak_megabytes(), time.perf_counter() - started,
                )
            if update_index == epochs // 2 + 1:
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=initial_learning_rate * 0.1,
                    momentum=float(settings.get("momentum", 0.9)),
                    weight_decay=float(settings.get("weight_decay", 0.0005)),
                )
        loader = build_loader(
            evaluation_dataset,
            config,
            train=False,
            batch_size=int(settings["evaluation_batch_size"]),
        )
        model.eval()
        correct = 0
        count = 0
        total_loss = 0.0
        metric_parts: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch_images, batch_labels = unpack_batch(batch)
                batch_images = batch_images.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)
                with autocast_context(config, device):
                    logits = model(batch_images)
                    loss = F.cross_entropy(
                        logits.float(), batch_labels, reduction="sum"
                    )
                metric_parts.append(
                    torch.stack(
                        (
                            loss.detach().double(),
                            (logits.argmax(1) == batch_labels)
                            .sum()
                            .double(),
                        )
                    )
                )
                count += int(batch_labels.numel())
        if metric_parts:
            # Preserve the original Python summation order, but transfer all
            # per-batch scalars from GPU in one operation.
            for loss_value, correct_value in (
                torch.stack(metric_parts).cpu().tolist()
            ):
                total_loss += float(loss_value)
                correct += int(correct_value)
        return {
            "iteration": int(iteration),
            "accuracy": correct / max(1, count),
            "loss": total_loss / max(1, count),
            "epochs": epochs,
            "training_updates": training_updates,
            "optimizer_steps": optimizer_steps,
            "logical_batch_size": batch_size,
            "train_microbatch": microbatch,
            "seconds": time.perf_counter() - started,
        }
    finally:
        del model, optimizer, loader, train_loader
        cleanup_memory()
        restore_rng_state(rng_state)


def _save_checkpoint(
    directory: Path,
    experiment: str,
    iteration: int,
    parameterization: SyntheticParameterization,
    optimizer: torch.optim.Optimizer,
    queue: OfficialIDMQueue,
    pool: IndexedImagePool,
    mixer: AuxiliaryGradientMixer,
    cluster_sampler: ClusterSampler | None,
    dream_sampler: DreamRepresentativeSampler | None,
    design_signature: Mapping[str, Any],
    pixel_constraints: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    algorithm_version: int,
) -> Path:
    payload = {
        "algorithm_version": int(algorithm_version),
        "experiment": str(experiment),
        "iteration": int(iteration),
        "parameterization": parameterization.state_dict(),
        "optimizer_name": optimizer.__class__.__name__.lower(),
        "optimizer": optimizer.state_dict(),
        "queue": queue.state_dict(),
        "real_pool": pool.state_dict(),
        "auxiliary_mixer": mixer.state_dict(),
        "cluster_sampler": (
            cluster_sampler.state_dict() if cluster_sampler is not None else None
        ),
        "dream_representative_sampler": (
            dream_sampler.state_dict() if dream_sampler is not None else None
        ),
        "design_signature": dict(design_signature),
        "pixel_constraints": dict(pixel_constraints),
        "diagnostics": dict(diagnostics),
        "rng_state": capture_rng_state(),
    }
    return atomic_torch_save(payload, directory / "checkpoint_last.pt")


def _formation_metadata(
    ipc: int, partition_factor: int
) -> dict[str, Any]:
    return {
        "stored_ipc": int(ipc),
        "effective_samples_per_class": int(
            ipc * int(partition_factor) ** 2
        ),
    }


def _export(
    config: Mapping[str, Any],
    directory: Path,
    experiment: str,
    parameterization: SyntheticParameterization,
    class_names: list[str],
    iteration: int,
    ipc: int,
    partition_factor: int,
    optimization_iterations: int | None = None,
    selection: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically overwrite the single best synthetic dataset."""

    images = parameterization.render_all().detach().cpu()
    labels = parameterization.labels.detach().cpu()
    path = directory / "synthetic.pt"
    atomic_torch_save(
        {
            "images": images,
            "labels": labels,
            "class_names": list(class_names),
            "ipc": int(ipc),
            "partition_expansion": int(partition_factor),
            **_formation_metadata(ipc, partition_factor),
            "method": str(experiment),
            "algorithm_version": _algorithm_version(config["condensation"]),
            "parameterization": parameterization.mode,
            "iteration": int(iteration),
            "optimization_iterations": int(
                optimization_iterations
                if optimization_iterations is not None
                else iteration
            ),
            "selection": dict(selection) if selection is not None else None,
            "normalization": dict(
                config["data"]["image"]["normalization"]
            ),
            "pixel_range": [
                float(images.min().item()),
                float(images.max().item()),
            ],
        },
        path,
    )
    del images, labels
    return path


def _best_online_record(
    records: list[Mapping[str, Any]],
    maximum_iteration: int,
) -> dict[str, Any] | None:
    """Return the best finite validation record using the paper tie-break."""

    candidates: list[dict[str, Any]] = []
    for raw_record in records:
        record = dict(raw_record)
        try:
            iteration = int(record["iteration"])
            loss = float(record["loss"])
            accuracy = float(record["accuracy"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            iteration <= 0
            or iteration > int(maximum_iteration)
            or not math.isfinite(loss)
            or not math.isfinite(accuracy)
            or str(record.get("split", "val")).lower() != "val"
        ):
            continue
        candidates.append(record)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            float(item["accuracy"]),
            -float(item["loss"]),
            -int(item["iteration"]),
        ),
    )


def _selection_from_record(
    record: Mapping[str, Any], path: Path, candidate_count: int
) -> dict[str, Any]:
    return {
        "metric": "validation_accuracy",
        "mode": "maximum",
        "split": "val",
        "iteration": int(record["iteration"]),
        "loss": float(record["loss"]),
        "accuracy": float(record["accuracy"]),
        "candidate_count": int(candidate_count),
        "synthetic_dataset": str(path.resolve()),
    }


def run_condensation(
    config: Mapping[str, Any],
    condensation_seed: int = 0,
    iteration_limit: int | None = None,
    ipc: int | None = None,
) -> dict[str, Any]:
    """Run size-weighted cluster IDM and resume compatible checkpoints."""

    condensation = config["condensation"]
    experiment = str(
        condensation.get("experiment_name", "size_weighted_cluster_idm")
    )
    ipc = _ipc(config, ipc)
    condensation["idm"]["ipc"] = int(ipc)
    partition_factor = _partition_factor(condensation, ipc)
    cluster_settings = condensation["cluster_matching"]
    cluster_initialization = bool(
        cluster_settings.get("initialization_enabled", False)
    )
    cluster_matching = bool(cluster_settings.get("matching_enabled", False))
    representative_settings = condensation.get(
        "representative_sampling", {}
    )
    representative_sampling = bool(
        representative_settings.get("enabled", False)
    )
    spread_enabled = bool(condensation["cluster_spread"]["enabled"])
    needs_clusters = cluster_initialization or cluster_matching
    device = resolve_device(config)
    seed = (
        int(config["project"]["seed"])
        + int(ipc) * 100000
        + int(condensation_seed) * 1009
    )
    seed_everything(
        seed, bool(config["project"].get("deterministic", False))
    )
    bundle = experiment_bundle(config)
    pool = IndexedImagePool(
        bundle.train,
        bundle.num_classes,
        seed + 17,
        cache_images=bool(
            condensation.get("memory", {}).get("cache_real_images", True)
        ),
    )
    cluster_counts = (
        _cluster_layouts(cluster_settings, int(ipc), pool.class_indices)
        if needs_clusters
        else {class_id: int(ipc) for class_id in range(bundle.num_classes)}
    )
    adaptive_capacity = bool(
        needs_clusters
        and any(int(count) < int(ipc) for count in cluster_counts.values())
    )
    design_signature = {
        "cluster_initialization": cluster_initialization,
        "cluster_matching": cluster_matching,
        "cluster_spread": spread_enabled,
        "representative_sampling": representative_sampling,
        "cluster_size_weighted": cluster_matching,
        "cluster_center_loss_weight": (
            float(cluster_settings["center_loss_weight"])
            if cluster_matching
            else 0.0
        ),
        "cluster_descriptor_mode": str(
            cluster_settings.get("descriptor_mode", "pixel_pca")
        ),
        "cluster_spread_mixer_mode": str(
            condensation["cluster_spread"].get("mixer_mode", "full")
        ),
        "cluster_spread_gradient_fraction": (
            float(condensation["cluster_spread"][
                "target_gradient_fraction"
            ])
            if spread_enabled
            else 0.0
        ),
        "ipc_clusters_by_class": {
            str(class_id): int(count)
            for class_id, count in sorted(cluster_counts.items())
        },
        "cluster_count_mode": str(
            cluster_settings.get("cluster_count_mode", "fixed")
        ),
        "images_per_cluster": int(
            cluster_settings.get("images_per_cluster", 100)
        ),
        "adaptive_capacity": (
            "exact_linear_cluster_size" if adaptive_capacity else "disabled"
        ),
        "partition_expansion": int(partition_factor),
        "real_image_cache": bool(pool.cache_images),
    }
    if spread_enabled and not cluster_matching:
        raise ValueError("Cluster spread requires cluster matching")
    pixel_constraints = dict(condensation.get("pixel_constraints", {}))
    algorithm_version = _algorithm_version(condensation)
    directory = (
        output_root(config)
        / f"ipc_{int(ipc)}"
        / f"condense_seed_{int(condensation_seed)}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    logger = get_stage_logger(
        f"idm_condense_{experiment}_{int(condensation_seed)}",
        directory,
    )
    cluster_index: PixelClusterIndex | None = None
    cluster_cache_path: Path | None = None
    if needs_clusters:
        cluster_index, cluster_cache_path = _load_or_build_cluster_index(
            config,
            pool,
            int(ipc),
            cluster_counts,
            seed + 41,
            device,
        )
        logger.info("cluster index=%s", cluster_cache_path)
        atomic_write_json(
            {
                "cache": str(cluster_cache_path.resolve()),
                "clusters_by_class": {
                    str(class_id): int(count)
                    for class_id, count in sorted(cluster_counts.items())
                },
                "stored_images_per_class": int(ipc),
                "minimum_images_per_cluster": 1,
                "coverage": "size_apportioned_canvases_per_cluster_centre_to_edge",
                "sizes": cluster_index.sizes,
                "allocations": cluster_index.allocations,
                "descriptor_size": list(cluster_index.descriptor_size),
                "pca_components": int(cluster_index.pca_components),
                "descriptor_mode": str(cluster_index.descriptor_mode),
                "deep_feature_extractor": bool(
                    cluster_index.descriptor_mode != "pixel_pca"
                ),
            },
            directory / "cluster_summary.json",
        )
    cluster_sampler = (
        ClusterSampler(cluster_index, seed + 53)
        if cluster_matching and cluster_index is not None
        else None
    )
    dream_sampler = (
        DreamRepresentativeSampler(config, pool, device, seed + 67)
        if representative_sampling
        else None
    )
    initial_pixels, labels = _initialize_synthetic_pixels(
        config,
        pool,
        cluster_index,
        bundle.num_classes,
        int(ipc),
        int(partition_factor),
    )
    parameterization = SyntheticParameterization(
        initial_pixels,
        labels,
        device,
    ).to(device)
    optimizer, optimizer_name, learning_rate = _build_synthetic_optimizer(
        parameterization.variable,
        condensation,
        int(ipc),
    )
    queue = OfficialIDMQueue(
        config, bundle.num_classes, device, seed + 29
    )
    mixer = AuxiliaryGradientMixer(condensation["cluster_spread"])
    completed = 0
    diagnostics: dict[str, Any] = {}
    checkpoint_path = find_latest_checkpoint(directory)
    if checkpoint_path is not None:
        payload = load_checkpoint(checkpoint_path, "cpu")
        checkpoint_version = int(payload.get("algorithm_version", 1))
        if checkpoint_version != algorithm_version:
            raise ValueError(
                "断点算法版本不兼容："
                f"{checkpoint_path} 使用 v{checkpoint_version}，"
                f"当前配置使用 v{algorithm_version}。"
                "请为新版实验使用新的输出目录，或只移走该 IPC 的旧断点。"
            )
        checkpoint_design = dict(payload.get("design_signature", {}))
        if checkpoint_design != design_signature:
            raise ValueError(
                "Checkpoint design does not match the current CACDM protocol: "
                f"checkpoint={checkpoint_design}, current={design_signature}"
            )
        checkpoint_pixel_constraints = payload.get("pixel_constraints")
        if (
            checkpoint_pixel_constraints is not None
            and dict(checkpoint_pixel_constraints) != pixel_constraints
        ):
            raise ValueError(
                "断点的像素投影设置与当前命令不一致："
                f"断点为 {checkpoint_pixel_constraints}，"
                f"当前为 {pixel_constraints}。"
                "请先把该 IPC 的现有输出移到独立目录，再开始新的对照实验。"
            )
        parameterization.load_state_dict(
            payload["parameterization"], strict=True
        )
        checkpoint_optimizer_name = str(
            payload.get("optimizer_name", "sgd")
        ).lower()
        if checkpoint_optimizer_name == optimizer_name:
            optimizer.load_state_dict(payload["optimizer"])
            _optimizer_to_device(optimizer, device)
        else:
            logger.warning(
                "断点优化器为 %s，当前配置为 %s；保留合成变量和迭代进度，"
                "但按当前配置重新初始化优化器状态",
                checkpoint_optimizer_name,
                optimizer_name,
            )
        for group in optimizer.param_groups:
            # 保留兼容优化器的历史状态，但允许 YAML 调整当前学习率。
            group["lr"] = float(learning_rate)
        queue.load_state_dict(payload.get("queue"))
        pool.load_state_dict(payload.get("real_pool"))
        mixer.load_state_dict(payload.get("auxiliary_mixer"))
        if cluster_sampler is not None:
            cluster_sampler.load_state_dict(payload.get("cluster_sampler"))
        if dream_sampler is not None:
            dream_sampler.load_state_dict(
                payload.get("dream_representative_sampler")
            )
        restore_rng_state(payload.get("rng_state"))
        completed = int(payload.get("iteration", 0))
        diagnostics = dict(payload.get("diagnostics", {}))
        logger.info(
            "恢复 %s seed=%d：iteration=%d",
            experiment,
            condensation_seed,
            completed,
        )
    else:
        # 官方实现从 3 个模型开始，并在 iteration 0 首先加入第 4 个。
        queue.grow(0)
    target = int(
        iteration_limit
        if iteration_limit is not None
        else condensation["idm"]["iterations"]
    )
    online_settings = condensation.get("online_evaluation", {})
    select_best_by_accuracy = bool(
        online_settings.get("select_best_by_accuracy", False)
    )
    online_path = directory / "online_evaluation.json"
    online_payload = read_json(online_path, default={}) or {}
    online_records = list(online_payload.get("records", []))
    best_path = directory / "synthetic.pt"
    best_record: dict[str, Any] | None = None
    if select_best_by_accuracy and best_path.is_file():
        best_payload = load_checkpoint(best_path, "cpu")
        best_iteration = int(best_payload.get("iteration", -1))
        best_images = best_payload.get("images")
        best_labels = best_payload.get("labels")
        expected_labels = parameterization.labels.detach().long().cpu()
        if not torch.is_tensor(best_images) or tuple(best_images.shape) != tuple(
            parameterization.variable.shape
        ):
            raise ValueError("Existing best synthetic dataset has an incompatible shape")
        if not torch.is_tensor(best_labels) or not torch.equal(
            best_labels.detach().long().cpu(), expected_labels
        ):
            raise ValueError("Existing best synthetic dataset has incompatible labels")
        matching_records = [
            record
            for record in online_records
            if int(record.get("iteration", -1)) == best_iteration
        ]
        best_record = _best_online_record(matching_records, target)
        if best_record is None:
            saved_selection = dict(
                best_payload.get("selection")
                or online_payload.get("selection")
                or {}
            )
            if int(saved_selection.get("iteration", -1)) == best_iteration:
                best_record = {
                    "iteration": best_iteration,
                    "accuracy": float(saved_selection["accuracy"]),
                    "loss": float(saved_selection["loss"]),
                    "split": "val",
                }
        if best_record is None:
            raise ValueError(
                "Existing synthetic.pt has no matching validation record"
            )
        logger.info(
            "loaded existing best synthetic dataset: iteration=%d "
            "accuracy=%.6f loss=%.6f",
            int(best_record["iteration"]),
            float(best_record["accuracy"]),
            float(best_record["loss"]),
        )
        del best_payload, best_images, best_labels, expected_labels
    diagnostic_path = directory / "diagnostic_history.json"
    diagnostic_payload = read_json(diagnostic_path, default={}) or {}
    diagnostic_records = list(diagnostic_payload.get("records", []))
    online_interval = int(online_settings.get("interval_iterations", 0))
    log_interval = int(condensation["idm"]["log_interval_iterations"])
    checkpoint_interval = int(
        condensation["idm"]["checkpoint_interval_iterations"]
    )
    batch_real = int(
        _idm_value(condensation["idm"], "batch_real", int(ipc))
    )
    dsa_strategy = str(condensation["idm"]["dsa_strategy"])
    augmentation = ParamDiffAug()
    final_iteration = int(completed)
    for iteration in range(completed + 1, target + 1):
        final_iteration = int(iteration)
        started = time.perf_counter()
        should_validate = (
            online_interval > 0 and iteration % online_interval == 0
        )
        should_log = (
            iteration == 1
            or iteration % log_interval == 0
            or iteration == target
        )
        checkpoint_due = (
            iteration % checkpoint_interval == 0 or iteration == target
        )
        collect_diagnostics = should_validate or should_log or checkpoint_due
        current_image_learning_rate = float(
            _idm_value(
                condensation["idm"], "image_learning_rate", int(ipc)
            )
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_image_learning_rate
        if device.type == "cuda" and collect_diagnostics:
            torch.cuda.reset_peak_memory_stats(device)
        queue.grow(iteration)
        reliability_was_reset = queue.reset_reliability_if_due(iteration)
        guidance = queue.guidance_members()
        loss_keys = (
            "mean",
            "cross_entropy",
            "weighted_cross_entropy",
            "cluster_spread",
            "prior",
        )
        loss_sums = (
            {
                key: torch.zeros((), device=device)
                for key in loss_keys
            }
            if collect_diagnostics
            else None
        )
        real_microbatch = int(
            condensation["memory"]["real_feature_microbatch"]
        )
        synthetic_microbatch = int(
            condensation["memory"]["synthetic_feature_microbatch"]
        )
        projection_samples: list[dict[str, float]] = []
        # Official IDM takes one synthetic optimizer step per class. Collapsing
        # these into one step under-trains a C-class dataset by about C times.
        for class_id in range(bundle.num_classes):
            optimizer.zero_grad(set_to_none=True)
            class_slice = parameterization.class_slice(class_id)
            stored_labels = parameterization.labels[class_slice]
            class_cluster_count = (
                cluster_index.cluster_count(class_id)
                if cluster_index is not None
                else int(ipc)
            )
            stored_groups = (
                _stored_cluster_groups(
                    cluster_index.class_allocations(class_id, int(ipc)),
                    device,
                )
                if cluster_matching
                else None
            )
            for member in guidance:
                if representative_sampling:
                    if dream_sampler is None:
                        raise RuntimeError("Missing DREAM representative sampler")
                    real_cpu = dream_sampler.sample(
                        member.model,
                        member.identifier,
                        class_id,
                        batch_real,
                        iteration,
                    )
                    real_groups_cpu = None
                elif cluster_matching:
                    if cluster_sampler is None:
                        raise RuntimeError("Missing cluster sampler")
                    real_cpu, real_groups_cpu = _sample_clustered_real_batch(
                        pool,
                        cluster_sampler,
                        class_id,
                        int(class_cluster_count),
                        batch_real,
                    )
                else:
                    real_cpu = pool.sample(class_id, batch_real)
                    real_groups_cpu = None
                dsa_seed = queue.random.randrange(0, 100000)
                real_features, real_microbatch = _real_features(
                    config,
                    member.model,
                    real_cpu,
                    dsa_seed,
                    device,
                    dsa_strategy,
                    augmentation,
                )
                del real_cpu
                real_groups = (
                    real_groups_cpu.to(device, non_blocking=True)
                    if real_groups_cpu is not None
                    else None
                )
                with autocast_context(config, device):
                    generated = parameterization.render_class(class_id)
                generated = generated.float()
                synthetic, synthetic_labels, synthetic_groups = (
                    partition_training_images(
                        generated,
                        stored_labels,
                        partition_factor=int(partition_factor),
                        group_ids=stored_groups,
                    )
                )
                synthetic = diff_augment(
                    synthetic,
                    dsa_strategy,
                    seed=dsa_seed,
                    param=augmentation,
                )
                synthetic_output = _synthetic_features(
                    member.model,
                    synthetic,
                    synthetic_microbatch,
                )
                if cluster_matching:
                    if real_groups is None or synthetic_groups is None:
                        raise RuntimeError("Cluster group metadata is missing")
                    spread_settings = condensation["cluster_spread"]
                    if cluster_index is None:
                        raise RuntimeError("Missing cluster-size metadata")
                    mean_loss, spread_raw = cluster_distribution_losses(
                        real_features,
                        real_groups,
                        synthetic_output.embedding,
                        synthetic_groups,
                        cluster_count=int(class_cluster_count),
                        cluster_weights=cluster_index.sizes[int(class_id)],
                        radial_weight=(
                            float(spread_settings["radial_weight"])
                            if spread_enabled
                            else 0.0
                        ),
                        standard_deviation_weight=(
                            float(spread_settings["standard_deviation_weight"])
                            if spread_enabled
                            else 0.0
                        ),
                        smooth_l1_beta=float(
                            spread_settings["smooth_l1_beta"]
                        ),
                    )
                    mean_loss = mean_loss * float(
                        cluster_settings["center_loss_weight"]
                    )
                else:
                    mean_loss = (
                        synthetic_output.embedding.float().mean(0)
                        - real_features.mean(0).detach()
                    ).square().sum()
                    spread_raw = generated.sum() * 0.0
                ce_loss = F.cross_entropy(
                    synthetic_output.logits.float(), synthetic_labels
                )
                reliability = member.reliability_percent_tensor().clamp_min(
                    float(condensation["idm"]["minimum_reliability"])
                ).to(dtype=ce_loss.dtype)
                weighted_ce_loss = (
                    _ce_weight(config, int(ipc))
                    * reliability
                    * ce_loss
                )
                base_loss = mean_loss + weighted_ce_loss
                base_image_gradient = torch.autograd.grad(
                    base_loss,
                    generated,
                    retain_graph=spread_enabled,
                )[0]
                image_gradient = base_image_gradient
                if spread_enabled:
                    spread_image_gradient = torch.autograd.grad(
                        spread_raw,
                        generated,
                        retain_graph=False,
                    )[0]
                    weighted_spread, _ = mixer.weight_against_base(
                        base_image_gradient,
                        spread_image_gradient,
                    )
                    image_gradient = image_gradient + weighted_spread
                    del spread_image_gradient, weighted_spread
                del base_image_gradient
                # 多网络指导时顺序计算、平均梯度，整个类别只更新一次；
                # 避免 train_net_num 同时偷偷放大有效优化步数和显存峰值。
                generated.backward(
                    image_gradient / float(max(1, len(guidance)))
                )
                if loss_sums is not None:
                    loss_sums["mean"] += mean_loss.detach()
                    loss_sums["cross_entropy"] += ce_loss.detach()
                    loss_sums["weighted_cross_entropy"] += (
                        weighted_ce_loss.detach()
                    )
                    loss_sums["cluster_spread"] += spread_raw.detach()
                del (
                    generated,
                    synthetic,
                    synthetic_labels,
                    synthetic_output,
                    mean_loss,
                    ce_loss,
                    weighted_ce_loss,
                    base_loss,
                    spread_raw,
                    image_gradient,
                    real_features,
                    real_groups,
                    real_groups_cpu,
                    synthetic_groups,
                )
            del stored_labels, stored_groups
            # 当前直接像素参数化的 prior_loss 恒等于零；原实现也不会反传。
            optimizer.step()
            if bool(pixel_constraints.get("enabled", False)):
                projection_sample = _project_synthetic_pixels(
                    parameterization.variable,
                    optimizer,
                    config,
                    collect_metrics=collect_diagnostics,
                )
                if projection_sample is not None:
                    projection_samples.append(projection_sample)
        if collect_diagnostics and projection_samples:
            projection_metrics = {
                key: statistics.fmean(
                    float(sample[key]) for sample in projection_samples
                )
                for key in projection_samples[0]
            }
        elif collect_diagnostics:
            projection_metrics = {
                "pixel/pre_projection_minimum": float(
                    parameterization.variable.detach().min().item()
                ),
                "pixel/pre_projection_maximum": float(
                    parameterization.variable.detach().max().item()
                ),
                "pixel/projected_fraction": 0.0,
                "pixel/momentum_cleared_fraction": 0.0,
                **_pixel_statistics(parameterization.variable, config),
            }
        else:
            projection_metrics = {}
        queue_metrics = queue.train_independent_members(
            pool, collect_metrics=collect_diagnostics
        )
        diagnostics = {}
        if collect_diagnostics:
            assert loss_sums is not None
            loss_values = torch.stack(
                tuple(loss_sums[key] for key in loss_keys)
            ).detach().cpu().tolist()
            loss_reports = dict(zip(loss_keys, loss_values, strict=True))
            divisor = float(bundle.num_classes * max(1, len(guidance)))
            peak_mib = cuda_peak_megabytes()
            total_mib = (
                float(torch.cuda.get_device_properties(device).total_memory)
                / (1024**2)
                if device.type == "cuda"
                else 0.0
            )
            recent_gradient_count = bundle.num_classes
            diagnostics = {
            **{
                f"loss/{key}": value / divisor
                for key, value in loss_reports.items()
            },
            **queue_metrics,
            **projection_metrics,
            "queue/reliability_reset": bool(reliability_was_reset),
            "design/cluster_initialization": cluster_initialization,
            "design/cluster_matching": cluster_matching,
            "design/representative_sampling": representative_sampling,
            "design/cluster_size_weighted": cluster_matching,
            "design/cluster_center_loss_weight": (
                float(cluster_settings["center_loss_weight"])
                if cluster_matching
                else 0.0
            ),
            "cluster_spread/enabled": spread_enabled,
            "cluster_spread/target_gradient_fraction": (
                mixer.target_fraction() if spread_enabled else 0.0
            ),
            "cluster_spread/gradient_scale": (
                mixer.recent_mean(
                    "gradient_scales", recent_gradient_count
                )
                if spread_enabled
                else 0.0
            ),
            "cluster_spread/base_gradient_norm": (
                mixer.recent_mean("base_norms", recent_gradient_count)
                if spread_enabled
                else 0.0
            ),
            "cluster_spread/raw_gradient_norm": (
                mixer.recent_mean("auxiliary_norms", recent_gradient_count)
                if spread_enabled
                else 0.0
            ),
            "cluster_spread/gradient_cosine": (
                mixer.recent_mean("gradient_cosines", recent_gradient_count)
                if spread_enabled
                else 0.0
            ),
            "cluster_spread/conflict_projection_rate": (
                mixer.recent_mean("conflict_projections", recent_gradient_count)
                if spread_enabled
                else 0.0
            ),
            "cluster_spread/achieved_gradient_fraction": (
                mixer.recent_mean(
                    "achieved_fractions", recent_gradient_count
                )
                if spread_enabled
                else 0.0
            ),
            "cluster_spread/mixed_gradients": int(mixer.mixed_gradients),
            "cluster_spread/skipped_gradients": int(mixer.skipped_gradients),
            "optimization/image_learning_rate": float(
                current_image_learning_rate
            ),
            "memory/real_feature_microbatch": int(real_microbatch),
            "memory/synthetic_feature_microbatch": int(
                synthetic_microbatch
            ),
            "memory/cuda_peak_mib": float(peak_mib),
            "memory/cuda_peak_fraction": (
                float(peak_mib / total_mib) if total_mib > 0 else 0.0
            ),
            "time/iteration_seconds": time.perf_counter() - started,
            }
        if should_validate:
            # Free inactive allocator blocks before the fresh online classifier.
            # Live synthetic parameters and the IDM guidance queue stay intact.
            cleanup_memory()
            synthetic_for_validation = (
                parameterization.render_all().detach().cpu()
            )
            monitor_split = str(
                config["data"].get("online_evaluation_split", "val")
            ).lower()
            monitor_dataset = (
                bundle.test if monitor_split == "test" else bundle.val
            )
            monitor = _quick_evaluate(
                config,
                synthetic_for_validation,
                parameterization.labels.detach().cpu(),
                monitor_dataset,
                bundle.num_classes,
                iteration,
                seed + 700001,
                int(ipc),
                int(partition_factor),
                device,
                logger=logger,
            )
            del synthetic_for_validation
            monitor["split"] = monitor_split
            monitor["selects_checkpoint"] = select_best_by_accuracy
            monitor["can_stop_training"] = False
            online_records.append(monitor)
            selection: dict[str, Any] | None = None
            if select_best_by_accuracy:
                candidates = (
                    [best_record, monitor]
                    if best_record is not None
                    else [monitor]
                )
                candidate_best = _best_online_record(candidates, iteration)
                if (
                    candidate_best is not None
                    and int(candidate_best["iteration"]) == int(iteration)
                ):
                    best_record = dict(monitor)
                    selection = _selection_from_record(
                        best_record, best_path, len(online_records)
                    )
                    _export(
                        config,
                        directory,
                        experiment,
                        parameterization,
                        bundle.class_names,
                        iteration,
                        int(ipc),
                        int(partition_factor),
                        optimization_iterations=iteration,
                        selection=selection,
                    )
                    logger.info(
                        "updated best synthetic dataset: iteration=%d "
                        "accuracy=%.6f loss=%.6f path=%s",
                        iteration,
                        float(monitor["accuracy"]),
                        float(monitor["loss"]),
                        best_path,
                    )
                elif best_record is not None:
                    selection = _selection_from_record(
                        best_record, best_path, len(online_records)
                    )
                selected_iteration = (
                    int(best_record["iteration"])
                    if best_record is not None
                    else -1
                )
                online_records = [
                    {
                        **dict(record),
                        "selected": int(record.get("iteration", -1))
                        == selected_iteration,
                    }
                    for record in online_records
                ]
            atomic_write_json(
                {
                    "purpose": (
                        "validation_accuracy_snapshot_selection"
                        if select_best_by_accuracy
                        else "training_progress_only"
                    ),
                    "split": monitor_split,
                    "selects_checkpoint": select_best_by_accuracy,
                    "can_stop_training": False,
                    "selection": selection,
                    "records": online_records,
                },
                online_path,
            )
            try:
                refresh_results_report(config)
            except Exception as error:
                logger.warning("results overview refresh failed: %s", error)
            diagnostics.update(
                {
                    "online/accuracy": float(monitor["accuracy"]),
                    "online/loss": float(monitor["loss"]),
                }
            )
        if should_log:
            message = (
                "%s seed=%d iter=%d/%d mean=%.5f image_lr=%.4g "
                "ce_coeff=%.3f ce_loss_w=%.5f"
            )
            values: list[Any] = [
                experiment,
                condensation_seed,
                iteration,
                target,
                diagnostics["loss/mean"],
                diagnostics["optimization/image_learning_rate"],
                _ce_weight(config, int(ipc)),
                diagnostics["loss/weighted_cross_entropy"],
            ]
            if spread_enabled:
                message += (
                    " spread=%.5f spread_scale=%.4g spread_fraction=%.4f"
                )
                values.extend(
                    [
                        diagnostics["loss/cluster_spread"],
                        diagnostics["cluster_spread/gradient_scale"],
                        diagnostics[
                            "cluster_spread/achieved_gradient_fraction"
                        ],
                    ]
                )
            if bool(pixel_constraints.get("enabled", False)):
                message += " projected=%.4f"
                values.append(diagnostics["pixel/projected_fraction"])
            message += " queue=%d updates=%.1f peak=%.0fMiB time=%.2fs"
            values.extend(
                [
                    int(diagnostics["queue/size"]),
                    diagnostics["queue/mean_updates"],
                    diagnostics["memory/cuda_peak_mib"],
                    diagnostics["time/iteration_seconds"],
                ]
            )
            logger.info(message, *values)
            diagnostic_records.append(
                {"iteration": int(iteration), **diagnostics}
            )
            atomic_write_json(
                {
                    "experiment": experiment,
                    "seed": int(seed),
                    "records": diagnostic_records,
                },
                diagnostic_path,
            )
        if checkpoint_due:
            _save_checkpoint(
                directory,
                experiment,
                iteration,
                parameterization,
                optimizer,
                queue,
                pool,
                mixer,
                cluster_sampler,
                dream_sampler,
                design_signature,
                pixel_constraints,
                diagnostics,
                algorithm_version,
            )
    selected_iteration = int(final_iteration)
    selection: dict[str, Any] | None = None
    if select_best_by_accuracy:
        if best_record is None or not best_path.is_file():
            raise FileNotFoundError(
                "Training finished without a validation-selected synthetic.pt"
            )
        selection = _selection_from_record(
            best_record, best_path, len(online_records)
        )
        selected_payload = load_checkpoint(best_path, "cpu")
        selected_images = selected_payload["images"].detach().float().cpu()
        selected_labels = selected_payload["labels"].detach().long().cpu()
        expected_labels = parameterization.labels.detach().long().cpu()
        if tuple(selected_images.shape) != tuple(parameterization.variable.shape):
            raise ValueError(
                "Selected synthetic dataset has an incompatible image shape"
            )
        if not torch.equal(selected_labels, expected_labels):
            raise ValueError(
                "Selected synthetic dataset has incompatible labels"
            )
        with torch.no_grad():
            parameterization.variable.copy_(selected_images.to(device))
        selected_iteration = int(selection["iteration"])
        logger.info(
            "selected validation-accuracy synthetic dataset iteration=%d "
            "accuracy=%.6f loss=%.6f candidates=%d",
            selected_iteration,
            float(selection["accuracy"]),
            float(selection["loss"]),
            int(selection["candidate_count"]),
        )
        del selected_payload, selected_images, selected_labels, expected_labels
        online_records = [
            {
                **dict(record),
                "selected": int(record.get("iteration", -1))
                == selected_iteration,
            }
            for record in online_records
        ]
        atomic_write_json(
            {
                "purpose": "validation_accuracy_snapshot_selection",
                "split": "val",
                "selects_checkpoint": True,
                "can_stop_training": False,
                "selection": selection,
                "records": online_records,
            },
            online_path,
        )
    synthetic_path = _export(
        config,
        directory,
        experiment,
        parameterization,
        bundle.class_names,
        selected_iteration,
        int(ipc),
        int(partition_factor),
        optimization_iterations=int(final_iteration),
        selection=selection,
    )
    summary = {
        "experiment": experiment,
        "condensation_seed": int(condensation_seed),
        "seed": int(seed),
        "parameterization": "pixel",
        "algorithm_version": int(algorithm_version),
        "design": design_signature,
        "ipc": int(ipc),
        "partition_expansion": int(partition_factor),
        **_formation_metadata(int(ipc), int(partition_factor)),
        "cluster_cache": (
            str(cluster_cache_path.resolve())
            if cluster_cache_path is not None
            else None
        ),
        "cluster_sizes": cluster_index.sizes if cluster_index is not None else None,
        "resolved_idm_protocol": {
            "image_learning_rate": float(
                _idm_value(
                    condensation["idm"], "image_learning_rate", int(ipc)
                )
            ),
            "image_momentum": float(condensation["idm"]["image_momentum"]),
            "ce_weight": float(_ce_weight(config, int(ipc))),
            "batch_real": int(
                _idm_value(condensation["idm"], "batch_real", int(ipc))
            ),
            "batch_train": int(
                _idm_value(condensation["idm"], "batch_train", int(ipc))
            ),
            "net_num": int(
                _idm_value(condensation["idm"], "net_num", int(ipc))
            ),
            "net_generate_interval": int(
                _idm_value(
                    condensation["idm"],
                    "net_generate_interval",
                    int(ipc),
                )
            ),
            "reliability_reset_interval": int(
                _idm_value(
                    condensation["idm"],
                    "reliability_reset_interval",
                    int(ipc),
                )
            ),
            "synthetic_updates_per_iteration": int(bundle.num_classes),
        },
        "iterations": int(final_iteration),
        "maximum_iterations": int(target),
        "selected_iteration": int(selected_iteration),
        "snapshot_selection": selection,
        "stopped_early": False,
        "synthetic_dataset": str(synthetic_path.resolve()),
        "auxiliary_gradient_mixer": mixer.state_dict(),
        "pixel_constraints": pixel_constraints,
        "diagnostics": diagnostics,
    }
    atomic_write_json(summary, directory / "summary.json")
    try:
        refresh_results_report(config)
    except Exception as error:
        logger.warning("results overview refresh failed: %s", error)
    del (
        parameterization,
        optimizer,
        queue,
        pool,
        mixer,
        cluster_sampler,
        dream_sampler,
    )
    cleanup_memory()
    return summary
