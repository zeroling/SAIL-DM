"""CPU-only structural smoke test for the 224x224 PathMNIST protocol."""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Apply the same Windows MKL safeguard as the real launcher before sklearn is
# imported by the clustering module.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Core.config import load_config
from Net.Condensation.cluster_multiform import build_pixel_cluster_index
from Net.Condensation.idm_official import build_idm_convnet
from Pipeline.Stages.condense import (
    IndexedImagePool,
    _partition_factor,
    _synthetic_features,
)


class TinyPath224(Dataset):
    def __init__(self) -> None:
        self.targets = [class_id for class_id in range(9) for _ in range(4)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(1000 + int(index))
        return {
            "image": torch.rand(3, 224, 224, generator=generator),
            "label": torch.tensor(self.targets[int(index)]),
        }


def main() -> None:
    config = load_config(dataset="pathmnist224")
    assert config["data"]["image"]["size"] == [224, 224]
    assert config["condensation"]["idm"]["network_depth"] == 5
    assert _partition_factor(config["condensation"], 1) == 2
    assert _partition_factor(config["condensation"], 10) == 2
    assert _partition_factor(config["condensation"], 100) == 2
    assert config["condensation"]["memory"][
        "synthetic_feature_microbatch"
    ] == 8

    dataset = TinyPath224()
    pool = IndexedImagePool(dataset, 9, seed=7, cache_images=False)
    assert not pool.cache_images
    sample = pool.sample(0, 2)
    assert sample.shape == (2, 3, 224, 224)

    class_indices = {
        class_id: [
            index for index, label in enumerate(dataset.targets)
            if label == class_id
        ]
        for class_id in range(9)
    }
    cluster_index = build_pixel_cluster_index(
        dataset,
        class_indices,
        clusters_per_class={class_id: 1 for class_id in range(9)},
        descriptor_size=(32, 32),
        pca_components=128,
        kmeans_n_init=1,
        normalization_mean=(0.0, 0.0, 0.0),
        normalization_std=(1.0, 1.0, 1.0),
        seed=11,
    )
    assert cluster_index.descriptor_mode == "pixel_pca"
    assert cluster_index.descriptor_size == (32, 32)

    model = build_idm_convnet(3, 9, (224, 224), depth=5)
    images = torch.randn(2, 3, 224, 224, requires_grad=True)
    output = _synthetic_features(model, images, microbatch=1)
    assert output.logits.shape == (2, 9)
    assert output.embedding.shape == (2, 128, 7, 7)
    base_loss = output.logits.square().mean()
    spread_loss = output.embedding.square().mean()
    base_gradient = torch.autograd.grad(
        base_loss, images, retain_graph=True
    )[0]
    spread_gradient = torch.autograd.grad(spread_loss, images)[0]
    assert torch.isfinite(base_gradient).all()
    assert torch.isfinite(spread_gradient).all()
    print("PathMNIST 224 smoke test passed")


if __name__ == "__main__":
    main()
