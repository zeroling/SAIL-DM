"""当前实验唯一的数据接口。"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from Core.data import build_data_bundle


def experiment_bundle(config: Mapping[str, Any]):
    """Build the registered public splits without training augmentation."""

    return build_data_bundle(
        config,
        train_augmentation={"enabled": False},
    )


class TensorImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images.detach().cpu().float()
        self.labels = labels.detach().cpu().long()
        self.targets = self.labels.tolist()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> dict[str, Any]:
        normalized = int(index)
        return {
            "image": self.images[normalized],
            "label": self.labels[normalized],
            "key": f"synthetic:{normalized}",
        }
