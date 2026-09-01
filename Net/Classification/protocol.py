"""HoP-TM cross-architecture evaluation networks.

The convolutional blocks and instance-normalization convention follow the
official HoP-TM ``networks.py``. Adaptive pooling removes the upstream
implementation's fixed classifier-shape assumptions.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from Net.Condensation.idm_official import build_idm_convnet


def _instance_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(int(channels), int(channels), affine=True)


class AlexNet(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Linear(192 * 4 * 4, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(images))
        return self.classifier(features.flatten(1))


class VGG11(nn.Module):
    _CONFIG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = int(channels)
        for item in self._CONFIG:
            if item == "M":
                layers.append(nn.MaxPool2d(2, 2))
                continue
            width = int(item)
            layers.extend(
                [
                    nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
                    _instance_norm(width),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, int(num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(images))
        return self.classifier(features.flatten(1))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = _instance_norm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, 3, stride=1, padding=1, bias=False
        )
        self.norm2 = _instance_norm(planes)
        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes, 1, stride=stride, bias=False
                ),
                _instance_norm(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.norm1(self.conv1(images)))
        output = self.norm2(self.conv2(output))
        return F.relu(output + self.shortcut(images))


class ResNet18(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(
            channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm1 = _instance_norm(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, int(num_classes))

    def _make_layer(
        self, planes: int, blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [int(stride)] + [1] * (int(blocks) - 1)
        layers = []
        for current_stride in strides:
            layers.append(
                BasicBlock(self.in_planes, planes, current_stride)
            )
            self.in_planes = int(planes)
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.norm1(self.conv1(images)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return self.classifier(self.pool(output).flatten(1))


def build_evaluation_model(
    architecture: str,
    channels: int,
    num_classes: int,
    image_size: Sequence[int],
    convnet_depth: int,
) -> nn.Module:
    name = str(architecture).strip().lower()
    if name == "convnet":
        return build_idm_convnet(
            channels,
            num_classes,
            image_size,
            depth=int(convnet_depth),
        )
    if name == "resnet18":
        return ResNet18(channels, num_classes)
    if name == "vgg11":
        return VGG11(channels, num_classes)
    if name == "alexnet":
        return AlexNet(channels, num_classes)
    raise ValueError(f"Unsupported evaluation architecture: {architecture}")
