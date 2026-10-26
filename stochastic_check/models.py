"""Build the original CIFAR adaptations of torchvision architectures.

ResNets use a 3x3 stride-one stem without initial max pooling; VGG uses
one linear classifier after adaptive pooling. This retains the original
He et al. / Zagoruyko-Komodakis CIFAR adaptation and initialization order.
"""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from torchvision.models import resnet18, resnet34, vgg11_bn


def _adapt_resnet(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the ImageNet stem and classifier for 32x32 inputs."""
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(cast(nn.Linear, model.fc).in_features, num_classes)
    return model


def resnet18_cifar(num_classes: int) -> nn.Module:
    """Initialize ResNet-18 with the CIFAR stem and requested class count."""
    return _adapt_resnet(resnet18(weights=None), num_classes)


def resnet34_cifar(num_classes: int) -> nn.Module:
    """Initialize ResNet-34 with the CIFAR stem and requested class count."""
    return _adapt_resnet(resnet34(weights=None), num_classes)


def vgg11_bn_cifar(num_classes: int) -> nn.Module:
    """Replace VGG's 4096-unit MLP with one CIFAR classification layer."""
    model = vgg11_bn(weights=None)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Linear(512, num_classes)
    return model


_MODEL_REGISTRY = {
    "resnet18": resnet18_cifar,
    "resnet34": resnet34_cifar,
    "vgg11": vgg11_bn_cifar,
}


def build_model(arch: str, num_classes: int) -> nn.Module:
    """Initialize a supported architecture without pretrained weights."""
    if arch not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown architecture '{arch}'. Known: {sorted(_MODEL_REGISTRY)}"
        )

    return _MODEL_REGISTRY[arch](num_classes)
