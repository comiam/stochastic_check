"""CIFAR-adapted architectures for the stochastic-check experiments.

The torchvision reference models are tailored to ImageNet-sized inputs
(224 x 224, 7 x 7 stem, max-pool).  For 32 x 32 CIFAR we use the
standard adaptation:

  - replace the 7 x 7 stride-2 stem by a 3 x 3 stride-1 conv,
  - drop the initial max-pool layer for ResNets,
  - replace the classifier head to match ``num_classes``.

This is the same recipe used in the He et al. and Zagoruyko / Komodakis
papers for CIFAR benchmarking.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18, resnet34, vgg11_bn

# ---------------------------------------------------------------------------
#  ResNets
# ---------------------------------------------------------------------------


def _adapt_resnet(model: nn.Module, num_classes: int) -> nn.Module:
    model.conv1 = nn.Conv2d(  # type: ignore[assignment]
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()  # type: ignore[assignment]
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # type: ignore[assignment]
    return model


def resnet18_cifar(num_classes: int) -> nn.Module:
    return _adapt_resnet(resnet18(weights=None), num_classes)


def resnet34_cifar(num_classes: int) -> nn.Module:
    return _adapt_resnet(resnet34(weights=None), num_classes)


# ---------------------------------------------------------------------------
#  VGG
# ---------------------------------------------------------------------------


def vgg11_bn_cifar(num_classes: int) -> nn.Module:
    """VGG-11 with BatchNorm, classifier adapted to CIFAR.

    The torchvision VGG ends in a 3-layer MLP with 4096-unit hidden
    layers; for CIFAR we collapse this to a single linear head, which is
    standard practice and matches the receptive field of the conv stack
    on 32 x 32 inputs.
    """
    model = vgg11_bn(weights=None)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # type: ignore[assignment]
    model.classifier = nn.Linear(512, num_classes)  # type: ignore[assignment]
    return model


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------


_MODEL_REGISTRY = {
    "resnet18": resnet18_cifar,
    "resnet34": resnet34_cifar,
    "vgg11": vgg11_bn_cifar,
}


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown architecture '{arch}'. Known: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[arch](num_classes)
