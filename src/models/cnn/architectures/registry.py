"""Explicit model registry for lightweight CNN architectures."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Mapping, Type

from torch import nn

from .ds_cnn import DepthwiseSeparableCNN
from .mobilenet_like import MobileNetLikeCNN
from .small_cnn import SmallCNN
from .tiny_cnn import TinyCNN
from .tiny_resnet import TinyResidualCNN
from .torchvision_mobilenet import TorchvisionMobileNet
from .ultra_tiny_cnn import UltraTinyCNN

MODEL_REGISTRY: Mapping[str, Type[nn.Module]] = {
    "tiny_cnn": TinyCNN,
    "small_cnn": SmallCNN,
    "ds_cnn": DepthwiseSeparableCNN,
    "mobilenet_like": MobileNetLikeCNN,
    "tiny_resnet": TinyResidualCNN,
    "ultra_tiny": UltraTinyCNN,
    "torchvision_mobilenet": TorchvisionMobileNet,
}


def model_names() -> list[str]:
    """Return available model names, checking torchvision availability dynamically."""
    return [
        name
        for name in MODEL_REGISTRY
        if name != "torchvision_mobilenet" or find_spec("torchvision") is not None
    ]


def create_model(name: str, num_classes: int = 2) -> nn.Module:
    """Instantiate a model by name without pretrained weights.

    Args:
        name: Identifier in MODEL_REGISTRY.
        num_classes: Number of target output classes (default: 2).

    Returns:
        Instantiated nn.Module.

    Raises:
        ValueError: If name is unknown or num_classes < 1.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    if num_classes < 1:
        raise ValueError("num_classes must be a positive integer.")
    return MODEL_REGISTRY[name](num_classes=num_classes)
