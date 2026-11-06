"""Lightweight CNN model zoo for grayscale image classification and benchmarking."""

from __future__ import annotations

from .common import model_stats, seed_everything, select_device
from .ds_cnn import DepthwiseSeparableCNN, DepthwiseSeparableConv
from .mobilenet_like import MobileNetLikeCNN
from .registry import MODEL_REGISTRY, create_model, model_names
from .small_cnn import SmallCNN
from .tiny_cnn import TinyCNN
from .tiny_resnet import ResidualBlock, TinyResidualCNN
from .torchvision_mobilenet import TorchvisionMobileNet
from .ultra_tiny_cnn import UltraTinyCNN

__all__ = [
    "TinyCNN",
    "SmallCNN",
    "DepthwiseSeparableCNN",
    "DepthwiseSeparableConv",
    "MobileNetLikeCNN",
    "TinyResidualCNN",
    "ResidualBlock",
    "UltraTinyCNN",
    "TorchvisionMobileNet",
    "create_model",
    "model_names",
    "model_stats",
    "seed_everything",
    "select_device",
    "MODEL_REGISTRY",
]
