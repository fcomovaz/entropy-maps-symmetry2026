"""Shared utility functions for lightweight CNN architectures and profiling."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    """Set random seed across Python, NumPy, and PyTorch for deterministic execution.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(name: str) -> torch.device:
    """Select execution device with strict fail-fast validation.

    Args:
        name: Device specification ('auto', 'cpu', 'cuda').

    Returns:
        torch.device instance.

    Raises:
        RuntimeError: If 'cuda' was requested but CUDA is not available.
        ValueError: If device name is unrecognized.
    """
    dev_str = name.lower()
    if dev_str == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    elif dev_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    elif dev_str not in {"cpu", "cuda"} and not dev_str.startswith("cuda:"):
        raise ValueError(f"Unsupported device name '{name}'. Expected 'auto', 'cpu', or 'cuda'.")
    return torch.device(dev_str)


def model_stats(model: nn.Module) -> Tuple[int, int, float]:
    """Compute parameter counts and decimal memory size.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Tuple of (total_parameters, trainable_parameters, size_mb).
    """
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return parameters, trainable, size_mb
