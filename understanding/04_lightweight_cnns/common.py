"""Utilidades pequeñas compartidas por los scripts; no ocultan arquitecturas."""
import argparse
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent


def positive_int(value: str) -> int:
    """Tipo CLI para cantidades positivas."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Debe ser un entero positivo.")
    return number


def input_size(value: str) -> int:
    """El laboratorio admite resoluciones cuadradas de al menos 32 píxeles."""
    size = positive_int(value)
    if size < 32:
        raise argparse.ArgumentTypeError("input-size debe ser >= 32.")
    return size


def seed_everything(seed: int) -> None:
    """Fija RNG y evita autotuning; no promete igualdad entre dispositivos."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(name: str) -> torch.device:
    """auto permite CPU; cuda explícito falla si no hay GPU disponible."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fue solicitada pero no está disponible.")
    return torch.device(name)


def model_stats(model: nn.Module) -> tuple[int, int, float]:
    """Parámetros, entrenables y MB decimales; excluye buffers/activaciones."""
    parameters = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return parameters, trainable, size_mb


def image_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    """Grayscale -> resize bilinear -> float32 [1, size, size] en [0, 1]."""
    image = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    pixels = np.array(image, dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(pixels).unsqueeze(0)
