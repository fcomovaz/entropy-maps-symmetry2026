"""Registro explícito: nombres cortos y constructores fáciles de localizar."""
from importlib.util import find_spec
from torch import nn
from .tiny_cnn import TinyCNN
from .small_cnn import SmallCNN
from .ds_cnn import DepthwiseSeparableCNN
from .mobilenet_like import MobileNetLikeCNN
from .tiny_resnet import TinyResidualCNN
from .ultra_tiny_cnn import UltraTinyCNN
from .torchvision_mobilenet import TorchvisionMobileNet

MODEL_REGISTRY = {
    "tiny_cnn": TinyCNN,
    "small_cnn": SmallCNN,
    "ds_cnn": DepthwiseSeparableCNN,
    "mobilenet_like": MobileNetLikeCNN,
    "tiny_resnet": TinyResidualCNN,
    "ultra_tiny": UltraTinyCNN,
    "torchvision_mobilenet": TorchvisionMobileNet,
}


def model_names() -> list[str]:
    """Omite la referencia solo si torchvision no está instalado."""
    return [name for name in MODEL_REGISTRY
            if name != "torchvision_mobilenet" or find_spec("torchvision") is not None]


def create_model(name: str, num_classes: int = 2) -> nn.Module:
    """Construye un clasificador sin pesos preentrenados ni descargas."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo desconocido {name!r}. Opciones: {list(MODEL_REGISTRY)}")
    if num_classes < 1:
        raise ValueError("num_classes debe ser positivo.")
    return MODEL_REGISTRY[name](num_classes=num_classes)
