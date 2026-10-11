"""Model zoo educativo para imágenes grayscale."""
from .tiny_cnn import TinyCNN
from .small_cnn import SmallCNN
from .ds_cnn import DepthwiseSeparableCNN, DepthwiseSeparableConv
from .mobilenet_like import MobileNetLikeCNN
from .tiny_resnet import TinyResidualCNN, ResidualBlock
from .ultra_tiny_cnn import UltraTinyCNN
from .torchvision_mobilenet import TorchvisionMobileNet
from .registry import create_model, model_names

__all__ = [
    "TinyCNN", "SmallCNN", "DepthwiseSeparableCNN", "DepthwiseSeparableConv",
    "MobileNetLikeCNN", "TinyResidualCNN", "ResidualBlock", "UltraTinyCNN",
    "TorchvisionMobileNet", "create_model", "model_names",
]
