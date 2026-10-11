"""Inspiración MobileNet: alternar reducción espacial y bloques separables."""
import torch
from torch import nn
from .ds_cnn import DepthwiseSeparableConv


class MobileNetLikeCNN(nn.Module):
    """Versión propia pequeña; no reproduce MobileNetV2/V3 ni usa expansión."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.block1 = DepthwiseSeparableConv(16, 32)
        self.block2 = DepthwiseSeparableConv(32, 64, stride=2)
        self.block3 = DepthwiseSeparableConv(64, 64)
        self.block4 = DepthwiseSeparableConv(64, 128, stride=2)
        self.block5 = DepthwiseSeparableConv(128, 128)
        self.block6 = DepthwiseSeparableConv(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforma [B, 1, H, W] en logits [B, num_classes]."""
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool(x)
        return self.classifier(torch.flatten(x, 1))
