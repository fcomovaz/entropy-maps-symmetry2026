"""Clasificador mínimo para explorar presupuestos conceptuales TinyML."""
import torch
from torch import nn
from .ds_cnn import DepthwiseSeparableConv


class UltraTinyCNN(nn.Module):
    """Pocos canales y reducción temprana; no es una implementación embebida."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        self.block1 = DepthwiseSeparableConv(8, 16, stride=2)
        self.block2 = DepthwiseSeparableConv(16, 32, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1, H, W] -> [B, num_classes] sin softmax."""
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        return self.classifier(torch.flatten(x, 1))
