"""Minimal TinyML budget convolutional classifier using stride-2 separable blocks."""

from __future__ import annotations

import torch
from torch import nn

from .ds_cnn import DepthwiseSeparableConv


class UltraTinyCNN(nn.Module):
    """Ultra-compact CNN designed for low parameter footprint.

    Receives grayscale images [B, 1, H, W] and outputs logits [B, num_classes].
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.block1 = DepthwiseSeparableConv(8, 16, stride=2)
        self.block2 = DepthwiseSeparableConv(16, 32, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, 1, H, W].

        Returns:
            Logits of shape [B, num_classes].
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        return self.classifier(torch.flatten(x, 1))
