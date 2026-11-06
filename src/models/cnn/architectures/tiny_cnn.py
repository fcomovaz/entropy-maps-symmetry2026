"""Baseline classic CNN: three convolutions with global average pooling."""

from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Classical CNN baseline.

    Receives grayscale images [B, 1, H, W] and outputs logits [B, num_classes].
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, 1, H, W].

        Returns:
            Logits of shape [B, num_classes].
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)  # [B, 64, 1, 1]
        x = torch.flatten(x, 1)  # [B, 64]
        return self.classifier(x)
