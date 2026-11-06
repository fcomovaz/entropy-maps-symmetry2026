"""Depthwise separable convolutions separating spatial filtering from cross-channel mixing."""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 3x3 convolution followed by pointwise 1x1 convolution with BN and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, Cin, H, W].

        Returns:
            Output tensor [B, Cout, H', W'].
        """
        x = self.relu(self.depthwise_bn(self.depthwise(x)))
        x = self.relu(self.pointwise_bn(self.pointwise(x)))
        return x


class DepthwiseSeparableCNN(nn.Module):
    """Educational lightweight CNN built with three depthwise separable blocks.

    Receives grayscale images [B, 1, H, W] and outputs logits [B, num_classes].
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.block1 = DepthwiseSeparableConv(16, 32, stride=2)
        self.block2 = DepthwiseSeparableConv(32, 64, stride=2)
        self.block3 = DepthwiseSeparableConv(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

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
        x = self.block3(x)
        x = self.pool(x)
        return self.classifier(torch.flatten(x, 1))
