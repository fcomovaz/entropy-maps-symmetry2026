"""Official torchvision MobileNetV3-Small reference architecture wrapper."""

from __future__ import annotations

import torch
from torch import nn


class TorchvisionMobileNet(nn.Module):
    """MobileNetV3-Small adapted for 1-channel grayscale input and num_classes output.

    Initializes strictly with random weights (weights=None, zero network downloads).
    """

    def __init__(self, num_classes: int = 2, weights: None = None) -> None:
        super().__init__()
        if weights is not None:
            raise ValueError("TorchvisionMobileNet requires weights=None for reproducible benchmarking.")
        from torchvision.models import mobilenet_v3_small

        self.network = mobilenet_v3_small(weights=None, num_classes=num_classes)
        original = self.network.features[0][0]
        self.network.features[0][0] = nn.Conv2d(
            1,
            original.out_channels,
            original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, 1, H, W].

        Returns:
            Logits of shape [B, num_classes].
        """
        return self.network(x)
