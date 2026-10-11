"""Conexiones residuales explícitas, sin construir un ResNet completo."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Aprende F(x); suma una identidad o una proyección compatible."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            # Ajusta canales y resolución para que la suma tenga igual shape.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """output = ReLU(F(x) + shortcut(x)); shortcut(x)=x si cabe identidad."""
        residual = self.shortcut(x)
        fx = self.relu(self.bn1(self.conv1(x)))
        fx = self.bn2(self.conv2(fx))
        output = fx + residual  # Conexión residual: F(x) + x o su proyección.
        return self.relu(output)


class TinyResidualCNN(nn.Module):
    """Tres bloques residuales; entrada grayscale y salida de logits."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.block1 = ResidualBlock(16, 16)
        self.block2 = ResidualBlock(16, 32, stride=2)
        self.block3 = ResidualBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 1, H, W] -> [B, num_classes], independientemente de H y W."""
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        return self.classifier(torch.flatten(x, 1))
