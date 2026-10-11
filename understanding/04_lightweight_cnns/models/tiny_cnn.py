"""Baseline clásico: tres convoluciones y clasificación tras promedio global."""
import torch
from torch import nn


class TinyCNN(nn.Module):
    """Recibe [B, 1, H, W] y devuelve logits [B, num_classes]."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """El promedio global elimina la dependencia de H y W en Linear."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)  # [B, 64, 1, 1]
        x = torch.flatten(x, 1)  # [B, 64]
        return self.classifier(x)
