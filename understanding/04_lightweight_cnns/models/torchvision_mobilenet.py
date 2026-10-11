"""Referencia oficial opcional, sin descargas de pesos."""
import torch
from torch import nn


class TorchvisionMobileNet(nn.Module):
    """MobileNetV3-Small original, cambiando solo entrada y cantidad de clases."""

    def __init__(self, num_classes: int = 2, weights: None = None) -> None:
        super().__init__()
        if weights is not None:
            raise ValueError("Este wrapper educativo solo admite weights=None.")
        from torchvision.models import mobilenet_v3_small

        self.network = mobilenet_v3_small(weights=None, num_classes=num_classes)
        original = self.network.features[0][0]
        self.network.features[0][0] = nn.Conv2d(
            1, original.out_channels, original.kernel_size,
            stride=original.stride, padding=original.padding, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Recibe [B, 1, H, W] y devuelve logits [B, num_classes]."""
        return self.network(x)
