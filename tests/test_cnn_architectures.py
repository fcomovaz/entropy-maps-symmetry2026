"""Unit and regression tests for lightweight CNN architectures.

Verifies:
1. Exact parameter counts matching the issue and model review specification.
2. Output shape [B, num_classes] across variable spatial resolutions (32, 64, 128).
3. Gradient backpropagation and optimizer parameter updates.
4. Numerical stability and absence of NaN/Inf activations.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_cnn_architectures.py
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_cnn_architectures.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn.architectures import (
    DepthwiseSeparableCNN,
    DepthwiseSeparableConv,
    MobileNetLikeCNN,
    ResidualBlock,
    SmallCNN,
    TinyCNN,
    TinyResidualCNN,
    TorchvisionMobileNet,
    UltraTinyCNN,
    create_model,
    model_names,
    model_stats,
    seed_everything,
)

EXPECTED_PARAMETER_COUNTS: dict[str, int] = {
    "tiny_cnn": 23_538,
    "small_cnn": 97_650,
    "ds_cnn": 12_866,
    "mobilenet_like": 70_786,
    "tiny_resnet": 77_234,
    "ultra_tiny": 1_154,
    "torchvision_mobilenet": 1_519_618,
}


class TestCnnArchitectures(unittest.TestCase):
    """Test suite for verifying lightweight CNN specifications and parameter counts."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        seed_everything(42)

    def test_parameter_counts_exact(self) -> None:
        """Verify each model in the registry has exactly the expected parameter count."""
        for name, expected_params in EXPECTED_PARAMETER_COUNTS.items():
            if name not in model_names():
                continue
            with self.subTest(model=name):
                model = create_model(name, num_classes=2)
                params, trainable, _ = model_stats(model)
                self.assertEqual(
                    params,
                    expected_params,
                    f"Parameter count mismatch for {name}: got {params:,}, expected {expected_params:,}",
                )
                self.assertEqual(
                    trainable,
                    expected_params,
                    f"All parameters should be trainable in {name}",
                )

    def test_forward_output_shapes_and_resolutions(self) -> None:
        """Verify output is [B, num_classes] for different spatial resolutions."""
        batch_size = 2
        for name in model_names():
            for num_classes in (2, 5):
                model = create_model(name, num_classes=num_classes).eval()
                for size in (32, 64, 128):
                    with self.subTest(model=name, classes=num_classes, size=size):
                        x = torch.randn(batch_size, 1, size, size)
                        with torch.inference_mode():
                            out = model(x)
                        self.assertEqual(tuple(out.shape), (batch_size, num_classes))
                        self.assertTrue(
                            torch.isfinite(out).all().item(),
                            f"Model {name} produced non-finite outputs at size {size}",
                        )

    def test_backward_gradient_flow(self) -> None:
        """Verify loss backward computes finite gradients and optimizer updates weights."""
        criterion = nn.CrossEntropyLoss()
        for name in model_names():
            with self.subTest(model=name):
                model = create_model(name, num_classes=2).train()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                first_param = next(model.parameters())
                before_weights = first_param.detach().clone()

                x = torch.randn(2, 1, 64, 64)
                y = torch.tensor([0, 1], dtype=torch.long)

                out = model(x)
                loss = criterion(out, y)
                self.assertTrue(torch.isfinite(loss).item())

                loss.backward()
                for p in model.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all().item())

                optimizer.step()
                after_weights = next(model.parameters()).detach()
                self.assertFalse(
                    torch.equal(before_weights, after_weights),
                    f"Optimizer step did not change weights in {name}",
                )

    def test_depthwise_separable_structure(self) -> None:
        """Verify groups equals in_channels and dimensions for DepthwiseSeparableConv."""
        block = DepthwiseSeparableConv(in_channels=16, out_channels=32, stride=2)
        self.assertEqual(block.depthwise.groups, 16)
        self.assertEqual(tuple(block.depthwise.weight.shape), (16, 1, 3, 3))
        self.assertEqual(tuple(block.pointwise.weight.shape), (32, 16, 1, 1))

    def test_residual_block_shortcuts(self) -> None:
        """Verify identity vs 1x1 projection shortcuts in ResidualBlock."""
        # Identity shortcut (stride=1, in=out)
        b_id = ResidualBlock(16, 16, stride=1)
        self.assertIsInstance(b_id.shortcut, nn.Identity)

        # Projection shortcut (stride=2 or in!=out)
        b_proj = ResidualBlock(16, 32, stride=2)
        self.assertIsInstance(b_proj.shortcut, nn.Sequential)


if __name__ == "__main__":
    unittest.main(verbosity=2)
