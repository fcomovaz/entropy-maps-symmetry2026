"""Pruebas sin pytest: CPU obligatorio y CUDA cuando está disponible."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from common import seed_everything
from models import DepthwiseSeparableConv, ResidualBlock, create_model, model_names
from train_demo import synthetic_dataset


class ModelTests(unittest.TestCase):
    """Shapes, gradientes reales, update del optimizer y operaciones pedagógicas."""

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        seed_everything(42)

    def check_sizes(self, device: str) -> None:
        """Cinco tamaños x dos números de clases x todas las arquitecturas."""
        cases = 0
        for name in model_names():
            for classes in (2, 5):
                model = create_model(name, classes).to(device).eval()
                for size in (32, 64, 128, 224, 256):
                    with self.subTest(device=device, model=name, classes=classes, size=size):
                        with torch.inference_mode():
                            output = model(torch.randn(2, 1, size, size, device=device))
                        self.assertEqual(tuple(output.shape), (2, classes))
                        self.assertTrue(torch.isfinite(output).all().item())
                        cases += 1
                del model
        print(f"\n{device}: {cases} forward/shape cases passed", flush=True)

    def check_backward(self, device: str) -> None:
        """Un paso real de entrenamiento por arquitectura y cantidad de clases."""
        for name in model_names():
            for classes in (2, 5):
                with self.subTest(device=device, model=name, classes=classes):
                    model = create_model(name, classes).to(device).train()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                    before = next(model.parameters()).detach().clone()
                    output = model(torch.randn(2, 1, 32, 32, device=device))
                    loss = nn.CrossEntropyLoss()(output, torch.tensor([0, classes - 1], device=device))
                    self.assertTrue(torch.isfinite(loss).item())
                    loss.backward()
                    for parameter in model.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    optimizer.step()
                    self.assertFalse(torch.equal(before, next(model.parameters()).detach()))
                    del optimizer, model

    def test_cpu_sizes(self) -> None:
        self.check_sizes("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_sizes(self) -> None:
        self.check_sizes("cuda")
        torch.cuda.synchronize()

    def test_cpu_backward(self) -> None:
        self.check_backward("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_backward(self) -> None:
        self.check_backward("cuda")
        torch.cuda.synchronize()

    def test_parameter_budgets(self) -> None:
        count = lambda name: sum(p.numel() for p in create_model(name, 2).parameters())
        self.assertTrue(20_000 <= count("tiny_cnn") <= 150_000)
        self.assertLess(count("ultra_tiny"), 50_000)
        self.assertLess(count("mobilenet_like"), 1_000_000)

    def test_depthwise_and_pointwise(self) -> None:
        block = DepthwiseSeparableConv(8, 16)
        self.assertEqual(block.depthwise.groups, 8)
        self.assertEqual(tuple(block.depthwise.weight.shape), (8, 1, 3, 3))
        self.assertEqual(tuple(block.pointwise.weight.shape), (16, 8, 1, 1))

    def test_residual_identity_and_projection(self) -> None:
        for channels, stride in ((8, 1), (16, 2)):
            block = ResidualBlock(8, channels, stride).eval()
            nn.init.zeros_(block.conv2.weight)  # Fuerza F(x)=0.
            x = torch.randn(2, 8, 32, 32)
            with torch.inference_mode():
                torch.testing.assert_close(block(x), torch.relu(block.shortcut(x)))

    def test_synthetic_seed(self) -> None:
        first = synthetic_dataset(8, 32, 42)
        second = synthetic_dataset(8, 32, 42)
        different = synthetic_dataset(8, 32, 43)
        torch.testing.assert_close(first.tensors[0], second.tensors[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(first.tensors[0], different.tensors[0]))
        self.assertEqual(set(first.tensors[1].tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
