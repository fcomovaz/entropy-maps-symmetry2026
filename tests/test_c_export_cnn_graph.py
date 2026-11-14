"""Unit tests covering all 7 CNN architectures: FX trace, BN folding, quantization, and arena planner."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from src.c_export.cnn_graph import extract_cnn_ir_from_module, load_cnn_ir, run_cnn_graph_f32
from src.c_export.cnn_quantize import quantize_cnn_graph, run_cnn_graph_q15
from src.c_export.config import load_c_imp_config
from src.c_export.memory_plan import build_memory_plan
from src.models.cnn.architectures.registry import MODEL_REGISTRY, create_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestCExportCnnGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_c_imp_config(Path("configs/c_imp/kolektorsdd2.toml"), PROJECT_ROOT)
        rng = np.random.default_rng(123)
        cls.probe_u8 = rng.integers(0, 256, size=(2, 128, 128), dtype=np.uint8)
        cls.probe_f32 = ((cls.probe_u8.astype(np.float32) / 255.0) - 0.5) / 0.5
        cls.probe_f32_nchw = cls.probe_f32[:, None, :, :]

    def test_all_seven_cnns_trace_fold_quantize_and_plan(self) -> None:
        torch.manual_seed(42)
        for model_name in MODEL_REGISTRY:
            with self.subTest(model=model_name):
                model = create_model(model_name, num_classes=2)
                # Populate non-trivial running BN stats so folding math is genuinely exercised
                for mod in model.modules():
                    if isinstance(mod, torch.nn.BatchNorm2d):
                        assert mod.running_mean is not None and mod.running_var is not None
                        mod.running_mean.normal_(mean=0.05, std=0.2)
                        mod.running_var.uniform_(0.2, 2.0)
                model.eval()

                graph = extract_cnn_ir_from_module(
                    model=model,
                    model_name=model_name,
                    input_shape=(1, 128, 128),
                    norm_mean=0.5,
                    norm_std=0.5,
                    selected_threshold=0.5,
                    fuse_relu=True,
                )
                with torch.no_grad():
                    pt_logits = model(torch.from_numpy(self.probe_f32_nchw[:1])).numpy()[0]
                ir_logits, _ = run_cnn_graph_f32(graph, self.probe_f32_nchw[0])
                np.testing.assert_allclose(pt_logits, ir_logits, atol=1e-4, rtol=0.0)

                qmodel = quantize_cnn_graph(graph, self.probe_f32_nchw, self.cfg)
                if model_name == "torchvision_mobilenet":
                    self.assertEqual(qmodel.weight_format, "int8_per_channel")
                else:
                    self.assertEqual(qmodel.weight_format, "q15")

                raw_q, f64_q = run_cnn_graph_q15(qmodel, self.probe_u8[0])
                self.assertEqual(raw_q.shape, (2,))
                self.assertTrue(np.all(np.isfinite(f64_q)))

                plan = build_memory_plan(graph, qmodel.weight_format, qmodel.weight_bytes, self.cfg)
                self.assertLessEqual(plan.weight_flash_bytes, self.cfg.cnn.flash_budget_bytes)
                self.assertLessEqual(plan.peak_arena_q15_bytes, self.cfg.cnn.psram_budget_bytes)
                # Verify live intervals do not collide in assigned arena offsets
                allocs = list(plan.allocations.values())
                for i in range(len(allocs)):
                    for j in range(i + 1, len(allocs)):
                        a, b = allocs[i], allocs[j]
                        overlaps_time = not (a.last_use < b.first_def or b.last_use < a.first_def)
                        if overlaps_time:
                            overlaps_q15 = not (
                                (a.offset_q15 + a.bytes_q15) <= b.offset_q15
                                or (b.offset_q15 + b.bytes_q15) <= a.offset_q15
                            )
                            self.assertFalse(
                                overlaps_q15,
                                f"{model_name} arena collision between {a.tensor_name} and {b.tensor_name}",
                            )

    def test_load_cnn_ir_from_real_checkpoint(self) -> None:
        ckpt_dir = self.cfg.cnn_cfg.effective_results_dir(self.cfg.object_name, "ultra_tiny")
        if not (ckpt_dir / "checkpoint.pt").is_file():
            self.skipTest(f"Checkpoint not present at {ckpt_dir / 'checkpoint.pt'}")
        graph = load_cnn_ir(self.cfg, "ultra_tiny")
        self.assertEqual(graph.model_name, "ultra_tiny")
        self.assertEqual(graph.input_shape, (1, 128, 128))
        self.assertTrue(0.0 < graph.selected_threshold < 1.0)
        self.assertTrue(graph.checkpoint_sha256)


if __name__ == "__main__":
    unittest.main()
