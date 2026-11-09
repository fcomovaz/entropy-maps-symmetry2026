"""Unit tests for CPU CNN layer hooks, ATen profiling, thread scaling, and benchmark extensions.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_profiling_cnn_layers.py -v
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_profiling_cnn_layers.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn.architectures.registry import create_model
from src.models.cnn.benchmark import BenchmarkRecord, benchmark_model_latency, run_benchmark
from src.models.cnn.config import (
    AugmentationConfig,
    BenchmarkConfig,
    CnnConfig,
    DataConfig,
    OutputConfig,
    TrainingConfig,
)
from src.profiling.cnn_layers import (
    CnnLayerProfileReport,
    profile_cnn_layers,
    profile_cnn_thread_scaling,
)


class _ToySeparableNet(nn.Module):
    """Minimal synthetic CNN exercising standard, depthwise, pointwise convs, BN, reused ReLU, pool, and linear."""

    def __init__(self) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(4)
        self.act = nn.ReLU()  # Reused multiple times intentionally
        self.dw = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4, bias=False)
        self.pw = nn.Conv2d(4, 8, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.stem_bn(self.stem_conv(x)))
        x = self.act(self.pw(self.dw(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _dummy_cnn_config(image_size: int = 32) -> CnnConfig:
    """Create minimal in-memory CnnConfig for backward-compatibility benchmark tests."""
    return CnnConfig(
        dataset_key="synth",
        dataset_name="Synthetic",
        preprocessed_dir=Path("."),
        preprocessed_dir_inverted=Path("."),
        metadata_csv_path=Path("."),
        metadata_csv_path_inverted=Path("."),
        results_dir=Path("."),
        inverted=False,
        overwrite=True,
        seed=42,
        seeds=(42,),
        default_model="ultra_tiny",
        models=("ultra_tiny",),
        default_object=None,
        subdatasets=(),
        data=DataConfig(
            split_column="split",
            label_column="type",
            object_column="",
            positive_label="defect",
            negative_label="good",
            split_mode="presplit",
            train_split_name="train",
            test_split_name="test",
            test_size=0.2,
            val_size=0.15,
            image_size=image_size,
            normalize_mean=0.5,
            normalize_std=0.5,
            use_metadata_index=False,
            metadata_csv_key="full_csv_path",
            metadata_csv_key_inverted="full_csv_path_inverted",
            allowed_splits=("train", "test"),
        ),
        training=TrainingConfig(
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=1e-4,
            optimizer="adamw",
            scheduler="cosine",
            early_stopping_patience=1,
            class_weighting="balanced",
            num_workers=0,
            device="cpu",
            amp=False,
            deterministic=True,
            threshold_mode="fixed_05",
        ),
        augmentation=AugmentationConfig(hflip=False, vflip=False, rotate_deg=0, brightness=0.0),
        benchmark=BenchmarkConfig(
            warmup_iters=2,
            timed_iters=5,
            batch_sizes=(1,),
            devices=("cpu",),
            threads=1,
            measure_flops=False,
        ),
        output=OutputConfig(
            results_dir=Path("."),
            dpi=100,
            float_precision=6,
            save_checkpoints=False,
        ),
    )


class TestProfilingCnnLayers(unittest.TestCase):
    """Test suite for CPU CNN leaf hooks, ATen profiling, thread scaling, and benchmark extensions."""

    def test_synthetic_net_reports_all_leaf_layers_and_aten_ops_on_cpu(self) -> None:
        """Verify all leaf layers appear, reused activations accumulate calls, shares sum to ~100%, and Chrome trace saves."""
        torch.manual_seed(42)
        net = _ToySeparableNet().cpu().eval()
        inputs = torch.randn(4, 1, 32, 32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.json"
            report = profile_cnn_layers(
                model=net,
                model_name="toy_sep",
                inputs=inputs,
                threads=1,
                batch_size=1,
                warmup_iters=2,
                timed_iters=10,
                torch_prof_warmup=1,
                torch_prof_active=3,
                hook_overhead_warn_pct=15.0,
                disable_gc=True,
                bootstrap_resamples=20,
                seed=42,
                chrome_trace_path=trace_path,
            )

            self.assertIsInstance(report, CnnLayerProfileReport)
            self.assertGreater(report.hookless_stats.median_ms, 0.0)
            self.assertGreater(report.sum_layers_median_ms, 0.0)
            self.assertGreater(report.total_flops, 0)
            self.assertGreater(report.max_activation_kb, 0.0)

            layer_names = {lay.layer_name: lay for lay in report.layers}
            for expected in ("stem_conv", "stem_bn", "act", "dw", "pw", "pool", "fc"):
                self.assertIn(expected, layer_names)

            self.assertEqual(layer_names["act"].call_count, 2)
            self.assertEqual(layer_names["dw"].category, "dwconv")
            self.assertEqual(layer_names["pw"].category, "pwconv")
            self.assertAlmostEqual(
                sum(lay.share_of_layers_pct for lay in report.layers), 100.0, places=3
            )

            self.assertTrue(len(report.aten_ops) > 0)
            self.assertTrue(len(report.aten_table_text.strip()) > 0)
            self.assertTrue(trace_path.is_file())
            trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertTrue(bool(trace_data))

    def test_ultra_tiny_cpu_only_profiling_and_thread_scaling(self) -> None:
        """Verify real registry architecture `ultra_tiny` profiles on CPU without CUDA."""
        torch.manual_seed(42)
        model = create_model("ultra_tiny", num_classes=2).cpu().eval()
        inputs = torch.randn(3, 1, 64, 64)

        report = profile_cnn_layers(
            model=model,
            model_name="ultra_tiny",
            inputs=inputs,
            threads=1,
            batch_size=1,
            warmup_iters=2,
            timed_iters=8,
            torch_prof_warmup=1,
            torch_prof_active=2,
            bootstrap_resamples=20,
        )
        self.assertGreater(len(report.layers), 4)
        self.assertGreater(report.total_params, 0)

        scaling = profile_cnn_thread_scaling(
            model=model,
            model_name="ultra_tiny",
            inputs=inputs,
            threads_list=(1, 2),
            batch_size=1,
            warmup_iters=2,
            timed_iters=6,
            bootstrap_resamples=20,
        )
        self.assertEqual([s.threads for s in scaling], [1, 2])
        self.assertAlmostEqual(scaling[0].speedup_vs_1thread, 1.0, places=6)

    def test_benchmark_py_backward_compatible_and_real_inputs(self) -> None:
        """Verify existing `benchmark.py` signature works with inputs=None and with real inputs."""
        model = create_model("ultra_tiny", num_classes=2).cpu().eval()
        cfg = _dummy_cnn_config(image_size=32)

        # Legacy call (no inputs supplied)
        rec_legacy = run_benchmark(
            model=model,
            model_name="ultra_tiny",
            cfg=cfg,
            device=torch.device("cpu"),
            batch_size=1,
        )
        self.assertIsInstance(rec_legacy, BenchmarkRecord)
        self.assertEqual(rec_legacy.device, "cpu")
        self.assertGreater(rec_legacy.median_ms, 0.0)

        # Extended call with real inputs and GC control
        real_x = torch.randn(4, 1, 32, 32)
        lat_ms, vram = benchmark_model_latency(
            model=model,
            device=torch.device("cpu"),
            batch_size=2,
            image_size=32,
            warmup_iters=2,
            timed_iters=6,
            threads=1,
            inputs=real_x,
            disable_gc=True,
        )
        self.assertEqual(len(lat_ms), 6)
        self.assertIsNone(vram)
        self.assertTrue(all(v > 0.0 for v in lat_ms))

        rec_real = run_benchmark(
            model=model,
            model_name="ultra_tiny",
            cfg=cfg,
            device=torch.device("cpu"),
            batch_size=2,
            inputs=real_x,
            threads=1,
            warmup_iters=2,
            timed_iters=5,
            disable_gc=True,
        )
        self.assertEqual(rec_real.batch_size, 2)
        self.assertGreater(rec_real.throughput_samples_per_s, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
