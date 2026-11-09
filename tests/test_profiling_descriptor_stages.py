"""Unit tests for staged x94 descriptor profiling, numerical parity, and scalability.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_profiling_descriptor_stages.py -v
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_profiling_descriptor_stages.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.asymmetry_maps import compute_asymmetry_features
from src.features.entropy_maps import build_entropy_plan, compute_entropy_features
from src.profiling.descriptor_stages import (
    DescriptorProfileReport,
    estimate_descriptor_ops,
    extract_x94_end_to_end,
    extract_x94_staged,
    profile_descriptor_scalability,
    profile_descriptor_stages,
)


class TestProfilingDescriptorStages(unittest.TestCase):
    """Test suite validating exact numerical equivalence and stage breakdown invariants."""

    def test_staged_extraction_matches_formal_x94_exact(self) -> None:
        """Verify `extract_x94_staged` matches formal end-to-end x94 within <1e-12."""
        rng = np.random.default_rng(123)
        images = rng.integers(0, 256, size=(8, 128, 128), dtype=np.uint8)
        levels = (1, 2, 4, 8)
        bins = 16
        asym_scales = (2, 4, 8)
        asym_kinds = ("lr", "tb", "rot180")
        plan = build_entropy_plan(128, levels, bins)

        out_staged = extract_x94_staged(
            images,
            grid_levels=levels,
            bins=bins,
            asymmetry_scales=asym_scales,
            asymmetry_kinds=asym_kinds,
            plan=plan,
        )
        out_e2e = extract_x94_end_to_end(
            images,
            grid_levels=levels,
            bins=bins,
            asymmetry_scales=asym_scales,
            asymmetry_kinds=asym_kinds,
            plan=plan,
        )

        # Independent oracle directly calling src.features functions
        x85 = compute_entropy_features(images, grid_levels=levels, bins=bins, plan=plan)
        e_by_scale = {}
        off = 0
        for g in levels:
            cols = g * g
            e_by_scale[g] = x85[:, off : off + cols]
            off += cols
        x9 = compute_asymmetry_features(e_by_scale, scales=asym_scales, kinds=asym_kinds)
        oracle = np.concatenate([x85, x9], axis=1)

        self.assertEqual(out_staged.shape, (8, 94))
        self.assertEqual(out_e2e.shape, (8, 94))
        self.assertTrue(np.all((out_staged >= 0.0) & (out_staged <= 1.0)))
        np.testing.assert_allclose(out_staged, out_e2e, atol=1e-12)
        np.testing.assert_allclose(out_staged, oracle, atol=1e-12)

    def test_2d_and_empty_inputs_handled_cleanly(self) -> None:
        """Verify [H, W] promotes to [1, 94] and [0, H, W] produces empty [0, 94]."""
        rng = np.random.default_rng(7)
        img2d = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        out_1 = extract_x94_staged(img2d)
        self.assertEqual(out_1.shape, (1, 94))

        empty = np.zeros((0, 64, 64), dtype=np.uint8)
        self.assertEqual(extract_x94_staged(empty).shape, (0, 94))
        self.assertEqual(extract_x94_end_to_end(empty).shape, (0, 94))

    def test_profile_descriptor_stages_percentage_sum_and_scaler(self) -> None:
        """Verify stage shares sum to 100%, analytical ops are positive, and cProfile runs."""
        rng = np.random.default_rng(99)
        images = rng.integers(0, 256, size=(6, 64, 64), dtype=np.uint8)
        x_sample = extract_x94_end_to_end(images)
        scaler = StandardScaler().fit(x_sample)

        report = profile_descriptor_stages(
            images_u8=images,
            dataset_tag="synthetic_64",
            grid_levels=(1, 2, 4, 8),
            bins=16,
            asymmetry_scales=(2, 4, 8),
            asymmetry_kinds=("lr", "tb", "rot180"),
            threads=1,
            batch_n_size=4,
            warmup_iters=2,
            timed_iters=10,
            batch_timed_iters=4,
            disable_gc_flag=True,
            bootstrap_resamples=30,
            cprofile_top_n=10,
            seed=42,
            scaler=scaler,
        )

        self.assertIsInstance(report, DescriptorProfileReport)
        self.assertLessEqual(report.max_numerical_diff, 1e-12)
        self.assertGreater(report.e2e_batch1_stats.median_ms, 0.0)
        self.assertGreater(report.staged_sum_batch1_ms, 0.0)
        self.assertIn("quantize_pixels", [s.stage for s in report.stages_batch1])
        self.assertIn("standard_scaler", [s.stage for s in report.stages_batch1])

        core_share_sum = sum(
            s.share_of_staged_sum_pct for s in report.stages_batch1 if s.stage != "standard_scaler"
        )
        self.assertAlmostEqual(core_share_sum, 100.0, places=4)
        self.assertTrue(len(report.cprofile_report_text.strip()) > 0)

        ops = estimate_descriptor_ops(
            128, grid_levels=(1, 2, 4, 8), bins=16, asymmetry_scales=(2, 4, 8), asymmetry_kinds=("lr", "tb", "rot180")
        )
        self.assertEqual(ops.quantize_ops, 128 * 128)
        self.assertEqual(ops.histogram_ops, 4 * 2 * 128 * 128)
        self.assertGreater(ops.total_ops, ops.histogram_ops)

    def test_profile_descriptor_scalability_sweep(self) -> None:
        """Verify parameter sweep across image sizes, bin counts, and grid levels executes cleanly."""
        rng = np.random.default_rng(11)
        images = rng.integers(0, 256, size=(3, 64, 64), dtype=np.uint8)
        records = profile_descriptor_scalability(
            images_u8=images,
            base_image_size=64,
            base_bins=16,
            base_grid_levels=(1, 2, 4),
            asymmetry_kinds=("lr", "tb", "rot180"),
            image_sizes=(32, 64),
            bins_list=(8, 16),
            grid_levels_list=((1, 2), (1, 2, 4)),
            threads=1,
            warmup_iters=2,
            timed_iters=6,
            disable_gc_flag=True,
            bootstrap_resamples=20,
            seed=42,
        )

        self.assertEqual(len(records), 6)
        self.assertEqual([r.sweep_dimension for r in records[:2]], ["image_size", "image_size"])
        self.assertEqual([r.sweep_dimension for r in records[2:4]], ["bins", "bins"])
        self.assertEqual([r.sweep_dimension for r in records[4:]], ["grid_levels", "grid_levels"])
        self.assertTrue(all(r.median_ms > 0.0 and r.analytical_ops_per_sample > 0 for r in records))


if __name__ == "__main__":
    unittest.main(verbosity=2)
