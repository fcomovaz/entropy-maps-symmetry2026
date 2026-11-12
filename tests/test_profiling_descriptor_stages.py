"""Unit tests for staged x94 descriptor profiling, numerical parity, and scalability.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_profiling_descriptor_stages.py -v
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_profiling_descriptor_stages.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.asymmetry_maps import compute_asymmetry_features
from src.features.entropy_maps import build_entropy_plan, compute_entropy_features
from src.features.native_entropy import resolve_default_native_dll_path
from src.profiling.descriptor_stages import (
    DescriptorProfileReport,
    estimate_descriptor_ops,
    extract_x94_end_to_end,
    extract_x94_staged,
    profile_descriptor_scalability,
    profile_descriptor_stages,
)
from src.profiling.environment import capture_environment
from src.profiling.report import generate_profiling_figures, write_profiling_readme


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
            backend="numpy_baseline",
        )

        self.assertIsInstance(report, DescriptorProfileReport)
        self.assertEqual(report.entropy_backend, "numpy_baseline")
        self.assertEqual(report.requested_backend, "numpy_baseline")
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
            128,
            grid_levels=(1, 2, 4, 8),
            bins=16,
            asymmetry_scales=(2, 4, 8),
            asymmetry_kinds=("lr", "tb", "rot180"),
            backend="numpy_baseline",
        )
        self.assertEqual(ops.quantize_ops, 128 * 128)
        self.assertEqual(ops.histogram_ops, 4 * 2 * 128 * 128)
        self.assertGreater(ops.total_ops, ops.histogram_ops)

        ops_c = estimate_descriptor_ops(
            128,
            grid_levels=(1, 2, 4, 8),
            bins=16,
            asymmetry_scales=(2, 4, 8),
            asymmetry_kinds=("lr", "tb", "rot180"),
            backend="c_single_pass",
        )
        self.assertLess(ops_c.histogram_ops, ops.histogram_ops)
        self.assertLess(ops_c.total_ops, ops.total_ops)

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
        self.assertTrue(all(bool(r.entropy_backend) for r in records))

    def test_fast_entropy_backends_staged_parity_stages_and_report_rendering(self) -> None:
        """Verify NumPy single-pass and C99 staged profiles match baseline x94 and render Fig 1."""
        rng = np.random.default_rng(2026)
        images = rng.integers(0, 256, size=(6, 64, 64), dtype=np.uint8)
        baseline = extract_x94_end_to_end(images, backend="numpy_baseline")

        expected_stages_map: dict[str, set[str]] = {
            "numpy_finest_pyramid": {"quantize_pixels", "fine_grid_histograms", "pyramid_sum", "normalized_entropy_g8"},
            "numpy_fused_entropy": {"quantize_pixels", "fine_grid_histograms", "pyramid_sum", "fused_entropy"},
        }
        dll_path = resolve_default_native_dll_path(PROJECT_ROOT)
        if dll_path is not None and dll_path.is_file():
            expected_stages_map["c_single_pass"] = {"c_fine_hist", "c_pyramid_sum", "c_entropy_lut"}

        last_report: DescriptorProfileReport | None = None
        for backend_name, expected_subset in expected_stages_map.items():
            staged = extract_x94_staged(images, backend=backend_name)
            e2e = extract_x94_end_to_end(images, backend=backend_name)
            np.testing.assert_allclose(staged, baseline, atol=1e-12)
            np.testing.assert_allclose(e2e, baseline, atol=1e-12)

            report = profile_descriptor_stages(
                images_u8=images,
                dataset_tag=f"synth_{backend_name}",
                grid_levels=(1, 2, 4, 8),
                bins=16,
                asymmetry_scales=(2, 4, 8),
                asymmetry_kinds=("lr", "tb", "rot180"),
                threads=1,
                batch_n_size=4,
                warmup_iters=2,
                timed_iters=8,
                batch_timed_iters=4,
                disable_gc_flag=True,
                bootstrap_resamples=20,
                cprofile_top_n=8,
                seed=42,
                backend=backend_name,
            )
            self.assertEqual(report.entropy_backend, backend_name)
            self.assertEqual(report.requested_backend, backend_name)
            self.assertLessEqual(report.max_numerical_diff, 1e-12)
            stage_names = {s.stage for s in report.stages_batch1}
            self.assertTrue(expected_subset.issubset(stage_names), f"{backend_name} missing {expected_subset} in {stage_names}")
            share_sum = sum(s.share_of_staged_sum_pct for s in report.stages_batch1 if s.stage != "standard_scaler")
            self.assertAlmostEqual(share_sum, 100.0, places=4)
            last_report = report

        assert last_report is not None
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            figs = generate_profiling_figures(
                output_dir=out_dir,
                dataset_label="synth_fast",
                desc_report=last_report,
                desc_scalability=(),
                layer_reports=(),
                cnn_scaling=(),
                cls_scaling=(),
                cpu_comparison_rows=(),
                dpi=80,
            )
            self.assertTrue((out_dir / "fig1_x94_stage_breakdown.png").is_file())
            self.assertGreaterEqual(len(figs), 1)
            env = capture_environment(PROJECT_ROOT, seed=42)
            readme_path = write_profiling_readme(
                output_path=out_dir / "README.md",
                dataset_label="synth_fast",
                env=env,
                desc_report=last_report,
                cpu_rows=(),
                cuda_rows=(),
                stat_tests=(),
                ablation_rows=(),
                layer_reports=(),
                primary_threads=1,
                primary_batch_size=1,
            )
            readme_txt = readme_path.read_text(encoding="utf-8")
            self.assertIn(last_report.entropy_backend, readme_txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
