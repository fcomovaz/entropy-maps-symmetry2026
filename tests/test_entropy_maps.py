"""Unit tests for pure mathematical visual region entropy feature calculations.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_entropy_maps.py
or:
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_entropy_maps.py -q
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.entropy_maps import (
    EntropyPlan,
    build_entropy_plan,
    build_quantization_lut,
    build_shannon_term_lut,
    compute_entropy_features,
    feature_names,
    normalized_entropy,
    quantize_pixels,
    region_edges,
    region_histograms,
    validate_bins,
    validate_grid_levels,
)


class TestEntropyValidation(unittest.TestCase):
    """Test defensive argument validation for grid levels and histogram bins."""

    def test_validate_grid_levels_valid(self) -> None:
        """Valid integer sequences without duplicates within image bounds succeed."""
        self.assertEqual(validate_grid_levels([1, 2, 4, 8], 128), (1, 2, 4, 8))
        self.assertEqual(validate_grid_levels([1], 16), (1,))
        self.assertEqual(validate_grid_levels([3, 5], 100), (3, 5))

    def test_validate_grid_levels_invalid(self) -> None:
        """Invalid grid level specifications raise ValueError."""
        with self.assertRaises(ValueError):
            validate_grid_levels([], 128)  # empty
        with self.assertRaises(ValueError):
            validate_grid_levels([1, 2, 2, 4], 128)  # duplicate
        with self.assertRaises(ValueError):
            validate_grid_levels([0, 1, 2], 128)  # zero
        with self.assertRaises(ValueError):
            validate_grid_levels([-1, 2], 128)  # negative
        with self.assertRaises(ValueError):
            validate_grid_levels([1, 2, 129], 128)  # exceeds size
        with self.assertRaises(ValueError):
            validate_grid_levels([1, True], 128)  # bool type
        with self.assertRaises(ValueError):
            validate_grid_levels([1, 2.5], 128)  # type float

    def test_validate_bins_valid(self) -> None:
        """Bins in [2, 256] validate successfully and return integers."""
        self.assertEqual(validate_bins(2), 2)
        self.assertEqual(validate_bins(16), 16)
        self.assertEqual(validate_bins(256), 256)

    def test_validate_bins_invalid(self) -> None:
        """Bins outside [2, 256] or non-integer types raise ValueError."""
        with self.assertRaises(ValueError):
            validate_bins(1)
        with self.assertRaises(ValueError):
            validate_bins(257)
        with self.assertRaises(ValueError):
            validate_bins(0)
        with self.assertRaises(ValueError):
            validate_bins(-16)
        with self.assertRaises(ValueError):
            validate_bins(True)
        with self.assertRaises(ValueError):
            validate_bins(16.0)
        with self.assertRaises(ValueError):
            validate_bins("16")  # type: ignore


class TestRegionPartition(unittest.TestCase):
    """Test proportional integer region boundaries against reference partitioning."""

    def test_region_edges_exact_division(self) -> None:
        """Evenly divisible dimensions match arithmetic progression."""
        edges = region_edges(128, 8)
        expected = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128], dtype=np.int64)
        np.testing.assert_array_equal(edges, expected)

    def test_region_edges_non_divisible(self) -> None:
        """Non-divisible dimensions partition via floor(k * size / g)."""
        edges = region_edges(10, 3)
        expected = np.array([0, 3, 6, 10], dtype=np.int64)
        np.testing.assert_array_equal(edges, expected)

    def test_region_edges_match_reference_boxes(self) -> None:
        """Region boundaries match partition_boxes from understanding/06."""
        try:
            mod = importlib.import_module(
                "understanding.06_entropy_pyramid_descriptor.partition"
            )
            partition_boxes = mod.partition_boxes
        except (ImportError, AttributeError):
            self.skipTest("understanding/06 reference partition module not importable")

        for size in (16, 32, 128):
            for g in (1, 2, 3, 4, 8):
                boxes = partition_boxes(size, size, g)
                edges = region_edges(size, g)
                box_idx = 0
                for r in range(g):
                    y0, y1 = edges[r], edges[r + 1]
                    for c in range(g):
                        x0, x1 = edges[c], edges[c + 1]
                        ref_x0, ref_y0, ref_x1, ref_y1 = boxes[box_idx]
                        self.assertEqual((x0, y0, x1, y1), (ref_x0, ref_y0, ref_x1, ref_y1))
                        box_idx += 1


class TestEntropyMathAndOracle(unittest.TestCase):
    """Test mathematical entropy properties and exact consistency with canonical oracle."""

    def test_constant_image_zero_entropy(self) -> None:
        """Constant image has single-bin occupancy; normalized entropy must be exactly 0.0."""
        image = np.full((1, 128, 128), 128, dtype=np.uint8)
        features = compute_entropy_features(image, [1, 2, 4, 8], bins=16)
        self.assertEqual(features.shape, (1, 85))
        np.testing.assert_array_equal(features, np.zeros((1, 85), dtype=np.float64))

    def test_equiprobable_maximum_entropy(self) -> None:
        """Image with equal bin counts per region must yield normalized entropy of exactly 1.0."""
        # Create 16x16 image (256 pixels) with 16 equiprobable bins (16 pixels each)
        bins = 16
        pixels_per_bin = 256 // bins
        flat = np.repeat(np.arange(bins, dtype=np.uint8) * (256 // bins), pixels_per_bin)
        image = flat.reshape(1, 16, 16)
        features = compute_entropy_features(image, [1], bins=bins)
        self.assertEqual(features.shape, (1, 1))
        self.assertEqual(features[0, 0], 1.0)

    def test_match_understanding06_oracle(self) -> None:
        """Vectorized feature extraction matches understanding/06 oracle within 1e-12 tolerance."""
        try:
            mod = importlib.import_module(
                "understanding.06_entropy_pyramid_descriptor.descriptor"
            )
            extract_descriptor = mod.extract_descriptor
        except (ImportError, AttributeError):
            self.skipTest("understanding/06 reference descriptor module not importable")

        rng = np.random.RandomState(42)
        random_img = rng.randint(0, 256, (128, 128), dtype=np.uint8)
        gradient_img = np.tile(np.linspace(0, 255, 128, dtype=np.uint8), (128, 1))

        test_bins = (16, 8, 32, 256, 7)
        for b in test_bins:
            for arr in (random_img, gradient_img):
                pil_img = Image.fromarray(arr, mode="L")
                oracle_result = extract_descriptor(pil_img, bins=b)
                oracle_entropies = np.array(oracle_result.entropy_features, dtype=np.float64)

                computed_features = compute_entropy_features(arr, [1, 2, 4, 8], bins=b)[0]
                self.assertEqual(len(computed_features), 85)
                np.testing.assert_allclose(
                    computed_features,
                    oracle_entropies,
                    atol=1e-12,
                    err_msg=f"Discrepancy against oracle with bins={b}",
                )

    def test_feature_counts_and_naming(self) -> None:
        """Feature dimensions match sum(g^2) and names match canonical schema."""
        levels_85 = [1, 2, 4, 8]
        names_85 = feature_names(levels_85)
        self.assertEqual(len(names_85), 85)
        self.assertEqual(names_85[0], "E_1x1_r0_c0")
        self.assertEqual(names_85[-1], "E_8x8_r7_c7")
        self.assertEqual(len(set(names_85)), 85)

        levels_341 = [1, 2, 4, 8, 16]
        names_341 = feature_names(levels_341)
        self.assertEqual(len(names_341), 341)

        levels_custom = [3, 5]
        names_custom = feature_names(levels_custom)
        self.assertEqual(len(names_custom), 3 * 3 + 5 * 5)
        self.assertEqual(names_custom[0], "E_3x3_r0_c0")
        self.assertEqual(names_custom[-1], "E_5x5_r4_c4")

    def test_values_bounded_and_deterministic(self) -> None:
        """Output features are strictly bounded within [0.0, 1.0] and deterministic."""
        rng = np.random.RandomState(123)
        images = rng.randint(0, 256, (4, 64, 64), dtype=np.uint8)
        feats1 = compute_entropy_features(images, [1, 2, 4], bins=16)
        feats2 = compute_entropy_features(images, [1, 2, 4], bins=16)

        np.testing.assert_array_equal(feats1, feats2)
        self.assertTrue(np.all(feats1 >= 0.0))
        self.assertTrue(np.all(feats1 <= 1.0))
        self.assertFalse(np.any(np.isnan(feats1)))

        # Batch consistency: computing 1-by-1 yields same rows
        single_feats = np.stack(
            [compute_entropy_features(images[i], [1, 2, 4], bins=16)[0] for i in range(4)]
        )
        np.testing.assert_allclose(feats1, single_feats, atol=1e-14)


class TestLookUpTablesAndPlans(unittest.TestCase):
    """Test Look-Up Table (LUT) precomputations, caching, and execution plan parity."""

    def test_quantization_lut_correctness(self) -> None:
        """LUT quantization exactly matches arithmetic (pixel * bins) // 256 for all 256 intensities."""
        for b in (8, 16, 32, 256, 7):
            lut = build_quantization_lut(b)
            self.assertEqual(len(lut), 256)
            for p in range(256):
                expected = (p * b) // 256
                self.assertEqual(lut[p], expected)

            # Test image batch with and without explicit LUT
            test_img = np.arange(256, dtype=np.uint8).reshape(16, 16)
            q1 = quantize_pixels(test_img, b)
            q2 = quantize_pixels(test_img, b, lut=lut)
            np.testing.assert_array_equal(q1, q2)

    def test_shannon_term_lut_correctness(self) -> None:
        """Shannon term LUT -p * log2(p) matches float definition for all counts c in [0, N]."""
        n_pixels = 256
        lut = build_shannon_term_lut(n_pixels)
        self.assertEqual(len(lut), n_pixels + 1)
        self.assertEqual(lut[0], 0.0)

        for c in range(1, n_pixels + 1):
            p = c / n_pixels
            expected = -p * math.log2(p)
            self.assertAlmostEqual(lut[c], expected, places=14)

    def test_entropy_plan_execution_parity(self) -> None:
        """Calling compute_entropy_features with a precomputed plan matches ad-hoc execution identically."""
        rng = np.random.RandomState(99)
        images = rng.randint(0, 256, (3, 64, 64), dtype=np.uint8)
        levels = [1, 2, 4]
        bins = 16

        plan = build_entropy_plan(64, levels, bins)
        self.assertEqual(plan.total_features, 1 + 4 + 16)
        self.assertEqual(len(plan.regions), 21)

        feats_with_plan = compute_entropy_features(images, levels, bins, plan=plan)
        feats_without_plan = compute_entropy_features(images, levels, bins, plan=None)

        np.testing.assert_array_equal(feats_with_plan, feats_without_plan)


if __name__ == "__main__":
    unittest.main()
