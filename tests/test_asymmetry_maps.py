"""Unit tests for pure mathematical spatial entropy asymmetry feature calculations.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\.venv\Scripts\python.exe -m unittest tests/test_asymmetry_maps.py
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.asymmetry_maps import (
    ALLOWED_ASYMMETRY_KINDS,
    asymmetry_column_names,
    compute_asymmetry_features,
    compute_asymmetry_for_grid,
    entropy_column_names,
    reconstruct_grids,
    validate_entropy_values,
    validate_kinds,
    validate_scales,
)


class TestScalesAndKindsValidation(unittest.TestCase):
    """Test defensive validation for spatial scales and asymmetry kinds."""

    def test_validate_scales_valid(self) -> None:
        """Valid ascending integer sequences >= 2 without duplicates succeed."""
        self.assertEqual(validate_scales([2, 4, 8]), (2, 4, 8))
        self.assertEqual(validate_scales([2]), (2,))
        self.assertEqual(validate_scales([3, 5, 7]), (3, 5, 7))

    def test_validate_scales_invalid(self) -> None:
        """Invalid scale specifications raise ValueError."""
        with self.assertRaises(ValueError):
            validate_scales([])  # empty
        with self.assertRaises(ValueError):
            validate_scales([1, 2, 4])  # contains global scale 1
        with self.assertRaises(ValueError):
            validate_scales([2, 2, 4])  # duplicate
        with self.assertRaises(ValueError):
            validate_scales([4, 2])  # unsorted
        with self.assertRaises(ValueError):
            validate_scales([0, 2])  # < 2
        with self.assertRaises(ValueError):
            validate_scales([-2, 4])  # negative
        with self.assertRaises(ValueError):
            validate_scales([2, True])  # bool type
        with self.assertRaises(ValueError):
            validate_scales([2, 4.0])  # float type

    def test_validate_kinds_valid(self) -> None:
        """Valid subsets of canonical asymmetry kinds succeed."""
        self.assertEqual(
            validate_kinds(["lr", "tb", "rot180"]), ("lr", "tb", "rot180")
        )
        self.assertEqual(validate_kinds(["lr"]), ("lr",))
        self.assertEqual(validate_kinds(["rot180", "tb"]), ("rot180", "tb"))

    def test_validate_kinds_invalid(self) -> None:
        """Invalid asymmetry kinds raise ValueError."""
        with self.assertRaises(ValueError):
            validate_kinds([])  # empty
        with self.assertRaises(ValueError):
            validate_kinds(["lr", "unknown"])  # unrecognized kind
        with self.assertRaises(ValueError):
            validate_kinds(["lr", "lr"])  # duplicate
        with self.assertRaises(ValueError):
            validate_kinds([123])  # type: ignore


class TestColumnNames(unittest.TestCase):
    """Test canonical column naming and schema compliance."""

    def test_asymmetry_column_names_canonical(self) -> None:
        """Default templates match exact schema x94-v1 order and naming."""
        names = asymmetry_column_names([2, 4, 8])
        expected = [
            "asym_lr_2x2",
            "asym_tb_2x2",
            "asym_rot180_2x2",
            "asym_lr_4x4",
            "asym_tb_4x4",
            "asym_rot180_4x4",
            "asym_lr_8x8",
            "asym_tb_8x8",
            "asym_rot180_8x8",
        ]
        self.assertEqual(names, expected)

    def test_asymmetry_column_names_custom_scales(self) -> None:
        """Custom scales and templates format appropriately."""
        names = asymmetry_column_names([2, 4], kinds=["lr", "tb"])
        expected = ["asym_lr_2x2", "asym_tb_2x2", "asym_lr_4x4", "asym_tb_4x4"]
        self.assertEqual(names, expected)

    def test_entropy_column_names_row_major(self) -> None:
        """Row-major entropy column naming matches expected format."""
        names_2x2 = entropy_column_names(2)
        expected_2x2 = ["E_2x2_r0_c0", "E_2x2_r0_c1", "E_2x2_r1_c0", "E_2x2_r1_c1"]
        self.assertEqual(names_2x2, expected_2x2)

        names_4x4 = entropy_column_names(4)
        self.assertEqual(len(names_4x4), 16)
        self.assertEqual(names_4x4[0], "E_4x4_r0_c0")
        self.assertEqual(names_4x4[-1], "E_4x4_r3_c3")


class TestGridReconstruction(unittest.TestCase):
    """Test 1D and 2D row-major reshaping into 3D square grids."""

    def test_reconstruct_grids_shapes(self) -> None:
        """Valid 1D and 2D arrays reshape correctly to [N, g, g]."""
        flat_1d = np.arange(16, dtype=np.float64)
        grid_1d = reconstruct_grids(flat_1d, 4)
        self.assertEqual(grid_1d.shape, (1, 4, 4))
        self.assertEqual(grid_1d[0, 1, 2], 1 * 4 + 2)

        flat_2d = np.arange(32, dtype=np.float64).reshape(2, 16)
        grid_2d = reconstruct_grids(flat_2d, 4)
        self.assertEqual(grid_2d.shape, (2, 4, 4))
        self.assertEqual(grid_2d[1, 2, 3], 16 + 2 * 4 + 3)

    def test_reconstruct_grids_invalid(self) -> None:
        """Shape mismatches raise ValueError."""
        with self.assertRaises(ValueError):
            reconstruct_grids(np.zeros(15), 4)  # 15 != 16
        with self.assertRaises(ValueError):
            reconstruct_grids(np.zeros((3, 10)), 4)  # 10 != 16
        with self.assertRaises(ValueError):
            reconstruct_grids(np.zeros((2, 4, 4)), 4)  # 3D input


class TestAsymmetryMath(unittest.TestCase):
    """Test mathematical correctness, symmetry invariances, and hand-computed examples."""

    def test_constant_grid_zero_asymmetry(self) -> None:
        """A uniform grid has zero asymmetry across all axes."""
        for val in (0.0, 0.5, 1.0):
            grid = np.full((1, 4, 4), val, dtype=np.float64)
            res = compute_asymmetry_for_grid(grid)
            for k in ALLOWED_ASYMMETRY_KINDS:
                self.assertAlmostEqual(res[k][0], 0.0, places=12)

    def test_axis_symmetric_grids(self) -> None:
        """Grids invariant under LR, TB, or 180 transformations evaluate to 0.0 on respective residuals."""
        rng = np.random.RandomState(42)
        half_lr = rng.uniform(0.0, 1.0, (1, 4, 2))
        grid_lr = np.concatenate([half_lr, half_lr[:, :, ::-1]], axis=2)
        res_lr = compute_asymmetry_for_grid(grid_lr)
        self.assertAlmostEqual(res_lr["lr"][0], 0.0, places=12)

        half_tb = rng.uniform(0.0, 1.0, (1, 2, 4))
        grid_tb = np.concatenate([half_tb, half_tb[:, ::-1, :]], axis=1)
        res_tb = compute_asymmetry_for_grid(grid_tb)
        self.assertAlmostEqual(res_tb["tb"][0], 0.0, places=12)

        # 180-degree symmetric grid: grid[r, c] == grid[g-1-r, g-1-c]
        mat = np.zeros((4, 4), dtype=np.float64)
        for r in range(4):
            for c in range(4):
                if r < 2:
                    mat[r, c] = rng.uniform(0.0, 1.0)
                else:
                    mat[r, c] = mat[3 - r, 3 - c]
        grid_180 = mat.reshape(1, 4, 4)
        res_180 = compute_asymmetry_for_grid(grid_180)
        self.assertAlmostEqual(res_180["rot180"][0], 0.0, places=12)

    def test_hand_computed_2x2_examples(self) -> None:
        """Hand-computed 2x2 grids match exact analytical residual values."""
        # Grid 1: columns alternating [[0, 1], [0, 1]]
        # LR diff: |0-1|*2 + |1-0|*2 = 4 -> mean = 4/4 = 1.0
        # TB diff: |0-0|*2 + |1-1|*2 = 0 -> mean = 0/4 = 0.0
        # R180 diff: |0-1|*2 + |1-0|*2 = 4 -> mean = 4/4 = 1.0
        grid1 = np.array([[[0.0, 1.0], [0.0, 1.0]]], dtype=np.float64)
        res1 = compute_asymmetry_for_grid(grid1)
        self.assertAlmostEqual(res1["lr"][0], 1.0, places=12)
        self.assertAlmostEqual(res1["tb"][0], 0.0, places=12)
        self.assertAlmostEqual(res1["rot180"][0], 1.0, places=12)

        # Grid 2: rows alternating [[1, 1], [0, 0]]
        # LR diff: 0.0, TB diff: 1.0, R180 diff: 1.0
        grid2 = np.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=np.float64)
        res2 = compute_asymmetry_for_grid(grid2)
        self.assertAlmostEqual(res2["lr"][0], 0.0, places=12)
        self.assertAlmostEqual(res2["tb"][0], 1.0, places=12)
        self.assertAlmostEqual(res2["rot180"][0], 1.0, places=12)

    def test_asymmetry_bounds_and_triangle_inequality(self) -> None:
        """Random entropy grids stay bounded in [0, 1] and satisfy triangle inequality."""
        rng = np.random.RandomState(123)
        grids = rng.uniform(0.0, 1.0, (10, 4, 4))
        res = compute_asymmetry_for_grid(grids)

        for k in ALLOWED_ASYMMETRY_KINDS:
            vals = res[k]
            self.assertTrue(np.all(vals >= 0.0))
            self.assertTrue(np.all(vals <= 1.0))
            self.assertFalse(np.any(np.isnan(vals)))

        # Contract inequality: |A_LR - A_TB| <= A_R180 <= A_LR + A_TB
        lr, tb, rot = res["lr"], res["tb"], res["rot180"]
        eps = 1e-12
        self.assertTrue(np.all(rot >= np.abs(lr - tb) - eps))
        self.assertTrue(np.all(rot <= (lr + tb) + eps))


class TestOracleParity(unittest.TestCase):
    """Test exact parity against canonical reference in understanding/05."""

    def test_parity_with_understanding_05_oracle(self) -> None:
        """Vectorized calculations match understanding/05 oracle within 1e-12 tolerance."""
        try:
            mod = importlib.import_module(
                "understanding.05_symmetry_features.symmetry_features"
            )
            reflection_asymmetry_lr = mod.reflection_asymmetry_lr
            reflection_asymmetry_tb = mod.reflection_asymmetry_tb
            rot180_asymmetry = mod.rot180_asymmetry
        except (ImportError, AttributeError):
            self.skipTest(
                "understanding.05_symmetry_features.symmetry_features not importable"
            )

        rng = np.random.RandomState(42)
        for g in (2, 4, 8):
            for _ in range(5):
                mat_np = rng.uniform(0.0, 1.0, (g, g))
                grid_list = mat_np.tolist()

                expected_lr = reflection_asymmetry_lr(grid_list)
                expected_tb = reflection_asymmetry_tb(grid_list)
                expected_rot = rot180_asymmetry(grid_list)

                computed = compute_asymmetry_for_grid(mat_np.reshape(1, g, g))
                self.assertAlmostEqual(
                    computed["lr"][0], expected_lr, places=12, msg=f"LR mismatch at scale {g}"
                )
                self.assertAlmostEqual(
                    computed["tb"][0], expected_tb, places=12, msg=f"TB mismatch at scale {g}"
                )
                self.assertAlmostEqual(
                    computed["rot180"][0],
                    expected_rot,
                    places=12,
                    msg=f"R180 mismatch at scale {g}",
                )


class TestDefensiveValidation(unittest.TestCase):
    """Test strict out-of-range rejection and tolerance handling without clipping."""

    def test_validate_entropy_values_valid(self) -> None:
        """Values in [0, 1] pass cleanly."""
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        validate_entropy_values(arr, tolerance=1e-6)

    def test_validate_entropy_values_tolerance_edge(self) -> None:
        """Values slightly exceeding 1.0 within tolerance pass without raising."""
        arr = np.array([1.0 + 1e-8], dtype=np.float64)
        validate_entropy_values(arr, tolerance=1e-6)

    def test_validate_entropy_values_out_of_range(self) -> None:
        """Values exceeding upper bound or below 0 raise ValueError."""
        with self.assertRaises(ValueError):
            validate_entropy_values(np.array([-0.01]))
        with self.assertRaises(ValueError):
            validate_entropy_values(np.array([1.02]), tolerance=1e-6)

    def test_validate_entropy_values_nan_inf(self) -> None:
        """NaN or Inf values raise ValueError."""
        with self.assertRaises(ValueError):
            validate_entropy_values(np.array([0.5, np.nan]))
        with self.assertRaises(ValueError):
            validate_entropy_values(np.array([0.5, np.inf]))


class TestComputeAsymmetryFeatures(unittest.TestCase):
    """Test full multi-scale feature matrix computation and concatenation."""

    def test_compute_asymmetry_features_shape_and_order(self) -> None:
        """Concatenated output has shape [N, 9] in canonical scale-then-kind order."""
        rng = np.random.RandomState(99)
        n_samples = 4
        scales = (2, 4, 8)
        entropy_by_scale = {
            g: rng.uniform(0.0, 1.0, (n_samples, g * g)) for g in scales
        }

        features = compute_asymmetry_features(entropy_by_scale, scales=scales)
        self.assertEqual(features.shape, (n_samples, 9))

        # Check individual scale parity
        col = 0
        for g in scales:
            grid = entropy_by_scale[g].reshape(n_samples, g, g)
            res = compute_asymmetry_for_grid(grid)
            for k in ("lr", "tb", "rot180"):
                np.testing.assert_allclose(features[:, col], res[k], atol=1e-14)
                col += 1

    def test_batch_mismatch_raises(self) -> None:
        """Different sample counts across scales raise ValueError."""
        entropy_by_scale = {
            2: np.zeros((3, 4)),
            4: np.zeros((4, 16)),  # 4 != 3
            8: np.zeros((3, 64)),
        }
        with self.assertRaises(ValueError):
            compute_asymmetry_features(entropy_by_scale, scales=(2, 4, 8))


if __name__ == "__main__":
    unittest.main()
