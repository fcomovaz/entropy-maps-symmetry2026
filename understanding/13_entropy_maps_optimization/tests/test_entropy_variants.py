"""Unit tests verifying numerical parity and fail-fast contracts across V0-V3."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

TESTS_DIR = Path(__file__).resolve().parent
LAB_ROOT = TESTS_DIR.parent
PROJECT_ROOT = TESTS_DIR.parents[2]
for _p in (PROJECT_ROOT, LAB_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config_loader import (
    EntropyOptimizationConfig,
    find_preprocessed_image_paths,
    generate_synthetic_images,
    load_optimization_config,
)
from native_binding import V3CSinglePassVariant, load_native_library
from src.features.entropy_maps import build_entropy_plan, compute_entropy_features
from variants import (
    EntropyVariant,
    V0BaselineVariant,
    V1NumpyFinestPyramidVariant,
    V2NumpyFusedEntropyVariant,
)

DEFAULT_CONFIG_REL = Path("configs/optimization/entropy_maps.toml")


class TestEntropyVariants(unittest.TestCase):
    """Contract and numerical parity suite for entropy pyramid variants V0-V3."""

    cfg: EntropyOptimizationConfig
    has_dll: bool

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_optimization_config(DEFAULT_CONFIG_REL, project_root=PROJECT_ROOT)
        cls.has_dll = cls.cfg.native.dll_path.is_file()

    def _make_variants(self, include_v3: bool = True) -> list[EntropyVariant]:
        variants: list[EntropyVariant] = [
            V0BaselineVariant(),
            V1NumpyFinestPyramidVariant(),
            V2NumpyFusedEntropyVariant(),
        ]
        if include_v3 and self.has_dll:
            variants.append(V3CSinglePassVariant(self.cfg.native.dll_path))
        return variants

    def _make_fast_variants(self) -> list[EntropyVariant]:
        fast: list[EntropyVariant] = [
            V1NumpyFinestPyramidVariant(),
            V2NumpyFusedEntropyVariant(),
        ]
        if self.has_dll:
            fast.append(V3CSinglePassVariant(self.cfg.native.dll_path))
        return fast

    def _assert_matches_v0_and_oracle(
        self,
        images: np.ndarray,
        image_size: int,
        grid_levels: tuple[int, ...],
        bins: int,
    ) -> np.ndarray:
        tol = self.cfg.protocol.tolerance_abs
        v0 = V0BaselineVariant()
        v0.prepare(image_size, grid_levels, bins)
        expected_v0 = v0.run(images)

        batch_src = images[None, ...] if images.ndim == 2 else images
        plan = build_entropy_plan(image_size, grid_levels, bins)
        expected_src = compute_entropy_features(batch_src, grid_levels=grid_levels, bins=bins, plan=plan)
        err_v0_src = (
            float(np.max(np.abs(expected_v0 - expected_src))) if expected_v0.size > 0 else 0.0
        )
        self.assertLessEqual(err_v0_src, tol)

        for var in self._make_variants(include_v3=True):
            var.prepare(image_size, grid_levels, bins)
            actual = var.run(images)
            self.assertEqual(actual.dtype, np.float64, msg=f"{var.name} returned wrong dtype")
            self.assertEqual(actual.shape, expected_v0.shape, msg=f"{var.name} returned wrong shape")
            max_err = float(np.max(np.abs(actual - expected_v0))) if actual.size > 0 else 0.0
            self.assertLessEqual(
                max_err,
                tol,
                msg=f"{var.name} exceeded tolerance {tol} (max_err={max_err:.3e})",
            )
        return expected_v0

    def test_constant_images_zero_entropy(self) -> None:
        sz = self.cfg.data.image_size
        imgs = np.stack(
            [
                np.zeros((sz, sz), dtype=np.uint8),
                np.full((sz, sz), 127, dtype=np.uint8),
                np.full((sz, sz), 255, dtype=np.uint8),
            ],
            axis=0,
        )
        out = self._assert_matches_v0_and_oracle(imgs, sz, self.cfg.grid_levels, self.cfg.bins)
        self.assertTrue(np.all(out == 0.0))

    def test_equiprobable_uniform_image_unit_entropy(self) -> None:
        sz = self.cfg.data.image_size
        g_max = max(self.cfg.grid_levels)
        cell_side = sz // g_max
        cell_pix = cell_side * cell_side
        bins = self.cfg.bins
        if cell_pix % bins != 0:
            self.skipTest("Cell pixel count not divisible by bins for exact uniform tiling")

        # Construct uniform tile inside each finest cell -> equiprobable histogram at every pyramid level
        bin_targets = np.repeat(np.arange(bins, dtype=np.uint16), cell_pix // bins)
        pix_tile = ((bin_targets * 256 + 128) // bins).clip(0, 255).astype(np.uint8).reshape(
            cell_side, cell_side
        )
        uniform_img = np.tile(pix_tile, (g_max, g_max))[None, ...]
        out = self._assert_matches_v0_and_oracle(uniform_img, sz, self.cfg.grid_levels, bins)
        self.assertTrue(np.all(out == 1.0))

    def test_synthetic_batch_matches_v0_src_and_canonical_oracle(self) -> None:
        sz = self.cfg.data.image_size
        batch = generate_synthetic_images(8, sz, self.cfg.seed)
        v0_out = self._assert_matches_v0_and_oracle(batch, sz, self.cfg.grid_levels, self.cfg.bins)

        if self.cfg.grid_levels == (1, 2, 4, 8):
            desc_mod = import_module("understanding.06_entropy_pyramid_descriptor.descriptor")
            pil_img = Image.fromarray(batch[0], mode="L")
            res06 = desc_mod.extract_descriptor(pil_img, bins=self.cfg.bins)
            expected_06 = np.asarray(res06.entropy_features, dtype=np.float64)
            err = float(np.max(np.abs(v0_out[0] - expected_06)))
            self.assertLessEqual(err, self.cfg.protocol.tolerance_abs)

    def test_real_preprocessed_image_parity(self) -> None:
        paths = find_preprocessed_image_paths(self.cfg.data)
        if not paths:
            self.skipTest(f"No preprocessed real images available under {self.cfg.data.preprocessed_dir}")

        sz = self.cfg.data.image_size
        sample_imgs: list[np.ndarray] = []
        for p in paths[:4]:
            with Image.open(p) as im:
                gray = im.convert("L")
                if gray.size != (sz, sz):
                    gray = gray.resize((sz, sz), resample=Image.Resampling.BILINEAR)
                sample_imgs.append(np.asarray(gray, dtype=np.uint8))
        batch = np.stack(sample_imgs, axis=0)
        self._assert_matches_v0_and_oracle(batch, sz, self.cfg.grid_levels, self.cfg.bins)

    def test_2d_single_and_batch1_and_empty_n0(self) -> None:
        sz = self.cfg.data.image_size
        single_3d = generate_synthetic_images(1, sz, self.cfg.seed)
        single_2d = single_3d[0]
        empty_0 = np.empty((0, sz, sz), dtype=np.uint8)

        out_3d = self._assert_matches_v0_and_oracle(single_3d, sz, self.cfg.grid_levels, self.cfg.bins)
        out_2d = self._assert_matches_v0_and_oracle(single_2d, sz, self.cfg.grid_levels, self.cfg.bins)
        self.assertTrue(np.array_equal(out_3d, out_2d))
        self._assert_matches_v0_and_oracle(empty_0, sz, self.cfg.grid_levels, self.cfg.bins)

    def test_non_contiguous_strided_views(self) -> None:
        sz = self.cfg.data.image_size
        oversize = generate_synthetic_images(4, sz * 2, self.cfg.seed + 1)
        strided = oversize[:, ::2, ::-2]
        self.assertFalse(strided.flags.c_contiguous)
        self._assert_matches_v0_and_oracle(strided, sz, self.cfg.grid_levels, self.cfg.bins)

    def test_bins_sweep_power2_and_arbitrary(self) -> None:
        sz = self.cfg.data.image_size
        batch = generate_synthetic_images(4, sz, self.cfg.seed + 2)
        for bins in (2, 7, 8, 13, 16, 32, 64, 256):
            self._assert_matches_v0_and_oracle(batch, sz, self.cfg.grid_levels, bins)

    def test_grid_levels_permutations_and_odd_nesting(self) -> None:
        sz = self.cfg.data.image_size
        batch = generate_synthetic_images(3, sz, self.cfg.seed + 3)
        for levels in ((1,), (1, 2, 4), (1, 4), (8, 4, 2, 1), (1, 2, 4, 8, 16)):
            self._assert_matches_v0_and_oracle(batch, sz, levels, self.cfg.bins)

        # Odd nested ratio k=3 and non-power-of-two image size (81 = 9 * 9)
        odd_sz = 81
        odd_batch = generate_synthetic_images(3, odd_sz, self.cfg.seed + 4)
        self._assert_matches_v0_and_oracle(odd_batch, odd_sz, (1, 3, 9), self.cfg.bins)

    def test_large_cell_u32_fallback_parity(self) -> None:
        # 256x256 level g=1 has 65536 pixels (> uint16 max 65535); exercises uint32 fallback in C
        batch_256 = generate_synthetic_images(2, 256, self.cfg.seed + 5)
        self._assert_matches_v0_and_oracle(batch_256, 256, (1, 2, 4, 8), self.cfg.bins)

        batch_512 = generate_synthetic_images(1, 512, self.cfg.seed + 6)
        self._assert_matches_v0_and_oracle(batch_512, 512, (1, 2, 4, 8), self.cfg.bins)

    def test_diagnostic_stages_parity(self) -> None:
        sz = self.cfg.data.image_size
        img = generate_synthetic_images(1, sz, self.cfg.seed + 7)
        v0 = V0BaselineVariant()
        v0.prepare(sz, self.cfg.grid_levels, self.cfg.bins)
        ref = v0.run(img)

        for var in self._make_variants(include_v3=True):
            var.prepare(sz, self.cfg.grid_levels, self.cfg.bins)
            out_staged, stages = var.measure_stages_once(img)  # type: ignore[attr-defined]
            self.assertLessEqual(
                float(np.max(np.abs(out_staged - ref))),
                self.cfg.protocol.tolerance_abs,
                msg=f"{var.name} measure_stages_once diverged from V0",
            )
            for key in ("quantize_ms", "histogram_ms", "pyramid_sum_ms", "entropy_ms"):
                self.assertIn(key, stages)
                self.assertGreaterEqual(stages[key], 0.0)

    def test_non_divisible_and_non_nested_raise_value_error(self) -> None:
        for var in self._make_fast_variants():
            with self.assertRaises(ValueError, msg=f"{var.name} accepted non-divisible size"):
                var.prepare(130, (1, 2, 4, 8), self.cfg.bins)
            with self.assertRaises(ValueError, msg=f"{var.name} accepted non-nested levels"):
                var.prepare(120, (1, 3, 4), self.cfg.bins)

    def test_invalid_inputs_fail_fast(self) -> None:
        sz = self.cfg.data.image_size
        for var in self._make_variants(include_v3=True):
            with self.assertRaises(RuntimeError):
                var.run(np.zeros((1, sz, sz), dtype=np.uint8))

            var.prepare(sz, self.cfg.grid_levels, self.cfg.bins)
            with self.assertRaises(ValueError):
                var.run(np.zeros((1, sz, sz), dtype=np.float32))
            with self.assertRaises(ValueError):
                var.run(np.zeros((1, sz, sz, 1), dtype=np.uint8))
            with self.assertRaises(ValueError):
                var.run(np.zeros((1, sz, sz - 2), dtype=np.uint8))
            with self.assertRaises(ValueError):
                var.run(np.zeros((1, sz // 2, sz // 2), dtype=np.uint8))

        with self.assertRaises(FileNotFoundError):
            load_native_library(self.cfg.native.build_dir / "non_existent_entropy_kernel.dll")

    def test_v3_specific_contracts_non_aliasing_and_noop(self) -> None:
        if not self.has_dll:
            self.skipTest("Native DLL not compiled yet; skipping V3 C-specific checks.")

        sz = self.cfg.data.image_size
        v3 = V3CSinglePassVariant(self.cfg.native.dll_path)
        self.assertGreaterEqual(v3.limits.max_finest_grid, max(self.cfg.grid_levels))
        self.assertGreaterEqual(v3.limits.max_bins, self.cfg.bins)
        self.assertGreaterEqual(v3.limits.max_levels, len(self.cfg.grid_levels))

        v3.prepare(sz, self.cfg.grid_levels, self.cfg.bins)
        self.assertEqual(v3.call_noop(), 0)

        imgs = generate_synthetic_images(2, sz, self.cfg.seed + 8)
        out_a = v3.run(imgs[0:1])
        snapshot_a = out_a.copy()
        out_b = v3.run(imgs[1:2])
        self.assertTrue(np.array_equal(out_a, snapshot_a), msg="V3 batch=1 output aliased across calls")
        self.assertFalse(np.array_equal(out_a, out_b))

        total_features = sum(g * g for g in self.cfg.grid_levels)
        buf = np.empty((2, total_features), dtype=np.float64)
        v3.run_into(imgs, buf)
        self.assertTrue(np.allclose(buf[0:1], out_a, rtol=0.0, atol=self.cfg.protocol.tolerance_abs))

        with self.assertRaises(ValueError):
            v3.run_into(imgs, np.empty((2, total_features), dtype=np.float32))
        with self.assertRaises(ValueError):
            v3.run_into(imgs, np.asfortranarray(buf))
        with self.assertRaises(ValueError):
            v3.prepare(sz, (1, 2, 4, 8, 32), self.cfg.bins)


if __name__ == "__main__":
    unittest.main(verbosity=2)
