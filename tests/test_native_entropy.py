"""Unit and numerical parity tests for the formal C99 single-pass entropy kernel.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_native_entropy.py -v
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.entropy_maps import compute_entropy_features as oracle_entropy_features


class TestNativeEntropyKernel(unittest.TestCase):
    """Test native C entropy DLL config loading, compatibility checks, and numerical parity."""

    def test_load_native_config_from_repo_toml(self) -> None:
        """Default native_entropy.toml resolves dynamically to src/features/native paths."""
        from src.features.native_entropy import load_native_config

        cfg = load_native_config(Path("configs/features/native_entropy.toml"), PROJECT_ROOT)
        self.assertEqual(cfg.library_name, "entropy_pyramid")
        self.assertEqual(cfg.source_dir, (PROJECT_ROOT / "src/features/native").resolve())
        self.assertEqual(cfg.build_dir, (PROJECT_ROOT / "src/features/native/build").resolve())
        self.assertEqual(cfg.c_sources, ("entropy_pyramid.c",))
        self.assertIn("/O2", cfg.compiler_flags)
        self.assertEqual(cfg.dll_path.name, "entropy_pyramid.dll")

    def test_load_native_config_fail_fast(self) -> None:
        """Missing TOML file or malformed required keys fail fast."""
        from src.features.native_entropy import load_native_config

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            with self.assertRaises(FileNotFoundError):
                load_native_config(Path("configs/features/missing.toml"), tmp)

            bad_cfg = tmp / "bad.toml"
            bad_cfg.write_text("[entropy_native]\nlibrary_name = \"\"\n", encoding="utf-8")
            with self.assertRaises((KeyError, ValueError)):
                load_native_config(bad_cfg, tmp)

    def test_missing_dll_raises_file_not_found(self) -> None:
        """Requesting a non-existent DLL raises FileNotFoundError immediately."""
        from src.features.native_entropy import prepare_native_buffers

        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_dll = Path(tmp_dir) / "does_not_exist.dll"
            with self.assertRaises(FileNotFoundError):
                prepare_native_buffers(128, [1, 2, 4, 8], 16, missing_dll)

    def test_compatibility_checks_structural_limits_and_nesting(self) -> None:
        """Compatibility check rejects non-nested grids, g_max > 16, coarse > 128, and bins > 256."""
        from src.features.native_entropy import NativeKernelLimits, check_native_compatibility

        limits = NativeKernelLimits(max_finest_grid=16, max_coarse_cells=128, max_bins=256, max_levels=16)

        ok, reason = check_native_compatibility(128, [1, 2, 4, 8], 16, limits=limits)
        self.assertTrue(ok, msg=str(reason))
        self.assertIsNone(reason)

        # Non-nested levels [3, 5] must be rejected
        ok_non_nested, _ = check_native_compatibility(128, [3, 5], 16, limits=limits)
        self.assertFalse(ok_non_nested)

        # Not divisible image_size % g_max
        ok_indiv, _ = check_native_compatibility(100, [1, 2, 4, 8], 16, limits=limits)
        self.assertFalse(ok_indiv)

        # g_max = 32 > 16
        ok_gmax, _ = check_native_compatibility(128, [1, 2, 4, 8, 16, 32], 16, limits=limits)
        self.assertFalse(ok_gmax)

        # Coarse cells exceeded: g=16 finest + (1, 2, 4, 8, 12? no, 1+4+16+64+144 > 128) -> e.g. with g_max=16:
        # [1, 2, 4, 8, 16] has coarse = 1+4+16+64 = 85 <= 128 (ok!)
        ok_85, reason_85 = check_native_compatibility(128, [1, 2, 4, 8, 16], 16, limits=limits)
        self.assertTrue(ok_85, msg=str(reason_85))

        # Tighter coarse-cell limit rejects [1, 2, 4, 8, 16] cleanly
        tight_limits = NativeKernelLimits(max_finest_grid=16, max_coarse_cells=80, max_bins=256, max_levels=16)
        ok_coarse, _ = check_native_compatibility(128, [1, 2, 4, 8, 16], 16, limits=tight_limits)
        self.assertFalse(ok_coarse)

        # Excessive bins
        ok_bins, _ = check_native_compatibility(128, [1, 2, 4, 8], 257, limits=limits)
        self.assertFalse(ok_bins)

    def test_compiled_dll_parity_all_cases_and_stages(self) -> None:
        """Compiled C DLL matches baseline oracle within 1e-12 across diverse inputs and stages."""
        from src.features.native_entropy import (
            load_native_config,
            prepare_native_buffers,
            run_native_entropy,
            run_native_noop,
            run_native_stage_entropy,
            run_native_stage_fine_hist,
            run_native_stage_sum,
        )

        cfg = load_native_config(Path("configs/features/native_entropy.toml"), PROJECT_ROOT)
        if not cfg.dll_path.is_file():
            self.fail(
                f"Native DLL not built at {cfg.dll_path}. "
                "Compile via '.\\.venv\\Scripts\\python.exe -m src.features.build_native'."
            )

        rng = np.random.default_rng(42)

        # 1. Standard 128x128 [1,2,4,8], bins=16 across batch, single 2D, non-contiguous, N=0, const, equiprob
        buf = prepare_native_buffers(128, [1, 2, 4, 8], 16, cfg.dll_path)
        self.assertEqual(run_native_noop(buf), 0)

        imgs = rng.integers(0, 256, size=(6, 128, 128), dtype=np.uint8)
        imgs[0] = 77  # constant -> entropy 0.0
        # Row-wise pattern of 16 bins evenly repeated for cell 0
        imgs[1] = np.tile((np.arange(16, dtype=np.uint8) * 16), (128, 8))

        ref = oracle_entropy_features(imgs, [1, 2, 4, 8], 16)
        got = run_native_entropy(buf, imgs)
        np.testing.assert_allclose(got, ref, atol=1e-12)
        np.testing.assert_array_equal(got[0], np.zeros(85, dtype=np.float64))
        self.assertEqual(got[1, 0], 1.0)

        # Empty N=0
        empty_got = run_native_entropy(buf, imgs[:0])
        self.assertEqual(empty_got.shape, (0, 85))

        # 2D input and non-aliasing across consecutive single-image runs
        r0 = run_native_entropy(buf, imgs[0])
        r2 = run_native_entropy(buf, imgs[2])
        self.assertEqual(r0.shape, (1, 85))
        np.testing.assert_array_equal(r0[0], np.zeros(85, dtype=np.float64))
        np.testing.assert_allclose(r2[0], ref[2], atol=1e-12)

        # Non-contiguous slice
        nc_slice = imgs[:, ::2, ::2]  # 64x64 non-contiguous
        buf64 = prepare_native_buffers(64, [1, 2, 4], 16, cfg.dll_path)
        ref_nc = oracle_entropy_features(np.ascontiguousarray(nc_slice), [1, 2, 4], 16)
        got_nc = run_native_entropy(buf64, nc_slice)
        np.testing.assert_allclose(got_nc, ref_nc, atol=1e-12)

        # 2. Various bins including non-power-of-two (2, 7, 8, 32, 64, 256)
        for bins in (2, 7, 8, 32, 64, 256):
            buf_b = prepare_native_buffers(128, [1, 2, 4, 8], bins, cfg.dll_path)
            ref_b = oracle_entropy_features(imgs[:3], [1, 2, 4, 8], bins)
            got_b = run_native_entropy(buf_b, imgs[:3])
            np.testing.assert_allclose(got_b, ref_b, atol=1e-12, err_msg=f"Mismatch at bins={bins}")

        # 3. Unordered levels (8, 4, 2, 1) and non-power-of-2 nested levels (1, 3, 9) on 108x108
        buf_unord = prepare_native_buffers(128, [8, 4, 2, 1], 16, cfg.dll_path)
        ref_unord = oracle_entropy_features(imgs[:2], [8, 4, 2, 1], 16)
        got_unord = run_native_entropy(buf_unord, imgs[:2])
        np.testing.assert_allclose(got_unord, ref_unord, atol=1e-12)

        img108 = rng.integers(0, 256, size=(2, 108, 108), dtype=np.uint8)
        buf_odd = prepare_native_buffers(108, [1, 3, 9], 12, cfg.dll_path)
        ref_odd = oracle_entropy_features(img108, [1, 3, 9], 12)
        got_odd = run_native_entropy(buf_odd, img108)
        np.testing.assert_allclose(got_odd, ref_odd, atol=1e-12)

        # 4. Larger image triggering u32 fallback path (512x512, level=[1, 2, 4, 8] -> 512^2 = 262144 > 65535)
        img512 = rng.integers(0, 256, size=(1, 512, 512), dtype=np.uint8)
        buf_512 = prepare_native_buffers(512, [1, 2, 4, 8], 16, cfg.dll_path)
        ref_512 = oracle_entropy_features(img512, [1, 2, 4, 8], 16)
        got_512 = run_native_entropy(buf_512, img512)
        np.testing.assert_allclose(got_512, ref_512, atol=1e-12)

        # 5. Diagnostic 3-stage execution matches single-pass C kernel identically
        fine_hist = run_native_stage_fine_hist(buf, imgs)
        all_hist = run_native_stage_sum(buf, fine_hist)
        staged_out = run_native_stage_entropy(buf, all_hist)
        np.testing.assert_allclose(staged_out, got, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
