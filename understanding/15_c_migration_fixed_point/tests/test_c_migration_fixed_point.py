"""Unit tests for Understanding Lab 15 fixed-point, entropy LUT, BN folding, and requantization.

Manual execution (PowerShell from 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest discover understanding/15_c_migration_fixed_point/tests -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAB_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_lab_module(module_name: str, file_name: str):
    """Load a numbered script from Lab 15 by path."""
    spec = importlib.util.spec_from_file_location(module_name, LAB_DIR / file_name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestLab15FixedPoint(unittest.TestCase):
    """Validate numerical properties studied in Lab 15."""

    def test_q15_mul_round_edges_and_accuracy(self) -> None:
        m1 = _load_lab_module("lab15_01", "01_q_formats.py")
        # -32768 * -32768 = +1.0 -> saturates to 32767 (not -32768 wrap)
        self.assertEqual(m1.q15_mul_round(-32768, -32768), 32767)
        # 0.5 * 0.5 = 0.25
        self.assertEqual(m1.q15_mul_round(16384, 16384), 8192)
        # -0.5 * 0.5 = -0.25
        self.assertEqual(m1.q15_mul_round(-16384, 16384), -8192)
        # -1 * 1 = -1 -> (-1 + 16384) >> 15 = 0
        self.assertEqual(m1.q15_mul_round(-1, 1), 0)

    def test_entropy_fixed_point_modes_within_bound(self) -> None:
        m2 = _load_lab_module("lab15_02", "02_entropy_fixed_point.py")
        from src.features.asymmetry_maps import compute_asymmetry_for_grid, reconstruct_grids
        from src.features.entropy_maps import build_entropy_plan, compute_entropy_features

        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
        plan = build_entropy_plan(128, [1, 2, 4, 8], 16, backend="numpy_baseline")
        x85 = compute_entropy_features(img[None, :, :], [1, 2, 4, 8], 16, plan=plan)[0]
        feats = list(x85)
        offs = {1: (0, 1), 2: (1, 5), 4: (5, 21), 8: (21, 85)}
        for g in (2, 4, 8):
            s, e = offs[g]
            ar = compute_asymmetry_for_grid(reconstruct_grids(x85[s:e], g), kinds=("lr", "tb", "rot180"))
            feats.extend([float(ar["lr"][0]), float(ar["tb"][0]), float(ar["rot180"][0])])
        ref = np.asarray(feats, dtype=np.float64)

        for mode, tol in [("nlogn_full", 1e-4), ("term_per_level", 5e-4), ("log2_mantissa", 2e-3)]:
            q_val = m2.compute_x94_fixed_point(img, [1, 2, 4, 8], 16, mode=mode, mantissa_bits=8)
            max_err = float(np.max(np.abs(q_val - ref)))
            self.assertLess(max_err, tol, msg=f"mode={mode} max_err={max_err}")

    def test_bn_folding_exact_match(self) -> None:
        m3 = _load_lab_module("lab15_03", "03_bn_folding.py")
        torch.manual_seed(42)
        conv = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)
        bn = nn.BatchNorm2d(8)
        # Simulate trained statistics
        with torch.no_grad():
            bn.running_mean.normal_(0.1, 0.5)
            bn.running_var.uniform_(0.2, 2.0)
            bn.weight.uniform_(0.5, 1.5)
            bn.bias.normal_(0.0, 0.2)
        x = torch.randn(2, 4, 16, 16)
        diff = m3.verify_conv_bn_fold_pair(conv, bn, x)
        self.assertLess(diff, 1e-5)

    def test_requantization_accuracy_and_shift(self) -> None:
        m4 = _load_lab_module("lab15_04", "04_requantization.py")
        rng = np.random.default_rng(7)
        x_f = rng.uniform(-0.8, 0.8, size=(4, 8, 8)).astype(np.float32)
        w_f = rng.uniform(-0.4, 0.4, size=(4, 4, 3, 3)).astype(np.float32)
        b_f = rng.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        with torch.no_grad():
            y_ref = torch.nn.functional.conv2d(
                torch.from_numpy(x_f).unsqueeze(0),
                torch.from_numpy(w_f),
                torch.from_numpy(b_f),
                padding=1,
            ).squeeze(0).numpy()

        xf = m4.choose_frac_bits(float(np.max(np.abs(x_f))), 16)
        wf = m4.choose_frac_bits(float(np.max(np.abs(w_f))), 16)
        yf = m4.choose_frac_bits(float(np.max(np.abs(y_ref))), 16, headroom_bits=1)
        yq, _ = m4.conv2d_q15_reference(
            m4.quantize_pow2(x_f, xf, 16),
            m4.quantize_pow2(w_f, wf, 16),
            m4.quantize_pow2(b_f, xf + wf, 64),
            shift_out=xf + wf - yf,
        )
        y_deq = yq.astype(np.float64) / float(1 << yf)
        self.assertLess(float(np.mean(np.abs(y_deq - y_ref))), 5e-4)


if __name__ == "__main__":
    unittest.main()
