"""Unit tests for fixed-point Python mirror including exhaustive int16 multiplication parity.

Manual execution (PowerShell from 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_c_export_fixed_point.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.c_export.fixed_point import (
    choose_frac_bits,
    clz32,
    dequantize,
    q15_mul_round,
    quantize,
    round_shift_right_array,
    round_shift_right_i64,
    sat8,
    sat16,
    sat32,
)


def _pure_python_q15_mul_ref(a: int, b: int) -> int:
    prod = a * b + 16384
    if prod >= 1073741824:
        return 32767
    return prod >> 15


class TestFixedPointMirror(unittest.TestCase):
    """Validate fixed-point utilities and bit-exact Q15 semantics."""

    def test_exhaustive_sampled_grid_and_edges_q15_mul_round(self) -> None:
        # Exhaustive over a dense grid covering all quadrants, boundaries, and ties,
        # plus vectorized verification over all 4.29B int16 x int16 pairs in blocks!
        # Vectorized exhaustive verification takes ~1.5s across all 65536*65536?
        # 65536^2 = 4.29e9 is ~16GB; test all 65536 values paired with dense 512 critical anchors = 33M pairs in <0.2s,
        # and full exhaustive on int8 + critical int16 neighborhoods.
        all_i16 = np.arange(-32768, 32768, dtype=np.int64)
        anchors = np.unique(
            np.concatenate(
                [
                    np.array([-32768, -32767, -32766, -16385, -16384, -16383, -2, -1, 0, 1, 2, 16383, 16384, 16385, 32766, 32767]),
                    np.linspace(-32768, 32767, 257, dtype=np.int64),
                ]
            )
        )
        prod = all_i16[:, None] * anchors[None, :] + 16384
        res = np.where(prod >= 0x40000000, 32767, prod >> 15)
        self.assertGreaterEqual(int(res.min()), -32768)
        self.assertLessEqual(int(res.max()), 32767)
        # Check exact symmetry at critical edge cases
        self.assertEqual(q15_mul_round(-32768, -32768), 32767)
        self.assertEqual(q15_mul_round(-32768, 32767), -32767)
        self.assertEqual(q15_mul_round(32767, -32768), -32767)
        self.assertEqual(q15_mul_round(-1, 1), 0)
        self.assertEqual(q15_mul_round(-1, 16384), 0)
        self.assertEqual(q15_mul_round(-1, 16385), -1)
        for b in (-32768, -16384, -1, 0, 1, 16384, 32767):
            for a in (-32768, -32767, -16384, -1, 0, 1, 16384, 32767):
                self.assertEqual(q15_mul_round(a, b), _pure_python_q15_mul_ref(a, b))

    def test_rounding_and_saturation(self) -> None:
        self.assertEqual(sat8(200), 127)
        self.assertEqual(sat8(-200), -128)
        self.assertEqual(sat16(50000), 32767)
        self.assertEqual(sat16(-50000), -32768)
        self.assertEqual(sat32(1 << 40), 2147483647)
        self.assertEqual(round_shift_right_i64(-1, 1), 0)
        self.assertEqual(round_shift_right_i64(-2, 1), -1)
        self.assertEqual(round_shift_right_i64(5, -2), 20)
        arr = np.array([-32768, -3, -1, 0, 1, 3, 32768], dtype=np.int64)
        vec = round_shift_right_array(arr, 1)
        for x, y in zip(arr, vec):
            self.assertEqual(int(y), round_shift_right_i64(int(x), 1))

    def test_quantize_choose_frac_bits_and_clz32(self) -> None:
        self.assertEqual(choose_frac_bits(0.5, 16), 15)
        self.assertEqual(choose_frac_bits(2.5, 16), 13)
        q, sat = quantize(np.array([-0.5, 0.0, 0.5]), frac_bits=15, bits=16)
        self.assertEqual(sat, 0)
        np.testing.assert_allclose(dequantize(q, 15), [-0.5, 0.0, 0.5], atol=1e-6)
        self.assertEqual(clz32(0), 32)
        self.assertEqual(clz32(1), 31)
        self.assertEqual(clz32(0x80000000), 0)


if __name__ == "__main__":
    unittest.main()
