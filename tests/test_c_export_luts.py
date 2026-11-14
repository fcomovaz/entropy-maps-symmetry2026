"""Unit tests for C99 config and LUT header generation.

Manual execution (PowerShell from 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_c_export_luts.py -v
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.c_export.config import load_c_imp_config
from src.c_export.luts import (
    build_bin_quant_lut,
    build_cnn_input_lut,
    build_shannon_lut,
    emit_config_and_lut_headers,
)
from src.features.entropy_maps import build_quantization_lut


class TestCExportLuts(unittest.TestCase):
    """Verify LUT correctness, determinism, and three Shannon modes."""

    def test_bin_quant_matches_src(self) -> None:
        for bins in (2, 16, 64, 256):
            np.testing.assert_array_equal(build_bin_quant_lut(bins), build_quantization_lut(bins))

    def test_cnn_input_lut_range_and_monotonicity(self) -> None:
        f32_lut, q15_lut, n_sat = build_cnn_input_lut(mean=0.5, std=0.5, frac_bits=15)
        self.assertEqual(f32_lut.shape, (256,))
        self.assertEqual(q15_lut.shape, (256,))
        self.assertLessEqual(n_sat, 1)  # p=255 maps +1.0 -> 32767
        self.assertTrue(np.all(np.diff(q15_lut) >= 0))
        self.assertEqual(int(q15_lut[0]), -32768)
        self.assertEqual(int(q15_lut[-1]), 32767)

    def test_shannon_lut_modes_and_deterministic_generation(self) -> None:
        for mode in ("nlogn_full", "term_per_level", "log2_mantissa"):
            b = build_shannon_lut(128, [1, 2, 4, 8], 16, mode=mode, feature_frac_bits=15, mantissa_bits=8)
            self.assertGreater(b.byte_size, 0)
            self.assertEqual(b.mantissa_table[0], 0)
            self.assertEqual(b.mantissa_table[-1], 32768)

        base_cfg = load_c_imp_config(Path("configs/c_imp/kolektorsdd2.toml"), PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            cfg1 = replace(base_cfg, paths=replace(base_cfg.paths, generated_root=Path(tmp1)))
            cfg2 = replace(base_cfg, paths=replace(base_cfg.paths, generated_root=Path(tmp2)))
            meta1 = emit_config_and_lut_headers(cfg1)
            meta2 = emit_config_and_lut_headers(cfg2)
            for fname in ("cimp_config.h", "lut_bin_quant.h", "lut_shannon.h", "lut_cnn_input.h"):
                self.assertEqual(meta1[fname]["sha256"], meta2[fname]["sha256"])
                self.assertTrue((cfg1.generated_dir / fname).is_file())


if __name__ == "__main__":
    unittest.main()
