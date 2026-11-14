"""Unit tests for C99 export TOML loader and fail-fast validation.

Manual execution (PowerShell from 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_c_export_config.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.c_export.config import load_c_imp_config


class TestCExportConfig(unittest.TestCase):
    """Validate single-source config resolution and fail-fast checks."""

    def test_load_kolektorsdd2_and_mvtec_bottle(self) -> None:
        cfg_k = load_c_imp_config(Path("configs/c_imp/kolektorsdd2.toml"), PROJECT_ROOT)
        self.assertEqual(cfg_k.effective_target_tag, "kolektorsdd2")
        self.assertEqual(cfg_k.descriptor_spec.image_size, 128)
        self.assertEqual(cfg_k.descriptor_spec.bins, 16)
        self.assertEqual(cfg_k.descriptor_spec.grid_levels, (1, 2, 4, 8))
        self.assertEqual(cfg_k.descriptor_spec.n_x94, 94)
        self.assertEqual(cfg_k.cnn_cfg.data.normalize_mean, 0.5)
        self.assertIn("torchvision_mobilenet", cfg_k.cnn.models)

        cfg_m = load_c_imp_config(
            Path("configs/c_imp/mvtec_ad.toml"), PROJECT_ROOT, object_name="bottle"
        )
        self.assertEqual(cfg_m.effective_target_tag, "mvtec_ad_bottle")
        self.assertEqual(cfg_m.object_name, "bottle")

        cfg_inv = load_c_imp_config(
            Path("configs/c_imp/kolektorsdd2.toml"), PROJECT_ROOT, inverted=True
        )
        self.assertEqual(cfg_inv.effective_target_tag, "kolektorsdd2-inverted")
        self.assertTrue(cfg_inv.classification_cfg.inverted)
        self.assertTrue(cfg_inv.cnn_cfg.inverted)

    def test_cwd_independence(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                cfg = load_c_imp_config(Path("configs/c_imp/kolektorsdd2.toml"), PROJECT_ROOT)
                self.assertEqual(cfg.project_root, PROJECT_ROOT.resolve())
                self.assertTrue(cfg.entropy_config_path.is_file())
            finally:
                os.chdir(old_cwd)

    def test_fail_fast_validation_errors(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_c_imp_config(Path("configs/c_imp/does_not_exist.toml"), PROJECT_ROOT)

        with self.assertRaises(ValueError):
            load_c_imp_config(
                Path("configs/c_imp/mvtec_ad.toml"), PROJECT_ROOT, object_name="invalid_obj"
            )

        with tempfile.TemporaryDirectory() as tmp:
            bad_toml = Path(tmp) / "bad.toml"
            text = (PROJECT_ROOT / "configs/c_imp/kolektorsdd2.toml").read_text(encoding="utf-8")
            bad_text = text.replace('shannon_lut_mode = "log2_mantissa"', 'shannon_lut_mode = "unknown_lut"')
            bad_toml.write_text(bad_text, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_c_imp_config(bad_toml, PROJECT_ROOT)

            bad_model = Path(tmp) / "bad_cnn.toml"
            bad_model.write_text(text.replace('"ultra_tiny"', '"no_such_cnn"'), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_c_imp_config(bad_model, PROJECT_ROOT)

            bad_frac = Path(tmp) / "bad_frac.toml"
            bad_frac.write_text(text.replace("feature_frac_bits = 15", "feature_frac_bits = 24"), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_c_imp_config(bad_frac, PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
