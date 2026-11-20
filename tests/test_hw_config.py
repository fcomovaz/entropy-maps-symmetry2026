"""Unit tests for `hw_test` TOML configuration loading and fail-fast validation.

Manual execution (PowerShell from `code/` directory):
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_hw_config.py -v
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

from src.hw_test.config import load_hw_test_config, load_wifi_secrets


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise AssertionError(f"Expected snippet {old!r} not found in TOML text")
    return text.replace(old, new, 1)


class TestHwConfig(unittest.TestCase):
    """Validate `configs/hw_test/boards.toml`, secrets loading, and error cases."""

    def setUp(self) -> None:
        self.boards_toml = PROJECT_ROOT / "configs" / "hw_test" / "boards.toml"
        self.base_text = self.boards_toml.read_text(encoding="utf-8")

    def _load_modified(self, tmp_dir: Path, mutated_text: str):
        cfg_file = tmp_dir / "boards.toml"
        cfg_file.write_text(mutated_text, encoding="utf-8")
        return load_hw_test_config(cfg_file, PROJECT_ROOT)

    def test_load_repo_boards_toml_and_effective_settings(self) -> None:
        cfg = load_hw_test_config(Path("configs/hw_test/boards.toml"), PROJECT_ROOT)
        self.assertEqual(cfg.project_root, PROJECT_ROOT.resolve())
        self.assertEqual(cfg.protocol.magic, 0x5345)
        self.assertEqual(cfg.protocol.version, 1)
        self.assertEqual(cfg.images.width, 128)
        self.assertEqual(cfg.images.height, 128)
        self.assertEqual(cfg.images.mode, "L")
        self.assertEqual(set(cfg.boards.keys()), {"nano", "uno", "nodemcu", "esp32cam"})

        nano = cfg.get_board("nano")
        self.assertEqual(nano.fqbn, "arduino:avr:nano:cpu=atmega328old")
        self.assertTrue(nano.streaming)
        self.assertEqual(nano.band_rows, 1)
        self.assertEqual(nano.max_payload_bytes, 132)
        self.assertEqual(nano.flash_app_bytes, 30720)
        self.assertEqual(nano.numeric_modes, ("q15",))
        self.assertTrue(nano.serial_reset_on_open)
        self.assertIsNone(nano.upload_baud)

        uno = cfg.get_board("uno")
        self.assertEqual(uno.fqbn, "arduino:avr:uno")
        self.assertEqual(uno.flash_app_bytes, 32256)
        self.assertTrue(uno.serial_reset_on_open)

        nodemcu = cfg.get_board("nodemcu")
        self.assertEqual(nodemcu.transports, ("uart", "wifi"))
        self.assertEqual(nodemcu.tcp_port, 5000)
        self.assertFalse(nodemcu.streaming)
        self.assertFalse(nodemcu.serial_reset_on_open)
        self.assertEqual(nodemcu.band_rows, 8)
        self.assertEqual(nodemcu.max_payload_bytes, 1028)

        esp32 = cfg.get_board("esp32cam")
        self.assertEqual(esp32.fqbn, "esp32:esp32:esp32cam:PartitionScheme=huge_app")
        self.assertNotIn("PSRAM=enabled", esp32.fqbn)
        self.assertEqual(esp32.upload_baud, 115200)
        self.assertFalse(esp32.serial_reset_on_open)
        self.assertEqual(esp32.psram_bytes, 4194304)
        self.assertEqual(esp32.arena_location, "auto")
        self.assertEqual(len(esp32.models), 9)

        # Effective settings resolution with and without model overrides
        eff_nano = cfg.effective_settings("nano", "lr_dot")
        self.assertEqual(eff_nano.warmup_iters, 0)
        self.assertEqual(eff_nano.timed_iters, 1)

        eff_ultra = cfg.effective_settings("esp32cam", "ultra_tiny")
        self.assertEqual(eff_ultra.warmup_iters, 3)
        self.assertEqual(eff_ultra.timed_iters, 10)
        self.assertEqual(eff_ultra.result_timeout_s, 30.0)
        self.assertEqual(eff_ultra.numeric_modes, ("q15", "f32"))

        eff_ds = cfg.effective_settings("esp32cam", "ds_cnn")
        self.assertEqual(eff_ds.warmup_iters, 1)
        self.assertEqual(eff_ds.timed_iters, 5)
        self.assertEqual(eff_ds.result_timeout_s, 120.0)

        eff_mv = cfg.effective_settings("esp32cam", "torchvision_mobilenet")
        self.assertEqual(eff_mv.numeric_modes, ("q15",))
        self.assertEqual(eff_mv.warmup_iters, 1)
        self.assertEqual(eff_mv.timed_iters, 3)
        self.assertEqual(eff_mv.result_timeout_s, 300.0)

        with self.assertRaises(ValueError):
            cfg.effective_settings("nano", "ultra_tiny")

    def test_cwd_independence_and_missing_firmware_secrets_allowed(self) -> None:
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mutated = _replace_once(
                self.base_text,
                'firmware_dir = "hw_test/firmware"',
                'firmware_dir = "nonexistent_firmware_dir"',
            )
            mutated = _replace_once(
                mutated,
                'secrets_config = "configs/hw_test/secrets.toml"',
                'secrets_config = "nonexistent_secrets.toml"',
            )
            try:
                os.chdir(tmp_path)
                cfg = self._load_modified(tmp_path, mutated)
                self.assertEqual(cfg.project_root, PROJECT_ROOT.resolve())
                self.assertFalse(cfg.paths.firmware_dir.exists())
                self.assertFalse(cfg.secrets_config_path.exists())
            finally:
                os.chdir(old_cwd)

    def test_load_wifi_secrets(self) -> None:
        secrets = load_wifi_secrets(PROJECT_ROOT / "configs" / "hw_test" / "secrets.example.toml")
        self.assertEqual(secrets.ssid, "your-ssid")
        self.assertEqual(secrets.password, "your-password")

        with self.assertRaises(FileNotFoundError):
            load_wifi_secrets(PROJECT_ROOT / "configs" / "hw_test" / "does_not_exist_secrets.toml")

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "secrets.toml"
            bad.write_text('[wifi]\npassword = "abc"\n', encoding="utf-8")
            with self.assertRaises(KeyError) as ctx:
                load_wifi_secrets(bad)
            self.assertIn("ssid", str(ctx.exception))

            bad.write_text('[wifi]\nssid = "   "\npassword = "abc"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_wifi_secrets(bad)

    def test_missing_keys_report_section_and_key(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_hw_test_config(Path("configs/hw_test/missing.toml"), PROJECT_ROOT)

        cases = [
            ('seed = 42\n', "[hw_test]", "seed"),
            ('images_dir = "hw_test/img2send"\n', "[hw_test.paths]", "images_dir"),
            ('magic = 0x5345\n', "[hw_test.protocol]", "magic"),
            ('max_latency_ms = 500.0\n', "[hw_test.fit]", "max_latency_ms"),
            ('fqbn = "arduino:avr:uno"\n', "[hw_test.boards.uno]", "fqbn"),
            ("serial_reset_on_open = true\n", "[hw_test.boards.nano]", "serial_reset_on_open"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for line, section, key in cases:
                with self.subTest(section=section, key=key):
                    mutated = _replace_once(self.base_text, line, "")
                    with self.assertRaises(KeyError) as ctx:
                        self._load_modified(tmp_path, mutated)
                    msg = str(ctx.exception)
                    self.assertIn(key, msg)
                    self.assertIn(section, msg)

    def test_invalid_transports_and_wifi_without_tcp_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with self.assertRaises(ValueError):
                self._load_modified(
                    tmp_path,
                    _replace_once(self.base_text, 'transports = ["uart"]', 'transports = ["ble"]'),
                )
            with self.assertRaises(ValueError):
                self._load_modified(
                    tmp_path,
                    _replace_once(
                        self.base_text,
                        'transports = ["uart", "wifi"]',
                        'transports = ["uart", "uart"]',
                    ),
                )
            with self.assertRaises(KeyError) as ctx:
                self._load_modified(
                    tmp_path,
                    _replace_once(self.base_text, "tcp_port = 5000\n", ""),
                )
            self.assertIn("tcp_port", str(ctx.exception))
            self.assertIn("nodemcu", str(ctx.exception))

    def test_invalid_models_images_band_rows_payload_and_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            invalid_mutations = [
                # Unknown model not in c_imp_cfg
                ('models = ["lr_dot", "mlp_16_dot"]', 'models = ["lr_dot", "unknown_net"]'),
                # Non-square image
                ("width = 128", "width = 64"),
                # Image size mismatch vs c_imp descriptor (128)
                ("width = 128\nheight = 128", "width = 64\nheight = 64"),
                # Invalid mode / extension
                ('mode = "L"', 'mode = "RGB"'),
                ('allowed_extensions = [".png"]', 'allowed_extensions = ["png"]'),
                # band_rows not dividing 128 or <= 0
                ("band_rows = 1", "band_rows = 3"),
                ("band_rows = 1", "band_rows = 0"),
                # max_payload_bytes < 4 + band_rows * width
                ("max_payload_bytes = 132", "max_payload_bytes = 131"),
                ("max_payload_bytes = 1028", "max_payload_bytes = 1027"),
                # Budgets <= 0 or usable > physical
                ("usable_sram_budget_bytes = 1792", "usable_sram_budget_bytes = 0"),
                ("usable_sram_budget_bytes = 1792", "usable_sram_budget_bytes = 4096"),
                ("flash_app_bytes = 30720", "flash_app_bytes = 0"),
                ("psram_bytes = 0", "psram_bytes = -1"),
                ("max_latency_ms = 500.0", "max_latency_ms = 0.0"),
                # Invalid family / arena / numeric_modes
                ('family = "avr"', 'family = "stm32"'),
                ('arena_location = "none"', 'arena_location = "sdram"'),
                ('arena_location = "none"', 'arena_location = "psram"'),
                ('numeric_modes = ["q15"]', 'numeric_modes = ["int8"]'),
                # Invalid model override key / model
                (
                    "[hw_test.boards.esp32cam.model_overrides.ds_cnn]",
                    "[hw_test.boards.esp32cam.model_overrides.no_such_cnn]",
                ),
                ("result_timeout_s = 120.0", "result_timeout_s = -5.0"),
                ("upload_baud = 115200", "upload_baud = 0"),
            ]
            for old, new in invalid_mutations:
                with self.subTest(old=old, new=new):
                    mutated = _replace_once(self.base_text, old, new)
                    with self.assertRaises(ValueError):
                        self._load_modified(tmp_path, mutated)


if __name__ == "__main__":
    unittest.main()
