"""Unit tests for firmware staging, mocked `arduino-cli` compilation, and `experiments.hw_test` CLI.

Manual execution (PowerShell from `code/` directory):
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_hw_cli.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.hw_test import main as hw_test_main
from src.hw_test.config import WifiSecrets, load_hw_test_config
from src.hw_test.firmware import compile_sketch, stage_sketch
from src.hw_test.images import GoldenLookup
from src.hw_test.transport import FakeTransport
from tests.test_hw_session import GoldenReplayDevice


class MockArduinoCliRunner:
    """Deterministic `subprocess.run` stub for `arduino-cli` and `git` commands."""

    def __init__(self, *, missing_core: str | None = None, fail_compile: bool = False) -> None:
        self.missing_core = missing_core
        self.fail_compile = fail_compile
        self.commands: list[list[str]] = []

    def __call__(self, cmd: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(cmd))
        if cmd[0] == "git":
            return subprocess.CompletedProcess(cmd, 0, stdout="a1b2c3d\n", stderr="")

        sub = cmd[1]
        if sub == "version":
            return subprocess.CompletedProcess(
                cmd, 0, stdout=json.dumps({"Application": "arduino-cli", "VersionString": "1.5.2-rc.1"}), stderr=""
            )
        if sub == "core" and cmd[2] == "list":
            platforms = [
                {"id": "arduino:avr", "installed_version": "1.8.8"},
                {"id": "esp8266:esp8266", "installed_version": "3.1.2"},
                {"id": "esp32:esp32", "installed_version": "3.3.11"},
            ]
            if self.missing_core:
                platforms = [p for p in platforms if p["id"] != self.missing_core]
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"platforms": platforms}), stderr="")

        if sub == "board" and cmd[2] == "list":
            data = {
                "detected_ports": [
                    {
                        "port": {
                            "address": "COM4",
                            "protocol": "serial",
                            "protocol_label": "Serial Port (USB)",
                            "properties": {"vid": "0x1A86", "pid": "0x7523"},
                        }
                    }
                ]
            }
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(data), stderr="")

        if sub == "compile":
            fqbn = cmd[cmd.index("--fqbn") + 1]
            if self.fail_compile:
                payload = {
                    "success": False,
                    "compiler_out": "",
                    "compiler_err": "region `flash' overflowed by 4096 bytes",
                    "builder_result": {},
                }
                return subprocess.CompletedProcess(cmd, 1, stdout=json.dumps(payload), stderr="")

            if fqbn.startswith("arduino:avr"):
                payload = {
                    "success": True,
                    "compiler_out": "",
                    "compiler_err": "",
                    "builder_result": {
                        "executable_sections_size": [
                            {"name": "text", "size": 3864, "max_size": 30720},
                            {"name": "data", "size": 1432, "max_size": 2048},
                        ],
                        "board_platform": {"id": "arduino:avr", "version": "1.8.8"},
                    },
                }
            elif fqbn.startswith("esp8266:esp8266"):
                payload = {
                    "success": True,
                    "compiler_out": "",
                    "compiler_err": (
                        ". Variables and constants in RAM (global, static), used 47200 / 80192 bytes (58%)\n"
                        "╠══ RODATA   920      constants\n"
                        ". Instruction RAM (IRAM_ATTR, ICACHE_RAM_ATTR), used 59143 / 65536 bytes (90%)\n"
                        ". Code in flash (default, ICACHE_FLASH_ATTR), used 245000 / 1048576 bytes (23%)\n"
                    ),
                    "builder_result": {
                        "board_platform": {"id": "esp8266:esp8266", "version": "3.1.2"},
                    },
                }
            else:
                payload = {
                    "success": True,
                    "compiler_out": "",
                    "compiler_err": "",
                    "builder_result": {
                        "executable_sections_size": [
                            {"name": "text", "size": 310000, "max_size": 3145728},
                            {"name": "data", "size": 38000, "max_size": 327680},
                        ],
                        "board_platform": {"id": "esp32:esp32", "version": "3.3.11"},
                    },
                }
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

        if sub == "upload":
            return subprocess.CompletedProcess(cmd, 0, stdout="Uploaded OK\n", stderr="")

        raise AssertionError(f"Unexpected mocked command: {cmd}")


class TestHwCliAndFirmware(unittest.TestCase):
    """Validate staging isolation, mocked builds/flashes, and every CLI subcommand."""

    def _make_temp_config(self, tmp_dir: Path) -> Path:
        raw = (PROJECT_ROOT / "configs" / "hw_test" / "boards.toml").read_text(encoding="utf-8")
        staging = (tmp_dir / "build").as_posix()
        results = (tmp_dir / "results").as_posix()
        mutated = raw.replace('staging_dir = "hw_test/build"', f'staging_dir = "{staging}"', 1)
        mutated = mutated.replace('results_dir = "results/hw_test"', f'results_dir = "{results}"', 1)
        cfg_path = tmp_dir / "boards.toml"
        cfg_path.write_text(mutated, encoding="utf-8")
        return cfg_path

    def test_stage_sketch_tabular_vs_cnn_and_compile_parsers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = self._make_temp_config(Path(tmp))
            cfg = load_hw_test_config(cfg_path, PROJECT_ROOT)
            runner = MockArduinoCliRunner()

            # 1. AVR tabular staging must exclude infra_pytorch.* and set CIMP_ENABLE_F32 0
            avr_dir = stage_sketch(cfg, "nano", "lr_dot", "q15", "uart", runner=runner)
            self.assertTrue((avr_dir / f"{avr_dir.name}.ino").is_file())
            self.assertTrue((avr_dir / "cimp_descriptor.c").is_file())
            self.assertTrue((avr_dir / "infra_sklearn.c").is_file())
            self.assertFalse((avr_dir / "infra_pytorch.c").exists())
            self.assertFalse((avr_dir / "infra_pytorch.h").exists())
            reg_txt = (avr_dir / "cimp_models_registry.h").read_text(encoding="utf-8")
            self.assertIn("#define CIMP_HAS_MODEL_LR_DOT 1", reg_txt)
            self.assertIn("#define CIMP_HAS_MODEL_MLP_16_DOT 0", reg_txt)
            bld_txt = (avr_dir / "hw_build_config.h").read_text(encoding="utf-8")
            self.assertIn("#define CIMP_ENABLE_F32 0", bld_txt)
            self.assertIn("#define HW_STREAMING 1", bld_txt)

            # 2. ESP32 CNN staging must exclude descriptor/sklearn .c and include infra_pytorch.c
            cnn_dir = stage_sketch(
                cfg,
                "esp32cam",
                "ultra_tiny",
                "q15",
                "wifi",
                wifi_secrets=WifiSecrets("TestNet", "SecretPass"),
                runner=runner,
            )
            self.assertTrue((cnn_dir / "infra_pytorch.c").is_file())
            self.assertTrue((cnn_dir / "model_ultra_tiny.c").is_file())
            self.assertTrue((cnn_dir / "cimp_descriptor.h").is_file())
            self.assertFalse((cnn_dir / "cimp_descriptor.c").exists())
            self.assertFalse((cnn_dir / "infra_sklearn.c").exists())
            cnn_bld = (cnn_dir / "hw_build_config.h").read_text(encoding="utf-8")
            self.assertIn('#define HW_WIFI_SSID "TestNet"', cnn_bld)
            self.assertIn("#define HW_ARENA_BYTES 163840UL", cnn_bld)
            self.assertIn("#define HW_ARENA_PREF 3", cnn_bld)

            # 3. Mocked compiles across AVR, ESP8266 (regex parser), and ESP32
            r_avr = compile_sketch(cfg, "nano", "lr_dot", "q15", runner=runner)
            self.assertTrue(r_avr.fits_budget)
            self.assertEqual(r_avr.flash_used_bytes, 3864)
            self.assertEqual(r_avr.ram_used_bytes, 1432)

            r_esp8266 = compile_sketch(cfg, "nodemcu", "lr_dot", "f32", runner=runner)
            self.assertTrue(r_esp8266.fits_budget)
            self.assertEqual(r_esp8266.flash_used_bytes, 245000)
            self.assertEqual(r_esp8266.ram_used_bytes, 47200)

            r_esp32 = compile_sketch(cfg, "esp32cam", "ultra_tiny", "q15", runner=runner)
            self.assertTrue(r_esp32.fits_budget)
            self.assertEqual(r_esp32.flash_used_bytes, 310000)

            # 4. Missing core raises explicit arduino-cli core install guidance
            missing_runner = MockArduinoCliRunner(missing_core="arduino:avr")
            with self.assertRaises(RuntimeError) as ctx:
                compile_sketch(cfg, "nano", "lr_dot", "q15", runner=missing_runner)
            self.assertIn("arduino-cli core install arduino:avr", str(ctx.exception))

    def test_cli_subcommands_dry_run_send_bench_report_and_error_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cfg_path = self._make_temp_config(tmp_dir)
            cfg = load_hw_test_config(cfg_path, PROJECT_ROOT)
            golden = GoldenLookup.from_config(cfg)
            runner = MockArduinoCliRunner()

            self.assertEqual(hw_test_main(["boards", "--config", str(cfg_path)], project_root=PROJECT_ROOT, runner=runner), 0)
            self.assertEqual(hw_test_main(["images", "--config", str(cfg_path), "--image", "20042.png"], project_root=PROJECT_ROOT, runner=runner), 0)
            self.assertEqual(hw_test_main(["build", "--config", str(cfg_path), "--board", "nano", "--model", "lr_dot", "--mode", "q15", "--dry-run"], project_root=PROJECT_ROOT, runner=runner), 0)
            self.assertEqual(hw_test_main(["flash", "--config", str(cfg_path), "--board", "nano", "--model", "lr_dot", "--mode", "q15", "--port", "COM4"], project_root=PROJECT_ROOT, runner=runner), 0)
            nano_uploads = [c for c in runner.commands if len(c) > 1 and c[1] == "upload"]
            self.assertEqual(len(nano_uploads), 1)
            self.assertNotIn("--upload-property", nano_uploads[0])

            self.assertEqual(hw_test_main(["flash", "--config", str(cfg_path), "--board", "esp32cam", "--model", "lr_dot", "--mode", "q15", "--port", "COM3"], project_root=PROJECT_ROOT, runner=runner), 0)
            esp_uploads = [c for c in runner.commands if len(c) > 1 and c[1] == "upload"]
            self.assertEqual(len(esp_uploads), 2)
            self.assertIn("upload.speed=115200", esp_uploads[1])

            # send --dry-run succeeds without port or device
            self.assertEqual(
                hw_test_main(
                    ["send", "--config", str(cfg_path), "--board", "nano", "--model", "lr_dot", "--mode", "q15", "--transport", "uart", "--image", "20042.png", "--dry-run"],
                    project_root=PROJECT_ROOT,
                    runner=runner,
                ),
                0,
            )

            # live send with FakeTransport factory
            def fake_factory(_cfg, board_spec, _tr, _port, _host):
                dev = GoldenReplayDevice(
                    board_id=board_spec.board_id,
                    model="lr_dot",
                    mode="q15",
                    streaming=board_spec.streaming,
                    max_payload=board_spec.max_payload_bytes,
                    is_cnn=False,
                    golden_lookup=golden,
                )
                return FakeTransport(handler=dev.handle_bytes)

            self.assertEqual(
                hw_test_main(
                    ["send", "--config", str(cfg_path), "--board", "nano", "--model", "lr_dot", "--mode", "q15", "--transport", "uart", "--port", "COM4", "--image", "20042.png"],
                    project_root=PROJECT_ROOT,
                    transport_factory=fake_factory,
                    runner=runner,
                ),
                0,
            )

            # bench --compile-only builds all models and produces fit_report.{csv,md}
            self.assertEqual(
                hw_test_main(
                    ["bench", "--config", str(cfg_path), "--board", "nano", "--compile-only"],
                    project_root=PROJECT_ROOT,
                    runner=runner,
                ),
                0,
            )
            self.assertTrue((tmp_dir / "results" / "fit_report.csv").is_file())
            self.assertTrue((tmp_dir / "results" / "fit_report.md").is_file())

            # ESP32 prompt skip and report command
            prompts: list[str] = []
            self.assertEqual(
                hw_test_main(
                    ["bench", "--config", str(cfg_path), "--board", "esp32cam", "--models", "ultra_tiny", "--modes", "q15", "--port", "COM5", "--image", "20042.png"],
                    project_root=PROJECT_ROOT,
                    runner=runner,
                    input_fn=lambda msg: (prompts.append(msg) or "s"),
                ),
                0,
            )
            self.assertEqual(len(prompts), 1)
            self.assertIn("GPIO0", prompts[0])
            self.assertEqual(hw_test_main(["report", "--config", str(cfg_path)], project_root=PROJECT_ROOT, runner=runner), 0)

            # Top-level SystemExit("error: ...") on invalid board
            with self.assertRaises(SystemExit) as ctx:
                hw_test_main(
                    ["send", "--config", str(cfg_path), "--board", "no_board", "--model", "lr_dot", "--mode", "q15", "--transport", "uart", "--all"],
                    project_root=PROJECT_ROOT,
                    runner=runner,
                )
            self.assertTrue(str(ctx.exception).startswith("error: "))

            # clean command: --dry-run, --staging-only, --keep-build-cache, and full clean preserving README.md
            readme_sentinel = tmp_dir / "results" / "README.md"
            readme_sentinel.write_text("# Keep me\n", encoding="utf-8")
            self.assertTrue((tmp_dir / "build" / "nano_lr_dot_q15_uart").is_dir())
            self.assertTrue((tmp_dir / "results" / "nano" / "build_cache.json").is_file())

            # 1. --dry-run removes nothing
            self.assertEqual(
                hw_test_main(["clean", "--config", str(cfg_path), "--dry-run"], project_root=PROJECT_ROOT, runner=runner),
                0,
            )
            self.assertTrue((tmp_dir / "build" / "nano_lr_dot_q15_uart").is_dir())
            self.assertTrue((tmp_dir / "results" / "fit_report.md").is_file())

            # 2. --staging-only --board nano removes only nano staging
            self.assertEqual(
                hw_test_main(
                    ["clean", "--config", str(cfg_path), "--board", "nano", "--staging-only"],
                    project_root=PROJECT_ROOT,
                    runner=runner,
                ),
                0,
            )
            self.assertFalse((tmp_dir / "build" / "nano_lr_dot_q15_uart").exists())
            self.assertTrue((tmp_dir / "build" / "esp32cam_lr_dot_q15_uart").is_dir())
            self.assertTrue((tmp_dir / "results" / "fit_report.md").is_file())

            # 3. --results-only --keep-build-cache removes runs and fit_report.* while keeping build_cache.json & README.md
            self.assertEqual(
                hw_test_main(
                    ["clean", "--config", str(cfg_path), "--results-only", "--keep-build-cache"],
                    project_root=PROJECT_ROOT,
                    runner=runner,
                ),
                0,
            )
            self.assertFalse((tmp_dir / "results" / "fit_report.md").exists())
            self.assertFalse((tmp_dir / "results" / "fit_report.csv").exists())
            self.assertTrue((tmp_dir / "results" / "nano" / "build_cache.json").is_file())
            self.assertTrue(readme_sentinel.is_file())

            # 4. Full clean removes remaining staging and build_cache.json while preserving README.md
            self.assertEqual(
                hw_test_main(["clean", "--config", str(cfg_path)], project_root=PROJECT_ROOT, runner=runner),
                0,
            )
            self.assertFalse((tmp_dir / "build" / "esp32cam_lr_dot_q15_uart").exists())
            self.assertFalse((tmp_dir / "results" / "nano").exists())
            self.assertTrue(readme_sentinel.is_file())


if __name__ == "__main__":
    unittest.main()
