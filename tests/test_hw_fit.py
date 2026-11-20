"""Unit tests for Stage 8 fit report verdict taxonomy and report generation.

Manual execution (PowerShell from `code/` directory):
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_hw_fit.py -v
"""

from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hw_test.config import load_hw_test_config
from src.hw_test.fit_report import (
    VERDICT_DOES_NOT_FIT,
    VERDICT_FITS_DOMINATED,
    VERDICT_FITS_EXCEEDS_LATENCY,
    VERDICT_FITS_VIABLE,
    VERDICT_NOT_MEASURED,
    FitRow,
    evaluate_verdicts,
    generate_fit_report,
)


def _make_row(
    *,
    board_id: str = "esp32cam",
    model: str,
    mode: str = "q15",
    fit_possible: bool = True,
    reason: str = "ok",
    median_total_ms: float | None = None,
    offline_roc_auc: float | None = 0.85,
) -> FitRow:
    return FitRow(
        board_id=board_id,
        model=model,
        mode=mode,
        fit_possible=fit_possible,
        verdict="",
        reason=reason,
        measured_transport="uart" if median_total_ms is not None else None,
        pred_weight_flash_bytes=4096,
        pred_arena_bytes=0,
        compile_ok=fit_possible,
        flash_used_bytes=280000,
        flash_max_bytes=3145728,
        ram_used_bytes=32000,
        ram_max_bytes=327680,
        free_ram_boot=200000,
        free_heap_min=150000,
        stack_hwm=2048,
        free_psram_min=None,
        arena_location="sram",
        median_selector_ms=0.0,
        median_model_ms=median_total_ms,
        median_total_ms=median_total_ms,
        parity_rate=1.0 if median_total_ms is not None else None,
        offline_roc_auc=offline_roc_auc,
        offline_f1=0.60,
    )


class TestHwFitReport(unittest.TestCase):
    """Validate Stage 8 verdict precedence, `lr_dot` baseline exemption, and CSV/MD tagging."""

    def test_verdict_precedence_and_lr_dot_exemption(self) -> None:
        rows = [
            # 1. does_not_fit takes precedence even if unmeasured or slow
            _make_row(
                board_id="esp32cam",
                model="small_cnn",
                fit_possible=False,
                reason="flash overflow",
                median_total_ms=999.0,
                offline_roc_auc=0.95,
            ),
            # 2. not_measured when fit_possible=True and no device measurement
            _make_row(
                board_id="esp32cam",
                model="ds_cnn",
                fit_possible=True,
                median_total_ms=None,
                offline_roc_auc=0.88,
            ),
            # 3. fits_exceeds_latency (> max_latency_ms=500.0)
            _make_row(
                board_id="esp32cam",
                model="torchvision_mobilenet",
                fit_possible=True,
                median_total_ms=750.0,
                offline_roc_auc=0.74,
            ),
            # 4. Fast + higher AUC model that dominates both lr_dot (exempt!) and tiny_cnn (dominated!)
            _make_row(
                board_id="esp32cam",
                model="ultra_tiny",
                fit_possible=True,
                median_total_ms=10.0,
                offline_roc_auc=0.89,
            ),
            # lr_dot baseline MUST remain fits_viable even though ultra_tiny is faster & higher AUC
            _make_row(
                board_id="esp32cam",
                model="lr_dot",
                fit_possible=True,
                median_total_ms=20.0,
                offline_roc_auc=0.80,
            ),
            # tiny_cnn is slower (150ms > 10ms) and lower AUC (0.87 < 0.89) -> fits_dominated
            _make_row(
                board_id="esp32cam",
                model="tiny_cnn",
                fit_possible=True,
                median_total_ms=150.0,
                offline_roc_auc=0.87,
            ),
            # tiny_resnet is slower (250ms) but has highest AUC (0.92) -> fits_viable (Pareto frontier)
            _make_row(
                board_id="esp32cam",
                model="tiny_resnet",
                fit_possible=True,
                median_total_ms=250.0,
                offline_roc_auc=0.92,
            ),
            # Separate board: nodemcu slower/lower AUC than esp32cam ultra_tiny must NOT be dominated across boards
            _make_row(
                board_id="nodemcu",
                model="mlp_16_dot",
                fit_possible=True,
                median_total_ms=30.0,
                offline_roc_auc=0.83,
            ),
        ]

        evaluated = evaluate_verdicts(rows, max_latency_ms=500.0)
        by_key = {(r.board_id, r.model): r for r in evaluated}

        self.assertEqual(by_key[("esp32cam", "small_cnn")].verdict, VERDICT_DOES_NOT_FIT)
        self.assertEqual(by_key[("esp32cam", "ds_cnn")].verdict, VERDICT_NOT_MEASURED)
        self.assertEqual(by_key[("esp32cam", "torchvision_mobilenet")].verdict, VERDICT_FITS_EXCEEDS_LATENCY)
        self.assertEqual(by_key[("esp32cam", "ultra_tiny")].verdict, VERDICT_FITS_VIABLE)
        self.assertEqual(by_key[("esp32cam", "lr_dot")].verdict, VERDICT_FITS_VIABLE)
        self.assertEqual(by_key[("esp32cam", "tiny_cnn")].verdict, VERDICT_FITS_DOMINATED)
        self.assertIn("ultra_tiny", by_key[("esp32cam", "tiny_cnn")].reason)
        self.assertEqual(by_key[("esp32cam", "tiny_resnet")].verdict, VERDICT_FITS_VIABLE)
        self.assertEqual(by_key[("nodemcu", "mlp_16_dot")].verdict, VERDICT_FITS_VIABLE)

    def test_generate_fit_report_end_to_end_with_synthetic_runs(self) -> None:
        base_cfg = load_hw_test_config(Path("configs/hw_test/boards.toml"), PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            # Seed one synthetic Nano run and one build cache entry
            nano_run = results_dir / "nano" / "20261119T120000Z_lr_dot_q15_uart"
            nano_run.mkdir(parents=True)
            (nano_run / "summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "20261119T120000Z_lr_dot_q15_uart",
                        "board_id": "nano",
                        "model": "lr_dot",
                        "mode": "q15",
                        "transport": "uart",
                        "parity": {"parity_rate": 1.0},
                        "timing_ms": {
                            "median_selector_ms": 18.0,
                            "median_model_ms": 0.4,
                            "median_total_ms": 18.4,
                        },
                        "memory": {
                            "free_ram_boot": 616,
                            "free_heap_min_overall": None,
                            "stack_hwm_min_overall": 420,
                            "free_psram_min_overall": None,
                            "arena_location": "none",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (results_dir / "nano" / "build_cache.json").write_text(
                json.dumps(
                    {
                        "board_id": "nano",
                        "builds": {
                            "lr_dot|q15": {
                                "model": "lr_dot",
                                "mode": "q15",
                                "compile_ok": True,
                                "flash_used_bytes": 3864,
                                "flash_max_bytes": 30720,
                                "ram_used_bytes": 1432,
                                "ram_max_bytes": 2048,
                                "fits_budget": True,
                                "reason": "ok",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            cfg = replace(base_cfg, paths=replace(base_cfg.paths, results_dir=results_dir))
            out = generate_fit_report(cfg)
            self.assertTrue(out.csv_path.is_file())
            self.assertTrue(out.md_path.is_file())

            with out.csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                rows = list(reader)

            # Verify every value column carries [predicted], [measured], or [offline]
            for h in headers:
                if h not in ("board", "model", "mode", "verdict", "reason", "measured_transport"):
                    self.assertTrue(
                        any(tag in h for tag in ("[predicted]", "[measured]", "[offline]")),
                        f"Missing source tag in column {h!r}",
                    )

            by_key = {(r["board"], r["model"], r["mode"]): r for r in rows}
            # Nano: lr_dot measured & viable; mlp_16_dot not_measured; all 7 CNNs does_not_fit
            self.assertEqual(by_key[("nano", "lr_dot", "q15")]["verdict"], VERDICT_FITS_VIABLE)
            self.assertEqual(by_key[("nano", "mlp_16_dot", "q15")]["verdict"], VERDICT_NOT_MEASURED)
            self.assertEqual(by_key[("nano", "ultra_tiny", "q15")]["verdict"], VERDICT_DOES_NOT_FIT)
            self.assertIn("163840", by_key[("nano", "ultra_tiny", "q15")]["reason"])

            # NodeMCU: CNNs explicitly listed as does_not_fit with arena reason
            self.assertEqual(by_key[("nodemcu", "ultra_tiny", "q15")]["verdict"], VERDICT_DOES_NOT_FIT)
            self.assertEqual(by_key[("nodemcu", "ultra_tiny", "f32")]["verdict"], VERDICT_DOES_NOT_FIT)

            # Decision D3: torchvision_mobilenet only in q15 on esp32cam
            self.assertIn(("esp32cam", "torchvision_mobilenet", "q15"), by_key)
            self.assertNotIn(("esp32cam", "torchvision_mobilenet", "f32"), by_key)


if __name__ == "__main__":
    unittest.main()
