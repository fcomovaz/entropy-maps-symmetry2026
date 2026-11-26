"""Unit and integration tests for `experiments.visualize_hw_test` and `src.hw_test.visualize`."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.visualize_hw_test import main
from src.hw_test.fit_report import FIT_CSV_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_c_imp_config(tmp_root: Path) -> Path:
    c_cfg = tmp_root / "configs" / "c_imp" / "kolektorsdd2.toml"
    c_cfg.parent.mkdir(parents=True, exist_ok=True)
    c_cfg.write_text(
        """
[c_imp]
dataset_key = "kolektorsdd2"
target_tag = "kolektorsdd2"
seed = 42
generated_dir = "c_imp/generated"
build_dir = "c_imp/build"
results_dir = "results/c_imp"
models = ["lr_dot", "mlp_16_dot", "ultra_tiny", "ds_cnn", "tiny_cnn", "mobilenet_like", "tiny_resnet", "small_cnn", "torchvision_mobilenet"]

[c_imp.features]
bins = 16
grid_levels = [1, 2, 4, 8]
x85_count = 85
x9_count = 9
x94_count = 94

[c_imp.quantization]
f32_tol = 1e-4
q15_frac_bits = 15

[c_imp.verification]
bench_warmup = 1
bench_reps = 1
""",
        encoding="utf-8",
    )
    return c_cfg


def _write_synthetic_run(
    results_dir: Path,
    board: str,
    model: str,
    mode: str,
    t_sel: float,
    t_mod: float,
    free_ram: int,
    stack_hwm: int,
    free_psram: int | None = None,
    arena: int = 0,
    arena_loc: str = "none",
) -> None:
    rdir = results_dir / board / f"20261126T000000Z_{model}_{mode}_uart"
    rdir.mkdir(parents=True, exist_ok=True)
    t_tot = t_sel + t_mod
    summary = {
        "run_id": rdir.name,
        "board_id": board,
        "model": model,
        "mode": mode,
        "transport": "uart",
        "n_images": 2,
        "n_golden_matched": 2,
        "parity": {
            "n_pass": 2,
            "n_fail": 0,
            "n_na": 0,
            "parity_rate": 1.0,
            "max_logit_abs_diff": 0.0,
            "x94_crc_pass": 2,
            "x94_crc_checked": 2,
        },
        "timing_us": {
            "median_selector_us": t_sel * 1000.0,
            "median_model_us": t_mod * 1000.0,
            "median_total_us": t_tot * 1000.0,
            "mean_total_us": t_tot * 1000.0,
            "min_total_us": t_tot * 1000.0,
            "max_total_us": t_tot * 1000.0,
            "first_total_us": t_tot * 1000.0,
        },
        "timing_ms": {
            "median_selector_ms": t_sel,
            "median_model_ms": t_mod,
            "median_total_ms": t_tot,
            "mean_total_ms": t_tot,
            "min_total_ms": t_tot,
            "max_total_ms": t_tot,
            "first_total_ms": t_tot,
        },
        "memory": {
            "free_ram_boot": free_ram,
            "free_heap_min_overall": 210000 if free_psram is not None else None,
            "stack_hwm_min_overall": stack_hwm,
            "free_psram_min_overall": free_psram,
            "arena_bytes": arena,
            "arena_location": arena_loc,
        },
    }
    (rdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (rdir / "per_image.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image", "true_label", "label", "logit0", "golden_logit0", "median_selector_us", "median_model_us", "median_total_us", "parity_ok"]
        )
        writer.writerow(["20000.png", 0, 1, 1.05, 1.05, t_sel * 1000.0, t_mod * 1000.0, t_tot * 1000.0, "True"])
        writer.writerow(["20042.png", 1, 1, 4.52, 4.52, t_sel * 1000.0, t_mod * 1000.0, t_tot * 1000.0, "True"])


class TestHwVisualize(unittest.TestCase):
    """Verify dry-run, overwrite semantics, and all 5 output PNG figures."""

    def test_synthetic_end_to_end_generation_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_root = Path(td)
            shutil.copytree(REPO_ROOT / "configs", tmp_root / "configs")

            res_dir = tmp_root / "results" / "hw_test"
            res_dir.mkdir(parents=True, exist_ok=True)
            fit_csv = res_dir / "fit_report.csv"

            # Populate minimal realistic fit_report.csv across all 4 boards
            rows_data = [
                ("nano", "lr_dot", "q15", "fits_viable", "ok", "uart", 384, 0, "True", 12780, 30720, 1636, 2048, 222, "", 146, "", "none", 76.8, 1.58, 78.38, 1.0, 0.806, 0.425),
                ("nano", "mlp_16_dot", "q15", "fits_viable", "viable", "uart", 6216, 0, "True", 19038, 30720, 1636, 2048, 230, "", 154, "", "none", 77.0, 24.8, 101.8, 1.0, 0.834, 0.489),
                ("nano", "ultra_tiny", "q15", "does_not_fit", "arena > SRAM", "", 3152, 163840, "", "", "", "", "", "", "", "", "", "", "", "", "", "", 0.882, 0.630),
                ("nano", "ds_cnn", "q15", "does_not_fit", "arena > SRAM", "", 27664, 1081344, "", "", "", "", "", "", "", "", "", "", "", "", "", "", 0.878, 0.656),
                ("uno", "lr_dot", "q15", "fits_viable", "ok", "uart", 384, 0, "True", 12780, 32256, 1636, 2048, 222, "", 146, "", "none", 76.8, 1.57, 78.37, 1.0, 0.806, 0.425),
                ("uno", "mlp_16_dot", "q15", "fits_viable", "viable", "uart", 6216, 0, "True", 19038, 32256, 1636, 2048, 230, "", 154, "", "none", 77.1, 24.8, 101.9, 1.0, 0.834, 0.489),
                ("uno", "ultra_tiny", "q15", "does_not_fit", "arena > SRAM", "", 3152, 163840, "", "", "", "", "", "", "", "", "", "", "", "", "", "", 0.882, 0.630),
                ("uno", "ds_cnn", "q15", "does_not_fit", "arena > SRAM", "", 27664, 1081344, "", "", "", "", "", "", "", "", "", "", "", "", "", "", 0.878, 0.656),
                ("nodemcu", "lr_dot", "q15", "not_measured", "no measurement", "", 384, 0, "True", 243572, 1048576, 49204, 80192, "", "", "", "", "", "", "", "", "", 0.806, 0.425),
                ("nodemcu", "lr_dot", "f32", "not_measured", "no measurement", "", 384, 0, "True", 243556, 1048576, 49396, 80192, "", "", "", "", "", "", "", "", "", 0.806, 0.426),
                ("nodemcu", "mlp_16_dot", "q15", "not_measured", "no measurement", "", 6216, 0, "True", 249796, 1048576, 49200, 80192, "", "", "", "", "", "", "", "", "", 0.834, 0.489),
                ("nodemcu", "mlp_16_dot", "f32", "not_measured", "no measurement", "", 6216, 0, "True", 249540, 1048576, 49392, 80192, "", "", "", "", "", "", "", "", "", 0.833, 0.490),
                ("nodemcu", "ultra_tiny", "q15", "does_not_fit", "arena > heap", "", 3152, 163840, "", "", "", "", "", "", "", "", "", "", "", "", "", "", 0.882, 0.630),
                ("nodemcu", "ultra_tiny", "f32", "does_not_fit", "arena > heap", "", 3152, 327680, "", "", "", "", "", "", "", "", "", "", "", "", "", "", 0.882, 0.630),
                ("nodemcu", "torchvision_mobilenet", "q15", "does_not_fit", "arena > heap", "", 1633714, 327680, "", "", "", "", "", "", "", "", "", "", "", "", "", "", 0.741, 0.589),
                ("nodemcu", "torchvision_mobilenet", "f32", "does_not_fit", "arena > heap", "", 1633714, 655360, "", "", "", "", "", "", "", "", "", "", "", "", "", "", 0.733, 0.598),
                ("esp32cam", "lr_dot", "q15", "fits_viable", "ok", "uart", 384, 0, "True", 289344, 3145728, 43116, 327680, 240764, 210296, 6908, 4192124, "none", 3.18, 0.012, 3.19, 1.0, 0.806, 0.425),
                ("esp32cam", "ultra_tiny", "q15", "fits_viable", "viable", "uart", 3152, 163840, "True", 292616, 3145728, 40204, 327680, 243676, 213208, 4944, 4024184, "psram", 0.0, 170.0, 170.0, 1.0, 0.882, 0.630),
                ("esp32cam", "ds_cnn", "q15", "fits_exceeds_latency", "slow", "uart", 27664, 1081344, "True", 317260, 3145728, 40204, 327680, 243676, 213208, 4944, 3078008, "psram", 0.0, 1991.2, 1991.2, 1.0, 0.878, 0.656),
            ]
            with fit_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(FIT_CSV_COLUMNS)
                for r in rows_data:
                    w.writerow(r)

            _write_synthetic_run(res_dir, "nano", "lr_dot", "q15", 76.8, 1.58, 222, 146)
            _write_synthetic_run(res_dir, "nano", "mlp_16_dot", "q15", 77.0, 24.8, 230, 154)
            _write_synthetic_run(res_dir, "uno", "lr_dot", "q15", 76.8, 1.57, 222, 146)
            _write_synthetic_run(res_dir, "uno", "mlp_16_dot", "q15", 77.1, 24.8, 230, 154)
            _write_synthetic_run(res_dir, "esp32cam", "lr_dot", "q15", 3.18, 0.012, 240764, 6908, 4192124, 0, "none")
            _write_synthetic_run(res_dir, "esp32cam", "ultra_tiny", "q15", 0.0, 170.0, 243676, 4944, 4024184, 163840, "psram")
            _write_synthetic_run(res_dir, "esp32cam", "ds_cnn", "q15", 0.0, 1991.2, 243676, 4944, 3078008, 1081344, "psram")

            cfg_arg = ["--config", "configs/visualization/hw_test.toml"]
            self.assertEqual(main([*cfg_arg, "--dry-run"], project_root=tmp_root), 0)
            fig_dir = res_dir / "figures"
            self.assertFalse(fig_dir.exists())

            self.assertEqual(main(cfg_arg, project_root=tmp_root), 0)
            expected = [
                fig_dir / "fig1_cross_board_all_mcus.png",
                fig_dir / "fig2_board_esp32cam.png",
                fig_dir / "fig3_board_nano.png",
                fig_dir / "fig4_board_uno.png",
                fig_dir / "fig5_board_nodemcu.png",
            ]
            for p in expected:
                self.assertTrue(p.is_file(), f"Missing generated figure: {p}")
                self.assertGreater(p.stat().st_size, 10_000)

            mtime_before = expected[0].stat().st_mtime_ns
            # Without --overwrite, should skip silently
            self.assertEqual(main(cfg_arg, project_root=tmp_root), 0)
            self.assertEqual(expected[0].stat().st_mtime_ns, mtime_before)

            # With --overwrite, should regenerate cleanly
            self.assertEqual(main([*cfg_arg, "--overwrite"], project_root=tmp_root), 0)
            for p in expected:
                self.assertTrue(p.is_file())


if __name__ == "__main__":
    unittest.main()
