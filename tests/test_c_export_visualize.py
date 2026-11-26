"""Unit and integration tests for C99 export visualization data aggregation, plots, and CLI.

Manual execution (PowerShell from 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_c_export_visualize.py -v
"""

from __future__ import annotations

import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.visualize_c_pipeline import main as visualize_cli_main
from src.c_export.config import CImpConfig, load_c_imp_config
from src.c_export.visualize_data import load_target_bundle
from src.c_export.visualize_plots import generate_c_pipeline_visualizations


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _clone_config_with_tmp_paths(base_cfg: CImpConfig, tmp_root: Path) -> CImpConfig:
    return replace(
        base_cfg,
        paths=replace(
            base_cfg.paths,
            generated_root=tmp_root / "generated",
            build_root=tmp_root / "build",
            results_root=tmp_root / "results",
        ),
        classifier=replace(base_cfg.classifier, models=("lr_dot",)),
        cnn=replace(base_cfg.cnn, models=("ultra_tiny",)),
    )


def _write_minimal_golden_fixtures(
    cfg: CImpConfig,
    single_class: bool = False,
) -> None:
    gen_dir = cfg.generated_dir
    gen_dir.mkdir(parents=True, exist_ok=True)

    y_true = [0, 0, 0, 0, 0, 0] if single_class else [0, 0, 0, 0, 1, 1]

    # Tabular lr_dot: threshold 0.0, exact agreement across precisions
    lr_f32_scores = [-1.4, -0.8, -0.3, 0.2, 1.1, 1.9]
    lr_f32_lbls = [1 if s > 0.0 else 0 for s in lr_f32_scores]
    lr_q15_scores = [-1.39, -0.81, -0.29, 0.21, 1.09, 1.91]
    lr_q15_lbls = [1 if s > 0.0 else 0 for s in lr_q15_scores]
    lr_q15_raw = [int(round(s * 32768)) for s in lr_q15_scores]

    _write_csv(
        gen_dir / "golden_lr_dot.csv",
        [
            "file_name",
            "true_label",
            "py_f32_label",
            "py_f32_logit",
            "py_q15_label",
            "py_q15_logit_f64",
            "py_q15_logit_raw",
        ],
        [
            [
                f"img_{i}.png",
                y_true[i],
                lr_f32_lbls[i],
                lr_f32_scores[i],
                lr_q15_lbls[i],
                lr_q15_scores[i],
                lr_q15_raw[i],
            ]
            for i in range(6)
        ],
    )

    # CNN ultra_tiny: explicit header threshold 0.75 and 1 deliberate Q15 flip near threshold
    (gen_dir / "model_ultra_tiny.h").write_text(
        "#pragma once\n#define CIMP_ULTRA_TINY_THRESHOLD_F32 (0.750000f)\n",
        encoding="utf-8",
    )
    cnn_f32_scores = [0.10, 0.25, 0.74, 0.40, 1.35, 1.85]
    cnn_f32_lbls = [0, 0, 0, 0, 1, 1]
    cnn_q15_scores = [0.11, 0.24, 0.76, 0.39, 1.34, 1.84]
    cnn_q15_lbls = [0, 0, 1, 0, 1, 1]  # sample 2 flips across 0.75 threshold
    cnn_q15_raw = [int(round(s * 32768)) for s in cnn_q15_scores]

    _write_csv(
        gen_dir / "golden_cnn_ultra_tiny.csv",
        [
            "file_name",
            "true_label",
            "py_f32_label",
            "py_f32_margin",
            "py_q15_label",
            "py_q15_margin_f64",
            "py_q15_margin_raw",
        ],
        [
            [
                f"img_{i}.png",
                y_true[i],
                cnn_f32_lbls[i],
                cnn_f32_scores[i],
                cnn_q15_lbls[i],
                cnn_q15_scores[i],
                cnn_q15_raw[i],
            ]
            for i in range(6)
        ],
    )


def _x94_feature_names() -> list[str]:
    names: list[str] = ["E_1x1_r0_c0"]
    for level in (2, 4, 8):
        for r in range(level):
            for c in range(level):
                names.append(f"E_{level}x{level}_r{r}_c{c}")
    for scale in ("2x2", "4x4", "8x8"):
        for kind in ("lr", "tb", "diag"):
            names.append(f"asym_{scale}_{kind}")
    return names


def _write_enriched_verify_fixtures(cfg: CImpConfig) -> None:
    gen_dir = cfg.generated_dir
    res_dir = cfg.results_dir
    res_dir.mkdir(parents=True, exist_ok=True)

    (gen_dir / "export_manifest.json").write_text(
        json.dumps({"tabular_models": {"lr_dot": {"weight_bytes": 416}}}),
        encoding="utf-8",
    )

    (res_dir / "memory_report.json").write_text(
        json.dumps(
            {
                "budgets": {
                    "flash_budget_bytes": cfg.cnn.flash_budget_bytes,
                    "sram_budget_bytes": cfg.cnn.sram_budget_bytes,
                    "psram_budget_bytes": cfg.cnn.psram_budget_bytes,
                },
                "models": [
                    {
                        "model_name": "ultra_tiny",
                        "weight_format": "q15",
                        "weight_flash_bytes": 3200,
                        "peak_arena_f32_bytes": 262144,
                        "peak_arena_q15_bytes": 131072,
                        "fits_internal_sram_q15": True,
                        "fits_psram_q15": True,
                        "fits_flash": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    (res_dir / "parity_report.json").write_text(
        json.dumps(
            {
                "compiler_version": "MSVC 19.x synthetic",
                "all_gates_passed": True,
                "models": [
                    {
                        "name": "descriptor_x94",
                        "n_samples": 6,
                        "f32_max_abs_diff": 2.5e-7,
                        "q15_vs_py_q15_max_raw_diff": 0,
                        "passed": True,
                    },
                    {
                        "name": "lr_dot",
                        "n_samples": 6,
                        "f32_max_abs_diff": 1.2e-6,
                        "f32_label_mismatches": 0,
                        "q15_vs_py_q15_max_raw_diff": 0,
                        "q15_vs_py_q15_label_mismatches": 0,
                        "passed": True,
                    },
                    {
                        "name": "ultra_tiny",
                        "n_samples": 6,
                        "f32_max_abs_diff": 8.5e-6,
                        "f32_label_mismatches": 0,
                        "q15_vs_py_q15_max_raw_diff": 0,
                        "q15_vs_py_q15_label_mismatches": 0,
                        "passed": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    _write_csv(
        res_dir / "parity_per_sample.csv",
        ["model", "sample_idx", "f32_abs_diff"],
        [
            *[("lr_dot", i, 1.0e-7 * (i + 1)) for i in range(6)],
            *[("ultra_tiny", i, 1.0e-6 * (i + 1)) for i in range(6)],
        ],
    )

    _write_csv(
        res_dir / "timing_benchmark.csv",
        ["model", "arith", "selector_ms_per_img", "model_ms_per_img", "total_ms_per_img"],
        [
            ["x94_only", "f32", "0.025", "0.000", "0.025"],
            ["x94_only", "q15", "0.015", "0.000", "0.015"],
            ["lr_dot", "f32", "0.025", "0.003", "0.028"],
            ["lr_dot", "q15", "0.015", "0.002", "0.017"],
            ["ultra_tiny", "f32", "0.000", "2.100", "2.100"],
            ["ultra_tiny", "q15", "0.000", "1.050", "1.050"],
        ],
    )

    _write_csv(
        res_dir / "c_outputs" / "lr_dot_f32.csv",
        ["index", "c_label", "c_logit"],
        [[i, 1 if s > 0 else 0, s] for i, s in enumerate([-1.4, -0.8, -0.3, 0.2, 1.1, 1.9])],
    )
    _write_csv(
        res_dir / "c_outputs" / "lr_dot_q15.csv",
        ["index", "c_label", "c_logit"],
        [[i, 1 if s > 0 else 0, s] for i, s in enumerate([-1.39, -0.81, -0.29, 0.21, 1.09, 1.91])],
    )
    _write_csv(
        res_dir / "c_outputs" / "cnn_ultra_tiny_f32.csv",
        ["index", "c_label", "c_margin"],
        [[i, l, s] for i, (l, s) in enumerate(zip([0, 0, 0, 0, 1, 1], [0.10, 0.25, 0.74, 0.40, 1.35, 1.85]))],
    )
    _write_csv(
        res_dir / "c_outputs" / "cnn_ultra_tiny_q15.csv",
        ["index", "c_label", "c_margin"],
        [[i, l, s] for i, (l, s) in enumerate(zip([0, 0, 1, 0, 1, 1], [0.11, 0.24, 0.76, 0.39, 1.34, 1.84]))],
    )

    feat_names = _x94_feature_names()
    assert len(feat_names) == 94
    rng = np.random.default_rng(7)
    labels = [0, 0, 0, 1, 1, 1]
    base_neg = rng.uniform(0.15, 0.35, size=(3, 94))
    base_pos = rng.uniform(0.40, 0.70, size=(3, 94))
    mat_f64 = np.vstack([base_neg, base_pos])
    scale = float(1 << cfg.numeric.feature_frac_bits)
    noise = rng.integers(-3, 4, size=mat_f64.shape)
    mat_q15 = np.round(mat_f64 * scale).astype(np.int64) + noise

    vec_header = ["index", "label", *feat_names]
    _write_csv(
        res_dir / "vectors" / "golden_x94_f64.csv",
        vec_header,
        [[i, labels[i], *mat_f64[i].tolist()] for i in range(6)],
    )
    _write_csv(
        res_dir / "vectors" / "golden_x94_q15.csv",
        vec_header,
        [[i, labels[i], *mat_q15[i].tolist()] for i in range(6)],
    )


class TestCExportVisualize(unittest.TestCase):
    """Verify [c_imp.visualization] parsing, data aggregation, plot rendering, and CLI."""

    def test_visualization_config_validation(self) -> None:
        cfg = load_c_imp_config(Path("configs/c_imp/kolektorsdd2.toml"), PROJECT_ROOT)
        self.assertEqual(cfg.visualization.output_subdir, "figures")
        self.assertEqual(cfg.visualization.dpi, 300)
        self.assertEqual(cfg.visualization.top_k_worst_errors, 10)
        self.assertEqual(cfg.visualization_dir, cfg.results_dir / "figures")

        base_text = (PROJECT_ROOT / "configs/c_imp/kolektorsdd2.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)

            no_vis = tmp_dir / "no_vis.toml"
            no_vis.write_text(base_text.split("[c_imp.visualization]")[0], encoding="utf-8")
            with self.assertRaises(KeyError):
                load_c_imp_config(no_vis, PROJECT_ROOT)

            bad_dpi = tmp_dir / "bad_dpi.toml"
            bad_dpi.write_text(re.sub(r"dpi\s*=\s*\d+", "dpi = 0", base_text), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_c_imp_config(bad_dpi, PROJECT_ROOT)

            bad_topk = tmp_dir / "bad_topk.toml"
            bad_topk.write_text(
                base_text.replace("top_k_worst_errors = 10", "top_k_worst_errors = -2"),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_c_imp_config(bad_topk, PROJECT_ROOT)

            bad_sub = tmp_dir / "bad_sub.toml"
            bad_sub.write_text(
                base_text.replace('output_subdir = "figures"', 'output_subdir = "   "'),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_c_imp_config(bad_sub, PROJECT_ROOT)

    def test_minimal_export_only_visualization_and_single_class_nan_safety(self) -> None:
        base_cfg = load_c_imp_config(Path("configs/c_imp/kolektorsdd2.toml"), PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _clone_config_with_tmp_paths(base_cfg, Path(tmp))
            _write_minimal_golden_fixtures(cfg, single_class=False)

            bundle = load_target_bundle(cfg)
            self.assertEqual(len(bundle.models), 2)

            lr_m, cnn_m = bundle.models
            self.assertEqual(lr_m.name, "lr_dot")
            self.assertEqual(lr_m.family, "tabular")
            self.assertEqual((lr_m.stats_q15.tn, lr_m.stats_q15.fp, lr_m.stats_q15.fn, lr_m.stats_q15.tp), (3, 1, 0, 2))
            self.assertEqual(lr_m.q15_vs_f32_flips, 0)

            self.assertEqual(cnn_m.name, "ultra_tiny")
            self.assertEqual(cnn_m.family, "cnn")
            self.assertAlmostEqual(cnn_m.decision_threshold, 0.75, places=6)
            self.assertEqual(cnn_m.q15_vs_f32_flips, 1)
            self.assertEqual((cnn_m.stats_f32.tn, cnn_m.stats_f32.fp, cnn_m.stats_f32.fn, cnn_m.stats_f32.tp), (4, 0, 0, 2))
            self.assertEqual((cnn_m.stats_q15.tn, cnn_m.stats_q15.fp, cnn_m.stats_q15.fn, cnn_m.stats_q15.tp), (3, 1, 0, 2))

            report = generate_c_pipeline_visualizations(cfg)
            fig_names = [p.name for p in report.figure_paths]
            self.assertEqual(
                fig_names,
                [
                    "fig1_classification_f32_vs_q15.png",
                    "fig2_score_separation_and_quantization.png",
                    "fig3_confusion_matrices_q15.png",
                ],
            )
            for p in report.figure_paths:
                self.assertTrue(p.is_file())
                self.assertGreater(p.stat().st_size, 1000)
            self.assertTrue(report.summary_csv_path.is_file())
            self.assertTrue(report.summary_md_path.is_file())

            # Single-class slice must not raise and must populate NaN for undefined curves
            single_dir = Path(tmp) / "single"
            cfg_single = _clone_config_with_tmp_paths(base_cfg, single_dir)
            _write_minimal_golden_fixtures(cfg_single, single_class=True)
            bundle_single = load_target_bundle(cfg_single)
            self.assertTrue(math.isnan(bundle_single.models[0].stats_q15.roc_auc))
            self.assertTrue(math.isnan(bundle_single.models[0].stats_q15.pr_auc))
            self.assertTrue(math.isnan(bundle_single.models[0].stats_q15.balanced_accuracy))
            report_single = generate_c_pipeline_visualizations(cfg_single)
            self.assertEqual(len(report_single.figure_paths), 3)

    def test_enriched_post_verify_visualization_generates_all_six_figures(self) -> None:
        base_cfg = load_c_imp_config(Path("configs/c_imp/kolektorsdd2.toml"), PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _clone_config_with_tmp_paths(base_cfg, Path(tmp))
            _write_minimal_golden_fixtures(cfg, single_class=False)
            _write_enriched_verify_fixtures(cfg)

            report = generate_c_pipeline_visualizations(cfg)
            fig_names = [p.name for p in report.figure_paths]
            self.assertEqual(
                fig_names,
                [
                    "fig1_classification_f32_vs_q15.png",
                    "fig2_score_separation_and_quantization.png",
                    "fig3_confusion_matrices_q15.png",
                    "fig4_c_parity_and_verification.png",
                    "fig5_embedded_memory_and_latency_pareto.png",
                    "fig6_x94_descriptor_insights.png",
                ],
            )
            for p in report.figure_paths:
                self.assertTrue(p.is_file())
                self.assertGreater(p.stat().st_size, 1000)

            with report.summary_csv_path.open("r", newline="", encoding="utf-8") as f:
                rows = {row["model"]: row for row in csv.DictReader(f)}
            self.assertIn("lr_dot", rows)
            self.assertIn("ultra_tiny", rows)
            self.assertEqual(rows["ultra_tiny"]["c_q15_max_raw_diff"], "0")
            self.assertEqual(rows["ultra_tiny"]["c_parity_passed"], "True")
            self.assertEqual(rows["ultra_tiny"]["fits_internal_sram_q15"], "True")
            self.assertEqual(rows["ultra_tiny"]["bench_q15_total_ms"], "1.0500")
            self.assertEqual(rows["lr_dot"]["weight_flash_kb"], "0.41")

            md_text = report.summary_md_path.read_text(encoding="utf-8")
            self.assertIn("ultra_tiny", md_text)
            self.assertIn("fig6_x94_descriptor_insights.png", md_text)

    def test_tabular_weight_format_reports_q31_when_weight_bits_32(self) -> None:
        base_cfg = load_c_imp_config(Path("configs/c_imp/kolektorsdd2.toml"), PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _clone_config_with_tmp_paths(base_cfg, Path(tmp))
            _write_minimal_golden_fixtures(cfg, single_class=False)
            _write_enriched_verify_fixtures(cfg)

            (cfg.generated_dir / "export_manifest.json").write_text(
                json.dumps(
                    {
                        "tabular_models": {
                            "lr_dot": {"weight_bits": 32, "weight_bytes": 384},
                        }
                    }
                ),
                encoding="utf-8",
            )
            bundle = load_target_bundle(cfg)
            by_name = {m.name: m for m in bundle.models}
            self.assertEqual(by_name["lr_dot"].weight_format, "q31")

            report = generate_c_pipeline_visualizations(cfg)
            with report.summary_csv_path.open("r", newline="", encoding="utf-8") as f:
                rows = {row["model"]: row for row in csv.DictReader(f)}
            self.assertEqual(rows["lr_dot"]["weight_format"], "q31")
            self.assertEqual(rows["ultra_tiny"]["weight_format"], "q15")

    def test_cli_dry_run_real_run_and_missing_dir_error(self) -> None:
        base_text = (PROJECT_ROOT / "configs/c_imp/kolektorsdd2.toml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            cfg_path = tmp_root / "test_vis.toml"
            cfg_text = (
                base_text.replace('generated_dir = "c_imp/generated"', f'generated_dir = "{(tmp_root / "generated").as_posix()}"')
                .replace('build_dir = "c_imp/build"', f'build_dir = "{(tmp_root / "build").as_posix()}"')
                .replace('results_dir = "results/c_imp"', f'results_dir = "{(tmp_root / "results").as_posix()}"')
            )
            cfg_path.write_text(cfg_text, encoding="utf-8")

            # Missing generated_dir must exit cleanly with SystemExit
            with self.assertRaises(SystemExit):
                visualize_cli_main(["--config", str(cfg_path)], project_root=PROJECT_ROOT)

            loaded_cfg = load_c_imp_config(cfg_path, PROJECT_ROOT)
            _write_minimal_golden_fixtures(loaded_cfg, single_class=False)

            # --dry-run succeeds and writes no files
            rc_dry = visualize_cli_main(["--config", str(cfg_path), "--dry-run"], project_root=PROJECT_ROOT)
            self.assertEqual(rc_dry, 0)
            self.assertFalse(loaded_cfg.visualization_dir.exists())

            # Real CLI run renders minimal figures + summary tables
            rc_real = visualize_cli_main(["--config", str(cfg_path)], project_root=PROJECT_ROOT)
            self.assertEqual(rc_real, 0)
            self.assertTrue((loaded_cfg.visualization_dir / "fig1_classification_f32_vs_q15.png").is_file())
            self.assertTrue((loaded_cfg.visualization_dir / "summary_models_metrics.csv").is_file())
            self.assertTrue((loaded_cfg.visualization_dir / "summary_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
