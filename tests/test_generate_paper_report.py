"""Unit and CLI tests for `experiments.generate_paper_report`."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.generate_paper_report import main as report_cli_main


def _write_cls_artifacts(
    root: Path,
    ds_folder: str,
    prefix: str,
    cls_name: str,
    dataset_name: str,
    obj_name: str | None,
    inverted: bool,
    f1_x85: float,
    f1_x9: float,
    f1_x94: float,
    bal_acc: float,
    roc_auc: float,
    pr_auc: float,
    x9_share: float,
) -> None:
    out = root / "results" / "classification" / ds_folder / f"{prefix}{cls_name}"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_name": dataset_name,
        "object_name": obj_name,
        "inverted": inverted,
        "source_csv": f"data/csv/{ds_folder}-entropy-full.csv",
        "split_mode": "presplit" if obj_name is None else "kfold_or_presplit",
        "seed": 42,
        "classifier": cls_name,
        "train_counts": {"good": 100, "defect": 20},
        "test_counts": {"good": 50, "defect": 10},
        "timings": {
            "T_data_load_s": 0.01,
            "T_model_train_s": 0.02,
            "T_model_inference_s": 0.001,
            "inference_ms_per_sample": 0.02,
        },
        "active_subset_metrics": {
            "subset_name": "x94_full",
            "n_features": 94,
            "accuracy": 0.85,
            "balanced_accuracy": bal_acc,
            "precision": 0.70,
            "recall": 0.80,
            "f1": f1_x94,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": [[45, 5], [2, 8]],
        },
        "research_findings": {
            "symmetry_gain_f1_x94_minus_x85": round(f1_x94 - f1_x85, 6),
            "symmetry_gain_roc_auc_x94_minus_x85": 0.05,
            "best_single_entropy_scale_by_f1": "E_8x8",
            "explainability_method": "permutation_importance",
            "family_breakdown": [
                {"group_name": "x85_entropy", "n_features": 85, "share_pct": round(100.0 - x9_share, 4), "mean_importance_per_feature_pct": 0.5},
                {"group_name": "x9_asymmetry", "n_features": 9, "share_pct": round(x9_share, 4), "mean_importance_per_feature_pct": 2.5},
            ],
            "scale_breakdown": [],
            "asymmetry_axis_breakdown": [
                {"group_name": "asym_lr", "n_features": 3, "share_within_x9_pct": 40.0},
                {"group_name": "asym_tb", "n_features": 3, "share_within_x9_pct": 35.0},
                {"group_name": "asym_rot180", "n_features": 3, "share_within_x9_pct": 25.0},
            ],
            "top_features": [{"rank": 1, "feature_name": "asym_lr_8x8", "share_pct": 12.0}],
        },
    }
    (out / "metrics_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (out / "ablation_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subset_name", "n_features", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "train_elapsed_s", "inference_elapsed_s", "inference_ms_per_sample"])
        w.writerow(["x85_only", 85, 0.80, bal_acc - 0.04, 0.65, 0.70, f1_x85, roc_auc - 0.05, pr_auc - 0.04, 0.01, 0.001, 0.01])
        w.writerow(["x9_only", 9, 0.78, bal_acc - 0.06, 0.60, 0.68, f1_x9, roc_auc - 0.07, pr_auc - 0.05, 0.005, 0.0005, 0.008])
        w.writerow(["x94_full", 94, 0.85, bal_acc, 0.70, 0.80, f1_x94, roc_auc, pr_auc, 0.02, 0.001, 0.02])
    (out / "confusion_matrix_and_curves.png").write_bytes(b"png")


def _write_cnn_artifacts(
    root: Path,
    ds_folder: str,
    mid_folder: str,
    model_name: str,
    dataset_name: str,
    obj_name: str | None,
    inverted: bool,
    f1: float,
    bal_acc: float,
    roc_auc: float,
    pr_auc: float,
    params: int,
) -> None:
    out = root / "results" / "cnn" / ds_folder / mid_folder / model_name
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "object_name": obj_name,
        "inverted": inverted,
        "seed": 42,
        "hyperparameters": {"epochs": 40, "batch_size": 32, "learning_rate": 0.001, "weight_decay": 0.0001, "optimizer": "adamw", "scheduler": "cosine", "class_weighting": "balanced", "image_size": 128},
        "split_counts": {"train": {"good": 80, "defect": 16}, "val": {"good": 20, "defect": 4}, "test": {"good": 50, "defect": 10}},
        "model_profile": {"parameters": params, "trainable_parameters": params, "size_mb": 0.01, "flops": 1000000},
        "metrics": {"accuracy": 0.90, "balanced_accuracy": bal_acc, "precision": 0.80, "recall": 0.70, "f1": f1, "roc_auc": roc_auc, "pr_auc": pr_auc, "confusion_matrix": [[48, 2], [3, 7]]},
        "timing": {"train_elapsed_s": 2.0, "inference_elapsed_s": 0.02, "inference_ms_per_sample": 0.3, "best_epoch": 5, "best_val_pr_auc": 0.8},
        "benchmarks": [],
        "environment": {"python": "3.11.9", "platform": "Windows-10", "processor": "Intel", "torch": "2.14.0+cu126", "cuda_available": True, "gpu": "NVIDIA GeForce GTX 1050"},
    }
    (out / "metrics_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (out / "confusion_matrix_and_curves.png").write_bytes(b"png")


def _write_cpu_csv(root: Path, tag: str, rows: list[dict[str, object]]) -> None:
    out = root / "results" / "comparison_cpu" / f"{tag}_baseline_vs_cnn_cpu.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "family", "device", "threads", "batch_size", "params_model", "params_descriptor",
        "macs", "flops", "ops_basis", "t_preprocess_ms_median", "t_descriptor_ms_median",
        "t_model_ms_median", "t_total_ms_median", "t_total_ms_iqr", "t_total_ms_p95",
        "t_total_ms_mean", "t_total_ms_std", "t_total_ci95_low_ms", "t_total_ci95_high_ms",
        "inter_run_median_ms", "inter_run_std_ms", "inter_run_cv_pct", "throughput_samples_s",
        "peak_ram_mb", "accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc",
        "metric_parity_max_diff", "seed", "n_samples", "timestamp", "git_commit",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row_full = {k: "" for k in fields}
            row_full.update(r)
            w.writerow(row_full)


def _write_cimp_target(root: Path, target_tag: str) -> None:
    rdir = root / "results" / "c_imp" / target_tag
    fdir = rdir / "figures"
    fdir.mkdir(parents=True, exist_ok=True)
    (rdir / "parity_report.json").write_text(
        json.dumps(
            {
                "target_tag": target_tag,
                "runtime_tests_passed": True,
                "all_gates_passed": True,
                "desktop_exe": "cimp_pipeline.exe",
                "compiler_version": "MSVC 19.51",
                "models": [
                    {"name": "descriptor_x94", "mode": "f32/q15", "n_samples": 60, "f32_max_abs_diff": 2e-8, "f32_label_mismatches": 0, "q15_vs_py_q15_max_raw_diff": 0, "q15_vs_py_q15_label_mismatches": 0, "q15_vs_f32_label_mismatches": 0, "passed": True},
                    {"name": "lr_dot", "mode": "f32/q15", "n_samples": 60, "f32_max_abs_diff": 1e-5, "f32_label_mismatches": 0, "q15_vs_py_q15_max_raw_diff": 0, "q15_vs_py_q15_label_mismatches": 0, "q15_vs_f32_label_mismatches": 0, "passed": True},
                    {"name": "ultra_tiny", "mode": "f32/q15", "n_samples": 60, "f32_max_abs_diff": 1e-6, "f32_label_mismatches": 0, "q15_vs_py_q15_max_raw_diff": 0, "q15_vs_py_q15_label_mismatches": 0, "q15_vs_f32_label_mismatches": 0, "passed": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    (rdir / "memory_report.json").write_text(
        json.dumps(
            {
                "target_tag": target_tag,
                "budgets": {"flash_budget_bytes": 2097152, "sram_budget_bytes": 200000, "psram_budget_bytes": 3000000},
                "models": [
                    {"model_name": "ultra_tiny", "weight_format": "q15", "weight_flash_bytes": 3152, "peak_arena_f32_bytes": 327680, "peak_arena_q15_bytes": 163840, "fits_internal_sram_q15": True, "fits_psram_q15": True, "fits_flash": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    with (rdir / "timing_benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "model", "arith", "n_images", "reps", "selector_ms_per_img", "model_ms_per_img", "total_ms_per_img", "sink"])
        w.writerow(["descriptor", "lr_dot", "f32", 60, 300, 0.018, 0.0002, 0.0182, 1])
        w.writerow(["descriptor", "lr_dot", "q15", 60, 300, 0.020, 0.0002, 0.0202, 1])
        w.writerow(["cnn", "ultra_tiny", "f32", 60, 300, 0.0, 0.90, 0.90, 1])
        w.writerow(["cnn", "ultra_tiny", "q15", 60, 300, 0.0, 1.40, 1.40, 1])
    with (fdir / "summary_models_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "family", "n_samples", "n_positives", "weight_format", "weight_flash_kb", "arena_q15_kb", "fits_internal_sram_q15", "c_f32_max_abs_diff", "c_q15_max_raw_diff", "c_parity_passed", "bench_f32_total_ms", "bench_q15_total_ms"])
        w.writerow(["lr_dot", "tabular", 60, 10, "q31", "0.38", "", "True", "1.0e-05", "0", "True", "0.0182", "0.0202"])
        w.writerow(["ultra_tiny", "cnn", 60, 10, "q15", "3.08", "160.00", "True", "1.0e-06", "0", "True", "0.9000", "1.4000"])
    (fdir / "fig5_embedded_memory_and_latency_pareto.png").write_bytes(b"png")


def _build_synthetic_workspace(root: Path) -> Path:
    (root / "configs" / "reporting").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "base.toml").write_text(
        """
[configs]
path = "configs/base.toml"

[datasets.mvtecad]
name = "mvtec-ad"
subdatasets = ["bottle", "tile"]

[datasets.kolektorsdd2]
name = "KolektorSDD2"
""",
        encoding="utf-8",
    )
    cfg_rel = Path("configs/reporting/paper_report.toml")
    (root / cfg_rel).write_text(
        """
[paper_report]
base_config = "configs/base.toml"
classification_dir = "results/classification"
cnn_dir = "results/cnn"
comparison_cpu_dir = "results/comparison_cpu"
profiling_dir = "results/profiling"
c_imp_dir = "results/c_imp"
methodology_dir = "results/methodology"
hw_test_fit_csv = "results/hw_test/fit_report.csv"
output_markdown = "results/FINAL_PAPER_REPORT.md"
seed = 42
primary_cpu_threads = 1
primary_cpu_batch_size = 1
classifiers = ["lr_dot", "rf"]
cnn_models = ["ultra_tiny", "tiny_resnet"]
mvtec_object_categories = ["bottle"]
mvtec_texture_categories = ["tile"]

[[paper_report.flagship_figures]]
path = "results/c_imp/kolektorsdd2/figures/fig5_embedded_memory_and_latency_pareto.png"
script = "experiments.visualize_c_pipeline"
meaning = "Flash/SRAM/PSRAM budgets and C99 Q15 Pareto."
caption = "Embedded memory feasibility and Q15 latency Pareto on KolektorSDD2."

[[paper_report.figure_templates]]
pattern = "results/classification/<dataset>/<obj>_<cls>/confusion_matrix_and_curves.png"
script = "experiments.evaluate_descriptors"
meaning = "Per-run confusion matrix, ROC, and PR curves."
caption = "Confusion matrix and curves for <cls>."

[[paper_report.scripts_catalog]]
stage = "1. Classical ML"
script = "experiments.evaluate_descriptors"
configs = "configs/classification/kolektorsdd2.toml"
status = "Executed"
outputs = "results/classification/**/metrics_summary.json"
summary = "Evaluates classical classifiers on x94."

[[paper_report.section_traceability]]
section = "Section 3"
artifacts = "results/classification/**/metrics_summary.json"
scripts = "experiments.evaluate_descriptors"
""",
        encoding="utf-8",
    )

    # Environment + methodology + deprecated comparison dir (must be excluded from latency claims/inventory)
    prof_kol = root / "results" / "profiling" / "kolektorsdd2"
    prof_kol.mkdir(parents=True, exist_ok=True)
    (prof_kol / "environment.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-11-25T22:21:08+00:00",
                "git_commit": "abc1234",
                "python_version": "3.11.9",
                "os_platform": "Windows-10",
                "cpu_model": "Intel(R) Core(TM) i7-7700HQ",
                "ram_total_gb": 16.0,
                "torch_version": "2.14.0+cu126",
                "cuda_available": True,
                "gpu_name": "NVIDIA GeForce GTX 1050",
                "numpy_version": "2.2.6",
                "scipy_version": "1.16.2",
                "sklearn_version": "1.7.2",
                "pandas_version": "2.3.3",
            }
        ),
        encoding="utf-8",
    )
    (prof_kol / "fig1_x94_stage_breakdown.png").write_bytes(b"png")
    (root / "results" / "methodology").mkdir(parents=True, exist_ok=True)
    (root / "results" / "methodology" / "kolektorsdd2_x94_methodology.png").write_bytes(b"png")
    (root / "results" / "hw_test").mkdir(parents=True, exist_ok=True)
    (root / "results" / "hw_test" / "fit_report.csv").write_text(
        "board,model,mode,verdict,reason,measured_transport,pred_weight_flash_bytes [predicted],pred_arena_bytes [predicted],compile_ok [measured],flash_used_bytes [measured],flash_max_bytes [measured],ram_used_bytes [measured],ram_max_bytes [measured],free_ram_boot [measured],free_heap_min [measured],stack_hwm [measured],free_psram_min [measured],arena_location [measured],median_selector_ms [measured],median_model_ms [measured],median_total_ms [measured],parity_rate [measured],offline_roc_auc [offline],offline_f1 [offline]\n"
        "nano,lr_dot,q15,not_measured,no device measurement yet,,384,0,True,12780,30720,1636,2048,,,,,,,,,,0.806,0.425\n"
        "esp32cam,lr_dot,q15,fits_viable,ok,uart,384,0,True,289344,3145728,43116,327680,240764,210296,6908,4192124,none,3.1780,0.0120,3.1900,1.0000,0.8063,0.4253\n",
        encoding="utf-8",
    )
    (root / "results" / "comparison").mkdir(parents=True, exist_ok=True)
    (root / "results" / "comparison" / "DEPRECATED_DO_NOT_CITE.png").write_bytes(b"png")

    # Populate KolektorSDD2 + bottle + tile (std + inv)
    for inv, kol_ds, mv_ds in ((False, "kolektorsdd2", "mvtec-ad"), (True, "kolektorsdd2-inverted", "mvtec-ad-inverted")):
        _write_cls_artifacts(root, kol_ds, "full_", "lr_dot", "KolektorSDD2", None, inv, 0.30, 0.35, 0.45 if not inv else 0.44, 0.74, 0.80, 0.45, 25.0)
        _write_cls_artifacts(root, kol_ds, "full_", "rf", "KolektorSDD2", None, inv, 0.40, 0.38, 0.43, 0.68, 0.79, 0.42, 47.0)
        _write_cnn_artifacts(root, kol_ds, "full", "ultra_tiny", "KolektorSDD2", None, inv, 0.63 if not inv else 0.52, 0.75, 0.88, 0.70, 1154)
        _write_cnn_artifacts(root, kol_ds, "full", "tiny_resnet", "KolektorSDD2", None, inv, 0.71 if not inv else 0.65, 0.85, 0.92, 0.78, 77762)
        tag_k = "kolektorsdd2" if not inv else "kolektorsdd2-inverted"
        _write_cpu_csv(
            root,
            tag_k,
            [
                {"model": "lr_dot", "family": "ml", "device": "cpu", "threads": 1, "batch_size": 1, "params_model": 95, "t_preprocess_ms_median": 0.0, "t_descriptor_ms_median": 0.35, "t_model_ms_median": 0.01, "t_total_ms_median": 0.36, "f1": 0.45 if not inv else 0.44, "balanced_accuracy": 0.74, "roc_auc": 0.80, "pr_auc": 0.45},
                {"model": "rf", "family": "ml", "device": "cpu", "threads": 1, "batch_size": 1, "params_model": 2000, "t_preprocess_ms_median": 0.0, "t_descriptor_ms_median": 0.35, "t_model_ms_median": 5.0, "t_total_ms_median": 5.35, "f1": 0.43, "balanced_accuracy": 0.68, "roc_auc": 0.79, "pr_auc": 0.42},
                {"model": "ultra_tiny", "family": "cnn", "device": "cpu", "threads": 1, "batch_size": 1, "params_model": 1154, "t_preprocess_ms_median": 0.05, "t_descriptor_ms_median": 0.0, "t_model_ms_median": 0.90, "t_total_ms_median": 0.95, "f1": 0.63 if not inv else 0.52, "balanced_accuracy": 0.75, "roc_auc": 0.88, "pr_auc": 0.70},
                {"model": "tiny_resnet", "family": "cnn", "device": "cpu", "threads": 1, "batch_size": 1, "params_model": 77762, "t_preprocess_ms_median": 0.05, "t_descriptor_ms_median": 0.0, "t_model_ms_median": 25.0, "t_total_ms_median": 25.05, "f1": 0.71 if not inv else 0.65, "balanced_accuracy": 0.85, "roc_auc": 0.92, "pr_auc": 0.78},
            ],
        )
        for obj, f1_lr_std, f1_ut_std, f1_ut_inv, sh in (("bottle", 0.80, 0.36, 0.41, 19.0), ("tile", 0.86, 0.67, 0.39, 18.5)):
            _write_cls_artifacts(root, mv_ds, f"{obj}_", "lr_dot", "mvtec-ad", obj, inv, f1_lr_std - 0.10, 0.70, f1_lr_std, 0.90, 0.93, 0.90, sh)
            _write_cls_artifacts(root, mv_ds, f"{obj}_", "rf", "mvtec-ad", obj, inv, 0.60, 0.55, 0.75, 0.80, 0.88, 0.85, 25.0)
            _write_cnn_artifacts(root, mv_ds, obj, "ultra_tiny", "mvtec-ad", obj, inv, f1_ut_std if not inv else f1_ut_inv, 0.65, 0.70, 0.55, 1154)
            _write_cnn_artifacts(root, mv_ds, obj, "tiny_resnet", "mvtec-ad", obj, inv, 0.55 if not inv else 0.30, 0.60, 0.72, 0.58, 77762)
            _write_cpu_csv(
                root,
                f"{mv_ds}_{obj}",
                [
                    {"model": "lr_dot", "family": "ml", "device": "cpu", "threads": 1, "batch_size": 1, "params_model": 95, "t_preprocess_ms_median": 0.0, "t_descriptor_ms_median": 0.30, "t_model_ms_median": 0.01, "t_total_ms_median": 0.31, "f1": f1_lr_std, "balanced_accuracy": 0.90, "roc_auc": 0.93, "pr_auc": 0.90},
                    {"model": "rf", "family": "ml", "device": "cpu", "threads": 1, "batch_size": 1, "params_model": 2000, "t_preprocess_ms_median": 0.0, "t_descriptor_ms_median": 0.30, "t_model_ms_median": 4.0, "t_total_ms_median": 4.30, "f1": 0.75, "balanced_accuracy": 0.80, "roc_auc": 0.88, "pr_auc": 0.85},
                    {"model": "ultra_tiny", "family": "cnn", "device": "cpu", "threads": 1, "batch_size": 1, "params_model": 1154, "t_preprocess_ms_median": 0.05, "t_descriptor_ms_median": 0.0, "t_model_ms_median": 0.85, "t_total_ms_median": 0.90, "f1": f1_ut_std if not inv else f1_ut_inv, "balanced_accuracy": 0.65, "roc_auc": 0.70, "pr_auc": 0.55},
                    {"model": "tiny_resnet", "family": "cnn", "device": "cpu", "threads": 1, "batch_size": 1, "params_model": 77762, "t_preprocess_ms_median": 0.05, "t_descriptor_ms_median": 0.0, "t_model_ms_median": 24.0, "t_total_ms_median": 24.05, "f1": 0.55 if not inv else 0.30, "balanced_accuracy": 0.60, "roc_auc": 0.72, "pr_auc": 0.58},
                ],
            )

    _write_cimp_target(root, "kolektorsdd2")
    _write_cimp_target(root, "mvtec_ad_bottle")
    _write_cimp_target(root, "mvtec_ad_tile")
    return cfg_rel


class TestGeneratePaperReport(unittest.TestCase):
    """Verify dry-run, overwrite protection, all 6 sections, and exclusion of deprecated artifacts."""

    def test_dry_run_render_and_overwrite_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_rel = _build_synthetic_workspace(root)
            out_md = root / "results" / "FINAL_PAPER_REPORT.md"

            rc_dry = report_cli_main(["--config", str(cfg_rel), "--dry-run"], project_root=root)
            self.assertEqual(rc_dry, 0)
            self.assertFalse(out_md.exists())

            rc_run = report_cli_main(["--config", str(cfg_rel)], project_root=root)
            self.assertEqual(rc_run, 0)
            self.assertTrue(out_md.is_file())
            text = out_md.read_text(encoding="utf-8")

            for required in (
                "## 1. Execution Audit & Environment",
                "reproduce_paper_pipeline.ps1",
                "## 2. Image Transformations (Section 0), Dataset Splits, Polarity Invariance & Hyperparameters",
                "### 2.0. Section 0 — Image Transformations, Standardization & Tabular `x94` Pipeline",
                "## 3. KolektorSDD2 Complete Results",
                "### 3B. KolektorSDD2 Confusion Matrix Counts",
                "## 4. MVTec-AD All-15-Objects Results",
                "### 4C. Complete Per-Category F1 Matrix",
                "## 5. Deep CPU Micro-Profiling, Scalability & Statistical Significance",
                "## 6. C99 f32 / Q15 Parity, Latency & Memory Summary",
                "### 6C. Target Microcontroller Memory & Feasibility Matrix",
                "### 6D. Live ESP32-CAM Hardware Execution & Parity",
                "3.190",
                "## 7. Generated Figures Inventory, Meaning & Suggested Captions",
                "### 7A. Flagship Paper Figures — Meaning, Source Script & Suggested Captions",
                "### 7B. Per-Run & Per-Category Figure Templates (Repeated Across Targets)",
                "## 8. Pipeline Scripts & Section-to-Artifact Traceability",
                "### 8B. Report Section-to-Artifact Traceability Matrix",
                "experiments.evaluate_descriptors",
                "Embedded memory feasibility and Q15 latency Pareto on KolektorSDD2.",
                "object",
                "texture",
                "Object Macro-Avg",
                "Texture Macro-Avg",
                "lr_dot",
                "ultra_tiny",
                "40% / 35% / 25%",
                "fig5_embedded_memory_and_latency_pareto.png",
            ):
                self.assertIn(required, text)
            self.assertNotIn("DEPRECATED_DO_NOT_CITE.png", text)

            mtime_before = out_md.stat().st_mtime_ns
            rc_skip = report_cli_main(["--config", str(cfg_rel)], project_root=root)
            self.assertEqual(rc_skip, 0)
            self.assertEqual(out_md.stat().st_mtime_ns, mtime_before)

            rc_ov = report_cli_main(["--config", str(cfg_rel), "--overwrite"], project_root=root)
            self.assertEqual(rc_ov, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
