"""CSV/JSON exporters, matplotlib figures, and data-driven markdown report generation."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .classifier_profile import ClassifierProfileRecord
from .cnn_layers import CnnLayerProfileReport, CnnThreadScalingRecord
from .descriptor_stages import DescriptorProfileReport, DescriptorScalabilityRecord
from .environment import EnvironmentSnapshot
from .timing import StatisticalComparisonResult


def _fmt_float(val: float | None, precision: int = 6) -> str:
    """Format float deterministically for CSV export."""
    if val is None:
        return ""
    return f"{float(val):.{precision}f}"


def compute_pareto_frontier_indices(
    rows: Sequence[Mapping[str, Any]],
    cost_key: str = "t_total_ms_median",
    quality_key: str = "f1",
) -> set[int]:
    """Identify non-dominated points maximizing `quality_key` while minimizing `cost_key`."""
    pareto_indices: set[int] = set()
    for i, ri in enumerate(rows):
        ci = float(ri[cost_key])
        qi = float(ri[quality_key])
        dominated = False
        for j, rj in enumerate(rows):
            if i == j:
                continue
            cj = float(rj[cost_key])
            qj = float(rj[quality_key])
            if cj <= ci and qj >= qi and (cj < ci or qj > qi):
                dominated = True
                break
        if not dominated:
            pareto_indices.add(i)
    return pareto_indices


CPU_COMPARISON_FIELDNAMES: tuple[str, ...] = (
    "model",
    "family",
    "device",
    "threads",
    "batch_size",
    "params_model",
    "params_descriptor",
    "macs",
    "flops",
    "ops_basis",
    "t_preprocess_ms_median",
    "t_descriptor_ms_median",
    "t_model_ms_median",
    "t_total_ms_median",
    "t_total_ms_iqr",
    "t_total_ms_p95",
    "t_total_ms_mean",
    "t_total_ms_std",
    "t_total_ci95_low_ms",
    "t_total_ci95_high_ms",
    "inter_run_median_ms",
    "inter_run_std_ms",
    "inter_run_cv_pct",
    "throughput_samples_s",
    "peak_ram_mb",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "roc_auc",
    "pr_auc",
    "metric_parity_max_diff",
    "seed",
    "n_samples",
    "timestamp",
    "git_commit",
)


def write_comparison_csv(
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    precision: int = 6,
) -> Path:
    """Write CPU-fair or CUDA-reference comparison rows with deterministic column ordering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(CPU_COMPARISON_FIELDNAMES), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            formatted: dict[str, Any] = {}
            for k in CPU_COMPARISON_FIELDNAMES:
                val = row.get(k, "")
                if isinstance(val, float):
                    formatted[k] = _fmt_float(val, precision=precision)
                elif val is None:
                    formatted[k] = ""
                else:
                    formatted[k] = val
            writer.writerow(formatted)
    return output_path


SUMMARY_ALL_FIELDNAMES: tuple[str, ...] = (
    "dataset_tag",
    "object_name",
    "inverted",
    "model",
    "family",
    "device",
    "threads",
    "batch_size",
    "params_model",
    "params_descriptor",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "roc_auc",
    "pr_auc",
    "t_total_ms_median",
    "t_total_ms_iqr",
    "throughput_samples_s",
    "timestamp",
)


def upsert_summary_all_datasets(
    summary_csv_path: Path,
    dataset_tag: str,
    object_name: str | None,
    inverted: bool,
    cpu_rows: Sequence[Mapping[str, Any]],
    precision: int = 6,
) -> Path:
    """Update `summary_all_datasets.csv` cleanly without duplicating existing dataset/model keys."""
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    if summary_csv_path.is_file():
        with summary_csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    str(row.get("dataset_tag", "")),
                    str(row.get("model", "")),
                    str(row.get("threads", "")),
                    str(row.get("batch_size", "")),
                )
                existing[key] = dict(row)

    obj_label = object_name if object_name else "full"
    for row in cpu_rows:
        if str(row.get("device", "")).lower() != "cpu":
            continue
        key = (
            dataset_tag,
            str(row.get("model", "")),
            str(row.get("threads", "")),
            str(row.get("batch_size", "")),
        )
        existing[key] = {
            "dataset_tag": dataset_tag,
            "object_name": obj_label,
            "inverted": str(inverted).lower(),
            "model": row["model"],
            "family": row["family"],
            "device": row["device"],
            "threads": row["threads"],
            "batch_size": row["batch_size"],
            "params_model": row["params_model"],
            "params_descriptor": row["params_descriptor"],
            "accuracy": _fmt_float(row.get("accuracy"), precision),
            "balanced_accuracy": _fmt_float(row.get("balanced_accuracy"), precision),
            "f1": _fmt_float(row.get("f1"), precision),
            "roc_auc": _fmt_float(row.get("roc_auc"), precision),
            "pr_auc": _fmt_float(row.get("pr_auc"), precision),
            "t_total_ms_median": _fmt_float(row.get("t_total_ms_median"), precision),
            "t_total_ms_iqr": _fmt_float(row.get("t_total_ms_iqr"), precision),
            "throughput_samples_s": _fmt_float(row.get("throughput_samples_s"), 2),
            "timestamp": row.get("timestamp", ""),
        }

    ordered_keys = sorted(
        existing.keys(),
        key=lambda k: (k[0], int(k[2]) if k[2].isdigit() else 0, int(k[3]) if k[3].isdigit() else 0, k[1]),
    )
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SUMMARY_ALL_FIELDNAMES))
        writer.writeheader()
        for k in ordered_keys:
            writer.writerow(existing[k])
    return summary_csv_path


def write_descriptor_stage_artifacts(
    output_dir: Path,
    report: DescriptorProfileReport,
    precision: int = 6,
) -> tuple[Path, Path]:
    """Write `descriptor_stage_times.csv` and `cprofile_descriptor.txt`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "descriptor_stage_times.csv"
    fieldnames = [
        "dataset_tag",
        "batch_size",
        "threads",
        "stage",
        "category",
        "scale",
        "median_ms",
        "q25_ms",
        "q75_ms",
        "iqr_ms",
        "p95_ms",
        "mean_ms",
        "std_ms",
        "ci95_low_ms",
        "ci95_high_ms",
        "share_of_staged_sum_pct",
        "tracemalloc_peak_kb",
        "rss_delta_mb",
        "analytical_ops_per_sample",
        "expected_complexity",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in (*report.stages_batch1, *report.stages_batch_n):
            d = asdict(rec)
            row = {
                "dataset_tag": report.dataset_tag,
                "batch_size": rec.batch_size,
                "threads": report.threads,
                **d,
            }
            for k in (
                "median_ms",
                "q25_ms",
                "q75_ms",
                "iqr_ms",
                "p95_ms",
                "mean_ms",
                "std_ms",
                "ci95_low_ms",
                "ci95_high_ms",
                "share_of_staged_sum_pct",
                "tracemalloc_peak_kb",
                "rss_delta_mb",
            ):
                row[k] = _fmt_float(row[k], precision=precision)
            writer.writerow(row)

    cprof_path = output_dir / "cprofile_descriptor.txt"
    cprof_path.write_text(report.cprofile_report_text, encoding="utf-8")
    return csv_path, cprof_path


def write_descriptor_scalability_csv(
    output_path: Path,
    records: Sequence[DescriptorScalabilityRecord],
    precision: int = 6,
) -> Path:
    """Write descriptor parameter sweep CSV (`image_size`, `bins`, `grid_levels`)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sweep_dimension",
        "config_label",
        "image_size",
        "bins",
        "grid_levels",
        "n_features",
        "threads",
        "init_plan_ms",
        "median_ms",
        "iqr_ms",
        "p95_ms",
        "ci95_low_ms",
        "ci95_high_ms",
        "throughput_samples_s",
        "analytical_ops_per_sample",
        "expected_complexity_note",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = asdict(rec)
            for k in ("init_plan_ms", "median_ms", "iqr_ms", "p95_ms", "ci95_low_ms", "ci95_high_ms"):
                row[k] = _fmt_float(row[k], precision=precision)
            row["throughput_samples_s"] = _fmt_float(row["throughput_samples_s"], precision=2)
            writer.writerow(row)
    return output_path


def write_classifier_times_csv(
    output_path: Path,
    records: Sequence[ClassifierProfileRecord],
    precision: int = 6,
) -> Path:
    """Write classical classifier latency, breakdown, params, and pickle size CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "classifier",
        "threads",
        "batch_size",
        "n_features",
        "n_train",
        "params_model",
        "scaler_params",
        "pickle_bytes",
        "analytical_macs",
        "analytical_flops",
        "predict_median_ms",
        "predict_iqr_ms",
        "predict_p95_ms",
        "predict_ci95_low_ms",
        "predict_ci95_high_ms",
        "predict_throughput_s",
        "predict_proba_median_ms",
        "predict_proba_iqr_ms",
        "scaler_median_ms",
        "estimator_only_median_ms",
        "direct_core_median_ms",
        "sklearn_dispatch_overhead_ms",
        "params_note",
        "ops_basis",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "classifier": rec.classifier_name,
                    "threads": rec.threads,
                    "batch_size": rec.batch_size,
                    "n_features": rec.n_features,
                    "n_train": rec.n_train,
                    "params_model": rec.params_info.params_model,
                    "scaler_params": rec.params_info.scaler_params,
                    "pickle_bytes": rec.params_info.pickle_bytes,
                    "analytical_macs": rec.ops_estimate.analytical_macs,
                    "analytical_flops": rec.ops_estimate.analytical_flops,
                    "predict_median_ms": _fmt_float(rec.predict_stats.median_ms, precision),
                    "predict_iqr_ms": _fmt_float(rec.predict_stats.iqr_ms, precision),
                    "predict_p95_ms": _fmt_float(rec.predict_stats.p95_ms, precision),
                    "predict_ci95_low_ms": _fmt_float(rec.predict_stats.ci95_low_ms, precision),
                    "predict_ci95_high_ms": _fmt_float(rec.predict_stats.ci95_high_ms, precision),
                    "predict_throughput_s": _fmt_float(rec.predict_stats.throughput_samples_s, 2),
                    "predict_proba_median_ms": _fmt_float(rec.predict_proba_stats.median_ms, precision),
                    "predict_proba_iqr_ms": _fmt_float(rec.predict_proba_stats.iqr_ms, precision),
                    "scaler_median_ms": _fmt_float(
                        rec.scaler_stats.median_ms if rec.scaler_stats is not None else None, precision
                    ),
                    "estimator_only_median_ms": _fmt_float(
                        rec.estimator_only_stats.median_ms if rec.estimator_only_stats is not None else None,
                        precision,
                    ),
                    "direct_core_median_ms": _fmt_float(
                        rec.direct_core_stats.median_ms if rec.direct_core_stats is not None else None,
                        precision,
                    ),
                    "sklearn_dispatch_overhead_ms": _fmt_float(
                        rec.sklearn_dispatch_overhead_ms, precision
                    ),
                    "params_note": rec.params_info.detail_note,
                    "ops_basis": rec.ops_estimate.estimate_basis,
                }
            )
    return output_path


def write_cnn_layer_artifacts(
    output_dir: Path,
    reports: Sequence[CnnLayerProfileReport],
    precision: int = 6,
) -> tuple[Path, list[Path]]:
    """Write per-model CNN layer/ATen CSVs and an aggregate CNN architecture summary CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    summary_rows: list[dict[str, Any]] = []
    for rep in reports:
        layer_csv = output_dir / f"cnn_layer_times_{rep.model_name}.csv"
        l_fields = [
            "model_name",
            "layer_name",
            "layer_type",
            "category",
            "call_count",
            "input_shape",
            "output_shape",
            "params",
            "param_mb_fp32",
            "activation_bytes",
            "activation_kb",
            "flops",
            "median_ms",
            "q25_ms",
            "q75_ms",
            "iqr_ms",
            "p95_ms",
            "share_of_layers_pct",
            "ms_per_mflop",
        ]
        with layer_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=l_fields)
            writer.writeheader()
            for lay in rep.layers:
                row = asdict(lay)
                for k in (
                    "param_mb_fp32",
                    "activation_kb",
                    "median_ms",
                    "q25_ms",
                    "q75_ms",
                    "iqr_ms",
                    "p95_ms",
                    "share_of_layers_pct",
                    "ms_per_mflop",
                ):
                    row[k] = _fmt_float(row[k], precision)
                writer.writerow(row)
        written_files.append(layer_csv)

        aten_csv = output_dir / f"cnn_aten_ops_{rep.model_name}.csv"
        a_fields = [
            "model_name",
            "op_name",
            "call_count",
            "cpu_time_total_ms",
            "self_cpu_time_ms",
            "share_cpu_time_pct",
            "share_self_cpu_time_pct",
            "cpu_memory_kb",
            "self_cpu_memory_kb",
            "flops",
        ]
        with aten_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=a_fields)
            writer.writeheader()
            for op in rep.aten_ops:
                row = asdict(op)
                for k in (
                    "cpu_time_total_ms",
                    "self_cpu_time_ms",
                    "share_cpu_time_pct",
                    "share_self_cpu_time_pct",
                    "cpu_memory_kb",
                    "self_cpu_memory_kb",
                ):
                    row[k] = _fmt_float(row[k], precision)
                writer.writerow(row)
        written_files.append(aten_csv)

        aten_txt = output_dir / f"cnn_aten_table_{rep.model_name}.txt"
        aten_txt.write_text(rep.aten_table_text, encoding="utf-8")
        written_files.append(aten_txt)

        summary_rows.append(
            {
                "model_name": rep.model_name,
                "threads": rep.threads,
                "batch_size": rep.batch_size,
                "total_params": rep.total_params,
                "model_size_mb": _fmt_float(rep.model_size_mb, 4),
                "max_activation_kb": _fmt_float(rep.max_activation_kb, 2),
                "total_flops": rep.total_flops,
                "hookless_median_ms": _fmt_float(rep.hookless_stats.median_ms, precision),
                "hookless_iqr_ms": _fmt_float(rep.hookless_stats.iqr_ms, precision),
                "hookless_p95_ms": _fmt_float(rep.hookless_stats.p95_ms, precision),
                "hooked_total_median_ms": _fmt_float(rep.hooked_total_median_ms, precision),
                "sum_layers_median_ms": _fmt_float(rep.sum_layers_median_ms, precision),
                "hook_overhead_pct": _fmt_float(rep.hook_overhead_pct, 2),
                "checkpoint_load_ms": _fmt_float(
                    rep.cold_start.checkpoint_load_ms if rep.cold_start is not None else None,
                    precision,
                ),
                "first_inference_ms": _fmt_float(
                    rep.cold_start.first_inference_ms if rep.cold_start is not None else None,
                    precision,
                ),
                "cold_to_steady_ratio": _fmt_float(
                    rep.cold_start.cold_to_steady_ratio if rep.cold_start is not None else None,
                    2,
                ),
                **{
                    f"share_{cat}_pct": _fmt_float(rep.category_shares_pct.get(cat, 0.0), 2)
                    for cat in ("conv", "dwconv", "pwconv", "bn", "activation", "pool", "linear", "other")
                },
            }
        )

    summary_csv = output_dir / "cnn_architectures_summary.csv"
    if summary_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    return summary_csv, written_files


def write_cnn_thread_scaling_csv(
    output_path: Path,
    records: Sequence[CnnThreadScalingRecord],
    precision: int = 6,
) -> Path:
    """Write CNN multi-thread scaling CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "threads",
        "batch_size",
        "median_ms",
        "iqr_ms",
        "p95_ms",
        "ci95_low_ms",
        "ci95_high_ms",
        "throughput_samples_s",
        "speedup_vs_1thread",
        "efficiency_pct",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = asdict(rec)
            for k in ("median_ms", "iqr_ms", "p95_ms", "ci95_low_ms", "ci95_high_ms", "speedup_vs_1thread"):
                row[k] = _fmt_float(row[k], precision)
            row["throughput_samples_s"] = _fmt_float(row["throughput_samples_s"], 2)
            row["efficiency_pct"] = _fmt_float(row["efficiency_pct"], 2)
            writer.writerow(row)
    return output_path


def write_statistical_tests_csv(
    output_path: Path,
    comparisons: Sequence[StatisticalComparisonResult],
    precision: int = 6,
) -> Path:
    """Write Mann-Whitney U hypothesis test comparisons CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group_a",
        "group_b",
        "n_a",
        "n_b",
        "median_a_ms",
        "median_b_ms",
        "median_diff_ms",
        "ratio_a_over_b",
        "u_statistic",
        "p_value",
        "alpha",
        "significant",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for comp in comparisons:
            row = asdict(comp)
            for k in ("median_a_ms", "median_b_ms", "median_diff_ms", "ratio_a_over_b", "u_statistic"):
                row[k] = _fmt_float(row[k], precision)
            row["p_value"] = f"{comp.p_value:.4e}"
            writer.writerow(row)
    return output_path


def write_ablation_cost_csv(
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    precision: int = 6,
) -> Path:
    """Write x85 vs x9 vs x94 quality-vs-latency ablation CSV (`dF1 / dms`)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "classifier",
        "subset_name",
        "n_features",
        "f1",
        "roc_auc",
        "pr_auc",
        "t_descriptor_ms",
        "t_model_ms",
        "t_total_ms",
        "delta_f1_vs_x85",
        "delta_ms_vs_x85",
        "marginal_f1_per_ms",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out_r = dict(r)
            for k in ("f1", "roc_auc", "pr_auc", "t_descriptor_ms", "t_model_ms", "t_total_ms", "delta_f1_vs_x85", "delta_ms_vs_x85", "marginal_f1_per_ms"):
                if k in out_r and isinstance(out_r[k], float):
                    out_r[k] = _fmt_float(out_r[k], precision)
            writer.writerow(out_r)
    return output_path


def plot_cpu_comparison_pareto(
    output_path: Path,
    rows: Sequence[Mapping[str, Any]],
    dataset_label: str,
    threads: int,
    primary_batch_size: int = 1,
    throughput_batch_size: int = 32,
    dpi: int = 160,
) -> Path | None:
    """Render CPU-only comparison panels for a specific thread count without mixing GPU rows."""
    cpu_primary = [
        r
        for r in rows
        if str(r.get("device", "")).lower() == "cpu"
        and int(r.get("threads", 0)) == threads
        and int(r.get("batch_size", 0)) == primary_batch_size
    ]
    if not cpu_primary:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=dpi)
    fig.suptitle(
        f"100% CPU Fair Comparison: {dataset_label} (threads={threads}, latency @ batch={primary_batch_size})",
        fontsize=13,
        fontweight="bold",
    )

    ml_rows = [r for r in cpu_primary if r["family"] == "ml"]
    cnn_rows = [r for r in cpu_primary if r["family"] == "cnn"]
    pareto_idx = compute_pareto_frontier_indices(cpu_primary, cost_key="t_total_ms_median", quality_key="f1")

    # 1. F1 vs Total CPU Latency (log x + IQR bars + Pareto front)
    ax1 = axes[0, 0]
    for fam, subset, color, marker, label in (
        ("ml", ml_rows, "#1f77b4", "o", "Classical ML (x94)"),
        ("cnn", cnn_rows, "#d62728", "^", "Lightweight CNNs"),
    ):
        if not subset:
            continue
        x = np.array([max(1e-4, float(r["t_total_ms_median"])) for r in subset])
        y = np.array([float(r["f1"]) for r in subset])
        xerr = np.array([float(r["t_total_ms_iqr"]) * 0.5 for r in subset])
        ax1.errorbar(
            x,
            y,
            xerr=xerr,
            fmt=marker,
            color=color,
            ecolor=color,
            alpha=0.9,
            markersize=8,
            capsize=3,
            linestyle="none",
            label=label,
            zorder=3,
        )
        for r in subset:
            ax1.annotate(
                str(r["model"]),
                (max(1e-4, float(r["t_total_ms_median"])), float(r["f1"])),
                fontsize=8,
                xytext=(5, 4),
                textcoords="offset points",
            )

    if pareto_idx:
        p_pts = sorted(
            [cpu_primary[i] for i in pareto_idx],
            key=lambda r: float(r["t_total_ms_median"]),
        )
        ax1.plot(
            [max(1e-4, float(r["t_total_ms_median"])) for r in p_pts],
            [float(r["f1"]) for r in p_pts],
            color="#2ca02c",
            linestyle="--",
            linewidth=1.6,
            marker="*",
            markersize=10,
            label="Pareto Frontier",
            zorder=4,
        )
    ax1.set_xscale("log")
    ax1.set_xlabel("Total CPU Latency ms/sample (log scale, +-0.5 IQR)")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("F1 vs Total CPU Latency")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)

    # 2. ROC-AUC vs Total CPU Latency (log x)
    ax2 = axes[0, 1]
    for subset, color, marker, label in (
        (ml_rows, "#1f77b4", "o", "Classical ML (x94)"),
        (cnn_rows, "#d62728", "^", "Lightweight CNNs"),
    ):
        if not subset:
            continue
        ax2.scatter(
            [max(1e-4, float(r["t_total_ms_median"])) for r in subset],
            [float(r["roc_auc"]) for r in subset],
            color=color,
            marker=marker,
            s=75,
            label=label,
            zorder=3,
        )
        for r in subset:
            ax2.annotate(
                str(r["model"]),
                (max(1e-4, float(r["t_total_ms_median"])), float(r["roc_auc"])),
                fontsize=8,
                xytext=(5, 4),
                textcoords="offset points",
            )
    ax2.set_xscale("log")
    ax2.set_xlabel("Total CPU Latency ms/sample (log scale)")
    ax2.set_ylabel("ROC-AUC")
    ax2.set_title("ROC-AUC vs Total CPU Latency")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)

    # 3. F1 vs Model Parameters (log x)
    ax3 = axes[1, 0]
    for subset, color, marker, label in (
        (ml_rows, "#1f77b4", "o", "Classical ML (params_model)"),
        (cnn_rows, "#d62728", "^", "Lightweight CNNs"),
    ):
        if not subset:
            continue
        ax3.scatter(
            [max(1, int(r["params_model"])) for r in subset],
            [float(r["f1"]) for r in subset],
            color=color,
            marker=marker,
            s=75,
            label=label,
            zorder=3,
        )
        for r in subset:
            ax3.annotate(
                str(r["model"]),
                (max(1, int(r["params_model"])), float(r["f1"])),
                fontsize=8,
                xytext=(5, 4),
                textcoords="offset points",
            )
    ax3.set_xscale("log")
    ax3.set_xlabel("Model Parameters (log scale)")
    ax3.set_ylabel("F1 Score")
    ax3.set_title("F1 vs Real Model Parameter Count")
    ax3.legend(fontsize=8)
    ax3.grid(True, which="both", linestyle=":", alpha=0.5)

    # 4. Throughput comparison (batch=1 vs batch=32 when available)
    ax4 = axes[1, 1]
    cpu_tp = [
        r
        for r in rows
        if str(r.get("device", "")).lower() == "cpu"
        and int(r.get("threads", 0)) == threads
        and int(r.get("batch_size", 0)) == throughput_batch_size
    ]
    tp_source = cpu_tp if cpu_tp else cpu_primary
    tp_sorted = sorted(tp_source, key=lambda r: float(r["throughput_samples_s"]), reverse=False)
    labels_tp = [str(r["model"]) for r in tp_sorted]
    vals_tp = [float(r["throughput_samples_s"]) for r in tp_sorted]
    cols_tp = ["#1f77b4" if r["family"] == "ml" else "#d62728" for r in tp_sorted]
    ax4.barh(labels_tp, vals_tp, color=cols_tp)
    ax4.set_xscale("log")
    ax4.set_xlabel(f"Throughput samples/s (log scale, batch={throughput_batch_size if cpu_tp else primary_batch_size})")
    ax4.set_title("CPU Throughput Ranking")
    ax4.grid(True, axis="x", which="both", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_profiling_figures(
    output_dir: Path,
    dataset_label: str,
    desc_report: DescriptorProfileReport,
    desc_scalability: Sequence[DescriptorScalabilityRecord],
    layer_reports: Sequence[CnnLayerProfileReport],
    cnn_scaling: Sequence[CnnThreadScalingRecord],
    cls_scaling: Sequence[ClassifierProfileRecord],
    cpu_comparison_rows: Sequence[Mapping[str, Any]],
    primary_threads: int = 1,
    primary_batch_size: int = 1,
    throughput_batch_size: int = 32,
    dpi: int = 160,
) -> list[Path]:
    """Generate the 6 structured publication figures for Phase B deep profiling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # Figure 1: Stacked horizontal bars for x94 descriptor stages
    fig1_path = output_dir / "fig1_x94_stage_breakdown.png"
    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5.5), dpi=dpi)
    fig1.suptitle(
        f"x94 Descriptor Stage Breakdown: {dataset_label} (CPU threads={desc_report.threads})",
        fontsize=12,
        fontweight="bold",
    )

    core_stages = [s for s in desc_report.stages_batch1 if s.stage != "standard_scaler"]
    stage_names = [s.stage for s in core_stages]
    b1_vals = [s.median_ms for s in core_stages]
    b1_pcts = [s.share_of_staged_sum_pct for s in core_stages]
    bn_map = {s.stage: s.median_ms for s in desc_report.stages_batch_n}
    bn_vals = [bn_map.get(s, 0.0) for s in stage_names]

    y_pos = np.arange(len(stage_names))
    axes1[0].barh(y_pos + 0.18, b1_vals[::-1], height=0.35, color="#1f77b4", label="batch=1")
    axes1[0].barh(
        y_pos - 0.18,
        bn_vals[::-1],
        height=0.35,
        color="#ff7f0e",
        label=f"batch={desc_report.batch_n_size} (amortized/img)",
    )
    axes1[0].set_yticks(y_pos)
    axes1[0].set_yticklabels(stage_names[::-1], fontsize=8)
    axes1[0].set_xlabel("Median Time per Sample (ms)")
    axes1[0].set_title(
        f"Stage Latency vs Batch Size (Gap={desc_report.python_call_gap_ms:+.3f} ms)"
    )
    axes1[0].legend(fontsize=8)
    axes1[0].grid(True, axis="x", linestyle=":", alpha=0.5)

    # Stacked percentage summary grouped by macro-family
    macro_groups = {
        "Quantization LUT": sum(s.share_of_staged_sum_pct for s in core_stages if s.stage == "quantize_pixels"),
        "Histograms (g=1,2,4,8)": sum(s.share_of_staged_sum_pct for s in core_stages if s.stage.startswith("region_histograms")),
        "Entropy LUT (g=1,2,4,8)": sum(s.share_of_staged_sum_pct for s in core_stages if s.stage.startswith("normalized_entropy")),
        "Asymmetry x9 (g=2,4,8)": sum(s.share_of_staged_sum_pct for s in core_stages if s.stage.startswith("asymmetry_")),
        "Concat / Assembly": sum(s.share_of_staged_sum_pct for s in core_stages if s.stage.startswith("concat_")),
    }
    colors = ["#4c72b0", "#dd8452", "#55a868", "#8172b3", "#937860"]
    left = 0.0
    for (m_name, pct), col in zip(macro_groups.items(), colors):
        axes1[1].barh([0], [pct], left=left, color=col, edgecolor="white", label=f"{m_name}: {pct:.1f}%")
        if pct >= 5.0:
            axes1[1].text(left + pct / 2.0, 0, f"{pct:.1f}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        left += pct
    axes1[1].set_xlim(0, max(100.0, left))
    axes1[1].set_yticks([0])
    axes1[1].set_yticklabels([f"Total={desc_report.staged_sum_batch1_ms:.3f} ms"])
    axes1[1].set_xlabel("Share of Staged Sum (%)")
    axes1[1].set_title(
        f"Stage Cost Composition (Init Plan={desc_report.init_entropy_plan_ms:.3f} ms once)"
    )
    axes1[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    axes1[1].grid(True, axis="x", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig1.savefig(fig1_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig1)
    generated.append(fig1_path)

    # Figure 2: CNN Layer Category Heatmap
    if layer_reports:
        fig2_path = output_dir / "fig2_cnn_layer_heatmap.png"
        cats = ["conv", "dwconv", "pwconv", "bn", "activation", "pool", "linear"]
        mat = np.zeros((len(layer_reports), len(cats)), dtype=np.float64)
        for idx_r, rep in enumerate(layer_reports):
            for idx_c, cat in enumerate(cats):
                mat[idx_r, idx_c] = rep.category_shares_pct.get(cat, 0.0)

        fig2, axes2 = plt.subplots(1, 2, figsize=(15, max(5.0, 0.6 * len(layer_reports) + 2.5)), dpi=dpi)
        fig2.suptitle(
            f"CNN Layer-Type CPU Time Heatmap & Total Latency: {dataset_label} (threads={primary_threads}, batch={primary_batch_size})",
            fontsize=12,
            fontweight="bold",
        )
        im = axes2[0].imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=100.0)
        axes2[0].set_xticks(np.arange(len(cats)))
        axes2[0].set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
        axes2[0].set_yticks(np.arange(len(layer_reports)))
        axes2[0].set_yticklabels([r.model_name for r in layer_reports], fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                axes2[0].text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if val > 55.0 else "black",
                    fontsize=8,
                )
        axes2[0].set_title("Share of Leaf Hook CPU Time (%) by Operator Family")
        plt.colorbar(im, ax=axes2[0], fraction=0.046, pad=0.04)

        sorted_reps = sorted(layer_reports, key=lambda r: r.hookless_stats.median_ms)
        axes2[1].barh(
            [r.model_name for r in sorted_reps],
            [r.hookless_stats.median_ms for r in sorted_reps],
            color="#d62728",
        )
        for idx_s, r in enumerate(sorted_reps):
            axes2[1].text(
                r.hookless_stats.median_ms,
                idx_s,
                f" {r.hookless_stats.median_ms:.3f} ms ({r.total_flops/1e6:.1f} MFLOPs)",
                va="center",
                fontsize=8,
            )
        axes2[1].set_xlabel("Hookless CPU Latency median (ms/sample)")
        axes2[1].set_title("CNN Forward Latency & MFLOPs")
        axes2[1].grid(True, axis="x", linestyle=":", alpha=0.5)

        plt.tight_layout()
        fig2.savefig(fig2_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig2)
        generated.append(fig2_path)

    # Figure 3: Pareto CPU (threads=primary_threads, batch=1)
    fig3_path = output_dir / "fig3_pareto_cpu_1t_b1.png"
    res3 = plot_cpu_comparison_pareto(
        output_path=fig3_path,
        rows=cpu_comparison_rows,
        dataset_label=dataset_label,
        threads=primary_threads,
        primary_batch_size=primary_batch_size,
        throughput_batch_size=throughput_batch_size,
        dpi=dpi,
    )
    if res3 is not None:
        generated.append(res3)

    # Figure 4: Thread scaling curves (CNNs + ML classifiers)
    if cnn_scaling or cls_scaling:
        fig4_path = output_dir / "fig4_thread_scaling.png"
        fig4, axes4 = plt.subplots(1, 2, figsize=(15, 5.5), dpi=dpi)
        fig4.suptitle(
            f"CPU Thread Scaling Sensitivity: {dataset_label} (batch={primary_batch_size})",
            fontsize=12,
            fontweight="bold",
        )

        # Left: absolute latency vs threads
        by_cnn: dict[str, list[CnnThreadScalingRecord]] = {}
        for rec in cnn_scaling:
            by_cnn.setdefault(rec.model_name, []).append(rec)
        for m_name, recs in sorted(by_cnn.items()):
            recs_s = sorted(recs, key=lambda r: r.threads)
            axes4[0].plot(
                [r.threads for r in recs_s],
                [r.median_ms for r in recs_s],
                marker="^",
                lw=1.8,
                label=f"CNN: {m_name}",
            )

        by_cls: dict[str, list[ClassifierProfileRecord]] = {}
        for rec in cls_scaling:
            by_cls.setdefault(rec.classifier_name, []).append(rec)
        for c_name, recs in sorted(by_cls.items()):
            recs_s = sorted(recs, key=lambda r: r.threads)
            axes4[0].plot(
                [r.threads for r in recs_s],
                [r.predict_stats.median_ms for r in recs_s],
                marker="o",
                linestyle="--",
                lw=1.8,
                label=f"ML predict: {c_name}",
            )

        axes4[0].set_yscale("log")
        axes4[0].set_xlabel("CPU Threads (PyTorch + threadpoolctl + n_jobs)")
        axes4[0].set_ylabel("Model Inference Median Latency ms (log)")
        axes4[0].set_title("Model Inference Latency vs Threads")
        axes4[0].legend(fontsize=8, ncol=2)
        axes4[0].grid(True, which="both", linestyle=":", alpha=0.5)

        # Right: Speedup vs threads for CNNs
        all_th = sorted({r.threads for r in cnn_scaling})
        if all_th:
            axes4[1].plot(all_th, all_th, color="black", linestyle=":", label="Ideal linear speedup")
            for m_name, recs in sorted(by_cnn.items()):
                recs_s = sorted(recs, key=lambda r: r.threads)
                axes4[1].plot(
                    [r.threads for r in recs_s],
                    [r.speedup_vs_1thread for r in recs_s],
                    marker="^",
                    lw=1.8,
                    label=m_name,
                )
        axes4[1].set_xlabel("CPU Threads")
        axes4[1].set_ylabel("Speedup vs 1 Thread")
        axes4[1].set_title("CNN Parallel Speedup Curve")
        axes4[1].legend(fontsize=8)
        axes4[1].grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        fig4.savefig(fig4_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig4)
        generated.append(fig4_path)

    # Figure 5: Descriptor scalability
    if desc_scalability:
        fig5_path = output_dir / "fig5_descriptor_scalability.png"
        dims = ("image_size", "bins", "grid_levels")
        fig5, axes5 = plt.subplots(1, 3, figsize=(16, 4.8), dpi=dpi)
        fig5.suptitle(
            f"x94 Descriptor Scalability (CPU threads={desc_report.threads}, batch=1)",
            fontsize=12,
            fontweight="bold",
        )
        for ax, dim_name in zip(axes5, dims):
            sub = [r for r in desc_scalability if r.sweep_dimension == dim_name]
            if not sub:
                ax.set_visible(False)
                continue
            labels = [r.config_label for r in sub]
            meds = [r.median_ms for r in sub]
            iqrs = [r.iqr_ms * 0.5 for r in sub]
            x_idx = np.arange(len(labels))
            ax.errorbar(x_idx, meds, yerr=iqrs, marker="o", capsize=4, lw=2, color="#1f77b4")
            for i_pt, val_ms in enumerate(meds):
                ax.annotate(f"{val_ms:.3f} ms", (x_idx[i_pt], val_ms), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
            ax.set_xticks(x_idx)
            ax.set_xticklabels(labels, fontsize=8, rotation=15)
            ax.set_ylabel("Median Time per Image (ms)")
            ax.set_title(f"Sweep: {dim_name}")
            ax.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        fig5.savefig(fig5_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig5)
        generated.append(fig5_path)

    # Figure 6: Throughput batch=1 vs batch=32 on CPU 1-thread
    cpu_b1 = {
        str(r["model"]): r
        for r in cpu_comparison_rows
        if str(r.get("device", "")).lower() == "cpu"
        and int(r.get("threads", 0)) == primary_threads
        and int(r.get("batch_size", 0)) == primary_batch_size
    }
    cpu_b32 = {
        str(r["model"]): r
        for r in cpu_comparison_rows
        if str(r.get("device", "")).lower() == "cpu"
        and int(r.get("threads", 0)) == primary_threads
        and int(r.get("batch_size", 0)) == throughput_batch_size
    }
    common_models = sorted(
        cpu_b32.keys(),
        key=lambda m: float(cpu_b32[m]["throughput_samples_s"]),
    )
    if common_models:
        fig6_path = output_dir / "fig6_throughput_batch32.png"
        fig6, ax6 = plt.subplots(figsize=(12, max(5.0, 0.48 * len(common_models) + 1.5)), dpi=dpi)
        y_i = np.arange(len(common_models))
        tp32 = [float(cpu_b32[m]["throughput_samples_s"]) for m in common_models]
        tp1 = [float(cpu_b1[m]["throughput_samples_s"]) if m in cpu_b1 else 0.0 for m in common_models]
        ax6.barh(y_i + 0.18, tp32, height=0.35, color="#2ca02c", label=f"batch={throughput_batch_size}")
        ax6.barh(y_i - 0.18, tp1, height=0.35, color="#1f77b4", label=f"batch={primary_batch_size}")
        ax6.set_xscale("log")
        ax6.set_yticks(y_i)
        ax6.set_yticklabels(common_models, fontsize=9)
        ax6.set_xlabel("CPU Throughput samples/s (log scale)")
        ax6.set_title(
            f"End-to-End CPU Throughput ({dataset_label}, threads={primary_threads}): batch={primary_batch_size} vs batch={throughput_batch_size}"
        )
        ax6.legend(fontsize=9)
        ax6.grid(True, axis="x", which="both", linestyle=":", alpha=0.5)
        plt.tight_layout()
        fig6.savefig(fig6_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig6)
        generated.append(fig6_path)

    return generated


def write_profiling_readme(
    output_path: Path,
    dataset_label: str,
    env: EnvironmentSnapshot,
    desc_report: DescriptorProfileReport,
    cpu_rows: Sequence[Mapping[str, Any]],
    cuda_rows: Sequence[Mapping[str, Any]],
    stat_tests: Sequence[StatisticalComparisonResult],
    ablation_rows: Sequence[Mapping[str, Any]],
    layer_reports: Sequence[CnnLayerProfileReport],
    primary_threads: int,
    primary_batch_size: int,
) -> Path:
    """Write a strictly data-driven profiling README separating measured findings from estimates."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    primary_rows = [
        r
        for r in cpu_rows
        if str(r.get("device", "")).lower() == "cpu"
        and int(r.get("threads", 0)) == primary_threads
        and int(r.get("batch_size", 0)) == primary_batch_size
    ]
    primary_sorted = sorted(primary_rows, key=lambda r: float(r["t_total_ms_median"]))

    top_stages = sorted(
        [s for s in desc_report.stages_batch1 if s.stage != "standard_scaler"],
        key=lambda s: s.median_ms,
        reverse=True,
    )[:5]

    lines: list[str] = [
        f"# Profiling Cientifico y Comparacion 100% CPU: `{dataset_label}`",
        "",
        "## 1. Objetivo y Protocolo Medido",
        "",
        f"- **Objetivo:** Comparar en igualdad estricta de dispositivo (**CPU**), hilos (`threads={primary_threads}` principal), tamano de lote (`batch_size={primary_batch_size}` principal para latencia embebida) y muestras reales el pipeline clasico **x94 + ML** frente a las arquitecturas **CNN ligeras**.",
        "- **Checkpoints CNN:** Se reutilizan los pesos ya entrenados en GPU (`checkpoint.pt`); la re-evaluacion en CPU verifica paridad numerica frente a las metricas originales sin reentrenar (la calidad no depende del dispositivo de entrenamiento).",
        f"- **Entorno registrado:** CPU `{env.cpu_model}` ({env.physical_cores} nucleos fisicos / {env.logical_cores} logicos), RAM {env.ram_total_gb:.1f} GB, SO `{env.os_platform}`, PyTorch `{env.torch_version}`, NumPy `{env.numpy_version}`, scikit-learn `{env.sklearn_version}`, seed `{env.seed}`, commit `{env.git_commit}`.",
        "- **Temporizacion y estadistica:** `time.perf_counter_ns()` con GC deshabilitado (`gc.disable()`), `torch.inference_mode()`, threads fijados en PyTorch + BLAS/OpenMP (`threadpoolctl`) + `n_jobs`, mediana, IQR, p95, IC95% por bootstrap y test no parametrico U de Mann-Whitney.",
        "",
        "## 2. Hallazgos Medidos: Comparacion 100% CPU (Principal: `threads=1, batch=1`)",
        "",
        "| Modelo | Familia | Device | Params (modelo / desc) | F1 | ROC-AUC | T_prep / T_desc (ms) | T_model (ms) | **T_total mediana (ms)** | IQR (ms) | IC95% (ms) | Throughput (img/s) | Paridad CPU diff |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in primary_sorted:
        tp_or_td = (
            f"{float(r['t_descriptor_ms_median']):.4f} (desc)"
            if r["family"] == "ml"
            else f"{float(r['t_preprocess_ms_median']):.4f} (prep)"
        )
        ci_str = f"[{float(r['t_total_ci95_low_ms']):.4f}, {float(r['t_total_ci95_high_ms']):.4f}]"
        par = r.get("metric_parity_max_diff")
        par_str = f"{float(par):.2e}" if isinstance(par, (float, int)) else "-"
        lines.append(
            f"| `{r['model']}` | {r['family']} | {r['device']} | {r['params_model']} / {r['params_descriptor']} | "
            f"{float(r['f1']):.4f} | {float(r['roc_auc']):.4f} | {tp_or_td} | "
            f"{float(r['t_model_ms_median']):.4f} | **{float(r['t_total_ms_median']):.4f}** | "
            f"{float(r['t_total_ms_iqr']):.4f} | {ci_str} | {float(r['throughput_samples_s']):.1f} | {par_str} |"
        )

    lines.extend(
        [
            "",
            "## 3. Desglose Medido por Etapas del Descriptor x94 (`batch=1` vs E2E)",
            "",
            f"- Inicializacion amortizable (`build_entropy_plan` + LUTs): **{desc_report.init_entropy_plan_ms:.4f} ms** (se computa una sola vez).",
            f"- Verificacion de equivalencia numerica (etapas vs `compute_entropy_features + compute_asymmetry_features`): max abs diff = **{desc_report.max_numerical_diff:.3e}**.",
            f"- Suma de medianas por etapa: **{desc_report.staged_sum_batch1_ms:.4f} ms** vs llamada formal E2E: **{desc_report.e2e_batch1_stats.median_ms:.4f} ms** (gap de dispatch/asignaciones Python = **{desc_report.python_call_gap_ms:+.4f} ms**, {desc_report.python_call_gap_pct:+.1f}%).",
            "",
            "| Etapa | Escala | Mediana batch=1 (ms) | % de suma | Peak tracemalloc (KB) | Complejidad esperada (analitica) |",
            "|---|---|---|---|---|---|",
        ]
    )
    for st in top_stages:
        lines.append(
            f"| `{st.stage}` | {st.scale} | {st.median_ms:.4f} | {st.share_of_staged_sum_pct:.1f}% | {st.tracemalloc_peak_kb:.1f} | `{st.expected_complexity}` |"
        )

    if layer_reports:
        lines.extend(
            [
                "",
                "## 4. Desglose Medido por Capas CNN en CPU (`threads=1, batch=1`)",
                "",
                "| Arquitectura | Params | Act Peak (KB) | FLOPs (PyTorch) | Hookless (ms) | Hook overhead (%) | Cold 1st (ms) | Familias dominantes (% hooks) |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for rep in layer_reports:
            dom = sorted(rep.category_shares_pct.items(), key=lambda kv: kv[1], reverse=True)[:3]
            dom_str = ", ".join(f"{k}:{v:.1f}%" for k, v in dom if v > 0.0)
            cold_str = f"{rep.cold_start.first_inference_ms:.3f}" if rep.cold_start is not None else "-"
            lines.append(
                f"| `{rep.model_name}` | {rep.total_params:,} | {rep.max_activation_kb:.1f} | {rep.total_flops:,} | "
                f"{rep.hookless_stats.median_ms:.4f} | {rep.hook_overhead_pct:+.1f}% | {cold_str} | {dom_str} |"
            )

    if stat_tests:
        lines.extend(
            [
                "",
                "## 5. Pruebas de Significancia Estadistica (Mann-Whitney U, 1 hilo, batch=1)",
                "",
                "| Comparacion | Mediana A (ms) | Mediana B (ms) | Ratio A/B | Estadistico U | p-valor | Significativo |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for st in stat_tests:
            lines.append(
                f"| `{st.group_a}` vs `{st.group_b}` | {st.median_a_ms:.4f} | {st.median_b_ms:.4f} | "
                f"{st.ratio_a_over_b:.3f}x | {st.u_statistic:.1f} | `{st.p_value:.3e}` | {st.significant} |"
            )

    if ablation_rows:
        lines.extend(
            [
                "",
                "## 6. Ablacion Calidad vs Coste (`x85` vs `x9` vs `x94`)",
                "",
                "| Clasificador | Subconjunto | Features | F1 | ROC-AUC | T_total (ms) | dF1 vs x85 | d_ms vs x85 | dF1 / d_ms |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for ab in ablation_rows:
            lines.append(
                f"| `{ab['classifier']}` | `{ab['subset_name']}` | {ab['n_features']} | {float(ab['f1']):.4f} | "
                f"{float(ab['roc_auc']):.4f} | {float(ab['t_total_ms']):.4f} | "
                f"{float(ab['delta_f1_vs_x85']):+.4f} | {float(ab['delta_ms_vs_x85']):+.4f} | "
                f"{float(ab['marginal_f1_per_ms']):+.4f} |"
            )

    if cuda_rows:
        lines.extend(
            [
                "",
                "## 7. Referencia Secundaria CUDA (Aislada, nunca mezclada en ranking CPU)",
                "",
                "| Modelo CNN | Device | Batch | T_model mediana (ms) | Throughput (img/s) |",
                "|---|---|---|---|---|",
            ]
        )
        for r in cuda_rows:
            lines.append(
                f"| `{r['model']}` | {r['device']} | {r['batch_size']} | "
                f"{float(r['t_model_ms_median']):.4f} | {float(r['throughput_samples_s']):.1f} |"
            )

    # Derived recommendations strictly anchored to measured dominant stage / Pareto points
    dominant_stage = top_stages[0] if top_stages else None
    best_f1_row = max(primary_rows, key=lambda r: float(r["f1"])) if primary_rows else None
    fastest_row = primary_sorted[0] if primary_sorted else None

    lines.extend(
        [
            "",
            "## 8. Separacion de Resultados Medidos vs Estimaciones Analiticas y Limitaciones",
            "",
            "- **Medido empiricamente:** Todas las latencias (`ms`), IQR, p95, IC95% bootstrap, pruebas Mann-Whitney U, tiempos por etapa/capa (`perf_counter_ns`, `torch.profiler`), consumo `psutil` RSS, asignaciones `tracemalloc`, metricas de clasificacion y `FlopCounterMode` FLOPs de PyTorch.",
            "- **Estimaciones analiticas (no equivalentes a MACs neuronales):** Los conteos de operaciones del descriptor x94 (indexacion LUT + incrementos enteros de histograma) y de clasificadores clasicos (comparaciones de arbol, producto punto LR, distancias kNN) se reportan con `ops_basis = analytical_estimate` para no mezclar ordenes de magnitud heterogeneos con `FlopCounterMode`.",
            "- **Limitaciones:** Una CPU x86-64 de escritorio no reproduce jerarquias de cache SRAM, ausencia de FPU/SIMD AVX2 ni consumo energetico de un microcontrolador embebido (Cortex-M / RISC-V). El runtime de Python/NumPy/scikit-learn anade overhead de despacho fijo en `batch=1` medible por separado.",
            "",
            "## 9. Recomendaciones Derivadas de los Datos",
            "",
        ]
    )
    if dominant_stage is not None:
        lines.append(
            f"- **Cuello de botella principal de x94:** `{dominant_stage.stage}` consume **{dominant_stage.share_of_staged_sum_pct:.1f}%** ({dominant_stage.median_ms:.4f} ms) del tiempo por etapas; cualquier optimizacion futura en C/Cython debe priorizar el bucle de histogramas multi-escala antes que cuantizacion o asimetria."
        )
    if fastest_row is not None and best_f1_row is not None:
        lines.append(
            f"- **Compromiso Pareto CPU (1 hilo, batch=1):** El pipeline mas rapido medido es `{fastest_row['model']}` (**{float(fastest_row['t_total_ms_median']):.4f} ms**, F1={float(fastest_row['f1']):.4f}), mientras el maximo F1 lo obtiene `{best_f1_row['model']}` (**F1={float(best_f1_row['f1']):.4f}**, {float(best_f1_row['t_total_ms_median']):.4f} ms)."
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
