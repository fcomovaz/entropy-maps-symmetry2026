"""Generate publication-quality comparative plots strictly from Stage 6 CSV/JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, LAB_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config_loader import EntropyOptimizationConfig, load_optimization_config, resolve_project_path
from configs.logging import configure_logging, logger

VARIANT_COLORS: dict[str, str] = {
    "V0_baseline": "#475569",
    "V1_numpy_finest_pyramid": "#2563eb",
    "V2_numpy_fused_entropy": "#0891b2",
    "V3_c_single_pass": "#16a34a",
    "V3_c_single_pass_into": "#15803d",
    "V3_ctypes_noop_overhead": "#64748b",
}

VARIANT_SHORT_LABELS: dict[str, str] = {
    "V0_baseline": "V0 Baseline\n(85 regions)",
    "V1_numpy_finest_pyramid": "V1 NumPy\nFinest Pyramid",
    "V2_numpy_fused_entropy": "V2 NumPy\nFused Entropy",
    "V3_c_single_pass": "V3 C DLL\nSingle Pass",
}


def _short_label(variant: str) -> str:
    return VARIANT_SHORT_LABELS.get(variant, variant)


def _variant_color(variant: str) -> str:
    return VARIANT_COLORS.get(variant, "#334155")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV records strictly using stdlib csv.DictReader, failing fast when absent."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Required benchmark CSV not found: '{path}'. Run benchmark_entropy_variants.py first."
        )
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_bool(raw: str) -> bool:
    return raw.strip().lower() in {"true", "1", "yes"}


def _save_fig(
    fig: plt.Figure,
    stem: str,
    output_dir: Path,
    dpi: int,
    formats: tuple[str, ...],
) -> list[Path]:
    """Save figure in all configured output formats and close matplotlib handle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        out_path = output_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        saved.append(out_path)
    plt.close(fig)
    return saved


def _plot_batch1_latency(
    latency_rows: list[dict[str, str]],
    manifest: dict[str, Any],
    output_dir: Path,
    dpi: int,
    formats: tuple[str, ...],
) -> list[Path]:
    """Figure 1: Batch=1 latency bar chart (log scale) with 95% CI and speedup vs V0."""
    primary = [
        r for r in latency_rows if int(r["batch_size"]) == 1 and not _as_bool(r["is_diagnostic"])
    ]
    if not primary:
        raise ValueError("No batch_size=1 primary variant rows found in benchmark_latency.csv")

    diag_map = {
        r["variant"]: r
        for r in latency_rows
        if int(r["batch_size"]) == 1 and _as_bool(r["is_diagnostic"])
    }

    names = [r["variant"] for r in primary]
    labels = [_short_label(n) for n in names]
    colors = [_variant_color(n) for n in names]
    medians = np.array([float(r["median_ms"]) for r in primary], dtype=np.float64)
    ci_lo = np.array([float(r["ci95_low_ms"]) for r in primary], dtype=np.float64)
    ci_hi = np.array([float(r["ci95_high_ms"]) for r in primary], dtype=np.float64)
    speedups = [float(r["speedup_vs_v0"]) for r in primary]
    sigs = [_as_bool(r["significant"]) for r in primary]

    yerr_low = np.maximum(medians - ci_lo, 0.0)
    yerr_high = np.maximum(ci_hi - medians, 0.0)

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=dpi)
    x = np.arange(len(names))
    bars = ax.bar(
        x,
        medians,
        width=0.58,
        color=colors,
        edgecolor="#0f172a",
        linewidth=0.8,
        yerr=[yerr_low, yerr_high],
        capsize=5,
        error_kw={"ecolor": "#0f172a", "linewidth": 1.1},
        zorder=3,
    )
    ax.set_yscale("log")
    y_min = max(1e-5, float(np.min(ci_lo)) * 0.45)
    y_max = float(np.max(ci_hi)) * 2.8
    ax.set_ylim(y_min, y_max)

    for idx, bar in enumerate(bars):
        top = max(float(ci_hi[idx]), float(medians[idx]))
        sp = speedups[idx]
        sig_mark = "*" if sigs[idx] else ""
        if names[idx] == manifest.get("reference_variant", "V0_baseline"):
            sp_text = "1.00x (ref)"
        else:
            sp_text = f"{sp:.2f}x{sig_mark}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            top * 1.16,
            f"{medians[idx]:.4f} ms\n({sp_text})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold" if idx == len(bars) - 1 else "normal",
        )

    subtitle_parts = [
        f"{manifest.get('image_size', '?')}x{manifest.get('image_size', '?')}",
        f"bins={manifest.get('bins', '?')}",
        f"threads={manifest.get('threads', 1)}",
        f"source={manifest.get('data_source', '?')}",
        "* p < alpha (Mann-Whitney U)",
    ]
    if "V3_ctypes_noop_overhead" in diag_map and "V3_c_single_pass_into" in diag_map:
        noop_ms = float(diag_map["V3_ctypes_noop_overhead"]["median_ms"])
        into_ms = float(diag_map["V3_c_single_pass_into"]["median_ms"])
        subtitle_parts.append(f"V3 run_into={into_ms:.4f} ms | ctypes noop={noop_ms * 1e3:.2f} us")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Median Per-Image Latency (ms, log scale)")
    ax.set_title(
        "Batch=1 CPU Entropy Descriptor Latency & Speedup (95% Bootstrap CI)\n"
        + " | ".join(subtitle_parts),
        fontsize=10,
        fontweight="bold",
    )
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.55, zorder=0)
    plt.tight_layout()
    return _save_fig(fig, "fig01_batch1_latency_speedup", output_dir, dpi, formats)


def _assign_staggered_y_offsets(
    log_y_values: list[float],
    min_gap_decades: float = 0.14,
    step_pts: float = 11.0,
) -> list[float]:
    """Compute vertical text offset in points so adjacent points in log-space do not overlap."""
    n = len(log_y_values)
    offsets = [0.0] * n
    order = sorted(range(n), key=lambda idx: log_y_values[idx])
    clusters: list[list[int]] = []
    for idx in order:
        if not clusters or (log_y_values[idx] - log_y_values[clusters[-1][-1]]) >= min_gap_decades:
            clusters.append([idx])
        else:
            clusters[-1].append(idx)

    for cluster in clusters:
        k = len(cluster)
        if k == 1:
            offsets[cluster[0]] = 5.0
        else:
            mid = (k - 1) / 2.0
            for pos, orig_idx in enumerate(cluster):
                offsets[orig_idx] = (pos - mid) * step_pts
    return offsets


def _plot_throughput_vs_batch(
    latency_rows: list[dict[str, str]],
    output_dir: Path,
    dpi: int,
    formats: tuple[str, ...],
) -> list[Path]:
    """Figure 2: Throughput (img/s) scaling curve vs batch size."""
    primary = [r for r in latency_rows if not _as_bool(r["is_diagnostic"])]
    batch_sizes = sorted({int(r["batch_size"]) for r in primary})
    variant_order: list[str] = []
    for r in primary:
        if r["variant"] not in variant_order:
            variant_order.append(r["variant"])

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=dpi)
    x_pos = np.arange(len(batch_sizes), dtype=np.float64)
    markers = ("s", "o", "^", "D", "v")
    all_y_vals: list[float] = []
    endpoints: list[tuple[str, float, float, float, float]] = []

    for v_idx, v_name in enumerate(variant_order):
        by_batch = {
            int(r["batch_size"]): float(r["throughput_samples_s"])
            for r in primary
            if r["variant"] == v_name
        }
        y_vals = [by_batch[b] for b in batch_sizes if b in by_batch]
        x_vals = [float(i) for i, b in enumerate(batch_sizes) if b in by_batch]
        all_y_vals.extend(y_vals)
        ax.plot(
            x_vals,
            y_vals,
            marker=markers[v_idx % len(markers)],
            linewidth=2.0,
            markersize=6.5,
            color=_variant_color(v_name),
            label=v_name,
        )
        if x_vals:
            endpoints.append((v_name, x_vals[0], y_vals[0], x_vals[-1], y_vals[-1]))

    if endpoints:
        start_offsets = _assign_staggered_y_offsets(
            [float(np.log10(ep[2])) for ep in endpoints], min_gap_decades=0.14, step_pts=12.0
        )
        end_offsets = _assign_staggered_y_offsets(
            [float(np.log10(ep[4])) for ep in endpoints], min_gap_decades=0.14, step_pts=12.0
        )
        for idx, (v_name, x0, y0, xn, yn) in enumerate(endpoints):
            col = _variant_color(v_name)
            ax.annotate(
                f"{y0:,.0f}",
                (x0, y0),
                textcoords="offset points",
                xytext=(7.0, start_offsets[idx]),
                va="center",
                ha="left",
                fontsize=7.8,
                color=col,
                fontweight="bold",
            )
            if xn > x0:
                ax.annotate(
                    f"{yn:,.0f}",
                    (xn, yn),
                    textcoords="offset points",
                    xytext=(-7.0, end_offsets[idx]),
                    va="center",
                    ha="right",
                    fontsize=7.8,
                    color=col,
                    fontweight="bold",
                )

    ax.set_yscale("log")
    if all_y_vals:
        ax.set_ylim(min(all_y_vals) * 0.65, max(all_y_vals) * 1.45)
    ax.set_xlim(-0.18, len(batch_sizes) - 1 + 0.18)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in batch_sizes], fontsize=9.5)
    ax.set_xlabel("Batch Size (images / call)")
    ax.set_ylabel("Throughput (images / second, log scale)")
    ax.set_title(
        "Batch Scaling Throughput by Implementation Variant (CPU Single-Thread)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.55)
    ax.legend(loc="best", fontsize=8.5)
    plt.tight_layout()
    return _save_fig(fig, "fig02_throughput_vs_batch", output_dir, dpi, formats)


def _plot_stage_breakdown(
    stage_rows: list[dict[str, str]],
    output_dir: Path,
    dpi: int,
    formats: tuple[str, ...],
) -> list[Path]:
    """Figure 3: Stage breakdown (quantize, histogram, pyramid sum, entropy) + E2E latency marker."""
    if not stage_rows:
        raise ValueError("benchmark_stages.csv is empty")

    variants = [r["variant"] for r in stage_rows]
    labels = [_short_label(v) for v in variants]
    quant = np.array([float(r["quantize_ms"]) for r in stage_rows], dtype=np.float64)
    hist = np.array([float(r["histogram_ms"]) for r in stage_rows], dtype=np.float64)
    pyr_sum = np.array([float(r["pyramid_sum_ms"]) for r in stage_rows], dtype=np.float64)
    entropy = np.array([float(r["entropy_ms"]) for r in stage_rows], dtype=np.float64)
    sums = quant + hist + pyr_sum + entropy
    e2e = np.array([float(r["e2e_median_ms"]) for r in stage_rows], dtype=np.float64)

    stage_specs = [
        ("Quantize", quant, "#4c72b0"),
        ("Fine / Region Histograms (incl. C fused quant)", hist, "#dd8452"),
        ("Pyramid Child Sum", pyr_sum, "#8172b3"),
        ("Shannon Entropy LUT", entropy, "#55a868"),
    ]

    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=dpi)
    x = np.arange(len(variants))

    # Left panel: absolute milliseconds stacked
    bottom_abs = np.zeros(len(variants), dtype=np.float64)
    for st_name, vals, col in stage_specs:
        ax_abs.bar(
            x, vals, bottom=bottom_abs, width=0.56, color=col, edgecolor="white", label=st_name
        )
        bottom_abs += vals
    ax_abs.scatter(
        x,
        e2e,
        marker="D",
        s=50,
        color="#111827",
        zorder=5,
        label="Un-instrumented E2E median (ms)",
    )
    for i in range(len(variants)):
        ax_abs.text(
            x[i],
            max(sums[i], e2e[i]) * 1.03,
            f"sum={sums[i]:.3f}\ne2e={e2e[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_abs.set_ylim(0.0, float(np.max(np.maximum(sums, e2e))) * 1.24)
    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(labels, fontsize=8.5)
    ax_abs.set_ylabel("Median Stage Latency (ms, batch=1)")
    ax_abs.set_title("Absolute Staged Time vs Un-instrumented E2E", fontsize=10, fontweight="bold")
    ax_abs.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax_abs.legend(loc="upper right", fontsize=7.8)

    # Right panel: 100%-normalized stage composition so sub-ms C stages are clearly visible
    safe_sums = np.where(sums > 0.0, sums, 1.0)
    bottom_pct = np.zeros(len(variants), dtype=np.float64)
    for st_name, vals, col in stage_specs:
        pct_vals = (vals / safe_sums) * 100.0
        ax_pct.bar(
            x,
            pct_vals,
            bottom=bottom_pct,
            width=0.56,
            color=col,
            edgecolor="white",
            label=st_name,
        )
        for i, pv in enumerate(pct_vals):
            if pv >= 6.5:
                ax_pct.text(
                    x[i],
                    bottom_pct[i] + pv / 2.0,
                    f"{pv:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )
        bottom_pct += pct_vals

    ax_pct.set_ylim(0.0, 108.0)
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(labels, fontsize=8.5)
    ax_pct.set_ylabel("Share of Summed Stage Time (%)")
    ax_pct.set_title("Relative Stage Cost Composition (%)", fontsize=10, fontweight="bold")
    ax_pct.grid(True, axis="y", linestyle=":", alpha=0.55)

    fig.suptitle(
        "Batch=1 Stage Decomposition Across Optimization Variants (Note: C fuses quantize into histogram pass)",
        fontsize=10.5,
        fontweight="bold",
    )
    plt.tight_layout()
    return _save_fig(fig, "fig03_stage_breakdown", output_dir, dpi, formats)


def _plot_sweeps_speedup(
    sweep_rows: list[dict[str, str]],
    output_dir: Path,
    dpi: int,
    formats: tuple[str, ...],
) -> list[Path]:
    """Figure 4: Speedup vs V0 across bins sweep and image_size sweep (grid_levels fixed)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), dpi=dpi)
    sweep_defs = [
        ("bins", "bins", "Histogram Bins B (fixed image_size)", axes[0]),
        ("image_size", "image_size", "Image Size H=W (pixels, fixed bins)", axes[1]),
    ]

    markers = ("s", "o", "^", "D")
    legend_handles = None
    legend_labels = None
    for axis_name, x_col, x_title, ax in sweep_defs:
        subset = [r for r in sweep_rows if r["sweep_axis"] == axis_name]
        if not subset:
            ax.text(0.5, 0.5, f"No data for sweep '{axis_name}'", ha="center", va="center")
            continue

        x_vals_unique = sorted({int(r[x_col]) for r in subset})
        x_pos = np.arange(len(x_vals_unique), dtype=np.float64)
        var_order: list[str] = []
        for r in subset:
            if r["variant"] not in var_order:
                var_order.append(r["variant"])

        y_max_all = 1.0
        for v_idx, v_name in enumerate(var_order):
            by_x = {int(r[x_col]): r for r in subset if r["variant"] == v_name}
            ys = np.array([float(by_x[xv]["speedup_vs_v0"]) for xv in x_vals_unique], dtype=np.float64)
            y_lo = np.array(
                [float(by_x[xv]["speedup_ci95_low"]) for xv in x_vals_unique], dtype=np.float64
            )
            y_hi = np.array(
                [float(by_x[xv]["speedup_ci95_high"]) for xv in x_vals_unique], dtype=np.float64
            )
            y_max_all = max(y_max_all, float(np.max(y_hi)))
            err_lo = np.maximum(ys - y_lo, 0.0)
            err_hi = np.maximum(y_hi - ys, 0.0)
            ax.errorbar(
                x_pos,
                ys,
                yerr=[err_lo, err_hi],
                marker=markers[v_idx % len(markers)],
                linewidth=1.9,
                markersize=6,
                capsize=4,
                color=_variant_color(v_name),
                label=v_name,
            )

        ax.axhline(1.0, color="#0f172a", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(0.8, y_max_all * 1.45)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(xv) for xv in x_vals_unique], fontsize=9.5)
        ax.set_xlabel(x_title)
        ax.set_ylabel("Speedup vs V0 Baseline (x, log scale)")
        ax.set_title(f"Batch=1 Speedup vs {x_col}", fontsize=10, fontweight="bold")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.suptitle(
        "Parameter Sensitivity Sweeps (Grid Levels Fixed, CPU Threads=1)",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=len(legend_handles),
            fontsize=8.5,
            frameon=True,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.85))
    return _save_fig(fig, "fig04_sweep_speedup", output_dir, dpi, formats)


def _plot_correctness_errors(
    correctness_rows: list[dict[str, str]],
    output_dir: Path,
    dpi: int,
    formats: tuple[str, ...],
) -> list[Path]:
    """Figure 5: Max absolute numerical error vs baseline V0 and src oracle (log axis)."""
    if not correctness_rows:
        raise ValueError("correctness.csv is empty")

    tol = float(correctness_rows[0]["tolerance_abs"])
    exact_floor = tol * 1e-6

    variants: list[str] = []
    for r in correctness_rows:
        if r["variant"] not in variants:
            variants.append(r["variant"])

    max_err_v0: list[float] = []
    max_err_src: list[float] = []
    exact_zero_flags: list[bool] = []
    for v_name in variants:
        rows_v = [r for r in correctness_rows if r["variant"] == v_name]
        e0 = max(float(r["max_abs_err_vs_v0"]) for r in rows_v)
        es = max(float(r["max_abs_err_vs_src"]) for r in rows_v)
        max_err_v0.append(e0)
        max_err_src.append(es)
        exact_zero_flags.append(max(e0, es) == 0.0)

    fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=dpi)
    x = np.arange(len(variants))
    width = 0.34

    plot_v0 = [max(v, exact_floor) for v in max_err_v0]
    plot_src = [max(v, exact_floor) for v in max_err_src]

    bars1 = ax.bar(
        x - width / 2,
        plot_v0,
        width=width,
        color=[_variant_color(v) for v in variants],
        edgecolor="#0f172a",
        alpha=0.9,
        label="max|variant - V0_baseline|",
        zorder=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        plot_src,
        width=width,
        color=[_variant_color(v) for v in variants],
        edgecolor="#0f172a",
        hatch="//",
        alpha=0.55,
        label="max|variant - src oracle|",
        zorder=3,
    )

    ax.axhline(
        tol,
        color="#dc2626",
        linestyle="--",
        linewidth=1.6,
        label=f"Pass threshold tolerance_abs = {tol:.0e}",
        zorder=4,
    )
    ax.axhline(
        exact_floor,
        color="#94a3b8",
        linestyle=":",
        linewidth=1.0,
        label=f"Exact zero floor marker ({exact_floor:.0e})",
    )

    for b, raw_val in zip(bars1, max_err_v0):
        lbl = "0.0\n(exact)" if raw_val == 0.0 else f"{raw_val:.2e}"
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() * 1.45,
            lbl,
            ha="center",
            va="bottom",
            fontsize=7.6,
        )
    for b, raw_val in zip(bars2, max_err_src):
        lbl = "0.0\n(exact)" if raw_val == 0.0 else f"{raw_val:.2e}"
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() * 1.45,
            lbl,
            ha="center",
            va="bottom",
            fontsize=7.6,
        )

    ax.set_yscale("log")
    ax.set_ylim(exact_floor * 0.25, tol * 50.0)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(v) for v in variants], fontsize=9)
    ax.set_ylabel("Max Absolute Feature Error (log scale)")
    ax.set_title(
        "Worst-Case Numerical Parity Across All Correctness & Sweep Cases",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.55, zorder=0)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return _save_fig(fig, "fig05_correctness_max_error", output_dir, dpi, formats)


def _plot_batch1_distribution(
    sample_rows: list[dict[str, str]],
    output_dir: Path,
    dpi: int,
    formats: tuple[str, ...],
) -> list[Path]:
    """Figure 6: Violin + boxplot of batch=1 sample latencies across variants."""
    if not sample_rows:
        raise ValueError("benchmark_batch1_samples.csv is empty")

    grouped: dict[str, list[float]] = {}
    for r in sample_rows:
        grouped.setdefault(r["variant"], []).append(float(r["latency_ms"]))

    variants = list(grouped.keys())
    arrays = [np.asarray(grouped[v], dtype=np.float64) for v in variants]
    positions = np.arange(1, len(variants) + 1)

    fig, ax = plt.subplots(figsize=(8.8, 5.1), dpi=dpi)
    parts = ax.violinplot(
        arrays,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.65,
    )
    for body, v_name in zip(parts["bodies"], variants):
        body.set_facecolor(_variant_color(v_name))
        body.set_edgecolor("#0f172a")
        body.set_alpha(0.45)

    ax.boxplot(
        arrays,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=True,
        boxprops={"facecolor": "white", "edgecolor": "#0f172a", "linewidth": 1.1},
        medianprops={"color": "#dc2626", "linewidth": 1.8},
        whiskerprops={"color": "#0f172a", "linewidth": 1.0},
        capprops={"color": "#0f172a", "linewidth": 1.0},
        flierprops={"marker": ".", "markersize": 3.5, "alpha": 0.45},
    )

    x_labels: list[str] = []
    for v_name, arr in zip(variants, arrays):
        med = float(np.median(arr))
        q25, q75 = np.percentile(arr, [25.0, 75.0])
        x_labels.append(f"{_short_label(v_name)}\nmed={med:.4f} ms | IQR={q75 - q25:.4f}")

    ax.set_yscale("log")
    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=8.3)
    ax.set_ylabel("Single-Image Latency (ms, log scale)")
    ax.set_title(
        f"Batch=1 Latency Distribution (n={len(arrays[0])} Timed Samples per Variant)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.55)
    plt.tight_layout()
    return _save_fig(fig, "fig06_batch1_distribution", output_dir, dpi, formats)


def _plot_vs_cnn_cpu(
    vs_cnn_rows: list[dict[str, str]],
    output_dir: Path,
    dpi: int,
    formats: tuple[str, ...],
) -> list[Path]:
    """Figure 7: End-to-end T_desc + T_model vs PyTorch CPU CNNs (latency & parameters)."""
    if not vs_cnn_rows:
        raise ValueError("benchmark_vs_cnn.csv is empty")

    # Show all CNN models, all descriptor + LR dot core pipelines, and fastest descriptor + LR sklearn wrapper
    cnn_rows = [r for r in vs_cnn_rows if r["category"] == "cnn_cpu"]
    dot_rows = [r for r in vs_cnn_rows if r["category"] == "descriptor_lr_dot"]
    sk_rows = [r for r in vs_cnn_rows if r["category"] == "descriptor_lr_sklearn"]

    selected: list[dict[str, str]] = list(cnn_rows) + list(dot_rows)
    if sk_rows:
        fastest_sk = min(sk_rows, key=lambda r: float(r["t_total_ms"]))
        selected.append(fastest_sk)

    fig, (ax_lat, ax_par) = plt.subplots(
        1, 2, figsize=(14.0, 5.6), dpi=dpi, gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    y = np.arange(len(selected))
    names = [r["pipeline_name"] for r in selected]
    t_desc = np.array([float(r["t_desc_ms"]) for r in selected], dtype=np.float64)
    t_model = np.array([float(r["t_model_ms"]) for r in selected], dtype=np.float64)
    t_total = np.array([float(r["t_total_ms"]) for r in selected], dtype=np.float64)
    ci_lo = np.array([float(r["t_total_ci95_low_ms"]) for r in selected], dtype=np.float64)
    ci_hi = np.array([float(r["t_total_ci95_high_ms"]) for r in selected], dtype=np.float64)
    params_tot = np.array([int(r["params_total"]) for r in selected], dtype=np.int64)
    speedups = [float(r["speedup_vs_ds_cnn"]) for r in selected]

    bar_colors: list[str] = []
    for r in selected:
        if r["category"] == "cnn_cpu":
            bar_colors.append("#dc2626" if r["model_name"] == "ds_cnn" else "#ea580c")
        else:
            bar_colors.append(_variant_color(r["descriptor_variant"]))

    xerr_lo = np.maximum(t_total - ci_lo, 0.0)
    xerr_hi = np.maximum(ci_hi - t_total, 0.0)
    ax_lat.barh(
        y,
        t_total,
        height=0.58,
        color=bar_colors,
        edgecolor="#0f172a",
        alpha=0.88,
        xerr=[xerr_lo, xerr_hi],
        capsize=4,
        zorder=3,
    )
    ax_lat.set_xscale("log")
    x_min = max(1e-4, float(np.min(ci_lo)) * 0.45)
    x_max = float(np.max(ci_hi)) * 28.0
    ax_lat.set_xlim(x_min, x_max)

    for idx in range(len(selected)):
        right_edge = max(float(ci_hi[idx]), float(t_total[idx]))
        if selected[idx]["category"] == "cnn_cpu":
            detail = (
                f"{t_total[idx]:.3f} ms ({speedups[idx]:.1f}x vs ds_cnn)\n"
                f"{params_tot[idx]:,} params"
            )
        else:
            detail = (
                f"{t_total[idx]:.4f} ms ({speedups[idx]:.1f}x vs ds_cnn)\n"
                f"desc={t_desc[idx]:.4f} + clf={t_model[idx]:.4f} | {params_tot[idx]} params"
            )
        ax_lat.text(
            right_edge * 1.16,
            y[idx],
            detail,
            va="center",
            ha="left",
            fontsize=7.3,
            linespacing=1.15,
        )

    ax_lat.set_yticks(y)
    ax_lat.set_yticklabels(names, fontsize=8.5)
    ax_lat.invert_yaxis()
    ax_lat.set_xlabel("Total CPU Batch=1 Latency T_desc + T_model (ms, log scale)")
    ax_lat.set_title("End-to-End CPU Inference Latency vs PyTorch CNNs", fontsize=10, fontweight="bold")
    ax_lat.grid(True, which="both", axis="x", linestyle=":", alpha=0.55, zorder=0)

    # Right panel: Parameters vs Total Latency Pareto frontier (log-log) with collision-free labels
    par_points: list[tuple[int, float, str]] = []
    for idx, r in enumerate(selected):
        marker = "s" if r["category"] == "cnn_cpu" else ("^" if r["category"] == "descriptor_lr_sklearn" else "o")
        ax_par.scatter(
            params_tot[idx],
            t_total[idx],
            s=85,
            marker=marker,
            color=bar_colors[idx],
            edgecolor="#0f172a",
            zorder=4,
        )
        short = r["model_name"] if r["category"] == "cnn_cpu" else f"{r['descriptor_variant'].split('_')[0]}+LR"
        if r["category"] == "descriptor_lr_sklearn":
            short += "(sk)"
        par_points.append((int(params_tot[idx]), float(t_total[idx]), short))

    # Stagger labels sharing nearly identical parameter counts (e.g., 256-param variants)
    clusters: dict[int, list[int]] = {}
    for idx, (p_cnt, _, _) in enumerate(par_points):
        clusters.setdefault(p_cnt, []).append(idx)

    y_offsets_par = [0.0] * len(par_points)
    for idxs in clusters.values():
        idxs_sorted = sorted(idxs, key=lambda i: par_points[i][1])
        sub_clusters: list[list[int]] = []
        for idx in idxs_sorted:
            ly = float(np.log10(par_points[idx][1]))
            if not sub_clusters or (ly - float(np.log10(par_points[sub_clusters[-1][-1]][1]))) >= 0.14:
                sub_clusters.append([idx])
            else:
                sub_clusters[-1].append(idx)
        for grp in sub_clusters:
            if len(grp) == 1:
                y_offsets_par[grp[0]] = 3.0
            else:
                mid = (len(grp) - 1) / 2.0
                for pos, orig_idx in enumerate(grp):
                    y_offsets_par[orig_idx] = (pos - mid) * 11.5

    for idx, (p_cnt, t_val, label_txt) in enumerate(par_points):
        ax_par.annotate(
            label_txt,
            (p_cnt, t_val),
            textcoords="offset points",
            xytext=(7.0, y_offsets_par[idx]),
            va="center",
            ha="left",
            fontsize=7.8,
        )

    ax_par.set_xscale("log")
    ax_par.set_yscale("log")
    ax_par.set_xlim(float(np.min(params_tot)) * 0.65, float(np.max(params_tot)) * 2.5)
    ax_par.set_ylim(float(np.min(t_total)) * 0.60, float(np.max(t_total)) * 1.85)
    ax_par.set_xlabel("Stored Classifier Parameters (count, log scale)")
    ax_par.set_ylabel("Total Batch=1 Latency (ms, log scale)")
    ax_par.set_title("Pareto Frontier: Lower-Left = Faster & Lighter", fontsize=10, fontweight="bold")
    ax_par.grid(True, which="both", linestyle=":", alpha=0.55)

    fig.suptitle(
        "Entropy Descriptor + LR vs Lightweight PyTorch CNNs (CPU, threads=1, batch=1)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    return _save_fig(fig, "fig07_descriptor_vs_cnn_cpu", output_dir, dpi, formats)


def generate_all_plots(
    cfg: EntropyOptimizationConfig,
    input_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Read Stage 6 artifacts exclusively and render all 7 Stage 7 figures."""
    manifest_path = input_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Run manifest not found: '{manifest_path}'. Run benchmark_entropy_variants.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    latency_rows = _read_csv_rows(input_dir / "benchmark_latency.csv")
    stage_rows = _read_csv_rows(input_dir / "benchmark_stages.csv")
    sweep_rows = _read_csv_rows(input_dir / "benchmark_sweeps.csv")
    correctness_rows = _read_csv_rows(input_dir / "correctness.csv")
    sample_rows = _read_csv_rows(input_dir / "benchmark_batch1_samples.csv")
    vs_cnn_rows = _read_csv_rows(input_dir / "benchmark_vs_cnn.csv")

    dpi = cfg.plots.dpi
    fmts = cfg.plots.formats

    saved_paths: list[Path] = []
    saved_paths.extend(_plot_batch1_latency(latency_rows, manifest, output_dir, dpi, fmts))
    saved_paths.extend(_plot_throughput_vs_batch(latency_rows, output_dir, dpi, fmts))
    saved_paths.extend(_plot_stage_breakdown(stage_rows, output_dir, dpi, fmts))
    saved_paths.extend(_plot_sweeps_speedup(sweep_rows, output_dir, dpi, fmts))
    saved_paths.extend(_plot_correctness_errors(correctness_rows, output_dir, dpi, fmts))
    saved_paths.extend(_plot_batch1_distribution(sample_rows, output_dir, dpi, fmts))
    saved_paths.extend(_plot_vs_cnn_cpu(vs_cnn_rows, output_dir, dpi, fmts))

    for p in saved_paths:
        logger.info("Saved scientific plot: %s", p)
    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for scientific plot rendering from CSV outputs."""
    parser = argparse.ArgumentParser(
        prog="plot_entropy_optimization.py",
        description="Render all 7 comparative scientific figures strictly from benchmark CSV outputs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/optimization/entropy_maps.toml"),
        help="Path to optimization TOML config.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory holding CSV/JSON outputs (defaults to [entropy_optimization.plots].output_dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write generated plots (defaults to --input-dir).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for Stage 7 visualization script."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg = load_optimization_config(args.config, project_root=PROJECT_ROOT)
        in_dir = (
            resolve_project_path(args.input_dir, PROJECT_ROOT)
            if args.input_dir is not None
            else cfg.plots.output_dir
        )
        out_dir = (
            resolve_project_path(args.output_dir, PROJECT_ROOT)
            if args.output_dir is not None
            else in_dir
        )
        _ = generate_all_plots(cfg=cfg, input_dir=in_dir, output_dir=out_dir)
    except (OSError, ValueError, KeyError) as exc:
        logger.error("Plot generation failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
