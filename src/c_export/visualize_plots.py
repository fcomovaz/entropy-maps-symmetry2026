"""Generate multi-panel diagnostic figures explaining C99 export, parity, memory, and timing CSVs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import PALETTES, annotate_points_non_overlapping, apply_style
from .config import CImpConfig
from .visualize_data import (
    DescriptorFeatureInsights,
    ModelSeriesBundle,
    TargetVisualizationBundle,
    VisualizationReport,
    load_target_bundle,
    write_summary_tables,
)

_MDPI_PAL = PALETTES["editorial"]
_COLOR_F32 = _MDPI_PAL[0]
_COLOR_Q15 = _MDPI_PAL[3]
_COLOR_INT8 = _MDPI_PAL[4]
_COLOR_NEG = _MDPI_PAL[5]
_COLOR_POS = _MDPI_PAL[1]
_COLOR_PARETO = _MDPI_PAL[2]
_COLOR_GATE = _MDPI_PAL[1]


def _pick_pareto_legend_loc(
    pts_norm: list[tuple[float, float]],
) -> tuple[str, tuple[float, float, float, float]]:
    """Select the sparse plot quadrant for the Pareto legend box in axes coordinates."""
    candidates: list[tuple[str, tuple[float, float, float, float]]] = [
        ("lower right", (0.60, 0.99, 0.02, 0.24)),
        ("center left", (0.02, 0.42, 0.38, 0.62)),
        ("center", (0.30, 0.70, 0.38, 0.62)),
        ("upper left", (0.02, 0.42, 0.76, 0.98)),
        ("center right", (0.58, 0.98, 0.38, 0.62)),
        ("upper right", (0.58, 0.98, 0.76, 0.98)),
        ("lower left", (0.02, 0.42, 0.02, 0.24)),
    ]

    def _clearance(rect: tuple[float, float, float, float]) -> tuple[int, float]:
        x0, x1, y0, y1 = rect
        hits = sum(1 for ux, vy in pts_norm if (x0 - 0.03) <= ux <= (x1 + 0.03) and (y0 - 0.04) <= vy <= (y1 + 0.04))
        min_d = min(
            (
                math.hypot(
                    max(x0 - ux, 0.0, ux - x1),
                    max(y0 - vy, 0.0, vy - y1),
                )
                for ux, vy in pts_norm
            ),
            default=1.0,
        )
        return (-hits, min_d)

    return max(candidates, key=lambda item: _clearance(item[1]))


def _compute_pareto_indices(costs: list[float], qualities: list[float]) -> set[int]:
    """Identify indices maximizing quality while minimizing cost."""
    pareto: set[int] = set()
    for i, (ci, qi) in enumerate(zip(costs, qualities)):
        dominated = any(
            j != i and cj <= ci and qj >= qi and (cj < ci or qj > qi)
            for j, (cj, qj) in enumerate(zip(costs, qualities))
        )
        if not dominated:
            pareto.add(i)
    return pareto


def _plot_fig1_classification(
    bundle: TargetVisualizationBundle,
    out_path: Path,
    dpi: int,
) -> Path:
    """Plot F1, PR-AUC, ROC-AUC (f32 vs Q15) alongside quantization drift and flips."""
    ordered = sorted(bundle.models, key=lambda m: (m.stats_q15.f1, m.stats_f32.f1))
    names = [m.name for m in ordered]
    y_pos = np.arange(len(ordered), dtype=np.float64)
    bar_h = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), dpi=dpi)
    fig.suptitle(
        f"Classification Quality & Fixed-Point Quantization Impact — {bundle.target_tag} "
        f"(N={ordered[0].stats_q15.n_samples}, pos={ordered[0].stats_q15.n_positives})",
        fontsize=13,
        fontweight="bold",
    )

    f32_f1 = np.array([m.stats_f32.f1 for m in ordered])
    q15_f1 = np.array([m.stats_q15.f1 for m in ordered])
    f32_pr = np.array([0.0 if math.isnan(m.stats_f32.pr_auc) else m.stats_f32.pr_auc for m in ordered])
    q15_pr = np.array([0.0 if math.isnan(m.stats_q15.pr_auc) else m.stats_q15.pr_auc for m in ordered])
    f32_roc = np.array([0.0 if math.isnan(m.stats_f32.roc_auc) else m.stats_f32.roc_auc for m in ordered])
    q15_roc = np.array([0.0 if math.isnan(m.stats_q15.roc_auc) else m.stats_q15.roc_auc for m in ordered])

    def _draw_metric_bars(
        ax: plt.Axes,
        v_f32: np.ndarray,
        v_q15: np.ndarray,
        title: str,
        xlabel: str,
        ref_line: float | None = None,
        ref_label: str | None = None,
    ) -> None:
        b1 = ax.barh(y_pos + bar_h / 2.0, v_f32, height=bar_h, color=_COLOR_F32, label="float32 (Python)")
        b2 = ax.barh(y_pos - bar_h / 2.0, v_q15, height=bar_h, color=_COLOR_Q15, label="Q15 / int8 (C/Py mirror)")
        if ref_line is not None and not math.isnan(ref_line):
            ax.axvline(ref_line, color="#555555", linestyle="--", linewidth=1.2, label=ref_label or "")
        for rect in b1:
            w = rect.get_width()
            ax.text(w + 0.012, rect.get_y() + rect.get_height() / 2.0, f"{w:.3f}", va="center", fontsize=7.5)
        for rect in b2:
            w = rect.get_width()
            ax.text(w + 0.012, rect.get_y() + rect.get_height() / 2.0, f"{w:.3f}", va="center", fontsize=7.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(0.0, 1.13)
        ax.set_xlabel(xlabel)
        ax.set_title(title, pad=22)
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        ax.legend(fontsize=7.8, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False)

    pos_rate = ordered[0].stats_q15.n_positives / max(1, ordered[0].stats_q15.n_samples)
    _draw_metric_bars(axes[0, 0], f32_f1, q15_f1, "F1 Score (Higher is Better)", "F1 Score")
    _draw_metric_bars(
        axes[0, 1],
        f32_pr,
        q15_pr,
        "Precision-Recall AUC (Imbalanced Defect Detection)",
        "PR-AUC",
        ref_line=pos_rate,
        ref_label=f"Random baseline ({pos_rate:.2f})",
    )
    _draw_metric_bars(
        axes[1, 0],
        f32_roc,
        q15_roc,
        "ROC-AUC (Threshold-Free Ranking)",
        "ROC-AUC",
        ref_line=0.5,
        ref_label="Random (0.50)",
    )

    ax_d = axes[1, 1]
    mean_d = np.array([max(1e-6, m.score_diff_mean) for m in ordered])
    max_d = np.array([max(1e-6, m.score_diff_max) for m in ordered])
    ax_d.barh(y_pos + bar_h / 2.0, mean_d, height=bar_h, color=_MDPI_PAL[5], label="Mean |score_Q15 - score_F32|")
    ax_d.barh(y_pos - bar_h / 2.0, max_d, height=bar_h, color=_MDPI_PAL[4], alpha=0.85, label="Max |score_Q15 - score_F32|")
    ax_d.set_xscale("log")
    min_xlim = min(1e-4, float(np.min(mean_d)) * 0.4)
    max_xlim = max(float(np.max(max_d)) * 28.0, 1e-1)
    ax_d.set_xlim(min_xlim, max_xlim)
    for idx, m in enumerate(ordered):
        tag = f"flips={m.q15_vs_f32_flips} [{m.weight_format}]"
        ax_d.text(max_d[idx] * 1.18, y_pos[idx], tag, va="center", fontsize=8)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(names, fontsize=9)
    ax_d.set_xlabel("Absolute Quantization Margin Drift (log scale)")
    ax_d.set_title("Quantization Drift & Decision Flips (Q15 vs float32)", pad=22)
    ax_d.grid(True, which="both", axis="x", linestyle=":", alpha=0.5)
    ax_d.legend(fontsize=7.8, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_fig2_score_separation(
    bundle: TargetVisualizationBundle,
    out_path: Path,
    dpi: int,
) -> Path:
    """Plot per-model Q15 vs float32 score/margin scatter colored by ground-truth label."""
    n = len(bundle.models)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes_arr = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.3 * nrows), dpi=dpi, squeeze=False)
    fig.suptitle(
        f"Score Agreement & Class Separation (Q15 vs float32) — {bundle.target_tag}",
        fontsize=13,
        fontweight="bold",
    )

    flat_axes = list(axes_arr.ravel())
    for idx, m in enumerate(bundle.models):
        ax = flat_axes[idx]
        neg = m.y_true == 0
        pos = m.y_true == 1
        flips = m.py_f32_label != m.py_q15_label

        ax.scatter(
            m.py_f32_score[neg],
            m.py_q15_score[neg],
            c=_COLOR_NEG,
            s=16,
            alpha=0.55,
            label="Normal (y=0)",
        )
        ax.scatter(
            m.py_f32_score[pos],
            m.py_q15_score[pos],
            c=_COLOR_POS,
            marker="^",
            s=22,
            alpha=0.75,
            label="Defect (y=1)",
        )
        if np.any(flips):
            ax.scatter(
                m.py_f32_score[flips],
                m.py_q15_score[flips],
                facecolors="none",
                edgecolors="#111111",
                linewidths=1.4,
                s=75,
                label=f"Q15 flip ({int(np.sum(flips))})",
                zorder=5,
            )

        thr = m.decision_threshold
        lo = float(min(np.min(m.py_f32_score), np.min(m.py_q15_score), thr))
        hi = float(max(np.max(m.py_f32_score), np.max(m.py_q15_score), thr))
        pad = 0.06 * max(1.0, hi - lo)
        lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, color="#444444", linestyle="--", linewidth=1.1, label="y = x")
        ax.axhline(thr, color="#666666", linestyle=":", linewidth=1.0, label=f"thr={thr:+.2f}")
        ax.axvline(thr, color="#666666", linestyle=":", linewidth=1.0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)

        r_txt = f"{m.score_pearson_r:.5f}" if not math.isnan(m.score_pearson_r) else "n/a"
        ax.set_title(f"{m.name} [{m.weight_format}]", fontsize=10, fontweight="bold")
        ax.set_xlabel(f"float32 score (> {thr:+.2f} => Defect)")
        ax.set_ylabel(f"Q15 score (> {thr:+.2f} => Defect)")
        ax.text(
            0.03,
            0.96,
            f"F1={m.stats_q15.f1:.3f} | thr={thr:+.2f}\nmax|d|={m.score_diff_max:.3f}\nr={r_txt}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(fontsize=7.2, loc="lower right")

    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_fig3_confusion_matrices(
    bundle: TargetVisualizationBundle,
    out_path: Path,
    dpi: int,
) -> Path:
    """Plot row-normalized Q15 confusion matrices with exact TP/FP/TN/FN counts."""
    n = len(bundle.models)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes_arr = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.9 * nrows), dpi=dpi, squeeze=False)
    fig.suptitle(
        f"Q15 Confusion Matrices (True vs Predicted) — {bundle.target_tag}",
        fontsize=13,
        fontweight="bold",
    )

    flat_axes = list(axes_arr.ravel())
    for idx, m in enumerate(bundle.models):
        ax = flat_axes[idx]
        s = m.stats_q15
        mat = np.array([[s.tn, s.fp], [s.fn, s.tp]], dtype=np.int64)
        row_sums = np.maximum(mat.sum(axis=1, keepdims=True), 1)
        pct = 100.0 * mat.astype(np.float64) / row_sums.astype(np.float64)

        ax.imshow(pct, cmap="Blues", vmin=0.0, vmax=100.0)
        for r in range(2):
            for c in range(2):
                val = int(mat[r, c])
                p = float(pct[r, c])
                txt_color = "white" if p > 55.0 else "#1a202c"
                role = (("TN", "FP"), ("FN", "TP"))[r][c]
                ax.text(
                    c,
                    r,
                    f"{role}\n{val}\n({p:.1f}%)",
                    ha="center",
                    va="center",
                    color=txt_color,
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Normal", "Pred Defect"], fontsize=8.5)
        ax.set_yticklabels(["True Normal", "True Defect"], fontsize=8.5)
        ax.set_title(
            f"{m.name}\nAcc={s.accuracy:.3f} | P={s.precision:.3f} | R={s.recall:.3f} | F1={s.f1:.3f}",
            fontsize=9.2,
        )

    for j in range(n, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_fig4_c_parity(
    bundle: TargetVisualizationBundle,
    cfg: CImpConfig,
    out_path: Path,
    dpi: int,
) -> Path | None:
    """Plot compiled C99 vs Python parity gates, per-sample diffs, and boundary flips."""
    parity_rows: list[tuple[str, str, float, int, int]] = []
    if bundle.descriptor_parity is not None:
        dp = bundle.descriptor_parity
        parity_rows.append(
            ("descriptor_x94", "descriptor", dp.f32_max_abs_diff, dp.q15_vs_py_q15_max_raw_diff, 0)
        )
    for m in bundle.models:
        if m.c_f32_max_abs_diff is not None and m.c_q15_max_raw_diff is not None:
            parity_rows.append(
                (
                    m.name,
                    m.family,
                    m.c_f32_max_abs_diff,
                    m.c_q15_max_raw_diff,
                    int(m.c_f32_label_mismatches or 0) + int(m.c_q15_label_mismatches or 0),
                )
            )
    if not parity_rows:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.8), dpi=dpi)
    fig.suptitle(
        f"Compiled C99 vs Python Numerical Parity Audit — {bundle.target_tag}",
        fontsize=13,
        fontweight="bold",
    )

    # Panel A: Max float32 error vs tolerance gates
    ax_a = axes[0, 0]
    p_names = [r[0] for r in parity_rows]
    p_f32 = np.array([max(1e-10, r[2]) for r in parity_rows])
    y_idx = np.arange(len(p_names), dtype=np.float64)
    colors_a = [_COLOR_PARETO if r[1] == "descriptor" else _COLOR_F32 for r in parity_rows]
    bars_a = ax_a.barh(y_idx, p_f32, color=colors_a)
    ax_a.axvline(
        cfg.verification.f32_logit_abs_tol,
        color=_COLOR_GATE,
        linestyle="--",
        linewidth=1.4,
        label=f"Logit/margin gate ({cfg.verification.f32_logit_abs_tol:.0e})",
    )
    ax_a.axvline(
        cfg.verification.f32_descriptor_abs_tol,
        color=_MDPI_PAL[3],
        linestyle=":",
        linewidth=1.4,
        label=f"Descriptor gate ({cfg.verification.f32_descriptor_abs_tol:.0e})",
    )
    ax_a.set_xscale("log")
    ax_a.set_xlim(
        min(1e-9, float(np.min(p_f32)) * 0.25),
        max(cfg.verification.f32_logit_abs_tol * 25.0, float(np.max(p_f32)) * 20.0),
    )
    for rect, raw_val in zip(bars_a, [r[2] for r in parity_rows]):
        w = rect.get_width()
        yc = rect.get_y() + rect.get_height() / 2.0
        if raw_val >= cfg.verification.f32_logit_abs_tol * 0.08:
            ax_a.text(w * 0.82, yc, f"{raw_val:.2e}", va="center", ha="right", fontsize=7.8, color="white", fontweight="bold")
        else:
            ax_a.text(w * 1.35, yc, f"{raw_val:.2e}", va="center", ha="left", fontsize=8)
    ax_a.set_yticks(y_idx)
    ax_a.set_yticklabels(p_names, fontsize=9)
    ax_a.set_xlabel("Max |C_f32 - Python_f32| (log scale)")
    ax_a.set_title("Float32 Numerical Error vs Acceptance Gates")
    ax_a.grid(True, which="both", axis="x", linestyle=":", alpha=0.5)
    ax_a.legend(fontsize=8, loc="lower right")

    # Panel B: C99 Q15 raw diff & C label mismatches
    ax_b = axes[0, 1]
    q_raw = np.array([r[3] for r in parity_rows], dtype=np.float64)
    lbl_mm = np.array([r[4] for r in parity_rows], dtype=np.float64)
    ax_b.barh(y_idx + 0.18, q_raw, height=0.34, color=_COLOR_Q15, label="C_q15 vs Py_q15 max raw bit diff")
    ax_b.barh(y_idx - 0.18, lbl_mm, height=0.34, color=_COLOR_GATE, label="C vs Py label mismatches")
    for idx, (qr, lm) in enumerate(zip(q_raw, lbl_mm)):
        status = "BIT-EXACT (0 diff)" if qr == 0 and lm == 0 else f"raw={int(qr)}, mm={int(lm)}"
        ax_b.text(max(qr, lm) + 0.08, y_idx[idx], status, va="center", fontsize=8.5, fontweight="bold", color=_COLOR_PARETO if qr == 0 and lm == 0 else _COLOR_GATE)
    ax_b.set_xlim(0.0, max(1.5, float(np.max(np.maximum(q_raw, lbl_mm))) + 1.0))
    ax_b.set_yticks(y_idx)
    ax_b.set_yticklabels(p_names, fontsize=9)
    ax_b.set_xlabel("Raw Integer / Label Discrepancy Count")
    ax_b.set_title("Q15 Bit-Exact Verification & Label Parity")
    ax_b.grid(axis="x", linestyle=":", alpha=0.5)
    ax_b.legend(fontsize=8, loc="lower right")

    # Panel C: Per-sample f32 error distribution from parity_per_sample.csv
    ax_c = axes[1, 0]
    diff_models = [m.name for m in bundle.models if m.name in bundle.per_sample_f32_abs_diff]
    if diff_models:
        data = [np.maximum(bundle.per_sample_f32_abs_diff[name], 1e-10) for name in diff_models]
        ax_c.boxplot(
            data,
            orientation="horizontal",
            tick_labels=diff_models,
            showfliers=True,
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.45},
        )
        ax_c.axvline(cfg.verification.f32_logit_abs_tol, color=_COLOR_GATE, linestyle="--", linewidth=1.2, label="Gate")
        ax_c.set_xscale("log")
        ax_c.set_xlabel("Per-Sample |C_f32 - Py_f32| (log scale)")
        ax_c.set_title("Per-Sample Error Spread Across All Test Images")
        ax_c.grid(True, which="both", axis="x", linestyle=":", alpha=0.5)
        ax_c.legend(fontsize=8)
    else:
        ax_c.text(0.5, 0.5, "parity_per_sample.csv not available", ha="center", va="center", transform=ax_c.transAxes)
        ax_c.set_axis_off()

    # Panel D: Near-threshold decision margin explanation for Q15-vs-F32 flips
    ax_d = axes[1, 1]
    all_margins: list[float] = []
    flip_margins: list[float] = []
    for m in bundle.models:
        abs_m = np.abs(m.py_f32_score - m.decision_threshold)
        flips = m.py_f32_label != m.py_q15_label
        all_margins.extend(abs_m.tolist())
        if np.any(flips):
            flip_margins.extend(abs_m[flips].tolist())
    if all_margins:
        bins = np.logspace(-4, max(0.5, math.log10(max(all_margins) + 1e-3)), 30)
        ax_d.hist(np.maximum(all_margins, 1e-4), bins=bins, alpha=0.65, color=_COLOR_F32, label=f"All evaluations (n={len(all_margins)})")
        if flip_margins:
            ax_d.hist(
                np.maximum(flip_margins, 1e-4),
                bins=bins,
                alpha=0.9,
                color=_COLOR_POS,
                label=f"Q15-vs-F32 flips (n={len(flip_margins)})",
            )
            ax_d.axvline(max(flip_margins), color=_COLOR_POS, linestyle="--", linewidth=1.2, label=f"Max flipped |score - thr|={max(flip_margins):.4f}")
        ax_d.set_xscale("log")
        ax_d.set_xlabel("Distance from Model Threshold |f32_score - thr| (log scale)")
        ax_d.set_ylabel("Count")
        ax_d.set_title("Why Rare Q15 Flips Occur Only Near Decision Threshold")
        ax_d.grid(True, which="both", linestyle=":", alpha=0.5)
        ax_d.legend(fontsize=8)
    else:
        ax_d.set_axis_off()

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_fig5_memory_and_latency(
    bundle: TargetVisualizationBundle,
    out_path: Path,
    dpi: int,
) -> Path | None:
    """Plot Flash/SRAM budgets and C99 Q15 accuracy-latency Pareto frontier."""
    has_mem = any(m.weight_flash_bytes is not None for m in bundle.models)
    has_time = any(m.bench_q15_total_ms is not None for m in bundle.models)
    if not has_mem and not has_time:
        return None

    nrows = 2 if (has_mem and has_time) else 1
    fig, axes_arr = plt.subplots(nrows, 2, figsize=(15.5, 5.0 * nrows), dpi=dpi, squeeze=False)
    fig.suptitle(
        f"Embedded Memory Feasibility & C99 Desktop Latency Pareto — {bundle.target_tag}",
        fontsize=13,
        fontweight="bold",
    )

    row_ptr = 0
    if has_mem:
        ax_f = axes_arr[row_ptr, 0]
        ax_a = axes_arr[row_ptr, 1]
        mem_models = [m for m in bundle.models if m.weight_flash_bytes is not None]
        names_m = [m.name for m in mem_models]
        y_m = np.arange(len(mem_models), dtype=np.float64)
        flash_kb = np.array([max(0.05, float(m.weight_flash_bytes or 0) / 1024.0) for m in mem_models])
        f_colors = [_COLOR_INT8 if m.weight_format == "int8_per_channel" else _COLOR_F32 for m in mem_models]
        bars_f = ax_f.barh(y_m, flash_kb, color=f_colors)
        flash_budget_kb = bundle.budgets.flash_budget_bytes / 1024.0
        ax_f.axvline(flash_budget_kb, color=_COLOR_GATE, linestyle="--", linewidth=1.3, label=f"Flash budget ({flash_budget_kb:.0f} KB)")
        ax_f.set_xscale("log")
        ax_f.set_xlim(0.1, max(flash_budget_kb * 2.5, float(np.max(flash_kb)) * 3.5))
        for rect, m, kb in zip(bars_f, mem_models, flash_kb):
            ax_f.text(rect.get_width() * 1.15, rect.get_y() + rect.get_height() / 2.0, f"{kb:.1f} KB [{m.weight_format}]", va="center", fontsize=8)
        ax_f.set_yticks(y_m)
        ax_f.set_yticklabels(names_m, fontsize=9)
        ax_f.set_xlabel("Weight Flash Footprint (KB, log scale)")
        ax_f.set_title("Model Weights Flash Storage vs MCU Budget")
        ax_f.grid(True, which="both", axis="x", linestyle=":", alpha=0.5)
        ax_f.legend(fontsize=8, loc="lower right")

        cnn_mem = [m for m in mem_models if m.peak_arena_q15_bytes is not None]
        if cnn_mem:
            c_names = [m.name for m in cnn_mem]
            y_c = np.arange(len(cnn_mem), dtype=np.float64)
            ar_f32 = np.array([float(m.peak_arena_f32_bytes or 0) / 1024.0 for m in cnn_mem])
            ar_q15 = np.array([float(m.peak_arena_q15_bytes or 0) / 1024.0 for m in cnn_mem])
            ax_a.barh(y_c + 0.18, ar_f32, height=0.35, color=_COLOR_F32, alpha=0.75, label="Arena float32 (KB)")
            ax_a.barh(y_c - 0.18, ar_q15, height=0.35, color=_COLOR_Q15, label="Arena Q15 (KB)")
            sram_kb = bundle.budgets.sram_budget_bytes / 1024.0
            psram_kb = bundle.budgets.psram_budget_bytes / 1024.0
            ax_a.axvline(sram_kb, color=_COLOR_PARETO, linestyle="--", linewidth=1.4, label=f"Internal SRAM ({sram_kb:.0f} KB)")
            ax_a.axvline(psram_kb, color=_COLOR_GATE, linestyle=":", linewidth=1.4, label=f"PSRAM ({psram_kb:.0f} KB)")
            for idx, m in enumerate(cnn_mem):
                fit_tag = "Internal SRAM OK" if m.fits_internal_sram_q15 else ("PSRAM OK" if m.fits_psram_q15 else "Exceeds PSRAM")
                ax_a.text(ar_f32[idx] + 35.0, y_c[idx], f"{ar_q15[idx]:.0f} KB ({fit_tag})", va="center", fontsize=8)
            ax_a.set_yticks(y_c)
            ax_a.set_yticklabels(c_names, fontsize=9)
            ax_a.set_xlim(0.0, max(psram_kb * 1.18, float(np.max(ar_f32)) * 1.32))
            ax_a.set_xlabel("Peak Activation Arena (KB)")
            ax_a.set_title("Static Activation Memory Arena: float32 vs Q15")
            ax_a.grid(axis="x", linestyle=":", alpha=0.5)
            ax_a.legend(fontsize=8, loc="upper right")
        else:
            ax_a.text(0.5, 0.5, "No CNN activation arena records present", ha="center", va="center", transform=ax_a.transAxes)
            ax_a.set_axis_off()
        row_ptr += 1

    if has_time:
        ax_p = axes_arr[row_ptr, 0]
        ax_l = axes_arr[row_ptr, 1]
        timed = [m for m in bundle.models if m.bench_q15_total_ms is not None]
        costs = [max(1e-4, float(m.bench_q15_total_ms or 0.0)) for m in timed]
        quals = [m.stats_q15.f1 for m in timed]
        pareto_idx = _compute_pareto_indices(costs, quals)

        for fam, color, marker, label in (
            ("tabular", _COLOR_F32, "o", "Classical x94 + ML (Q15 C)"),
            ("cnn", _COLOR_POS, "^", "Lightweight CNN (Q15 C)"),
        ):
            sub = [(m, c, q) for m, c, q in zip(timed, costs, quals) if m.family == fam]
            if not sub:
                continue
            ax_p.scatter([x[1] for x in sub], [x[2] for x in sub], c=color, marker=marker, s=85, label=label, zorder=3)

        lx_vals = [math.log10(c) for c in costs]
        lx_min, lx_max = min(lx_vals), max(lx_vals)
        lx_span = max(1.0, lx_max - lx_min)
        x_lim_log = (lx_min - 0.14 * lx_span, lx_max + 0.24 * lx_span)
        ax_p.set_xlim(10.0 ** x_lim_log[0], 10.0 ** x_lim_log[1])

        q_min, q_max = min(quals), max(quals)
        q_span = max(0.05, q_max - q_min)
        y_lim = (max(-0.02, q_min - 0.16 * q_span), min(1.03, q_max + 0.15 * q_span))
        ax_p.set_ylim(*y_lim)

        pt_norms = [
            ((math.log10(c) - x_lim_log[0]) / (x_lim_log[1] - x_lim_log[0]), (q - y_lim[0]) / (y_lim[1] - y_lim[0]))
            for c, q in zip(costs, quals)
        ]
        legend_loc, legend_rect = _pick_pareto_legend_loc(pt_norms)

        if pareto_idx:
            pts = sorted([(costs[i], quals[i]) for i in pareto_idx], key=lambda item: item[0])
            ax_p.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                color=_COLOR_PARETO,
                linestyle="--",
                linewidth=1.6,
                marker="*",
                markersize=10,
                label="Pareto Frontier",
                zorder=4,
            )
        ax_p.set_xscale("log")
        ax_p.set_xlabel("C99 Q15 Total Latency ms/img (log scale)")
        ax_p.set_ylabel("Q15 F1 Score")
        ax_p.set_title("Quality vs Embedded Q15 Latency Pareto")
        ax_p.grid(True, which="both", linestyle=":", alpha=0.5)
        ax_p.legend(fontsize=8, loc=legend_loc)
        annotate_points_non_overlapping(
            ax_p,
            [m.name for m in timed],
            costs,
            quals,
            xscale="log",
            fontsize=8.0,
            reserved_boxes_axes=[legend_rect],
            zorder=5,
        )

        # Latency breakdown T_selector + T_model for f32 and q15
        timed_sorted = sorted(timed, key=lambda m: float(m.bench_q15_total_ms or 0.0))
        t_names = [m.name for m in timed_sorted]
        y_t = np.arange(len(timed_sorted), dtype=np.float64)
        sel_q15 = np.array([float(m.bench_q15_selector_ms or 0.0) for m in timed_sorted])
        mod_q15 = np.array([float(m.bench_q15_model_ms or 0.0) for m in timed_sorted])
        tot_q15 = np.array([max(1e-4, float(m.bench_q15_total_ms or 0.0)) for m in timed_sorted])
        tot_f32 = np.array([max(1e-4, float(m.bench_f32_total_ms or 0.0)) for m in timed_sorted])

        ax_l.barh(y_t + 0.18, tot_f32, height=0.34, color=_COLOR_F32, alpha=0.75, label="Total C float32 (ms/img)")
        ax_l.barh(y_t - 0.18, np.maximum(sel_q15, 1e-4), height=0.34, color=_MDPI_PAL[5], label="Q15 Selector (T_selector)")
        ax_l.barh(
            y_t - 0.18,
            np.maximum(mod_q15, 0.0),
            left=np.maximum(sel_q15, 0.0),
            height=0.34,
            color=_COLOR_Q15,
            label="Q15 Classifier/CNN (T_model)",
        )
        ax_l.set_xscale("log")
        ax_l.set_xlim(1e-3, max(float(np.max(tot_f32)), float(np.max(tot_q15))) * 65.0)
        for idx, (qf, qt) in enumerate(zip(tot_f32, tot_q15)):
            ax_l.text(max(qf, qt) * 1.18, y_t[idx], f"f32={qf:.3f} | q15={qt:.3f} ms", va="center", fontsize=7.8)
        ax_l.set_yticks(y_t)
        ax_l.set_yticklabels(t_names, fontsize=9)
        ax_l.set_xlabel("Desktop Single-Thread Latency ms/img (log scale)")
        ax_l.set_title("C99 Latency Breakdown (T_total = T_selector + T_model)")
        ax_l.grid(True, which="both", axis="x", linestyle=":", alpha=0.5)
        ax_l.legend(fontsize=7.6, loc="lower right")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _plot_fig6_x94_descriptor(
    insights: DescriptorFeatureInsights,
    target_tag: str,
    out_path: Path,
    dpi: int,
) -> Path:
    """Plot Normal vs Defect 8x8 spatial entropy grids, asymmetry bars, effect sizes, and Q0.15 error."""
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6), dpi=dpi)
    fig.suptitle(
        f"Multiscale Entropy & Asymmetry Descriptor (x94) Summary — {target_tag} "
        f"(Normal={insights.n_neg}, Defect={insights.n_pos})",
        fontsize=13,
        fontweight="bold",
    )

    vmax = float(max(np.max(insights.mean_neg_8x8), np.max(insights.mean_pos_8x8), 1e-6))
    vmin = float(min(np.min(insights.mean_neg_8x8), np.min(insights.mean_pos_8x8)))

    im0 = axes[0, 0].imshow(insights.mean_neg_8x8, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Mean 8x8 Entropy: Normal (y=0)")
    axes[0, 0].set_xlabel("Grid Column")
    axes[0, 0].set_ylabel("Grid Row")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(insights.mean_pos_8x8, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Mean 8x8 Entropy: Defect (y=1)")
    axes[0, 1].set_xlabel("Grid Column")
    axes[0, 1].set_ylabel("Grid Row")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    diff_8x8 = insights.mean_pos_8x8 - insights.mean_neg_8x8
    dlim = float(max(np.max(np.abs(diff_8x8)), 1e-6))
    im2 = axes[0, 2].imshow(diff_8x8, cmap="coolwarm", vmin=-dlim, vmax=dlim)
    axes[0, 2].set_title("Spatial Entropy Shift (Defect - Normal)")
    axes[0, 2].set_xlabel("Grid Column")
    axes[0, 2].set_ylabel("Grid Row")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Bottom-left: Multiscale asymmetry comparison
    ax_as = axes[1, 0]
    if insights.asym_names:
        x_a = np.arange(len(insights.asym_names), dtype=np.float64)
        w_a = 0.36
        ax_as.bar(x_a - w_a / 2.0, insights.asym_mean_neg, width=w_a, color=_COLOR_NEG, label="Normal (y=0)")
        ax_as.bar(x_a + w_a / 2.0, insights.asym_mean_pos, width=w_a, color=_COLOR_POS, label="Defect (y=1)")
        short_labels = [n.replace("asym_", "") for n in insights.asym_names]
        ax_as.set_xticks(x_a)
        ax_as.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8.5)
        ax_as.set_ylabel("Mean Spatial Entropy Asymmetry")
        ax_as.set_title("Multiscale Entropy Asymmetry by Class")
        ax_as.grid(axis="y", linestyle=":", alpha=0.5)
        ax_as.legend(fontsize=8)
    else:
        ax_as.set_axis_off()

    # Bottom-center: Top-12 Cohen's d effect size features
    ax_ef = axes[1, 1]
    y_e = np.arange(len(insights.top_effect_names), dtype=np.float64)
    vals_e = insights.top_effect_values
    colors_e = [_COLOR_POS if v >= 0 else _COLOR_NEG for v in vals_e]
    ax_ef.barh(y_e[::-1], vals_e, color=colors_e)
    ax_ef.axvline(0.0, color="#333333", linewidth=1.0)
    ax_ef.set_yticks(y_e[::-1])
    ax_ef.set_yticklabels(insights.top_effect_names, fontsize=8.5)
    ax_ef.set_xlabel("Cohen's d Effect Size (Defect - Normal)")
    ax_ef.set_title("Top-12 Most Discriminative x94 Features")
    ax_ef.grid(axis="x", linestyle=":", alpha=0.5)

    # Bottom-right: Q0.15 & log2_mantissa quantization error across features
    ax_q = axes[1, 2]
    if float(np.max(insights.q15_abs_err_max)) > 0.0:
        feat_idx = np.arange(len(insights.feature_names), dtype=np.int64)
        ax_q.plot(feat_idx, np.maximum(insights.q15_abs_err_mean, 1e-8), color=_COLOR_F32, linewidth=1.5, label="Mean |Q15 - float64|")
        ax_q.plot(feat_idx, np.maximum(insights.q15_abs_err_max, 1e-8), color=_COLOR_Q15, linewidth=1.1, alpha=0.8, label="Max |Q15 - float64|")
        ax_q.set_yscale("log")
        ax_q.set_xlabel("x94 Feature Index (0..84 Entropy, 85..93 Asymmetry)")
        ax_q.set_ylabel("Absolute Error (log scale)")
        ax_q.set_title("Q0.15 + LUT Quantization Residual per Feature")
        ax_q.grid(True, which="both", linestyle=":", alpha=0.5)
        ax_q.legend(fontsize=8)
    else:
        ax_q.text(0.5, 0.5, "golden_x94_q15.csv not present", ha="center", va="center", transform=ax_q.transAxes)
        ax_q.set_axis_off()

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def generate_c_pipeline_visualizations(cfg: CImpConfig) -> VisualizationReport:
    """Load target CSV/JSON artifacts and render diagnostic figures + human-readable summary."""
    apply_style("mdpi", palette="editorial")
    bundle = load_target_bundle(cfg)
    out_dir = cfg.visualization_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = cfg.visualization.dpi

    generated_figs: list[Path] = [
        _plot_fig1_classification(bundle, out_dir / "fig1_classification_f32_vs_q15.png", dpi=dpi),
        _plot_fig2_score_separation(bundle, out_dir / "fig2_score_separation_and_quantization.png", dpi=dpi),
        _plot_fig3_confusion_matrices(bundle, out_dir / "fig3_confusion_matrices_q15.png", dpi=dpi),
    ]

    fig4 = _plot_fig4_c_parity(bundle, cfg, out_dir / "fig4_c_parity_and_verification.png", dpi=dpi)
    if fig4 is not None:
        generated_figs.append(fig4)

    fig5 = _plot_fig5_memory_and_latency(bundle, out_dir / "fig5_embedded_memory_and_latency_pareto.png", dpi=dpi)
    if fig5 is not None:
        generated_figs.append(fig5)

    if bundle.descriptor_insights is not None:
        generated_figs.append(
            _plot_fig6_x94_descriptor(
                bundle.descriptor_insights,
                bundle.target_tag,
                out_dir / "fig6_x94_descriptor_insights.png",
                dpi=dpi,
            )
        )

    csv_path, md_path = write_summary_tables(bundle, out_dir, generated_figs)
    return VisualizationReport(
        target_tag=bundle.target_tag,
        output_dir=out_dir,
        summary_csv_path=csv_path,
        summary_md_path=md_path,
        figure_paths=tuple(generated_figs),
    )
