"""Publication-ready Matplotlib figure generators for x94 descriptor evaluation and explainability.

Renders four diagnostic figures per experiment run:
1. Confusion Matrix, ROC Curve, and Precision-Recall Curve.
2. Descriptor Subset Ablation (x85_only vs x9_only vs x94_full and per-scale E_1x1..E_8x8).
3. Explainability Breakdown (Top-K features, Total Share % by scale, and Mean Importance per feature).
4. 2D Spatial Importance Heatmaps across 2x2, 4x4, and 8x8 entropy grids.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.models.descriptor_classifier import ClassificationMetrics, ExplainabilityReport

COLOR_ENTROPY = "#1f618d"
COLOR_ASYMMETRY = "#d35400"
COLOR_FULL = "#117a65"


def plot_confusion_and_curves(
    metrics: ClassificationMetrics,
    title_prefix: str,
    out_path: Path,
    dpi: int,
) -> Path:
    """Render 1x3 figure with Confusion Matrix, ROC Curve, and Precision-Recall Curve."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel 1: Confusion Matrix
    ax_cm = axes[0]
    cm = np.asarray(metrics.confusion_matrix, dtype=np.int64)
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_pct = (cm / row_sums) * 100.0

    im = ax_cm.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=100.0)
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04, label="Row %")
    classes = ["good (0)", "defect (1)"]
    ax_cm.set_xticks([0, 1], labels=classes)
    ax_cm.set_yticks([0, 1], labels=classes)
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.set_title(
        f"Confusion Matrix\nAcc={metrics.accuracy:.3f} | BalAcc={metrics.balanced_accuracy:.3f} | F1={metrics.f1:.3f}"
    )

    tag_labels = [["TN", "FP"], ["FN", "TP"]]
    for r in range(2):
        for c in range(2):
            text_color = "white" if cm_pct[r, c] > 55.0 else "#1b2631"
            ax_cm.text(
                c,
                r,
                f"{tag_labels[r][c]}\n{cm[r, c]} ({cm_pct[r, c]:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
                fontsize=10,
            )

    # Panel 2: ROC Curve
    ax_roc = axes[1]
    ax_roc.plot(
        metrics.fpr,
        metrics.tpr,
        color=COLOR_ENTROPY,
        lw=2.2,
        label=f"ROC-AUC = {metrics.roc_auc:.4f}",
    )
    ax_roc.plot([0.0, 1.0], [0.0, 1.0], color="#7f8c8d", ls="--", lw=1.2, label="Random (0.50)")
    ax_roc.set_xlim(-0.02, 1.02)
    ax_roc.set_ylim(-0.02, 1.05)
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (TPR / Recall)")
    ax_roc.set_title("ROC Curve")
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    ax_roc.legend(loc="lower right")

    # Panel 3: Precision-Recall Curve
    ax_pr = axes[2]
    ax_pr.plot(
        metrics.pr_recall,
        metrics.pr_precision,
        color=COLOR_ASYMMETRY,
        lw=2.2,
        label=f"PR-AUC (AP) = {metrics.pr_auc:.4f}",
    )
    ax_pr.set_xlim(-0.02, 1.02)
    ax_pr.set_ylim(-0.02, 1.05)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(
        f"Precision-Recall Curve\nPrec={metrics.precision:.3f} | Rec={metrics.recall:.3f}"
    )
    ax_pr.grid(True, linestyle=":", alpha=0.6)
    ax_pr.legend(loc="lower left")

    fig.suptitle(
        f"{title_prefix} — {metrics.classifier_name.upper()} ({metrics.subset_name}, {metrics.n_features} features)",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_subset_ablation(
    ablation_metrics: Sequence[ClassificationMetrics],
    title_prefix: str,
    out_path: Path,
    dpi: int,
) -> Path:
    """Render grouped bar chart comparing F1, Balanced Accuracy, and ROC-AUC across feature subsets."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 5.2))

    labels = [f"{m.subset_name}\n({m.n_features}f)" for m in ablation_metrics]
    f1_vals = [m.f1 for m in ablation_metrics]
    bal_vals = [m.balanced_accuracy for m in ablation_metrics]
    auc_vals = [m.roc_auc for m in ablation_metrics]

    x = np.arange(len(labels))
    width = 0.25

    bars_f1 = ax.bar(x - width, f1_vals, width, label="F1-Score (defect)", color=COLOR_FULL)
    bars_bal = ax.bar(x, bal_vals, width, label="Balanced Accuracy", color=COLOR_ENTROPY)
    bars_auc = ax.bar(x + width, auc_vals, width, label="ROC-AUC", color=COLOR_ASYMMETRY)

    for bars in (bars_f1, bars_bal, bars_auc):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    by_sub = {m.subset_name: m for m in ablation_metrics}
    delta_note = ""
    if "x94_full" in by_sub and "x85_only" in by_sub:
        df1 = by_sub["x94_full"].f1 - by_sub["x85_only"].f1
        dauc = by_sub["x94_full"].roc_auc - by_sub["x85_only"].roc_auc
        delta_note = f" | Symmetry Gain (x94 - x85): dF1={df1:+.4f}, dROC-AUC={dauc:+.4f}"

    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0.0, 1.14)
    ax.set_ylabel("Score")
    ax.set_title(
        f"{title_prefix} — Descriptor Subset & Scale Ablation{delta_note}",
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_explainability_breakdown(
    report: ExplainabilityReport,
    title_prefix: str,
    out_path: Path,
    dpi: int,
) -> Path:
    """Render 1x3 figure with Top-K features, Total Share % by scale, and Mean Importance per feature."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # Panel 1: Top-K Individual Features
    ax_top = axes[0]
    top_indices = list(report.top_k_indices)[::-1]
    top_names = [report.feature_names[i] for i in top_indices]
    top_vals = [report.importances[i] * 100.0 for i in top_indices]
    top_colors = [
        COLOR_ASYMMETRY if name.startswith("asym_") else COLOR_ENTROPY for name in top_names
    ]

    ax_top.barh(range(len(top_names)), top_vals, color=top_colors, alpha=0.9)
    ax_top.set_yticks(range(len(top_names)), labels=top_names, fontsize=8.5)
    ax_top.set_xlabel("Relative Importance (%)")
    ax_top.set_title(f"Top-{len(top_names)} Discriminative Features ({report.method})")
    ax_top.grid(axis="x", linestyle=":", alpha=0.6)

    # Panel 2: Total Share (%) by Pyramid Scale (Entropy vs Asymmetry)
    ax_share = axes[1]
    scale_names = [g.group_name for g in report.scale_breakdown]
    scale_shares = [g.share_pct for g in report.scale_breakdown]
    scale_colors = [
        COLOR_ASYMMETRY if name.startswith("asym_") else COLOR_ENTROPY for name in scale_names
    ]
    bars_s = ax_share.bar(scale_names, scale_shares, color=scale_colors, alpha=0.9)
    for bar in bars_s:
        h = bar.get_height()
        ax_share.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fam_summary = " | ".join(
        f"{g.group_name}: {g.share_pct:.1f}% ({g.n_features}f)" for g in report.family_breakdown
    )
    ax_share.set_ylabel("Total Importance Share (%)")
    ax_share.set_title(f"Total Contribution by Scale\n{fam_summary}", fontsize=10)
    ax_share.tick_params(axis="x", rotation=30)
    ax_share.set_ylim(0.0, max(scale_shares + [10.0]) * 1.18)
    ax_share.grid(axis="y", linestyle=":", alpha=0.6)

    # Panel 3: Mean Importance per Feature (Cardinality-Normalized)
    ax_mean = axes[2]
    mean_vals = [g.mean_importance_per_feature * 100.0 for g in report.scale_breakdown]
    bars_m = ax_mean.bar(scale_names, mean_vals, color=scale_colors, alpha=0.9)
    for bar in bars_m:
        h = bar.get_height()
        ax_mean.annotate(
            f"{h:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    kind_summary = " | ".join(
        f"{g.group_name}: {g.share_pct:.1f}%" for g in report.asymmetry_kind_breakdown
    )
    ax_mean.set_ylabel("Mean Importance per Feature (%)")
    ax_mean.set_title(
        f"Efficiency per Feature (Normalized by Group Size)\nAsym Axes: {kind_summary}",
        fontsize=10,
    )
    ax_mean.tick_params(axis="x", rotation=30)
    ax_mean.set_ylim(0.0, max(mean_vals + [1.0]) * 1.18)
    ax_mean.grid(axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(
        f"{title_prefix} — Explainability: Entropy (Blue) vs Symmetry/Asymmetry (Orange)",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_spatial_importance_heatmaps(
    report: ExplainabilityReport,
    title_prefix: str,
    out_path: Path,
    dpi: int,
) -> Path:
    """Render 1x3 figure of 2D spatial importance heatmaps for 2x2, 4x4, and 8x8 entropy grids."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scales = sorted(report.spatial_heatmaps.keys())
    n_panels = max(1, len(scales))
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.6))
    if n_panels == 1:
        axes = [axes]

    for ax, g in zip(axes, scales):
        grid_pct = report.spatial_heatmaps[g] * 100.0
        vmax = max(float(grid_pct.max()), 1e-6)
        im = ax.imshow(grid_pct, cmap="YlOrRd", vmin=0.0, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Importance (%)")
        ax.set_xticks(range(g), labels=[f"c{c}" for c in range(g)])
        ax.set_yticks(range(g), labels=[f"r{r}" for r in range(g)])
        peak_r, peak_c = np.unravel_index(np.argmax(grid_pct), grid_pct.shape)
        ax.set_title(
            f"Scale E_{g}x{g} (Sum={grid_pct.sum():.1f}%)\nPeak: r{int(peak_r)}_c{int(peak_c)} ({grid_pct[peak_r, peak_c]:.2f}%)"
        )

        # Annotate cell values on 2x2 and 4x4, and top-5 cells on 8x8
        thresh = 0.55 * vmax
        if g <= 4:
            for r in range(g):
                for c in range(g):
                    val = grid_pct[r, c]
                    ax.text(
                        c,
                        r,
                        f"{val:.2f}%",
                        ha="center",
                        va="center",
                        color="white" if val > thresh else "#1b2631",
                        fontsize=9 if g == 2 else 8,
                        fontweight="bold",
                    )
        else:
            flat_top = np.argsort(grid_pct.ravel())[::-1][:5]
            for flat_idx in flat_top:
                r, c = np.unravel_index(int(flat_idx), grid_pct.shape)
                val = grid_pct[r, c]
                ax.text(
                    c,
                    r,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "#1b2631",
                    fontsize=7,
                    fontweight="bold",
                )

    fig.suptitle(
        f"{title_prefix} — 2D Spatial Entropy Importance Maps",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all_objects_summary(
    summary_rows: Sequence[Mapping[str, object]],
    title_prefix: str,
    out_path: Path,
    dpi: int,
) -> Path:
    """Render multi-object comparison chart for MVTec-AD (F1 for x85_only vs x94_full and x9 share %)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.2), sharex=True)

    objects = [str(r["object_name"]) for r in summary_rows]
    f1_x85 = [float(r["f1_x85_only"]) for r in summary_rows]
    f1_x94 = [float(r["f1_x94_full"]) for r in summary_rows]
    auc_x94 = [float(r["roc_auc_x94_full"]) for r in summary_rows]
    x9_share = [float(r["x9_share_pct"]) for r in summary_rows]

    x = np.arange(len(objects))
    width = 0.28

    ax0 = axes[0]
    ax0.bar(x - width, f1_x85, width, label="F1 (x85 Entropy only)", color=COLOR_ENTROPY)
    ax0.bar(x, f1_x94, width, label="F1 (x94 Full = x85 + x9)", color=COLOR_FULL)
    ax0.bar(x + width, auc_x94, width, label="ROC-AUC (x94 Full)", color="#8e44ad")
    ax0.set_ylim(0.0, 1.18)
    ax0.set_ylabel("Score")
    ax0.set_title(f"{title_prefix} — Per-Object Classification Performance (x85 vs x94)", fontweight="bold")
    ax0.grid(axis="y", linestyle=":", alpha=0.6)
    ax0.legend(loc="upper right", ncol=3)

    ax1 = axes[1]
    bars = ax1.bar(x, x9_share, width=0.5, color=COLOR_ASYMMETRY, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2.0, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.axhline(9.0 / 94.0 * 100.0, color="#2c3e50", ls="--", lw=1.2, label="Uniform expectation (9/94 = 9.6%)")
    ax1.set_xticks(x, labels=objects, rotation=30, ha="right")
    ax1.set_ylabel("x9 Symmetry Share (%)")
    ax1.set_ylim(0.0, max(x9_share + [20.0]) * 1.2)
    ax1.set_title("Total Contribution of the 9 Symmetry/Asymmetry Features (x9) per Object", fontweight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
