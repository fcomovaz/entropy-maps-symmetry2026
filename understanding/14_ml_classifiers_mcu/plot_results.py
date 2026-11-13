"""Generate publication-quality matplotlib visualizations for Lab 14 MCU classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, LAB_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from configs.logging import configure_logging, logger
from config_loader import McuLabConfig, load_lab_config

GROUP_STYLE: dict[str, tuple[str, str, str]] = {
    "src_baseline": ("#6c757d", "s", "src/ Baseline"),
    "linear_dense": ("#1f77b4", "o", "Folded Linear"),
    "linear_folded": ("#1f77b4", "o", "Folded Linear"),
    "linear_sparse": ("#17becf", "P", "Sparse L1 Linear"),
    "quadratic_diag": ("#9467bd", "D", "Folded Diag Quad (GNB)"),
    "tree_core": ("#ff7f0e", "^", "Flat C99 Tree / GBDT"),
    "mlp_folded": ("#d62728", "*", "Folded ReLU MLP"),
}


def _group_color_marker(group: str, model_family: str | None = None) -> tuple[str, str, str]:
    """Resolve consistent Okabe-Ito-inspired color and marker style by model group."""
    if model_family == "linear_dense" and group == "src_baseline":
        return ("#2b2d42", "s", "src/ Baseline")
    return GROUP_STYLE.get(group, ("#333333", "o", group))


def _plot_pareto_flash_vs_quality(df_b1: pd.DataFrame, out_path: Path, dpi: int) -> None:
    """Plot Flash memory footprint (KB) vs ROC-AUC across primary datasets."""
    datasets = [ds for ds in df_b1["dataset_tag"].unique() if not str(ds).endswith("-inverted")]
    if not datasets:
        return
    fig, axes = plt.subplots(2, len(datasets), figsize=(5.2 * len(datasets), 8.2), sharex=True)
    if len(datasets) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    label_offsets: dict[str, tuple[int, int]] = {
        "lr_dot": (5, 4),
        "lr_l1_dot": (5, -9),
        "lda_shrink_dot": (-38, -10),
        "lsvc_dot": (-28, 5),
        "dt": (5, 4),
        "dt_core": (5, -8),
        "gnb_quad": (5, 4),
        "gbdt_tiny_core": (5, 4),
        "mlp_16_dot": (-16, -11),
        "mlp_16_skip_dot": (5, 5),
        "mlp_32_16_dot": (5, 4),
        "rf": (-14, 5),
    }

    metrics = [("roc_auc", "ROC-AUC"), ("f1", "F1 Score")]
    for row_idx, (metric_col, metric_label) in enumerate(metrics):
        for col_idx, ds in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            sub = df_b1[df_b1["dataset_tag"] == ds].copy()
            seen_labels: set[str] = set()
            for _, r in sub.iterrows():
                color, marker, lbl = _group_color_marker(str(r["group"]), str(r["model_family"]))
                show_lbl = lbl if (row_idx == 0 and col_idx == 0 and lbl not in seen_labels) else None
                if show_lbl:
                    seen_labels.add(lbl)
                size = 150 if r["group"] == "mlp_folded" else 75
                x_val = max(0.05, float(r["flash_kb_f32"]))
                y_val = float(r[metric_col])
                ax.scatter(
                    x_val,
                    y_val,
                    c=color,
                    marker=marker,
                    s=size,
                    edgecolors="black",
                    linewidths=0.6,
                    alpha=0.92,
                    label=show_lbl,
                    zorder=4,
                )
                m_key = str(r["model"])
                ax.annotate(
                    m_key.replace("_dot", "").replace("_core", ""),
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=label_offsets.get(m_key, (4, 4)),
                    fontsize=7.3,
                    alpha=0.9,
                )
            ax.set_xscale("log")
            ax.set_xlim(0.12, float(sub["flash_kb_f32"].max()) * 2.2)
            ax.axvline(1.0, color="#aaaaaa", linestyle=":", linewidth=0.9)
            ax.axvline(16.0, color="#aaaaaa", linestyle="--", linewidth=0.9)
            if row_idx == 0:
                ax.set_title(ds, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            if row_idx == 1:
                ax.set_xlabel("Flash Footprint (KB, float32, log scale)", fontsize=9.5)
            ax.grid(True, which="both", linestyle=":", alpha=0.35)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(5, len(labels)),
            frameon=True,
            fontsize=8.5,
            bbox_to_anchor=(0.5, 0.995),
        )
    fig.suptitle(
        "MCU Flash Footprint vs Predictive Quality (x94, seed=42)",
        fontsize=12.5,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_latency_breakdown(df_b1: pd.DataFrame, out_path: Path, dpi: int) -> None:
    """Plot T_desc vs T_model (B=1, threads=1) plus native C99 kernel latency."""
    sub = df_b1[df_b1["dataset_tag"] == "kolektorsdd2"].copy()
    if sub.empty:
        sub = df_b1[~df_b1["inverted"].astype(bool)].copy()
        first_ds = str(sub["dataset_tag"].iloc[0])
        sub = sub[sub["dataset_tag"] == first_ds].copy()
    else:
        first_ds = "kolektorsdd2"

    sub = sub.sort_values("t_total_ms_est", ascending=True)
    y_pos = np.arange(len(sub))

    fig, (ax_e2e, ax_c99) = plt.subplots(1, 2, figsize=(13.5, 6.2))

    t_desc = sub["t_descriptor_ms_median"].to_numpy(dtype=np.float64)
    t_mod = sub["t_model_ms_median"].to_numpy(dtype=np.float64)
    labels = sub["model"].astype(str).tolist()
    totals = t_desc + t_mod
    non_rf = totals[sub["model"].astype(str).to_numpy() != "rf"]
    x_cap = float(np.max(non_rf) * 1.85) if non_rf.size else float(np.max(totals) * 1.50)

    t_mod_plot = np.minimum(t_mod, np.maximum(0.0, x_cap * 0.94 - t_desc))
    ax_e2e.barh(y_pos, t_desc, color="#457b9d", alpha=0.88, label="T_desc (x94 c_single_pass)")
    ax_e2e.barh(y_pos, t_mod_plot, left=t_desc, color="#e63946", alpha=0.9, label="T_model (Python/NumPy or sklearn)")
    ax_e2e.set_yticks(y_pos)
    ax_e2e.set_yticklabels(labels, fontsize=8.5)
    ax_e2e.set_xlim(0.0, x_cap)
    ax_e2e.set_xlabel("End-to-End Pipeline Latency T_total = T_desc + T_model (ms, linear; broken bar for rf)", fontsize=9.2)
    ax_e2e.set_title(f"End-to-End CPU Latency (threads=1, B=1, {first_ds})", fontsize=10.5, fontweight="bold")
    ax_e2e.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax_e2e.legend(loc="lower right", fontsize=8.2, framealpha=0.95)
    for idx, (td, tm) in enumerate(zip(t_desc, t_mod)):
        total = td + tm
        if total > x_cap * 0.94:
            ax_e2e.text(x_cap * 0.92, idx, f"// {total:.2f} ms (off-scale)", va="center", ha="right", fontsize=7.5, color="white", fontweight="bold")
        else:
            ax_e2e.text(total + x_cap * 0.015, idx, f"{total:.3f} ms", va="center", fontsize=7.5)

    c_sub = sub[sub["t_c99_kernel_us_per_sample"].notna()].copy()
    if not c_sub.empty:
        c_sub = c_sub.sort_values("t_c99_kernel_us_per_sample", ascending=True)
        cy = np.arange(len(c_sub))
        vals_us = c_sub["t_c99_kernel_us_per_sample"].to_numpy(dtype=np.float64)
        colors = [_group_color_marker(str(g), str(f))[0] for g, f in zip(c_sub["group"], c_sub["model_family"])]
        ax_c99.barh(cy, vals_us, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)
        ax_c99.set_yticks(cy)
        ax_c99.set_yticklabels(c_sub["model"].astype(str).tolist(), fontsize=8.5)
        ax_c99.set_xlim(0.0, float(np.max(vals_us)) * 1.24)
        ax_c99.set_xlabel("Pure C99 Classifier Core Latency (microseconds / sample)", fontsize=9.5)
        ax_c99.set_title("Native C99 Kernel Execution (float32, 0 malloc)", fontsize=10.5, fontweight="bold")
        ax_c99.grid(True, axis="x", linestyle=":", alpha=0.4)
        for idx, v in enumerate(vals_us):
            ax_c99.text(v + max(vals_us) * 0.015, idx, f"{v:.2f} us", va="center", fontsize=8)
    else:
        ax_c99.text(0.5, 0.5, "C99 DLL unavailable (--skip-native)", ha="center", va="center")
        ax_c99.set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_mvtec15_boxplot(df_sweep: pd.DataFrame, out_path: Path, dpi: int) -> None:
    """Plot cross-object generalization distributions across all 15 MVTec-AD categories."""
    mv = df_sweep[df_sweep["is_mvtec"].astype(bool)].copy()
    if mv.empty:
        return

    order = (
        mv.groupby("model")["roc_auc"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )
    fig, (ax_auc, ax_f1) = plt.subplots(1, 2, figsize=(13.5, 6.0), sharey=True)

    for ax, col, title in [
        (ax_auc, "roc_auc", "ROC-AUC Across 15 MVTec-AD Objects"),
        (ax_f1, "f1", "F1 Score Across 15 MVTec-AD Objects"),
    ]:
        data = [mv[mv["model"] == m][col].dropna().to_numpy(dtype=np.float64) for m in order]
        bp = ax.boxplot(
            data,
            vert=False,
            tick_labels=order,
            patch_artist=True,
            widths=0.6,
            showfliers=False,
        )
        for idx, (box, m_name) in enumerate(zip(bp["boxes"], order)):
            grp = str(mv[mv["model"] == m_name]["group"].iloc[0])
            c, _, _ = _group_color_marker(grp)
            box.set_facecolor(c)
            box.set_alpha(0.45)
            vals = data[idx]
            jitter = np.linspace(-0.15, 0.15, num=max(1, vals.size))
            ax.scatter(vals, np.full_like(vals, idx + 1) + jitter, color=c, s=16, alpha=0.85, zorder=3)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel(col.upper(), fontsize=9.5)
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    fig.suptitle(
        "MVTec-AD All-15 Object Generalization (x94, seed=42, ordered by median ROC-AUC)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_quantization_and_sparsity(
    df_b1: pd.DataFrame,
    df_sweep: pd.DataFrame | None,
    out_path: Path,
    dpi: int,
) -> None:
    """Visualize float32 quantization fidelity and L1 sparse feature reduction across datasets."""
    fig, (ax_q, ax_sp) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    q_sub = df_b1[df_b1["parity_max_abs_diff_f32"].notna()].copy()
    q_sub = q_sub[~q_sub["model"].isin(["dt", "rf"])]
    if not q_sub.empty:
        agg = q_sub.groupby("model")[["parity_max_abs_diff_f32", "pred_agreement_f32"]].max()
        agg_min_agree = q_sub.groupby("model")["pred_agreement_f32"].min()
        models = agg.index.tolist()
        y_idx = np.arange(len(models))
        deltas = np.maximum(1e-9, agg["parity_max_abs_diff_f32"].to_numpy(dtype=np.float64))
        ax_q.barh(y_idx, deltas, color="#2a9d8f", edgecolor="black", linewidth=0.5, alpha=0.88)
        ax_q.set_xscale("log")
        ax_q.set_xlim(3e-10, float(np.max(deltas)) * 45.0)
        ax_q.set_yticks(y_idx)
        ax_q.set_yticklabels(models, fontsize=8.5)
        ax_q.set_xlabel("Worst-Case |score_ref - score_f32| Across Primary Datasets (log)", fontsize=9)
        ax_q.set_title("Float32 / C99 Numerical Parity & Label Agreement", fontsize=10.5, fontweight="bold")
        ax_q.grid(True, axis="x", linestyle=":", alpha=0.4)
        for idx, m_name in enumerate(models):
            agree = float(agg_min_agree.loc[m_name]) * 100.0
            ax_q.text(deltas[idx] * 1.15, idx, f"agree={agree:.2f}%", va="center", fontsize=7.5)

    if df_sweep is not None and not df_sweep.empty:
        l1 = df_sweep[df_sweep["model"] == "lr_l1_dot"].sort_values("n_active_features")
        if not l1.empty:
            y_sp = np.arange(len(l1))
            active = l1["n_active_features"].to_numpy(dtype=np.float64)
            ax_sp.barh(y_sp, active, color="#17becf", edgecolor="black", linewidth=0.5, alpha=0.9)
            ax_sp.axvline(94, color="#d62728", linestyle="--", linewidth=1.2, label="Full x94 (94 features)")
            ax_sp.set_yticks(y_sp)
            ax_sp.set_yticklabels(l1["dataset_tag"].astype(str).tolist(), fontsize=8)
            ax_sp.set_xlim(0, 102)
            ax_sp.set_xlabel("Nonzero L1 Weights Kept at Inference (out of 94)", fontsize=9)
            ax_sp.set_title("L1 Logistic Regression Feature Sparsity per Dataset", fontsize=10.5, fontweight="bold")
            ax_sp.legend(loc="lower right", fontsize=8)
            ax_sp.grid(True, axis="x", linestyle=":", alpha=0.4)
            for idx, v in enumerate(active):
                ax_sp.text(v + 1.0, idx, f"{int(v)}/94", va="center", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(cfg: McuLabConfig) -> list[Path]:
    """Read Lab 14 benchmark CSV outputs and write matplotlib figures."""
    out_dir = cfg.experiment.output_dir
    primary_csv = out_dir / "benchmark_primary_models.csv"
    if not primary_csv.is_file():
        raise FileNotFoundError(
            f"Benchmark results CSV not found: {primary_csv}. Run benchmark_classifiers.py first."
        )

    df_primary = pd.read_csv(primary_csv)
    df_b1 = df_primary[df_primary["batch_size"] == 1].copy()

    sweep_csv = out_dir / "all_objects_quality.csv"
    df_sweep = pd.read_csv(sweep_csv) if sweep_csv.is_file() else None

    generated: list[Path] = []
    p1 = out_dir / "fig1_pareto_flash_vs_quality.png"
    _plot_pareto_flash_vs_quality(df_b1, p1, dpi=cfg.experiment.dpi)
    generated.append(p1)

    p2 = out_dir / "fig2_latency_breakdown_b1.png"
    _plot_latency_breakdown(df_b1, p2, dpi=cfg.experiment.dpi)
    generated.append(p2)

    if df_sweep is not None:
        p3 = out_dir / "fig3_mvtec15_cross_object.png"
        _plot_mvtec15_boxplot(df_sweep, p3, dpi=cfg.experiment.dpi)
        generated.append(p3)

    p4 = out_dir / "fig4_quantization_and_sparsity.png"
    _plot_quantization_and_sparsity(df_b1, df_sweep, p4, dpi=cfg.experiment.dpi)
    generated.append(p4)

    for p in generated:
        logger.info("Saved figure: %s", p)
    return generated


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for Lab 14 plotting."""
    parser = argparse.ArgumentParser(description="Generate Lab 14 MCU classifier matplotlib plots.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/understanding/ml_classifiers_mcu.toml"),
        help="Path to Lab 14 TOML configuration.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Lab 14 plotting."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg = load_lab_config(args.config, PROJECT_ROOT)
        generate_all_plots(cfg)
        return 0
    except (OSError, KeyError, ValueError, FileNotFoundError) as exc:
        logger.error("Lab 14 plotting failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
