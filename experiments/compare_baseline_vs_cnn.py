"""Cross-paradigm comparison between classical ML (x94 descriptors) and lightweight CNNs.

Scans evaluation results from results/classification/ and results/cnn/, aligns metrics,
and enforces fairness (AGENTS.md performance rule) by measuring descriptor extraction
latency so that T_total = T_descriptor + T_model is compared against CNN end-to-end latency.

Manual execution commands (PowerShell from repository 'code/' directory):
    # 1. Compare on KolektorSDD2 (standard):
    .\\.venv\\Scripts\\python.exe -m experiments.compare_baseline_vs_cnn --dataset kolektorsdd2

    # 2. Compare on KolektorSDD2 (inverted):
    .\\.venv\\Scripts\\python.exe -m experiments.compare_baseline_vs_cnn --dataset kolektorsdd2 --inverted

    # 3. Compare on MVTec-AD single object (e.g. bottle):
    .\\.venv\\Scripts\\python.exe -m experiments.compare_baseline_vs_cnn --dataset mvtecad --object bottle

    # 4. Compare on all MVTec-AD objects:
    .\\.venv\\Scripts\\python.exe -m experiments.compare_baseline_vs_cnn --dataset mvtecad
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import configure_logging, logger
from src.features.asymmetry_maps import compute_asymmetry_features
from src.features.entropy_maps import compute_entropy_features


def measure_descriptor_latency_ms(
    image_size: int = 128,
    n_samples: int = 32,
    warmup: int = 5,
) -> float:
    """Measure mean CPU time in ms required to extract the x94 descriptor per 128x128 image.

    This ensures fair total latency accounting: T_total = T_descriptor + T_classifier.
    """
    rng = np.random.default_rng(42)
    sample_images = rng.integers(0, 256, size=(n_samples, image_size, image_size), dtype=np.uint8)
    grid_levels = (1, 2, 4, 8)
    asym_scales = (2, 4, 8)
    bins = 16

    # Warmup
    for _ in range(warmup):
        e_feats = compute_entropy_features(sample_images[:4], grid_levels=grid_levels, bins=bins)
        e_by_scale = {}
        offset = 0
        for g in grid_levels:
            cols = g * g
            e_by_scale[g] = e_feats[:4, offset : offset + cols]
            offset += cols
        _ = compute_asymmetry_features(e_by_scale, scales=asym_scales)

    # Timed run
    t0 = time.perf_counter()
    e_feats = compute_entropy_features(sample_images, grid_levels=grid_levels, bins=bins)
    e_by_scale = {}
    offset = 0
    for g in grid_levels:
        cols = g * g
        e_by_scale[g] = e_feats[:, offset : offset + cols]
        offset += cols
    _ = compute_asymmetry_features(e_by_scale, scales=asym_scales)
    elapsed_s = time.perf_counter() - t0

    return (elapsed_s * 1000.0) / n_samples


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for baseline vs CNN comparison."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.compare_baseline_vs_cnn",
        description="Compare x94 classical ML baseline against lightweight CNN architectures.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kolektorsdd2", "mvtecad"],
        default="kolektorsdd2",
        help="Dataset to compare ('kolektorsdd2' or 'mvtecad').",
    )
    parser.add_argument(
        "--object",
        dest="object_name",
        type=str,
        default=None,
        help="Object name for MVTec-AD (e.g. 'bottle', 'metal_nut', 'tile').",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        default=False,
        help="Compare on inverted variant.",
    )
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=Path("results/classification"),
        help="Base directory for classical classification results.",
    )
    parser.add_argument(
        "--cnn-dir",
        type=Path,
        default=Path("results/cnn"),
        help="Base directory for CNN evaluation results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparison"),
        help="Directory to save comparison table and figures.",
    )
    return parser


def _collect_metrics(
    target_dir: Path, family: str, filter_obj: str | None
) -> list[dict[str, object]]:
    """Scan directory recursively for metrics_summary.json files matching filters."""
    records: list[dict[str, object]] = []
    if not target_dir.is_dir():
        return records

    for summary_path in target_dir.glob("**/metrics_summary.json"):
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        obj_name = data.get("object_name")
        if filter_obj is not None and obj_name != filter_obj:
            continue

        raw_model = (
            data.get("model_name")
            or data.get("classifier")
            or data.get("classifier_name")
            or summary_path.parent.name
        )
        raw_str = str(raw_model).lower()
        if raw_str.startswith("full_"):
            raw_str = raw_str.replace("full_", "")

        if family == "ml":
            model_name = f"{raw_str} (x94)"
        else:
            model_name = raw_str

        metrics = data.get("metrics") or data.get("active_subset_metrics", {})
        timing = data.get("timing") or data.get("timings", {})
        profile = data.get("model_profile", {})

        # Classical baseline records might place timing at root or in timings
        t_train = timing.get(
            "train_elapsed_s",
            timing.get("T_model_train_s", data.get("train_elapsed_s", 0.0)),
        )
        t_inf_ms = timing.get(
            "inference_ms_per_sample",
            timing.get(
                "T_model_inference_s",
                data.get("inference_ms_per_sample", 0.0),
            ),
        )
        params = profile.get(
            "parameters",
            metrics.get("n_features", data.get("n_features", 94 if family == "ml" else 0)),
        )
        flops = profile.get("flops", 0)

        record = {
            "model": model_name,
            "family": family,
            "object_name": obj_name if obj_name else "full",
            "seed": data.get("seed", 42),
            "params": params,
            "flops": flops,
            "accuracy": metrics.get("accuracy", 0.0),
            "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
            "f1": metrics.get("f1", 0.0),
            "roc_auc": metrics.get("roc_auc", 0.0),
            "pr_auc": metrics.get("pr_auc", 0.0),
            "t_train_s": t_train,
            "t_inf_model_ms": t_inf_ms,
        }
        records.append(record)

    return records


def plot_comparison_pareto(
    output_path: Path,
    rows: list[dict[str, object]],
    dataset_name: str,
    object_name: str | None,
    dpi: int = 160,
) -> None:
    """Plot Pareto frontier: F1 vs Total Latency, ROC-AUC vs Total Latency, and F1 vs Params."""
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=dpi)
    obj_str = f" [{object_name}]" if object_name else ""
    fig.suptitle(
        f"Baseline (x94 Classical ML) vs. Lightweight CNNs: {dataset_name}{obj_str}\n"
        "(Fair Latency Accounting: T_total = T_descriptor + T_model for ML; Raw Pixels for CNN)",
        fontsize=12,
        fontweight="bold",
    )

    ml_rows = [r for r in rows if r["family"] == "ml"]
    cnn_rows = [r for r in rows if r["family"] == "cnn"]

    # 1. F1 vs T_total ms/sample
    ax1 = axes[0]
    if ml_rows:
        ax1.scatter(
            [r["t_total_ms"] for r in ml_rows],
            [r["f1"] for r in ml_rows],
            color="#1f77b4",
            marker="o",
            s=80,
            label="Classical ML (x94)",
            zorder=3,
        )
        for r in ml_rows:
            ax1.annotate(
                f"{r['model']}",
                (r["t_total_ms"], r["f1"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    if cnn_rows:
        ax1.scatter(
            [r["t_total_ms"] for r in cnn_rows],
            [r["f1"] for r in cnn_rows],
            color="#d62728",
            marker="^",
            s=80,
            label="Lightweight CNNs",
            zorder=3,
        )
        for r in cnn_rows:
            ax1.annotate(
                f"{r['model']}",
                (r["t_total_ms"], r["f1"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )

    ax1.set_xlabel("Total Inference Latency (ms/sample)", fontsize=10)
    ax1.set_ylabel("F1 Score", fontsize=10)
    ax1.set_title("F1 vs. End-to-End Latency", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. ROC-AUC vs T_total ms/sample
    ax2 = axes[1]
    if ml_rows:
        ax2.scatter(
            [r["t_total_ms"] for r in ml_rows],
            [r["roc_auc"] for r in ml_rows],
            color="#1f77b4",
            marker="o",
            s=80,
            label="Classical ML (x94)",
            zorder=3,
        )
    if cnn_rows:
        ax2.scatter(
            [r["t_total_ms"] for r in cnn_rows],
            [r["roc_auc"] for r in cnn_rows],
            color="#d62728",
            marker="^",
            s=80,
            label="Lightweight CNNs",
            zorder=3,
        )
    ax2.set_xlabel("Total Inference Latency (ms/sample)", fontsize=10)
    ax2.set_ylabel("ROC-AUC", fontsize=10)
    ax2.set_title("ROC-AUC vs. End-to-End Latency", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. F1 vs Parameters (log scale)
    ax3 = axes[2]
    if ml_rows:
        ax3.scatter(
            [max(1, int(r["params"])) for r in ml_rows],
            [r["f1"] for r in ml_rows],
            color="#1f77b4",
            marker="o",
            s=80,
            label="Classical ML (x94)",
            zorder=3,
        )
    if cnn_rows:
        ax3.scatter(
            [max(1, int(r["params"])) for r in cnn_rows],
            [r["f1"] for r in cnn_rows],
            color="#d62728",
            marker="^",
            s=80,
            label="Lightweight CNNs",
            zorder=3,
        )
    ax3.set_xscale("log")
    ax3.set_xlabel("Model Parameters (log scale)", fontsize=10)
    ax3.set_ylabel("F1 Score", fontsize=10)
    ax3.set_title("F1 vs. Parameter Footprint", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """Execute cross-paradigm comparison pipeline.

    Args:
        argv: Command-line arguments.
        project_root: Repository root directory.

    Returns:
        Exit code (0 on success).
    """
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    ds_key = args.dataset.lower()
    tag_name = ds_key if ds_key == "kolektorsdd2" else "mvtec-ad"
    if args.inverted:
        tag_name = f"{tag_name}-inverted"

    # 1. Measure descriptor extraction latency for fair total timing
    logger.info("Benchmarking x94 descriptor extraction latency per 128x128 image...")
    t_desc_ms = measure_descriptor_latency_ms(image_size=128)
    logger.info("Measured T_descriptor = %.4f ms/sample on host CPU", t_desc_ms)

    # 2. Collect classification baseline results
    cls_tag_dir = (project_root / args.classification_dir / tag_name).resolve()
    ml_records = _collect_metrics(cls_tag_dir, family="ml", filter_obj=args.object_name)

    # 3. Collect CNN results
    cnn_tag_dir = (project_root / args.cnn_dir / tag_name).resolve()
    cnn_records = _collect_metrics(cnn_tag_dir, family="cnn", filter_obj=args.object_name)

    all_records = ml_records + cnn_records
    if not all_records:
        logger.warning(
            "No evaluation records found under %s or %s matching dataset '%s' (object=%s)",
            cls_tag_dir,
            cnn_tag_dir,
            tag_name,
            args.object_name,
        )
        return 0

    # 4. Integrate fair latency: T_total = T_descriptor + T_model for ML; T_inf for CNN
    for r in all_records:
        if r["family"] == "ml":
            r["t_descriptor_ms"] = t_desc_ms
            r["t_total_ms"] = float(r["t_inf_model_ms"]) + t_desc_ms
        else:
            r["t_descriptor_ms"] = 0.0
            r["t_total_ms"] = float(r["t_inf_model_ms"])

    # 5. Export combined comparison CSV
    out_dir = (project_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_tag = f"_{args.object_name}" if args.object_name else ""
    csv_file = out_dir / f"{tag_name}{obj_tag}_baseline_vs_cnn.csv"
    png_file = out_dir / f"{tag_name}{obj_tag}_baseline_vs_cnn.png"

    fieldnames = [
        "model",
        "family",
        "object_name",
        "seed",
        "params",
        "flops",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "pr_auc",
        "t_train_s",
        "t_inf_model_ms",
        "t_descriptor_ms",
        "t_total_ms",
    ]

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    # 6. Plot comparison figure
    plot_comparison_pareto(
        output_path=png_file,
        rows=all_records,
        dataset_name=tag_name,
        object_name=args.object_name,
    )

    logger.info(
        "Saved cross-paradigm comparison: %d models -> %s and %s",
        len(all_records),
        csv_file,
        png_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
