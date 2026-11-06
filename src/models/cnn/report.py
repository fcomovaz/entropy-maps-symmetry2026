"""Report generation, metrics serialization, training history export, and diagnostic plotting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.descriptor_classifier import ClassificationMetrics
from .benchmark import BenchmarkRecord
from .config import CnnConfig
from .data import CnnDatasetPartition
from .evaluate import ThresholdCalibrationInfo
from .train import TrainResult


def _format_float(val: float, precision: int = 6) -> float:
    """Round float to fixed precision for JSON serialization."""
    return round(float(val), precision)


def save_metrics_summary(
    output_dir: Path,
    cfg: CnnConfig,
    model_name: str,
    metrics: ClassificationMetrics,
    train_res: TrainResult,
    benchmarks: Sequence[BenchmarkRecord],
    partition: CnnDatasetPartition,
    seed: int,
    th_info: ThresholdCalibrationInfo | None = None,
) -> Path:
    """Save structured JSON metrics summary including threshold calibration details.

    Args:
        output_dir: Destination directory.
        cfg: Validated CnnConfig instance.
        model_name: Architecture name.
        metrics: Evaluated ClassificationMetrics.
        train_res: Training run results and timing.
        benchmarks: Benchmarking records across devices and batch sizes.
        partition: Partition with dataset counts.
        seed: Random seed of the evaluation.
        th_info: Optional ThresholdCalibrationInfo record.

    Returns:
        Path to written metrics_summary.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "metrics_summary.json"

    bench_list = [
        {
            "device": b.device,
            "batch_size": b.batch_size,
            "median_ms": _format_float(b.median_ms, 4),
            "iqr_ms": _format_float(b.iqr_ms, 4),
            "mean_ms": _format_float(b.mean_ms, 4),
            "std_ms": _format_float(b.std_ms, 4),
            "min_ms": _format_float(b.min_ms, 4),
            "max_ms": _format_float(b.max_ms, 4),
            "throughput_samples_per_s": _format_float(b.throughput_samples_per_s, 2),
            "peak_vram_mb": (
                _format_float(b.peak_vram_mb, 2) if b.peak_vram_mb is not None else None
            ),
        }
        for b in benchmarks
    ]

    summary: dict[str, object] = {
        "model_name": model_name,
        "dataset_name": cfg.dataset_name,
        "object_name": partition.object_name,
        "inverted": cfg.inverted,
        "seed": seed,
        "hyperparameters": {
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "optimizer": cfg.training.optimizer,
            "scheduler": cfg.training.scheduler,
            "class_weighting": cfg.training.class_weighting,
            "image_size": cfg.data.image_size,
        },
        "split_counts": {
            "train": partition.train_counts,
            "val": partition.val_counts,
            "test": partition.test_counts,
        },
        "model_profile": {
            "parameters": benchmarks[0].parameters if benchmarks else 0,
            "trainable_parameters": (
                benchmarks[0].trainable_parameters if benchmarks else 0
            ),
            "size_mb": (
                _format_float(benchmarks[0].size_mb, 4) if benchmarks else 0.0
            ),
            "flops": benchmarks[0].flops if benchmarks else 0,
        },
        "metrics": {
            "accuracy": _format_float(metrics.accuracy, cfg.output.float_precision),
            "balanced_accuracy": _format_float(
                metrics.balanced_accuracy, cfg.output.float_precision
            ),
            "precision": _format_float(metrics.precision, cfg.output.float_precision),
            "recall": _format_float(metrics.recall, cfg.output.float_precision),
            "f1": _format_float(metrics.f1, cfg.output.float_precision),
            "roc_auc": _format_float(metrics.roc_auc, cfg.output.float_precision),
            "pr_auc": _format_float(metrics.pr_auc, cfg.output.float_precision),
            "confusion_matrix": metrics.confusion_matrix,
        },
        "timing": {
            "train_elapsed_s": _format_float(
                train_res.train_elapsed_s, cfg.output.float_precision
            ),
            "inference_elapsed_s": _format_float(
                metrics.inference_elapsed_s, cfg.output.float_precision
            ),
            "inference_ms_per_sample": _format_float(
                metrics.inference_ms_per_sample, 4
            ),
            "best_epoch": train_res.best_epoch,
            "best_val_pr_auc": _format_float(
                train_res.best_val_pr_auc, cfg.output.float_precision
            ),
        },
        "benchmarks": bench_list,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }

    if th_info is not None:
        summary["threshold_calibration"] = {
            "threshold_mode": th_info.threshold_mode,
            "selected_threshold": _format_float(th_info.selected_threshold, 4),
            "val_score_at_selected": _format_float(th_info.val_score_at_selected, 4),
            "test_at_selected_threshold": {
                "accuracy": _format_float(th_info.test_accuracy, cfg.output.float_precision),
                "balanced_accuracy": _format_float(
                    th_info.test_balanced_accuracy, cfg.output.float_precision
                ),
                "precision": _format_float(th_info.test_precision, cfg.output.float_precision),
                "recall": _format_float(th_info.test_recall, cfg.output.float_precision),
                "f1": _format_float(th_info.test_f1, cfg.output.float_precision),
            },
            "test_at_fixed_05": {
                "accuracy": _format_float(th_info.acc_at_05, cfg.output.float_precision),
                "balanced_accuracy": _format_float(
                    th_info.bal_acc_at_05, cfg.output.float_precision
                ),
                "precision": _format_float(th_info.prec_at_05, cfg.output.float_precision),
                "recall": _format_float(th_info.rec_at_05, cfg.output.float_precision),
                "f1": _format_float(th_info.f1_at_05, cfg.output.float_precision),
            },
            "oracle_test_optimal": {
                "oracle_test_threshold": _format_float(th_info.oracle_test_threshold, 4),
                "oracle_test_f1": _format_float(
                    th_info.oracle_test_f1, cfg.output.float_precision
                ),
            },
        }

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out_file


def save_training_history(output_dir: Path, train_res: TrainResult) -> Path:
    """Export training history progression as CSV.

    Args:
        output_dir: Destination directory.
        train_res: Training run results.

    Returns:
        Path to written training_history.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "training_history.csv"

    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_pr_auc",
        "val_f1",
        "lr",
    ]

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in train_res.history:
            writer.writerow(row)

    return out_file


def save_checkpoint(
    output_dir: Path, state_dict: dict[str, torch.Tensor] | object
) -> Path:
    """Save PyTorch model weights checkpoint.

    Args:
        output_dir: Destination directory.
        state_dict: Model state dictionary.

    Returns:
        Path to written checkpoint.pt.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "checkpoint.pt"
    torch.save(state_dict, out_file)
    return out_file


def plot_confusion_and_curves(
    output_dir: Path,
    metrics: ClassificationMetrics,
    model_name: str,
    dataset_name: str,
    object_name: str | None,
    th_info: ThresholdCalibrationInfo | None = None,
    dpi: int = 160,
) -> Path:
    """Render 3-panel figure: Confusion Matrix, ROC Curve, and PR Curve."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "confusion_matrix_and_curves.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=dpi)
    obj_str = f" - {object_name}" if object_name else ""
    fig.suptitle(
        f"{model_name.upper()} on {dataset_name}{obj_str} (Test Split Evaluation)",
        fontsize=13,
        fontweight="bold",
    )

    # 1. Confusion Matrix
    cm = np.array(metrics.confusion_matrix)
    ax_cm = axes[0]
    cax = ax_cm.matshow(cm, cmap="Blues", alpha=0.8)
    fig.colorbar(cax, ax=ax_cm, fraction=0.046, pad=0.04)

    classes = ["Good (0)", "Defect (1)"]
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(classes, fontsize=9)
    ax_cm.set_yticklabels(classes, fontsize=9)
    ax_cm.set_xlabel("Predicted label", fontsize=10, labelpad=8)
    ax_cm.set_ylabel("True label", fontsize=10)

    th_label = f"{th_info.selected_threshold:.2f}" if th_info is not None else "0.50"
    ax_cm.set_title(f"Confusion Matrix (Threshold={th_label})", fontsize=11, pad=12)

    total_samples = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total_samples * 100) if total_samples > 0 else 0
            row_pct = (count / cm[i].sum() * 100) if cm[i].sum() > 0 else 0
            text = f"{count}\n({pct:.1f}%)\n[Row: {row_pct:.1f}%]"
            color = "white" if count > (cm.max() / 2) else "black"
            ax_cm.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    # 2. ROC Curve
    ax_roc = axes[1]
    ax_roc.plot(
        metrics.fpr,
        metrics.tpr,
        color="#1f77b4",
        lw=2,
        label=f"ROC (AUC = {metrics.roc_auc:.4f})",
    )
    ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax_roc.set_title("Receiver Operating Characteristic", fontsize=11)
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.grid(True, linestyle=":", alpha=0.6)

    # 3. Precision-Recall Curve
    ax_pr = axes[2]
    ax_pr.plot(
        metrics.pr_recall,
        metrics.pr_precision,
        color="#ff7f0e",
        lw=2,
        label=f"PR (AUC = {metrics.pr_auc:.4f})",
    )
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    ax_pr.set_xlabel("Recall", fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.set_title(f"Precision-Recall Curve (F1 = {metrics.f1:.4f})", fontsize=11)
    ax_pr.legend(loc="lower left", fontsize=9)
    ax_pr.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_file


def plot_training_curves(
    output_dir: Path,
    train_res: TrainResult,
    model_name: str,
    dpi: int = 160,
) -> Path:
    """Render training and validation progression curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "training_curves.png"

    if not train_res.history:
        return out_file

    epochs = [int(r["epoch"]) for r in train_res.history]
    train_loss = [r["train_loss"] for r in train_res.history]
    val_loss = [r["val_loss"] for r in train_res.history]
    val_pr_auc = [r["val_pr_auc"] for r in train_res.history]
    val_f1 = [r["val_f1"] for r in train_res.history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=dpi)
    fig.suptitle(f"{model_name.upper()} Training Dynamics", fontsize=12, fontweight="bold")

    # Loss
    axes[0].plot(epochs, train_loss, label="Train Loss", color="#1f77b4", lw=1.8)
    if any(vl > 0 for vl in val_loss):
        axes[0].plot(epochs, val_loss, label="Val Loss", color="#d62728", lw=1.8, linestyle="--")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=10)
    axes[0].set_title("Loss Progression", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Validation Metrics
    if any(v > 0 for v in val_pr_auc):
        axes[1].plot(epochs, val_pr_auc, label="Val PR-AUC", color="#2ca02c", lw=1.8)
    if any(v > 0 for v in val_f1):
        axes[1].plot(epochs, val_f1, label="Val F1", color="#ff7f0e", lw=1.8, linestyle=":")
    axes[1].axvline(
        train_res.best_epoch,
        color="black",
        linestyle="-.",
        lw=1,
        label=f"Best Epoch ({train_res.best_epoch})",
    )
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Metric Score", fontsize=10)
    axes[1].set_title("Validation Discriminability", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_file
