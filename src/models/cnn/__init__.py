"""Lightweight CNN models, data pipelines, training, evaluation, and benchmarking."""

from __future__ import annotations

from .architectures import (
    DepthwiseSeparableCNN,
    MobileNetLikeCNN,
    SmallCNN,
    TinyCNN,
    TinyResidualCNN,
    TorchvisionMobileNet,
    UltraTinyCNN,
    create_model,
    model_names,
)
from .benchmark import BenchmarkRecord, run_benchmark
from .config import CnnConfig, load_cnn_config
from .data import CnnDatasetPartition, load_cnn_partition
from .evaluate import evaluate_cnn, predict_proba
from .report import (
    plot_confusion_and_curves,
    plot_training_curves,
    save_checkpoint,
    save_metrics_summary,
    save_training_history,
)
from .train import TrainResult, train_model

__all__ = [
    "CnnConfig",
    "load_cnn_config",
    "CnnDatasetPartition",
    "load_cnn_partition",
    "train_model",
    "TrainResult",
    "predict_proba",
    "evaluate_cnn",
    "BenchmarkRecord",
    "run_benchmark",
    "save_metrics_summary",
    "save_training_history",
    "save_checkpoint",
    "plot_confusion_and_curves",
    "plot_training_curves",
    "create_model",
    "model_names",
    "TinyCNN",
    "SmallCNN",
    "DepthwiseSeparableCNN",
    "MobileNetLikeCNN",
    "TinyResidualCNN",
    "UltraTinyCNN",
    "TorchvisionMobileNet",
]
