"""Inference, probability calculation, threshold calibration, and evaluation metrics for CNNs."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
from torch import nn

from src.models.descriptor_classifier import ClassificationMetrics
from .config import CnnConfig
from .data import CnnDatasetPartition


@dataclass(frozen=True, slots=True)
class ThresholdCalibrationInfo:
    """Detailed record of decision threshold calibration and comparative operating points."""

    selected_threshold: float
    threshold_mode: str
    val_score_at_selected: float
    # Test metrics at selected threshold
    test_accuracy: float
    test_balanced_accuracy: float
    test_f1: float
    test_precision: float
    test_recall: float
    # Test metrics at fixed 0.50 threshold (for uncalibrated baseline parity)
    f1_at_05: float
    acc_at_05: float
    bal_acc_at_05: float
    prec_at_05: float
    rec_at_05: float
    # Oracle test optimal (for upper-bound ranking analysis only)
    oracle_test_threshold: float
    oracle_test_f1: float


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Find decision threshold maximizing a metric on validation set without test leakage.

    Args:
        y_true: Ground truth binary labels in {0, 1}.
        y_prob: Predicted positive class probabilities in [0, 1].
        metric: 'f1' or 'balanced_accuracy'.

    Returns:
        Tuple of (optimal_threshold, best_val_score).
    """
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.50, 0.0

    thresholds = np.linspace(0.05, 0.95, 91)
    best_th = 0.50
    best_score = -1.0

    for th in thresholds:
        preds = (y_prob >= th).astype(np.int64)
        if metric == "f1":
            score = float(f1_score(y_true, preds, zero_division=0))
        elif metric == "balanced_accuracy":
            score = float(balanced_accuracy_score(y_true, preds))
        else:
            raise ValueError(f"Unsupported threshold optimization metric '{metric}'")

        # In case of ties, prefer threshold closest to 0.50 to avoid extreme edge drift
        if score > best_score or (
            score == best_score and abs(th - 0.50) < abs(best_th - 0.50)
        ):
            best_score = score
            best_th = round(float(th), 4)

    return round(best_th, 4), best_score


def predict_proba(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute softmax probabilities for an input image tensor.

    Args:
        model: Trained nn.Module.
        images: Input tensor of shape [N, 1, H, W].
        device: Device on which to run inference.
        batch_size: Evaluation batch size.

    Returns:
        NumPy array of shape [N, 2] with class probabilities.
    """
    model.eval()
    n = images.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    probs_list: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            xb = images[i : min(i + batch_size, n)].to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(prob)

    return np.concatenate(probs_list, axis=0)


def evaluate_cnn(
    model: nn.Module,
    partition: CnnDatasetPartition,
    cfg: CnnConfig,
    model_name: str,
    train_elapsed_s: float,
    device: torch.device,
    batch_size: int = 64,
    threshold_mode: str = "val_f1",
    fixed_threshold: float | None = None,
) -> Tuple[ClassificationMetrics, np.ndarray, np.ndarray, ThresholdCalibrationInfo]:
    """Evaluate CNN on test partition with validation-based threshold calibration.

    Addresses the class-weighting logit shift: weighted CrossEntropyLoss shifts
    logits by log(w1 / w0). Applying a static threshold of 0.50 without calibration
    causes severe False Positive inflation when defect prevalence is low.
    Calibrating the threshold on the validation split restores high F1 without
    any test-set leakage.

    Args:
        model: Evaluated nn.Module.
        partition: CnnDatasetPartition containing X_train, X_val, y_val, X_test, y_test.
        cfg: Validated CnnConfig instance.
        model_name: Architecture identifier.
        train_elapsed_s: Measured training duration.
        device: Device used for evaluation.
        batch_size: Batch size for test inference.
        threshold_mode: Strategy for decision threshold ('val_f1', 'fixed_05', 'val_bal_acc').
        fixed_threshold: Optional explicit threshold override.

    Returns:
        Tuple of (ClassificationMetrics, y_pred, y_prob, ThresholdCalibrationInfo).
    """
    model.eval()
    x_test = partition.X_test
    y_test = partition.y_test.cpu().numpy()
    n_test = max(1, len(y_test))

    # 1. Timed inference on test set
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0_inf = time.perf_counter()

    probs_test = predict_proba(model, x_test, device, batch_size=batch_size)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_elapsed_s = time.perf_counter() - t0_inf
    inference_ms_per_sample = (inference_elapsed_s * 1000.0) / n_test

    y_prob_test = probs_test[:, 1] if len(probs_test) > 0 else np.empty(0, dtype=np.float64)

    # 2. Threshold Calibration on Validation Set (strictly without test leakage)
    val_probs = np.empty(0, dtype=np.float64)
    y_val = partition.y_val.cpu().numpy()
    if len(partition.X_val) > 0 and len(np.unique(y_val)) > 1:
        probs_val = predict_proba(model, partition.X_val, device, batch_size=batch_size)
        val_probs = probs_val[:, 1]

    if fixed_threshold is not None:
        selected_th = float(fixed_threshold)
        mode_used = "fixed_explicit"
        val_score = 0.0
    elif threshold_mode == "fixed_05":
        selected_th = 0.50
        mode_used = "fixed_05"
        val_score = 0.0
    elif threshold_mode == "val_bal_acc":
        selected_th, val_score = find_optimal_threshold(
            y_val, val_probs, metric="balanced_accuracy"
        )
        mode_used = "val_bal_acc"
    else:  # default 'val_f1'
        selected_th, val_score = find_optimal_threshold(
            y_val, val_probs, metric="f1"
        )
        mode_used = "val_f1"

    # 3. Test predictions at calibrated threshold
    y_pred_selected = (y_prob_test >= selected_th).astype(np.int64)
    acc = float(accuracy_score(y_test, y_pred_selected))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred_selected))
    prec = float(precision_score(y_test, y_pred_selected, zero_division=0))
    rec = float(recall_score(y_test, y_pred_selected, zero_division=0))
    f1 = float(f1_score(y_test, y_pred_selected, zero_division=0))

    # 4. Comparative test predictions at static 0.50 cutoff
    y_pred_05 = (y_prob_test >= 0.50).astype(np.int64)
    acc_05 = float(accuracy_score(y_test, y_pred_05))
    bal_acc_05 = float(balanced_accuracy_score(y_test, y_pred_05))
    prec_05 = float(precision_score(y_test, y_pred_05, zero_division=0))
    rec_05 = float(recall_score(y_test, y_pred_05, zero_division=0))
    f1_05 = float(f1_score(y_test, y_pred_05, zero_division=0))

    # 5. Oracle test threshold (upper bound reference for ranking capacity analysis)
    oracle_th, oracle_f1 = find_optimal_threshold(y_test, y_prob_test, metric="f1")

    # 6. Threshold-free ranking metrics (ROC-AUC & PR-AUC)
    if len(np.unique(y_test)) > 1:
        roc_auc = float(roc_auc_score(y_test, y_prob_test))
        pr_auc = float(average_precision_score(y_test, y_prob_test))
        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob_test)
    else:
        roc_auc = 0.0
        pr_auc = 0.0
        fpr = np.array([0.0, 1.0], dtype=np.float64)
        tpr = np.array([0.0, 1.0], dtype=np.float64)
        pr_precision = np.array([0.0, 1.0], dtype=np.float64)
        pr_recall = np.array([1.0, 0.0], dtype=np.float64)

    cm_raw = confusion_matrix(y_test, y_pred_selected, labels=[0, 1])
    cm_tuple = (
        (int(cm_raw[0, 0]), int(cm_raw[0, 1])),
        (int(cm_raw[1, 0]), int(cm_raw[1, 1])),
    )

    metrics = ClassificationMetrics(
        classifier_name=model_name.lower(),
        subset_name=f"raw_pixels_128x128_th{selected_th:.2f}",
        n_features=cfg.data.image_size * cfg.data.image_size,
        accuracy=acc,
        balanced_accuracy=bal_acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm_tuple,
        fpr=np.asarray(fpr, dtype=np.float64),
        tpr=np.asarray(tpr, dtype=np.float64),
        pr_precision=np.asarray(pr_precision, dtype=np.float64),
        pr_recall=np.asarray(pr_recall, dtype=np.float64),
        train_elapsed_s=train_elapsed_s,
        inference_elapsed_s=inference_elapsed_s,
        inference_ms_per_sample=inference_ms_per_sample,
    )

    th_info = ThresholdCalibrationInfo(
        selected_threshold=selected_th,
        threshold_mode=mode_used,
        val_score_at_selected=val_score,
        test_accuracy=acc,
        test_balanced_accuracy=bal_acc,
        test_f1=f1,
        test_precision=prec,
        test_recall=rec,
        f1_at_05=f1_05,
        acc_at_05=acc_05,
        bal_acc_at_05=bal_acc_05,
        prec_at_05=prec_05,
        rec_at_05=rec_05,
        oracle_test_threshold=oracle_th,
        oracle_test_f1=oracle_f1,
    )

    return metrics, y_pred_selected, y_prob_test, th_info
