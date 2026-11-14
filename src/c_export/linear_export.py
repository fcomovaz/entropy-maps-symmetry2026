"""Export and quantize classical tabular models (`lr_dot`, `mlp_16_dot`) to C99 headers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from configs.logging import logger
from src.models.descriptor_classifier import (
    ClassificationMetrics,
    DatasetPartition,
    LinearDotCoreClassifier,
    load_and_split_descriptors,
    train_and_evaluate,
)
from src.models.mlp_dot import MlpDotCoreClassifier

from .c_writer import format_c_array, format_c_float, write_header
from .config import CImpConfig
from .fixed_point import choose_frac_bits, dequantize, quantize, round_shift_right_i64, sat16, sat32


@dataclass(frozen=True, slots=True)
class LinearQuantParams:
    """Quantized weights and shifts for `lr_dot`."""

    d_in: int
    w_bits: int
    w_frac: int
    out_frac: int
    out_shift: int
    w_q: np.ndarray      # int64 array quantized to w_bits
    b_acc: int           # int64 in Q(in_frac + w_frac)
    threshold_q: int     # int32 in Q(out_frac)


@dataclass(frozen=True, slots=True)
class MlpQuantParams:
    """Quantized weights, hidden activation Q-format, and shifts for `mlp_16_dot`."""

    d_in: int
    d_hidden: int
    w1_bits: int
    w1_frac: int
    hidden_frac: int
    shift1: int
    w2_bits: int
    w2_frac: int
    out_frac: int
    shift2: int
    w1_q: np.ndarray     # [d_hidden, d_in] int64
    b1_acc: np.ndarray   # [d_hidden] int64 in Q(in_frac + w1_frac)
    w2_q: np.ndarray     # [d_hidden] int64
    b2_acc: int          # int64 in Q(hidden_frac + w2_frac)
    threshold_q: int     # int32 in Q(out_frac)


@dataclass(frozen=True, slots=True)
class ExportedTabularModel:
    """Metadata, provenance, and generated file paths for one exported tabular classifier."""

    model_name: str
    header_path: Path
    header_sha256: str
    golden_csv_path: Path
    metrics_crosschecked: bool
    weight_bits: int
    weight_bytes: int
    eval_f32_logits: np.ndarray
    eval_q15_logits: np.ndarray
    eval_q15_labels: np.ndarray
    q15_label_mismatches: int


def eval_lr_dot_q15(
    x_q_batch: np.ndarray,
    params: LinearQuantParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact Python mirror of `cimp_dense_classify_q15` over `[N, D]` inputs."""
    n_samples = int(x_q_batch.shape[0])
    logits_q = np.empty(n_samples, dtype=np.int32)
    labels_q = np.empty(n_samples, dtype=np.int32)
    w_q = params.w_q.astype(np.int64)
    for i in range(n_samples):
        acc = int(params.b_acc) + int(np.sum(x_q_batch[i].astype(np.int64) * w_q))
        logit = sat32(round_shift_right_i64(acc, params.out_shift))
        logits_q[i] = logit
        labels_q[i] = 1 if logit > params.threshold_q else 0
    return logits_q, labels_q


def eval_mlp_16_dot_q15(
    x_q_batch: np.ndarray,
    params: MlpQuantParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact Python mirror of `cimp_mlp_1hidden_classify_q15` over `[N, D]` inputs."""
    n_samples = int(x_q_batch.shape[0])
    logits_q = np.empty(n_samples, dtype=np.int32)
    labels_q = np.empty(n_samples, dtype=np.int32)
    w1_q = params.w1_q.astype(np.int64)
    b1_acc = params.b1_acc.astype(np.int64)
    w2_q = params.w2_q.astype(np.int64)
    hidden = np.empty(params.d_hidden, dtype=np.int64)

    for i in range(n_samples):
        row = x_q_batch[i].astype(np.int64)
        for h in range(params.d_hidden):
            acc1 = int(b1_acc[h]) + int(np.sum(row * w1_q[h]))
            shifted1 = round_shift_right_i64(acc1, params.shift1)
            hidden[h] = 0 if shifted1 <= 0 else sat16(shifted1)
        acc2 = int(params.b2_acc) + int(np.sum(hidden * w2_q))
        logit = sat32(round_shift_right_i64(acc2, params.shift2))
        logits_q[i] = logit
        labels_q[i] = 1 if logit > params.threshold_q else 0
    return logits_q, labels_q


def _crosscheck_saved_metrics(
    cfg: CImpConfig,
    cls_name: str,
    metrics: ClassificationMetrics,
) -> bool:
    """Verify deterministic refit metrics match saved classification `metrics_summary.json` if present."""
    dataset_tag = cfg.classification_cfg.effective_dataset_tag
    target_tag = cfg.object_name if cfg.object_name else "full"
    summary_path = (
        cfg.classifier.metrics_crosscheck_dir
        / dataset_tag
        / f"{target_tag}_{cls_name}"
        / "metrics_summary.json"
    )
    if not summary_path.exists():
        logger.info(
            "No existing %s baseline found at %s; skipping cross-check.",
            cls_name,
            summary_path,
        )
        return False

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    saved = payload.get("active_subset_metrics", {})
    prec = cfg.classification_cfg.float_precision
    tol = cfg.classifier.metrics_crosscheck_abs_tol

    for metric_key, current_val in (
        ("accuracy", metrics.accuracy),
        ("f1", metrics.f1),
        ("roc_auc", metrics.roc_auc),
    ):
        if metric_key not in saved:
            raise KeyError(f"Missing '{metric_key}' in {summary_path}")
        diff = abs(round(float(current_val), prec) - float(saved[metric_key]))
        if diff > tol:
            raise ValueError(
                f"Metric cross-check failed for '{cls_name}' metric '{metric_key}' against "
                f"{summary_path}: current={round(float(current_val), prec)}, "
                f"saved={saved[metric_key]}, diff={diff:.3e} > tol={tol:.3e}"
            )
    logger.info("Cross-checked '%s' refit metrics against %s (diff <= %.1e).", cls_name, summary_path, tol)
    return True


def _quantize_lr_dot(
    model: LinearDotCoreClassifier,
    x_eval_f64: np.ndarray,
    x_eval_q15: np.ndarray,
    in_frac: int,
) -> LinearQuantParams:
    """Select weight bit-width (q15 or automatic q31 upgrade) and build `LinearQuantParams`."""
    w_eff = np.asarray(model.w_eff_, dtype=np.float64)
    b_eff = float(model.b_eff_)
    logits_f32 = (x_eval_f64.astype(np.float32) @ w_eff.astype(np.float32) + np.float32(b_eff))
    labels_f32 = (logits_f32 > 0.0).astype(np.int32)
    out_frac = 16

    for w_bits in (16, 32):
        w_frac = choose_frac_bits(float(np.max(np.abs(w_eff))), storage_bits=w_bits)
        w_q, n_sat = quantize(w_eff, w_frac, bits=w_bits)
        if n_sat > 0 and w_bits == 16:
            continue
        acc_frac = in_frac + w_frac
        b_q, _ = quantize(b_eff, acc_frac, bits=64)
        params = LinearQuantParams(
            d_in=int(w_eff.size),
            w_bits=w_bits,
            w_frac=w_frac,
            out_frac=out_frac,
            out_shift=acc_frac - out_frac,
            w_q=w_q,
            b_acc=int(b_q.item()),
            threshold_q=0,
        )
        _, labels_q = eval_lr_dot_q15(x_eval_q15, params)
        if int(np.sum(labels_q != labels_f32)) == 0 or w_bits == 32:
            return params
    raise RuntimeError("Unreachable: w_bits=32 always returns")


def _quantize_mlp_16_dot(
    model: MlpDotCoreClassifier,
    x_calib_f64: np.ndarray,
    x_eval_f64: np.ndarray,
    x_eval_q15: np.ndarray,
    in_frac: int,
) -> MlpQuantParams:
    """Calibrate hidden activations and quantize `MlpDotCoreClassifier` cleanly."""
    w1_eff = np.asarray(model.w1_eff_, dtype=np.float64)
    b1_eff = np.asarray(model.b1_eff_, dtype=np.float64)
    w2_eff = np.asarray(model.w2_eff_, dtype=np.float64)
    b2_eff = float(model.b2_eff_)

    h_calib = np.maximum(0.0, x_calib_f64 @ w1_eff.T + b1_eff)
    max_h = float(np.max(h_calib)) if h_calib.size else 1.0
    hidden_frac = choose_frac_bits(max(max_h, 1.0), storage_bits=16, headroom_bits=1)

    h_eval_f32 = np.maximum(
        0.0,
        x_eval_f64.astype(np.float32) @ w1_eff.astype(np.float32).T
        + b1_eff.astype(np.float32),
    )
    logits_f32 = h_eval_f32 @ w2_eff.astype(np.float32) + np.float32(b2_eff)
    labels_f32 = (logits_f32 > 0.0).astype(np.int32)
    out_frac = 16

    for w_bits in (16, 32):
        w1_frac = choose_frac_bits(float(np.max(np.abs(w1_eff))), storage_bits=w_bits)
        w2_frac = choose_frac_bits(float(np.max(np.abs(w2_eff))), storage_bits=w_bits)
        w1_q, sat1 = quantize(w1_eff, w1_frac, bits=w_bits)
        w2_q, sat2 = quantize(w2_eff, w2_frac, bits=w_bits)
        if (sat1 > 0 or sat2 > 0) and w_bits == 16:
            continue

        acc1_frac = in_frac + w1_frac
        b1_q, _ = quantize(b1_eff, acc1_frac, bits=64)
        acc2_frac = hidden_frac + w2_frac
        b2_q, _ = quantize(b2_eff, acc2_frac, bits=64)

        params = MlpQuantParams(
            d_in=int(w1_eff.shape[1]),
            d_hidden=int(w1_eff.shape[0]),
            w1_bits=w_bits,
            w1_frac=w1_frac,
            hidden_frac=hidden_frac,
            shift1=acc1_frac - hidden_frac,
            w2_bits=w_bits,
            w2_frac=w2_frac,
            out_frac=out_frac,
            shift2=acc2_frac - out_frac,
            w1_q=w1_q,
            b1_acc=b1_q,
            w2_q=w2_q,
            b2_acc=int(b2_q.item()),
            threshold_q=0,
        )
        _, labels_q = eval_mlp_16_dot_q15(x_eval_q15, params)
        if int(np.sum(labels_q != labels_f32)) == 0 or w_bits == 32:
            return params
    raise RuntimeError("Unreachable: w_bits=32 always returns")


def _write_golden_tabular_csv(
    csv_path: Path,
    file_names: tuple[str, ...],
    true_labels: np.ndarray,
    logits_f32: np.ndarray,
    labels_f32: np.ndarray,
    logits_q: np.ndarray,
    logits_q_float: np.ndarray,
    labels_q: np.ndarray,
) -> None:
    """Write golden per-sample tabular predictions CSV for Python-vs-C parity checks."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "file_name",
                "true_label",
                "py_f32_logit",
                "py_f32_label",
                "py_q15_logit_raw",
                "py_q15_logit_f64",
                "py_q15_label",
            ]
        )
        for i, fname in enumerate(file_names):
            writer.writerow(
                [
                    i,
                    fname,
                    int(true_labels[i]),
                    f"{float(logits_f32[i]):.9g}",
                    int(labels_f32[i]),
                    int(logits_q[i]),
                    f"{float(logits_q_float[i]):.9g}",
                    int(labels_q[i]),
                ]
            )


def _export_lr_dot(
    cfg: CImpConfig,
    partition: DatasetPartition,
    eval_file_names: tuple[str, ...],
    eval_true_labels: np.ndarray,
    eval_x94_f64: np.ndarray,
    eval_x94_q15: np.ndarray,
) -> ExportedTabularModel:
    """Fit, cross-check, quantize, and emit `model_lr_dot.h` + `golden_lr_dot.csv`."""
    fitted, metrics = train_and_evaluate(partition, "lr_dot", cfg.classification_cfg, subset_name="x94_full")
    if not isinstance(fitted, LinearDotCoreClassifier):
        raise TypeError(f"Expected LinearDotCoreClassifier, got {type(fitted).__name__}")
    crosschecked = _crosscheck_saved_metrics(cfg, "lr_dot", metrics)

    w_eff = np.asarray(fitted.w_eff_, dtype=np.float32)
    b_eff = float(fitted.b_eff_)
    q_params = _quantize_lr_dot(
        fitted, eval_x94_f64, eval_x94_q15, in_frac=cfg.numeric.feature_frac_bits
    )

    f32_logits = (eval_x94_f64.astype(np.float32) @ w_eff + np.float32(b_eff)).astype(np.float32)
    f32_labels = (f32_logits > 0.0).astype(np.int32)
    q_logits_raw, q_labels = eval_lr_dot_q15(eval_x94_q15, q_params)
    q_logits_float = dequantize(q_logits_raw, q_params.out_frac)
    mismatches = int(np.sum(q_labels != f32_labels))

    w_c_type = "int32_t" if q_params.w_bits == 32 else "q15_t"
    body = "\n\n".join(
        [
            f"#define CIMP_LR_DOT_D_IN {q_params.d_in}\n"
            f"#define CIMP_LR_DOT_W_BITS {q_params.w_bits}\n"
            f"#define CIMP_LR_DOT_W_FRAC {q_params.w_frac}\n"
            f"#define CIMP_LR_DOT_OUT_FRAC {q_params.out_frac}\n"
            f"#define CIMP_LR_DOT_OUT_SHIFT {q_params.out_shift}\n"
            f"#define CIMP_LR_DOT_B_EFF_F32 ({format_c_float(b_eff)})\n"
            f"#define CIMP_LR_DOT_THRESHOLD_F32 0.0f\n"
            f"#define CIMP_LR_DOT_B_ACC ((int64_t){q_params.b_acc}LL)\n"
            f"#define CIMP_LR_DOT_THRESHOLD_Q ((int32_t){q_params.threshold_q})",
            format_c_array("CIMP_LR_DOT_W_F32", w_eff, "float", per_line=8),
            format_c_array("CIMP_LR_DOT_W_Q", q_params.w_q, w_c_type, per_line=10),
        ]
    )
    header_path = cfg.generated_dir / "model_lr_dot.h"
    header_sha = write_header(
        path=header_path,
        guard="CIMP_MODEL_LR_DOT_H",
        includes=("<stdint.h>", '"cimp_fixed_point.h"'),
        body=body,
        provenance_lines=(
            "Model: lr_dot (scaler folded into w_eff, b_eff)",
            f"Target: {cfg.effective_target_tag} | seed={cfg.seed}",
            f"Weight storage: {w_c_type} ({q_params.w_bits}-bit), w_frac={q_params.w_frac}, out_shift={q_params.out_shift}",
        ),
    )

    golden_csv = cfg.generated_dir / "golden_lr_dot.csv"
    _write_golden_tabular_csv(
        golden_csv,
        eval_file_names,
        eval_true_labels,
        f32_logits,
        f32_labels,
        q_logits_raw,
        q_logits_float,
        q_labels,
    )
    weight_bytes = int((q_params.d_in * (q_params.w_bits // 8)) + 8)
    return ExportedTabularModel(
        model_name="lr_dot",
        header_path=header_path,
        header_sha256=header_sha,
        golden_csv_path=golden_csv,
        metrics_crosschecked=crosschecked,
        weight_bits=q_params.w_bits,
        weight_bytes=weight_bytes,
        eval_f32_logits=f32_logits,
        eval_q15_logits=q_logits_raw,
        eval_q15_labels=q_labels,
        q15_label_mismatches=mismatches,
    )


def _export_mlp_16_dot(
    cfg: CImpConfig,
    partition: DatasetPartition,
    eval_file_names: tuple[str, ...],
    eval_true_labels: np.ndarray,
    eval_x94_f64: np.ndarray,
    eval_x94_q15: np.ndarray,
) -> ExportedTabularModel:
    """Fit, cross-check, quantize, and emit `model_mlp_16_dot.h` + `golden_mlp_16_dot.csv`."""
    fitted, metrics = train_and_evaluate(partition, "mlp_16_dot", cfg.classification_cfg, subset_name="x94_full")
    if not isinstance(fitted, MlpDotCoreClassifier):
        raise TypeError(f"Expected MlpDotCoreClassifier, got {type(fitted).__name__}")
    crosschecked = _crosscheck_saved_metrics(cfg, "mlp_16_dot", metrics)

    w1_eff = np.asarray(fitted.w1_eff_, dtype=np.float32)
    b1_eff = np.asarray(fitted.b1_eff_, dtype=np.float32)
    w2_eff = np.asarray(fitted.w2_eff_, dtype=np.float32)
    b2_eff = float(fitted.b2_eff_)

    # Calibrate hidden activations using training descriptors
    q_params = _quantize_mlp_16_dot(
        fitted,
        x_calib_f64=partition.X_train,
        x_eval_f64=eval_x94_f64,
        x_eval_q15=eval_x94_q15,
        in_frac=cfg.numeric.feature_frac_bits,
    )

    h_f32 = np.maximum(0.0, eval_x94_f64.astype(np.float32) @ w1_eff.T + b1_eff)
    f32_logits = (h_f32 @ w2_eff + np.float32(b2_eff)).astype(np.float32)
    f32_labels = (f32_logits > 0.0).astype(np.int32)
    q_logits_raw, q_labels = eval_mlp_16_dot_q15(eval_x94_q15, q_params)
    q_logits_float = dequantize(q_logits_raw, q_params.out_frac)
    mismatches = int(np.sum(q_labels != f32_labels))

    w1_c_type = "int32_t" if q_params.w1_bits == 32 else "q15_t"
    w2_c_type = "int32_t" if q_params.w2_bits == 32 else "q15_t"
    body = "\n\n".join(
        [
            f"#define CIMP_MLP_16_DOT_D_IN {q_params.d_in}\n"
            f"#define CIMP_MLP_16_DOT_D_HIDDEN {q_params.d_hidden}\n"
            f"#define CIMP_MLP_16_DOT_W1_BITS {q_params.w1_bits}\n"
            f"#define CIMP_MLP_16_DOT_W1_FRAC {q_params.w1_frac}\n"
            f"#define CIMP_MLP_16_DOT_HIDDEN_FRAC {q_params.hidden_frac}\n"
            f"#define CIMP_MLP_16_DOT_SHIFT1 {q_params.shift1}\n"
            f"#define CIMP_MLP_16_DOT_W2_BITS {q_params.w2_bits}\n"
            f"#define CIMP_MLP_16_DOT_W2_FRAC {q_params.w2_frac}\n"
            f"#define CIMP_MLP_16_DOT_OUT_FRAC {q_params.out_frac}\n"
            f"#define CIMP_MLP_16_DOT_SHIFT2 {q_params.shift2}\n"
            f"#define CIMP_MLP_16_DOT_B2_EFF_F32 ({format_c_float(b2_eff)})\n"
            f"#define CIMP_MLP_16_DOT_THRESHOLD_F32 0.0f\n"
            f"#define CIMP_MLP_16_DOT_B2_ACC ((int64_t){q_params.b2_acc}LL)\n"
            f"#define CIMP_MLP_16_DOT_THRESHOLD_Q ((int32_t){q_params.threshold_q})",
            format_c_array("CIMP_MLP_16_DOT_W1_F32", w1_eff.ravel(), "float", per_line=8),
            format_c_array("CIMP_MLP_16_DOT_B1_F32", b1_eff, "float", per_line=8),
            format_c_array("CIMP_MLP_16_DOT_W2_F32", w2_eff, "float", per_line=8),
            format_c_array("CIMP_MLP_16_DOT_W1_Q", q_params.w1_q.ravel(), w1_c_type, per_line=10),
            format_c_array("CIMP_MLP_16_DOT_B1_ACC", q_params.b1_acc, "int64_t", per_line=6),
            format_c_array("CIMP_MLP_16_DOT_W2_Q", q_params.w2_q, w2_c_type, per_line=10),
        ]
    )
    header_path = cfg.generated_dir / "model_mlp_16_dot.h"
    header_sha = write_header(
        path=header_path,
        guard="CIMP_MODEL_MLP_16_DOT_H",
        includes=("<stdint.h>", '"cimp_fixed_point.h"'),
        body=body,
        provenance_lines=(
            "Model: mlp_16_dot (StandardScaler folded into fc1 weights/bias)",
            f"Target: {cfg.effective_target_tag} | seed={cfg.seed}",
            f"W1: {w1_c_type} (w1_frac={q_params.w1_frac}, shift1={q_params.shift1}), "
            f"W2: {w2_c_type} (w2_frac={q_params.w2_frac}, shift2={q_params.shift2})",
        ),
    )

    golden_csv = cfg.generated_dir / "golden_mlp_16_dot.csv"
    _write_golden_tabular_csv(
        golden_csv,
        eval_file_names,
        eval_true_labels,
        f32_logits,
        f32_labels,
        q_logits_raw,
        q_logits_float,
        q_labels,
    )
    weight_bytes = int(
        (q_params.w1_q.size * (q_params.w1_bits // 8))
        + (q_params.b1_acc.size * 8)
        + (q_params.w2_q.size * (q_params.w2_bits // 8))
        + 8
    )
    return ExportedTabularModel(
        model_name="mlp_16_dot",
        header_path=header_path,
        header_sha256=header_sha,
        golden_csv_path=golden_csv,
        metrics_crosschecked=crosschecked,
        weight_bits=q_params.w1_bits,
        weight_bytes=weight_bytes,
        eval_f32_logits=f32_logits,
        eval_q15_logits=q_logits_raw,
        eval_q15_labels=q_labels,
        q15_label_mismatches=mismatches,
    )


def export_classical_models(
    cfg: CImpConfig,
    eval_file_names: tuple[str, ...],
    eval_true_labels: np.ndarray,
    eval_x94_f64: np.ndarray,
    eval_x94_q15: np.ndarray,
) -> dict[str, ExportedTabularModel]:
    """Fit, cross-check, quantize, and emit C99 headers for configured classical models."""
    partition = load_and_split_descriptors(
        cfg.classification_cfg,
        object_name=cfg.object_name,
        subset_name="x94_full",
    )
    exported: dict[str, ExportedTabularModel] = {}
    for cls_name in cfg.classifier.models:
        if cls_name == "lr_dot":
            exported[cls_name] = _export_lr_dot(
                cfg, partition, eval_file_names, eval_true_labels, eval_x94_f64, eval_x94_q15
            )
        elif cls_name == "mlp_16_dot":
            exported[cls_name] = _export_mlp_16_dot(
                cfg, partition, eval_file_names, eval_true_labels, eval_x94_f64, eval_x94_q15
            )
        else:
            raise ValueError(f"Unsupported C export tabular classifier: {cls_name}")
    return exported
