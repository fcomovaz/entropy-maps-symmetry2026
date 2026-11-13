"""MCU-portable folded classifier cores, C99 export helpers, and parity verification."""

from __future__ import annotations

from dataclasses import dataclass, replace
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import torch
import torch.nn as nn

LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import ModelsConfig, ThresholdsConfig
from native_bridge import NativeClassifierLibrary, measure_c_inner_us_per_sample
from src.models.descriptor_classifier import ClassificationConfig, DatasetPartition, train_and_evaluate


def sigmoid_stable(scores: np.ndarray) -> np.ndarray:
    """Compute numerically stable binary probabilities from raw decision logits."""
    clipped = np.clip(np.asarray(scores, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _safe_scaler_stats(scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    """Extract float64 mean and positive scale matching scikit-learn zero-variance handling."""
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64).copy()
    scale[scale == 0.0] = 1.0
    return mean, scale


@dataclass(frozen=True, slots=True)
class McuCostProfile:
    """Exact static memory footprint and inference arithmetic budget for MCU deployment."""

    model_family: str
    n_params: int
    flash_bytes_f32: int
    sram_scratch_bytes_f32: int
    macs: int
    adds: int
    comparisons: int
    math_calls_at_threshold: int
    n_active_features: int


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    """Predictive quality metrics evaluated cleanly on a fixed test partition."""

    accuracy: float
    balanced_accuracy: float
    f1: float
    roc_auc: float
    pr_auc: float


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Numerical fidelity checks across reference, folded float64/float32, and C99."""

    max_abs_diff_f64: float | None
    max_abs_diff_f32: float | None
    pred_agreement_f32: float
    max_abs_diff_c99: float | None
    pred_agreement_c99: float | None


def compute_quality_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> QualityMetrics:
    """Compute canonical evaluation metrics aligned with src/models/descriptor_classifier."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    prob = np.asarray(y_prob, dtype=np.float64)
    has_both = np.unique(y_t).size >= 2
    roc = float(roc_auc_score(y_t, prob)) if has_both else float("nan")
    pr = float(average_precision_score(y_t, prob)) if has_both else float("nan")
    return QualityMetrics(
        accuracy=float(accuracy_score(y_t, y_p)),
        balanced_accuracy=float(balanced_accuracy_score(y_t, y_p)),
        f1=float(f1_score(y_t, y_p, zero_division=0)),
        roc_auc=roc,
        pr_auc=pr,
    )


class FoldedMcuModel:
    """Base strategy interface for MCU-folded inference models and C99 parity."""

    name: str
    display_group: str
    reference_model: Any
    cost_profile: McuCostProfile
    extra_note: str

    def predict_scores_ref(self, X: np.ndarray) -> np.ndarray:
        """Evaluate original reference model logits / continuous scores for parity validation."""
        raise NotImplementedError

    def predict_scores_f64(self, X: np.ndarray) -> np.ndarray:
        """Evaluate algebraically folded float64 NumPy inference core."""
        raise NotImplementedError

    def predict_scores_f32(self, X: np.ndarray) -> np.ndarray:
        """Evaluate quantized float32 NumPy inference core."""
        raise NotImplementedError

    def predict_batch_for_latency(self, X_f64: np.ndarray, X_f32: np.ndarray) -> object:
        """Run realistic B-sample inference entrypoint measured by Python/NumPy profiler."""
        return self.predict_scores_f64(X_f64)

    def predict_probs_and_labels(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute canonical probability scores and binary labels from folded core."""
        scores = self.predict_scores_f64(X)
        return sigmoid_stable(scores), (scores > 0.0).astype(np.int64)

    def supports_c99(self) -> bool:
        """Return True when this model maps directly to a compiled C99 kernel."""
        return True

    def numerical_scale_bound(self) -> float:
        """Return characteristic pre-cancellation score scale for IEEE-754 float32 parity."""
        return 1.0

    def predict_scores_c99(self, lib: NativeClassifierLibrary, X_f32: np.ndarray) -> np.ndarray:
        """Execute matching C99 batch entrypoint cleanly via ctypes."""
        raise NotImplementedError

    def measure_c99_inner_us(
        self,
        lib: NativeClassifierLibrary,
        X_f32: np.ndarray,
        inner_iters: int,
    ) -> float:
        """Measure pure C99 kernel execution time in microseconds/sample."""
        raise NotImplementedError

    def export_c99_snippet(self, prefix: str) -> str:
        """Render standalone C99 static weight declarations for embedded integration."""
        return f"/* Export snippet not implemented for {self.name} */\n"


def _fmt_f32_array(name: str, arr: np.ndarray) -> str:
    """Format a 1D or flat float32 NumPy array into clean C99 static const initialization."""
    flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
    lines = [f"static const float {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, 6):
        chunk = ", ".join(f"{float(v):.8e}f" for v in flat[i : i + 6])
        comma = "," if (i + 6) < flat.size else ""
        lines.append(f"    {chunk}{comma}")
    lines.append("};")
    return "\n".join(lines)


def _fmt_int_array(ctype: str, name: str, arr: np.ndarray) -> str:
    """Format integer NumPy array into C99 static const array initializer."""
    flat = np.ascontiguousarray(arr).ravel()
    lines = [f"static const {ctype} {name}[{flat.size}] = {{"]
    for i in range(0, flat.size, 12):
        chunk = ", ".join(str(int(v)) for v in flat[i : i + 12])
        comma = "," if (i + 12) < flat.size else ""
        lines.append(f"    {chunk}{comma}")
    lines.append("};")
    return "\n".join(lines)


class BaselineSrcModelWrapper(FoldedMcuModel):
    """Wraps unmodified src/ models (lr_dot, dt, rf) for uniform benchmarking and reporting."""

    def __init__(
        self,
        name: str,
        estimator: Any,
        n_features: int,
        zero_weight_eps: float = 1e-12,
    ) -> None:
        self.name = name
        self.display_group = "src_baseline"
        self.reference_model = estimator
        self.extra_note = "src_unmodified"
        if name == "lr_dot":
            w = np.asarray(estimator.w_eff_, dtype=np.float64)
            b = float(estimator.b_eff_)
            self._w64 = w
            self._b64 = b
            self._w32 = np.ascontiguousarray(w, dtype=np.float32)
            self._b32 = np.float32(b)
            nnz = int(np.count_nonzero(np.abs(w) > zero_weight_eps))
            self.cost_profile = McuCostProfile(
                model_family="linear_dense",
                n_params=int(w.size + 1),
                flash_bytes_f32=int((w.size + 1) * 4),
                sram_scratch_bytes_f32=0,
                macs=int(w.size),
                adds=int(w.size),
                comparisons=1,
                math_calls_at_threshold=0,
                n_active_features=nnz,
            )
        elif name == "dt":
            n_nodes = int(estimator.tree_.node_count)
            feats = np.asarray(estimator.tree_.feature, dtype=np.int16)
            used = {int(f) for f in feats if f >= 0}
            self.cost_profile = McuCostProfile(
                model_family="dt_sklearn",
                n_params=n_nodes,
                flash_bytes_f32=int(n_nodes * 14 + 4),
                sram_scratch_bytes_f32=0,
                macs=0,
                adds=0,
                comparisons=int(estimator.get_depth()),
                math_calls_at_threshold=0,
                n_active_features=len(used),
            )
        elif name == "rf":
            estimators = getattr(estimator, "estimators_", [])
            n_nodes = int(sum(int(t.tree_.node_count) for t in estimators))
            depths = [int(t.get_depth()) for t in estimators]
            used = set()
            for t in estimators:
                for f in t.tree_.feature:
                    if int(f) >= 0:
                        used.add(int(f))
            # Pin single-thread execution for fair threads=1 profiling cleanly without refitting.
            if hasattr(estimator, "n_jobs"):
                estimator.n_jobs = 1
            self.cost_profile = McuCostProfile(
                model_family="rf_forest",
                n_params=n_nodes,
                flash_bytes_f32=int(n_nodes * 14 + len(estimators) * 4),
                sram_scratch_bytes_f32=0,
                macs=0,
                adds=max(0, len(estimators) - 1),
                comparisons=int(sum(depths)),
                math_calls_at_threshold=0,
                n_active_features=len(used),
            )
        else:
            raise ValueError(f"Unsupported src baseline wrapper name: {name}")

    def supports_c99(self) -> bool:
        return self.name == "lr_dot"

    def predict_scores_ref(self, X: np.ndarray) -> np.ndarray:
        if self.name == "lr_dot":
            return np.asarray(self.reference_model.decision_function(X), dtype=np.float64)
        probs = np.asarray(self.reference_model.predict_proba(X)[:, 1], dtype=np.float64)
        return probs - 0.5

    def predict_scores_f64(self, X: np.ndarray) -> np.ndarray:
        if self.name == "lr_dot":
            return np.asarray(self.reference_model.decision_function(X), dtype=np.float64)
        return self.predict_scores_ref(X)

    def predict_scores_f32(self, X: np.ndarray) -> np.ndarray:
        if self.name == "lr_dot":
            X_f32 = np.asarray(X, dtype=np.float32)
            return np.asarray(X_f32 @ self._w32 + self._b32, dtype=np.float64)
        return self.predict_scores_ref(X)

    def predict_probs_and_labels(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probs = np.asarray(self.reference_model.predict_proba(X)[:, 1], dtype=np.float64)
        preds = np.asarray(self.reference_model.predict(X), dtype=np.int64)
        return probs, preds

    def predict_batch_for_latency(self, X_f64: np.ndarray, X_f32: np.ndarray) -> object:
        return self.reference_model.predict(X_f64)

    def predict_scores_c99(self, lib: NativeClassifierLibrary, X_f32: np.ndarray) -> np.ndarray:
        out = np.empty(X_f32.shape[0], dtype=np.float32)
        lib.raw.mcu_dense_dot_batch(
            lib.f32_ptr(X_f32),
            int(X_f32.shape[0]),
            int(X_f32.shape[1]),
            lib.f32_ptr(self._w32),
            float(self._b32),
            lib.f32_ptr(out),
        )
        return out.astype(np.float64)

    def measure_c99_inner_us(
        self,
        lib: NativeClassifierLibrary,
        X_f32: np.ndarray,
        inner_iters: int,
    ) -> float:
        x_ptr = lib.f32_ptr(X_f32)
        w_ptr = lib.f32_ptr(self._w32)
        n_samp, n_feat = int(X_f32.shape[0]), int(X_f32.shape[1])
        b_val = float(self._b32)
        return measure_c_inner_us_per_sample(
            lambda: lib.raw.mcu_bench_dense_dot(x_ptr, n_samp, n_feat, w_ptr, b_val, inner_iters),
            inner_iters=inner_iters,
        )


class FoldedLinearModel(FoldedMcuModel):
    """Algebraically folded dense linear dot-product classifier (LDA, LinearSVC, etc.)."""

    def __init__(
        self,
        name: str,
        reference_model: Any,
        w_eff: np.ndarray,
        b_eff: float,
        zero_weight_eps: float = 1e-12,
        extra_note: str = "",
    ) -> None:
        self.name = name
        self.display_group = "linear_folded"
        self.reference_model = reference_model
        self.extra_note = extra_note
        self.w64 = np.ascontiguousarray(w_eff, dtype=np.float64).ravel()
        self.b64 = float(b_eff)
        self.w32 = np.ascontiguousarray(self.w64, dtype=np.float32)
        self.b32 = np.float32(self.b64)
        nnz = int(np.count_nonzero(np.abs(self.w64) > zero_weight_eps))
        d = int(self.w64.size)
        self.cost_profile = McuCostProfile(
            model_family="linear_dense",
            n_params=d + 1,
            flash_bytes_f32=(d + 1) * 4,
            sram_scratch_bytes_f32=0,
            macs=d,
            adds=d,
            comparisons=1,
            math_calls_at_threshold=0,
            n_active_features=nnz,
        )

    def predict_scores_ref(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.reference_model.decision_function(X), dtype=np.float64).ravel()

    def predict_scores_f64(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(np.asarray(X, dtype=np.float64) @ self.w64 + self.b64, dtype=np.float64)

    def predict_scores_f32(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(np.asarray(X, dtype=np.float32) @ self.w32 + self.b32, dtype=np.float64)

    def predict_batch_for_latency(self, X_f64: np.ndarray, X_f32: np.ndarray) -> object:
        return (X_f64 @ self.w64 + self.b64) > 0.0

    def predict_scores_c99(self, lib: NativeClassifierLibrary, X_f32: np.ndarray) -> np.ndarray:
        out = np.empty(X_f32.shape[0], dtype=np.float32)
        lib.raw.mcu_dense_dot_batch(
            lib.f32_ptr(X_f32),
            int(X_f32.shape[0]),
            int(X_f32.shape[1]),
            lib.f32_ptr(self.w32),
            float(self.b32),
            lib.f32_ptr(out),
        )
        return out.astype(np.float64)

    def measure_c99_inner_us(
        self,
        lib: NativeClassifierLibrary,
        X_f32: np.ndarray,
        inner_iters: int,
    ) -> float:
        x_ptr = lib.f32_ptr(X_f32)
        w_ptr = lib.f32_ptr(self.w32)
        n_samp, n_feat = int(X_f32.shape[0]), int(X_f32.shape[1])
        b_val = float(self.b32)
        return measure_c_inner_us_per_sample(
            lambda: lib.raw.mcu_bench_dense_dot(x_ptr, n_samp, n_feat, w_ptr, b_val, inner_iters),
            inner_iters=inner_iters,
        )

    def export_c99_snippet(self, prefix: str) -> str:
        return (
            f"/* {self.name}: dense dot-product core (n={self.w32.size}) */\n"
            + _fmt_f32_array(f"{prefix}_w", self.w32)
            + f"\nstatic const float {prefix}_b = {float(self.b32):.8e}f;\n"
        )


class FoldedSparseLinearModel(FoldedMcuModel):
    """L1-sparsified linear dot-product classifier skipping zero-weight features."""

    def __init__(
        self,
        name: str,
        reference_model: Any,
        w_eff: np.ndarray,
        b_eff: float,
        zero_weight_eps: float = 1e-12,
    ) -> None:
        self.name = name
        self.display_group = "linear_sparse"
        self.reference_model = reference_model
        w64_full = np.ascontiguousarray(w_eff, dtype=np.float64).ravel()
        self.b64 = float(b_eff)
        self.b32 = np.float32(self.b64)
        idx = np.flatnonzero(np.abs(w64_full) > zero_weight_eps).astype(np.int32)
        self.idx_nz = np.ascontiguousarray(idx, dtype=np.int32)
        self.w_nz64 = np.ascontiguousarray(w64_full[self.idx_nz], dtype=np.float64)
        self.w_nz32 = np.ascontiguousarray(self.w_nz64, dtype=np.float32)
        nnz = int(self.idx_nz.size)
        d = int(w64_full.size)
        self.extra_note = f"nnz={nnz}/{d}"
        # Flash stores w_nz (4B), idx_nz (4B int32 to match C header), and bias (4B).
        self.cost_profile = McuCostProfile(
            model_family="linear_sparse",
            n_params=nnz + 1,
            flash_bytes_f32=8 * nnz + 4,
            sram_scratch_bytes_f32=0,
            macs=nnz,
            adds=nnz,
            comparisons=1,
            math_calls_at_threshold=0,
            n_active_features=nnz,
        )

    def predict_scores_ref(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.reference_model.decision_function(X), dtype=np.float64).ravel()

    def predict_scores_f64(self, X: np.ndarray) -> np.ndarray:
        if self.idx_nz.size == 0:
            return np.full(X.shape[0], self.b64, dtype=np.float64)
        return np.asarray(
            np.asarray(X, dtype=np.float64)[:, self.idx_nz] @ self.w_nz64 + self.b64,
            dtype=np.float64,
        )

    def predict_scores_f32(self, X: np.ndarray) -> np.ndarray:
        if self.idx_nz.size == 0:
            return np.full(X.shape[0], float(self.b32), dtype=np.float64)
        X32 = np.asarray(X, dtype=np.float32)
        return np.asarray(X32[:, self.idx_nz] @ self.w_nz32 + self.b32, dtype=np.float64)

    def predict_batch_for_latency(self, X_f64: np.ndarray, X_f32: np.ndarray) -> object:
        if self.idx_nz.size == 0:
            return np.full(X_f64.shape[0], self.b64 > 0.0, dtype=bool)
        return (X_f64[:, self.idx_nz] @ self.w_nz64 + self.b64) > 0.0

    def predict_scores_c99(self, lib: NativeClassifierLibrary, X_f32: np.ndarray) -> np.ndarray:
        out = np.empty(X_f32.shape[0], dtype=np.float32)
        lib.raw.mcu_sparse_dot_batch(
            lib.f32_ptr(X_f32),
            int(X_f32.shape[0]),
            int(X_f32.shape[1]),
            lib.f32_ptr(self.w_nz32),
            lib.i32_ptr(self.idx_nz),
            float(self.b32),
            int(self.idx_nz.size),
            lib.f32_ptr(out),
        )
        return out.astype(np.float64)

    def measure_c99_inner_us(
        self,
        lib: NativeClassifierLibrary,
        X_f32: np.ndarray,
        inner_iters: int,
    ) -> float:
        x_ptr = lib.f32_ptr(X_f32)
        w_ptr = lib.f32_ptr(self.w_nz32)
        idx_ptr = lib.i32_ptr(self.idx_nz)
        n_samp, n_feat = int(X_f32.shape[0]), int(X_f32.shape[1])
        b_val = float(self.b32)
        nnz = int(self.idx_nz.size)
        return measure_c_inner_us_per_sample(
            lambda: lib.raw.mcu_bench_sparse_dot(x_ptr, n_samp, n_feat, w_ptr, idx_ptr, b_val, nnz, inner_iters),
            inner_iters=inner_iters,
        )

    def export_c99_snippet(self, prefix: str) -> str:
        return (
            f"/* {self.name}: L1 sparse dot-product ({self.extra_note}) */\n"
            + _fmt_int_array("int32_t", f"{prefix}_idx", self.idx_nz)
            + "\n"
            + _fmt_f32_array(f"{prefix}_w_nz", self.w_nz32)
            + f"\nstatic const float {prefix}_b = {float(self.b32):.8e}f;\n"
        )


class FoldedDiagQuadModel(FoldedMcuModel):
    """GaussianNB folded analytically into c + sum x_i * (a_i * x_i + b_i) with 0 log/div."""

    def __init__(self, name: str, gnb: GaussianNB) -> None:
        if gnb.theta_.shape[0] != 2:
            raise ValueError("FoldedDiagQuadModel requires binary GaussianNB")
        self.name = name
        self.display_group = "quadratic_diag"
        self.reference_model = gnb
        self.extra_note = "horner_diag_llr"

        mu0 = np.asarray(gnb.theta_[0], dtype=np.float64)
        mu1 = np.asarray(gnb.theta_[1], dtype=np.float64)
        v0 = np.asarray(gnb.var_[0], dtype=np.float64)
        v1 = np.asarray(gnb.var_[1], dtype=np.float64)
        p0, p1 = float(gnb.class_prior_[0]), float(gnb.class_prior_[1])

        a = 0.5 * (1.0 / v0 - 1.0 / v1)
        b_lin = mu1 / v1 - mu0 / v0
        c_scalar = float(np.log(p1 / p0) + 0.5 * np.sum(np.log(v0 / v1) + (mu0 * mu0) / v0 - (mu1 * mu1) / v1))

        self.a64 = np.ascontiguousarray(a, dtype=np.float64)
        self.b64 = np.ascontiguousarray(b_lin, dtype=np.float64)
        self.c64 = c_scalar
        self.a32 = np.ascontiguousarray(self.a64, dtype=np.float32)
        self.b32 = np.ascontiguousarray(self.b64, dtype=np.float32)
        self.c32 = np.float32(self.c64)

        d = int(self.a64.size)
        self.cost_profile = McuCostProfile(
            model_family="quadratic_diag",
            n_params=2 * d + 1,
            flash_bytes_f32=(2 * d + 1) * 4,
            sram_scratch_bytes_f32=0,
            macs=2 * d,
            adds=2 * d,
            comparisons=1,
            math_calls_at_threshold=0,
            n_active_features=d,
        )

    def predict_scores_ref(self, X: np.ndarray) -> np.ndarray:
        jll = np.asarray(self.reference_model._joint_log_likelihood(X), dtype=np.float64)
        return jll[:, 1] - jll[:, 0]

    def predict_scores_f64(self, X: np.ndarray) -> np.ndarray:
        Xf = np.asarray(X, dtype=np.float64)
        return np.asarray(np.sum(Xf * (Xf * self.a64 + self.b64), axis=1) + self.c64, dtype=np.float64)

    def numerical_scale_bound(self) -> float:
        """GaussianNB log-ratio offset c can cancel large opposite-signed per-feature sums."""
        return float(max(1.0, abs(self.c64)))

    def predict_scores_f32(self, X: np.ndarray) -> np.ndarray:
        Xf = np.asarray(X, dtype=np.float32)
        return np.asarray(
            np.sum(Xf * (Xf * self.a32 + self.b32), axis=1, dtype=np.float32) + self.c32,
            dtype=np.float64,
        )

    def predict_batch_for_latency(self, X_f64: np.ndarray, X_f32: np.ndarray) -> object:
        return (np.sum(X_f64 * (X_f64 * self.a64 + self.b64), axis=1) + self.c64) > 0.0

    def predict_scores_c99(self, lib: NativeClassifierLibrary, X_f32: np.ndarray) -> np.ndarray:
        out = np.empty(X_f32.shape[0], dtype=np.float32)
        lib.raw.mcu_diag_quad_batch(
            lib.f32_ptr(X_f32),
            int(X_f32.shape[0]),
            int(X_f32.shape[1]),
            lib.f32_ptr(self.a32),
            lib.f32_ptr(self.b32),
            float(self.c32),
            lib.f32_ptr(out),
        )
        return out.astype(np.float64)

    def measure_c99_inner_us(
        self,
        lib: NativeClassifierLibrary,
        X_f32: np.ndarray,
        inner_iters: int,
    ) -> float:
        x_ptr = lib.f32_ptr(X_f32)
        a_ptr = lib.f32_ptr(self.a32)
        b_ptr = lib.f32_ptr(self.b32)
        n_samp, n_feat = int(X_f32.shape[0]), int(X_f32.shape[1])
        c_val = float(self.c32)
        return measure_c_inner_us_per_sample(
            lambda: lib.raw.mcu_bench_diag_quad(x_ptr, n_samp, n_feat, a_ptr, b_ptr, c_val, inner_iters),
            inner_iters=inner_iters,
        )

    def export_c99_snippet(self, prefix: str) -> str:
        return (
            f"/* {self.name}: diagonal quadratic GaussianNB LLR (0 log/div at inference) */\n"
            + _fmt_f32_array(f"{prefix}_a", self.a32)
            + "\n"
            + _fmt_f32_array(f"{prefix}_b", self.b32)
            + f"\nstatic const float {prefix}_c = {float(self.c32):.8e}f;\n"
        )


def _flatten_sklearn_trees(
    trees: list[Any],
    is_classifier_proba: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, set[int]]:
    """Flatten sklearn Tree objects into contiguous C99-indexed node tables."""
    offsets: list[int] = []
    feats: list[int] = []
    thresh: list[float] = []
    lefts: list[int] = []
    rights: list[int] = []
    vals: list[float] = []
    max_depth_sum = 0
    used_features: set[int] = set()

    cursor = 0
    for tr in trees:
        offsets.append(cursor)
        n = int(tr.node_count)
        max_depth_sum += int(tr.max_depth)
        f_arr = tr.feature
        t_arr = tr.threshold
        l_arr = tr.children_left
        r_arr = tr.children_right
        v_arr = tr.value
        for node in range(n):
            f_id = int(f_arr[node])
            l_id = int(l_arr[node])
            r_id = int(r_arr[node])
            feats.append(f_id if f_id >= 0 else -1)
            if f_id >= 0:
                used_features.add(f_id)
            thresh.append(float(t_arr[node]) if l_id >= 0 else 0.0)
            lefts.append(cursor + l_id if l_id >= 0 else -1)
            rights.append(cursor + r_id if r_id >= 0 else -1)
            if is_classifier_proba:
                counts = np.asarray(v_arr[node, 0], dtype=np.float64)
                tot = float(np.sum(counts))
                vals.append(float(counts[1] / tot) if tot > 0.0 else 0.0)
            else:
                vals.append(float(v_arr[node, 0, 0]))
        cursor += n

    return (
        np.ascontiguousarray(offsets, dtype=np.int32),
        np.ascontiguousarray(feats, dtype=np.int16),
        np.ascontiguousarray(thresh, dtype=np.float64),
        np.ascontiguousarray(lefts, dtype=np.int16),
        np.ascontiguousarray(rights, dtype=np.int16),
        np.ascontiguousarray(vals, dtype=np.float64),
        max_depth_sum,
        used_features,
    )


class FoldedTreeEnsembleModel(FoldedMcuModel):
    """Flat C-table decision tree or boosted tree ensemble (0 MACs at inference)."""

    def __init__(
        self,
        name: str,
        reference_model: Any,
        trees: list[Any],
        base_score: float,
        scale: float,
        is_classifier_proba: bool,
        model_family: str,
        extra_note: str = "",
    ) -> None:
        self.name = name
        self.display_group = "tree_core"
        self.reference_model = reference_model
        self.is_classifier_proba = is_classifier_proba
        self.base64 = float(base_score)
        self.scale64 = float(scale)
        self.base32 = np.float32(self.base64)
        self.scale32 = np.float32(self.scale64)

        (
            self.tree_offsets,
            self.features,
            self.thresholds64,
            self.left_children,
            self.right_children,
            self.leaf_values64,
            max_depth_sum,
            used_features,
        ) = _flatten_sklearn_trees(trees, is_classifier_proba=is_classifier_proba)

        self.thresholds32 = np.ascontiguousarray(self.thresholds64, dtype=np.float32)
        self.leaf_values32 = np.ascontiguousarray(self.leaf_values64, dtype=np.float32)
        n_nodes = int(self.features.size)
        n_trees = int(self.tree_offsets.size)
        self.extra_note = extra_note or f"trees={n_trees},nodes={n_nodes}"
        # Per node: int16 feat + float32 thresh + int16 left + int16 right + float32 leaf = 14B.
        flash_bytes = n_nodes * 14 + n_trees * 4 + 8
        self.cost_profile = McuCostProfile(
            model_family=model_family,
            n_params=n_nodes,
            flash_bytes_f32=flash_bytes,
            sram_scratch_bytes_f32=0,
            macs=0,
            adds=max(0, n_trees),
            comparisons=max_depth_sum + 1,
            math_calls_at_threshold=0,
            n_active_features=len(used_features),
        )

    def predict_scores_ref(self, X: np.ndarray) -> np.ndarray:
        if self.is_classifier_proba:
            return np.asarray(self.reference_model.predict_proba(X)[:, 1], dtype=np.float64) - 0.5
        return np.asarray(self.reference_model.decision_function(X), dtype=np.float64).ravel()

    def _eval_py(self, X: np.ndarray, thresholds: np.ndarray, leaf_values: np.ndarray, base: float, scale: float) -> np.ndarray:
        X_arr = np.asarray(X)
        out = np.empty(X_arr.shape[0], dtype=np.float64)
        offs = self.tree_offsets
        feats = self.features
        lefts = self.left_children
        rights = self.right_children
        for row in range(X_arr.shape[0]):
            xr = X_arr[row]
            s = 0.0
            for t in range(offs.size):
                node = int(offs[t])
                while lefts[node] >= 0:
                    f_id = int(feats[node])
                    node = int(lefts[node]) if xr[f_id] <= thresholds[node] else int(rights[node])
                s += float(leaf_values[node])
            out[row] = base + scale * s
        return out

    def predict_scores_f64(self, X: np.ndarray) -> np.ndarray:
        # sklearn Cython tree routines cast X to float32 internally before splitting.
        X_sk = np.asarray(X, dtype=np.float32)
        return self._eval_py(X_sk, self.thresholds64, self.leaf_values64, self.base64, self.scale64)

    def predict_scores_f32(self, X: np.ndarray) -> np.ndarray:
        X32 = np.asarray(X, dtype=np.float32)
        return self._eval_py(X32, self.thresholds32, self.leaf_values32, float(self.base32), float(self.scale32))

    def predict_probs_and_labels(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = self.predict_scores_f64(X)
        if self.is_classifier_proba:
            probs = np.clip(scores + 0.5, 0.0, 1.0)
            return probs, (scores > 0.0).astype(np.int64)
        return sigmoid_stable(scores), (scores > 0.0).astype(np.int64)

    def predict_batch_for_latency(self, X_f64: np.ndarray, X_f32: np.ndarray) -> object:
        # Use sklearn's compiled Cython tree kernel directly for fast Python-side timing,
        # avoiding pure-Python loop artifact; C99 DLL is also timed separately via ctypes.
        if self.is_classifier_proba:
            return self.reference_model.tree_.predict(X_f32)
        return self.reference_model.decision_function(X_f64)

    def predict_scores_c99(self, lib: NativeClassifierLibrary, X_f32: np.ndarray) -> np.ndarray:
        out = np.empty(X_f32.shape[0], dtype=np.float32)
        lib.raw.mcu_tree_ensemble_batch(
            lib.f32_ptr(X_f32),
            int(X_f32.shape[0]),
            int(X_f32.shape[1]),
            int(self.tree_offsets.size),
            lib.i32_ptr(self.tree_offsets),
            lib.i16_ptr(self.features),
            lib.f32_ptr(self.thresholds32),
            lib.i16_ptr(self.left_children),
            lib.i16_ptr(self.right_children),
            lib.f32_ptr(self.leaf_values32),
            float(self.base32),
            float(self.scale32),
            lib.f32_ptr(out),
        )
        return out.astype(np.float64)

    def measure_c99_inner_us(
        self,
        lib: NativeClassifierLibrary,
        X_f32: np.ndarray,
        inner_iters: int,
    ) -> float:
        x_ptr = lib.f32_ptr(X_f32)
        offs_ptr = lib.i32_ptr(self.tree_offsets)
        feat_ptr = lib.i16_ptr(self.features)
        th_ptr = lib.f32_ptr(self.thresholds32)
        left_ptr = lib.i16_ptr(self.left_children)
        right_ptr = lib.i16_ptr(self.right_children)
        val_ptr = lib.f32_ptr(self.leaf_values32)
        n_samp, n_feat = int(X_f32.shape[0]), int(X_f32.shape[1])
        n_trees = int(self.tree_offsets.size)
        b_val = float(self.base32)
        s_val = float(self.scale32)
        return measure_c_inner_us_per_sample(
            lambda: lib.raw.mcu_bench_tree_ensemble(
                x_ptr, n_samp, n_feat, n_trees,
                offs_ptr, feat_ptr, th_ptr, left_ptr, right_ptr, val_ptr,
                b_val, s_val, inner_iters,
            ),
            inner_iters=inner_iters,
        )

    def export_c99_snippet(self, prefix: str) -> str:
        return (
            f"/* {self.name}: flat C99 node table ({self.extra_note}, 0 MACs) */\n"
            + f"static const int32_t {prefix}_n_trees = {self.tree_offsets.size};\n"
            + f"static const float {prefix}_base = {float(self.base32):.8e}f;\n"
            + f"static const float {prefix}_scale = {float(self.scale32):.8e}f;\n"
            + _fmt_int_array("int32_t", f"{prefix}_offsets", self.tree_offsets)
            + "\n"
            + _fmt_int_array("int16_t", f"{prefix}_features", self.features)
            + "\n"
            + _fmt_f32_array(f"{prefix}_thresholds", self.thresholds32)
            + "\n"
            + _fmt_int_array("int16_t", f"{prefix}_left", self.left_children)
            + "\n"
            + _fmt_int_array("int16_t", f"{prefix}_right", self.right_children)
            + "\n"
            + _fmt_f32_array(f"{prefix}_leaf_values", self.leaf_values32)
            + "\n"
        )


class _TorchMlp1Hidden(nn.Module):
    """Single hidden layer ReLU MLP supporting optional linear skip path."""

    def __init__(self, d_in: int, d_hidden: int, dropout: float, use_skip: bool) -> None:
        super().__init__()
        self.use_skip = use_skip
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(d_hidden, 1, bias=not use_skip)
        self.skip = nn.Linear(d_in, 1) if use_skip else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.fc1(x)))
        out = self.fc2(h)
        if self.skip is not None:
            out = out + self.skip(x)
        return out


class _TorchMlp2Hidden(nn.Module):
    """Two hidden layer ReLU MLP trained with balanced weighted BCE loss."""

    def __init__(self, d_in: int, h1: int, h2: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, h1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(h1, h2)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.drop1(self.act1(self.fc1(x)))
        z2 = self.drop2(self.act2(self.fc2(z1)))
        return self.fc3(z2)


def _train_torch_binary_mlp(
    net_factory: Any,
    scaler: StandardScaler,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    train_threads: int,
) -> nn.Module:
    """Deterministically initialize and fit compact PyTorch MLP with pos_weight balancing."""
    torch.manual_seed(seed)
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, int(train_threads)))
    try:
        net = net_factory()
        Xt = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.float32)
        n_pos = float(np.sum(y_train == 1))
        n_neg = float(np.sum(y_train == 0))
        pos_w = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = net(Xt).squeeze(1)
            loss = loss_fn(logits, yt)
            loss.backward()
            opt.step()
        net.eval()
        return net
    finally:
        torch.set_num_threads(prev_threads)


class FoldedMlp1HiddenModel(FoldedMcuModel):
    """Scaler-folded 1-hidden-layer ReLU MLP with zero runtime scaling overhead."""

    def __init__(
        self,
        name: str,
        net: _TorchMlp1Hidden,
        scaler: StandardScaler,
    ) -> None:
        self.name = name
        self.display_group = "mlp_folded"
        self.reference_model = (net, scaler)
        self.extra_note = f"arch={net.fc1.in_features}->{net.fc1.out_features}->1"

        mu, sigma = _safe_scaler_stats(scaler)
        inv_sigma = 1.0 / sigma
        with torch.no_grad():
            w1_raw = net.fc1.weight.detach().cpu().numpy().astype(np.float64)
            b1_raw = net.fc1.bias.detach().cpu().numpy().astype(np.float64)
            w2_raw = net.fc2.weight.detach().cpu().numpy().astype(np.float64).ravel()
            b2_raw = float(net.fc2.bias.detach().cpu().numpy().astype(np.float64).item())

        self.w1_64 = np.ascontiguousarray(w1_raw * inv_sigma[None, :], dtype=np.float64)
        self.b1_64 = np.ascontiguousarray(b1_raw - self.w1_64 @ mu, dtype=np.float64)
        self.w2_64 = np.ascontiguousarray(w2_raw, dtype=np.float64)
        self.b2_64 = b2_raw

        self.w1_32 = np.ascontiguousarray(self.w1_64, dtype=np.float32)
        self.b1_32 = np.ascontiguousarray(self.b1_64, dtype=np.float32)
        self.w2_32 = np.ascontiguousarray(self.w2_64, dtype=np.float32)
        self.b2_32 = np.float32(self.b2_64)

        d_in, d_h = self.w1_64.shape[1], self.w1_64.shape[0]
        n_params = d_h * d_in + d_h + d_h + 1
        self.cost_profile = McuCostProfile(
            model_family="mlp_1h",
            n_params=n_params,
            flash_bytes_f32=n_params * 4,
            sram_scratch_bytes_f32=d_h * 4,
            macs=d_h * d_in + d_h,
            adds=d_h * d_in + d_h,
            comparisons=d_h + 1,
            math_calls_at_threshold=0,
            n_active_features=d_in,
        )

    def predict_scores_ref(self, X: np.ndarray) -> np.ndarray:
        net, scaler = self.reference_model
        X_scaled = torch.tensor(scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():
            return net(X_scaled).squeeze(1).cpu().numpy().astype(np.float64)

    def predict_scores_f64(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, np.asarray(X, dtype=np.float64) @ self.w1_64.T + self.b1_64)
        return np.asarray(h @ self.w2_64 + self.b2_64, dtype=np.float64)

    def predict_scores_f32(self, X: np.ndarray) -> np.ndarray:
        Xf = np.asarray(X, dtype=np.float32)
        h = np.maximum(np.float32(0.0), Xf @ self.w1_32.T + self.b1_32)
        return np.asarray(h @ self.w2_32 + self.b2_32, dtype=np.float64)

    def predict_batch_for_latency(self, X_f64: np.ndarray, X_f32: np.ndarray) -> object:
        h = np.maximum(0.0, X_f64 @ self.w1_64.T + self.b1_64)
        return (h @ self.w2_64 + self.b2_64) > 0.0

    def predict_scores_c99(self, lib: NativeClassifierLibrary, X_f32: np.ndarray) -> np.ndarray:
        out = np.empty(X_f32.shape[0], dtype=np.float32)
        d_h, d_in = int(self.w1_32.shape[0]), int(self.w1_32.shape[1])
        lib.raw.mcu_mlp_1hidden_batch(
            lib.f32_ptr(X_f32),
            int(X_f32.shape[0]),
            d_in,
            d_h,
            lib.f32_ptr(self.w1_32),
            lib.f32_ptr(self.b1_32),
            lib.f32_ptr(self.w2_32),
            float(self.b2_32),
            lib.f32_ptr(out),
        )
        return out.astype(np.float64)

    def measure_c99_inner_us(
        self,
        lib: NativeClassifierLibrary,
        X_f32: np.ndarray,
        inner_iters: int,
    ) -> float:
        x_ptr = lib.f32_ptr(X_f32)
        w1_ptr = lib.f32_ptr(self.w1_32)
        b1_ptr = lib.f32_ptr(self.b1_32)
        w2_ptr = lib.f32_ptr(self.w2_32)
        n_samp, d_in = int(X_f32.shape[0]), int(X_f32.shape[1])
        d_h = int(self.w1_32.shape[0])
        b2_val = float(self.b2_32)
        return measure_c_inner_us_per_sample(
            lambda: lib.raw.mcu_bench_mlp_1hidden(
                x_ptr, n_samp, d_in, d_h, w1_ptr, b1_ptr, w2_ptr, b2_val, inner_iters
            ),
            inner_iters=inner_iters,
        )

    def export_c99_snippet(self, prefix: str) -> str:
        d_h, d_in = self.w1_32.shape
        return (
            f"/* {self.name}: scaler folded into Layer 1 ({d_in}->{d_h}->1, ReLU) */\n"
            + f"static const int32_t {prefix}_n_in = {d_in};\n"
            + f"static const int32_t {prefix}_n_h = {d_h};\n"
            + _fmt_f32_array(f"{prefix}_w1", self.w1_32)
            + "\n"
            + _fmt_f32_array(f"{prefix}_b1", self.b1_32)
            + "\n"
            + _fmt_f32_array(f"{prefix}_w2", self.w2_32)
            + f"\nstatic const float {prefix}_b2 = {float(self.b2_32):.8e}f;\n"
        )


class FoldedMlp1HiddenSkipModel(FoldedMcuModel):
    """Scaler-folded residual skip + 1-hidden-layer ReLU MLP."""

    def __init__(
        self,
        name: str,
        net: _TorchMlp1Hidden,
        scaler: StandardScaler,
    ) -> None:
        if net.skip is None:
            raise ValueError("Expected skip branch on _TorchMlp1Hidden")
        self.name = name
        self.display_group = "mlp_folded"
        self.reference_model = (net, scaler)
        self.extra_note = f"arch={net.fc1.in_features}->{net.fc1.out_features}->1+skip"

        mu, sigma = _safe_scaler_stats(scaler)
        inv_sigma = 1.0 / sigma
        with torch.no_grad():
            w1_raw = net.fc1.weight.detach().cpu().numpy().astype(np.float64)
            b1_raw = net.fc1.bias.detach().cpu().numpy().astype(np.float64)
            w2_raw = net.fc2.weight.detach().cpu().numpy().astype(np.float64).ravel()
            w_skip_raw = net.skip.weight.detach().cpu().numpy().astype(np.float64).ravel()
            b_skip_raw = float(net.skip.bias.detach().cpu().numpy().astype(np.float64).item())

        self.w1_64 = np.ascontiguousarray(w1_raw * inv_sigma[None, :], dtype=np.float64)
        self.b1_64 = np.ascontiguousarray(b1_raw - self.w1_64 @ mu, dtype=np.float64)
        self.w2_64 = np.ascontiguousarray(w2_raw, dtype=np.float64)
        self.w_skip_64 = np.ascontiguousarray(w_skip_raw * inv_sigma, dtype=np.float64)
        self.b_skip_64 = float(b_skip_raw - self.w_skip_64 @ mu)

        self.w1_32 = np.ascontiguousarray(self.w1_64, dtype=np.float32)
        self.b1_32 = np.ascontiguousarray(self.b1_64, dtype=np.float32)
        self.w2_32 = np.ascontiguousarray(self.w2_64, dtype=np.float32)
        self.w_skip_32 = np.ascontiguousarray(self.w_skip_64, dtype=np.float32)
        self.b_skip_32 = np.float32(self.b_skip_64)

        d_in, d_h = self.w1_64.shape[1], self.w1_64.shape[0]
        n_params = d_h * d_in + d_h + d_h + d_in + 1
        self.cost_profile = McuCostProfile(
            model_family="mlp_1h_skip",
            n_params=n_params,
            flash_bytes_f32=n_params * 4,
            sram_scratch_bytes_f32=d_h * 4,
            macs=d_h * d_in + d_h + d_in,
            adds=d_h * d_in + d_h + d_in,
            comparisons=d_h + 1,
            math_calls_at_threshold=0,
            n_active_features=d_in,
        )

    def predict_scores_ref(self, X: np.ndarray) -> np.ndarray:
        net, scaler = self.reference_model
        X_scaled = torch.tensor(scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():
            return net(X_scaled).squeeze(1).cpu().numpy().astype(np.float64)

    def predict_scores_f64(self, X: np.ndarray) -> np.ndarray:
        Xf = np.asarray(X, dtype=np.float64)
        lin = Xf @ self.w_skip_64 + self.b_skip_64
        h = np.maximum(0.0, Xf @ self.w1_64.T + self.b1_64)
        return np.asarray(lin + h @ self.w2_64, dtype=np.float64)

    def predict_scores_f32(self, X: np.ndarray) -> np.ndarray:
        Xf = np.asarray(X, dtype=np.float32)
        lin = Xf @ self.w_skip_32 + self.b_skip_32
        h = np.maximum(np.float32(0.0), Xf @ self.w1_32.T + self.b1_32)
        return np.asarray(lin + h @ self.w2_32, dtype=np.float64)

    def predict_batch_for_latency(self, X_f64: np.ndarray, X_f32: np.ndarray) -> object:
        h = np.maximum(0.0, X_f64 @ self.w1_64.T + self.b1_64)
        return (X_f64 @ self.w_skip_64 + self.b_skip_64 + h @ self.w2_64) > 0.0

    def predict_scores_c99(self, lib: NativeClassifierLibrary, X_f32: np.ndarray) -> np.ndarray:
        out = np.empty(X_f32.shape[0], dtype=np.float32)
        d_h, d_in = int(self.w1_32.shape[0]), int(self.w1_32.shape[1])
        lib.raw.mcu_mlp_1hidden_skip_batch(
            lib.f32_ptr(X_f32),
            int(X_f32.shape[0]),
            d_in,
            d_h,
            lib.f32_ptr(self.w_skip_32),
            float(self.b_skip_32),
            lib.f32_ptr(self.w1_32),
            lib.f32_ptr(self.b1_32),
            lib.f32_ptr(self.w2_32),
            lib.f32_ptr(out),
        )
        return out.astype(np.float64)

    def measure_c99_inner_us(
        self,
        lib: NativeClassifierLibrary,
        X_f32: np.ndarray,
        inner_iters: int,
    ) -> float:
        x_ptr = lib.f32_ptr(X_f32)
        ws_ptr = lib.f32_ptr(self.w_skip_32)
        w1_ptr = lib.f32_ptr(self.w1_32)
        b1_ptr = lib.f32_ptr(self.b1_32)
        w2_ptr = lib.f32_ptr(self.w2_32)
        n_samp, d_in = int(X_f32.shape[0]), int(X_f32.shape[1])
        d_h = int(self.w1_32.shape[0])
        bs_val = float(self.b_skip_32)
        return measure_c_inner_us_per_sample(
            lambda: lib.raw.mcu_bench_mlp_1hidden_skip(
                x_ptr, n_samp, d_in, d_h, ws_ptr, bs_val, w1_ptr, b1_ptr, w2_ptr, inner_iters
            ),
            inner_iters=inner_iters,
        )

    def export_c99_snippet(self, prefix: str) -> str:
        d_h, d_in = self.w1_32.shape
        return (
            f"/* {self.name}: scaler folded into skip + hidden Layer 1 ({d_in}->{d_h}->1+skip) */\n"
            + _fmt_f32_array(f"{prefix}_w_skip", self.w_skip_32)
            + f"\nstatic const float {prefix}_b_skip = {float(self.b_skip_32):.8e}f;\n"
            + _fmt_f32_array(f"{prefix}_w1", self.w1_32)
            + "\n"
            + _fmt_f32_array(f"{prefix}_b1", self.b1_32)
            + "\n"
            + _fmt_f32_array(f"{prefix}_w2", self.w2_32)
            + "\n"
        )


class FoldedMlp2HiddenModel(FoldedMcuModel):
    """Scaler-folded 2-hidden-layer ReLU MLP (94->32->16->1)."""

    def __init__(
        self,
        name: str,
        net: _TorchMlp2Hidden,
        scaler: StandardScaler,
    ) -> None:
        self.name = name
        self.display_group = "mlp_folded"
        self.reference_model = (net, scaler)
        h1, h2 = net.fc1.out_features, net.fc2.out_features
        d_in = net.fc1.in_features
        self.extra_note = f"arch={d_in}->{h1}->{h2}->1"

        mu, sigma = _safe_scaler_stats(scaler)
        inv_sigma = 1.0 / sigma
        with torch.no_grad():
            w1_raw = net.fc1.weight.detach().cpu().numpy().astype(np.float64)
            b1_raw = net.fc1.bias.detach().cpu().numpy().astype(np.float64)
            w2_raw = net.fc2.weight.detach().cpu().numpy().astype(np.float64)
            b2_raw = net.fc2.bias.detach().cpu().numpy().astype(np.float64)
            w3_raw = net.fc3.weight.detach().cpu().numpy().astype(np.float64).ravel()
            b3_raw = float(net.fc3.bias.detach().cpu().numpy().astype(np.float64).item())

        self.w1_64 = np.ascontiguousarray(w1_raw * inv_sigma[None, :], dtype=np.float64)
        self.b1_64 = np.ascontiguousarray(b1_raw - self.w1_64 @ mu, dtype=np.float64)
        self.w2_64 = np.ascontiguousarray(w2_raw, dtype=np.float64)
        self.b2_64 = np.ascontiguousarray(b2_raw, dtype=np.float64)
        self.w3_64 = np.ascontiguousarray(w3_raw, dtype=np.float64)
        self.b3_64 = b3_raw

        self.w1_32 = np.ascontiguousarray(self.w1_64, dtype=np.float32)
        self.b1_32 = np.ascontiguousarray(self.b1_64, dtype=np.float32)
        self.w2_32 = np.ascontiguousarray(self.w2_64, dtype=np.float32)
        self.b2_32 = np.ascontiguousarray(self.b2_64, dtype=np.float32)
        self.w3_32 = np.ascontiguousarray(self.w3_64, dtype=np.float32)
        self.b3_32 = np.float32(self.b3_64)

        n_params = h1 * d_in + h1 + h2 * h1 + h2 + h2 + 1
        macs = h1 * d_in + h2 * h1 + h2
        self.cost_profile = McuCostProfile(
            model_family="mlp_2h",
            n_params=n_params,
            flash_bytes_f32=n_params * 4,
            sram_scratch_bytes_f32=(h1 + h2) * 4,
            macs=macs,
            adds=macs,
            comparisons=h1 + h2 + 1,
            math_calls_at_threshold=0,
            n_active_features=d_in,
        )

    def predict_scores_ref(self, X: np.ndarray) -> np.ndarray:
        net, scaler = self.reference_model
        X_scaled = torch.tensor(scaler.transform(X), dtype=torch.float32)
        with torch.no_grad():
            return net(X_scaled).squeeze(1).cpu().numpy().astype(np.float64)

    def predict_scores_f64(self, X: np.ndarray) -> np.ndarray:
        z1 = np.maximum(0.0, np.asarray(X, dtype=np.float64) @ self.w1_64.T + self.b1_64)
        z2 = np.maximum(0.0, z1 @ self.w2_64.T + self.b2_64)
        return np.asarray(z2 @ self.w3_64 + self.b3_64, dtype=np.float64)

    def predict_scores_f32(self, X: np.ndarray) -> np.ndarray:
        Xf = np.asarray(X, dtype=np.float32)
        z1 = np.maximum(np.float32(0.0), Xf @ self.w1_32.T + self.b1_32)
        z2 = np.maximum(np.float32(0.0), z1 @ self.w2_32.T + self.b2_32)
        return np.asarray(z2 @ self.w3_32 + self.b3_32, dtype=np.float64)

    def predict_batch_for_latency(self, X_f64: np.ndarray, X_f32: np.ndarray) -> object:
        z1 = np.maximum(0.0, X_f64 @ self.w1_64.T + self.b1_64)
        z2 = np.maximum(0.0, z1 @ self.w2_64.T + self.b2_64)
        return (z2 @ self.w3_64 + self.b3_64) > 0.0

    def predict_scores_c99(self, lib: NativeClassifierLibrary, X_f32: np.ndarray) -> np.ndarray:
        out = np.empty(X_f32.shape[0], dtype=np.float32)
        h1, d_in = int(self.w1_32.shape[0]), int(self.w1_32.shape[1])
        h2 = int(self.w2_32.shape[0])
        lib.raw.mcu_mlp_2hidden_batch(
            lib.f32_ptr(X_f32),
            int(X_f32.shape[0]),
            d_in,
            h1,
            h2,
            lib.f32_ptr(self.w1_32),
            lib.f32_ptr(self.b1_32),
            lib.f32_ptr(self.w2_32),
            lib.f32_ptr(self.b2_32),
            lib.f32_ptr(self.w3_32),
            float(self.b3_32),
            lib.f32_ptr(out),
        )
        return out.astype(np.float64)

    def measure_c99_inner_us(
        self,
        lib: NativeClassifierLibrary,
        X_f32: np.ndarray,
        inner_iters: int,
    ) -> float:
        x_ptr = lib.f32_ptr(X_f32)
        w1_ptr = lib.f32_ptr(self.w1_32)
        b1_ptr = lib.f32_ptr(self.b1_32)
        w2_ptr = lib.f32_ptr(self.w2_32)
        b2_ptr = lib.f32_ptr(self.b2_32)
        w3_ptr = lib.f32_ptr(self.w3_32)
        n_samp, d_in = int(X_f32.shape[0]), int(X_f32.shape[1])
        h1, h2 = int(self.w1_32.shape[0]), int(self.w2_32.shape[0])
        b3_val = float(self.b3_32)
        return measure_c_inner_us_per_sample(
            lambda: lib.raw.mcu_bench_mlp_2hidden(
                x_ptr, n_samp, d_in, h1, h2,
                w1_ptr, b1_ptr, w2_ptr, b2_ptr, w3_ptr, b3_val, inner_iters,
            ),
            inner_iters=inner_iters,
        )

    def export_c99_snippet(self, prefix: str) -> str:
        h1, d_in = self.w1_32.shape
        h2 = self.w2_32.shape[0]
        return (
            f"/* {self.name}: scaler folded 2-hidden ReLU MLP ({d_in}->{h1}->{h2}->1) */\n"
            + _fmt_f32_array(f"{prefix}_w1", self.w1_32)
            + "\n"
            + _fmt_f32_array(f"{prefix}_b1", self.b1_32)
            + "\n"
            + _fmt_f32_array(f"{prefix}_w2", self.w2_32)
            + "\n"
            + _fmt_f32_array(f"{prefix}_b2", self.b2_32)
            + "\n"
            + _fmt_f32_array(f"{prefix}_w3", self.w3_32)
            + f"\nstatic const float {prefix}_b3 = {float(self.b3_32):.8e}f;\n"
        )


def _fold_linear_pipeline(scaler: StandardScaler | None, w_s: np.ndarray, b_s: float) -> tuple[np.ndarray, float]:
    """Fold affine linear weights and intercept around an optional StandardScaler."""
    w = np.asarray(w_s, dtype=np.float64).ravel()
    b = float(b_s)
    if scaler is None:
        return w, b
    mu, sigma = _safe_scaler_stats(scaler)
    w_eff = w / sigma
    b_eff = float(b - np.dot(w_eff, mu))
    return w_eff, b_eff


def train_and_build_mcu_model(
    model_name: str,
    partition: DatasetPartition,
    src_cfg: ClassificationConfig,
    models_cfg: ModelsConfig,
    thresholds: ThresholdsConfig,
    seed: int,
) -> FoldedMcuModel:
    """Factory training one classifier cleanly on partition.X_train and folding into MCU core."""
    X_tr, y_tr = partition.X_train, partition.y_train
    d_in = int(X_tr.shape[1])

    if model_name in ("lr_dot", "dt", "rf"):
        src_cfg_local = src_cfg if model_name != "rf" else replace(
            src_cfg, rf_params=replace(src_cfg.rf_params, n_jobs=1)
        )
        est, _ = train_and_evaluate(partition, model_name, src_cfg_local)
        return BaselineSrcModelWrapper(
            name=model_name,
            estimator=est,
            n_features=d_in,
            zero_weight_eps=thresholds.zero_weight_eps,
        )

    hp = models_cfg.params_by_model.get(model_name, {})

    if model_name == "lr_l1_dot":
        from sklearn.linear_model import LogisticRegression

        scaler = StandardScaler() if bool(hp.get("scale_features", True)) else None
        lr = LogisticRegression(
            C=float(hp["C"]),
            penalty="l1",
            solver=str(hp["solver"]),
            class_weight=str(hp["class_weight"]),
            max_iter=int(hp["max_iter"]),
            random_state=seed,
        )
        pipe: Any = make_pipeline(scaler, lr) if scaler is not None else lr
        pipe.fit(X_tr, y_tr)
        fitted_scaler = pipe.named_steps["standardscaler"] if isinstance(pipe, Pipeline) else None
        fitted_lr = pipe.named_steps["logisticregression"] if isinstance(pipe, Pipeline) else pipe
        w_eff, b_eff = _fold_linear_pipeline(fitted_scaler, fitted_lr.coef_, float(fitted_lr.intercept_[0]))
        return FoldedSparseLinearModel(
            name=model_name,
            reference_model=pipe,
            w_eff=w_eff,
            b_eff=b_eff,
            zero_weight_eps=thresholds.zero_weight_eps,
        )

    if model_name == "lda_shrink_dot":
        lda = LinearDiscriminantAnalysis(
            solver=str(hp["solver"]),
            shrinkage=hp["shrinkage"],
            priors=list(float(p) for p in hp["priors"]),
        )
        lda.fit(X_tr, y_tr)
        return FoldedLinearModel(
            name=model_name,
            reference_model=lda,
            w_eff=np.asarray(lda.coef_[0], dtype=np.float64),
            b_eff=float(lda.intercept_[0]),
            zero_weight_eps=thresholds.zero_weight_eps,
            extra_note="lsqr_shrinkage_auto",
        )

    if model_name == "lsvc_dot":
        scaler = StandardScaler() if bool(hp.get("scale_features", True)) else None
        lsvc = LinearSVC(
            C=float(hp["C"]),
            class_weight=str(hp["class_weight"]),
            max_iter=int(hp["max_iter"]),
            random_state=seed,
        )
        pipe = make_pipeline(scaler, lsvc) if scaler is not None else lsvc
        pipe.fit(X_tr, y_tr)
        fitted_scaler = pipe.named_steps["standardscaler"] if isinstance(pipe, Pipeline) else None
        fitted_svc = pipe.named_steps["linearsvc"] if isinstance(pipe, Pipeline) else pipe
        w_eff, b_eff = _fold_linear_pipeline(fitted_scaler, fitted_svc.coef_, float(fitted_svc.intercept_[0]))
        return FoldedLinearModel(
            name=model_name,
            reference_model=pipe,
            w_eff=w_eff,
            b_eff=b_eff,
            zero_weight_eps=thresholds.zero_weight_eps,
            extra_note=f"hinge_C={float(hp['C'])}",
        )

    if model_name == "gnb_quad":
        gnb = GaussianNB(
            priors=list(float(p) for p in hp["priors"]),
            var_smoothing=float(hp["var_smoothing"]),
        )
        gnb.fit(X_tr, y_tr)
        return FoldedDiagQuadModel(name=model_name, gnb=gnb)

    if model_name == "dt_core":
        dt = DecisionTreeClassifier(
            max_depth=int(hp["max_depth"]),
            min_samples_leaf=int(hp["min_samples_leaf"]),
            class_weight=str(hp["class_weight"]),
            random_state=seed,
        )
        dt.fit(X_tr, y_tr)
        return FoldedTreeEnsembleModel(
            name=model_name,
            reference_model=dt,
            trees=[dt.tree_],
            base_score=-0.5,
            scale=1.0,
            is_classifier_proba=True,
            model_family="tree_single",
            extra_note=f"depth={dt.get_depth()},nodes={dt.tree_.node_count}",
        )

    if model_name == "gbdt_tiny_core":
        gbdt = GradientBoostingClassifier(
            n_estimators=int(hp["n_estimators"]),
            max_depth=int(hp["max_depth"]),
            learning_rate=float(hp["learning_rate"]),
            subsample=float(hp["subsample"]),
            random_state=seed,
        )
        if bool(hp.get("use_balanced_sample_weight", True)):
            sw = compute_sample_weight("balanced", y_tr)
            gbdt.fit(X_tr, y_tr, sample_weight=sw)
        else:
            gbdt.fit(X_tr, y_tr)
        raw_prior = float(gbdt._raw_predict_init(X_tr[:1])[0, 0])
        trees = [est.tree_ for est in gbdt.estimators_.ravel()]
        return FoldedTreeEnsembleModel(
            name=model_name,
            reference_model=gbdt,
            trees=trees,
            base_score=raw_prior,
            scale=float(hp["learning_rate"]),
            is_classifier_proba=False,
            model_family="tree_boost",
        )

    if model_name in ("mlp_16_dot", "mlp_16_skip_dot", "mlp_32_16_dot"):
        hidden = [int(v) for v in hp["hidden_dims"]]
        scaler = StandardScaler().fit(X_tr)
        dropout = float(hp["dropout"])
        use_skip = bool(hp.get("use_skip", False))

        if model_name == "mlp_32_16_dot":
            def _make_net() -> nn.Module:
                return _TorchMlp2Hidden(d_in=d_in, h1=hidden[0], h2=hidden[1], dropout=dropout)
        else:
            def _make_net() -> nn.Module:
                return _TorchMlp1Hidden(d_in=d_in, d_hidden=hidden[0], dropout=dropout, use_skip=use_skip)

        fitted_net = _train_torch_binary_mlp(
            net_factory=_make_net,
            scaler=scaler,
            X_train=X_tr,
            y_train=y_tr,
            epochs=int(hp["epochs"]),
            lr=float(hp["learning_rate"]),
            weight_decay=float(hp["weight_decay"]),
            seed=seed,
            train_threads=int(hp.get("train_threads", 1)),
        )
        if model_name == "mlp_32_16_dot":
            return FoldedMlp2HiddenModel(name=model_name, net=fitted_net, scaler=scaler)
        if use_skip:
            return FoldedMlp1HiddenSkipModel(name=model_name, net=fitted_net, scaler=scaler)
        return FoldedMlp1HiddenModel(name=model_name, net=fitted_net, scaler=scaler)

    raise ValueError(f"Unknown model name '{model_name}' in Lab 14")


def verify_model_parity(
    model: FoldedMcuModel,
    X_test: np.ndarray,
    native_lib: NativeClassifierLibrary | None,
    thresholds: ThresholdsConfig,
) -> ParityReport:
    """Verify numerical equivalence between reference estimator, folded f64/f32, and C99."""
    scores_ref = model.predict_scores_ref(X_test)
    scores_f64 = model.predict_scores_f64(X_test)
    scores_f32 = model.predict_scores_f32(X_test)

    max_diff_f64: float | None = float(np.max(np.abs(scores_ref - scores_f64)))
    max_diff_f32: float | None = float(np.max(np.abs(scores_ref - scores_f32)))

    pred_f64 = (scores_f64 > 0.0).astype(np.int64)
    pred_f32 = (scores_f32 > 0.0).astype(np.int64)
    agree_f32 = float(np.mean(pred_f64 == pred_f32))

    # NOTE: PyTorch weights are inherently float32, so ref vs f64 compares float32 PyTorch against float64 folded.
    is_torch_ref = isinstance(model, (FoldedMlp1HiddenModel, FoldedMlp1HiddenSkipModel, FoldedMlp2HiddenModel))
    tol_f64 = thresholds.parity_abs_tol_f32 if is_torch_ref else thresholds.parity_abs_tol_f64
    if model.name not in ("dt", "rf") and max_diff_f64 > tol_f64:
        raise AssertionError(
            f"Parity violation (ref vs folded f64) for {model.name}: {max_diff_f64:.3e} > {tol_f64:.3e}"
        )

    max_diff_c99: float | None = None
    agree_c99: float | None = None
    if native_lib is not None and model.supports_c99():
        X_f32 = np.ascontiguousarray(X_test, dtype=np.float32)
        scores_c99 = model.predict_scores_c99(native_lib, X_f32)
        max_diff_c99 = float(np.max(np.abs(scores_f32 - scores_c99)))
        agree_c99 = float(np.mean(pred_f32 == (scores_c99 > 0.0).astype(np.int64)))
        scale_bound = max(
            1.0,
            float(np.max(np.abs(scores_f32))) if scores_f32.size else 1.0,
            model.numerical_scale_bound(),
        )
        eff_c_tol = thresholds.c_parity_abs_tol_f32 + thresholds.c_parity_rel_tol_f32 * scale_bound
        if max_diff_c99 > eff_c_tol:
            raise AssertionError(
                f"C99 vs f32 parity violation for {model.name}: "
                f"{max_diff_c99:.3e} > {eff_c_tol:.3e}"
            )

    return ParityReport(
        max_abs_diff_f64=max_diff_f64,
        max_abs_diff_f32=max_diff_f32,
        pred_agreement_f32=agree_f32,
        max_abs_diff_c99=max_diff_c99,
        pred_agreement_c99=agree_c99,
    )
