"""Classical ML classifier latency, parameter count, memory, and sklearn dispatch profiling."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import pickle
from typing import Any, Sequence

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.models.descriptor_classifier import LinearDotCoreClassifier

from .timing import (
    LatencyStats,
    measure_indexed_latencies,
    summarize_latencies,
    thread_limits,
)

logger = logging.getLogger("research")


@dataclass(frozen=True, slots=True)
class ClassifierParamInfo:
    """Exact model parameter accounting and serialized memory footprint."""

    params_model: int
    scaler_params: int
    pickle_bytes: int
    detail_note: str


@dataclass(frozen=True, slots=True)
class ClassifierOpsEstimate:
    """Analytical operation estimate for classical ML (not equivalent to neural MACs)."""

    analytical_macs: int
    analytical_flops: int
    estimate_basis: str


@dataclass(frozen=True, slots=True)
class ClassifierProfileRecord:
    """Detailed CPU latency, overhead breakdown, params, and throughput for a classical classifier."""

    classifier_name: str
    threads: int
    batch_size: int
    n_features: int
    n_train: int
    params_info: ClassifierParamInfo
    ops_estimate: ClassifierOpsEstimate
    predict_stats: LatencyStats
    predict_proba_stats: LatencyStats
    scaler_stats: LatencyStats | None
    estimator_only_stats: LatencyStats | None
    direct_core_stats: LatencyStats | None
    sklearn_dispatch_overhead_ms: float | None
    raw_predict_samples_ms: tuple[float, ...]


def _unwrap_estimator(model: ClassifierMixin | Pipeline) -> tuple[StandardScaler | None, Any]:
    """Extract optional StandardScaler and core estimator from a fitted sklearn model or Pipeline."""
    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("scaler")
        core = model.steps[-1][1]
        return (scaler if isinstance(scaler, StandardScaler) else None), core
    return None, model


def configure_classifier_threads(model: ClassifierMixin | Pipeline, threads: int) -> None:
    """Set classifier n_jobs cleanly via sklearn `set_params` without altering TOML configs."""
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    if isinstance(model, Pipeline):
        last_name, last_est = model.steps[-1]
        if hasattr(last_est, "n_jobs"):
            model.set_params(**{f"{last_name}__n_jobs": threads})
    elif hasattr(model, "n_jobs"):
        model.set_params(n_jobs=threads)


def count_classical_model_params(model: ClassifierMixin | Pipeline) -> ClassifierParamInfo:
    """Count real stored parameters of a fitted sklearn estimator (distinct from descriptor features)."""
    scaler, core = _unwrap_estimator(model)
    scaler_params = 0
    if scaler is not None:
        if hasattr(scaler, "mean_") and scaler.mean_ is not None:
            scaler_params += int(scaler.mean_.size)
        if hasattr(scaler, "scale_") and scaler.scale_ is not None:
            scaler_params += int(scaler.scale_.size)

    pickle_bytes = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))

    if isinstance(core, LogisticRegression):
        coef_sz = int(core.coef_.size)
        inter_sz = int(core.intercept_.size)
        total = coef_sz + inter_sz
        note = f"LR coef({coef_sz})+intercept({inter_sz}); scaler_params={scaler_params}"
    elif isinstance(core, LinearDotCoreClassifier):
        coef_sz = int(core.w_eff_.size)
        total = coef_sz + 1
        note = f"LinearDotCore w_eff({coef_sz})+b_eff(1); scaler folded at fit (scaler_params=0)"
    elif isinstance(core, DecisionTreeClassifier):
        nodes = int(core.tree_.node_count)
        leaves = int(core.tree_.n_leaves)
        total = nodes
        note = f"DT tree_.node_count={nodes} (leaves={leaves}, max_depth={core.get_depth()})"
    elif isinstance(core, RandomForestClassifier):
        nodes = sum(int(est.tree_.node_count) for est in core.estimators_)
        n_trees = len(core.estimators_)
        total = nodes
        note = f"RF sum(tree_.node_count)={nodes} across {n_trees} trees"
    elif isinstance(core, KNeighborsClassifier):
        n_tr, n_feat = core._fit_X.shape  # type: ignore[attr-defined]
        total = int(n_tr * n_feat)
        note = f"kNN training memory N_train({n_tr}) x F({n_feat}) + labels({n_tr}); scaler_params={scaler_params}"
    else:
        total = 0
        note = f"Unsupported estimator type {type(core).__name__}"

    return ClassifierParamInfo(
        params_model=total,
        scaler_params=scaler_params,
        pickle_bytes=pickle_bytes,
        detail_note=note,
    )


def _mean_expected_tree_depth(tree_obj: Any) -> float:
    """Compute sample-weighted expected root-to-leaf path length (comparisons) of an sklearn Tree."""
    children_left = tree_obj.children_left
    children_right = tree_obj.children_right
    samples = tree_obj.weighted_n_node_samples
    root_weight = float(samples[0]) if len(samples) > 0 and samples[0] > 0 else 1.0

    expected = 0.0
    stack: list[tuple[int, int]] = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        left = int(children_left[node_id])
        right = int(children_right[node_id])
        if left == right:  # leaf node
            expected += float(depth) * (float(samples[node_id]) / root_weight)
        else:
            stack.append((left, depth + 1))
            stack.append((right, depth + 1))
    return expected


def estimate_classical_inference_ops(
    model: ClassifierMixin | Pipeline,
    n_features: int,
) -> ClassifierOpsEstimate:
    """Estimate analytical inference cost per sample (strictly tagged as analytical estimate)."""
    scaler, core = _unwrap_estimator(model)
    scaler_ops = (2 * n_features) if scaler is not None else 0

    if isinstance(core, LogisticRegression):
        macs = int(core.coef_.size)
        flops = 2 * macs + scaler_ops
        basis = (
            f"Analytical estimate: dot product ({macs} MACs)"
            + (f" + StandardScaler ({scaler_ops} ops)" if scaler_ops > 0 else "")
        )
    elif isinstance(core, LinearDotCoreClassifier):
        macs = int(core.w_eff_.size)
        flops = 2 * macs
        basis = f"Analytical estimate: folded dot product ({macs} MACs, scaler folded at fit)"
    elif isinstance(core, DecisionTreeClassifier):
        avg_cmp = max(1.0, _mean_expected_tree_depth(core.tree_))
        macs = 0
        flops = int(round(avg_cmp))
        basis = f"Analytical estimate: ~{avg_cmp:.2f} expected threshold comparisons along tree path (0 MACs)"
    elif isinstance(core, RandomForestClassifier):
        total_cmp = sum(max(1.0, _mean_expected_tree_depth(t.tree_)) for t in core.estimators_)
        macs = 0
        flops = int(round(total_cmp))
        basis = (
            f"Analytical estimate: sum of expected tree path comparisons "
            f"(~{total_cmp:.1f} total across {len(core.estimators_)} trees, 0 MACs)"
        )
    elif isinstance(core, KNeighborsClassifier):
        n_tr, n_feat = core._fit_X.shape  # type: ignore[attr-defined]
        # Euclidean squared distance: diff + square + sum ~= 3*F ops per stored vector
        macs = int(n_tr * n_feat)
        flops = int(3 * n_tr * n_feat) + scaler_ops
        basis = (
            f"Analytical estimate: brute-force/index distances to N_train={n_tr} x {n_feat} features"
            + (f" + StandardScaler ({scaler_ops} ops)" if scaler_ops > 0 else "")
        )
    else:
        macs = 0
        flops = 0
        basis = "Analytical estimate unavailable"

    return ClassifierOpsEstimate(
        analytical_macs=macs,
        analytical_flops=flops,
        estimate_basis=basis,
    )


def _prepare_feature_batches(
    X_test: np.ndarray,
    batch_size: int,
    max_samples: int,
) -> list[np.ndarray]:
    """Slice contiguous [batch_size, F] float64 arrays from real test features outside timing."""
    n_avail = int(X_test.shape[0])
    if n_avail == 0:
        raise ValueError("X_test must contain at least 1 sample")
    n_use = min(n_avail, max(1, max_samples))
    x_sub = np.ascontiguousarray(X_test[:n_use], dtype=np.float64)

    batches: list[np.ndarray] = []
    if batch_size == 1:
        for i in range(n_use):
            batches.append(np.ascontiguousarray(x_sub[i : i + 1]))
        return batches

    n_batches = max(1, (n_use + batch_size - 1) // batch_size)
    for b in range(n_batches):
        idx = (np.arange(b * batch_size, (b + 1) * batch_size) % n_use).astype(np.int64)
        batches.append(np.ascontiguousarray(x_sub[idx]))
    return batches


def profile_classifier(
    model: ClassifierMixin | Pipeline,
    classifier_name: str,
    X_test: np.ndarray,
    n_train: int,
    threads: int = 1,
    batch_size: int = 1,
    warmup_iters: int = 20,
    timed_iters: int = 200,
    max_real_samples: int = 200,
    disable_gc: bool = True,
    bootstrap_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> ClassifierProfileRecord:
    """Profile a fitted classical ML classifier on real features under strict CPU thread limits."""
    configure_classifier_threads(model, threads=threads)
    n_features = int(X_test.shape[1])
    params_info = count_classical_model_params(model)
    ops_estimate = estimate_classical_inference_ops(model, n_features=n_features)
    batches = _prepare_feature_batches(X_test, batch_size=batch_size, max_samples=max_real_samples)
    n_batches = len(batches)

    scaler, core = _unwrap_estimator(model)

    with thread_limits(threads):
        pred_ms = measure_indexed_latencies(
            fn=lambda idx: model.predict(batches[idx % n_batches]),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            disable_gc=disable_gc,
        )
        proba_ms = measure_indexed_latencies(
            fn=lambda idx: model.predict_proba(batches[idx % n_batches]),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            disable_gc=disable_gc,
        )

        scaler_stats: LatencyStats | None = None
        estimator_only_stats: LatencyStats | None = None
        if scaler is not None:
            scaled_batches = [np.ascontiguousarray(scaler.transform(b)) for b in batches]
            sc_ms = measure_indexed_latencies(
                fn=lambda idx: scaler.transform(batches[idx % n_batches]),
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                disable_gc=disable_gc,
            )
            est_ms = measure_indexed_latencies(
                fn=lambda idx: core.predict(scaled_batches[idx % n_batches]),
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                disable_gc=disable_gc,
            )
            scaler_stats = summarize_latencies(
                sc_ms,
                per_sample_divisor=batch_size,
                bootstrap_resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=seed,
            )
            estimator_only_stats = summarize_latencies(
                est_ms,
                per_sample_divisor=batch_size,
                bootstrap_resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=seed,
            )

        # Direct low-level kernel diagnostic (isolated from baseline) to expose sklearn validation overhead
        direct_stats: LatencyStats | None = None
        if isinstance(core, DecisionTreeClassifier):
            batches_f32 = [np.ascontiguousarray(b, dtype=np.float32) for b in batches]
            dir_ms = measure_indexed_latencies(
                fn=lambda idx: core.tree_.predict(batches_f32[idx % n_batches]),
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                disable_gc=disable_gc,
            )
            direct_stats = summarize_latencies(
                dir_ms,
                per_sample_divisor=batch_size,
                bootstrap_resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=seed,
            )
        elif isinstance(core, LogisticRegression):
            src_batches = [np.ascontiguousarray(scaler.transform(b)) for b in batches] if scaler is not None else batches
            w_t = np.ascontiguousarray(core.coef_.T)
            b_vec = np.ascontiguousarray(core.intercept_)
            dir_ms = measure_indexed_latencies(
                fn=lambda idx: src_batches[idx % n_batches] @ w_t + b_vec,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                disable_gc=disable_gc,
            )
            direct_stats = summarize_latencies(
                dir_ms,
                per_sample_divisor=batch_size,
                bootstrap_resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=seed,
            )
        elif isinstance(core, LinearDotCoreClassifier):
            w_vec = np.ascontiguousarray(core.w_eff_, dtype=np.float64)
            b_val = float(core.b_eff_)
            c0, c1 = core.classes_[0], core.classes_[1]
            dir_ms = measure_indexed_latencies(
                fn=lambda idx: np.where((batches[idx % n_batches] @ w_vec + b_val) > 0.0, c1, c0),
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                disable_gc=disable_gc,
            )
            direct_stats = summarize_latencies(
                dir_ms,
                per_sample_divisor=batch_size,
                bootstrap_resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=seed,
            )

    predict_stats = summarize_latencies(
        pred_ms,
        per_sample_divisor=batch_size,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    predict_proba_stats = summarize_latencies(
        proba_ms,
        per_sample_divisor=batch_size,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )

    dispatch_overhead_ms: float | None = None
    if direct_stats is not None:
        base_est_med = (
            estimator_only_stats.median_ms if estimator_only_stats is not None else predict_stats.median_ms
        )
        dispatch_overhead_ms = base_est_med - direct_stats.median_ms

    return ClassifierProfileRecord(
        classifier_name=classifier_name,
        threads=threads,
        batch_size=batch_size,
        n_features=n_features,
        n_train=n_train,
        params_info=params_info,
        ops_estimate=ops_estimate,
        predict_stats=predict_stats,
        predict_proba_stats=predict_proba_stats,
        scaler_stats=scaler_stats,
        estimator_only_stats=estimator_only_stats,
        direct_core_stats=direct_stats,
        sklearn_dispatch_overhead_ms=dispatch_overhead_ms,
        raw_predict_samples_ms=tuple(float(v / batch_size) for v in pred_ms),
    )


def profile_classifier_thread_scaling(
    model: ClassifierMixin | Pipeline,
    classifier_name: str,
    X_test: np.ndarray,
    n_train: int,
    threads_list: Sequence[int],
    batch_size: int = 1,
    warmup_iters: int = 20,
    timed_iters: int = 200,
    max_real_samples: int = 200,
    disable_gc: bool = True,
    bootstrap_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[ClassifierProfileRecord, ...]:
    """Measure classical classifier latency across thread counts (restoring original n_jobs afterwards)."""
    _, core = _unwrap_estimator(model)
    prev_n_jobs = getattr(core, "n_jobs", None)
    records: list[ClassifierProfileRecord] = []
    try:
        for th in threads_list:
            rec = profile_classifier(
                model=model,
                classifier_name=classifier_name,
                X_test=X_test,
                n_train=n_train,
                threads=int(th),
                batch_size=batch_size,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                max_real_samples=max_real_samples,
                disable_gc=disable_gc,
                bootstrap_resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=seed,
            )
            records.append(rec)
    finally:
        if prev_n_jobs is not None:
            configure_classifier_threads(model, int(prev_n_jobs) if int(prev_n_jobs) != 0 else 1)
    return tuple(records)
