"""High-resolution nanosecond timing, thread pinning, GC control, memory, and statistical tests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gc
import os
import time
import tracemalloc
from typing import Callable, Iterator, Sequence, TypeVar

import numpy as np
import psutil
from scipy import stats
from threadpoolctl import threadpool_limits
import torch

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Distribution summary with bootstrap confidence interval for latency measurements (ms)."""

    n_samples: int
    median_ms: float
    q25_ms: float
    q75_ms: float
    iqr_ms: float
    p95_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    ci95_low_ms: float
    ci95_high_ms: float
    throughput_samples_s: float


@dataclass(frozen=True, slots=True)
class MultiRunStats:
    """Inter-run repeatability summary across multiple independent repetitions."""

    repeats: int
    run_medians_ms: tuple[float, ...]
    overall_median_ms: float
    inter_run_std_ms: float
    inter_run_cv_pct: float


@dataclass(frozen=True, slots=True)
class StatisticalComparisonResult:
    """Two-sided Mann-Whitney U comparison between two latency sample arrays."""

    group_a: str
    group_b: str
    n_a: int
    n_b: int
    median_a_ms: float
    median_b_ms: float
    median_diff_ms: float
    ratio_a_over_b: float
    u_statistic: float
    p_value: float
    significant: bool
    alpha: float


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Process RSS delta and Python/NumPy heap allocation peak measured around an execution."""

    rss_before_mb: float
    rss_after_mb: float
    rss_delta_mb: float
    rss_peak_mb: float
    tracemalloc_current_kb: float
    tracemalloc_peak_kb: float


@contextmanager
def disabled_gc(enabled: bool = True) -> Iterator[None]:
    """Temporarily disable Python cyclic garbage collector and guarantee prior state restoration."""
    was_enabled = gc.isenabled()
    if enabled and was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if enabled and was_enabled:
            gc.enable()


@contextmanager
def thread_limits(threads: int) -> Iterator[None]:
    """Apply uniform CPU thread caps across PyTorch intra-op and BLAS/OpenMP thread pools."""
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    prev_torch_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with threadpool_limits(limits=threads):
            yield
    finally:
        torch.set_num_threads(prev_torch_threads)


def measure_rss_mb() -> float:
    """Return current process Resident Set Size (RSS) in MiB using psutil."""
    proc = psutil.Process(os.getpid())
    return float(proc.memory_info().rss) / (1024.0 * 1024.0)


def measure_indexed_latencies(
    fn: Callable[[int], object],
    warmup_iters: int,
    timed_iters: int,
    disable_gc: bool = True,
    sync_cuda_device: torch.device | None = None,
) -> list[float]:
    """Measure per-iteration elapsed wall times (ms) using time.perf_counter_ns().

    Args:
        fn: Callable receiving the current iteration index.
        warmup_iters: Un-timed warmup iterations.
        timed_iters: Timed iterations returned in the latency sample list.
        disable_gc: When True, pauses Python GC during timing.
        sync_cuda_device: Optional CUDA device to synchronize around each measurement.

    Returns:
        List of positive elapsed durations in milliseconds of length `timed_iters`.
    """
    if warmup_iters < 0 or timed_iters < 1:
        raise ValueError("warmup_iters must be >= 0 and timed_iters must be >= 1")

    samples_ms: list[float] = []
    with disabled_gc(enabled=disable_gc):
        for idx in range(warmup_iters):
            _ = fn(idx)
        if sync_cuda_device is not None and sync_cuda_device.type == "cuda":
            torch.cuda.synchronize(sync_cuda_device)

        for idx in range(timed_iters):
            if sync_cuda_device is not None and sync_cuda_device.type == "cuda":
                torch.cuda.synchronize(sync_cuda_device)
            t0_ns = time.perf_counter_ns()
            _ = fn(idx)
            if sync_cuda_device is not None and sync_cuda_device.type == "cuda":
                torch.cuda.synchronize(sync_cuda_device)
            samples_ms.append((time.perf_counter_ns() - t0_ns) / 1e6)

    return samples_ms


def measure_callable_latencies(
    fn: Callable[[], object],
    warmup_iters: int,
    timed_iters: int,
    disable_gc: bool = True,
    sync_cuda_device: torch.device | None = None,
) -> list[float]:
    """Measure zero-argument callable latency distribution in milliseconds."""
    return measure_indexed_latencies(
        fn=lambda _idx: fn(),
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        disable_gc=disable_gc,
        sync_cuda_device=sync_cuda_device,
    )


def summarize_latencies(
    samples_ms: Sequence[float],
    per_sample_divisor: int = 1,
    bootstrap_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> LatencyStats:
    """Compute quantiles, median, IQR, p95, bootstrap median CI, and throughput.

    Args:
        samples_ms: Raw elapsed latency measurements (ms). When `per_sample_divisor > 1`
            (e.g. batch_size=32), divides each measurement first to obtain amortized ms/sample.
        per_sample_divisor: Batch divisor (>= 1) converting per-batch ms to per-sample ms.
        bootstrap_resamples: Number of bootstrap replicates for confidence interval of the median.
        confidence_level: Confidence level in (0.5, 1.0), default 0.95.
        seed: Random seed for reproducible bootstrap resampling.

    Returns:
        Populated LatencyStats record.
    """
    arr = np.asarray(samples_ms, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("samples_ms must be non-empty")
    if per_sample_divisor < 1:
        raise ValueError(f"per_sample_divisor must be >= 1, got {per_sample_divisor}")

    arr = arr / float(per_sample_divisor)
    n = int(arr.size)

    q25, median_val, q75, p95 = np.percentile(arr, [25.0, 50.0, 75.0, 95.0])
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    iqr_val = float(q75 - q25)

    # Chunked vectorized bootstrap for median CI without large temporary allocations
    if n == 1:
        ci_low = float(median_val)
        ci_high = float(median_val)
    else:
        rng = np.random.default_rng(seed)
        boot_medians = np.empty(bootstrap_resamples, dtype=np.float64)
        block = 250
        for start in range(0, bootstrap_resamples, block):
            size_b = min(block, bootstrap_resamples - start)
            idx = rng.integers(0, n, size=(size_b, n))
            boot_medians[start : start + size_b] = np.median(arr[idx], axis=1)
        alpha_half = (1.0 - confidence_level) * 50.0
        ci_low = float(np.percentile(boot_medians, alpha_half))
        ci_high = float(np.percentile(boot_medians, 100.0 - alpha_half))

    throughput = (1000.0 / float(median_val)) if median_val > 0.0 else 0.0

    return LatencyStats(
        n_samples=n,
        median_ms=float(median_val),
        q25_ms=float(q25),
        q75_ms=float(q75),
        iqr_ms=iqr_val,
        p95_ms=float(p95),
        mean_ms=mean_val,
        std_ms=std_val,
        min_ms=min_val,
        max_ms=max_val,
        ci95_low_ms=ci_low,
        ci95_high_ms=ci_high,
        throughput_samples_s=throughput,
    )


def summarize_multi_run(run_medians_ms: Sequence[float]) -> MultiRunStats:
    """Compute inter-run median stability and coefficient of variation (%)."""
    if not run_medians_ms:
        raise ValueError("run_medians_ms cannot be empty")
    arr = np.asarray(run_medians_ms, dtype=np.float64)
    med = float(np.median(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    cv_pct = (std / med * 100.0) if med > 0.0 else 0.0
    return MultiRunStats(
        repeats=int(arr.size),
        run_medians_ms=tuple(float(v) for v in arr.tolist()),
        overall_median_ms=med,
        inter_run_std_ms=std,
        inter_run_cv_pct=cv_pct,
    )


def compare_latency_distributions(
    samples_a_ms: Sequence[float],
    samples_b_ms: Sequence[float],
    group_a: str,
    group_b: str,
    alpha: float = 0.05,
) -> StatisticalComparisonResult:
    """Run two-sided Mann-Whitney U non-parametric test between two latency distributions."""
    a = np.asarray(samples_a_ms, dtype=np.float64)
    b = np.asarray(samples_b_ms, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        raise ValueError("Both sample arrays must be non-empty for Mann-Whitney U test")

    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    med_a = float(np.median(a))
    med_b = float(np.median(b))
    diff = med_a - med_b
    ratio = (med_a / med_b) if med_b > 0.0 else 0.0

    return StatisticalComparisonResult(
        group_a=group_a,
        group_b=group_b,
        n_a=int(a.size),
        n_b=int(b.size),
        median_a_ms=med_a,
        median_b_ms=med_b,
        median_diff_ms=diff,
        ratio_a_over_b=ratio,
        u_statistic=float(res.statistic),
        p_value=float(res.pvalue),
        significant=bool(float(res.pvalue) < alpha),
        alpha=float(alpha),
    )


def measure_memory_alloc(fn: Callable[[], T]) -> tuple[T, MemorySnapshot]:
    """Execute `fn` while tracking psutil process RSS and Python tracemalloc peak allocations."""
    rss_before = measure_rss_mb()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()

    try:
        result = fn()
        cur_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    rss_after = measure_rss_mb()
    snapshot = MemorySnapshot(
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        rss_delta_mb=rss_after - rss_before,
        rss_peak_mb=max(rss_before, rss_after),
        tracemalloc_current_kb=float(cur_bytes) / 1024.0,
        tracemalloc_peak_kb=float(peak_bytes) / 1024.0,
    )
    return result, snapshot
