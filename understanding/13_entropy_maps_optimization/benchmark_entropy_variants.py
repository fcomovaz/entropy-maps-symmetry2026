"""Reproducible CPU-fair correctness gate and performance benchmark suite for V0-V3."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch

LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, LAB_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from build_native import read_build_info
from config_loader import (
    EntropyOptimizationConfig,
    ProtocolOptimizationConfig,
    generate_synthetic_images,
    load_benchmark_images,
    load_optimization_config,
    resize_u8_batch,
    resolve_project_path,
)
from configs.logging import configure_logging, logger
from native_binding import V3CSinglePassVariant
from src.features.entropy_maps import (
    build_entropy_plan,
    build_shannon_term_lut,
    compute_entropy_features,
)
from src.models.cnn.architectures import create_model, model_stats, seed_everything
from src.models.cnn.benchmark import benchmark_model_latency, estimate_flops
from src.profiling.classifier_profile import (
    count_classical_model_params,
    estimate_classical_inference_ops,
    profile_classifier,
)
from src.profiling.environment import capture_environment, query_git_commit, save_environment_json
from src.profiling.timing import (
    LatencyStats,
    compare_latency_distributions,
    disabled_gc,
    measure_indexed_latencies,
    measure_memory_alloc,
    summarize_latencies,
    summarize_multi_run,
    thread_limits,
)
from variants import (
    EntropyVariant,
    V0BaselineVariant,
    V1NumpyFinestPyramidVariant,
    V2NumpyFusedEntropyVariant,
)


def _instantiate_variants(
    cfg: EntropyOptimizationConfig,
) -> tuple[dict[str, EntropyVariant], list[str]]:
    """Create configured variants; skip V3 cleanly with unverified marker if DLL is absent."""
    instances: dict[str, EntropyVariant] = {}
    unverified: list[str] = []

    for name in cfg.variants:
        if name == "V0_baseline":
            instances[name] = V0BaselineVariant()
        elif name == "V1_numpy_finest_pyramid":
            instances[name] = V1NumpyFinestPyramidVariant()
        elif name == "V2_numpy_fused_entropy":
            instances[name] = V2NumpyFusedEntropyVariant()
        elif name == "V3_c_single_pass":
            dll_path = cfg.native.dll_path
            if dll_path.is_file():
                instances[name] = V3CSinglePassVariant(dll_path)
            else:
                msg = (
                    f"V3_c_single_pass disabled: DLL not found at {dll_path}. "
                    "Build with build_native.py to verify native C speedup."
                )
                logger.warning(msg)
                unverified.append(msg)
        else:
            raise ValueError(f"Unknown variant name '{name}' in config")

    if cfg.reference_variant not in instances:
        raise RuntimeError(
            f"Reference variant '{cfg.reference_variant}' could not be instantiated."
        )
    return instances, unverified


def _bootstrap_speedup_ci(
    ref_samples_ms: np.ndarray,
    cand_samples_ms: np.ndarray,
    n_resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float]:
    """Compute point speedup median(ref)/median(cand) and non-parametric bootstrap CI."""
    ref_med = float(np.median(ref_samples_ms))
    cand_med = float(np.median(cand_samples_ms))
    point = (ref_med / cand_med) if cand_med > 0.0 else 0.0
    if ref_samples_ms.size <= 1 or cand_samples_ms.size <= 1 or cand_med <= 0.0:
        return point, point, point

    rng = np.random.default_rng(seed)
    n_ref = int(ref_samples_ms.size)
    n_cand = int(cand_samples_ms.size)
    ratios = np.empty(n_resamples, dtype=np.float64)
    block = 250
    for start in range(0, n_resamples, block):
        cur = min(block, n_resamples - start)
        idx_ref = rng.integers(0, n_ref, size=(cur, n_ref))
        idx_cand = rng.integers(0, n_cand, size=(cur, n_cand))
        med_r = np.median(ref_samples_ms[idx_ref], axis=1)
        med_c = np.median(cand_samples_ms[idx_cand], axis=1)
        ratios[start : start + cur] = np.divide(
            med_r, med_c, out=np.zeros_like(med_r), where=med_c > 0.0
        )

    alpha_half = (1.0 - confidence_level) * 50.0
    ci_low = float(np.percentile(ratios, alpha_half))
    ci_high = float(np.percentile(ratios, 100.0 - alpha_half))
    return point, ci_low, ci_high


def _prepare_batch_inputs(images_u8: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Slice deterministic contiguous [batch_size, H, W] arrays outside timed loops."""
    n_avail = int(images_u8.shape[0])
    if n_avail == 0:
        raise ValueError("Cannot construct batches from empty image dataset")
    if batch_size == 1:
        return [np.ascontiguousarray(images_u8[i : i + 1]) for i in range(n_avail)]

    n_batches = max(1, (n_avail + batch_size - 1) // batch_size)
    batches: list[np.ndarray] = []
    for b_idx in range(n_batches):
        idx = (np.arange(b_idx * batch_size, (b_idx + 1) * batch_size) % n_avail).astype(np.int64)
        batches.append(np.ascontiguousarray(images_u8[idx]))
    return batches


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Serialize list of dictionaries cleanly to CSV creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _measure_prepare_ms(
    variant: EntropyVariant,
    image_size: int,
    grid_levels: tuple[int, ...],
    bins: int,
    disable_gc_flag: bool,
) -> float:
    """Measure cold one-time initialization/LUT precomputation cost of a variant in milliseconds."""
    build_shannon_term_lut.cache_clear()
    with disabled_gc(enabled=disable_gc_flag):
        t0 = time.perf_counter_ns()
        variant.prepare(image_size, grid_levels, bins)
        elapsed_ns = time.perf_counter_ns() - t0
    return elapsed_ns / 1e6


def _run_correctness_gate(
    variants: dict[str, EntropyVariant],
    cfg: EntropyOptimizationConfig,
    bench_images: np.ndarray,
    bins_sweep: tuple[int, ...],
    size_sweep: tuple[int, ...],
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Verify every active variant matches V0 and src oracle within tolerance_abs before timing."""
    logger.info("Running correctness gate (tolerance_abs=%.1e)...", cfg.protocol.tolerance_abs)
    ref_name = cfg.reference_variant
    tol = cfg.protocol.tolerance_abs
    grid_str = "x".join(str(g) for g in cfg.grid_levels)

    # 1. Fixed edge cases + representative real/benchmark subset
    base_sz = cfg.data.image_size
    n_edge = min(4, int(bench_images.shape[0]))
    edge_imgs = np.stack(
        [
            np.zeros((base_sz, base_sz), dtype=np.uint8),
            np.full((base_sz, base_sz), 255, dtype=np.uint8),
            generate_synthetic_images(1, base_sz, cfg.seed + 101)[0],
            bench_images[0],
        ],
        axis=0,
    )
    cases: list[tuple[str, int, int, np.ndarray]] = [
        ("benchmark_batch", cfg.bins, base_sz, bench_images),
        ("edge_const_and_single", cfg.bins, base_sz, edge_imgs),
    ]
    for b_val in bins_sweep:
        if b_val != cfg.bins:
            cases.append((f"sweep_bins_{b_val}", b_val, base_sz, bench_images[:n_edge]))
    for sz_val in size_sweep:
        if sz_val != base_sz:
            resized = resize_u8_batch(bench_images[:n_edge], sz_val)
            cases.append((f"sweep_size_{sz_val}", cfg.bins, sz_val, resized))

    rows: list[dict[str, Any]] = []
    any_failed = False

    for case_name, bins_val, sz_val, case_imgs in cases:
        plan = build_entropy_plan(sz_val, cfg.grid_levels, bins_val)
        src_out = compute_entropy_features(
            case_imgs, grid_levels=cfg.grid_levels, bins=bins_val, plan=plan
        )
        ref_var = variants[ref_name]
        ref_var.prepare(sz_val, cfg.grid_levels, bins_val)
        v0_out = ref_var.run(case_imgs)

        for v_name, var in variants.items():
            var.prepare(sz_val, cfg.grid_levels, bins_val)
            actual = var.run(case_imgs)
            err_v0 = float(np.max(np.abs(actual - v0_out))) if actual.size > 0 else 0.0
            err_src = float(np.max(np.abs(actual - src_out))) if actual.size > 0 else 0.0
            passed = (err_v0 <= tol) and (err_src <= tol)
            if not passed:
                any_failed = True
                logger.error(
                    "CORRECTNESS FAILURE: variant=%s case=%s max|err_v0|=%.3e max|err_src|=%.3e (tol=%.1e)",
                    v_name,
                    case_name,
                    err_v0,
                    err_src,
                    tol,
                )
            rows.append(
                {
                    "variant": v_name,
                    "case_name": case_name,
                    "bins": bins_val,
                    "image_size": sz_val,
                    "grid_levels": grid_str,
                    "max_abs_err_vs_v0": f"{err_v0:.16e}",
                    "max_abs_err_vs_src": f"{err_src:.16e}",
                    "tolerance_abs": f"{tol:.3e}",
                    "passed": passed,
                }
            )

    _write_csv(
        output_dir / "correctness.csv",
        [
            "variant",
            "case_name",
            "bins",
            "image_size",
            "grid_levels",
            "max_abs_err_vs_v0",
            "max_abs_err_vs_src",
            "tolerance_abs",
            "passed",
        ],
        rows,
    )

    if any_failed:
        raise ValueError(
            "One or more entropy variants failed the numerical correctness gate. "
            f"Inspect {output_dir / 'correctness.csv'} before benchmarking."
        )
    logger.info("Correctness gate PASSED across all %d checks.", len(rows))
    return rows


def _latency_row(
    variant_name: str,
    is_diagnostic: bool,
    batch_size: int,
    image_size: int,
    bins: int,
    threads: int,
    repeats: int,
    prepare_ms: float,
    stats: LatencyStats,
    inter_run_cv_pct: float,
    speedup: float,
    speedup_ci_low: float,
    speedup_ci_high: float,
    mw_u: float,
    mw_p: float,
    significant: bool,
    tracemalloc_kb: float,
    rss_delta_mb: float,
) -> dict[str, Any]:
    """Format one latency measurement row for CSV export."""
    return {
        "variant": variant_name,
        "is_diagnostic": is_diagnostic,
        "batch_size": batch_size,
        "image_size": image_size,
        "bins": bins,
        "threads": threads,
        "n_samples": stats.n_samples,
        "repeats": repeats,
        "prepare_ms": round(prepare_ms, 6),
        "median_ms": round(stats.median_ms, 6),
        "q25_ms": round(stats.q25_ms, 6),
        "q75_ms": round(stats.q75_ms, 6),
        "iqr_ms": round(stats.iqr_ms, 6),
        "p95_ms": round(stats.p95_ms, 6),
        "mean_ms": round(stats.mean_ms, 6),
        "std_ms": round(stats.std_ms, 6),
        "ci95_low_ms": round(stats.ci95_low_ms, 6),
        "ci95_high_ms": round(stats.ci95_high_ms, 6),
        "throughput_samples_s": round(stats.throughput_samples_s, 3),
        "inter_run_cv_pct": round(inter_run_cv_pct, 4),
        "speedup_vs_v0": round(speedup, 4),
        "speedup_ci95_low": round(speedup_ci_low, 4),
        "speedup_ci95_high": round(speedup_ci_high, 4),
        "mann_whitney_u": round(mw_u, 2),
        "mann_whitney_p": f"{mw_p:.6e}",
        "significant": significant,
        "tracemalloc_peak_kb": round(tracemalloc_kb, 3),
        "rss_delta_mb": round(rss_delta_mb, 4),
    }


def _benchmark_latencies(
    variants: dict[str, EntropyVariant],
    cfg: EntropyOptimizationConfig,
    proto: ProtocolOptimizationConfig,
    images_u8: np.ndarray,
    batch_sizes: tuple[int, ...],
    warmup_iters: int,
    timed_iters: int,
    batch_timed_iters: int,
    repeats: int,
    bootstrap_resamples: int,
    output_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, LatencyStats],
    dict[str, np.ndarray],
]:
    """Benchmark batch sizes 1..256 cleanly, returning CSV rows + batch=1 stats & sample vectors."""
    sz = cfg.data.image_size
    bins = cfg.bins
    threads = proto.threads
    ref_name = cfg.reference_variant

    prepare_times_ms: dict[str, float] = {}
    for v_name, var in variants.items():
        prepare_times_ms[v_name] = _measure_prepare_ms(
            var, sz, cfg.grid_levels, bins, proto.disable_gc
        )

    rows: list[dict[str, Any]] = []
    batch1_stats: dict[str, LatencyStats] = {}
    batch1_samples: dict[str, np.ndarray] = {}
    sample_csv_rows: list[dict[str, Any]] = []

    with thread_limits(threads):
        for batch_size in batch_sizes:
            batches = _prepare_batch_inputs(images_u8, batch_size)
            n_batches = len(batches)
            eff_timed = timed_iters if batch_size == 1 else batch_timed_iters
            eff_repeats = repeats if batch_size == 1 else 1

            # Interleave repeats across variants to eliminate slow thermal/clock drift bias
            per_var_repeat_samples: dict[str, list[list[float]]] = {v: [] for v in variants}
            for _rep in range(eff_repeats):
                for v_name, var in variants.items():
                    samples_ms = measure_indexed_latencies(
                        fn=lambda idx, _var=var: _var.run(batches[idx % n_batches]),
                        warmup_iters=warmup_iters,
                        timed_iters=eff_timed,
                        disable_gc=proto.disable_gc,
                    )
                    per_var_repeat_samples[v_name].append(samples_ms)

            # Reference batch per-sample latencies from repeat 0
            ref_per_sample = np.asarray(
                per_var_repeat_samples[ref_name][0], dtype=np.float64
            ) / float(batch_size)

            # Memory snapshot outside timing loop + summary per variant
            for v_name, var in variants.items():
                rep_lists = per_var_repeat_samples[v_name]
                primary_raw = rep_lists[0]
                primary_per_sample = np.asarray(primary_raw, dtype=np.float64) / float(batch_size)

                stats = summarize_latencies(
                    primary_raw,
                    per_sample_divisor=batch_size,
                    bootstrap_resamples=bootstrap_resamples,
                    confidence_level=proto.confidence_level,
                    seed=cfg.seed,
                )

                if eff_repeats > 1:
                    rep_medians = [
                        float(np.median(np.asarray(r, dtype=np.float64) / float(batch_size)))
                        for r in rep_lists
                    ]
                    multi = summarize_multi_run(rep_medians)
                    cv_pct = multi.inter_run_cv_pct
                else:
                    cv_pct = 0.0

                if v_name == ref_name:
                    speedup, sp_low, sp_high = 1.0, 1.0, 1.0
                    mw_u, mw_p, sig = 0.0, 1.0, False
                else:
                    speedup, sp_low, sp_high = _bootstrap_speedup_ci(
                        ref_per_sample,
                        primary_per_sample,
                        n_resamples=bootstrap_resamples,
                        confidence_level=proto.confidence_level,
                        seed=cfg.seed,
                    )
                    cmp_res = compare_latency_distributions(
                        ref_per_sample,
                        primary_per_sample,
                        group_a=ref_name,
                        group_b=v_name,
                        alpha=proto.mann_whitney_alpha,
                    )
                    mw_u = cmp_res.u_statistic
                    mw_p = cmp_res.p_value
                    sig = cmp_res.significant

                _, mem_snap = measure_memory_alloc(lambda _v=var: _v.run(batches[0]))

                rows.append(
                    _latency_row(
                        variant_name=v_name,
                        is_diagnostic=False,
                        batch_size=batch_size,
                        image_size=sz,
                        bins=bins,
                        threads=threads,
                        repeats=eff_repeats,
                        prepare_ms=prepare_times_ms[v_name],
                        stats=stats,
                        inter_run_cv_pct=cv_pct,
                        speedup=speedup,
                        speedup_ci_low=sp_low,
                        speedup_ci_high=sp_high,
                        mw_u=mw_u,
                        mw_p=mw_p,
                        significant=sig,
                        tracemalloc_kb=mem_snap.tracemalloc_peak_kb,
                        rss_delta_mb=mem_snap.rss_delta_mb,
                    )
                )

                if batch_size == 1:
                    batch1_stats[v_name] = stats
                    batch1_samples[v_name] = primary_per_sample
                    for s_idx, s_val in enumerate(primary_per_sample.tolist()):
                        sample_csv_rows.append(
                            {
                                "variant": v_name,
                                "sample_idx": s_idx,
                                "latency_ms": round(s_val, 8),
                            }
                        )

            # V3 diagnostic metrics at batch=1: ctypes empty FFI overhead + zero-alloc run_into
            if batch_size == 1 and "V3_c_single_pass" in variants:
                v3 = variants["V3_c_single_pass"]
                assert isinstance(v3, V3CSinglePassVariant)
                noop_ms = measure_indexed_latencies(
                    fn=lambda _idx: v3.call_noop(),
                    warmup_iters=warmup_iters,
                    timed_iters=eff_timed,
                    disable_gc=proto.disable_gc,
                )
                noop_stats = summarize_latencies(
                    noop_ms,
                    per_sample_divisor=1,
                    bootstrap_resamples=bootstrap_resamples,
                    confidence_level=proto.confidence_level,
                    seed=cfg.seed,
                )
                rows.append(
                    _latency_row(
                        variant_name="V3_ctypes_noop_overhead",
                        is_diagnostic=True,
                        batch_size=1,
                        image_size=sz,
                        bins=bins,
                        threads=threads,
                        repeats=1,
                        prepare_ms=0.0,
                        stats=noop_stats,
                        inter_run_cv_pct=0.0,
                        speedup=0.0,
                        speedup_ci_low=0.0,
                        speedup_ci_high=0.0,
                        mw_u=0.0,
                        mw_p=1.0,
                        significant=False,
                        tracemalloc_kb=0.0,
                        rss_delta_mb=0.0,
                    )
                )

                total_features = sum(g * g for g in cfg.grid_levels)
                prealloc_out = np.empty((1, total_features), dtype=np.float64)
                into_ms = measure_indexed_latencies(
                    fn=lambda idx: v3.run_into(batches[idx % n_batches], prealloc_out),
                    warmup_iters=warmup_iters,
                    timed_iters=eff_timed,
                    disable_gc=proto.disable_gc,
                )
                into_stats = summarize_latencies(
                    into_ms,
                    per_sample_divisor=1,
                    bootstrap_resamples=bootstrap_resamples,
                    confidence_level=proto.confidence_level,
                    seed=cfg.seed,
                )
                sp_in, sp_in_lo, sp_in_hi = _bootstrap_speedup_ci(
                    ref_per_sample,
                    np.asarray(into_ms, dtype=np.float64),
                    n_resamples=bootstrap_resamples,
                    confidence_level=proto.confidence_level,
                    seed=cfg.seed,
                )
                cmp_in = compare_latency_distributions(
                    ref_per_sample,
                    into_ms,
                    group_a=ref_name,
                    group_b="V3_c_single_pass_into",
                    alpha=proto.mann_whitney_alpha,
                )
                _, into_mem = measure_memory_alloc(lambda: v3.run_into(batches[0], prealloc_out))
                rows.append(
                    _latency_row(
                        variant_name="V3_c_single_pass_into",
                        is_diagnostic=True,
                        batch_size=1,
                        image_size=sz,
                        bins=bins,
                        threads=threads,
                        repeats=1,
                        prepare_ms=prepare_times_ms["V3_c_single_pass"],
                        stats=into_stats,
                        inter_run_cv_pct=0.0,
                        speedup=sp_in,
                        speedup_ci_low=sp_in_lo,
                        speedup_ci_high=sp_in_hi,
                        mw_u=cmp_in.u_statistic,
                        mw_p=cmp_in.p_value,
                        significant=cmp_in.significant,
                        tracemalloc_kb=into_mem.tracemalloc_peak_kb,
                        rss_delta_mb=into_mem.rss_delta_mb,
                    )
                )

    latency_cols = [
        "variant",
        "is_diagnostic",
        "batch_size",
        "image_size",
        "bins",
        "threads",
        "n_samples",
        "repeats",
        "prepare_ms",
        "median_ms",
        "q25_ms",
        "q75_ms",
        "iqr_ms",
        "p95_ms",
        "mean_ms",
        "std_ms",
        "ci95_low_ms",
        "ci95_high_ms",
        "throughput_samples_s",
        "inter_run_cv_pct",
        "speedup_vs_v0",
        "speedup_ci95_low",
        "speedup_ci95_high",
        "mann_whitney_u",
        "mann_whitney_p",
        "significant",
        "tracemalloc_peak_kb",
        "rss_delta_mb",
    ]
    _write_csv(output_dir / "benchmark_latency.csv", latency_cols, rows)
    _write_csv(
        output_dir / "benchmark_batch1_samples.csv",
        ["variant", "sample_idx", "latency_ms"],
        sample_csv_rows,
    )
    return rows, batch1_stats, batch1_samples


def _benchmark_stages(
    variants: dict[str, EntropyVariant],
    cfg: EntropyOptimizationConfig,
    proto: ProtocolOptimizationConfig,
    images_u8: np.ndarray,
    batch1_stats: dict[str, LatencyStats],
    warmup_iters: int,
    timed_iters: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Measure stage-level latency breakdown (quantize, histogram, pyramid sum, entropy) at batch=1."""
    batches = _prepare_batch_inputs(images_u8, 1)
    n_b = len(batches)
    stage_rows: list[dict[str, Any]] = []

    with thread_limits(proto.threads):
        for v_name, var in variants.items():
            var.prepare(cfg.data.image_size, cfg.grid_levels, cfg.bins)
            quant_list: list[float] = []
            hist_list: list[float] = []
            sum_list: list[float] = []
            ent_list: list[float] = []

            with disabled_gc(enabled=proto.disable_gc):
                for idx in range(warmup_iters):
                    _ = var.measure_stages_once(batches[idx % n_b])  # type: ignore[attr-defined]
                for idx in range(timed_iters):
                    _, st = var.measure_stages_once(batches[idx % n_b])  # type: ignore[attr-defined]
                    quant_list.append(st["quantize_ms"])
                    hist_list.append(st["histogram_ms"])
                    sum_list.append(st["pyramid_sum_ms"])
                    ent_list.append(st["entropy_ms"])

            q_med = float(np.median(quant_list))
            h_med = float(np.median(hist_list))
            s_med = float(np.median(sum_list))
            e_med = float(np.median(ent_list))
            sum_stages = q_med + h_med + s_med + e_med
            e2e_med = batch1_stats[v_name].median_ms
            stage_rows.append(
                {
                    "variant": v_name,
                    "quantize_ms": round(q_med, 6),
                    "histogram_ms": round(h_med, 6),
                    "pyramid_sum_ms": round(s_med, 6),
                    "entropy_ms": round(e_med, 6),
                    "sum_stages_ms": round(sum_stages, 6),
                    "e2e_median_ms": round(e2e_med, 6),
                    "dispatch_gap_ms": round(sum_stages - e2e_med, 6),
                    "quant_fused_into_hist": v_name == "V3_c_single_pass",
                }
            )

    _write_csv(
        output_dir / "benchmark_stages.csv",
        [
            "variant",
            "quantize_ms",
            "histogram_ms",
            "pyramid_sum_ms",
            "entropy_ms",
            "sum_stages_ms",
            "e2e_median_ms",
            "dispatch_gap_ms",
            "quant_fused_into_hist",
        ],
        stage_rows,
    )
    return stage_rows


def _benchmark_sweeps(
    variants: dict[str, EntropyVariant],
    cfg: EntropyOptimizationConfig,
    proto: ProtocolOptimizationConfig,
    images_u8: np.ndarray,
    bins_sweep: tuple[int, ...],
    size_sweep: tuple[int, ...],
    warmup_iters: int,
    timed_iters: int,
    bootstrap_resamples: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Measure batch=1 latency and speedup across bins and image_size axes (grid_levels fixed)."""
    sweep_rows: list[dict[str, Any]] = []
    ref_name = cfg.reference_variant

    sweep_configs: list[tuple[str, int, int]] = []
    for b in bins_sweep:
        sweep_configs.append(("bins", b, cfg.data.image_size))
    for sz in size_sweep:
        sweep_configs.append(("image_size", cfg.bins, sz))

    batches_by_size: dict[int, list[np.ndarray]] = {}
    for _, _, sz in sweep_configs:
        if sz not in batches_by_size:
            resized = resize_u8_batch(images_u8, sz)
            batches_by_size[sz] = _prepare_batch_inputs(resized, 1)

    with thread_limits(proto.threads):
        for axis, b_val, sz_val in sweep_configs:
            batches = batches_by_size[sz_val]
            n_b = len(batches)
            per_var_samples: dict[str, np.ndarray] = {}
            prep_ms_map: dict[str, float] = {}
            stats_map: dict[str, LatencyStats] = {}

            for v_name, var in variants.items():
                prep_ms_map[v_name] = _measure_prepare_ms(
                    var, sz_val, cfg.grid_levels, b_val, proto.disable_gc
                )
                samples = measure_indexed_latencies(
                    fn=lambda idx, _v=var, _b=batches, _nb=n_b: _v.run(_b[idx % _nb]),
                    warmup_iters=warmup_iters,
                    timed_iters=timed_iters,
                    disable_gc=proto.disable_gc,
                )
                per_var_samples[v_name] = np.asarray(samples, dtype=np.float64)
                stats_map[v_name] = summarize_latencies(
                    samples,
                    per_sample_divisor=1,
                    bootstrap_resamples=bootstrap_resamples,
                    confidence_level=proto.confidence_level,
                    seed=cfg.seed,
                )

            ref_arr = per_var_samples[ref_name]
            for v_name in variants:
                st = stats_map[v_name]
                if v_name == ref_name:
                    sp, sp_lo, sp_hi = 1.0, 1.0, 1.0
                    mw_p, sig = 1.0, False
                else:
                    sp, sp_lo, sp_hi = _bootstrap_speedup_ci(
                        ref_arr,
                        per_var_samples[v_name],
                        n_resamples=bootstrap_resamples,
                        confidence_level=proto.confidence_level,
                        seed=cfg.seed,
                    )
                    cmp_res = compare_latency_distributions(
                        ref_arr,
                        per_var_samples[v_name],
                        group_a=ref_name,
                        group_b=v_name,
                        alpha=proto.mann_whitney_alpha,
                    )
                    mw_p, sig = cmp_res.p_value, cmp_res.significant

                sweep_rows.append(
                    {
                        "sweep_axis": axis,
                        "bins": b_val,
                        "image_size": sz_val,
                        "variant": v_name,
                        "prepare_ms": round(prep_ms_map[v_name], 6),
                        "median_ms": round(st.median_ms, 6),
                        "ci95_low_ms": round(st.ci95_low_ms, 6),
                        "ci95_high_ms": round(st.ci95_high_ms, 6),
                        "iqr_ms": round(st.iqr_ms, 6),
                        "p95_ms": round(st.p95_ms, 6),
                        "throughput_samples_s": round(st.throughput_samples_s, 3),
                        "speedup_vs_v0": round(sp, 4),
                        "speedup_ci95_low": round(sp_lo, 4),
                        "speedup_ci95_high": round(sp_hi, 4),
                        "mann_whitney_p": f"{mw_p:.6e}",
                        "significant": sig,
                    }
                )

    _write_csv(
        output_dir / "benchmark_sweeps.csv",
        [
            "sweep_axis",
            "bins",
            "image_size",
            "variant",
            "prepare_ms",
            "median_ms",
            "ci95_low_ms",
            "ci95_high_ms",
            "iqr_ms",
            "p95_ms",
            "throughput_samples_s",
            "speedup_vs_v0",
            "speedup_ci95_low",
            "speedup_ci95_high",
            "mann_whitney_p",
            "significant",
        ],
        sweep_rows,
    )
    return sweep_rows


def _fit_reference_lr(
    cfg: EntropyOptimizationConfig,
    x_desc: np.ndarray,
) -> tuple[Pipeline | LogisticRegression, int, int, int, int, str]:
    """Instantiate and fit reference LogisticRegression on real descriptor shape for inference profiling."""
    lr_cfg = cfg.lr_ref
    lr_kwargs: dict[str, Any] = {
        "C": lr_cfg.c_value,
        "penalty": lr_cfg.penalty,
        "solver": lr_cfg.solver,
        "class_weight": lr_cfg.class_weight,
        "max_iter": lr_cfg.max_iter,
        "random_state": cfg.seed,
    }
    lr = LogisticRegression(**lr_kwargs)
    model: Pipeline | LogisticRegression = (
        Pipeline([("scaler", StandardScaler()), ("clf", lr)])
        if lr_cfg.scale_features
        else lr
    )

    # Deterministic median-split labels populate valid weights solely for inference latency profiling.
    if x_desc.shape[0] < 2:
        x_fit = np.vstack([x_desc[0], np.clip(x_desc[0] * 0.9 + 0.05, 0.0, 1.0)])
        y_fit = np.array([0, 1], dtype=np.int64)
    else:
        mean_ent = x_desc.mean(axis=1)
        y_fit = (mean_ent > float(np.median(mean_ent))).astype(np.int64)
        if np.unique(y_fit).size < 2:
            y_fit = (np.arange(x_desc.shape[0]) % 2).astype(np.int64)
        x_fit = x_desc
    model.fit(x_fit, y_fit)

    n_features = int(x_desc.shape[1])
    param_info = count_classical_model_params(model)
    ops_est = estimate_classical_inference_ops(model, n_features=n_features)
    return (
        model,
        param_info.params_model,
        param_info.scaler_params,
        ops_est.analytical_flops,
        n_features,
        ops_est.estimate_basis,
    )


def _benchmark_vs_cnn_cpu(
    variants: dict[str, EntropyVariant],
    cfg: EntropyOptimizationConfig,
    proto: ProtocolOptimizationConfig,
    images_u8: np.ndarray,
    batch1_stats: dict[str, LatencyStats],
    warmup_iters: int,
    timed_iters: int,
    bootstrap_resamples: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Compare entropy variants (+ LR) vs lightweight PyTorch CNNs on CPU (threads=1, batch=1)."""
    sz_desc = cfg.data.image_size
    ref_var = variants[cfg.reference_variant]
    ref_var.prepare(sz_desc, cfg.grid_levels, cfg.bins)
    x_desc = ref_var.run(images_u8)

    lr_model, lr_params, lr_scaler_params, lr_flops, n_feats, lr_basis = _fit_reference_lr(
        cfg, x_desc
    )
    lr_profile = profile_classifier(
        model=lr_model,
        classifier_name="lr_entropy_x85",
        X_test=x_desc,
        n_train=int(x_desc.shape[0]),
        threads=proto.threads,
        batch_size=1,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        max_real_samples=int(images_u8.shape[0]),
        disable_gc=proto.disable_gc,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=proto.confidence_level,
        seed=cfg.seed,
    )
    lr_sklearn_ms = lr_profile.predict_stats.median_ms
    lr_dot_ms = (
        lr_profile.direct_core_stats.median_ms
        if lr_profile.direct_core_stats is not None
        else lr_sklearn_ms
    )

    # Analytical descriptor ops basis:
    # ~1 LUT quant read/px + 1 fine hist increment/px + 1 LUT term lookup per (cell * bin) + child additions
    pixels_per_img = sz_desc * sz_desc
    n_levels = len(cfg.grid_levels)
    sum_cells = sum(g * g for g in cfg.grid_levels)
    v0_analytical_ops = (n_levels * pixels_per_img) + (2 * sum_cells * cfg.bins) + pixels_per_img
    fast_analytical_ops = pixels_per_img + (2 * sum_cells * cfg.bins)

    # Benchmark PyTorch CPU CNN reference models using exact linked cnn.data normalization
    sz_cnn = cfg.cnn_ref.image_size
    cnn_u8 = resize_u8_batch(images_u8, sz_cnn)
    cnn_norm = (
        (cnn_u8.astype(np.float32) / 255.0) - float(cfg.cnn_ref.normalize_mean)
    ) / float(cfg.cnn_ref.normalize_std)
    cnn_tensor = torch.from_numpy(cnn_norm[:, None, :, :]).contiguous()

    cnn_rows: list[dict[str, Any]] = []
    ds_cnn_total_ms: float | None = None

    with thread_limits(proto.threads):
        for model_name in cfg.cnn_ref.reference_models:
            seed_everything(cfg.seed)
            model = create_model(model_name, num_classes=2).cpu().eval()
            params, _, _ = model_stats(model)
            cnn_flops = estimate_flops(model, sz_cnn)
            cnn_samples_ms, _ = benchmark_model_latency(
                model=model,
                device=torch.device("cpu"),
                batch_size=1,
                image_size=sz_cnn,
                warmup_iters=warmup_iters,
                timed_iters=timed_iters,
                threads=proto.threads,
                inputs=cnn_tensor,
                disable_gc=proto.disable_gc,
            )
            cnn_stats = summarize_latencies(
                cnn_samples_ms,
                per_sample_divisor=1,
                bootstrap_resamples=bootstrap_resamples,
                confidence_level=proto.confidence_level,
                seed=cfg.seed,
            )
            if model_name == "ds_cnn":
                ds_cnn_total_ms = cnn_stats.median_ms
            cnn_rows.append(
                {
                    "pipeline_name": f"CNN_{model_name}_cpu",
                    "category": "cnn_cpu",
                    "descriptor_variant": "none",
                    "model_name": model_name,
                    "image_size": sz_cnn,
                    "threads": proto.threads,
                    "t_desc_ms": 0.0,
                    "t_model_ms": round(cnn_stats.median_ms, 6),
                    "t_total_ms": round(cnn_stats.median_ms, 6),
                    "t_total_ci95_low_ms": round(cnn_stats.ci95_low_ms, 6),
                    "t_total_ci95_high_ms": round(cnn_stats.ci95_high_ms, 6),
                    "speedup_vs_ds_cnn": 0.0,  # backfilled below once ds_cnn known
                    "params_model": params,
                    "params_scaler": 0,
                    "params_total": params,
                    "analytical_flops": cnn_flops,
                    "ops_basis": "PyTorch FlopCounterMode / hook estimate on [1,1,H,W]",
                    "notes": "Untrained random weights; CPU threads=1 batch=1 latency",
                }
            )

    if ds_cnn_total_ms is None and cnn_rows:
        ds_cnn_total_ms = float(cnn_rows[0]["t_total_ms"])

    all_rows: list[dict[str, Any]] = []
    for row in cnn_rows:
        tot = float(row["t_total_ms"])
        row["speedup_vs_ds_cnn"] = round(
            (ds_cnn_total_ms / tot) if (ds_cnn_total_ms and tot > 0.0) else 0.0, 4
        )
        all_rows.append(row)

    for v_name in variants:
        desc_st = batch1_stats[v_name]
        desc_ops = v0_analytical_ops if v_name == "V0_baseline" else fast_analytical_ops
        desc_ops_note = (
            f"Analytical descriptor ops (~{desc_ops} integer/LUT ops) + {lr_basis}"
        )

        # Row A: Descriptor + direct LR dot-product (isolates descriptor + real linear core)
        tot_dot = desc_st.median_ms + lr_dot_ms
        all_rows.append(
            {
                "pipeline_name": f"{v_name} + LR (dot core)",
                "category": "descriptor_lr_dot",
                "descriptor_variant": v_name,
                "model_name": f"LR_dot_{n_feats}f",
                "image_size": sz_desc,
                "threads": proto.threads,
                "t_desc_ms": round(desc_st.median_ms, 6),
                "t_model_ms": round(lr_dot_ms, 6),
                "t_total_ms": round(tot_dot, 6),
                "t_total_ci95_low_ms": round(desc_st.ci95_low_ms + lr_dot_ms, 6),
                "t_total_ci95_high_ms": round(desc_st.ci95_high_ms + lr_dot_ms, 6),
                "speedup_vs_ds_cnn": round(
                    (ds_cnn_total_ms / tot_dot) if (ds_cnn_total_ms and tot_dot > 0.0) else 0.0, 4
                ),
                "params_model": lr_params,
                "params_scaler": lr_scaler_params,
                "params_total": lr_params + lr_scaler_params,
                "analytical_flops": desc_ops + lr_flops,
                "ops_basis": desc_ops_note,
                "notes": "LR weights fitted on median-split real descriptors solely for latency/param measurement",
            }
        )

        # Row B: Descriptor + full sklearn Pipeline.predict() (exposes Python/sklearn wrapper dispatch)
        tot_sk = desc_st.median_ms + lr_sklearn_ms
        all_rows.append(
            {
                "pipeline_name": f"{v_name} + LR (sklearn)",
                "category": "descriptor_lr_sklearn",
                "descriptor_variant": v_name,
                "model_name": f"LR_sklearn_{n_feats}f",
                "image_size": sz_desc,
                "threads": proto.threads,
                "t_desc_ms": round(desc_st.median_ms, 6),
                "t_model_ms": round(lr_sklearn_ms, 6),
                "t_total_ms": round(tot_sk, 6),
                "t_total_ci95_low_ms": round(desc_st.ci95_low_ms + lr_sklearn_ms, 6),
                "t_total_ci95_high_ms": round(desc_st.ci95_high_ms + lr_sklearn_ms, 6),
                "speedup_vs_ds_cnn": round(
                    (ds_cnn_total_ms / tot_sk) if (ds_cnn_total_ms and tot_sk > 0.0) else 0.0, 4
                ),
                "params_model": lr_params,
                "params_scaler": lr_scaler_params,
                "params_total": lr_params + lr_scaler_params,
                "analytical_flops": desc_ops + lr_flops,
                "ops_basis": desc_ops_note,
                "notes": "Includes full scikit-learn predict() validation overhead",
            }
        )

    _write_csv(
        output_dir / "benchmark_vs_cnn.csv",
        [
            "pipeline_name",
            "category",
            "descriptor_variant",
            "model_name",
            "image_size",
            "threads",
            "t_desc_ms",
            "t_model_ms",
            "t_total_ms",
            "t_total_ci95_low_ms",
            "t_total_ci95_high_ms",
            "speedup_vs_ds_cnn",
            "params_model",
            "params_scaler",
            "params_total",
            "analytical_flops",
            "ops_basis",
            "notes",
        ],
        all_rows,
    )
    return all_rows


def run_benchmark_suite(
    cfg: EntropyOptimizationConfig,
    source_override: str | None = None,
    output_dir_override: Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Execute complete Stage 5 correctness gate + Stage 6 CPU-fair benchmarks."""
    output_dir = (
        resolve_project_path(output_dir_override, PROJECT_ROOT)
        if output_dir_override is not None
        else cfg.plots.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)

    images_u8, actual_source = load_benchmark_images(cfg, source_override=source_override)
    variants, unverified_items = _instantiate_variants(cfg)

    proto = cfg.protocol
    if dry_run:
        batch_sizes = tuple(b for b in proto.batch_sizes if b <= 8) or (proto.batch_sizes[0],)
        warmup_iters = min(2, proto.warmup_iters)
        timed_iters = min(10, proto.timed_iters)
        batch_timed_iters = min(5, proto.batch_timed_iters)
        repeats = 1
        bootstrap_resamples = min(50, proto.bootstrap_resamples)
        bins_sweep = (cfg.bins,)
        size_sweep = (cfg.data.image_size,)
        logger.warning("Running in --dry-run mode with shortened iterations.")
    else:
        batch_sizes = proto.batch_sizes
        warmup_iters = proto.warmup_iters
        timed_iters = proto.timed_iters
        batch_timed_iters = proto.batch_timed_iters
        repeats = proto.repeats
        bootstrap_resamples = proto.bootstrap_resamples
        bins_sweep = cfg.sweeps.bins_list
        size_sweep = cfg.sweeps.image_sizes

    logger.info(
        "Benchmark setup: source=%s, shape=%s, grid_levels=%s, bins=%d, threads=%d, variants=%s",
        actual_source,
        images_u8.shape,
        cfg.grid_levels,
        cfg.bins,
        proto.threads,
        list(variants.keys()),
    )

    # Step 1: Correctness verification gate before any timing
    _run_correctness_gate(
        variants=variants,
        cfg=cfg,
        bench_images=images_u8,
        bins_sweep=bins_sweep,
        size_sweep=size_sweep,
        output_dir=output_dir,
    )

    # Step 2: Latency and throughput across batch sizes + batch=1 raw distribution
    latency_rows, batch1_stats, _ = _benchmark_latencies(
        variants=variants,
        cfg=cfg,
        proto=proto,
        images_u8=images_u8,
        batch_sizes=batch_sizes,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        batch_timed_iters=batch_timed_iters,
        repeats=repeats,
        bootstrap_resamples=bootstrap_resamples,
        output_dir=output_dir,
    )

    # Step 3: Stage-level breakdown at batch=1
    _benchmark_stages(
        variants=variants,
        cfg=cfg,
        proto=proto,
        images_u8=images_u8,
        batch1_stats=batch1_stats,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        output_dir=output_dir,
    )

    # Step 4: Bins & image_size sweeps
    _benchmark_sweeps(
        variants=variants,
        cfg=cfg,
        proto=proto,
        images_u8=images_u8,
        bins_sweep=bins_sweep,
        size_sweep=size_sweep,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        bootstrap_resamples=bootstrap_resamples,
        output_dir=output_dir,
    )

    # Step 5: Reference comparison against CPU lightweight CNNs + LR
    _benchmark_vs_cnn_cpu(
        variants=variants,
        cfg=cfg,
        proto=proto,
        images_u8=images_u8,
        batch1_stats=batch1_stats,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        bootstrap_resamples=bootstrap_resamples,
        output_dir=output_dir,
    )

    # Step 6: Hardware / SW environment & manifest provenance
    env_snapshot = capture_environment(PROJECT_ROOT, cfg.seed)
    save_environment_json(env_snapshot, output_dir / "environment.json")

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": query_git_commit(PROJECT_ROOT),
        "seed": cfg.seed,
        "dry_run": dry_run,
        "config_path": str(cfg.config_path),
        "entropy_config_path": str(cfg.entropy_config_path),
        "base_config_path": str(cfg.base_config_path),
        "cnn_config_path": str(cfg.cnn_config_path),
        "classification_config_path": str(cfg.classification_config_path),
        "data_source": actual_source,
        "n_images_loaded": int(images_u8.shape[0]),
        "image_size": cfg.data.image_size,
        "grid_levels": list(cfg.grid_levels),
        "bins": cfg.bins,
        "threads": proto.threads,
        "batch_sizes": list(batch_sizes),
        "warmup_iters": warmup_iters,
        "timed_iters": timed_iters,
        "batch_timed_iters": batch_timed_iters,
        "repeats": repeats,
        "disable_gc": proto.disable_gc,
        "tolerance_abs": proto.tolerance_abs,
        "active_variants": list(variants.keys()),
        "unverified_items": unverified_items,
        "native_dll_path": str(cfg.native.dll_path),
        "native_build_info": read_build_info(cfg.native.build_dir),
        "library_versions": {
            "python": env_snapshot.python_version,
            "numpy": env_snapshot.numpy_version,
            "scipy": env_snapshot.scipy_version,
            "sklearn": env_snapshot.sklearn_version,
            "pandas": env_snapshot.pandas_version,
            "torch": env_snapshot.torch_version,
            "psutil": env_snapshot.psutil_version,
            "threadpoolctl": env_snapshot.threadpoolctl_version,
        },
        "latency_rows_count": len(latency_rows),
    }
    manifest_path = output_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Benchmark artifacts saved under %s", output_dir)
    for r in latency_rows:
        if r["batch_size"] == 1:
            logger.info(
                "  [batch=1] %-26s | med=%8.4f ms (IQR=%7.4f) | sp=%6.2fx | p=%s",
                r["variant"],
                r["median_ms"],
                r["iqr_ms"],
                r["speedup_vs_v0"],
                r["mann_whitney_p"],
            )
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for entropy optimization benchmark suite."""
    parser = argparse.ArgumentParser(
        prog="benchmark_entropy_variants.py",
        description="Verify numerical parity and benchmark V0-V3 entropy pyramid variants.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/optimization/entropy_maps.toml"),
        help="Path to optimization TOML config.",
    )
    parser.add_argument(
        "--source",
        choices=["real", "synthetic"],
        default=None,
        help="Override [entropy_optimization.data].source ('real' or 'synthetic').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override output directory for CSVs and JSON metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run minimal smoke benchmark (reduced iterations) to verify pipeline mechanics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for Stage 6 benchmark runner."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg = load_optimization_config(args.config, project_root=PROJECT_ROOT)
        _ = run_benchmark_suite(
            cfg=cfg,
            source_override=args.source,
            output_dir_override=args.output_dir,
            dry_run=bool(args.dry_run),
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        logger.error("Benchmark suite failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
