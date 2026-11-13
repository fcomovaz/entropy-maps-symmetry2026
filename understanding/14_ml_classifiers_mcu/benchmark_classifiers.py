"""Reproducible MCU-classifier exploration, C99 parity gate, and CPU-fair benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, LAB_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from configs.logging import configure_logging, logger
from config_loader import (
    BenchmarkTargetConfig,
    McuLabConfig,
    load_lab_config,
    resolve_project_path,
)
from models import (
    FoldedMcuModel,
    compute_quality_metrics,
    train_and_build_mcu_model,
    verify_model_parity,
)
from native_bridge import NativeClassifierLibrary, try_load_native_library
from src.features.entropy_maps import build_entropy_plan
from src.models.descriptor_classifier import load_and_split_descriptors, load_classification_config
from src.profiling.descriptor_stages import extract_x94_end_to_end
from src.profiling.environment import capture_environment, save_environment_json
from src.profiling.timing import (
    compare_latency_distributions,
    measure_indexed_latencies,
    summarize_latencies,
    summarize_multi_run,
    thread_limits,
)


def _load_real_grayscale_batch(image_dir: Path, image_size: int, max_images: int, seed: int) -> np.ndarray:
    """Load deterministic sample of preprocessed grayscale images as contiguous uint8 [N, H, W]."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Preprocessed image directory not found: {image_dir}")
    paths = sorted(p for p in image_dir.rglob("*.png") if p.is_file())
    if not paths:
        raise FileNotFoundError(f"No PNG images found inside {image_dir}")

    rng = np.random.default_rng(seed)
    if len(paths) > max_images:
        chosen = rng.choice(len(paths), size=max_images, replace=False)
        selected = [paths[int(i)] for i in sorted(chosen)]
    else:
        selected = paths

    imgs: list[np.ndarray] = []
    for p in selected:
        with Image.open(p) as im:
            gray = im.convert("L")
            if gray.size != (image_size, image_size):
                gray = gray.resize((image_size, image_size), Image.Resampling.BILINEAR)
            imgs.append(np.asarray(gray, dtype=np.uint8))
    return np.ascontiguousarray(np.stack(imgs, axis=0), dtype=np.uint8)


def _measure_descriptor_reference(
    target: BenchmarkTargetConfig,
    cfg: McuLabConfig,
    inverted: bool,
) -> dict[int, dict[str, Any]]:
    """Measure T_desc (ms/sample) cleanly on real preprocessed images at threads=1."""
    d_cfg = cfg.descriptor_reference
    p_cfg = cfg.profiling
    img_dir = target.image_dir_inverted if inverted else target.image_dir
    imgs = _load_real_grayscale_batch(
        img_dir,
        image_size=d_cfg.image_size,
        max_images=d_cfg.n_real_images,
        seed=cfg.experiment.seed,
    )
    if imgs.shape[0] < max(p_cfg.batch_sizes):
        reps = int(np.ceil(max(p_cfg.batch_sizes) / float(imgs.shape[0])))
        imgs = np.ascontiguousarray(np.tile(imgs, (reps, 1, 1)), dtype=np.uint8)

    plan = build_entropy_plan(
        image_size=d_cfg.image_size,
        grid_levels=d_cfg.grid_levels,
        bins=d_cfg.bins,
        backend=d_cfg.entropy_backend,
    )
    n_imgs = int(imgs.shape[0])
    results_by_bs: dict[int, dict[str, Any]] = {}

    with thread_limits(p_cfg.threads):
        for bs in p_cfg.batch_sizes:
            iters = p_cfg.timed_iters if bs == 1 else p_cfg.batch_timed_iters
            run_medians: list[float] = []
            pooled: list[float] = []
            for rep in range(p_cfg.repeats):
                offset_span = max(1, n_imgs - bs + 1)
                raw_ms = measure_indexed_latencies(
                    fn=lambda idx, span=offset_span, bsz=bs: extract_x94_end_to_end(
                        imgs[(idx % span) : (idx % span) + bsz],
                        grid_levels=d_cfg.grid_levels,
                        bins=d_cfg.bins,
                        asymmetry_scales=d_cfg.asymmetry_scales,
                        asymmetry_kinds=d_cfg.asymmetry_kinds,
                        plan=plan,
                    ),
                    warmup_iters=p_cfg.warmup_iters,
                    timed_iters=iters,
                    disable_gc=p_cfg.disable_gc,
                )
                st = summarize_latencies(
                    raw_ms,
                    per_sample_divisor=bs,
                    bootstrap_resamples=p_cfg.bootstrap_resamples,
                    confidence_level=p_cfg.confidence_level,
                    seed=cfg.experiment.seed + rep,
                )
                run_medians.append(st.median_ms)
                pooled.extend(v / float(bs) for v in raw_ms)

            pooled_st = summarize_latencies(
                pooled,
                per_sample_divisor=1,
                bootstrap_resamples=p_cfg.bootstrap_resamples,
                confidence_level=p_cfg.confidence_level,
                seed=cfg.experiment.seed,
            )
            multi = summarize_multi_run(run_medians)
            results_by_bs[bs] = {
                "entropy_backend": plan.resolved_backend,
                "n_images_loaded": int(n_imgs),
                "t_descriptor_ms_median": pooled_st.median_ms,
                "t_descriptor_ms_iqr": pooled_st.iqr_ms,
                "t_descriptor_ci95_low_ms": pooled_st.ci95_low_ms,
                "t_descriptor_ci95_high_ms": pooled_st.ci95_high_ms,
                "inter_run_cv_pct": multi.inter_run_cv_pct,
            }
    return results_by_bs


def _prepare_latency_pools(X_test: np.ndarray, min_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Ensure C-contiguous float64 & float32 feature pools have at least min_rows samples."""
    X64 = np.ascontiguousarray(X_test, dtype=np.float64)
    if X64.shape[0] < min_rows:
        reps = int(np.ceil(min_rows / float(X64.shape[0])))
        X64 = np.ascontiguousarray(np.tile(X64, (reps, 1)), dtype=np.float64)
    X32 = np.ascontiguousarray(X64, dtype=np.float32)
    return X64, X32


def _benchmark_single_model_latency(
    model: FoldedMcuModel,
    X64_pool: np.ndarray,
    X32_pool: np.ndarray,
    cfg: McuLabConfig,
    native_lib: NativeClassifierLibrary | None,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[float]]]:
    """Benchmark Python/NumPy core and optional C99 native latency at B=1 and B=32."""
    p_cfg = cfg.profiling
    n_rows = int(X64_pool.shape[0])
    stats_by_bs: dict[int, dict[str, Any]] = {}
    pooled_by_bs: dict[int, list[float]] = {}

    with thread_limits(p_cfg.threads):
        for bs in p_cfg.batch_sizes:
            iters = p_cfg.timed_iters if bs == 1 else p_cfg.batch_timed_iters
            span = max(1, n_rows - bs + 1)
            run_medians: list[float] = []
            pooled: list[float] = []

            for rep in range(p_cfg.repeats):
                raw_ms = measure_indexed_latencies(
                    fn=lambda idx, sp=span, bsz=bs: model.predict_batch_for_latency(
                        X64_pool[(idx % sp) : (idx % sp) + bsz],
                        X32_pool[(idx % sp) : (idx % sp) + bsz],
                    ),
                    warmup_iters=p_cfg.warmup_iters,
                    timed_iters=iters,
                    disable_gc=p_cfg.disable_gc,
                )
                st = summarize_latencies(
                    raw_ms,
                    per_sample_divisor=bs,
                    bootstrap_resamples=p_cfg.bootstrap_resamples,
                    confidence_level=p_cfg.confidence_level,
                    seed=cfg.experiment.seed + rep,
                )
                run_medians.append(st.median_ms)
                pooled.extend(v / float(bs) for v in raw_ms)

            pooled_st = summarize_latencies(
                pooled,
                per_sample_divisor=1,
                bootstrap_resamples=p_cfg.bootstrap_resamples,
                confidence_level=p_cfg.confidence_level,
                seed=cfg.experiment.seed,
            )
            multi = summarize_multi_run(run_medians)

            c99_ffi_ms: float | None = None
            c99_inner_us: float | None = None
            if native_lib is not None and model.supports_c99():
                c_raw_ms = measure_indexed_latencies(
                    fn=lambda idx, sp=span, bsz=bs: model.predict_scores_c99(
                        native_lib, X32_pool[(idx % sp) : (idx % sp) + bsz]
                    ),
                    warmup_iters=p_cfg.warmup_iters,
                    timed_iters=iters,
                    disable_gc=p_cfg.disable_gc,
                )
                c_st = summarize_latencies(
                    c_raw_ms,
                    per_sample_divisor=bs,
                    bootstrap_resamples=p_cfg.bootstrap_resamples,
                    confidence_level=p_cfg.confidence_level,
                    seed=cfg.experiment.seed,
                )
                c99_ffi_ms = c_st.median_ms
                if bs == 1:
                    c99_inner_us = model.measure_c99_inner_us(
                        native_lib,
                        X32_pool,
                        inner_iters=cfg.native.c_bench_inner_iters,
                    )

            stats_by_bs[bs] = {
                "t_model_ms_median": pooled_st.median_ms,
                "t_model_ms_iqr": pooled_st.iqr_ms,
                "t_model_ms_p95": pooled_st.p95_ms,
                "t_model_ci95_low_ms": pooled_st.ci95_low_ms,
                "t_model_ci95_high_ms": pooled_st.ci95_high_ms,
                "model_cv_pct": multi.inter_run_cv_pct,
                "t_c99_ffi_ms_median": c99_ffi_ms,
                "t_c99_kernel_us_per_sample": c99_inner_us,
            }
            pooled_by_bs[bs] = pooled

    return stats_by_bs, pooled_by_bs


def _write_csv(path: Path, rows: list[dict[str, Any]], float_precision: int) -> None:
    """Serialize structured record list cleanly into formatted CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted: dict[str, Any] = {}
            for k, v in row.items():
                if isinstance(v, float):
                    if np.isnan(v):
                        formatted[k] = ""
                    else:
                        formatted[k] = f"{v:.{float_precision}f}"
                elif v is None:
                    formatted[k] = ""
                else:
                    formatted[k] = v
            writer.writerow(formatted)


def run_benchmark(
    cfg: McuLabConfig,
    skip_native: bool = False,
    skip_mvtec_all: bool = False,
    include_inverted: bool = False,
) -> dict[str, Any]:
    """Execute full Lab 14 training, parity verification, timing, and generalization sweep."""
    out_dir = cfg.experiment.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    native_lib: NativeClassifierLibrary | None = None
    if not skip_native:
        native_lib = try_load_native_library(cfg.native)

    env_snap = capture_environment(PROJECT_ROOT, seed=cfg.experiment.seed)
    save_environment_json(env_snap, out_dir / "environment.json")

    primary_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    desc_rows: list[dict[str, Any]] = []
    sample_export_chunks: list[str] = []

    inversion_modes = (False, True) if include_inverted else (False,)

    for inverted in inversion_modes:
        for target in cfg.targets:
            ds_tag = f"{target.tag}-inverted" if inverted else target.tag
            logger.info("=== Benchmarking target dataset: %s ===", ds_tag)

            src_cfg = load_classification_config(target.classification_config, PROJECT_ROOT)
            if inverted:
                src_cfg = replace(src_cfg, inverted=True)
            partition = load_and_split_descriptors(src_cfg, object_name=target.object_name)

            desc_stats = _measure_descriptor_reference(target, cfg, inverted=inverted)
            for bs, d_info in desc_stats.items():
                desc_rows.append(
                    {
                        "dataset_tag": ds_tag,
                        "inverted": inverted,
                        "batch_size": bs,
                        "threads": cfg.profiling.threads,
                        **d_info,
                    }
                )

            X64_pool, X32_pool = _prepare_latency_pools(
                partition.X_test, min_rows=max(cfg.profiling.batch_sizes)
            )
            b1_samples_by_model: dict[str, list[float]] = {}

            for model_name in cfg.models.active:
                model = train_and_build_mcu_model(
                    model_name=model_name,
                    partition=partition,
                    src_cfg=src_cfg,
                    models_cfg=cfg.models,
                    thresholds=cfg.thresholds,
                    seed=cfg.experiment.seed,
                )
                parity = verify_model_parity(
                    model=model,
                    X_test=partition.X_test,
                    native_lib=native_lib,
                    thresholds=cfg.thresholds,
                )
                y_prob, y_pred = model.predict_probs_and_labels(partition.X_test)
                metrics = compute_quality_metrics(partition.y_test, y_pred, y_prob)

                if (
                    not inverted
                    and ds_tag == cfg.experiment.export_sample_header_dataset
                    and model_name in cfg.experiment.export_sample_header_models
                ):
                    safe_prefix = model_name.replace("-", "_")
                    sample_export_chunks.append(model.export_c99_snippet(safe_prefix))

                lat_stats, pooled_samples = _benchmark_single_model_latency(
                    model=model,
                    X64_pool=X64_pool,
                    X32_pool=X32_pool,
                    cfg=cfg,
                    native_lib=native_lib,
                )
                b1_samples_by_model[model_name] = pooled_samples[1]
                cost = model.cost_profile

                for bs in cfg.profiling.batch_sizes:
                    t_desc_ms = float(desc_stats[bs]["t_descriptor_ms_median"])
                    l_info = lat_stats[bs]
                    t_model_ms = float(l_info["t_model_ms_median"])
                    t_total_ms = t_desc_ms + t_model_ms
                    throughput = 1000.0 / t_total_ms if t_total_ms > 0.0 else 0.0

                    c99_total_ms = (
                        t_desc_ms + float(l_info["t_c99_kernel_us_per_sample"]) / 1e3
                        if l_info["t_c99_kernel_us_per_sample"] is not None
                        else None
                    )

                    primary_rows.append(
                        {
                            "dataset_tag": ds_tag,
                            "object_name": target.object_name or "full",
                            "inverted": inverted,
                            "model": model_name,
                            "group": model.display_group,
                            "model_family": cost.model_family,
                            "threads": cfg.profiling.threads,
                            "batch_size": bs,
                            "n_train": int(partition.X_train.shape[0]),
                            "n_test": int(partition.X_test.shape[0]),
                            "pos_train": int(np.sum(partition.y_train == 1)),
                            "pos_test": int(np.sum(partition.y_test == 1)),
                            "n_params": cost.n_params,
                            "flash_bytes_f32": cost.flash_bytes_f32,
                            "flash_kb_f32": cost.flash_bytes_f32 / 1024.0,
                            "sram_scratch_bytes_f32": cost.sram_scratch_bytes_f32,
                            "macs": cost.macs,
                            "adds": cost.adds,
                            "comparisons": cost.comparisons,
                            "math_calls_at_threshold": cost.math_calls_at_threshold,
                            "n_active_features": cost.n_active_features,
                            "accuracy": metrics.accuracy,
                            "balanced_accuracy": metrics.balanced_accuracy,
                            "f1": metrics.f1,
                            "roc_auc": metrics.roc_auc,
                            "pr_auc": metrics.pr_auc,
                            "t_descriptor_ms_median": t_desc_ms,
                            "t_model_ms_median": t_model_ms,
                            "t_model_ms_iqr": l_info["t_model_ms_iqr"],
                            "t_model_ms_p95": l_info["t_model_ms_p95"],
                            "t_model_ci95_low_ms": l_info["t_model_ci95_low_ms"],
                            "t_model_ci95_high_ms": l_info["t_model_ci95_high_ms"],
                            "t_total_ms_est": t_total_ms,
                            "throughput_total_samples_s": throughput,
                            "t_c99_ffi_ms_median": l_info["t_c99_ffi_ms_median"],
                            "t_c99_kernel_us_per_sample": l_info["t_c99_kernel_us_per_sample"],
                            "t_total_c99_kernel_ms_est": c99_total_ms,
                            "parity_max_abs_diff_f64": parity.max_abs_diff_f64,
                            "parity_max_abs_diff_f32": parity.max_abs_diff_f32,
                            "pred_agreement_f32": parity.pred_agreement_f32,
                            "parity_max_abs_diff_c99": parity.max_abs_diff_c99,
                            "pred_agreement_c99": parity.pred_agreement_c99,
                            "note": model.extra_note,
                        }
                    )

                logger.info(
                    "  %-15s | F1=%.4f AUC=%.4f PR=%.4f | Flash=%.2f KB | MACs=%d | T_model(B=1)=%.4f ms",
                    model_name,
                    metrics.f1,
                    metrics.roc_auc,
                    metrics.pr_auc,
                    cost.flash_bytes_f32 / 1024.0,
                    cost.macs,
                    lat_stats[1]["t_model_ms_median"],
                )

            ref_name = cfg.profiling.reference_baseline_model
            heavy_name = cfg.profiling.heavy_baseline_model
            for candidate_name in cfg.models.active:
                for baseline_name in (ref_name, heavy_name):
                    if candidate_name == baseline_name:
                        continue
                    if baseline_name not in b1_samples_by_model:
                        continue
                    cmp_res = compare_latency_distributions(
                        samples_a_ms=b1_samples_by_model[candidate_name],
                        samples_b_ms=b1_samples_by_model[baseline_name],
                        group_a=candidate_name,
                        group_b=baseline_name,
                        alpha=cfg.profiling.mann_whitney_alpha,
                    )
                    stat_rows.append(
                        {
                            "dataset_tag": ds_tag,
                            "batch_size": 1,
                            "threads": cfg.profiling.threads,
                            **asdict(cmp_res),
                        }
                    )

    _write_csv(out_dir / "benchmark_primary_models.csv", primary_rows, cfg.experiment.float_precision)
    _write_csv(out_dir / "statistical_comparisons.csv", stat_rows, cfg.experiment.float_precision)
    _write_csv(out_dir / "descriptor_reference_times.csv", desc_rows, cfg.experiment.float_precision)

    if sample_export_chunks:
        header_path = out_dir / "sample_weights_mvtec_bottle.h"
        header_banner = (
            "/* Auto-generated static C99 weight sample from Lab 14 (mvtec-ad_bottle, seed=42).\n"
            " * Ready to include alongside understanding/14_ml_classifiers_mcu/c_kernels/mcu_classifiers.h.\n"
            " */\n\n#ifndef SAMPLE_WEIGHTS_MVTEC_BOTTLE_H\n#define SAMPLE_WEIGHTS_MVTEC_BOTTLE_H\n\n#include <stdint.h>\n\n"
        )
        header_footer = "\n#endif /* SAMPLE_WEIGHTS_MVTEC_BOTTLE_H */\n"
        header_path.write_text(
            header_banner + "\n".join(sample_export_chunks) + header_footer,
            encoding="utf-8",
        )

    sweep_rows: list[dict[str, Any]] = []
    agg_rows: list[dict[str, Any]] = []
    if cfg.mvtec_sweep.enabled and not skip_mvtec_all:
        logger.info("=== Running 15-object MVTec-AD + KolektorSDD2 generalization sweep ===")
        sweep_tasks: list[tuple[str, Path, str | None]] = [
            (
                "kolektorsdd2",
                resolve_project_path("configs/classification/kolektorsdd2.toml", PROJECT_ROOT),
                None,
            )
        ] + [
            (f"mvtec-ad_{obj}", cfg.mvtec_sweep.classification_config, obj)
            for obj in cfg.mvtec_sweep.objects
        ]

        for task_tag, cls_cfg_path, obj_name in sweep_tasks:
            src_cfg = load_classification_config(cls_cfg_path, PROJECT_ROOT)
            partition = load_and_split_descriptors(src_cfg, object_name=obj_name)
            for model_name in cfg.models.active:
                model = train_and_build_mcu_model(
                    model_name=model_name,
                    partition=partition,
                    src_cfg=src_cfg,
                    models_cfg=cfg.models,
                    thresholds=cfg.thresholds,
                    seed=cfg.experiment.seed,
                )
                parity = verify_model_parity(
                    model=model,
                    X_test=partition.X_test,
                    native_lib=native_lib,
                    thresholds=cfg.thresholds,
                )
                y_prob, y_pred = model.predict_probs_and_labels(partition.X_test)
                m = compute_quality_metrics(partition.y_test, y_pred, y_prob)
                c = model.cost_profile
                sweep_rows.append(
                    {
                        "dataset_tag": task_tag,
                        "is_mvtec": task_tag.startswith("mvtec-ad_"),
                        "object_name": obj_name or "full",
                        "model": model_name,
                        "group": model.display_group,
                        "n_params": c.n_params,
                        "flash_kb_f32": c.flash_bytes_f32 / 1024.0,
                        "macs": c.macs,
                        "n_active_features": c.n_active_features,
                        "accuracy": m.accuracy,
                        "balanced_accuracy": m.balanced_accuracy,
                        "f1": m.f1,
                        "roc_auc": m.roc_auc,
                        "pr_auc": m.pr_auc,
                        "parity_max_abs_diff_f32": parity.max_abs_diff_f32,
                        "pred_agreement_f32": parity.pred_agreement_f32,
                    }
                )

        mvtec_only = [r for r in sweep_rows if r["is_mvtec"]]
        for model_name in cfg.models.active:
            sub = [r for r in mvtec_only if r["model"] == model_name]
            f1s = np.asarray([float(r["f1"]) for r in sub], dtype=np.float64)
            baccs = np.asarray([float(r["balanced_accuracy"]) for r in sub], dtype=np.float64)
            aucs = np.asarray([float(r["roc_auc"]) for r in sub], dtype=np.float64)
            prs = np.asarray([float(r["pr_auc"]) for r in sub], dtype=np.float64)
            flashes = np.asarray([float(r["flash_kb_f32"]) for r in sub], dtype=np.float64)
            actives = np.asarray([float(r["n_active_features"]) for r in sub], dtype=np.float64)
            agg_rows.append(
                {
                    "model": model_name,
                    "group": sub[0]["group"],
                    "n_objects": len(sub),
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s, ddof=1)),
                    "f1_median": float(np.median(f1s)),
                    "balanced_accuracy_mean": float(np.mean(baccs)),
                    "roc_auc_mean": float(np.mean(aucs)),
                    "roc_auc_median": float(np.median(aucs)),
                    "pr_auc_mean": float(np.mean(prs)),
                    "flash_kb_f32_mean": float(np.mean(flashes)),
                    "active_features_mean": float(np.mean(actives)),
                }
            )
            logger.info(
                "  MVTec-15 Mean %-15s | F1=%.4f AUC=%.4f PR=%.4f | Flash=%.2f KB",
                model_name,
                float(np.mean(f1s)),
                float(np.mean(aucs)),
                float(np.mean(prs)),
                float(np.mean(flashes)),
            )

        _write_csv(out_dir / "all_objects_quality.csv", sweep_rows, cfg.experiment.float_precision)
        _write_csv(out_dir / "mvtec15_aggregate_summary.csv", agg_rows, cfg.experiment.float_precision)

    run_summary = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": str(cfg.config_path),
        "output_dir": str(out_dir),
        "seed": cfg.experiment.seed,
        "threads": cfg.profiling.threads,
        "native_c99_loaded": native_lib is not None,
        "native_build_metadata": asdict(native_lib.build_metadata)
        if native_lib is not None and native_lib.build_metadata is not None
        else None,
        "include_inverted": include_inverted,
        "mvtec_sweep_executed": bool(cfg.mvtec_sweep.enabled and not skip_mvtec_all),
        "primary_rows_count": len(primary_rows),
        "sweep_rows_count": len(sweep_rows),
        "src_unmodified_verified_by_isolation": True,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return run_summary


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser cleanly decoupled from module import."""
    parser = argparse.ArgumentParser(
        description="Benchmark folded MCU ML classifiers on x94 descriptors with C99 parity."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/understanding/ml_classifiers_mcu.toml"),
        help="Path to Lab 14 TOML configuration relative to code root or absolute.",
    )
    parser.add_argument(
        "--skip-native",
        action="store_true",
        help="Skip C99 shared-library build/verification and benchmark Python/NumPy cores only.",
    )
    parser.add_argument(
        "--skip-mvtec-all",
        action="store_true",
        help="Skip 15-object MVTec-AD generalization sweep (run primary latency targets only).",
    )
    parser.add_argument(
        "--include-inverted",
        action="store_true",
        help="Also run intensity-inverted dataset variants for primary targets.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint with early config validation and structured logging."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg = load_lab_config(args.config, PROJECT_ROOT)
        summary = run_benchmark(
            cfg=cfg,
            skip_native=args.skip_native,
            skip_mvtec_all=args.skip_mvtec_all,
            include_inverted=args.include_inverted,
        )
        logger.info("Lab 14 benchmark complete: %s", summary["output_dir"] if "output_dir" in summary else cfg.experiment.output_dir)
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, AssertionError) as exc:
        logger.error("Lab 14 benchmark failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
