"""CLI and core runner for 100% CPU fair benchmarking: x94 classical ML vs lightweight CNNs.

Enforces strict device parity (CPU primary, configurable thread pinning, batch=1 latency focus,
GC disabled during timing, real test samples, GPU checkpoint reuse with CPU parity check) while
keeping optional CUDA timings strictly isolated in a separate secondary reference CSV.

Manual execution commands (PowerShell from repository 'code/' directory):
    # 1. Dry-run validation:
    .\\.venv\\Scripts\\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml --dry-run
    .\\.venv\\Scripts\\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle --dry-run

    # 2. Full CPU fair benchmark on KolektorSDD2 (normal & inverted):
    .\\.venv\\Scripts\\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml
    .\\.venv\\Scripts\\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml --inverted

    # 3. Full CPU fair benchmark on MVTec-AD bottle (normal & inverted):
    .\\.venv\\Scripts\\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle
    .\\.venv\\Scripts\\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle --inverted
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import configure_logging, logger
from src.features.entropy_maps import build_entropy_plan
from src.models.cnn import (
    CnnConfig,
    CnnDatasetPartition,
    create_model,
    evaluate_cnn,
    load_cnn_config,
    load_cnn_partition,
    model_names,
    run_benchmark,
)
from src.models.cnn.architectures.common import model_stats, seed_everything
from src.models.cnn.benchmark import benchmark_model_latency, estimate_flops
from src.models.descriptor_classifier import (
    ClassificationConfig,
    ClassificationMetrics,
    DatasetPartition,
    load_and_split_descriptors,
    load_classification_config,
    train_and_evaluate,
)
from src.profiling import (
    ClassifierOpsEstimate,
    ClassifierParamInfo,
    EnvironmentSnapshot,
    ProfilingConfig,
    capture_environment,
    configure_classifier_threads,
    effective_variant_tag,
    estimate_descriptor_ops,
    extract_x94_end_to_end,
    load_profiling_config,
    measure_indexed_latencies,
    measure_memory_alloc,
    profile_classifier,
    resolve_target_objects,
    summarize_latencies,
    summarize_multi_run,
    thread_limits,
)
from src.profiling.report import (
    plot_cpu_comparison_pareto,
    upsert_summary_all_datasets,
    write_comparison_csv,
)


@dataclass(frozen=True, slots=True)
class CpuFairBenchmarkBundle:
    """In-memory and on-disk outputs from Phase A CPU-fair benchmarking for one target."""

    dataset_tag: str
    object_name: str | None
    inverted: bool
    environment: EnvironmentSnapshot
    cls_partition: DatasetPartition
    images_u8: np.ndarray
    x_real_cnn: torch.Tensor
    x_ml_real: np.ndarray
    fitted_classifiers: dict[str, tuple[Any, ClassificationMetrics, ClassifierParamInfo, ClassifierOpsEstimate]]
    loaded_cnns: dict[str, tuple[nn.Module, Path, ClassificationMetrics, float | None]]
    cpu_rows: tuple[dict[str, Any], ...]
    cuda_rows: tuple[dict[str, Any], ...]
    raw_batch1_primary_latencies_ms: dict[str, tuple[float, ...]]
    cpu_csv_path: Path | None
    cuda_csv_path: Path | None
    summary_csv_path: Path | None
    pareto_plot_paths: tuple[Path, ...]


def resolve_profiling_config_path(
    config_arg: Path | None,
    dataset_arg: str | None,
) -> Path:
    """Resolve profiling TOML config path from `--config` and/or `--dataset` flags."""
    if config_arg is not None:
        return config_arg
    if dataset_arg is None:
        return Path("configs/profiling/kolektorsdd2.toml")
    ds_norm = dataset_arg.lower().strip()
    if ds_norm == "kolektorsdd2":
        return Path("configs/profiling/kolektorsdd2.toml")
    if ds_norm in {"mvtecad", "mvtec-ad"}:
        return Path("configs/profiling/mvtec_ad.toml")
    raise ValueError(f"Unsupported --dataset '{dataset_arg}'. Expected 'kolektorsdd2' or 'mvtecad'.")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for 100% CPU fair cross-paradigm comparison."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.benchmark_cpu_fair",
        description="Execute 100% CPU fair benchmark comparing x94 classical ML vs lightweight CNNs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to profiling TOML config (defaults to configs/profiling/kolektorsdd2.toml or derived from --dataset).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kolektorsdd2", "mvtecad", "mvtec-ad"],
        default=None,
        help="Optional dataset shorthand ('kolektorsdd2' or 'mvtecad').",
    )
    parser.add_argument(
        "--object",
        dest="object_name",
        type=str,
        default=None,
        help="Single object name for MVTec-AD (e.g. 'bottle').",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        type=str,
        default=None,
        help="Explicit list of MVTec-AD object names (e.g. --objects bottle metal_nut tile).",
    )
    parser.add_argument(
        "--all-objects",
        action="store_true",
        default=False,
        help="Benchmark across all MVTec-AD objects configured in base.toml.",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        default=False,
        help="Run on inverted grayscale dataset variant.",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=None,
        help="CPU thread counts to benchmark (primary thread count uses first/primary value).",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Batch sizes to benchmark (default from TOML: [1, 32]).",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        type=str,
        default=None,
        help="Classical ML classifiers subset (e.g. dt knn lr rf, or 'all').",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=None,
        help="CNN architecture subset (or 'all').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Primary comparison device (strictly must be 'cpu'; passing 'cuda' fails fast).",
    )
    parser.add_argument(
        "--no-cuda-reference",
        action="store_true",
        default=False,
        help="Skip optional secondary CUDA reference table even if GPU is available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for CPU comparison CSVs and Pareto plots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Fail-fast validate TOMLs, CSVs, PNG datasets, splits, and existing CNN checkpoints without benchmarking.",
    )
    return parser


def resolve_selected_classifiers(raw_arg: Sequence[str] | None, cfg: ProfilingConfig) -> tuple[str, ...]:
    """Validate and normalize requested classical ML classifiers."""
    allowed = {"dt", "knn", "lr", "rf"}
    if not raw_arg:
        chosen = cfg.classifiers
    elif len(raw_arg) == 1 and raw_arg[0].lower() == "all":
        chosen = cfg.classifiers
    else:
        chosen = tuple(str(c).lower() for c in raw_arg)
    for c in chosen:
        if c not in allowed:
            raise ValueError(f"Unsupported classifier '{c}'. Expected one of {sorted(allowed)}.")
    return chosen


def resolve_selected_models(raw_arg: Sequence[str] | None, cfg: ProfilingConfig) -> tuple[str, ...]:
    """Validate and normalize requested CNN architecture names."""
    valid = set(model_names())
    if not raw_arg:
        chosen = cfg.models
    elif len(raw_arg) == 1 and raw_arg[0].lower() == "all":
        chosen = cfg.models
    else:
        chosen = tuple(str(m).lower() for m in raw_arg)
    for m in chosen:
        if m not in valid:
            raise ValueError(f"Unsupported CNN model '{m}'. Available: {model_names()}")
    return chosen


def validate_required_inputs_upfront(
    prof_cfg: ProfilingConfig,
    cls_cfg: ClassificationConfig,
    cnn_cfg: CnnConfig,
    target_objects: Sequence[str | None],
    models: Sequence[str],
) -> None:
    """Fail-fast verify that descriptor CSVs, CNN PNG directories, and checkpoints exist before execution."""
    if not cls_cfg.effective_input_csv.is_file():
        raise FileNotFoundError(
            f"Missing x94 descriptor CSV: {cls_cfg.effective_input_csv}. "
            "Run experiments.append_asymmetry_features first."
        )
    if not cnn_cfg.effective_metadata_csv.is_file():
        raise FileNotFoundError(
            f"Missing CNN metadata CSV: {cnn_cfg.effective_metadata_csv}."
        )
    if not cnn_cfg.effective_preprocessed_dir.is_dir():
        raise FileNotFoundError(
            f"Missing preprocessed PNG directory: {cnn_cfg.effective_preprocessed_dir}."
        )

    missing_ckpts: list[tuple[str | None, str, Path]] = []
    for obj in target_objects:
        for m_name in models:
            ckpt = cnn_cfg.effective_results_dir(obj, m_name) / "checkpoint.pt"
            if not ckpt.is_file():
                missing_ckpts.append((obj, m_name, ckpt))

    if missing_ckpts:
        cfg_rel = (
            "configs/cnn/kolektorsdd2.toml"
            if prof_cfg.dataset_key == "kolektorsdd2"
            else "configs/cnn/mvtec_ad.toml"
        )
        inv_flag = " --inverted" if prof_cfg.inverted else ""
        preview = "; ".join(f"{o or 'full'}/{m} -> {p}" for o, m, p in missing_ckpts[:4])
        extra = f" (+{len(missing_ckpts) - 4} more)" if len(missing_ckpts) > 4 else ""
        sample_obj = missing_ckpts[0][0]
        obj_flag = f" --object {sample_obj}" if sample_obj else ""
        raise FileNotFoundError(
            f"Missing {len(missing_ckpts)} GPU-trained CNN checkpoint(s): {preview}{extra}. "
            "Per protocol §10.1, train checkpoints once before CPU profiling via: "
            f".\\.venv\\Scripts\\python.exe -m experiments.evaluate_cnn --config {cfg_rel}{obj_flag}{inv_flag} --model all"
        )


def _prepare_real_inputs(
    cnn_part: CnnDatasetPartition,
    cls_part: DatasetPartition,
    cnn_cfg: CnnConfig,
    n_real_samples: int,
    seed: int,
) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """Deterministically slice real test uint8 images, normalized CNN tensors, and ML feature rows."""
    n_cnn_test = int(cnn_part.X_test.shape[0])
    n_use = min(n_real_samples, n_cnn_test)
    rng = np.random.default_rng(seed)
    sel_idx = np.sort(rng.choice(n_cnn_test, size=n_use, replace=False))
    x_real_cnn = cnn_part.X_test[torch.from_numpy(sel_idx).to(torch.long)].contiguous()

    # Exact integer inverse of (u8 / 255.0 - mean) / std
    arr_f = x_real_cnn.squeeze(1).cpu().numpy()
    images_u8 = np.rint(
        (arr_f * cnn_cfg.data.normalize_std + cnn_cfg.data.normalize_mean) * 255.0
    ).clip(0, 255).astype(np.uint8)
    images_u8 = np.ascontiguousarray(images_u8)

    n_ml_test = int(cls_part.X_test.shape[0])
    if n_ml_test >= n_cnn_test:
        ml_sel = sel_idx
    else:
        n_ml_use = min(n_real_samples, n_ml_test)
        ml_sel = np.sort(np.random.default_rng(seed).choice(n_ml_test, size=n_ml_use, replace=False))
    x_ml_real = np.ascontiguousarray(cls_part.X_test[ml_sel], dtype=np.float64)
    return images_u8, x_real_cnn, x_ml_real


def _make_u8_batches(images_u8: np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Slice contiguous [batch_size, H, W] uint8 arrays outside timed blocks."""
    n = int(images_u8.shape[0])
    if batch_size == 1:
        return [np.ascontiguousarray(images_u8[i : i + 1]) for i in range(n)]
    n_batches = max(1, (n + batch_size - 1) // batch_size)
    batches: list[np.ndarray] = []
    for b in range(n_batches):
        idx = (np.arange(b * batch_size, (b + 1) * batch_size) % n).astype(np.int64)
        batches.append(np.ascontiguousarray(images_u8[idx]))
    return batches


def _measure_cnn_preprocess_raw_ms(
    images_u8: np.ndarray,
    normalize_mean: float,
    normalize_std: float,
    batch_size: int,
    warmup_iters: int,
    timed_iters: int,
    disable_gc: bool,
) -> list[float]:
    """Return per-sample normalized preprocessing latencies (ms) aligned with timed_iters."""
    n_avail = int(images_u8.shape[0])
    batches_u8: list[np.ndarray] = []
    if batch_size == 1:
        for i in range(n_avail):
            batches_u8.append(np.ascontiguousarray(images_u8[i]))
    else:
        n_batches = max(1, (n_avail + batch_size - 1) // batch_size)
        for b in range(n_batches):
            idx = (np.arange(b * batch_size, (b + 1) * batch_size) % n_avail).astype(np.int64)
            batches_u8.append(np.ascontiguousarray(images_u8[idx]))
    n_b = len(batches_u8)

    def _prep(idx: int) -> torch.Tensor:
        sample = batches_u8[idx % n_b]
        arr = sample.astype(np.float32) / 255.0
        arr = (arr - normalize_mean) / normalize_std
        t = torch.from_numpy(arr)
        return t.unsqueeze(0).unsqueeze(0) if sample.ndim == 2 else t.unsqueeze(1)

    raw_batch_ms = measure_indexed_latencies(
        fn=_prep,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        disable_gc=disable_gc,
    )
    return [float(v / batch_size) for v in raw_batch_ms]


def _verify_cnn_checkpoint_on_cpu(
    m_name: str,
    ckpt_path: Path,
    cnn_cfg: CnnConfig,
    object_name: str | None,
    partition_cache: dict[int, CnnDatasetPartition],
    metrics_tolerance: float,
) -> tuple[nn.Module, ClassificationMetrics, float | None]:
    """Load saved CNN weights on CPU, re-evaluate metrics, and verify parity against GPU summary JSON."""
    summary_path = ckpt_path.parent / "metrics_summary.json"
    summary_data: dict[str, Any] = {}
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as f:
            summary_data = json.load(f)

    eval_seed = int(summary_data.get("seed", cnn_cfg.seed))
    th_cal = summary_data.get("threshold_calibration", {})
    th_mode = str(th_cal.get("threshold_mode", cnn_cfg.training.threshold_mode))

    if eval_seed not in partition_cache:
        partition_cache[eval_seed] = load_cnn_partition(
            cnn_cfg, object_name=object_name, seed=eval_seed
        )
    eval_part = partition_cache[eval_seed]

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = create_model(m_name, num_classes=2).to("cpu")
    model.load_state_dict(state)
    model.eval()

    cpu_metrics, _, _, _ = evaluate_cnn(
        model=model,
        partition=eval_part,
        cfg=cnn_cfg,
        model_name=m_name,
        train_elapsed_s=0.0,
        device=torch.device("cpu"),
        batch_size=cnn_cfg.training.batch_size,
        threshold_mode=th_mode,
    )

    parity_max_diff: float | None = None
    saved_metrics = summary_data.get("metrics")
    if isinstance(saved_metrics, Mapping):
        diffs: list[float] = []
        for k in ("accuracy", "balanced_accuracy", "f1", "roc_auc", "pr_auc"):
            if k in saved_metrics:
                diffs.append(abs(float(getattr(cpu_metrics, k)) - float(saved_metrics[k])))
        if diffs:
            parity_max_diff = max(diffs)
            if parity_max_diff > metrics_tolerance:
                logger.warning(
                    "CPU parity difference %.3e exceeds tolerance %.1e for CNN '%s' (%s). "
                    "Saved seed=%d, th_mode=%s; check deterministic settings.",
                    parity_max_diff,
                    metrics_tolerance,
                    m_name,
                    object_name or "full",
                    eval_seed,
                    th_mode,
                )

    return model, cpu_metrics, parity_max_diff


def run_cpu_fair_target(
    prof_cfg: ProfilingConfig,
    cls_cfg: ClassificationConfig,
    cnn_cfg: CnnConfig,
    object_name: str | None,
    classifiers: Sequence[str],
    models: Sequence[str],
    threads_list: Sequence[int],
    batch_sizes: Sequence[int],
    primary_threads: int,
    primary_batch_size: int,
    include_cuda_reference: bool,
    output_dir: Path,
    dry_run: bool = False,
    write_outputs: bool = True,
) -> CpuFairBenchmarkBundle:
    """Execute Phase A CPU-fair benchmarking for a single dataset/object variant."""
    tag = effective_variant_tag(prof_cfg.dataset_name, object_name, prof_cfg.inverted)
    cls_part = load_and_split_descriptors(cls_cfg, object_name=object_name, subset_name="x94_full")
    base_cnn_part = load_cnn_partition(cnn_cfg, object_name=object_name, seed=prof_cfg.seed)

    images_u8, x_real_cnn, x_ml_real = _prepare_real_inputs(
        cnn_part=base_cnn_part,
        cls_part=cls_part,
        cnn_cfg=cnn_cfg,
        n_real_samples=prof_cfg.protocol.n_real_samples,
        seed=prof_cfg.seed,
    )

    partition_cache: dict[int, CnnDatasetPartition] = {prof_cfg.seed: base_cnn_part}
    desc_cfg = prof_cfg.descriptor
    env = capture_environment(prof_cfg.project_root, seed=prof_cfg.seed)

    if dry_run:
        # Lightweight shape & checkpoint verification without writing artifacts
        x_sample = extract_x94_end_to_end(
            images_u8[:1],
            grid_levels=desc_cfg.grid_levels,
            bins=desc_cfg.bins,
            asymmetry_scales=desc_cfg.asymmetry_scales,
            asymmetry_kinds=desc_cfg.asymmetry_kinds,
        )
        for m_name in models:
            ckpt = cnn_cfg.effective_results_dir(object_name, m_name) / "checkpoint.pt"
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            mdl = create_model(m_name, num_classes=2).to("cpu")
            mdl.load_state_dict(state)
            mdl.eval()
            with torch.inference_mode():
                out = mdl(x_real_cnn[:1])
            if tuple(out.shape) != (1, 2):
                raise RuntimeError(f"Model '{m_name}' unexpected output shape: {tuple(out.shape)}")

        logger.info(
            "Dry-run OK: %s | ML split test=%s | CNN split test=%s | real_samples=%d | x94=%s | checkpoints=%d OK",
            tag,
            cls_part.test_counts,
            base_cnn_part.test_counts,
            int(images_u8.shape[0]),
            tuple(x_sample.shape),
            len(models),
        )
        return CpuFairBenchmarkBundle(
            dataset_tag=tag,
            object_name=object_name,
            inverted=prof_cfg.inverted,
            environment=env,
            cls_partition=cls_part,
            images_u8=images_u8,
            x_real_cnn=x_real_cnn,
            x_ml_real=x_ml_real,
            fitted_classifiers={},
            loaded_cnns={},
            cpu_rows=(),
            cuda_rows=(),
            raw_batch1_primary_latencies_ms={},
            cpu_csv_path=None,
            cuda_csv_path=None,
            summary_csv_path=None,
            pareto_plot_paths=(),
        )

    # 1. Fit ML models once deterministically and get test metrics + real param/ops definitions
    fitted_classifiers: dict[
        str, tuple[Any, ClassificationMetrics, ClassifierParamInfo, ClassifierOpsEstimate]
    ] = {}
    for cls_name in classifiers:
        seed_everything(cls_cfg.seed)
        ml_model, cls_metrics = train_and_evaluate(
            cls_part, cls_name, cls_cfg, subset_name="x94_full"
        )
        # Inspect real params/ops outside timing
        from src.profiling import count_classical_model_params, estimate_classical_inference_ops

        p_info = count_classical_model_params(ml_model)
        ops_est = estimate_classical_inference_ops(ml_model, n_features=cls_metrics.n_features)
        fitted_classifiers[cls_name] = (ml_model, cls_metrics, p_info, ops_est)

    # 2. Load GPU-trained CNN checkpoints on CPU and verify CPU metric parity
    loaded_cnns: dict[str, tuple[nn.Module, Path, ClassificationMetrics, float | None]] = {}
    cnn_static_stats: dict[str, tuple[int, int]] = {}
    for m_name in models:
        ckpt_path = cnn_cfg.effective_results_dir(object_name, m_name) / "checkpoint.pt"
        cnn_model, cpu_metrics, parity_diff = _verify_cnn_checkpoint_on_cpu(
            m_name=m_name,
            ckpt_path=ckpt_path,
            cnn_cfg=cnn_cfg,
            object_name=object_name,
            partition_cache=partition_cache,
            metrics_tolerance=prof_cfg.cnn.metrics_tolerance,
        )
        loaded_cnns[m_name] = (cnn_model, ckpt_path, cpu_metrics, parity_diff)
        n_params, _, _ = model_stats(cnn_model)
        n_flops = estimate_flops(cnn_model, input_size=cnn_cfg.data.image_size)
        cnn_static_stats[m_name] = (n_params, n_flops)

    # 3. CPU Fair Timing Grid: same real samples, same thread limits, same batch sizes
    desc_ops = estimate_descriptor_ops(
        image_size=desc_cfg.image_size,
        grid_levels=desc_cfg.grid_levels,
        bins=desc_cfg.bins,
        asymmetry_scales=desc_cfg.asymmetry_scales,
        asymmetry_kinds=desc_cfg.asymmetry_kinds,
    )
    n_desc_features = (
        sum(int(g) * int(g) for g in desc_cfg.grid_levels)
        + len(desc_cfg.asymmetry_scales) * len(desc_cfg.asymmetry_kinds)
    )
    plan = build_entropy_plan(desc_cfg.image_size, desc_cfg.grid_levels, desc_cfg.bins)

    cpu_rows: list[dict[str, Any]] = []
    raw_batch1_primary_ms: dict[str, tuple[float, ...]] = {}
    n_real_used = int(images_u8.shape[0])

    for th in threads_list:
        for b_size in batch_sizes:
            n_iters = (
                prof_cfg.protocol.timed_iters
                if b_size == 1
                else prof_cfg.protocol.batch_timed_iters
            )
            desc_batches = _make_u8_batches(images_u8, batch_size=b_size)
            n_db = len(desc_batches)
            is_primary = th == primary_threads and b_size == primary_batch_size
            n_repeats = prof_cfg.protocol.repeats if is_primary else 1

            with thread_limits(th):
                # Measure descriptor x94 extraction latencies once for all ML classifiers
                raw_desc_ms = [
                    float(v / b_size)
                    for v in measure_indexed_latencies(
                        fn=lambda idx: extract_x94_end_to_end(
                            desc_batches[idx % n_db],
                            grid_levels=desc_cfg.grid_levels,
                            bins=desc_cfg.bins,
                            asymmetry_scales=desc_cfg.asymmetry_scales,
                            asymmetry_kinds=desc_cfg.asymmetry_kinds,
                            plan=plan,
                        ),
                        warmup_iters=prof_cfg.protocol.warmup_iters,
                        timed_iters=n_iters,
                        disable_gc=prof_cfg.protocol.disable_gc,
                    )
                ]
                desc_stats = summarize_latencies(
                    raw_desc_ms,
                    per_sample_divisor=1,
                    bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
                    confidence_level=prof_cfg.statistics.confidence_level,
                    seed=prof_cfg.seed,
                )

                for cls_name in classifiers:
                    ml_model, cls_metrics, p_info, ops_est = fitted_classifiers[cls_name]
                    cls_rec = profile_classifier(
                        model=ml_model,
                        classifier_name=cls_name,
                        X_test=x_ml_real,
                        n_train=int(cls_part.X_train.shape[0]),
                        threads=th,
                        batch_size=b_size,
                        warmup_iters=prof_cfg.protocol.warmup_iters,
                        timed_iters=n_iters,
                        max_real_samples=int(x_ml_real.shape[0]),
                        disable_gc=prof_cfg.protocol.disable_gc,
                        bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
                        confidence_level=prof_cfg.statistics.confidence_level,
                        seed=prof_cfg.seed,
                    )
                    raw_total_ms = [
                        d + m for d, m in zip(raw_desc_ms, cls_rec.raw_predict_samples_ms)
                    ]
                    total_stats = summarize_latencies(
                        raw_total_ms,
                        per_sample_divisor=1,
                        bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
                        confidence_level=prof_cfg.statistics.confidence_level,
                        seed=prof_cfg.seed,
                    )

                    # Multi-run inter-run stability when in primary protocol (repeats > 1)
                    inter_med: float | None = None
                    inter_std: float | None = None
                    inter_cv: float | None = None
                    if is_primary and n_repeats > 1:
                        run_meds = [total_stats.median_ms]
                        ml_batches = [
                            np.ascontiguousarray(x_ml_real[i : i + 1])
                            for i in range(int(x_ml_real.shape[0]))
                        ]
                        n_mb = len(ml_batches)
                        configure_classifier_threads(ml_model, th)
                        for _rep in range(1, n_repeats):
                            r_d = [
                                float(v / b_size)
                                for v in measure_indexed_latencies(
                                    fn=lambda idx: extract_x94_end_to_end(
                                        desc_batches[idx % n_db],
                                        grid_levels=desc_cfg.grid_levels,
                                        bins=desc_cfg.bins,
                                        asymmetry_scales=desc_cfg.asymmetry_scales,
                                        asymmetry_kinds=desc_cfg.asymmetry_kinds,
                                        plan=plan,
                                    ),
                                    warmup_iters=max(5, prof_cfg.protocol.warmup_iters // 2),
                                    timed_iters=n_iters,
                                    disable_gc=prof_cfg.protocol.disable_gc,
                                )
                            ]
                            r_m = [
                                float(v / b_size)
                                for v in measure_indexed_latencies(
                                    fn=lambda idx: ml_model.predict(ml_batches[idx % n_mb]),
                                    warmup_iters=max(5, prof_cfg.protocol.warmup_iters // 2),
                                    timed_iters=n_iters,
                                    disable_gc=prof_cfg.protocol.disable_gc,
                                )
                            ]
                            run_meds.append(statistics.median([a + b for a, b in zip(r_d, r_m)]))
                        multi = summarize_multi_run(run_meds)
                        inter_med = multi.overall_median_ms
                        inter_std = multi.inter_run_std_ms
                        inter_cv = multi.inter_run_cv_pct

                    _, mem_snap = measure_memory_alloc(
                        lambda: ml_model.predict(
                            extract_x94_end_to_end(
                                desc_batches[0],
                                grid_levels=desc_cfg.grid_levels,
                                bins=desc_cfg.bins,
                                asymmetry_scales=desc_cfg.asymmetry_scales,
                                asymmetry_kinds=desc_cfg.asymmetry_kinds,
                                plan=plan,
                            )
                        )
                    )

                    if is_primary:
                        raw_batch1_primary_ms[cls_name] = tuple(raw_total_ms)
                        raw_batch1_primary_ms[f"{cls_name} (x94)"] = tuple(raw_total_ms)

                    cpu_rows.append(
                        {
                            "model": f"{cls_name} (x94)",
                            "family": "ml",
                            "device": "cpu",
                            "threads": th,
                            "batch_size": b_size,
                            "params_model": p_info.params_model,
                            "params_descriptor": n_desc_features,
                            "macs": ops_est.analytical_macs,
                            "flops": desc_ops.total_ops + ops_est.analytical_flops,
                            "ops_basis": "analytical_estimate",
                            "t_preprocess_ms_median": 0.0,
                            "t_descriptor_ms_median": desc_stats.median_ms,
                            "t_model_ms_median": cls_rec.predict_stats.median_ms,
                            "t_total_ms_median": total_stats.median_ms,
                            "t_total_ms_iqr": total_stats.iqr_ms,
                            "t_total_ms_p95": total_stats.p95_ms,
                            "t_total_ms_mean": total_stats.mean_ms,
                            "t_total_ms_std": total_stats.std_ms,
                            "t_total_ci95_low_ms": total_stats.ci95_low_ms,
                            "t_total_ci95_high_ms": total_stats.ci95_high_ms,
                            "inter_run_median_ms": inter_med,
                            "inter_run_std_ms": inter_std,
                            "inter_run_cv_pct": inter_cv,
                            "throughput_samples_s": total_stats.throughput_samples_s,
                            "peak_ram_mb": mem_snap.rss_peak_mb,
                            "accuracy": cls_metrics.accuracy,
                            "balanced_accuracy": cls_metrics.balanced_accuracy,
                            "f1": cls_metrics.f1,
                            "roc_auc": cls_metrics.roc_auc,
                            "pr_auc": cls_metrics.pr_auc,
                            "metric_parity_max_diff": None,
                            "seed": cls_cfg.seed,
                            "n_samples": n_real_used,
                            "timestamp": env.timestamp_utc,
                            "git_commit": env.git_commit,
                        }
                    )

                # CNNs: preprocessing (uint8 -> normalized [B, 1, H, W]) + forward pass on CPU
                raw_prep_ms = _measure_cnn_preprocess_raw_ms(
                    images_u8=images_u8,
                    normalize_mean=cnn_cfg.data.normalize_mean,
                    normalize_std=cnn_cfg.data.normalize_std,
                    batch_size=b_size,
                    warmup_iters=prof_cfg.protocol.warmup_iters,
                    timed_iters=n_iters,
                    disable_gc=prof_cfg.protocol.disable_gc,
                )
                prep_stats = summarize_latencies(
                    raw_prep_ms,
                    per_sample_divisor=1,
                    bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
                    confidence_level=prof_cfg.statistics.confidence_level,
                    seed=prof_cfg.seed,
                )

                for m_name in models:
                    cnn_model, _, cpu_metrics, parity_diff = loaded_cnns[m_name]
                    n_params, n_flops = cnn_static_stats[m_name]

                    batch_lat_ms, _ = benchmark_model_latency(
                        model=cnn_model,
                        device=torch.device("cpu"),
                        batch_size=b_size,
                        image_size=cnn_cfg.data.image_size,
                        warmup_iters=prof_cfg.protocol.warmup_iters,
                        timed_iters=n_iters,
                        threads=th,
                        inputs=x_real_cnn,
                        disable_gc=prof_cfg.protocol.disable_gc,
                    )
                    raw_cnn_model_ms = [float(v / b_size) for v in batch_lat_ms]
                    model_lat_stats = summarize_latencies(
                        raw_cnn_model_ms,
                        per_sample_divisor=1,
                        bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
                        confidence_level=prof_cfg.statistics.confidence_level,
                        seed=prof_cfg.seed,
                    )
                    raw_cnn_total_ms = [p + m for p, m in zip(raw_prep_ms, raw_cnn_model_ms)]
                    total_stats = summarize_latencies(
                        raw_cnn_total_ms,
                        per_sample_divisor=1,
                        bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
                        confidence_level=prof_cfg.statistics.confidence_level,
                        seed=prof_cfg.seed,
                    )

                    inter_med = None
                    inter_std = None
                    inter_cv = None
                    if is_primary and n_repeats > 1:
                        run_meds = [total_stats.median_ms]
                        for _rep in range(1, n_repeats):
                            rep_prep = _measure_cnn_preprocess_raw_ms(
                                images_u8=images_u8,
                                normalize_mean=cnn_cfg.data.normalize_mean,
                                normalize_std=cnn_cfg.data.normalize_std,
                                batch_size=b_size,
                                warmup_iters=max(5, prof_cfg.protocol.warmup_iters // 2),
                                timed_iters=n_iters,
                                disable_gc=prof_cfg.protocol.disable_gc,
                            )
                            rep_mod_batch, _ = benchmark_model_latency(
                                model=cnn_model,
                                device=torch.device("cpu"),
                                batch_size=b_size,
                                image_size=cnn_cfg.data.image_size,
                                warmup_iters=max(5, prof_cfg.protocol.warmup_iters // 2),
                                timed_iters=n_iters,
                                threads=th,
                                inputs=x_real_cnn,
                                disable_gc=prof_cfg.protocol.disable_gc,
                            )
                            rep_mod = [float(v / b_size) for v in rep_mod_batch]
                            run_meds.append(statistics.median([p + m for p, m in zip(rep_prep, rep_mod)]))
                        multi = summarize_multi_run(run_meds)
                        inter_med = multi.overall_median_ms
                        inter_std = multi.inter_run_std_ms
                        inter_cv = multi.inter_run_cv_pct

                    with torch.inference_mode():
                        _, mem_snap = measure_memory_alloc(
                            lambda: cnn_model(x_real_cnn[: min(b_size, x_real_cnn.shape[0])])
                        )

                    if is_primary:
                        raw_batch1_primary_ms[m_name] = tuple(raw_cnn_total_ms)

                    cpu_rows.append(
                        {
                            "model": m_name,
                            "family": "cnn",
                            "device": "cpu",
                            "threads": th,
                            "batch_size": b_size,
                            "params_model": n_params,
                            "params_descriptor": 0,
                            "macs": n_flops // 2,
                            "flops": n_flops,
                            "ops_basis": "torch_flop_counter",
                            "t_preprocess_ms_median": prep_stats.median_ms,
                            "t_descriptor_ms_median": 0.0,
                            "t_model_ms_median": model_lat_stats.median_ms,
                            "t_total_ms_median": total_stats.median_ms,
                            "t_total_ms_iqr": total_stats.iqr_ms,
                            "t_total_ms_p95": total_stats.p95_ms,
                            "t_total_ms_mean": total_stats.mean_ms,
                            "t_total_ms_std": total_stats.std_ms,
                            "t_total_ci95_low_ms": total_stats.ci95_low_ms,
                            "t_total_ci95_high_ms": total_stats.ci95_high_ms,
                            "inter_run_median_ms": inter_med,
                            "inter_run_std_ms": inter_std,
                            "inter_run_cv_pct": inter_cv,
                            "throughput_samples_s": total_stats.throughput_samples_s,
                            "peak_ram_mb": mem_snap.rss_peak_mb,
                            "accuracy": cpu_metrics.accuracy,
                            "balanced_accuracy": cpu_metrics.balanced_accuracy,
                            "f1": cpu_metrics.f1,
                            "roc_auc": cpu_metrics.roc_auc,
                            "pr_auc": cpu_metrics.pr_auc,
                            "metric_parity_max_diff": parity_diff,
                            "seed": cnn_cfg.seed,
                            "n_samples": n_real_used,
                            "timestamp": env.timestamp_utc,
                            "git_commit": env.git_commit,
                        }
                    )

    # 4. Isolated CUDA secondary reference (never written to CPU comparison CSV or summary)
    cuda_rows: list[dict[str, Any]] = []
    if include_cuda_reference and torch.cuda.is_available():
        cuda_dev = torch.device("cuda")
        for b_size in batch_sizes:
            n_iters = (
                prof_cfg.protocol.timed_iters
                if b_size == 1
                else prof_cfg.protocol.batch_timed_iters
            )
            for m_name in models:
                cnn_model, _, cpu_metrics, parity_diff = loaded_cnns[m_name]
                n_params, n_flops = cnn_static_stats[m_name]
                batch_lat_ms, peak_vram = benchmark_model_latency(
                    model=cnn_model,
                    device=cuda_dev,
                    batch_size=b_size,
                    image_size=cnn_cfg.data.image_size,
                    warmup_iters=prof_cfg.protocol.warmup_iters,
                    timed_iters=n_iters,
                    threads=primary_threads,
                    inputs=x_real_cnn,
                    disable_gc=prof_cfg.protocol.disable_gc,
                )
                cnn_model.to("cpu")
                raw_cuda_ms = [float(v / b_size) for v in batch_lat_ms]
                c_stats = summarize_latencies(
                    raw_cuda_ms,
                    per_sample_divisor=1,
                    bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
                    confidence_level=prof_cfg.statistics.confidence_level,
                    seed=prof_cfg.seed,
                )
                cuda_rows.append(
                    {
                        "model": m_name,
                        "family": "cnn",
                        "device": "cuda",
                        "threads": 0,
                        "batch_size": b_size,
                        "params_model": n_params,
                        "params_descriptor": 0,
                        "macs": n_flops // 2,
                        "flops": n_flops,
                        "ops_basis": "torch_flop_counter",
                        "t_preprocess_ms_median": 0.0,
                        "t_descriptor_ms_median": 0.0,
                        "t_model_ms_median": c_stats.median_ms,
                        "t_total_ms_median": c_stats.median_ms,
                        "t_total_ms_iqr": c_stats.iqr_ms,
                        "t_total_ms_p95": c_stats.p95_ms,
                        "t_total_ms_mean": c_stats.mean_ms,
                        "t_total_ms_std": c_stats.std_ms,
                        "t_total_ci95_low_ms": c_stats.ci95_low_ms,
                        "t_total_ci95_high_ms": c_stats.ci95_high_ms,
                        "inter_run_median_ms": None,
                        "inter_run_std_ms": None,
                        "inter_run_cv_pct": None,
                        "throughput_samples_s": c_stats.throughput_samples_s,
                        "peak_ram_mb": peak_vram if peak_vram is not None else 0.0,
                        "accuracy": cpu_metrics.accuracy,
                        "balanced_accuracy": cpu_metrics.balanced_accuracy,
                        "f1": cpu_metrics.f1,
                        "roc_auc": cpu_metrics.roc_auc,
                        "pr_auc": cpu_metrics.pr_auc,
                        "metric_parity_max_diff": parity_diff,
                        "seed": cnn_cfg.seed,
                        "n_samples": n_real_used,
                        "timestamp": env.timestamp_utc,
                        "git_commit": env.git_commit,
                    }
                )

    cpu_csv_path: Path | None = None
    cuda_csv_path: Path | None = None
    summary_csv_path: Path | None = None
    pareto_paths: list[Path] = []

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        cpu_csv_path = write_comparison_csv(
            output_path=output_dir / f"{tag}_baseline_vs_cnn_cpu.csv",
            rows=cpu_rows,
            precision=prof_cfg.output.float_precision,
        )
        if cuda_rows:
            cuda_csv_path = write_comparison_csv(
                output_path=output_dir / f"{tag}_cuda_reference.csv",
                rows=cuda_rows,
                precision=prof_cfg.output.float_precision,
            )
        summary_csv_path = upsert_summary_all_datasets(
            summary_csv_path=output_dir / "summary_all_datasets.csv",
            dataset_tag=tag,
            object_name=object_name,
            inverted=prof_cfg.inverted,
            cpu_rows=cpu_rows,
            precision=prof_cfg.output.float_precision,
        )
        tp_batch = max(batch_sizes)
        for th in threads_list:
            p_png = plot_cpu_comparison_pareto(
                output_path=output_dir / f"{tag}_pareto_threads{th}.png",
                rows=cpu_rows,
                dataset_label=tag,
                threads=th,
                primary_batch_size=primary_batch_size,
                throughput_batch_size=tp_batch,
                dpi=prof_cfg.output.dpi,
            )
            if p_png is not None:
                pareto_paths.append(p_png)

        logger.info(
            "Completed CPU-fair benchmark '%s': %d CPU rows -> %s (summary -> %s)",
            tag,
            len(cpu_rows),
            cpu_csv_path,
            summary_csv_path,
        )

    return CpuFairBenchmarkBundle(
        dataset_tag=tag,
        object_name=object_name,
        inverted=prof_cfg.inverted,
        environment=env,
        cls_partition=cls_part,
        images_u8=images_u8,
        x_real_cnn=x_real_cnn,
        x_ml_real=x_ml_real,
        fitted_classifiers=fitted_classifiers,
        loaded_cnns=loaded_cnns,
        cpu_rows=tuple(cpu_rows),
        cuda_rows=tuple(cuda_rows),
        raw_batch1_primary_latencies_ms=raw_batch1_primary_ms,
        cpu_csv_path=cpu_csv_path,
        cuda_csv_path=cuda_csv_path,
        summary_csv_path=summary_csv_path,
        pareto_plot_paths=tuple(pareto_paths),
    )


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """CLI entry point for 100% CPU fair benchmarking."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.device.lower() != "cpu":
            raise ValueError(
                f"Primary benchmark mode must run on 'cpu' for fair comparison, got '--device {args.device}'. "
                "CUDA is recorded only in an isolated secondary reference CSV."
            )

        cfg_rel = resolve_profiling_config_path(args.config, args.dataset)
        prof_cfg = load_profiling_config(cfg_rel, project_root)

        if args.dataset is not None:
            ds_req = "kolektorsdd2" if args.dataset.lower() == "kolektorsdd2" else "mvtecad"
            if prof_cfg.dataset_key != ds_req:
                raise ValueError(
                    f"--dataset '{args.dataset}' conflicts with loaded config dataset_key '{prof_cfg.dataset_key}'"
                )

        if args.inverted:
            prof_cfg = replace(prof_cfg, inverted=True)

        cls_cfg = replace(
            load_classification_config(prof_cfg.classification_config_path, prof_cfg.project_root),
            inverted=prof_cfg.inverted,
            overwrite=prof_cfg.overwrite,
        )
        cnn_cfg = replace(
            load_cnn_config(prof_cfg.cnn_config_path, prof_cfg.project_root),
            inverted=prof_cfg.inverted,
            overwrite=prof_cfg.overwrite,
        )

        target_objects = resolve_target_objects(
            prof_cfg,
            object_name=args.object_name,
            objects=args.objects,
            all_objects=args.all_objects,
        )
        classifiers = resolve_selected_classifiers(args.classifiers, prof_cfg)
        models = resolve_selected_models(args.models, prof_cfg)

        validate_required_inputs_upfront(
            prof_cfg=prof_cfg,
            cls_cfg=cls_cfg,
            cnn_cfg=cnn_cfg,
            target_objects=target_objects,
            models=models,
        )

        threads_list = (
            tuple(sorted({int(t) for t in args.threads if int(t) >= 1}))
            if args.threads
            else prof_cfg.protocol.effective_threads_list()
        )
        if not threads_list:
            raise ValueError("Resolved threads list cannot be empty")
        primary_threads = (
            prof_cfg.protocol.primary_threads
            if prof_cfg.protocol.primary_threads in threads_list
            else threads_list[0]
        )

        batch_sizes = (
            tuple(sorted({int(b) for b in args.batch_sizes if int(b) >= 1}))
            if args.batch_sizes
            else prof_cfg.protocol.batch_sizes
        )
        if not batch_sizes:
            raise ValueError("Resolved batch_sizes list cannot be empty")
        primary_batch_size = (
            prof_cfg.protocol.primary_batch_size
            if prof_cfg.protocol.primary_batch_size in batch_sizes
            else batch_sizes[0]
        )

        include_cuda_ref = prof_cfg.protocol.include_cuda_reference and not args.no_cuda_reference
        out_dir = (
            (prof_cfg.project_root / args.output_dir).resolve()
            if args.output_dir is not None
            else prof_cfg.output.comparison_cpu_dir
        )

        for obj in target_objects:
            run_cpu_fair_target(
                prof_cfg=prof_cfg,
                cls_cfg=cls_cfg,
                cnn_cfg=cnn_cfg,
                object_name=obj,
                classifiers=classifiers,
                models=models,
                threads_list=threads_list,
                batch_sizes=batch_sizes,
                primary_threads=primary_threads,
                primary_batch_size=primary_batch_size,
                include_cuda_reference=include_cuda_ref,
                output_dir=out_dir,
                dry_run=args.dry_run,
                write_outputs=not args.dry_run,
            )

        return 0

    except (OSError, ValueError, KeyError, RuntimeError, FileExistsError) as exc:
        logger.error("CPU fair benchmark error: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
