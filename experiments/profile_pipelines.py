"""CLI and orchestrator for Phase B deep scientific CPU profiling and reporting.

Performs stage-level x94 descriptor profiling, cProfile diagnostics, parameter scalability
sweeps, classical classifier breakdown, CNN leaf-hook + torch.profiler ATen breakdowns,
thread scaling, quality-vs-cost ablation (x85 vs x9 vs x94), Mann-Whitney U tests, 6 figures,
and an automatically populated data-driven README.

Manual execution commands (PowerShell from repository 'code/' directory):
    # 1. Dry-run validation:
    .\\.venv\\Scripts\\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml --dry-run
    .\\.venv\\Scripts\\python.exe -m experiments.profile_pipelines --config configs/profiling/mvtec_ad.toml --objects bottle --dry-run

    # 2. Full Phase B profiling (normal variants per section 10 methodology):
    .\\.venv\\Scripts\\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml
    .\\.venv\\Scripts\\python.exe -m experiments.profile_pipelines --config configs/profiling/mvtec_ad.toml --objects bottle

    # 3. Short descriptor verification on inverted variant (automatic with --inverted):
    .\\.venv\\Scripts\\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml --inverted
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import configure_logging, logger
from experiments.benchmark_cpu_fair import (
    _prepare_real_inputs,
    resolve_profiling_config_path,
    resolve_selected_classifiers,
    resolve_selected_models,
    run_cpu_fair_target,
    validate_required_inputs_upfront,
)
from src.features.asymmetry_maps import compute_asymmetry_features
from src.features.entropy_maps import build_entropy_plan, compute_entropy_features
from src.models.cnn import (
    CnnConfig,
    create_model,
    load_cnn_config,
    load_cnn_partition,
)
from src.models.cnn.architectures.common import seed_everything
from src.models.descriptor_classifier import (
    ClassificationConfig,
    DatasetPartition,
    load_and_split_descriptors,
    load_classification_config,
    resolve_feature_groups,
    slice_partition_subset,
    train_and_evaluate,
)
from src.profiling import (
    ClassifierProfileRecord,
    CnnLayerProfileReport,
    CnnThreadScalingRecord,
    ProfilingConfig,
    StatisticalComparisonResult,
    capture_environment,
    compare_latency_distributions,
    configure_classifier_threads,
    effective_variant_tag,
    extract_x94_end_to_end,
    extract_x94_staged,
    load_profiling_config,
    measure_checkpoint_cold_start,
    measure_indexed_latencies,
    profile_classifier,
    profile_classifier_thread_scaling,
    profile_cnn_layers,
    profile_cnn_thread_scaling,
    profile_descriptor_scalability,
    profile_descriptor_stages,
    resolve_target_objects,
    save_environment_json,
    summarize_latencies,
    thread_limits,
)
from src.profiling.report import (
    generate_profiling_figures,
    write_ablation_cost_csv,
    write_classifier_times_csv,
    write_cnn_layer_artifacts,
    write_cnn_thread_scaling_csv,
    write_comparison_csv,
    write_descriptor_scalability_csv,
    write_descriptor_stage_artifacts,
    write_profiling_readme,
    write_statistical_tests_csv,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for deep pipeline profiling."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.profile_pipelines",
        description="Execute Phase B deep CPU profiling for x94 ML and lightweight CNN pipelines.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to profiling TOML config (defaults to configs/profiling/kolektorsdd2.toml or --dataset).",
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
        help="Explicit list of MVTec-AD object names (e.g. --objects bottle metal_nut).",
    )
    parser.add_argument(
        "--all-objects",
        action="store_true",
        default=False,
        help="Profile across all configured MVTec-AD objects.",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        default=False,
        help="Run on inverted dataset variant (runs short descriptor verification by default).",
    )
    parser.add_argument(
        "--short-descriptor-check",
        action="store_true",
        default=False,
        help="Run only short x94 descriptor verification instead of full CNN/ML profiling.",
    )
    parser.add_argument(
        "--full-inverted",
        action="store_true",
        default=False,
        help="Force full Phase B profiling even when --inverted is enabled.",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=None,
        help="Override thread sweep list (primary latency still uses first/configured primary thread).",
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
        "--no-cuda-reference",
        action="store_true",
        default=False,
        help="Skip optional secondary CUDA reference table in report.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override base output directory (default from TOML: results/profiling).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate configs, splits, numerical equivalence, and checkpoints without running profiling.",
    )
    return parser


def _load_existing_comparison_rows(csv_path: Path) -> list[dict[str, Any]]:
    """Read an existing CPU/CUDA comparison CSV into typed dictionaries when available."""
    if not csv_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    int_keys = {"threads", "batch_size", "params_model", "params_descriptor", "macs", "flops", "seed", "n_samples"}
    float_keys = {
        "t_preprocess_ms_median",
        "t_descriptor_ms_median",
        "t_model_ms_median",
        "t_total_ms_median",
        "t_total_ms_iqr",
        "t_total_ms_p95",
        "t_total_ms_mean",
        "t_total_ms_std",
        "t_total_ci95_low_ms",
        "t_total_ci95_high_ms",
        "throughput_samples_s",
        "peak_ram_mb",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "pr_auc",
        "metric_parity_max_diff",
    }
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            parsed: dict[str, Any] = dict(raw)
            for k in int_keys:
                v = raw.get(k, "")
                if v != "":
                    parsed[k] = int(v)
            for k in float_keys:
                v = raw.get(k, "")
                if v != "":
                    parsed[k] = float(v)
            rows.append(parsed)
    return rows


def _measure_subset_descriptor_ms(
    subset_name: str,
    images_u8: np.ndarray,
    prof_cfg: ProfilingConfig,
    primary_threads: int,
) -> float:
    """Measure CPU single-image descriptor extraction median latency for an ablation subset."""
    desc_cfg = prof_cfg.descriptor
    n_imgs = int(images_u8.shape[0])
    batches = [np.ascontiguousarray(images_u8[i : i + 1]) for i in range(n_imgs)]
    n_b = len(batches)
    n_warmup = min(10, prof_cfg.protocol.warmup_iters)
    n_iters = min(100, prof_cfg.protocol.timed_iters)

    with thread_limits(primary_threads):
        if subset_name == "x85_only":
            plan_85 = build_entropy_plan(desc_cfg.image_size, desc_cfg.grid_levels, desc_cfg.bins)

            def _fn85(idx: int) -> np.ndarray:
                return compute_entropy_features(
                    batches[idx % n_b],
                    grid_levels=desc_cfg.grid_levels,
                    bins=desc_cfg.bins,
                    plan=plan_85,
                )

            raw_ms = measure_indexed_latencies(
                fn=_fn85,
                warmup_iters=n_warmup,
                timed_iters=n_iters,
                disable_gc=prof_cfg.protocol.disable_gc,
            )
        elif subset_name == "x9_only":
            asym_scales = tuple(int(g) for g in desc_cfg.asymmetry_scales)
            plan_asym = build_entropy_plan(desc_cfg.image_size, asym_scales, desc_cfg.bins)

            def _fn9(idx: int) -> np.ndarray:
                e_sub = compute_entropy_features(
                    batches[idx % n_b],
                    grid_levels=asym_scales,
                    bins=desc_cfg.bins,
                    plan=plan_asym,
                )
                e_by: dict[int, np.ndarray] = {}
                off = 0
                for g in asym_scales:
                    cols = g * g
                    e_by[g] = e_sub[:, off : off + cols]
                    off += cols
                return compute_asymmetry_features(
                    e_by, scales=asym_scales, kinds=desc_cfg.asymmetry_kinds
                )

            raw_ms = measure_indexed_latencies(
                fn=_fn9,
                warmup_iters=n_warmup,
                timed_iters=n_iters,
                disable_gc=prof_cfg.protocol.disable_gc,
            )
        elif subset_name == "x94_full":
            plan_94 = build_entropy_plan(desc_cfg.image_size, desc_cfg.grid_levels, desc_cfg.bins)

            def _fn94(idx: int) -> np.ndarray:
                return extract_x94_end_to_end(
                    batches[idx % n_b],
                    grid_levels=desc_cfg.grid_levels,
                    bins=desc_cfg.bins,
                    asymmetry_scales=desc_cfg.asymmetry_scales,
                    asymmetry_kinds=desc_cfg.asymmetry_kinds,
                    plan=plan_94,
                )

            raw_ms = measure_indexed_latencies(
                fn=_fn94,
                warmup_iters=n_warmup,
                timed_iters=n_iters,
                disable_gc=prof_cfg.protocol.disable_gc,
            )
        else:
            raise ValueError(f"Unsupported ablation subset '{subset_name}' for cost profiling.")

    stats = summarize_latencies(raw_ms, per_sample_divisor=1, seed=prof_cfg.seed)
    return stats.median_ms


def _run_ablation_quality_vs_cost(
    prof_cfg: ProfilingConfig,
    cls_cfg: ClassificationConfig,
    cls_partition: DatasetPartition,
    images_u8: np.ndarray,
    classifiers: Sequence[str],
    primary_threads: int,
) -> list[dict[str, Any]]:
    """Quantify quality gain vs marginal CPU latency (`dF1 / dms`) across x85, x9, and x94."""
    groups = resolve_feature_groups(cls_partition.feature_columns, cls_cfg)
    subset_names = tuple(prof_cfg.statistics.ablation_subsets)
    subset_desc_ms = {
        sub: _measure_subset_descriptor_ms(sub, images_u8, prof_cfg, primary_threads)
        for sub in subset_names
    }

    rows: list[dict[str, Any]] = []
    n_real_cls = min(prof_cfg.protocol.n_real_samples, int(cls_partition.X_test.shape[0]))
    n_iters = min(150, prof_cfg.protocol.timed_iters)

    for cls_name in classifiers:
        per_subset_results: dict[str, tuple[int, float, float, float, float, float, float]] = {}
        for sub in subset_names:
            sub_part = slice_partition_subset(cls_partition, sub, groups)
            seed_everything(cls_cfg.seed)
            mdl, metrics = train_and_evaluate(sub_part, cls_name, cls_cfg, subset_name=sub)
            configure_classifier_threads(mdl, primary_threads)

            x_sub_test = np.ascontiguousarray(sub_part.X_test[:n_real_cls], dtype=np.float64)
            batches = [np.ascontiguousarray(x_sub_test[i : i + 1]) for i in range(n_real_cls)]
            n_b = len(batches)
            with thread_limits(primary_threads):
                m_ms = measure_indexed_latencies(
                    fn=lambda idx: mdl.predict(batches[idx % n_b]),
                    warmup_iters=min(10, prof_cfg.protocol.warmup_iters),
                    timed_iters=n_iters,
                    disable_gc=prof_cfg.protocol.disable_gc,
                )
            t_mod_ms = summarize_latencies(m_ms, per_sample_divisor=1, seed=prof_cfg.seed).median_ms
            t_desc_ms = subset_desc_ms[sub]
            t_tot_ms = t_desc_ms + t_mod_ms
            per_subset_results[sub] = (
                metrics.n_features,
                metrics.f1,
                metrics.roc_auc,
                metrics.pr_auc,
                t_desc_ms,
                t_mod_ms,
                t_tot_ms,
            )

        base_f1 = per_subset_results["x85_only"][1] if "x85_only" in per_subset_results else 0.0
        base_ms = per_subset_results["x85_only"][6] if "x85_only" in per_subset_results else 0.0

        for sub in subset_names:
            n_feat, f1_val, roc_val, pr_val, td_ms, tm_ms, tt_ms = per_subset_results[sub]
            d_f1 = f1_val - base_f1
            d_ms = tt_ms - base_ms
            marg = (d_f1 / d_ms) if (sub != "x85_only" and abs(d_ms) > 1e-9) else 0.0
            rows.append(
                {
                    "classifier": cls_name,
                    "subset_name": sub,
                    "n_features": n_feat,
                    "f1": f1_val,
                    "roc_auc": roc_val,
                    "pr_auc": pr_val,
                    "t_descriptor_ms": td_ms,
                    "t_model_ms": tm_ms,
                    "t_total_ms": tt_ms,
                    "delta_f1_vs_x85": d_f1,
                    "delta_ms_vs_x85": d_ms,
                    "marginal_f1_per_ms": marg,
                }
            )
    return rows


def run_profile_target(
    prof_cfg: ProfilingConfig,
    cls_cfg: ClassificationConfig,
    cnn_cfg: CnnConfig,
    object_name: str | None,
    classifiers: Sequence[str],
    models: Sequence[str],
    threads_list: Sequence[int],
    primary_threads: int,
    primary_batch_size: int,
    short_mode: bool,
    include_cuda_reference: bool,
    profiling_base_dir: Path,
    dry_run: bool = False,
) -> Path | None:
    """Execute Phase B profiling (or short inverted descriptor check) for one target variant."""
    tag = effective_variant_tag(prof_cfg.dataset_name, object_name, prof_cfg.inverted)
    out_dir = profiling_base_dir / tag
    desc_cfg = prof_cfg.descriptor

    if short_mode:
        cls_part = load_and_split_descriptors(cls_cfg, object_name=object_name, subset_name="x94_full")
        base_cnn_part = load_cnn_partition(cnn_cfg, object_name=object_name, seed=prof_cfg.seed)
        images_u8, _, _ = _prepare_real_inputs(
            cnn_part=base_cnn_part,
            cls_part=cls_part,
            cnn_cfg=cnn_cfg,
            n_real_samples=prof_cfg.protocol.n_real_samples,
            seed=prof_cfg.seed,
        )

        if dry_run:
            diff = float(
                np.max(
                    np.abs(
                        extract_x94_staged(
                            images_u8[:4],
                            grid_levels=desc_cfg.grid_levels,
                            bins=desc_cfg.bins,
                            asymmetry_scales=desc_cfg.asymmetry_scales,
                            asymmetry_kinds=desc_cfg.asymmetry_kinds,
                        )
                        - extract_x94_end_to_end(
                            images_u8[:4],
                            grid_levels=desc_cfg.grid_levels,
                            bins=desc_cfg.bins,
                            asymmetry_scales=desc_cfg.asymmetry_scales,
                            asymmetry_kinds=desc_cfg.asymmetry_kinds,
                        )
                    )
                )
            )
            logger.info(
                "Dry-run OK (short descriptor check): %s | samples=%d | x94 staged parity diff=%.3e",
                tag,
                int(images_u8.shape[0]),
                diff,
            )
            return None

        env = capture_environment(prof_cfg.project_root, seed=prof_cfg.seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_environment_json(env, out_dir / "environment.json")

        batch_n = max(prof_cfg.protocol.batch_sizes)
        desc_report = profile_descriptor_stages(
            images_u8=images_u8,
            dataset_tag=tag,
            grid_levels=desc_cfg.grid_levels,
            bins=desc_cfg.bins,
            asymmetry_scales=desc_cfg.asymmetry_scales,
            asymmetry_kinds=desc_cfg.asymmetry_kinds,
            threads=primary_threads,
            batch_n_size=batch_n,
            warmup_iters=prof_cfg.protocol.warmup_iters,
            timed_iters=min(100, prof_cfg.protocol.timed_iters),
            batch_timed_iters=min(30, prof_cfg.protocol.batch_timed_iters),
            disable_gc_flag=prof_cfg.protocol.disable_gc,
            bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
            confidence_level=prof_cfg.statistics.confidence_level,
            cprofile_top_n=desc_cfg.cprofile_top_n,
            seed=prof_cfg.seed,
            scaler=None,
        )
        write_descriptor_stage_artifacts(out_dir, desc_report, precision=prof_cfg.output.float_precision)

        existing_cpu_rows = _load_existing_comparison_rows(
            prof_cfg.output.comparison_cpu_dir / f"{tag}_baseline_vs_cnn_cpu.csv"
        )
        existing_cuda_rows = _load_existing_comparison_rows(
            prof_cfg.output.comparison_cpu_dir / f"{tag}_cuda_reference.csv"
        )
        generate_profiling_figures(
            output_dir=out_dir,
            dataset_label=tag,
            desc_report=desc_report,
            desc_scalability=(),
            layer_reports=(),
            cnn_scaling=(),
            cls_scaling=(),
            cpu_comparison_rows=existing_cpu_rows,
            primary_threads=primary_threads,
            primary_batch_size=primary_batch_size,
            throughput_batch_size=batch_n,
            dpi=prof_cfg.output.dpi,
        )
        readme_path = write_profiling_readme(
            output_path=out_dir / "README.md",
            dataset_label=tag,
            env=env,
            desc_report=desc_report,
            cpu_rows=existing_cpu_rows,
            cuda_rows=existing_cuda_rows,
            stat_tests=(),
            ablation_rows=(),
            layer_reports=(),
            primary_threads=primary_threads,
            primary_batch_size=primary_batch_size,
        )
        extra_note = (
            "\n## Nota Metodologica (Variante Invertida — Short Descriptor Check)\n\n"
            "Siguiendo el protocolo §10, las variantes invertidas ejecutan comparacion CPU completa (Fase A) "
            "y esta verificacion corta del descriptor x94 para confirmar empiricamente que la inversion fotometrica "
            f"(`255 - x`) conserva el mismo perfil computacional (E2E mediana batch=1 = **{desc_report.e2e_batch1_stats.median_ms:.4f} ms**, "
            f"staged sum = **{desc_report.staged_sum_batch1_ms:.4f} ms**).\n"
        )
        readme_path.write_text(readme_path.read_text(encoding="utf-8") + extra_note, encoding="utf-8")
        logger.info("Completed short descriptor verification '%s' -> %s", tag, out_dir)
        return out_dir

    # Full Phase B Mode
    if dry_run:
        run_cpu_fair_target(
            prof_cfg=prof_cfg,
            cls_cfg=cls_cfg,
            cnn_cfg=cnn_cfg,
            object_name=object_name,
            classifiers=classifiers,
            models=models,
            threads_list=threads_list,
            batch_sizes=prof_cfg.protocol.batch_sizes,
            primary_threads=primary_threads,
            primary_batch_size=primary_batch_size,
            include_cuda_reference=False,
            output_dir=out_dir,
            dry_run=True,
            write_outputs=False,
        )
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_bundle = run_cpu_fair_target(
        prof_cfg=prof_cfg,
        cls_cfg=cls_cfg,
        cnn_cfg=cnn_cfg,
        object_name=object_name,
        classifiers=classifiers,
        models=models,
        threads_list=threads_list,
        batch_sizes=prof_cfg.protocol.batch_sizes,
        primary_threads=primary_threads,
        primary_batch_size=primary_batch_size,
        include_cuda_reference=include_cuda_reference,
        output_dir=out_dir,
        dry_run=False,
        write_outputs=False,
    )

    save_environment_json(cpu_bundle.environment, out_dir / "environment.json")
    write_comparison_csv(
        output_path=out_dir / "comparison_cpu.csv",
        rows=cpu_bundle.cpu_rows,
        precision=prof_cfg.output.float_precision,
    )

    # Extract StandardScaler from LogisticRegression if available to measure scaler cost in stage breakdown
    lr_scaler: StandardScaler | None = None
    if "lr" in cpu_bundle.fitted_classifiers:
        lr_model = cpu_bundle.fitted_classifiers["lr"][0]
        if isinstance(lr_model, Pipeline) and "scaler" in lr_model.named_steps:
            sc_cand = lr_model.named_steps["scaler"]
            if isinstance(sc_cand, StandardScaler):
                lr_scaler = sc_cand

    batch_n = max(prof_cfg.protocol.batch_sizes)

    # B.1 Descriptor Stages + cProfile + Scalability Sweep
    desc_report = profile_descriptor_stages(
        images_u8=cpu_bundle.images_u8,
        dataset_tag=tag,
        grid_levels=desc_cfg.grid_levels,
        bins=desc_cfg.bins,
        asymmetry_scales=desc_cfg.asymmetry_scales,
        asymmetry_kinds=desc_cfg.asymmetry_kinds,
        threads=primary_threads,
        batch_n_size=batch_n,
        warmup_iters=prof_cfg.protocol.warmup_iters,
        timed_iters=prof_cfg.protocol.timed_iters,
        batch_timed_iters=prof_cfg.protocol.batch_timed_iters,
        disable_gc_flag=prof_cfg.protocol.disable_gc,
        bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
        confidence_level=prof_cfg.statistics.confidence_level,
        cprofile_top_n=desc_cfg.cprofile_top_n,
        seed=prof_cfg.seed,
        scaler=lr_scaler,
    )
    write_descriptor_stage_artifacts(out_dir, desc_report, precision=prof_cfg.output.float_precision)

    desc_scalability = profile_descriptor_scalability(
        images_u8=cpu_bundle.images_u8,
        base_image_size=desc_cfg.image_size,
        base_bins=desc_cfg.bins,
        base_grid_levels=desc_cfg.grid_levels,
        asymmetry_kinds=desc_cfg.asymmetry_kinds,
        image_sizes=desc_cfg.scalability_image_sizes,
        bins_list=desc_cfg.scalability_bins,
        grid_levels_list=desc_cfg.scalability_grid_levels,
        threads=primary_threads,
        warmup_iters=min(15, prof_cfg.protocol.warmup_iters),
        timed_iters=min(100, prof_cfg.protocol.timed_iters),
        disable_gc_flag=prof_cfg.protocol.disable_gc,
        bootstrap_resamples=min(500, prof_cfg.statistics.bootstrap_resamples),
        confidence_level=prof_cfg.statistics.confidence_level,
        seed=prof_cfg.seed,
    )
    write_descriptor_scalability_csv(
        output_path=out_dir / "descriptor_scalability.csv",
        records=desc_scalability,
        precision=prof_cfg.output.float_precision,
    )

    # B.3 Classical ML Profiling (batch=1 vs batch=32, scaler + dispatch overhead, thread scaling)
    cls_records: list[ClassifierProfileRecord] = []
    cls_scaling_records: list[ClassifierProfileRecord] = []
    n_train_ml = int(cpu_bundle.cls_partition.X_train.shape[0])

    for cls_name in classifiers:
        ml_model = cpu_bundle.fitted_classifiers[cls_name][0]
        for b_size in prof_cfg.protocol.batch_sizes:
            n_it = (
                prof_cfg.protocol.timed_iters
                if b_size == 1
                else prof_cfg.protocol.batch_timed_iters
            )
            cls_records.append(
                profile_classifier(
                    model=ml_model,
                    classifier_name=cls_name,
                    X_test=cpu_bundle.x_ml_real,
                    n_train=n_train_ml,
                    threads=primary_threads,
                    batch_size=b_size,
                    warmup_iters=prof_cfg.protocol.warmup_iters,
                    timed_iters=n_it,
                    max_real_samples=int(cpu_bundle.x_ml_real.shape[0]),
                    disable_gc=prof_cfg.protocol.disable_gc,
                    bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
                    confidence_level=prof_cfg.statistics.confidence_level,
                    seed=prof_cfg.seed,
                )
            )
        cls_scaling_records.extend(
            profile_classifier_thread_scaling(
                model=ml_model,
                classifier_name=cls_name,
                X_test=cpu_bundle.x_ml_real,
                n_train=n_train_ml,
                threads_list=threads_list,
                batch_size=primary_batch_size,
                warmup_iters=prof_cfg.protocol.warmup_iters,
                timed_iters=prof_cfg.protocol.timed_iters,
                max_real_samples=int(cpu_bundle.x_ml_real.shape[0]),
                disable_gc=prof_cfg.protocol.disable_gc,
                bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
                confidence_level=prof_cfg.statistics.confidence_level,
                seed=prof_cfg.seed,
            )
        )

    write_classifier_times_csv(
        output_path=out_dir / "classifier_times.csv",
        records=cls_records,
        precision=prof_cfg.output.float_precision,
    )
    write_classifier_times_csv(
        output_path=out_dir / "classifier_thread_scaling.csv",
        records=cls_scaling_records,
        precision=prof_cfg.output.float_precision,
    )

    # B.4 Quality vs Cost Ablation (x85 vs x9 vs x94 -> dF1 / dms)
    ablation_rows = _run_ablation_quality_vs_cost(
        prof_cfg=prof_cfg,
        cls_cfg=cls_cfg,
        cls_partition=cpu_bundle.cls_partition,
        images_u8=cpu_bundle.images_u8,
        classifiers=classifiers,
        primary_threads=primary_threads,
    )
    write_ablation_cost_csv(
        output_path=out_dir / "ablation_quality_vs_cost.csv",
        rows=ablation_rows,
        precision=prof_cfg.output.float_precision,
    )

    # B.2 CNN Layer Profiling, ATen Ops, Chrome Trace, Cold Start & Thread Scaling
    layer_reports: list[CnnLayerProfileReport] = []
    cnn_scaling_records: list[CnnThreadScalingRecord] = []

    for m_name in models:
        cnn_model, ckpt_path, _, _ = cpu_bundle.loaded_cnns[m_name]
        cold_start = None
        if prof_cfg.cnn.include_checkpoint_load_and_cold_start:
            cold_start = measure_checkpoint_cold_start(
                model_factory=create_model,
                model_name=m_name,
                checkpoint_path=ckpt_path,
                sample_input=cpu_bundle.x_real_cnn,
                threads=primary_threads,
                disable_gc=prof_cfg.protocol.disable_gc,
            )

        l_rep = profile_cnn_layers(
            model=cnn_model,
            model_name=m_name,
            inputs=cpu_bundle.x_real_cnn,
            threads=primary_threads,
            batch_size=primary_batch_size,
            warmup_iters=prof_cfg.protocol.warmup_iters,
            timed_iters=min(100, prof_cfg.protocol.timed_iters),
            torch_prof_warmup=prof_cfg.cnn.torch_profiler_warmup,
            torch_prof_active=prof_cfg.cnn.torch_profiler_active,
            hook_overhead_warn_pct=prof_cfg.cnn.hook_overhead_warn_pct,
            disable_gc=prof_cfg.protocol.disable_gc,
            bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
            confidence_level=prof_cfg.statistics.confidence_level,
            seed=prof_cfg.seed,
            chrome_trace_path=out_dir / f"cnn_chrome_trace_{m_name}.json",
            cold_start=cold_start,
        )
        layer_reports.append(l_rep)

        cnn_scaling_records.extend(
            profile_cnn_thread_scaling(
                model=cnn_model,
                model_name=m_name,
                inputs=cpu_bundle.x_real_cnn,
                threads_list=threads_list,
                batch_size=primary_batch_size,
                warmup_iters=prof_cfg.protocol.warmup_iters,
                timed_iters=prof_cfg.protocol.timed_iters,
                disable_gc=prof_cfg.protocol.disable_gc,
                bootstrap_resamples=prof_cfg.statistics.bootstrap_resamples,
                confidence_level=prof_cfg.statistics.confidence_level,
                seed=prof_cfg.seed,
            )
        )

    write_cnn_layer_artifacts(out_dir, layer_reports, precision=prof_cfg.output.float_precision)
    write_cnn_thread_scaling_csv(
        output_path=out_dir / "cnn_thread_scaling.csv",
        records=cnn_scaling_records,
        precision=prof_cfg.output.float_precision,
    )

    # B.4 Non-parametric Mann-Whitney U Statistical Tests
    stat_tests: list[StatisticalComparisonResult] = []
    for ml_key in prof_cfg.statistics.ml_baseline_models:
        if ml_key not in cpu_bundle.raw_batch1_primary_latencies_ms:
            continue
        for cnn_key in prof_cfg.statistics.cnn_reference_models:
            if cnn_key not in cpu_bundle.raw_batch1_primary_latencies_ms:
                continue
            stat_tests.append(
                compare_latency_distributions(
                    samples_a_ms=cpu_bundle.raw_batch1_primary_latencies_ms[ml_key],
                    samples_b_ms=cpu_bundle.raw_batch1_primary_latencies_ms[cnn_key],
                    group_a=f"{ml_key} (x94)",
                    group_b=cnn_key,
                    alpha=prof_cfg.statistics.mann_whitney_alpha,
                )
            )
    write_statistical_tests_csv(
        output_path=out_dir / "statistical_tests.csv",
        comparisons=stat_tests,
        precision=prof_cfg.output.float_precision,
    )

    # B.5 Figures + Markdown Report
    generate_profiling_figures(
        output_dir=out_dir,
        dataset_label=tag,
        desc_report=desc_report,
        desc_scalability=desc_scalability,
        layer_reports=layer_reports,
        cnn_scaling=cnn_scaling_records,
        cls_scaling=cls_scaling_records,
        cpu_comparison_rows=cpu_bundle.cpu_rows,
        primary_threads=primary_threads,
        primary_batch_size=primary_batch_size,
        throughput_batch_size=batch_n,
        dpi=prof_cfg.output.dpi,
    )

    write_profiling_readme(
        output_path=out_dir / "README.md",
        dataset_label=tag,
        env=cpu_bundle.environment,
        desc_report=desc_report,
        cpu_rows=cpu_bundle.cpu_rows,
        cuda_rows=cpu_bundle.cuda_rows,
        stat_tests=stat_tests,
        ablation_rows=ablation_rows,
        layer_reports=layer_reports,
        primary_threads=primary_threads,
        primary_batch_size=primary_batch_size,
    )

    logger.info("Completed full Phase B profiling '%s' -> %s", tag, out_dir)
    return out_dir


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """CLI entry point for Phase B scientific CPU pipeline profiling."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
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

        short_mode = bool(
            args.short_descriptor_check
            or (
                prof_cfg.inverted
                and not prof_cfg.output.run_full_profiling_on_inverted
                and not args.full_inverted
            )
        )

        if not short_mode:
            validate_required_inputs_upfront(
                prof_cfg=prof_cfg,
                cls_cfg=cls_cfg,
                cnn_cfg=cnn_cfg,
                target_objects=target_objects,
                models=models,
            )
        else:
            if not cls_cfg.effective_input_csv.is_file():
                raise FileNotFoundError(
                    f"Missing x94 descriptor CSV: {cls_cfg.effective_input_csv}"
                )
            if not cnn_cfg.effective_metadata_csv.is_file():
                raise FileNotFoundError(
                    f"Missing CNN metadata CSV: {cnn_cfg.effective_metadata_csv}"
                )
            if not cnn_cfg.effective_preprocessed_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing preprocessed PNG directory: {cnn_cfg.effective_preprocessed_dir}"
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
        primary_batch_size = prof_cfg.protocol.primary_batch_size
        include_cuda_ref = prof_cfg.protocol.include_cuda_reference and not args.no_cuda_reference
        prof_base_dir = (
            (prof_cfg.project_root / args.output_dir).resolve()
            if args.output_dir is not None
            else prof_cfg.output.profiling_dir
        )

        for obj in target_objects:
            run_profile_target(
                prof_cfg=prof_cfg,
                cls_cfg=cls_cfg,
                cnn_cfg=cnn_cfg,
                object_name=obj,
                classifiers=classifiers,
                models=models,
                threads_list=threads_list,
                primary_threads=primary_threads,
                primary_batch_size=primary_batch_size,
                short_mode=short_mode,
                include_cuda_reference=include_cuda_ref,
                profiling_base_dir=prof_base_dir,
                dry_run=args.dry_run,
            )

        return 0

    except (OSError, ValueError, KeyError, RuntimeError, FileExistsError) as exc:
        logger.error("Pipeline profiling error: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
