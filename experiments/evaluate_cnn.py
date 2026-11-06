"""CLI entry point for lightweight CNN training, evaluation, and latency benchmarking.

Trains and evaluates lightweight CNN architectures on 128x128 grayscale images
for KolektorSDD2 (official presplit) and MVTec-AD (stratified split per object).
Matches the protocol and split indices of task_06A (descriptor_classifier.py)
for fair and rigorous scientific comparison.

Manual execution commands (PowerShell from repository 'code/' directory):
    # 1. Dry-run validation (KolektorSDD2 and MVTec-AD):
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --dry-run
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object bottle --dry-run

    # 2. KolektorSDD2 single model evaluation (normal & inverted):
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model ds_cnn
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model ds_cnn --inverted

    # 3. MVTec-AD single object evaluation:
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object bottle --model ds_cnn
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object tile --model tiny_cnn

    # 4. MVTec-AD 15-objects sweep:
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --all-objects --model ds_cnn
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import logging
from pathlib import Path
import sys
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import configure_logging, logger
from src.models.cnn import (
    BenchmarkRecord,
    CnnConfig,
    CnnDatasetPartition,
    create_model,
    evaluate_cnn,
    load_cnn_config,
    load_cnn_partition,
    model_names,
    plot_confusion_and_curves,
    plot_training_curves,
    run_benchmark,
    save_checkpoint,
    save_metrics_summary,
    save_training_history,
    train_model,
)
from src.models.cnn.architectures.common import seed_everything, select_device


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for CNN evaluation."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.evaluate_cnn",
        description="Train and evaluate lightweight CNNs on grayscale anomaly detection benchmarks.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cnn/kolektorsdd2.toml"),
        help="Path to CNN TOML configuration (default: configs/cnn/kolektorsdd2.toml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture name ('ultra_tiny', 'ds_cnn', 'tiny_cnn', 'mobilenet_like', "
        "'tiny_resnet', 'small_cnn', 'torchvision_mobilenet', or 'all'). Defaults to TOML setting.",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        default=False,
        help="Evaluate on inverted grayscale images (255 - x).",
    )
    parser.add_argument(
        "--object",
        dest="object_name",
        type=str,
        default=None,
        help="Object category for MVTec-AD (e.g. bottle, metal_nut, screw, tile).",
    )
    parser.add_argument(
        "--all-objects",
        action="store_true",
        default=False,
        help="Run evaluation across all subdatasets configured in base.toml for MVTec-AD.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds to evaluate (default: read from TOML).",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["val_f1", "fixed_05", "val_bal_acc"],
        default=None,
        help="Strategy for decision threshold calibration (default: from TOML or 'val_f1').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Explicit positive decision threshold override (e.g. 0.50, 0.70).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Device override ('auto', 'cpu', 'cuda'). Defaults to TOML setting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files in results directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate configuration, image dimensions, split counts, and model forward pass without training.",
    )
    return parser


def _run_single_evaluation(
    cfg: CnnConfig,
    model_name: str,
    object_name: str | None,
    seed: int,
    device_name: str,
    dry_run: bool,
    threshold_mode: str | None = None,
    fixed_threshold: float | None = None,
) -> dict[str, object]:
    """Execute training, evaluation, and benchmarking for a single model and seed."""
    device = select_device(device_name)
    target_object = object_name if object_name is not None else cfg.default_object

    seed_everything(seed)

    # Load partition directly from preprocessed PNG files
    partition = load_cnn_partition(cfg, object_name=target_object, seed=seed)

    # Instantiate model
    model = create_model(model_name, num_classes=2).to(device)

    obj_tag = f" [object={target_object}]" if target_object else ""
    inv_tag = " (inverted)" if cfg.inverted else ""

    if dry_run:
        # Validate forward pass shape and finiteness
        dummy = torch.randn(2, 1, cfg.data.image_size, cfg.data.image_size, device=device)
        model.eval()
        with torch.inference_mode():
            out = model(dummy)
        if tuple(out.shape) != (2, 2) or not torch.isfinite(out).all():
            raise RuntimeError(f"Model {model_name} dry-run output shape or values invalid: {out.shape}")

        logger.info(
            "Dry-run OK: %s%s%s | model=%s | split_mode=%s | train=%s | val=%s | test=%s",
            cfg.dataset_name,
            inv_tag,
            obj_tag,
            model_name,
            partition.split_mode,
            partition.train_counts,
            partition.val_counts,
            partition.test_counts,
        )
        return {
            "model_name": model_name,
            "object_name": target_object,
            "seed": seed,
            "dry_run": True,
        }

    out_dir = cfg.effective_results_dir(target_object, model_name)
    if (out_dir / "metrics_summary.json").is_file() and not cfg.overwrite:
        logger.info(
            "Skipping %s%s%s [%s]: results already exist in %s (use --overwrite to re-run)",
            cfg.dataset_name,
            inv_tag,
            obj_tag,
            model_name.upper(),
            out_dir,
        )
        return {"model_name": model_name, "object_name": target_object, "skipped": True}

    # Train model
    train_res = train_model(
        model=model,
        partition=partition,
        cfg=cfg,
        seed=seed,
        device=device,
    )

    # Evaluate model on test set with validation threshold calibration
    active_th_mode = threshold_mode or cfg.training.threshold_mode
    metrics, y_pred, y_prob, th_info = evaluate_cnn(
        model=model,
        partition=partition,
        cfg=cfg,
        model_name=model_name,
        train_elapsed_s=train_res.train_elapsed_s,
        device=device,
        batch_size=cfg.training.batch_size,
        threshold_mode=active_th_mode,
        fixed_threshold=fixed_threshold,
    )

    # Run standalone latency benchmarks
    benchmarks: list[BenchmarkRecord] = []
    for b_dev_name in cfg.benchmark.devices:
        try:
            b_dev = select_device(b_dev_name)
        except RuntimeError:
            continue
        for b_size in cfg.benchmark.batch_sizes:
            rec = run_benchmark(
                model=model,
                model_name=model_name,
                cfg=cfg,
                device=b_dev,
                batch_size=b_size,
            )
            benchmarks.append(rec)

    # Export artifacts
    save_metrics_summary(
        output_dir=out_dir,
        cfg=cfg,
        model_name=model_name,
        metrics=metrics,
        train_res=train_res,
        benchmarks=benchmarks,
        partition=partition,
        seed=seed,
        th_info=th_info,
    )
    save_training_history(out_dir, train_res)
    if cfg.output.save_checkpoints:
        save_checkpoint(out_dir, train_res.best_state_dict)

    plot_confusion_and_curves(
        output_dir=out_dir,
        metrics=metrics,
        model_name=model_name,
        dataset_name=cfg.dataset_name,
        object_name=target_object,
        th_info=th_info,
        dpi=cfg.output.dpi,
    )
    plot_training_curves(
        output_dir=out_dir,
        train_res=train_res,
        model_name=model_name,
        dpi=cfg.output.dpi,
    )

    logger.info(
        "Completed %s%s%s [%s]: Acc=%.4f | BalAcc=%.4f | F1=%.4f (th=%.2f, F1@0.5=%.4f) | "
        "ROC-AUC=%.4f | PR-AUC=%.4f | T_train=%.4fs, T_inf=%.4fs (%.3f ms/sample) -> %s",
        cfg.dataset_name,
        inv_tag,
        obj_tag,
        model_name.upper(),
        metrics.accuracy,
        metrics.balanced_accuracy,
        metrics.f1,
        th_info.selected_threshold,
        th_info.f1_at_05,
        metrics.roc_auc,
        metrics.pr_auc,
        metrics.train_elapsed_s,
        metrics.inference_elapsed_s,
        metrics.inference_ms_per_sample,
        out_dir,
    )

    return {
        "model_name": model_name,
        "object_name": target_object,
        "seed": seed,
        "metrics": metrics,
        "train_res": train_res,
        "benchmarks": benchmarks,
        "th_info": th_info,
        "out_dir": out_dir,
    }


def _run_all_objects_sweep(
    cfg: CnnConfig,
    model_name: str,
    device_name: str,
    seeds: Sequence[int],
    dry_run: bool,
    threshold_mode: str | None = None,
    fixed_threshold: float | None = None,
) -> None:
    """Execute evaluation across all configured MVTec-AD subdatasets."""
    if not cfg.subdatasets:
        raise ValueError(f"--all-objects requires subdatasets in base.toml for '{cfg.dataset_key}'")

    summary_rows: list[dict[str, object]] = []

    for obj in cfg.subdatasets:
        for seed in seeds:
            res = _run_single_evaluation(
                cfg=cfg,
                model_name=model_name,
                object_name=obj,
                seed=seed,
                device_name=device_name,
                dry_run=dry_run,
                threshold_mode=threshold_mode,
                fixed_threshold=fixed_threshold,
            )
            if dry_run or res.get("skipped"):
                continue

            metrics = res["metrics"]
            train_res = res["train_res"]
            benchmarks = res["benchmarks"]
            th_info = res["th_info"]
            b_sample = benchmarks[0] if benchmarks else None

            summary_rows.append(
                {
                    "object_name": obj,
                    "model_name": model_name,
                    "seed": seed,
                    "inverted": cfg.inverted,
                    "threshold_used": th_info.selected_threshold,
                    "accuracy": metrics.accuracy,
                    "balanced_accuracy": metrics.balanced_accuracy,
                    "f1": metrics.f1,
                    "f1_at_05": th_info.f1_at_05,
                    "roc_auc": metrics.roc_auc,
                    "pr_auc": metrics.pr_auc,
                    "train_elapsed_s": train_res.train_elapsed_s,
                    "inference_ms_per_sample": metrics.inference_ms_per_sample,
                    "params": b_sample.parameters if b_sample else 0,
                    "flops": b_sample.flops if b_sample else 0,
                }
            )

    if summary_rows and not dry_run:
        summary_dir = cfg.results_dir / cfg.effective_dataset_tag
        summary_dir.mkdir(parents=True, exist_ok=True)
        csv_path = summary_dir / f"mvtec_all_objects_{model_name.lower()}_summary.csv"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        logger.info("Saved multi-object sweep summary: %s", csv_path)


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """Execute lightweight CNN evaluation pipeline CLI.

    Args:
        argv: Command-line arguments.
        project_root: Repository root directory.

    Returns:
        Exit code (0 on success).
    """
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_cnn_config(args.config, project_root)
        if args.inverted:
            cfg = replace(cfg, inverted=True)
        if args.overwrite:
            cfg = replace(cfg, overwrite=True)

        device_name = args.device or cfg.training.device
        seeds = tuple(args.seeds) if args.seeds else (cfg.seeds if cfg.seeds else (cfg.seed,))

        active_models: list[str]
        req_model = (args.model or cfg.default_model).lower()
        if req_model == "all":
            active_models = list(cfg.models)
        else:
            if req_model not in model_names():
                raise ValueError(
                    f"Requested model '{req_model}' not found in registry. Available: {model_names()}"
                )
            active_models = [req_model]

        for m_name in active_models:
            if args.all_objects:
                _run_all_objects_sweep(
                    cfg=cfg,
                    model_name=m_name,
                    device_name=device_name,
                    seeds=seeds,
                    dry_run=args.dry_run,
                    threshold_mode=args.threshold_mode,
                    fixed_threshold=args.threshold,
                )
            else:
                for seed in seeds:
                    _run_single_evaluation(
                        cfg=cfg,
                        model_name=m_name,
                        object_name=args.object_name,
                        seed=seed,
                        device_name=device_name,
                        dry_run=args.dry_run,
                        threshold_mode=args.threshold_mode,
                        fixed_threshold=args.threshold,
                    )
        return 0

    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        logger.error("CNN evaluation pipeline error: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
