"""CLI entry point for x94 descriptor classification, subset ablation, and explainability.

Trains and evaluates lightweight supervised classifiers (Random Forest or k-NN) on
the 94-feature multiscale entropy and spatial asymmetry descriptors (x94 = [x85, x9])
for KolektorSDD2 (official train/test presplit) and MVTec-AD (in-memory stratified
split per object after excluding ground_truth masks). Automatically produces:
  - metrics_summary.json
  - ablation_table.csv (x85_only vs x9_only vs x94_full and per-scale E_1x1..E_8x8)
  - feature_importance.csv & scale_and_family_breakdown.csv
  - confusion_matrix_and_curves.png
  - descriptor_ablation_x85_x9_x94.png
  - explainability_importance_breakdown.png
  - spatial_importance_heatmaps.png

Manual execution commands (PowerShell from repository 'code/' directory):
    # 1. Dry-run validation (KolektorSDD2 and MVTec-AD):
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --dry-run
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --dry-run

    # 2. KolektorSDD2 (Random Forest & k-NN, standard & inverted):
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier rf
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier knn
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier rf --inverted

    # 3. MVTec-AD single object (e.g., bottle, metal_nut, screw, tile):
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier rf
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object metal_nut --classifier rf
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier knn --inverted

    # 4. MVTec-AD all 15 objects batch evaluation + summary chart:
    .\\.venv\\Scripts\\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier rf
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import configure_logging, logger
from src.models.classification_plots import plot_all_objects_summary
from src.models.descriptor_classifier import (
    ClassificationConfig,
    EvaluationRunSummary,
    load_classification_config,
    run_evaluation_pipeline,
)


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for descriptor evaluation."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.evaluate_descriptors",
        description="Evaluate x94 descriptors with Random Forest or k-NN and generate explainability plots.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/classification/kolektorsdd2.toml"),
        help="Path to classification task TOML configuration (default: configs/classification/kolektorsdd2.toml).",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["rf", "knn", "dt", "lr", "lr_dot", "all"],
        default=None,
        help="Classifier to evaluate: 'lr', 'lr_dot', 'rf', 'knn', 'dt', or 'all' to evaluate all available classifiers in sequence. Defaults to TOML setting.",
    )
    parser.add_argument(
        "--linear-mode",
        type=str,
        choices=["sklearn", "dot_core"],
        default=None,
        help="Override [classification.models.lr].linear_mode when evaluating 'lr' ('sklearn' or 'dot_core').",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        default=False,
        help="Evaluate on the inverted dataset variant (255 - x).",
    )
    parser.add_argument(
        "--object",
        dest="object_name",
        type=str,
        default=None,
        help="Object category name for MVTec-AD (e.g., bottle, cable, metal_nut, tile).",
    )
    parser.add_argument(
        "--all-objects",
        action="store_true",
        default=False,
        help="Run across all subdatasets configured for MVTec-AD and generate an aggregate summary.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing output files in the results directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate configuration, CSV schema, and train/test class counts without training or writing files.",
    )
    return parser


def _log_summary(summary: EvaluationRunSummary) -> None:
    """Log structured summary of a single evaluation run."""
    obj_tag = f" [object={summary.object_name}]" if summary.object_name else ""
    inv_tag = " (inverted)" if summary.inverted else ""
    if summary.dry_run:
        logger.info(
            "Dry-run OK: %s%s%s | split_mode=%s | train=%s | test=%s",
            summary.dataset_name,
            inv_tag,
            obj_tag,
            summary.split_mode,
            summary.train_counts,
            summary.test_counts,
        )
        return

    assert summary.active_metrics is not None
    assert summary.explainability is not None
    m = summary.active_metrics
    by_sub = {ab.subset_name: ab for ab in summary.ablation_metrics}
    df1 = (
        by_sub["x94_full"].f1 - by_sub["x85_only"].f1
        if "x94_full" in by_sub and "x85_only" in by_sub
        else 0.0
    )
    fam_map = {g.group_name: g for g in summary.explainability.family_breakdown}
    x85_pct = fam_map["x85_entropy"].share_pct if "x85_entropy" in fam_map else 0.0
    x9_pct = fam_map["x9_asymmetry"].share_pct if "x9_asymmetry" in fam_map else 0.0
    top5 = [
        f"{summary.explainability.feature_names[i]} ({summary.explainability.importances[i]*100:.1f}%)"
        for i in summary.explainability.top_k_indices[:5]
    ]

    logger.info(
        "Completed %s%s%s [%s]: Acc=%.4f | BalAcc=%.4f | F1=%.4f | ROC-AUC=%.4f | PR-AUC=%.4f | "
        "dF1(x94-x85)=%+.4f | Share(x85=%.1f%%, x9=%.1f%%) | Top-5=%s | "
        "T_train=%.4fs, T_inf=%.4fs (%.3f ms/sample) -> %s",
        summary.dataset_name,
        inv_tag,
        obj_tag,
        summary.classifier_name.upper(),
        m.accuracy,
        m.balanced_accuracy,
        m.f1,
        m.roc_auc,
        m.pr_auc,
        df1,
        x85_pct,
        x9_pct,
        ", ".join(top5),
        m.train_elapsed_s,
        m.inference_elapsed_s,
        m.inference_ms_per_sample,
        summary.output_dir,
    )


def _run_all_objects(
    cfg: ClassificationConfig,
    classifier_name: str | None,
    dry_run: bool,
) -> None:
    """Execute pipeline across all configured MVTec-AD subdatasets and export aggregate report."""
    if not cfg.subdatasets:
        raise ValueError(
            f"--all-objects requires subdatasets in base.toml for dataset '{cfg.dataset_key}'"
        )

    cls_name = (classifier_name or cfg.default_classifier).lower()
    summary_rows: list[dict[str, object]] = []

    for obj_name in cfg.subdatasets:
        summary = run_evaluation_pipeline(
            cfg,
            classifier_name=cls_name,
            object_name=obj_name,
            dry_run=dry_run,
        )
        _log_summary(summary)
        if dry_run:
            continue

        assert summary.active_metrics is not None
        assert summary.explainability is not None
        by_sub = {m.subset_name: m for m in summary.ablation_metrics}
        fam_map = {g.group_name: g for g in summary.explainability.family_breakdown}
        scale_candidates = [
            m for m in summary.ablation_metrics if m.subset_name.startswith("E_")
        ]
        best_scale = (
            max(scale_candidates, key=lambda m: (m.f1, m.roc_auc)).subset_name
            if scale_candidates
            else ""
        )
        top1_idx = summary.explainability.top_k_indices[0]
        top1_name = summary.explainability.feature_names[top1_idx]

        f1_x85 = by_sub["x85_only"].f1 if "x85_only" in by_sub else 0.0
        f1_x9 = by_sub["x9_only"].f1 if "x9_only" in by_sub else 0.0
        f1_x94 = (
            by_sub["x94_full"].f1 if "x94_full" in by_sub else summary.active_metrics.f1
        )

        summary_rows.append(
            {
                "object_name": obj_name,
                "classifier": cls_name,
                "inverted": cfg.inverted,
                "train_good": summary.train_counts.get(cfg.negative_label, 0),
                "train_defect": summary.train_counts.get(cfg.positive_label, 0),
                "test_good": summary.test_counts.get(cfg.negative_label, 0),
                "test_defect": summary.test_counts.get(cfg.positive_label, 0),
                "accuracy_x94_full": summary.active_metrics.accuracy,
                "balanced_accuracy_x94_full": summary.active_metrics.balanced_accuracy,
                "f1_x85_only": f1_x85,
                "f1_x9_only": f1_x9,
                "f1_x94_full": f1_x94,
                "delta_f1_x94_minus_x85": f1_x94 - f1_x85,
                "roc_auc_x94_full": summary.active_metrics.roc_auc,
                "pr_auc_x94_full": summary.active_metrics.pr_auc,
                "best_single_entropy_scale": best_scale,
                "x85_share_pct": (
                    fam_map["x85_entropy"].share_pct
                    if "x85_entropy" in fam_map
                    else 0.0
                ),
                "x9_share_pct": (
                    fam_map["x9_asymmetry"].share_pct
                    if "x9_asymmetry" in fam_map
                    else 0.0
                ),
                "top1_feature": top1_name,
                "train_elapsed_s": summary.active_metrics.train_elapsed_s,
                "inference_ms_per_sample": summary.active_metrics.inference_ms_per_sample,
            }
        )

    if dry_run or not summary_rows:
        return

    ds_root = cfg.results_dir / cfg.effective_dataset_tag
    ds_root.mkdir(parents=True, exist_ok=True)
    csv_out = ds_root / f"mvtec_all_objects_{cls_name}_summary.csv"
    p_fmt = f"%.{cfg.float_precision}f"

    fieldnames = list(summary_rows[0].keys())
    with csv_out.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            formatted_row = {
                k: (p_fmt % v if isinstance(v, float) else v) for k, v in row.items()
            }
            writer.writerow(formatted_row)

    title_prefix = f"{cfg.dataset_name}{' (Inverted)' if cfg.inverted else ''} [{cls_name.upper()}]"
    png_out = ds_root / f"mvtec_all_objects_{cls_name}_comparison.png"
    plot_all_objects_summary(
        summary_rows, title_prefix=title_prefix, out_path=png_out, dpi=cfg.dpi
    )
    logger.info("Saved aggregate multi-object report to %s and %s", csv_out, png_out)


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """Execute descriptor classification and explainability CLI pipeline.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).
        project_root: Repository root directory for relative path resolution.

    Returns:
        Exit code (0 on success).
    """
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_classification_config(args.config, project_root)
        if args.inverted:
            cfg = replace(cfg, inverted=True)
        if args.overwrite:
            cfg = replace(cfg, overwrite=True)
        if args.linear_mode is not None:
            cfg = replace(
                cfg,
                lr_params=replace(cfg.lr_params, linear_mode=args.linear_mode),
            )

        active_classifiers: list[str]
        req_cls = (args.classifier or cfg.default_classifier).lower()
        if req_cls == "all":
            active_classifiers = list(cfg.available_classifiers)
        else:
            active_classifiers = [req_cls]

        for cls_name in active_classifiers:
            if args.all_objects:
                _run_all_objects(cfg, classifier_name=cls_name, dry_run=args.dry_run)
            else:
                summary = run_evaluation_pipeline(
                    cfg,
                    classifier_name=cls_name,
                    object_name=args.object_name,
                    dry_run=args.dry_run,
                )
                _log_summary(summary)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        logger.error("Classification pipeline error: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
