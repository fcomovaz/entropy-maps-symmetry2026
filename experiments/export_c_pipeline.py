"""CLI to export descriptor LUTs, classical models, CNNs, and test vectors to C99 headers/sources."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from configs.logging import configure_logging, logger
from src.c_export.build_c import build_desktop_runner
from src.c_export.cnn_codegen import ExportedCnnModel, export_cnn_models, write_registry_and_reports
from src.c_export.config import CImpConfig, load_c_imp_config
from src.c_export.descriptor_reference import extract_x94_f64_oracle, extract_x94_q15_batch
from src.c_export.linear_export import export_classical_models
from src.c_export.luts import emit_config_and_lut_headers
from src.c_export.test_vectors import _partition_to_u8, _select_indices, export_test_vectors
from src.c_export.visualize_plots import generate_c_pipeline_visualizations
from src.models.cnn.data import load_cnn_partition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALL_STAGES: tuple[str, ...] = ("luts", "descriptor", "classifier", "cnn", "vectors")


def _validate_dry_run(cfg: CImpConfig) -> None:
    """Validate referenced dataset images and checkpoints without writing generated files."""
    preproc = cfg.cnn_cfg.effective_preprocessed_dir
    if not preproc.is_dir():
        raise FileNotFoundError(f"Preprocessed image directory not found: {preproc}")
    for mname in cfg.cnn.models:
        rdir = cfg.cnn_cfg.effective_results_dir(cfg.object_name, mname)
        ckpt = (
            rdir / "checkpoint.pt"
            if (rdir / "checkpoint.pt").is_file()
            else rdir / f"{cfg.cnn.checkpoint_variant}_model.pt"
        )
        summary = rdir / "metrics_summary.json"
        if not ckpt.is_file():
            raise FileNotFoundError(f"CNN checkpoint missing for '{mname}': {ckpt}")
        if not summary.is_file():
            raise FileNotFoundError(f"CNN metrics_summary.json missing for '{mname}': {summary}")
    logger.info(
        "Dry-run OK: target=%s | image_size=%d | shannon_mode=%s | tabular=%s | cnn=%s",
        cfg.effective_target_tag,
        cfg.descriptor_spec.image_size,
        cfg.descriptor.shannon_lut_mode,
        cfg.classifier.models,
        cfg.cnn.models,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for C99 export pipeline."""
    parser = argparse.ArgumentParser(
        description="Export deterministic C99 LUTs, descriptors, linear/MLP models, CNNs, and golden vectors."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Relative path to C export TOML config (e.g. configs/c_imp/kolektorsdd2.toml).",
    )
    parser.add_argument(
        "--object",
        "--object-name",
        dest="object_name",
        type=str,
        default=None,
        help="Optional MVTec-AD subdataset object override (e.g. bottle).",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        default=False,
        help="Use inverted image orientation variant.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional override for number of exported test-vector images.",
    )
    parser.add_argument(
        "--cnn-models",
        nargs="*",
        default=None,
        help="Optional subset of CNN names to export (defaults to TOML [c_imp.cnn].models).",
    )
    parser.add_argument(
        "--classifier-models",
        nargs="*",
        default=None,
        help="Optional subset of tabular classifiers to export (defaults to TOML [c_imp.classifier].models).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ALL_STAGES,
        default=None,
        help="Optional subset of export stages to run.",
    )
    parser.add_argument(
        "--skip-cnn",
        action="store_true",
        help="Skip CNN graph extraction and code generation.",
    )
    parser.add_argument(
        "--skip-tabular",
        action="store_true",
        help="Skip classical descriptor model export.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, paths, and checkpoints without writing generated files.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also compile desktop C99 binary after code generation.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate summary CSV/Markdown and matplotlib diagnostic figures after export.",
    )
    return parser


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """Run end-to-end C99 pipeline export CLI."""
    configure_logging()
    args = build_parser().parse_args(argv)

    try:
        cfg = load_c_imp_config(
            config_path=args.config,
            project_root=project_root,
            object_name=args.object_name,
            inverted=True if args.inverted else None,
        )
        if args.cnn_models is not None:
            cfg = replace(cfg, cnn=replace(cfg.cnn, models=tuple(args.cnn_models)))
        if args.classifier_models is not None:
            cfg = replace(
                cfg, classifier=replace(cfg.classifier, models=tuple(args.classifier_models))
            )

        if args.dry_run:
            _validate_dry_run(cfg)
            return 0

        stages = set(args.only) if args.only is not None else set(ALL_STAGES)
        if args.skip_cnn:
            stages.discard("cnn")
        if args.skip_tabular:
            stages.discard("classifier")

        logger.info("Exporting C99 pipeline for target '%s'...", cfg.effective_target_tag)
        luts_info: dict[str, object] = {}
        if "luts" in stages or "descriptor" in stages:
            luts_info = emit_config_and_lut_headers(cfg)

        need_partition = bool({"vectors", "classifier", "cnn"} & stages)
        vectors_manifest_str = ""
        n_test_vectors = 0
        exported_tabular: dict[str, object] = {}
        exported_cnns: list[ExportedCnnModel] = []

        if need_partition:
            partition = load_cnn_partition(cfg.cnn_cfg, object_name=cfg.object_name, seed=cfg.seed)
            if "vectors" in stages:
                vectors = export_test_vectors(
                    cfg, partition=partition, max_images_override=args.max_images
                )
                vectors_manifest_str = str(vectors.manifest_path)
                file_names = vectors.file_names
                labels = vectors.labels
                images_u8 = vectors.images_u8
                images_f32 = vectors.images_f32
                x94_f64 = vectors.x94_f64
                x94_q15 = vectors.x94_q15
            else:
                max_imgs = (
                    cfg.test_vectors.max_images
                    if args.max_images is None
                    else args.max_images
                )
                y_full = partition.y_test.cpu().numpy().astype(np.int64)
                sel = _select_indices(y_full, max_imgs, seed=cfg.seed)
                images_f32 = partition.X_test[sel].cpu().numpy().astype(np.float32)
                labels = y_full[sel]
                file_names = tuple(partition.test_file_names[int(i)] for i in sel)
                images_u8 = _partition_to_u8(
                    images_f32, cfg.cnn_cfg.data.normalize_mean, cfg.cnn_cfg.data.normalize_std
                )
                x94_f64 = (
                    extract_x94_f64_oracle(images_u8, cfg)
                    if "classifier" in stages
                    else np.empty((0, cfg.descriptor_spec.n_x94), dtype=np.float64)
                )
                x94_q15 = (
                    extract_x94_q15_batch(images_u8, cfg)
                    if "classifier" in stages
                    else np.empty((0, cfg.descriptor_spec.n_x94), dtype=np.int16)
                )
            n_test_vectors = int(len(labels))

            if "classifier" in stages and cfg.classifier.models:
                exported_tabular = export_classical_models(
                    cfg,
                    eval_file_names=file_names,
                    eval_true_labels=labels,
                    eval_x94_f64=x94_f64,
                    eval_x94_q15=x94_q15,
                )

            if "cnn" in stages and cfg.cnn.models:
                y_cal = partition.y_train.cpu().numpy().astype(np.int64)
                cal_idx = _select_indices(y_cal, cfg.cnn.calibration_max_images, seed=cfg.seed)
                calib_f32 = partition.X_train[cal_idx].cpu().numpy().astype(np.float32)
                exported_cnns = export_cnn_models(
                    cfg=cfg,
                    calib_images_f32=calib_f32,
                    eval_file_names=file_names,
                    eval_true_labels=labels,
                    eval_images_u8=images_u8,
                    eval_images_f32=images_f32,
                )

        if {"classifier", "cnn", "vectors"} & stages:
            registry_path, mem_report_path, manifest_path = write_registry_and_reports(
                cfg,
                exported_tabular=exported_tabular,
                exported_cnns=exported_cnns,
                extra_manifest={
                    "luts": luts_info,
                    "vectors_manifest": vectors_manifest_str,
                    "n_test_vectors": n_test_vectors,
                },
            )
            logger.info(
                "Export complete: registry=%s, manifest=%s, memory_report=%s",
                registry_path,
                manifest_path,
                mem_report_path,
            )

        if args.compile:
            build_desktop_runner(cfg)

        if args.visualize:
            vis_report = generate_c_pipeline_visualizations(cfg)
            logger.info(
                "Visualizations written (%d figures): %s",
                len(vis_report.figure_paths),
                vis_report.output_dir,
            )

        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        logger.error("C export failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
