"""CLI entry point for multiscale spatial entropy asymmetry feature extraction.

Appends 9 multiscale spatial asymmetry features (reflection LR, reflection TB,
and 180-degree rotation across scales 2x2, 4x4, 8x8) to existing tabular entropy
CSVs to generate complete x94 descriptor tables.

Manual execution commands (PowerShell from repository 'code/' directory):
    # Standard MVTec-AD asymmetry append (creates data/csv/mvtec-ad-entropy-full.csv):
    .\\.venv\\Scripts\\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml

    # Dry-run MVTec-AD (validates header and entropy ranges without writing output):
    .\\.venv\\Scripts\\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml --dry-run

    # Inverted MVTec-AD asymmetry append:
    .\\.venv\\Scripts\\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml --inverted

    # Standard KolektorSDD2 asymmetry append (creates data/csv/kolektorsdd2-entropy-full.csv):
    .\\.venv\\Scripts\\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/kolektorsdd2.toml

    # Dry-run KolektorSDD2:
    .\\.venv\\Scripts\\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/kolektorsdd2.toml --dry-run

    # Inverted KolektorSDD2 asymmetry append:
    .\\.venv\\Scripts\\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/kolektorsdd2.toml --inverted

    # Run all datasets and variants (MVTec-AD & KolektorSDD2, standard & inverted):
    .\\.venv\\Scripts\\python.exe -m experiments.append_asymmetry_features --all

    # Force overwrite if target output CSV already exists:
    .\\.venv\\Scripts\\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml --overwrite
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import configure_logging, logger
from src.features.asymmetry_csv import append_asymmetry_csv, load_asymmetry_config


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for asymmetry feature appending."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.append_asymmetry_features",
        description="Append 9 multiscale spatial asymmetry features to existing entropy CSV tables.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/asymmetry/mvtec_ad.toml"),
        help="Path to asymmetry task TOML configuration (default: configs/asymmetry/mvtec_ad.toml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite destination CSV and sidecar files if they already exist.",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        default=False,
        help="Process inverted dataset variant to append asymmetry features to inverted entropy CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate source CSV header and entropy value ranges without creating output files.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Process all datasets in configs/asymmetry/*.toml across both standard and inverted variants.",
    )
    return parser


def _process_single(
    config_path: Path,
    overwrite: bool,
    inverted: bool | None,
    dry_run: bool,
    project_root: Path,
) -> None:
    """Execute pipeline for a single configuration and inversion setting."""
    cfg = load_asymmetry_config(config_path, project_root)
    if overwrite:
        cfg = replace(cfg, overwrite=True)
    if inverted is not None:
        cfg = replace(cfg, inverted=inverted)

    logger.info("Starting spatial entropy asymmetry appending pipeline")
    logger.info("Configuration:         %s", config_path)
    logger.info("Input CSV:             %s", cfg.effective_input_csv)
    logger.info("Output CSV:            %s", cfg.effective_output_csv)
    logger.info("Inverted variant:      %s", cfg.inverted)
    logger.info("Overwrite active:      %s", cfg.overwrite)
    logger.info("Scales:                %s", list(cfg.scales))
    logger.info("Kinds:                 %s", list(cfg.kinds))
    logger.info("Chunk size:            %d", cfg.chunk_size)
    logger.info("Dry-run mode:          %s", dry_run)

    summary = append_asymmetry_csv(cfg, dry_run=dry_run)

    if summary.dry_run:
        logger.info(
            "Dry-run complete: validated %d rows, projected output columns=%d (appended %d asymmetry features)",
            summary.rows,
            summary.columns,
            summary.n_asymmetry_features,
        )
    else:
        logger.info(
            "Export complete: wrote %d rows, %d total columns (%d asymmetry features) "
            "to %s (%.2f MB) in %.2fs (T_feature_extraction_asymmetry)",
            summary.rows,
            summary.columns,
            summary.n_asymmetry_features,
            summary.output_csv,
            summary.bytes_written / (1024 * 1024),
            summary.elapsed_s,
        )


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """Execute asymmetry appending CLI pipeline.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].
        project_root: Root directory for relative path resolution (enables test injection).

    Returns:
        Integer exit code (0 for success).
    """
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.all:
            asym_configs_dir = project_root / "configs" / "asymmetry"
            toml_files = sorted(asym_configs_dir.glob("*.toml"))
            if not toml_files:
                raise FileNotFoundError(
                    f"No configuration files found in {asym_configs_dir}"
                )

            logger.info("Running --all over %d configurations", len(toml_files))
            for cfg_path in toml_files:
                rel_cfg = cfg_path.relative_to(project_root)
                # Standard variant
                _process_single(
                    config_path=rel_cfg,
                    overwrite=args.overwrite,
                    inverted=False,
                    dry_run=args.dry_run,
                    project_root=project_root,
                )
                # Inverted variant
                _process_single(
                    config_path=rel_cfg,
                    overwrite=args.overwrite,
                    inverted=True,
                    dry_run=args.dry_run,
                    project_root=project_root,
                )
        else:
            _process_single(
                config_path=args.config,
                overwrite=args.overwrite,
                inverted=True if args.inverted else None,
                dry_run=args.dry_run,
                project_root=project_root,
            )
        return 0
    except (OSError, ValueError, KeyError) as exc:
        logger.error("Pipeline failure: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
