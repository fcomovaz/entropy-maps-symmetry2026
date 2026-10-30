"""CLI entry point for MVTec-AD dataset CSV export.

Manual execution commands:
    # Standard export (creates data/csv/mvtec-ad.csv):
    .venv\\Scripts\\python.exe -m experiments.export_mvtec_ad_csv

    # Standard dry-run (counts rows without writing CSV):
    .venv\\Scripts\\python.exe -m experiments.export_mvtec_ad_csv --dry-run

    # Inverted export (creates data/csv/mvtec-ad-inverted.csv):
    .venv\\Scripts\\python.exe -m experiments.export_mvtec_ad_csv --inverted

    # Inverted dry-run:
    .venv\\Scripts\\python.exe -m experiments.export_mvtec_ad_csv --inverted --dry-run

    # Force overwrite if CSV already exists:
    .venv\\Scripts\\python.exe -m experiments.export_mvtec_ad_csv --overwrite
    .venv\\Scripts\\python.exe -m experiments.export_mvtec_ad_csv --inverted --overwrite
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
from src.datasets.mvtec_ad_csv import export_csv, load_csv_config


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for CSV export."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.export_mvtec_ad_csv",
        description="Flatten preprocessed MVTec-AD images into a single structured CSV file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/csv/mvtec_ad.toml"),
        help="Path to CSV export TOML configuration file (default: configs/csv/mvtec_ad.toml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite target CSV file if it already exists.",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        default=False,
        help="Export preprocessed inverted images to inverted CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Count and validate images without creating or modifying CSV files.",
    )
    return parser


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """Execute CSV export pipeline.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].
        project_root: Root directory for path resolution (enables test injection).

    Returns:
        Integer exit code (0 for success).
    """
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_csv_config(args.config, project_root)
        if args.overwrite:
            cfg = replace(cfg, overwrite=True)
        if args.inverted:
            cfg = replace(cfg, inverted=True)

        logger.info("Starting MVTec-AD CSV export pipeline")
        logger.info("Source directory:      %s", cfg.effective_source_root)
        logger.info("Output CSV file:       %s", cfg.effective_output_csv)
        logger.info("Inverted variant:      %s", cfg.inverted)
        logger.info("Overwrite active:      %s", cfg.overwrite)
        logger.info("Target size:           %dx%d", cfg.image_size, cfg.image_size)

        summary = export_csv(cfg, dry_run=args.dry_run)

        if args.dry_run:
            logger.info("Dry-run completed. Discovered %d rows.", summary.rows)
        else:
            logger.info(
                "CSV export completed successfully: %d rows written to %s (%.2fs)",
                summary.rows,
                summary.output_csv,
                summary.elapsed_s,
            )
        return 0

    except (OSError, ValueError, KeyError) as exc:
        logger.error("MVTec-AD CSV export failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
