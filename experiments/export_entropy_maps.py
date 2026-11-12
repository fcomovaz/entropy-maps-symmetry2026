"""CLI entry point for visual region entropy feature extraction.

Manual execution commands (PowerShell from repository 'code/' directory):
    # Standard MVTec-AD entropy export (creates data/csv/mvtec-ad-entropy.csv):
    .\\.venv\\Scripts\\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml

    # Dry-run MVTec-AD (validates dimensions and schema without writing output):
    .\\.venv\\Scripts\\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml --dry-run

    # Inverted MVTec-AD entropy export:
    .\\.venv\\Scripts\\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml --inverted

    # Standard KolektorSDD2 entropy export (creates data/csv/kolektorsdd2-entropy.csv):
    .\\.venv\\Scripts\\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml

    # Dry-run KolektorSDD2:
    .\\.venv\\Scripts\\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml --dry-run

    # Inverted KolektorSDD2 entropy export:
    .\\.venv\\Scripts\\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml --inverted

    # Force overwrite if target output CSV already exists:
    .\\.venv\\Scripts\\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml --overwrite
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
from src.features.entropy_csv import export_entropy_csv, load_entropy_config
from src.features.entropy_maps import ENTROPY_BACKENDS


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser for entropy map extraction."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.export_entropy_maps",
        description="Extract multi-scale region visual entropy features from flat pixel CSV datasets.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/entropy/mvtec_ad.toml"),
        help="Path to entropy feature extraction TOML configuration (default: configs/entropy/mvtec_ad.toml).",
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
        help="Process inverted dataset variant to generate inverted entropy features CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate source CSV, headers, and image dimensions without creating output files.",
    )
    parser.add_argument(
        "--backend",
        choices=list(ENTROPY_BACKENDS),
        default=None,
        help="Override configured entropy extraction backend (default: TOML setting or 'auto').",
    )
    return parser


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """Execute entropy extraction CLI pipeline.

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
        cfg = load_entropy_config(args.config, project_root)
        if args.overwrite:
            cfg = replace(cfg, overwrite=True)
        if args.inverted:
            cfg = replace(cfg, inverted=True)
        if args.backend is not None:
            cfg = replace(cfg, entropy_backend=args.backend)

        logger.info("Starting visual region entropy extraction pipeline")
        logger.info("Input CSV:             %s", cfg.effective_input_csv)
        logger.info("Output CSV:            %s", cfg.effective_output_csv)
        logger.info("Inverted variant:      %s", cfg.inverted)
        logger.info("Overwrite active:      %s", cfg.overwrite)
        logger.info("Entropy backend:       %s", cfg.entropy_backend)
        logger.info("Image size:            %dx%d", cfg.image_size, cfg.image_size)
        logger.info("Grid levels:           %s", list(cfg.grid_levels))
        logger.info("Histogram bins:        %d", cfg.bins)
        logger.info("Chunk size:            %d", cfg.chunk_size)

        summary = export_entropy_csv(cfg, dry_run=args.dry_run)

        if args.dry_run:
            logger.info(
                "Dry-run completed successfully. Validated %d rows (resolved_backend=%s).",
                summary.rows,
                summary.resolved_entropy_backend,
            )
        else:
            logger.info(
                "Entropy extraction completed successfully (resolved_backend=%s).",
                summary.resolved_entropy_backend,
            )
            logger.info("Wrote %d rows with %d columns (%d entropy features).", summary.rows, summary.columns, summary.n_features)
            logger.info("Output file size:      %.2f MB", summary.bytes_written / (1024 * 1024))
            logger.info("Elapsed execution time: %.2f seconds", summary.elapsed_s)

        return 0

    except (OSError, ValueError, KeyError) as exc:
        logger.error("Entropy extraction execution failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
