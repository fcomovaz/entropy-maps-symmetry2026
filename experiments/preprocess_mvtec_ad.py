"""CLI entry point for MVTec-AD dataset preprocessing.

Manual execution command:
    .venv\\Scripts\\python.exe -m experiments.preprocess_mvtec_ad --config configs/preprocessing/mvtec_ad.toml

Dry-run command (scans dataset and counts files without writing):
    .venv\\Scripts\\python.exe -m experiments.preprocess_mvtec_ad --dry-run

Inverted intensity execution (255 - x for images, writes to mvtec-ad-inverted):
    .venv\\Scripts\\python.exe -m experiments.preprocess_mvtec_ad --inverted

Inverted dry-run:
    .venv\\Scripts\\python.exe -m experiments.preprocess_mvtec_ad --inverted --dry-run

Overwrite command (forces reprocessing of all images):
    .venv\\Scripts\\python.exe -m experiments.preprocess_mvtec_ad --overwrite
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# An anchor to project root guarantees module resolution regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import configure_logging, logger
from src.preprocessing.mvtec_ad import load_config, run


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m experiments.preprocess_mvtec_ad",
        description="Clean and preprocess MVTec-AD dataset images to 128x128 grayscale with letterboxing.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/preprocessing/mvtec_ad.toml"),
        help="Path to task configuration TOML file (default: configs/preprocessing/mvtec_ad.toml).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite previously preprocessed files if they already exist.",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        default=False,
        help="Invert grayscale intensity (255 - x) for images; masks untouched; writes to inverted output folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Traverse and count dataset images without performing image writes.",
    )
    return parser


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """Main execution routine.

    Args:
        argv: Command-line arguments. If None, defaults to sys.argv[1:].
        project_root: Root directory of the repository for path anchoring.

    Returns:
        Integer exit code (0 for success, non-zero on failure).
    """
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, project_root)
        if args.overwrite:
            cfg = replace(cfg, overwrite=True)
        if args.inverted:
            cfg = replace(cfg, invert=True)

        logger.info("Starting MVTec-AD preprocessing pipeline")
        logger.info("Input directory:       %s", cfg.input_root)
        logger.info("Output directory:      %s", cfg.output_root)
        logger.info("Inverted directory:    %s", cfg.output_root_inverted)
        logger.info("Effective destination: %s", cfg.effective_output_root)
        logger.info("Invert active:         %s", cfg.invert)
        logger.info("Target size:           %dx%d", cfg.target_size, cfg.target_size)
        logger.info("Overwrite:             %s", cfg.overwrite)

        summary = run(cfg, dry_run=args.dry_run)

        if summary.failed > 0:
            logger.error(
                "Preprocessing completed with %d failed files out of %d",
                summary.failed,
                summary.total_discovered,
            )
            return 1

        logger.info("Preprocessing completed successfully")
        return 0

    except (OSError, ValueError, KeyError, FileNotFoundError) as exc:
        logger.error("Execution failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
