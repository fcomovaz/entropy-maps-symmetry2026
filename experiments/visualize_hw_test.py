"""Generate publication-ready MDPI 300-DPI figures for embedded MCU benchmarks (`hw_test/`)."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.logging import configure_logging, logger
from src.hw_test.visualize import load_hw_test_viz_config, run_hw_test_visualization

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for embedded hardware visualization."""
    parser = argparse.ArgumentParser(
        description="Render MDPI 300-DPI cross-board and per-MCU hardware figures from results/hw_test/."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/visualization/hw_test.toml"),
        help="Path to TOML configuration file (default: configs/visualization/hw_test.toml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate fit_report.csv, run summaries, and output path without writing PNGs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing output PNG figures.",
    )
    return parser


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """CLI entrypoint for embedded MCU figure generation."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg = load_hw_test_viz_config(args.config, project_root=project_root)
        run_hw_test_visualization(
            cfg,
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
        )
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        logger.error("Embedded hardware visualization failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
