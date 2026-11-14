"""CLI to generate diagnostic figures and summary tables from C99 export/parity CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.logging import configure_logging, logger
from src.c_export.config import CImpConfig, load_c_imp_config
from src.c_export.visualize_data import load_target_bundle
from src.c_export.visualize_plots import generate_c_pipeline_visualizations


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for C99 export/verification visualization."""
    parser = argparse.ArgumentParser(
        description="Render multi-panel matplotlib figures and readable summary CSV/Markdown from C99 outputs."
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
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate target CSV inputs without writing PNG or summary files.",
    )
    return parser


def _validate_dry_run(cfg: CImpConfig) -> None:
    bundle = load_target_bundle(cfg)
    logger.info(
        "Dry-run OK: target=%s | models=%s | out_dir=%s | dpi=%d",
        bundle.target_tag,
        tuple(m.name for m in bundle.models),
        cfg.visualization_dir,
        cfg.visualization.dpi,
    )


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """Load generated C99 export/parity CSVs and render summary figures + tables."""
    configure_logging()
    args = build_parser().parse_args(argv)

    try:
        cfg = load_c_imp_config(
            config_path=args.config,
            project_root=project_root,
            object_name=args.object_name,
            inverted=True if args.inverted else None,
        )
        if args.dry_run:
            _validate_dry_run(cfg)
            return 0

        report = generate_c_pipeline_visualizations(cfg)
        logger.info(
            "Visualization complete for '%s': %d figures, summary=%s, report=%s",
            report.target_tag,
            len(report.figure_paths),
            report.summary_csv_path,
            report.summary_md_path,
        )
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        logger.error("C pipeline visualization failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
