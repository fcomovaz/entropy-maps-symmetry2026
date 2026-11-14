"""CLI to compile desktop C99 runtime/models and verify bit-exact Python-vs-C parity."""

from __future__ import annotations

import argparse
from pathlib import Path

from configs.logging import configure_logging, logger
from src.c_export.config import load_c_imp_config
from src.c_export.parity import run_full_parity_verification
from src.c_export.visualize_plots import generate_c_pipeline_visualizations


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for C99 parity verification."""
    parser = argparse.ArgumentParser(
        description="Compile generated C99 models and verify parity + desktop timing against Python."
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
        "--benchmark",
        action="store_true",
        default=False,
        help="Run warmup + timed repetitions in C and write timing_benchmark.csv.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Render diagnostic summary tables and PNG figures after verification.",
    )
    return parser


def main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    """Run C compilation, parity verification, and optional desktop benchmarking."""
    configure_logging()
    args = build_parser().parse_args(argv)

    try:
        cfg = load_c_imp_config(
            config_path=args.config,
            project_root=project_root,
            object_name=args.object_name,
            inverted=True if args.inverted else None,
        )
        report = run_full_parity_verification(cfg, benchmark=args.benchmark)
        logger.info(
            "Verification complete (all_gates_passed=%s): report=%s",
            report.all_gates_passed,
            report.report_path,
        )
        if args.visualize:
            vis_report = generate_c_pipeline_visualizations(cfg)
            logger.info(
                "Visualizations written (%d figures): %s",
                len(vis_report.figure_paths),
                vis_report.output_dir,
            )
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        logger.error("C verification failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
