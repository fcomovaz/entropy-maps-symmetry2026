"""Post-export verification and round-trip visual integrity validation for KolektorSDD2 CSV.

Streams the generated CSV without loading the full file into memory, validates schema,
bounds, and counts, reconstructs sample images, and asserts exact bit-level equality
with source PNG files.

Manual execution commands:
    # Verify standard CSV (data/csv/kolektorsdd2.csv):
    .venv\\Scripts\\python.exe understanding/11_verify_kolektorsdd2_csv/verify.py

    # Verify inverted CSV (data/csv/kolektorsdd2-inverted.csv):
    .venv\\Scripts\\python.exe understanding/11_verify_kolektorsdd2_csv/verify.py --inverted
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import configure_logging, logger
from src.datasets.kolektorsdd2_csv import build_header, load_kolektor_csv_config


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for KolektorSDD2 CSV verification utility."""
    parser = argparse.ArgumentParser(
        prog="verify.py",
        description="Verify integrity, row count, and exact pixel reconstruction of KolektorSDD2 CSV.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/csv/kolektorsdd2.toml"),
        help="Path to CSV export TOML configuration file.",
    )
    parser.add_argument(
        "--inverted",
        action="store_true",
        default=False,
        help="Verify the inverted CSV export (kolektorsdd2-inverted.csv).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of random rows to reconstruct and bit-compare against source PNGs (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sample selection.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Stream CSV, validate header and values, check round-trip equality, and save plot."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_kolektor_csv_config(args.config, PROJECT_ROOT)
    if args.inverted:
        cfg = replace(cfg, inverted=True)

    csv_file = cfg.effective_output_csv
    if not csv_file.is_file():
        logger.error("Target CSV file does not exist: %s", csv_file)
        return 1

    expected_header = build_header(cfg)
    expected_cols = len(expected_header)

    logger.info("Opening KolektorSDD2 CSV for stream verification: %s", csv_file)
    logger.info("Expected column count: %d", expected_cols)

    # Pass 1: validate header and count rows
    total_rows = 0
    split_type_counts: dict[tuple[str, str], int] = {}
    with csv_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            actual_header = next(reader)
        except StopIteration:
            logger.error("CSV file is empty: %s", csv_file)
            return 1

        if actual_header != expected_header:
            logger.error("CSV header does not match expected schema")
            return 1

        for row_idx, row in enumerate(reader):
            if len(row) != expected_cols:
                logger.error(
                    "Row %d has %d columns, expected %d", row_idx, len(row), expected_cols
                )
                return 1

            split = row[0]
            row_type = row[1]
            key = (split, row_type)
            split_type_counts[key] = split_type_counts.get(key, 0) + 1
            total_rows += 1

    logger.info("Total data rows discovered: %d", total_rows)
    for k, count in sorted(split_type_counts.items()):
        logger.info("  Split/type %s: %d", k, count)

    if total_rows == 0:
        logger.error("No data rows found in CSV.")
        return 1

    # Select random row indices to verify and visualize
    rng = np.random.default_rng(args.seed)
    n_samples = min(args.samples, total_rows)
    sample_indices = set(rng.choice(total_rows, size=n_samples, replace=False).tolist())

    logger.info(
        "Selecting %d sample rows for round-trip pixel verification (seed=%d)...",
        n_samples,
        args.seed,
    )

    # Pass 2: retrieve sampled rows and validate pixel values
    sampled_records: list[tuple[list[str], np.ndarray, np.ndarray]] = []
    mismatches = 0

    with csv_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row_idx, row in enumerate(reader):
            width = int(row[-2])
            height = int(row[-1])
            if width != cfg.image_size or height != cfg.image_size:
                logger.error(
                    "Row %d dimensions (%dx%d) do not match configured image_size %d",
                    row_idx,
                    width,
                    height,
                    cfg.image_size,
                )
                mismatches += 1

            raw_px = [int(v) for v in row[3:-2]]

            # Range checks
            min_px = min(raw_px)
            max_px = max(raw_px)
            if min_px < 0 or max_px > 255:
                logger.error(
                    "Row %d pixel intensity outside [0, 255]: [%d, %d]",
                    row_idx,
                    min_px,
                    max_px,
                )
                mismatches += 1

            if row_idx in sample_indices:
                split = row[0]
                row_type = row[1]
                file_name = row[2]
                source_png = (
                    cfg.effective_source_root
                    / split
                    / row_type
                    / file_name
                )
                if not source_png.is_file():
                    logger.error("Source PNG not found for row %d: %s", row_idx, source_png)
                    mismatches += 1
                    continue

                orig_img = np.asarray(Image.open(source_png), dtype=np.uint8)
                recon_img = np.array(raw_px, dtype=np.uint8).reshape((height, width))

                if not np.array_equal(orig_img, recon_img):
                    diff = np.abs(orig_img.astype(np.int32) - recon_img.astype(np.int32))
                    logger.error(
                        "Pixel mismatch in row %d (%s)! Max diff: %d",
                        row_idx,
                        source_png,
                        np.max(diff),
                    )
                    mismatches += 1
                else:
                    sampled_records.append((row, orig_img, recon_img))

    if mismatches > 0:
        logger.error("Verification failed with %d mismatches.", mismatches)
        return 1

    logger.info("All %d sampled rows verified with 0 pixel discrepancies.", len(sampled_records))

    # Generate visual figure
    if sampled_records:
        n_plots = min(len(sampled_records), 4)
        fig, axes = plt.subplots(n_plots, 3, figsize=(9, 3 * n_plots))
        if n_plots == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(n_plots):
            rec_row, orig_img, recon_img = sampled_records[i]
            split = rec_row[0]
            row_type = rec_row[1]
            file_name = rec_row[2]
            diff = np.abs(orig_img.astype(np.int32) - recon_img.astype(np.int32))

            axes[i, 0].imshow(orig_img, cmap="gray", vmin=0, vmax=255)
            axes[i, 0].set_title(f"Source PNG\n{split}/{row_type}/{file_name}", fontsize=8)
            axes[i, 0].axis("off")

            axes[i, 1].imshow(recon_img, cmap="gray", vmin=0, vmax=255)
            axes[i, 1].set_title(f"Reconstructed CSV\n({split}/{row_type})", fontsize=8)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(diff, cmap="hot", vmin=0, vmax=255)
            axes[i, 2].set_title(f"Absolute Diff (max={diff.max()})", fontsize=8)
            axes[i, 2].axis("off")

        plt.tight_layout()
        output_fig = Path(__file__).resolve().parent / (
            "csv_roundtrip_inverted.png" if args.inverted else "csv_roundtrip.png"
        )
        plt.savefig(output_fig, dpi=150)
        plt.close(fig)
        logger.info("Saved verification figure to: %s", output_fig)

    logger.info("KolektorSDD2 CSV verification completed successfully with 0 errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
