"""KolektorSDD2 dataset flat CSV export pipeline.

Loads preprocessed grayscale PNGs and writes a single flat CSV where each row
contains metadata (split, type, file_name), flattened row-major pixel intensities,
and spatial dimensions (width, height).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import logging
from pathlib import Path
import time
import tomllib
from typing import Iterator

import numpy as np
from tqdm import tqdm

from src.datasets.mvtec_ad_csv import (
    CsvExportSummary,
    load_pixels,
    pixel_column_names,
)

logger = logging.getLogger("research")

METADATA_COLUMNS: tuple[str, ...] = (
    "split",
    "type",
    "file_name",
)
SIZE_COLUMNS: tuple[str, ...] = ("width", "height")


@dataclass(frozen=True, slots=True)
class KolektorCsvConfig:
    """Immutable configuration for exporting KolektorSDD2 images to CSV."""

    source_root: Path
    source_root_inverted: Path
    output_csv: Path
    output_csv_inverted: Path
    splits: tuple[str, ...]
    image_extensions: tuple[str, ...]
    good_folder: str
    label_good: str
    label_defect: str
    image_size: int
    pixel_column_prefix: str
    pixel_index_digits: int
    inverted: bool
    overwrite: bool

    @property
    def effective_source_root(self) -> Path:
        """Return the active source root directory based on the inversion flag."""
        return self.source_root_inverted if self.inverted else self.source_root

    @property
    def effective_output_csv(self) -> Path:
        """Return the active output CSV file path based on the inversion flag."""
        return self.output_csv_inverted if self.inverted else self.output_csv


@dataclass(frozen=True, slots=True)
class KolektorRecord:
    """Provenance record identifying a discovered KolektorSDD2 image file and its metadata."""

    split: str
    type: str
    file_name: str
    path: Path


def _require(table: dict[str, object], key: str, where: str) -> object:
    """Extract a required key from a dictionary or raise KeyError with context."""
    if key not in table:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return table[key]


def validate_sources(cfg: KolektorCsvConfig) -> None:
    """Validate that the effective source directory and all split folders exist."""
    root = cfg.effective_source_root
    if not root.exists():
        raise FileNotFoundError(f"Effective source root directory does not exist: {root}")
    for split in cfg.splits:
        split_path = root / split
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Configured split directory '{split}' not found at: {split_path}"
            )


def load_kolektor_csv_config(
    config_path: Path, project_root: Path
) -> KolektorCsvConfig:
    """Load, resolve, and validate CSV export configuration from TOML files.

    Args:
        config_path: Path to the task TOML file (e.g. configs/csv/kolektorsdd2.toml).
        project_root: Root directory of the repository for path anchoring.

    Returns:
        Validated KolektorCsvConfig instance.

    Raises:
        FileNotFoundError: If configuration files or effective source directories do not exist.
        KeyError: If required configuration keys or sections are missing.
        ValueError: If configuration values violate operational constraints.
    """
    resolved_config_path = (project_root / config_path).resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved_config_path}")

    with resolved_config_path.open("rb") as f:
        task_data = tomllib.load(f)

    csv_table_obj = _require(task_data, "csv", "task config")
    if not isinstance(csv_table_obj, dict):
        raise ValueError("[csv] section must be a table")
    csv_table: dict[str, object] = csv_table_obj

    base_config_rel = str(_require(csv_table, "base_config", "[csv]"))
    dataset_key = str(_require(csv_table, "dataset_key", "[csv]"))
    preprocessing_config_rel = str(_require(csv_table, "preprocessing_config", "[csv]"))
    inverted = _require(csv_table, "inverted", "[csv]")
    overwrite = _require(csv_table, "overwrite", "[csv]")

    if not isinstance(inverted, bool):
        raise ValueError("'inverted' in [csv] must be a boolean")
    if not isinstance(overwrite, bool):
        raise ValueError("'overwrite' in [csv] must be a boolean")

    raw_extensions = _require(csv_table, "image_extensions", "[csv]")
    raw_splits = _require(csv_table, "splits", "[csv]")

    labels_table_obj = _require(csv_table, "labels", "[csv]")
    if not isinstance(labels_table_obj, dict):
        raise ValueError("[csv.labels] section must be a table")
    good_folder = str(_require(labels_table_obj, "good_folder", "[csv.labels]"))
    label_good = str(_require(labels_table_obj, "good", "[csv.labels]"))
    label_defect = str(_require(labels_table_obj, "defect", "[csv.labels]"))

    pixels_table_obj = _require(csv_table, "pixels", "[csv]")
    if not isinstance(pixels_table_obj, dict):
        raise ValueError("[csv.pixels] section must be a table")
    pixel_column_prefix = str(_require(pixels_table_obj, "column_prefix", "[csv.pixels]"))
    index_digits = int(_require(pixels_table_obj, "index_digits", "[csv.pixels]"))

    # Load base dataset configuration
    base_config_path = (project_root / base_config_rel).resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")

    with base_config_path.open("rb") as f:
        base_data = tomllib.load(f)

    datasets_table = _require(base_data, "datasets", "base config")
    if not isinstance(datasets_table, dict):
        raise ValueError("[datasets] section in base config must be a table")

    ds_spec = _require(datasets_table, dataset_key, f"base config [datasets.{dataset_key}]")
    if not isinstance(ds_spec, dict):
        raise ValueError(f"[datasets.{dataset_key}] in base config must be a table")

    output_path_rel = str(_require(ds_spec, "output_path", f"[datasets.{dataset_key}]"))
    output_path_inverted_rel = str(
        _require(ds_spec, "output_path_inverted", f"[datasets.{dataset_key}]")
    )
    csv_path_rel = str(_require(ds_spec, "csv_path", f"[datasets.{dataset_key}]"))
    csv_path_inverted_rel = str(
        _require(ds_spec, "csv_path_inverted", f"[datasets.{dataset_key}]")
    )

    # Load preprocessing configuration for target image size
    prep_config_path = (project_root / preprocessing_config_rel).resolve()
    if not prep_config_path.is_file():
        raise FileNotFoundError(
            f"Preprocessing configuration file not found: {prep_config_path}"
        )

    with prep_config_path.open("rb") as f:
        prep_data = tomllib.load(f)

    prep_table = _require(prep_data, "preprocessing", "preprocessing config")
    if not isinstance(prep_table, dict):
        raise ValueError("[preprocessing] section must be a table")
    image_size = int(_require(prep_table, "target_size", "[preprocessing]"))

    # Path anchors
    source_root = (project_root / output_path_rel).resolve()
    source_root_inverted = (project_root / output_path_inverted_rel).resolve()
    output_csv = (project_root / csv_path_rel).resolve()
    output_csv_inverted = (project_root / csv_path_inverted_rel).resolve()

    splits = tuple(str(s) for s in raw_splits)
    image_extensions = tuple(str(ext).lower() for ext in raw_extensions)

    # Validations
    if len(splits) == 0:
        raise ValueError("Configured 'splits' list cannot be empty")
    if image_size < 1:
        raise ValueError(f"image_size must be >= 1, got {image_size}")

    max_pixel_idx = image_size * image_size - 1
    min_digits = len(str(max_pixel_idx))
    if index_digits < min_digits:
        raise ValueError(
            f"index_digits ({index_digits}) too small for image_size {image_size} "
            f"(requires >= {min_digits})"
        )

    if output_csv == output_csv_inverted:
        raise ValueError(
            f"output_csv ({output_csv}) and output_csv_inverted ({output_csv_inverted}) must differ"
        )
    if output_csv.suffix.lower() != ".csv":
        raise ValueError(f"output_csv must have .csv extension, got {output_csv}")
    if output_csv_inverted.suffix.lower() != ".csv":
        raise ValueError(
            f"output_csv_inverted must have .csv extension, got {output_csv_inverted}"
        )

    cfg = KolektorCsvConfig(
        source_root=source_root,
        source_root_inverted=source_root_inverted,
        output_csv=output_csv,
        output_csv_inverted=output_csv_inverted,
        splits=splits,
        image_extensions=image_extensions,
        good_folder=good_folder,
        label_good=label_good,
        label_defect=label_defect,
        image_size=image_size,
        pixel_column_prefix=pixel_column_prefix,
        pixel_index_digits=index_digits,
        inverted=inverted,
        overwrite=overwrite,
    )

    validate_sources(cfg)
    return cfg


def classify_type(class_folder: str, cfg: KolektorCsvConfig) -> str:
    """Classify row type based on class folder name.

    Returns label_good if class_folder equals good_folder; otherwise label_defect.
    """
    return cfg.label_good if class_folder == cfg.good_folder else cfg.label_defect


def iter_records(cfg: KolektorCsvConfig) -> Iterator[KolektorRecord]:
    """Discover image files deterministically according to configured splits.

    Yields:
        KolektorRecord instances sorted by split (TOML order), class folder, and filename.
    """
    root = cfg.effective_source_root
    for split in cfg.splits:
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

        class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
        for class_dir in class_dirs:
            class_folder = class_dir.name
            row_type = classify_type(class_folder, cfg)
            image_files = sorted(
                f
                for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in cfg.image_extensions
            )
            for img_path in image_files:
                yield KolektorRecord(
                    split=split,
                    type=row_type,
                    file_name=img_path.name,
                    path=img_path,
                )


def build_header(cfg: KolektorCsvConfig) -> list[str]:
    """Construct full CSV header list: metadata columns + pixel columns + dimensions."""
    px_names = pixel_column_names(
        cfg.image_size * cfg.image_size,
        cfg.pixel_column_prefix,
        cfg.pixel_index_digits,
    )
    return list(METADATA_COLUMNS) + px_names + list(SIZE_COLUMNS)


def record_to_row(record: KolektorRecord, pixels: np.ndarray) -> list[str | int]:
    """Flatten 2D image pixels in row-major order and assemble the complete CSV row."""
    height, width = pixels.shape
    flat_pixels = pixels.ravel(order="C").tolist()
    row: list[str | int] = [
        record.split,
        record.type,
        record.file_name,
    ]
    row.extend(flat_pixels)
    row.extend([width, height])
    return row


def export_kolektor_csv(
    cfg: KolektorCsvConfig, dry_run: bool = False
) -> CsvExportSummary:
    """Export KolektorSDD2 dataset images into a single flattened CSV file.

    Writes to a temporary file first and atomically renames on success.

    Args:
        cfg: Validated CSV export configuration.
        dry_run: If True, count records without writing files.

    Returns:
        CsvExportSummary with execution metrics.

    Raises:
        FileExistsError: If output CSV exists and overwrite is False.
    """
    validate_sources(cfg)
    output = cfg.effective_output_csv
    if output.exists() and not cfg.overwrite:
        raise FileExistsError(
            f"Output CSV file already exists and overwrite=False: {output}"
        )

    records = list(iter_records(cfg))
    counts: dict[tuple[str, str], int] = {}
    for r in records:
        k = (r.split, r.type)
        counts[k] = counts.get(k, 0) + 1

    header = build_header(cfg)
    cols_count = len(header)

    if dry_run:
        logger.info(
            "Dry-run: %d records discovered across %d columns (no files written).",
            len(records),
            cols_count,
        )
        for k, v in sorted(counts.items()):
            logger.info("  Split/type %s: %d", k, v)
        return CsvExportSummary(
            output_csv=output,
            rows=len(records),
            counts=counts,
            columns=cols_count,
            bytes_written=0,
            elapsed_s=0.0,
            dry_run=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    start_time = time.perf_counter()
    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for rec in tqdm(records, desc=f"Exporting {output.name}", unit="img"):
                pixels = load_pixels(rec.path, cfg.image_size)
                writer.writerow(record_to_row(rec, pixels))
        tmp.replace(output)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    elapsed = time.perf_counter() - start_time
    bytes_written = output.stat().st_size
    mb = bytes_written / (1024 * 1024)
    logger.info(
        "KolektorSDD2 CSV export completed: %s (%d rows, %d columns, %.2f MB, %.2fs)",
        output,
        len(records),
        cols_count,
        mb,
        elapsed,
    )
    for k, v in sorted(counts.items()):
        logger.info("  Split/type %s: %d", k, v)

    return CsvExportSummary(
        output_csv=output,
        rows=len(records),
        counts=counts,
        columns=cols_count,
        bytes_written=bytes_written,
        elapsed_s=elapsed,
        dry_run=False,
    )
