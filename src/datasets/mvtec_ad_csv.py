"""MVTec-AD dataset flat CSV export pipeline.

Loads preprocessed grayscale PNGs and writes a single flat CSV where each row
contains metadata, flattened row-major pixel intensities, and dimensions.
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
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("research")

METADATA_COLUMNS: tuple[str, ...] = (
    "object_name",
    "split",
    "type",
    "defect_name",
    "file_name",
)
SIZE_COLUMNS: tuple[str, ...] = ("width", "height")


@dataclass(frozen=True, slots=True)
class CsvExportConfig:
    """Immutable configuration for exporting MVTec-AD images to CSV."""

    source_root: Path
    source_root_inverted: Path
    output_csv: Path
    output_csv_inverted: Path
    subdatasets: tuple[str, ...]
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
class ImageRecord:
    """Provenance record identifying a discovered image file and its metadata."""

    object_name: str
    split: str
    type: str
    defect_name: str
    file_name: str
    path: Path


@dataclass(frozen=True, slots=True)
class CsvExportSummary:
    """Execution summary statistics produced by the CSV export process."""

    output_csv: Path
    rows: int
    counts: dict[tuple[str, str], int]
    columns: int
    bytes_written: int
    elapsed_s: float
    dry_run: bool


def _require(table: dict[str, object], key: str, where: str) -> object:
    """Extract a required key from a dictionary or raise KeyError with context."""
    if key not in table:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return table[key]


def validate_sources(cfg: CsvExportConfig) -> None:
    """Validate that the effective source directory and all subdataset folders exist."""
    root = cfg.effective_source_root
    if not root.exists():
        raise FileNotFoundError(f"Effective source root directory does not exist: {root}")
    for sub in cfg.subdatasets:
        sub_path = root / sub
        if not sub_path.is_dir():
            raise FileNotFoundError(
                f"Configured subdataset category directory '{sub}' not found at: {sub_path}"
            )


def load_csv_config(config_path: Path, project_root: Path) -> CsvExportConfig:
    """Load, resolve, and validate CSV export configuration from TOML files.

    Args:
        config_path: Path to the task TOML file (e.g. configs/csv/mvtec_ad.toml).
        project_root: Root directory of the repository for path anchoring.

    Returns:
        Validated CsvExportConfig instance.

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
    subdatasets_raw = _require(ds_spec, "subdatasets", f"[datasets.{dataset_key}]")
    if not isinstance(subdatasets_raw, list):
        raise ValueError(f"'subdatasets' in [datasets.{dataset_key}] must be a list")

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

    subdatasets = tuple(str(s) for s in subdatasets_raw)
    splits = tuple(str(s) for s in raw_splits)
    image_extensions = tuple(str(ext).lower() for ext in raw_extensions)

    # Validations
    if len(splits) == 0:
        raise ValueError("Configured 'splits' list cannot be empty")
    if len(subdatasets) == 0:
        raise ValueError("Configured 'subdatasets' list cannot be empty")
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

    cfg = CsvExportConfig(
        source_root=source_root,
        source_root_inverted=source_root_inverted,
        output_csv=output_csv,
        output_csv_inverted=output_csv_inverted,
        subdatasets=subdatasets,
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


def classify_type(defect_name: str, cfg: CsvExportConfig) -> str:
    """Classify row type based on defect folder name.

    Returns label_good if defect_name equals good_folder; otherwise label_defect.
    """
    return cfg.label_good if defect_name == cfg.good_folder else cfg.label_defect


def iter_records(cfg: CsvExportConfig) -> Iterator[ImageRecord]:
    """Discover image files deterministically according to configured subdatasets and splits.

    Yields:
        ImageRecord instances sorted by subdataset, split, defect folder, and filename.
    """
    root = cfg.effective_source_root
    for obj_name in cfg.subdatasets:
        obj_dir = root / obj_name
        if not obj_dir.is_dir():
            raise FileNotFoundError(f"Subdataset directory does not exist: {obj_dir}")

        for split in cfg.splits:
            split_dir = obj_dir / split
            if not split_dir.exists():
                raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

            defect_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
            for defect_dir in defect_dirs:
                defect_name = defect_dir.name
                row_type = classify_type(defect_name, cfg)
                image_files = sorted(
                    f
                    for f in defect_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in cfg.image_extensions
                )
                for img_path in image_files:
                    yield ImageRecord(
                        object_name=obj_name,
                        split=split,
                        type=row_type,
                        defect_name=defect_name,
                        file_name=img_path.name,
                        path=img_path,
                    )


def pixel_column_names(n_pixels: int, prefix: str, digits: int) -> list[str]:
    """Generate formatted column headers for flat image pixels."""
    return [f"{prefix}{i:0{digits}d}" for i in range(n_pixels)]


def build_header(cfg: CsvExportConfig) -> list[str]:
    """Construct full CSV header list: metadata columns + pixel columns + dimensions."""
    px_names = pixel_column_names(
        cfg.image_size * cfg.image_size,
        cfg.pixel_column_prefix,
        cfg.pixel_index_digits,
    )
    return list(METADATA_COLUMNS) + px_names + list(SIZE_COLUMNS)


def load_pixels(path: Path, image_size: int) -> np.ndarray:
    """Load image from disk and return as 2D uint8 numpy array with size and mode checks.

    Args:
        path: Path to the image file.
        image_size: Expected square dimension in pixels.

    Returns:
        2D numpy ndarray of shape [image_size, image_size] and uint8 dtype.

    Raises:
        ValueError: If the image mode is not 'L' or dimensions do not match image_size.
    """
    with Image.open(path) as img:
        if img.mode != "L":
            raise ValueError(
                f"Expected grayscale image mode 'L', got '{img.mode}' in {path}"
            )
        if img.size != (image_size, image_size):
            raise ValueError(
                f"Expected image dimensions {(image_size, image_size)}, "
                f"got {img.size} in {path}"
            )
        return np.asarray(img, dtype=np.uint8)


def record_to_row(record: ImageRecord, pixels: np.ndarray) -> list[str | int]:
    """Flatten 2D image pixels in row-major order and assemble the complete CSV row."""
    height, width = pixels.shape
    flat_pixels = pixels.ravel(order="C").tolist()
    row: list[str | int] = [
        record.object_name,
        record.split,
        record.type,
        record.defect_name,
        record.file_name,
    ]
    row.extend(flat_pixels)
    row.extend([width, height])
    return row


def export_csv(cfg: CsvExportConfig, dry_run: bool = False) -> CsvExportSummary:
    """Export dataset images into a single flattened CSV file.

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
        "CSV export completed: %s (%d rows, %d columns, %.2f MB, %.2fs)",
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
