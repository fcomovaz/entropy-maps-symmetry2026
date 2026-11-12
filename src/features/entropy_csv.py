"""CSV pipeline for region visual entropy features.

Loads flat preprocessed grayscale image CSVs in streaming chunks, extracts multi-scale
spatial entropy features, and writes a self-contained tabular CSV with metadata provenance
and sidecar JSON documentation.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
import tomllib
from typing import Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.entropy_maps import (
    build_entropy_plan,
    compute_entropy_features,
    feature_names,
    resolve_entropy_backend,
    validate_bins,
    validate_entropy_backend,
    validate_grid_levels,
)
from src.features.native_entropy import load_native_config

logger = logging.getLogger("research")


@dataclass(frozen=True, slots=True)
class EntropyCsvConfig:
    """Immutable configuration for region entropy feature extraction."""

    input_csv: Path
    input_csv_inverted: Path
    output_csv: Path
    output_csv_inverted: Path
    image_size: int
    grid_levels: tuple[int, ...]
    bins: int
    pixel_column_prefix: str
    metadata_columns: tuple[str, ...]
    size_columns: tuple[str, ...]
    feature_name_template: str
    bins_column: str
    levels_column: str
    levels_separator: str
    float_precision: int
    write_sidecar: bool
    chunk_size: int
    inverted: bool
    overwrite: bool
    entropy_backend: str = "auto"
    native_dll_path: Path | None = None

    @property
    def effective_input_csv(self) -> Path:
        """Return the active input CSV path according to the inversion setting."""
        return self.input_csv_inverted if self.inverted else self.input_csv

    @property
    def effective_output_csv(self) -> Path:
        """Return the active output CSV path according to the inversion setting."""
        return self.output_csv_inverted if self.inverted else self.output_csv


@dataclass(frozen=True, slots=True)
class EntropyExportSummary:
    """Summary metrics produced upon completing entropy feature extraction."""

    output_csv: Path
    rows: int
    columns: int
    n_features: int
    feature_names: tuple[str, ...]
    bytes_written: int
    elapsed_s: float
    dry_run: bool
    entropy_backend: str = "auto"
    resolved_entropy_backend: str = "numpy_baseline"


def _require(table: dict[str, object], key: str, where: str) -> object:
    """Extract required key from dictionary or raise KeyError with context."""
    if key not in table:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return table[key]


def load_entropy_config(config_path: Path, project_root: Path) -> EntropyCsvConfig:
    """Load, resolve, and validate entropy feature extraction configuration from TOML.

    Args:
        config_path: Path to the task TOML file (e.g. configs/entropy/mvtec_ad.toml).
        project_root: Root directory of the repository for path anchoring.

    Returns:
        Validated EntropyCsvConfig instance.

    Raises:
        FileNotFoundError: If configuration files do not exist.
        KeyError: If required sections or keys are missing.
        ValueError: If parameters violate schema or validation constraints.
    """
    resolved_task_path = (project_root / config_path).resolve()
    if not resolved_task_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved_task_path}")

    with resolved_task_path.open("rb") as f:
        task_data = tomllib.load(f)

    entropy_table_obj = _require(task_data, "entropy_maps", "task config")
    if not isinstance(entropy_table_obj, dict):
        raise ValueError("[entropy_maps] section must be a table")
    entropy_table: dict[str, object] = entropy_table_obj

    base_config_rel = str(_require(entropy_table, "base_config", "[entropy_maps]"))
    dataset_key = str(_require(entropy_table, "dataset_key", "[entropy_maps]"))
    preprocessing_config_rel = str(
        _require(entropy_table, "preprocessing_config", "[entropy_maps]")
    )
    inverted = _require(entropy_table, "inverted", "[entropy_maps]")
    overwrite = _require(entropy_table, "overwrite", "[entropy_maps]")

    if not isinstance(inverted, bool):
        raise ValueError("'inverted' in [entropy_maps] must be a boolean")
    if not isinstance(overwrite, bool):
        raise ValueError("'overwrite' in [entropy_maps] must be a boolean")

    regions_table_obj = _require(entropy_table, "regions", "[entropy_maps]")
    if not isinstance(regions_table_obj, dict):
        raise ValueError("[entropy_maps.regions] must be a table")
    raw_levels = _require(regions_table_obj, "grid_levels", "[entropy_maps.regions]")
    raw_bins = _require(regions_table_obj, "bins", "[entropy_maps.regions]")

    if not isinstance(raw_levels, list):
        raise ValueError("'grid_levels' in [entropy_maps.regions] must be a list of integers")

    src_table_obj = _require(entropy_table, "source", "[entropy_maps]")
    if not isinstance(src_table_obj, dict):
        raise ValueError("[entropy_maps.source] must be a table")
    pixel_column_prefix = str(_require(src_table_obj, "pixel_column_prefix", "[entropy_maps.source]"))
    raw_metadata_cols = _require(src_table_obj, "metadata_columns", "[entropy_maps.source]")
    raw_size_cols = _require(src_table_obj, "size_columns", "[entropy_maps.source]")

    if not isinstance(raw_metadata_cols, list) or len(raw_metadata_cols) == 0:
        raise ValueError("'metadata_columns' in [entropy_maps.source] must be a non-empty list")
    if not isinstance(raw_size_cols, list) or len(raw_size_cols) != 2:
        raise ValueError("'size_columns' in [entropy_maps.source] must specify exactly 2 columns (e.g. ['width', 'height'])")

    out_table_obj = _require(entropy_table, "output", "[entropy_maps]")
    if not isinstance(out_table_obj, dict):
        raise ValueError("[entropy_maps.output] must be a table")
    feature_name_template = str(_require(out_table_obj, "feature_name_template", "[entropy_maps.output]"))
    float_precision = int(_require(out_table_obj, "float_precision", "[entropy_maps.output]"))
    write_sidecar = bool(_require(out_table_obj, "write_sidecar", "[entropy_maps.output]"))
    bins_column = str(out_table_obj.get("bins_column", "bins"))
    levels_column = str(out_table_obj.get("levels_column", "grid_levels"))
    levels_separator = str(out_table_obj.get("levels_separator", "|"))

    runtime_table_obj = _require(entropy_table, "runtime", "[entropy_maps]")
    if not isinstance(runtime_table_obj, dict):
        raise ValueError("[entropy_maps.runtime] must be a table")
    chunk_size = int(_require(runtime_table_obj, "chunk_size", "[entropy_maps.runtime]"))
    entropy_backend = validate_entropy_backend(str(runtime_table_obj.get("backend", "auto")))
    native_dll_path: Path | None = None
    raw_native_cfg = runtime_table_obj.get("native_config")
    if raw_native_cfg is not None and str(raw_native_cfg).strip():
        native_cfg = load_native_config(Path(str(raw_native_cfg).strip()), project_root=project_root)
        native_dll_path = native_cfg.dll_path

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if float_precision < 1:
        raise ValueError(f"float_precision must be >= 1, got {float_precision}")

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

    input_csv_rel = str(_require(ds_spec, "csv_path", f"[datasets.{dataset_key}]"))
    input_csv_inverted_rel = str(
        _require(ds_spec, "csv_path_inverted", f"[datasets.{dataset_key}]")
    )
    output_csv_rel = str(
        _require(ds_spec, "entropy_csv_path", f"[datasets.{dataset_key}]")
    )
    output_csv_inverted_rel = str(
        _require(ds_spec, "entropy_csv_path_inverted", f"[datasets.{dataset_key}]")
    )

    # Load target image size from preprocessing configuration
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

    if image_size < 1:
        raise ValueError(f"image_size must be >= 1, got {image_size}")

    grid_levels = validate_grid_levels(raw_levels, image_size)
    bins = validate_bins(raw_bins)

    input_csv = (project_root / input_csv_rel).resolve()
    input_csv_inverted = (project_root / input_csv_inverted_rel).resolve()
    output_csv = (project_root / output_csv_rel).resolve()
    output_csv_inverted = (project_root / output_csv_inverted_rel).resolve()

    if input_csv.suffix.lower() != ".csv" or input_csv_inverted.suffix.lower() != ".csv":
        raise ValueError("Input CSV paths must have a .csv extension")
    if output_csv.suffix.lower() != ".csv" or output_csv_inverted.suffix.lower() != ".csv":
        raise ValueError("Output CSV paths must have a .csv extension")

    metadata_columns = tuple(str(col) for col in raw_metadata_cols)
    size_columns = tuple(str(col) for col in raw_size_cols)

    cfg = EntropyCsvConfig(
        input_csv=input_csv,
        input_csv_inverted=input_csv_inverted,
        output_csv=output_csv,
        output_csv_inverted=output_csv_inverted,
        image_size=image_size,
        grid_levels=grid_levels,
        bins=bins,
        pixel_column_prefix=pixel_column_prefix,
        metadata_columns=metadata_columns,
        size_columns=size_columns,
        feature_name_template=feature_name_template,
        bins_column=bins_column,
        levels_column=levels_column,
        levels_separator=levels_separator,
        float_precision=float_precision,
        write_sidecar=write_sidecar,
        chunk_size=chunk_size,
        inverted=inverted,
        overwrite=overwrite,
        entropy_backend=entropy_backend,
        native_dll_path=native_dll_path,
    )

    if cfg.effective_input_csv == cfg.effective_output_csv:
        raise ValueError(
            f"Effective input and output CSV paths cannot be identical: {cfg.effective_input_csv}"
        )

    return cfg


def validate_header(header: list[str], cfg: EntropyCsvConfig) -> None:
    """Validate that the source CSV header satisfies required columns and pixel counts.

    Args:
        header: List of column names in the source CSV.
        cfg: Validated EntropyCsvConfig instance.

    Raises:
        ValueError: If metadata columns, dimension columns, or required pixel columns are missing
            or improperly formed.
    """
    for col in cfg.metadata_columns:
        if col not in header:
            raise ValueError(f"Required metadata column '{col}' missing from source CSV header")

    for col in cfg.size_columns:
        if col not in header:
            raise ValueError(f"Required size column '{col}' missing from source CSV header")

    px_cols = [c for c in header if c.startswith(cfg.pixel_column_prefix)]
    expected_px_count = cfg.image_size * cfg.image_size
    if len(px_cols) != expected_px_count:
        raise ValueError(
            f"Expected {expected_px_count} pixel columns starting with '{cfg.pixel_column_prefix}', "
            f"found {len(px_cols)}"
        )

    first_idx = header.index(px_cols[0])
    last_idx = header.index(px_cols[-1])
    if last_idx - first_idx + 1 != len(px_cols):
        raise ValueError("Pixel columns are not contiguous in the source CSV header")


def export_entropy_csv(cfg: EntropyCsvConfig, dry_run: bool = False) -> EntropyExportSummary:
    """Run chunked region entropy extraction pipeline from input CSV to output CSV.

    Args:
        cfg: Validated EntropyCsvConfig instance.
        dry_run: When True, validates input and schema without writing output files.

    Returns:
        EntropyExportSummary containing execution statistics.

    Raises:
        FileNotFoundError: If effective input CSV does not exist.
        FileExistsError: If effective output CSV exists and overwrite is False.
        ValueError: If headers, image dimensions, or rows violate integrity constraints.
    """
    if not cfg.effective_input_csv.is_file():
        raise FileNotFoundError(
            f"Effective input CSV file not found: {cfg.effective_input_csv}"
        )

    if cfg.effective_output_csv.exists() and not cfg.overwrite and not dry_run:
        raise FileExistsError(
            f"Target output CSV already exists: {cfg.effective_output_csv}. "
            "Use overwrite=True or --overwrite to replace it."
        )

    # Read header from input CSV
    with cfg.effective_input_csv.open("r", encoding="utf-8", newline="") as f_header:
        reader_header = csv.reader(f_header)
        header = next(reader_header, None)
        if header is None:
            raise ValueError(f"Input CSV is empty: {cfg.effective_input_csv}")

    validate_header(header, cfg)

    feat_names = tuple(feature_names(cfg.grid_levels, cfg.feature_name_template))
    output_header = (
        list(cfg.metadata_columns)
        + [cfg.bins_column, cfg.levels_column]
        + list(feat_names)
    )

    metadata_cols = list(cfg.metadata_columns)
    size_cols = list(cfg.size_columns)
    px_cols = [c for c in header if c.startswith(cfg.pixel_column_prefix)]
    usecols = metadata_cols + size_cols + px_cols

    dtypes: dict[str, object] = {c: np.uint8 for c in px_cols}
    for sc in size_cols:
        dtypes[sc] = np.int32
    for mc in metadata_cols:
        dtypes[mc] = str

    bins_val_str = str(cfg.bins)
    levels_val_str = cfg.levels_separator.join(str(lvl) for lvl in cfg.grid_levels)
    float_fmt = f"%.{cfg.float_precision}f"

    start_time = time.perf_counter()
    total_rows = 0
    bytes_written = 0

    resolved_backend = resolve_entropy_backend(
        cfg.entropy_backend,
        cfg.image_size,
        cfg.grid_levels,
        cfg.bins,
        dll_path=cfg.native_dll_path,
    )

    if dry_run:
        logger.info("Executing dry-run validation for %s", cfg.effective_input_csv)
        with pd.read_csv(
            cfg.effective_input_csv,
            usecols=usecols,
            dtype=dtypes,
            chunksize=cfg.chunk_size,
        ) as reader:
            for chunk in reader:
                w_vals = chunk[size_cols[0]].to_numpy()
                h_vals = chunk[size_cols[1]].to_numpy()
                if np.any(w_vals != cfg.image_size) or np.any(h_vals != cfg.image_size):
                    raise ValueError(
                        f"Row image dimensions do not match configured size {cfg.image_size}x{cfg.image_size}"
                    )
                total_rows += len(chunk)
        elapsed_s = time.perf_counter() - start_time
        return EntropyExportSummary(
            output_csv=cfg.effective_output_csv,
            rows=total_rows,
            columns=len(output_header),
            n_features=len(feat_names),
            feature_names=feat_names,
            bytes_written=0,
            elapsed_s=elapsed_s,
            dry_run=True,
            entropy_backend=cfg.entropy_backend,
            resolved_entropy_backend=resolved_backend,
        )

    entropy_plan = build_entropy_plan(
        cfg.image_size,
        cfg.grid_levels,
        cfg.bins,
        backend=cfg.entropy_backend,
        dll_path=cfg.native_dll_path,
    )
    static_cols = [bins_val_str, levels_val_str]
    tmp_output = cfg.effective_output_csv.with_suffix(".csv.tmp")
    tmp_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with (
            tmp_output.open("w", encoding="utf-8", newline="") as f_out,
            pd.read_csv(
                cfg.effective_input_csv,
                usecols=usecols,
                dtype=dtypes,
                chunksize=cfg.chunk_size,
            ) as reader,
        ):
            writer = csv.writer(f_out)
            writer.writerow(output_header)

            for chunk in tqdm(reader, desc="Extracting entropy features", unit="chunk"):
                w_vals = chunk[size_cols[0]].to_numpy()
                h_vals = chunk[size_cols[1]].to_numpy()
                if np.any(w_vals != cfg.image_size) or np.any(h_vals != cfg.image_size):
                    raise ValueError(
                        f"Row image dimensions do not match configured size {cfg.image_size}x{cfg.image_size}"
                    )

                pixels = chunk[px_cols].to_numpy(dtype=np.uint8).reshape(
                    -1, cfg.image_size, cfg.image_size
                )
                entropy_feats = compute_entropy_features(
                    pixels, cfg.grid_levels, cfg.bins, plan=entropy_plan
                )
                formatted_feats = [[float_fmt % val for val in row] for row in entropy_feats]
                meta_vals = chunk[metadata_cols].to_numpy(dtype=str)

                chunk_rows: list[list[str]] = []
                for m_row, e_row in zip(meta_vals, formatted_feats):
                    chunk_rows.append(list(m_row) + static_cols + e_row)

                writer.writerows(chunk_rows)
                total_rows += len(chunk_rows)

        tmp_output.replace(cfg.effective_output_csv)
    except Exception:
        if tmp_output.exists():
            tmp_output.unlink()
        raise

    elapsed_s = time.perf_counter() - start_time
    bytes_written = cfg.effective_output_csv.stat().st_size

    if cfg.write_sidecar:
        sidecar_path = cfg.effective_output_csv.with_suffix(".json")
        sidecar_data = {
            "source_csv": str(cfg.effective_input_csv),
            "output_csv": str(cfg.effective_output_csv),
            "image_size": cfg.image_size,
            "grid_levels": list(cfg.grid_levels),
            "bins": cfg.bins,
            "rows": total_rows,
            "features_count": len(feat_names),
            "feature_columns": list(feat_names),
            "total_columns": len(output_header),
            "float_precision": cfg.float_precision,
            "elapsed_seconds": round(elapsed_s, 4),
            "numpy_version": np.__version__,
            "entropy_backend": cfg.entropy_backend,
            "resolved_entropy_backend": entropy_plan.resolved_backend,
            "timing_metric": "T_feature_extraction_entropy",
        }
        with sidecar_path.open("w", encoding="utf-8") as f_sidecar:
            json.dump(sidecar_data, f_sidecar, indent=2)

    return EntropyExportSummary(
        output_csv=cfg.effective_output_csv,
        rows=total_rows,
        columns=len(output_header),
        n_features=len(feat_names),
        feature_names=feat_names,
        bytes_written=bytes_written,
        elapsed_s=elapsed_s,
        dry_run=False,
        entropy_backend=cfg.entropy_backend,
        resolved_entropy_backend=entropy_plan.resolved_backend,
    )
