"""CSV pipeline for multiscale spatial entropy asymmetry features.

Reads existing tabular entropy CSV files (x85) in streaming chunks, computes the
9 spatial asymmetry measurements (x9), and appends them to produce the complete
multiscale descriptor tables (x94) with sidecar JSON documentation.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
import tomllib
from typing import Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.asymmetry_maps import (
    asymmetry_column_names,
    compute_asymmetry_features,
    entropy_column_names,
    validate_entropy_values,
    validate_kinds,
    validate_scales,
)

logger = logging.getLogger("research")


@dataclass(frozen=True, slots=True)
class AsymmetryCsvConfig:
    """Immutable configuration for appending spatial asymmetry features to entropy CSVs."""

    input_csv: Path
    input_csv_inverted: Path
    output_csv: Path
    output_csv_inverted: Path
    scales: tuple[int, ...]
    kinds: tuple[str, ...]
    column_templates: dict[str, str]
    feature_name_template: str
    global_scale: int
    bins_column: str
    levels_column: str
    levels_separator: str
    suffix: str
    float_precision: int
    value_tolerance: float
    write_sidecar: bool
    sidecar_timing_metric: str
    schema: str
    chunk_size: int
    inverted: bool
    overwrite: bool

    @property
    def effective_input_csv(self) -> Path:
        """Return the active input CSV path according to the inversion setting."""
        return self.input_csv_inverted if self.inverted else self.input_csv

    @property
    def effective_output_csv(self) -> Path:
        """Return the active output CSV path according to the inversion setting."""
        return self.output_csv_inverted if self.inverted else self.output_csv


@dataclass(frozen=True, slots=True)
class AsymmetryExportSummary:
    """Summary metrics produced upon completing asymmetry feature appending."""

    output_csv: Path
    rows: int
    columns: int
    n_asymmetry_features: int
    asymmetry_feature_names: tuple[str, ...]
    bytes_written: int
    elapsed_s: float
    dry_run: bool


def _require(table: Mapping[str, object], key: str, where: str) -> object:
    """Extract required key from dictionary or raise KeyError with context."""
    if key not in table:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return table[key]


def load_asymmetry_config(config_path: Path, project_root: Path) -> AsymmetryCsvConfig:
    """Load, resolve, and validate asymmetry feature appending configuration from TOML.

    Args:
        config_path: Path to the asymmetry task TOML file (e.g. configs/asymmetry/mvtec_ad.toml).
        project_root: Root directory of the repository for dynamic path anchoring.

    Returns:
        Validated AsymmetryCsvConfig instance.

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

    asym_table_obj = _require(task_data, "asymmetry_features", "task config")
    if not isinstance(asym_table_obj, dict):
        raise ValueError("[asymmetry_features] section must be a table")
    asym_table: dict[str, object] = asym_table_obj

    base_config_rel = str(_require(asym_table, "base_config", "[asymmetry_features]"))
    dataset_key = str(_require(asym_table, "dataset_key", "[asymmetry_features]"))
    inverted = _require(asym_table, "inverted", "[asymmetry_features]")
    overwrite = _require(asym_table, "overwrite", "[asymmetry_features]")
    schema = str(_require(asym_table, "schema", "[asymmetry_features]"))

    if not isinstance(inverted, bool):
        raise ValueError("'inverted' in [asymmetry_features] must be a boolean")
    if not isinstance(overwrite, bool):
        raise ValueError("'overwrite' in [asymmetry_features] must be a boolean")

    # Optional linkage to entropy config for single-source parameters
    entropy_config_rel = asym_table.get("entropy_config")
    entropy_out_table: dict[str, object] = {}
    if entropy_config_rel is not None:
        entropy_config_path = (project_root / str(entropy_config_rel)).resolve()
        if entropy_config_path.is_file():
            with entropy_config_path.open("rb") as f_ent:
                ent_data = tomllib.load(f_ent)
                ent_maps = ent_data.get("entropy_maps", {})
                if isinstance(ent_maps, dict):
                    out_t = ent_maps.get("output", {})
                    if isinstance(out_t, dict):
                        entropy_out_table = out_t

    # Regions table
    regions_table_obj = _require(asym_table, "regions", "[asymmetry_features]")
    if not isinstance(regions_table_obj, dict):
        raise ValueError("[asymmetry_features.regions] must be a table")
    raw_scales = _require(regions_table_obj, "scales", "[asymmetry_features.regions]")
    raw_kinds = _require(regions_table_obj, "kinds", "[asymmetry_features.regions]")

    if not isinstance(raw_scales, list):
        raise ValueError("'scales' in [asymmetry_features.regions] must be a list of integers")
    if not isinstance(raw_kinds, list):
        raise ValueError("'kinds' in [asymmetry_features.regions] must be a list of strings")

    # Source table
    source_table_obj = asym_table.get("source", {})
    if not isinstance(source_table_obj, dict):
        raise ValueError("[asymmetry_features.source] must be a table")

    feature_name_template = str(
        source_table_obj.get(
            "feature_name_template",
            entropy_out_table.get("feature_name_template", "E_{g}x{g}_r{r}_c{c}"),
        )
    )
    global_scale = int(source_table_obj.get("global_scale", 1))
    bins_column = str(
        source_table_obj.get(
            "bins_column",
            entropy_out_table.get("bins_column", "bins"),
        )
    )
    levels_column = str(
        source_table_obj.get(
            "levels_column",
            entropy_out_table.get("levels_column", "grid_levels"),
        )
    )
    levels_separator = str(
        source_table_obj.get(
            "levels_separator",
            entropy_out_table.get("levels_separator", "|"),
        )
    )

    scales = validate_scales(raw_scales, global_scale=global_scale)
    kinds = validate_kinds([str(k) for k in raw_kinds])

    # Output table
    out_table_obj = _require(asym_table, "output", "[asymmetry_features]")
    if not isinstance(out_table_obj, dict):
        raise ValueError("[asymmetry_features.output] must be a table")

    suffix = str(_require(out_table_obj, "suffix", "[asymmetry_features.output]"))
    raw_templates = _require(out_table_obj, "column_templates", "[asymmetry_features.output]")
    if not isinstance(raw_templates, dict):
        raise ValueError("'column_templates' in [asymmetry_features.output] must be a table")
    column_templates = {str(k): str(v) for k, v in raw_templates.items()}

    float_precision = int(_require(out_table_obj, "float_precision", "[asymmetry_features.output]"))
    value_tolerance = float(out_table_obj.get("value_tolerance", 1e-6))
    write_sidecar = bool(_require(out_table_obj, "write_sidecar", "[asymmetry_features.output]"))
    sidecar_timing_metric = str(
        out_table_obj.get("sidecar_timing_metric", "T_feature_extraction_asymmetry")
    )

    # Runtime table
    runtime_table_obj = _require(asym_table, "runtime", "[asymmetry_features]")
    if not isinstance(runtime_table_obj, dict):
        raise ValueError("[asymmetry_features.runtime] must be a table")
    chunk_size = int(_require(runtime_table_obj, "chunk_size", "[asymmetry_features.runtime]"))

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if float_precision < 1:
        raise ValueError(f"float_precision must be >= 1, got {float_precision}")
    if value_tolerance < 0.0:
        raise ValueError(f"value_tolerance must be >= 0.0, got {value_tolerance}")

    # Load base dataset paths
    base_config_path = (project_root / base_config_rel).resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")

    with base_config_path.open("rb") as f_base:
        base_data = tomllib.load(f_base)

    datasets_table = _require(base_data, "datasets", "base config")
    if not isinstance(datasets_table, dict):
        raise ValueError("[datasets] section in base config must be a table")

    ds_spec = _require(datasets_table, dataset_key, f"base config [datasets.{dataset_key}]")
    if not isinstance(ds_spec, dict):
        raise ValueError(f"[datasets.{dataset_key}] in base config must be a table")

    input_csv_rel = str(_require(ds_spec, "entropy_csv_path", f"[datasets.{dataset_key}]"))
    input_csv_inverted_rel = str(
        _require(ds_spec, "entropy_csv_path_inverted", f"[datasets.{dataset_key}]")
    )

    input_csv = (project_root / input_csv_rel).resolve()
    input_csv_inverted = (project_root / input_csv_inverted_rel).resolve()

    if input_csv.suffix.lower() != ".csv" or input_csv_inverted.suffix.lower() != ".csv":
        raise ValueError("Input entropy CSV paths must have a .csv extension")

    output_csv = input_csv.with_name(f"{input_csv.stem}{suffix}{input_csv.suffix}")
    output_csv_inverted = input_csv_inverted.with_name(
        f"{input_csv_inverted.stem}{suffix}{input_csv_inverted.suffix}"
    )

    if input_csv == output_csv or input_csv_inverted == output_csv_inverted:
        raise ValueError(
            f"Input and output CSV paths cannot be identical. Check suffix: '{suffix}'"
        )

    return AsymmetryCsvConfig(
        input_csv=input_csv,
        input_csv_inverted=input_csv_inverted,
        output_csv=output_csv,
        output_csv_inverted=output_csv_inverted,
        scales=scales,
        kinds=kinds,
        column_templates=column_templates,
        feature_name_template=feature_name_template,
        global_scale=global_scale,
        bins_column=bins_column,
        levels_column=levels_column,
        levels_separator=levels_separator,
        suffix=suffix,
        float_precision=float_precision,
        value_tolerance=value_tolerance,
        write_sidecar=write_sidecar,
        sidecar_timing_metric=sidecar_timing_metric,
        schema=schema,
        chunk_size=chunk_size,
        inverted=inverted,
        overwrite=overwrite,
    )


def validate_header(header: list[str], cfg: AsymmetryCsvConfig) -> list[str]:
    """Validate that the source CSV header contains required entropy columns and no collisions.

    Args:
        header: List of column names in the source CSV header.
        cfg: Validated AsymmetryCsvConfig instance.

    Returns:
        List of all required entropy feature column names across the configured scales.

    Raises:
        ValueError: If the global scale column or fine scale entropy columns are missing,
            or if any output asymmetry column already exists in the header.
    """
    global_cols = entropy_column_names(cfg.global_scale, cfg.feature_name_template)
    for col in global_cols:
        if col not in header:
            raise ValueError(
                f"Required global entropy column '{col}' missing from source CSV header"
            )

    required_entropy_cols: list[str] = []
    for g in cfg.scales:
        cols_g = entropy_column_names(g, cfg.feature_name_template)
        for col in cols_g:
            if col not in header:
                raise ValueError(
                    f"Required entropy column '{col}' for scale {g}x{g} missing from source CSV header"
                )
            required_entropy_cols.append(col)

    # Prevent double-append: no output column should already exist in the input header
    asym_names = asymmetry_column_names(cfg.scales, cfg.kinds, cfg.column_templates)
    for col in asym_names:
        if col in header:
            raise ValueError(
                f"Output asymmetry column '{col}' already exists in source CSV header; "
                "cannot double-append"
            )

    return required_entropy_cols


def append_asymmetry_csv(
    cfg: AsymmetryCsvConfig, dry_run: bool = False
) -> AsymmetryExportSummary:
    """Run chunked spatial entropy asymmetry appending pipeline from input CSV to output CSV.

    Args:
        cfg: Validated AsymmetryCsvConfig instance.
        dry_run: When True, validates input CSV, header, and value ranges without writing files.

    Returns:
        AsymmetryExportSummary containing execution statistics.

    Raises:
        FileNotFoundError: If effective input CSV does not exist.
        FileExistsError: If effective output CSV exists and overwrite is False.
        ValueError: If headers or entropy values violate integrity constraints.
    """
    if not cfg.effective_input_csv.is_file():
        raise FileNotFoundError(
            f"Effective input entropy CSV file not found: {cfg.effective_input_csv}"
        )

    if cfg.effective_output_csv.exists() and not cfg.overwrite and not dry_run:
        raise FileExistsError(
            f"Target output CSV already exists: {cfg.effective_output_csv}. "
            "Use overwrite=True or --overwrite to replace it."
        )

    with cfg.effective_input_csv.open("r", encoding="utf-8", newline="") as f_header:
        reader_header = csv.reader(f_header)
        header = next(reader_header, None)
        if header is None:
            raise ValueError(f"Input CSV is empty: {cfg.effective_input_csv}")

    validate_header(header, cfg)

    asym_names = tuple(asymmetry_column_names(cfg.scales, cfg.kinds, cfg.column_templates))
    output_header = list(header) + list(asym_names)
    float_fmt = f"%.{cfg.float_precision}f"

    entropy_cols_by_scale = {
        g: entropy_column_names(g, cfg.feature_name_template) for g in cfg.scales
    }

    start_time = time.perf_counter()
    total_rows = 0

    if dry_run:
        logger.info("Executing dry-run validation for %s", cfg.effective_input_csv)
        all_scale_cols = [c for cols in entropy_cols_by_scale.values() for c in cols]
        usecols = all_scale_cols
        if cfg.levels_column in header:
            usecols = usecols + [cfg.levels_column]

        with pd.read_csv(
            cfg.effective_input_csv,
            usecols=usecols,
            chunksize=cfg.chunk_size,
            dtype=str,
            keep_default_na=False,
        ) as reader:
            for chunk in reader:
                if cfg.levels_column in chunk.columns:
                    first_levels_str = str(chunk[cfg.levels_column].iloc[0])
                    available_levels = {
                        int(x.strip())
                        for x in first_levels_str.split(cfg.levels_separator)
                        if x.strip().isdigit()
                    }
                    for g in cfg.scales:
                        if g not in available_levels:
                            raise ValueError(
                                f"Configured scale {g} not found in '{cfg.levels_column}' "
                                f"value '{first_levels_str}'"
                            )

                for g in cfg.scales:
                    vals_g = chunk[entropy_cols_by_scale[g]].to_numpy(dtype=np.float64)
                    validate_entropy_values(vals_g, tolerance=cfg.value_tolerance)

                total_rows += len(chunk)

        elapsed_s = time.perf_counter() - start_time
        return AsymmetryExportSummary(
            output_csv=cfg.effective_output_csv,
            rows=total_rows,
            columns=len(output_header),
            n_asymmetry_features=len(asym_names),
            asymmetry_feature_names=asym_names,
            bytes_written=0,
            elapsed_s=elapsed_s,
            dry_run=True,
        )

    tmp_output = cfg.effective_output_csv.with_suffix(".csv.tmp")
    tmp_output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with (
            tmp_output.open("w", encoding="utf-8", newline="") as f_out,
            pd.read_csv(
                cfg.effective_input_csv,
                chunksize=cfg.chunk_size,
                dtype=str,
                keep_default_na=False,
            ) as reader,
        ):
            writer = csv.writer(f_out)
            writer.writerow(output_header)

            for chunk in tqdm(
                reader, desc="Appending asymmetry features", unit="chunk"
            ):
                if cfg.levels_column in chunk.columns:
                    first_levels_str = str(chunk[cfg.levels_column].iloc[0])
                    available_levels = {
                        int(x.strip())
                        for x in first_levels_str.split(cfg.levels_separator)
                        if x.strip().isdigit()
                    }
                    for g in cfg.scales:
                        if g not in available_levels:
                            raise ValueError(
                                f"Configured scale {g} not found in '{cfg.levels_column}' "
                                f"value '{first_levels_str}'"
                            )

                entropy_by_scale: dict[int, np.ndarray] = {}
                for g in cfg.scales:
                    vals_g = chunk[entropy_cols_by_scale[g]].to_numpy(dtype=np.float64)
                    validate_entropy_values(vals_g, tolerance=cfg.value_tolerance)
                    entropy_by_scale[g] = vals_g

                asym_matrix = compute_asymmetry_features(
                    entropy_by_scale=entropy_by_scale,
                    scales=cfg.scales,
                    kinds=cfg.kinds,
                    tolerance=cfg.value_tolerance,
                )

                formatted_asym = [
                    [float_fmt % val for val in row] for row in asym_matrix
                ]
                raw_chunk_vals = chunk.to_numpy(dtype=str)

                chunk_rows: list[list[str]] = []
                for orig_row, asym_row in zip(raw_chunk_vals, formatted_asym):
                    chunk_rows.append(list(orig_row) + asym_row)

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
        input_sidecar_path = cfg.effective_input_csv.with_suffix(".json")

        inherited: dict[str, object] = {}
        if input_sidecar_path.is_file():
            try:
                with input_sidecar_path.open("r", encoding="utf-8") as f_in_sidecar:
                    inherited = json.load(f_in_sidecar)
            except Exception as exc:
                logger.warning(
                    "Could not load input sidecar JSON %s: %s", input_sidecar_path, exc
                )
        else:
            logger.warning(
                "Input sidecar JSON not found: %s; creating new sidecar with computed defaults",
                input_sidecar_path,
            )

        existing_feature_cols = inherited.get("feature_columns")
        if isinstance(existing_feature_cols, list):
            feature_cols_list = list(existing_feature_cols) + list(asym_names)
        else:
            # Fallback if input sidecar had no feature_columns list
            base_entropy_names = entropy_column_names(
                cfg.global_scale, cfg.feature_name_template
            )
            for g in cfg.scales:
                base_entropy_names.extend(
                    entropy_column_names(g, cfg.feature_name_template)
                )
            feature_cols_list = base_entropy_names + list(asym_names)

        sidecar_data = dict(inherited)
        sidecar_data.update(
            {
                "source_csv": str(cfg.effective_input_csv),
                "output_csv": str(cfg.effective_output_csv),
                "rows": total_rows,
                "features_count": len(feature_cols_list),
                "feature_columns": feature_cols_list,
                "asymmetry_columns": list(asym_names),
                "asymmetry_scales": list(cfg.scales),
                "total_columns": len(output_header),
                "float_precision": cfg.float_precision,
                "schema": cfg.schema,
                "elapsed_seconds": round(elapsed_s, 4),
                "numpy_version": np.__version__,
                "timing_metric": cfg.sidecar_timing_metric,
            }
        )

        with sidecar_path.open("w", encoding="utf-8") as f_sidecar:
            json.dump(sidecar_data, f_sidecar, indent=2)

    return AsymmetryExportSummary(
        output_csv=cfg.effective_output_csv,
        rows=total_rows,
        columns=len(output_header),
        n_asymmetry_features=len(asym_names),
        asymmetry_feature_names=asym_names,
        bytes_written=bytes_written,
        elapsed_s=elapsed_s,
        dry_run=False,
    )
