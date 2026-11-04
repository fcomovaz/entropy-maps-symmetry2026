"""Unit and integration tests for visual region entropy CSV pipeline and CLI.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_entropy_csv.py
or:
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_entropy_csv.py -q
"""
from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.export_entropy_maps import main as cli_main
from src.features.entropy_csv import (
    EntropyCsvConfig,
    export_entropy_csv,
    load_entropy_config,
    validate_header,
)


def _create_synthetic_csv(
    path: Path,
    metadata_cols: list[str],
    rows_meta: list[list[str]],
    size: int,
    invert: bool = False,
) -> None:
    """Generate a valid flat pixel CSV with given metadata schema and dimensions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    px_cols = [f"px_{i:05d}" for i in range(size * size)]
    header = metadata_cols + px_cols + ["width", "height"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, meta in enumerate(rows_meta):
            base_val = ((i * 37) % 256)
            if invert:
                base_val = 255 - base_val
            px_values = [(base_val + px_idx) % 256 for px_idx in range(size * size)]
            writer.writerow(meta + px_values + [str(size), str(size)])


def _setup_synthetic_project(
    tmp: Path,
    dataset_key: str = "mvtecad",
    size: int = 16,
    grid_levels: list[int] | None = None,
    bins: int = 16,
) -> Path:
    """Create complete minimal directory tree and TOML configuration files in tmp.

    Returns the relative path to the entropy task configuration.
    """
    if grid_levels is None:
        grid_levels = [1, 2, 4]

    configs_dir = tmp / "configs"
    (configs_dir / "entropy").mkdir(parents=True, exist_ok=True)
    (configs_dir / "preprocessing").mkdir(parents=True, exist_ok=True)

    base_toml = f"""[configs]
path = "configs/base.toml"

[datasets.mvtecad]
name = "mvtec-ad"
input_path = "data/raw/mvtec-ad"
csv_path = "data/csv/mvtec-ad.csv"
csv_path_inverted = "data/csv/mvtec-ad-inverted.csv"
entropy_csv_path = "data/csv/mvtec-ad-entropy.csv"
entropy_csv_path_inverted = "data/csv/mvtec-ad-inverted-entropy.csv"
output_path = "data/preprocessed/mvtec-ad"
output_path_inverted = "data/preprocessed/mvtec-ad-inverted"
subdatasets = ["bottle"]

[datasets.kolektorsdd2]
name = "KolektorSDD2"
input_path = "data/raw/KolektorSDD2"
csv_path = "data/csv/kolektorsdd2.csv"
csv_path_inverted = "data/csv/kolektorsdd2-inverted.csv"
entropy_csv_path = "data/csv/kolektorsdd2-entropy.csv"
entropy_csv_path_inverted = "data/csv/kolektorsdd2-inverted-entropy.csv"
output_path = "data/preprocessed/KolektorSDD2"
output_path_inverted = "data/preprocessed/KolektorSDD2-inverted"
splits = ["train", "test"]
"""
    (configs_dir / "base.toml").write_text(base_toml, encoding="utf-8")

    prep_toml = f"""[preprocessing]
base_config = "configs/base.toml"
dataset_key = "{dataset_key}"
target_size = {size}
pad_value = 0
padding_anchor = "center"
image_extensions = [".png"]
overwrite = false
"""
    (configs_dir / f"preprocessing/{dataset_key}.toml").write_text(prep_toml, encoding="utf-8")

    if dataset_key == "mvtecad":
        meta_cols = '["object_name", "split", "type", "defect_name", "file_name"]'
    else:
        meta_cols = '["split", "type", "file_name"]'

    task_toml = f"""[entropy_maps]
base_config = "configs/base.toml"
dataset_key = "{dataset_key}"
preprocessing_config = "configs/preprocessing/{dataset_key}.toml"
inverted = false
overwrite = false

[entropy_maps.regions]
grid_levels = {grid_levels}
bins = {bins}

[entropy_maps.source]
pixel_column_prefix = "px_"
metadata_columns = {meta_cols}
size_columns = ["width", "height"]

[entropy_maps.output]
feature_name_template = "E_{{g}}x{{g}}_r{{r}}_c{{c}}"
float_precision = 8
write_sidecar = true
bins_column = "bins"
levels_column = "grid_levels"
levels_separator = "|"

[entropy_maps.runtime]
chunk_size = 4
"""
    task_config_rel = Path(f"configs/entropy/{dataset_key}.toml")
    (tmp / task_config_rel).write_text(task_toml, encoding="utf-8")
    return task_config_rel


class TestEntropyConfigLoading(unittest.TestCase):
    """Test configuration loading, validation, and path resolution."""

    def test_load_entropy_config_mvtec(self) -> None:
        """Standard MVTec config loads and resolves all paths anchored to project_root."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            cfg_rel = _setup_synthetic_project(tmp, "mvtecad", size=16, grid_levels=[1, 2, 4], bins=16)
            cfg = load_entropy_config(cfg_rel, tmp)

            self.assertEqual(cfg.image_size, 16)
            self.assertEqual(cfg.grid_levels, (1, 2, 4))
            self.assertEqual(cfg.bins, 16)
            self.assertEqual(cfg.chunk_size, 4)
            self.assertEqual(cfg.float_precision, 8)
            self.assertEqual(cfg.bins_column, "bins")
            self.assertEqual(cfg.levels_column, "grid_levels")
            self.assertEqual(cfg.levels_separator, "|")
            self.assertEqual(cfg.effective_input_csv, (tmp / "data/csv/mvtec-ad.csv").resolve())
            self.assertEqual(cfg.effective_output_csv, (tmp / "data/csv/mvtec-ad-entropy.csv").resolve())

    def test_load_entropy_config_missing_file(self) -> None:
        """Non-existent config file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            with self.assertRaises(FileNotFoundError):
                load_entropy_config(Path("non_existent.toml"), tmp)

    def test_load_entropy_config_missing_key(self) -> None:
        """Missing required section or key raises KeyError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_rel = _setup_synthetic_project(tmp, "mvtecad")
            full_path = tmp / cfg_rel
            content = full_path.read_text(encoding="utf-8").replace("grid_levels = [1, 2, 4]", "")
            full_path.write_text(content, encoding="utf-8")
            with self.assertRaises(KeyError):
                load_entropy_config(cfg_rel, tmp)


class TestHeaderValidation(unittest.TestCase):
    """Test source CSV header structural validation."""

    def test_validate_header_valid(self) -> None:
        """Complete, ordered, contiguous header passes validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_rel = _setup_synthetic_project(tmp, "mvtecad", size=4)
            cfg = load_entropy_config(cfg_rel, tmp)
            px_cols = [f"px_{i:05d}" for i in range(16)]
            header = list(cfg.metadata_columns) + px_cols + list(cfg.size_columns)
            validate_header(header, cfg)

    def test_validate_header_missing_metadata(self) -> None:
        """Missing metadata column raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_rel = _setup_synthetic_project(tmp, "mvtecad", size=4)
            cfg = load_entropy_config(cfg_rel, tmp)
            px_cols = [f"px_{i:05d}" for i in range(16)]
            header = ["object_name", "split"] + px_cols + list(cfg.size_columns)
            with self.assertRaises(ValueError):
                validate_header(header, cfg)

    def test_validate_header_pixel_count_mismatch(self) -> None:
        """Insufficient pixel column count raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_rel = _setup_synthetic_project(tmp, "mvtecad", size=4)
            cfg = load_entropy_config(cfg_rel, tmp)
            px_cols = [f"px_{i:05d}" for i in range(15)]  # requires 16
            header = list(cfg.metadata_columns) + px_cols + list(cfg.size_columns)
            with self.assertRaises(ValueError):
                validate_header(header, cfg)


class TestEntropyCsvPipeline(unittest.TestCase):
    """Integration tests for streaming extraction, output formatting, and sidecar emission."""

    def test_export_mvtec_ad_synthetic(self) -> None:
        """MVTec synthetic pipeline produces exact expected schema, values, and sidecar."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_rel = _setup_synthetic_project(tmp, "mvtecad", size=16, grid_levels=[1, 2, 4], bins=16)
            cfg = load_entropy_config(cfg_rel, tmp)

            meta_rows = [
                ["bottle", "train", "good", "good", "000.png"],
                ["bottle", "train", "good", "good", "001.png"],
                ["bottle", "test", "defect", "broken_large", "000.png"],
            ]
            _create_synthetic_csv(
                cfg.effective_input_csv,
                list(cfg.metadata_columns),
                meta_rows,
                size=16,
            )

            summary = export_entropy_csv(cfg)
            self.assertEqual(summary.rows, 3)
            self.assertEqual(summary.n_features, 1 * 1 + 2 * 2 + 4 * 4)  # 21 features
            self.assertTrue(cfg.effective_output_csv.is_file())

            # Read and verify exported CSV
            with cfg.effective_output_csv.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            expected_header = (
                list(cfg.metadata_columns)
                + ["bins", "grid_levels"]
                + [f"E_{g}x{g}_r{r}_c{c}" for g in (1, 2, 4) for r in range(g) for c in range(g)]
            )
            self.assertEqual(header, expected_header)
            self.assertEqual(len(rows), 3)

            for idx, r in enumerate(rows):
                # Verify metadata preserved
                self.assertEqual(r[:5], meta_rows[idx])
                # Verify configuration columns
                self.assertEqual(r[5], "16")
                self.assertEqual(r[6], "1|2|4")
                # Verify feature values formatted to 8 decimals in [0.0, 1.0]
                for val_str in r[7:]:
                    val = float(val_str)
                    self.assertTrue(0.0 <= val <= 1.0)
                    self.assertEqual(len(val_str.split(".")[1]), 8)

            # Verify sidecar JSON
            sidecar_path = cfg.effective_output_csv.with_suffix(".json")
            self.assertTrue(sidecar_path.is_file())
            sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            self.assertEqual(sidecar_data["rows"], 3)
            self.assertEqual(sidecar_data["features_count"], 21)
            self.assertEqual(sidecar_data["grid_levels"], [1, 2, 4])
            self.assertEqual(sidecar_data["bins"], 16)

    def test_export_kolektor_synthetic(self) -> None:
        """Kolektor synthetic pipeline produces 3 metadata cols + config cols + features."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_rel = _setup_synthetic_project(tmp, "kolektorsdd2", size=16, grid_levels=[1, 2], bins=8)
            cfg = load_entropy_config(cfg_rel, tmp)

            meta_rows = [
                ["train", "good", "10001.png"],
                ["train", "defect", "10002.png"],
            ]
            _create_synthetic_csv(
                cfg.effective_input_csv,
                list(cfg.metadata_columns),
                meta_rows,
                size=16,
            )

            summary = export_entropy_csv(cfg)
            self.assertEqual(summary.rows, 2)
            self.assertEqual(summary.n_features, 5)

            with cfg.effective_output_csv.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            expected_header = ["split", "type", "file_name", "bins", "grid_levels"] + [
                f"E_{g}x{g}_r{r}_c{c}" for g in (1, 2) for r in range(g) for c in range(g)
            ]
            self.assertEqual(header, expected_header)
            self.assertEqual(rows[0][3], "8")
            self.assertEqual(rows[0][4], "1|2")

    def test_inverted_variant(self) -> None:
        """Inverted variant reads inverted CSV and writes to inverted entropy output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_rel = _setup_synthetic_project(tmp, "mvtecad", size=16)
            cfg = load_entropy_config(cfg_rel, tmp)
            cfg = replace(cfg, inverted=True)

            meta_rows = [["bottle", "train", "good", "good", "000.png"]]
            _create_synthetic_csv(
                cfg.effective_input_csv,
                list(cfg.metadata_columns),
                meta_rows,
                size=16,
                invert=True,
            )

            summary = export_entropy_csv(cfg)
            self.assertEqual(summary.rows, 1)
            self.assertTrue(cfg.output_csv_inverted.is_file())
            self.assertFalse(cfg.output_csv.exists())

    def test_overwrite_protection(self) -> None:
        """Existing output file raises FileExistsError unless overwrite is True."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_rel = _setup_synthetic_project(tmp, "mvtecad", size=16)
            cfg = load_entropy_config(cfg_rel, tmp)

            meta_rows = [["bottle", "train", "good", "good", "000.png"]]
            _create_synthetic_csv(
                cfg.effective_input_csv,
                list(cfg.metadata_columns),
                meta_rows,
                size=16,
            )

            # First run writes file
            export_entropy_csv(cfg)

            # Second run with overwrite=False fails
            with self.assertRaises(FileExistsError):
                export_entropy_csv(cfg)

            # Run with overwrite=True succeeds
            cfg_overwrite = replace(cfg, overwrite=True)
            summary = export_entropy_csv(cfg_overwrite)
            self.assertEqual(summary.rows, 1)

    def test_dry_run_no_files_written(self) -> None:
        """Dry-run validates rows and returns summary without writing output or sidecar."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_rel = _setup_synthetic_project(tmp, "mvtecad", size=16)
            cfg = load_entropy_config(cfg_rel, tmp)

            meta_rows = [["bottle", "train", "good", "good", "000.png"]]
            _create_synthetic_csv(
                cfg.effective_input_csv,
                list(cfg.metadata_columns),
                meta_rows,
                size=16,
            )

            summary = export_entropy_csv(cfg, dry_run=True)
            self.assertTrue(summary.dry_run)
            self.assertEqual(summary.rows, 1)
            self.assertFalse(cfg.effective_output_csv.exists())
            self.assertFalse(cfg.effective_output_csv.with_suffix(".json").exists())

    def test_atomic_failure_cleanup(self) -> None:
        """Exception during streaming chunk write leaves no temporary files on disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_rel = _setup_synthetic_project(tmp, "mvtecad", size=16)
            cfg = load_entropy_config(cfg_rel, tmp)

            # Write corrupted CSV with mismatched dimensions
            path = cfg.effective_input_csv
            path.parent.mkdir(parents=True, exist_ok=True)
            px_cols = [f"px_{i:05d}" for i in range(256)]
            header = list(cfg.metadata_columns) + px_cols + ["width", "height"]
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                # First valid row
                writer.writerow(["bottle", "train", "good", "good", "000.png"] + [0] * 256 + ["16", "16"])
                # Second row with wrong size to trigger ValueError inside chunk loop
                writer.writerow(["bottle", "train", "good", "good", "001.png"] + [0] * 256 + ["32", "32"])

            tmp_file = cfg.effective_output_csv.with_suffix(".csv.tmp")
            with self.assertRaises(ValueError):
                export_entropy_csv(cfg)

            self.assertFalse(tmp_file.exists())
            self.assertFalse(cfg.effective_output_csv.exists())


class TestCliIntegration(unittest.TestCase):
    """Test CLI entry point flags, overrides, and exit codes."""

    def test_cli_main_success(self) -> None:
        """CLI main executes successfully with injected project root and returns 0."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_rel = _setup_synthetic_project(tmp, "mvtecad", size=16)
            cfg = load_entropy_config(cfg_rel, tmp)

            meta_rows = [["bottle", "train", "good", "good", "000.png"]]
            _create_synthetic_csv(
                cfg.effective_input_csv,
                list(cfg.metadata_columns),
                meta_rows,
                size=16,
            )

            exit_code = cli_main(["--config", str(cfg_rel)], project_root=tmp)
            self.assertEqual(exit_code, 0)
            self.assertTrue(cfg.effective_output_csv.is_file())

    def test_cli_main_dry_run(self) -> None:
        """CLI dry-run returns exit code 0 without creating files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cfg_rel = _setup_synthetic_project(tmp, "mvtecad", size=16)
            cfg = load_entropy_config(cfg_rel, tmp)

            meta_rows = [["bottle", "train", "good", "good", "000.png"]]
            _create_synthetic_csv(
                cfg.effective_input_csv,
                list(cfg.metadata_columns),
                meta_rows,
                size=16,
            )

            exit_code = cli_main(["--config", str(cfg_rel), "--dry-run"], project_root=tmp)
            self.assertEqual(exit_code, 0)
            self.assertFalse(cfg.effective_output_csv.exists())

    def test_cli_main_error_exit(self) -> None:
        """Missing config file raises SystemExit with non-zero exit."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            with self.assertRaises(SystemExit) as cm:
                cli_main(["--config", "non_existent.toml"], project_root=tmp)
            self.assertIn("error:", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
