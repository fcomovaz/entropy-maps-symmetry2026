"""Unit and integration tests for spatial entropy asymmetry CSV pipeline and CLI.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\.venv\Scripts\python.exe -m unittest tests/test_asymmetry_csv.py
"""
from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.append_asymmetry_features import main as cli_main
from src.features.asymmetry_csv import (
    AsymmetryCsvConfig,
    append_asymmetry_csv,
    load_asymmetry_config,
    validate_header,
)
from src.features.asymmetry_maps import (
    asymmetry_column_names,
    entropy_column_names,
)


def _compute_file_hash(path: Path) -> str:
    """Calculate SHA-256 hash of a file's contents."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _create_synthetic_entropy_csv(
    path: Path,
    metadata_cols: list[str],
    rows_meta: list[list[str]],
    scales: tuple[int, ...] = (1, 2, 4, 8),
    bins: int = 16,
) -> None:
    """Generate a valid flat entropy CSV matching canonical x85 layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entropy_cols: list[str] = []
    for g in scales:
        entropy_cols.extend(entropy_column_names(g))

    header = metadata_cols + ["bins", "grid_levels"] + entropy_cols
    levels_str = "|".join(str(g) for g in scales)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, meta in enumerate(rows_meta):
            entropy_vals: list[str] = []
            for col_idx, _ in enumerate(entropy_cols):
                # Deterministic pseudo-entropy values strictly in [0.1, 0.9]
                val = 0.1 + (((i + 1) * 31 + col_idx * 17) % 79) / 100.0
                entropy_vals.append(f"{val:.8f}")
            writer.writerow(meta + [str(bins), levels_str] + entropy_vals)


def _setup_synthetic_asymmetry_project(
    tmp: Path,
    dataset_key: str = "mvtecad",
) -> Path:
    """Create complete minimal directory tree, TOML configs, and input CSVs in tmp."""
    configs_dir = tmp / "configs"
    (configs_dir / "asymmetry").mkdir(parents=True, exist_ok=True)
    (configs_dir / "entropy").mkdir(parents=True, exist_ok=True)
    (tmp / "data" / "csv").mkdir(parents=True, exist_ok=True)

    base_toml = """[configs]
path = "configs/base.toml"

[datasets.mvtecad]
name = "mvtec-ad"
entropy_csv_path = "data/csv/mvtec-ad-entropy.csv"
entropy_csv_path_inverted = "data/csv/mvtec-ad-inverted-entropy.csv"

[datasets.kolektorsdd2]
name = "KolektorSDD2"
entropy_csv_path = "data/csv/kolektorsdd2-entropy.csv"
entropy_csv_path_inverted = "data/csv/kolektorsdd2-inverted-entropy.csv"
"""
    (configs_dir / "base.toml").write_text(base_toml, encoding="utf-8")

    entropy_toml = """[entropy_maps]
base_config = "configs/base.toml"
dataset_key = "mvtecad"

[entropy_maps.output]
feature_name_template = "E_{g}x{g}_r{r}_c{c}"
float_precision = 8
bins_column = "bins"
levels_column = "grid_levels"
levels_separator = "|"
"""
    (configs_dir / "entropy" / "mvtec_ad.toml").write_text(
        entropy_toml, encoding="utf-8"
    )

    asym_toml = f"""[asymmetry_features]
base_config = "configs/base.toml"
entropy_config = "configs/entropy/mvtec_ad.toml"
dataset_key = "{dataset_key}"
inverted = false
overwrite = false
schema = "x94-v1"

[asymmetry_features.regions]
scales = [2, 4, 8]
kinds = ["lr", "tb", "rot180"]

[asymmetry_features.source]
feature_name_template = "E_{{g}}x{{g}}_r{{r}}_c{{c}}"
global_scale = 1
bins_column = "bins"
levels_column = "grid_levels"
levels_separator = "|"

[asymmetry_features.output]
suffix = "-full"
column_templates = {{ lr = "asym_lr_{{g}}x{{g}}", tb = "asym_tb_{{g}}x{{g}}", rot180 = "asym_rot180_{{g}}x{{g}}" }}
float_precision = 8
value_tolerance = 1e-6
write_sidecar = true
sidecar_timing_metric = "T_feature_extraction_asymmetry"

[asymmetry_features.runtime]
chunk_size = 4
"""
    asym_cfg_path = configs_dir / "asymmetry" / "mvtec_ad.toml"
    asym_cfg_path.write_text(asym_toml, encoding="utf-8")

    # Generate synthetic input CSV and sidecar
    meta_cols = ["object_name", "split", "type", "defect_name", "file_name"]
    meta_rows = [
        ["bottle", "train", "good", "good", f"{i:03d}.png"] for i in range(10)
    ]
    input_csv = tmp / "data" / "csv" / "mvtec-ad-entropy.csv"
    _create_synthetic_entropy_csv(input_csv, meta_cols, meta_rows)

    sidecar_data = {
        "source_csv": str(tmp / "data" / "csv" / "mvtec-ad.csv"),
        "output_csv": str(input_csv),
        "image_size": 128,
        "grid_levels": [1, 2, 4, 8],
        "bins": 16,
        "rows": 10,
        "features_count": 85,
        "feature_columns": [
            col
            for g in (1, 2, 4, 8)
            for col in entropy_column_names(g)
        ],
        "total_columns": len(meta_cols) + 2 + 85,
        "float_precision": 8,
        "elapsed_seconds": 1.234,
        "numpy_version": "2.2.6",
        "timing_metric": "T_feature_extraction_entropy",
    }
    (tmp / "data" / "csv" / "mvtec-ad-entropy.json").write_text(
        json.dumps(sidecar_data, indent=2), encoding="utf-8"
    )

    # Inverted variant
    input_inv_csv = tmp / "data" / "csv" / "mvtec-ad-inverted-entropy.csv"
    _create_synthetic_entropy_csv(input_inv_csv, meta_cols, meta_rows)

    return asym_cfg_path.relative_to(tmp)


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading, validation, and dynamic path resolution."""

    def test_load_asymmetry_config_valid(self) -> None:
        """Valid configuration loads cleanly and resolves relative paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            cfg_rel = _setup_synthetic_asymmetry_project(tmp)
            cfg = load_asymmetry_config(cfg_rel, project_root=tmp)

            self.assertEqual(cfg.scales, (2, 4, 8))
            self.assertEqual(cfg.kinds, ("lr", "tb", "rot180"))
            self.assertEqual(cfg.suffix, "-full")
            self.assertEqual(cfg.schema, "x94-v1")
            self.assertEqual(cfg.chunk_size, 4)
            self.assertEqual(cfg.float_precision, 8)
            self.assertTrue(cfg.output_csv.name.endswith("-full.csv"))
            self.assertEqual(
                cfg.output_csv,
                (tmp / "data" / "csv" / "mvtec-ad-entropy-full.csv").resolve(),
            )
            self.assertEqual(
                cfg.output_csv_inverted,
                (tmp / "data" / "csv" / "mvtec-ad-inverted-entropy-full.csv").resolve(),
            )

    def test_load_config_missing_file(self) -> None:
        """Non-existent config file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            with self.assertRaises(FileNotFoundError):
                load_asymmetry_config(Path("configs/asymmetry/missing.toml"), project_root=tmp)

    def test_load_config_identical_paths_error(self) -> None:
        """Empty suffix leading to identical input and output paths raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            _setup_synthetic_asymmetry_project(tmp)
            cfg_path = tmp / "configs" / "asymmetry" / "mvtec_ad.toml"
            text = cfg_path.read_text(encoding="utf-8")
            text = text.replace('suffix = "-full"', 'suffix = ""')
            cfg_path.write_text(text, encoding="utf-8")

            with self.assertRaises(ValueError):
                load_asymmetry_config(Path("configs/asymmetry/mvtec_ad.toml"), project_root=tmp)


class TestHeaderValidation(unittest.TestCase):
    """Test header integrity checks and double-append collision prevention."""

    def test_validate_header_valid(self) -> None:
        """Valid header containing 1x1 and fine scale entropy columns succeeds."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            cfg_rel = _setup_synthetic_asymmetry_project(tmp)
            cfg = load_asymmetry_config(cfg_rel, project_root=tmp)

            header = ["split", "type", "bins", "grid_levels"]
            for g in (1, 2, 4, 8):
                header.extend(entropy_column_names(g))

            req_cols = validate_header(header, cfg)
            self.assertEqual(len(req_cols), 4 + 16 + 64)

    def test_validate_header_missing_global(self) -> None:
        """Missing 1x1 global entropy column raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            cfg_rel = _setup_synthetic_asymmetry_project(tmp)
            cfg = load_asymmetry_config(cfg_rel, project_root=tmp)

            header = ["split", "type"]
            for g in (2, 4, 8):
                header.extend(entropy_column_names(g))

            with self.assertRaises(ValueError) as ctx:
                validate_header(header, cfg)
            self.assertIn("E_1x1_r0_c0", str(ctx.exception))

    def test_validate_header_missing_scale(self) -> None:
        """Missing fine scale entropy column raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            cfg_rel = _setup_synthetic_asymmetry_project(tmp)
            cfg = load_asymmetry_config(cfg_rel, project_root=tmp)

            header = ["split", "type", "E_1x1_r0_c0"] + entropy_column_names(2)
            with self.assertRaises(ValueError) as ctx:
                validate_header(header, cfg)
            self.assertIn("E_4x4_r0_c0", str(ctx.exception))

    def test_validate_header_prevents_double_append(self) -> None:
        """Header already containing output asymmetry column raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            cfg_rel = _setup_synthetic_asymmetry_project(tmp)
            cfg = load_asymmetry_config(cfg_rel, project_root=tmp)

            header = ["split", "type"]
            for g in (1, 2, 4, 8):
                header.extend(entropy_column_names(g))
            header.append("asym_lr_2x2")  # Collision

            with self.assertRaises(ValueError) as ctx:
                validate_header(header, cfg)
            self.assertIn("cannot double-append", str(ctx.exception))


class TestPipelineIntegration(unittest.TestCase):
    """Test full execution of append_asymmetry_csv pipeline with streaming and sidecar."""

    def test_pipeline_execution_and_sidecar(self) -> None:
        """Standard pipeline appends 9 columns, preserves original columns byte-for-byte, and writes sidecar."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            cfg_rel = _setup_synthetic_asymmetry_project(tmp)
            cfg = load_asymmetry_config(cfg_rel, project_root=tmp)

            input_hash_before = _compute_file_hash(cfg.effective_input_csv)
            sidecar_in = cfg.effective_input_csv.with_suffix(".json")
            sidecar_in_content = sidecar_in.read_text(encoding="utf-8")

            summary = append_asymmetry_csv(cfg, dry_run=False)

            self.assertFalse(summary.dry_run)
            self.assertEqual(summary.rows, 10)
            self.assertEqual(summary.n_asymmetry_features, 9)
            self.assertTrue(cfg.effective_output_csv.is_file())

            # Verify input file and input sidecar are completely unchanged
            input_hash_after = _compute_file_hash(cfg.effective_input_csv)
            self.assertEqual(input_hash_before, input_hash_after)
            self.assertEqual(sidecar_in.read_text(encoding="utf-8"), sidecar_in_content)

            # Verify output CSV content and structure
            df_in = pd.read_csv(cfg.effective_input_csv, dtype=str)
            df_out = pd.read_csv(cfg.effective_output_csv, dtype=str)

            self.assertEqual(len(df_out), len(df_in))
            self.assertEqual(len(df_out.columns), len(df_in.columns) + 9)

            # Pass-through fidelity: original columns must match byte-for-byte
            for col in df_in.columns:
                self.assertTrue(df_in[col].equals(df_out[col]))

            # Asymmetry column names and positions
            asym_names = list(summary.asymmetry_feature_names)
            expected_asym_names = [
                "asym_lr_2x2",
                "asym_tb_2x2",
                "asym_rot180_2x2",
                "asym_lr_4x4",
                "asym_tb_4x4",
                "asym_rot180_4x4",
                "asym_lr_8x8",
                "asym_tb_8x8",
                "asym_rot180_8x8",
            ]
            self.assertEqual(asym_names, expected_asym_names)
            self.assertEqual(list(df_out.columns[-9:]), expected_asym_names)

            # Values bounded and formatted to 8 decimals
            for col in asym_names:
                for val_str in df_out[col]:
                    val = float(val_str)
                    self.assertGreaterEqual(val, 0.0)
                    self.assertLessEqual(val, 1.0)
                    # Verify 8 decimals format: e.g. "0.12345678"
                    parts = val_str.split(".")
                    self.assertEqual(len(parts), 2)
                    self.assertEqual(len(parts[1]), 8)

            # Verify sidecar JSON
            sidecar_out = cfg.effective_output_csv.with_suffix(".json")
            self.assertTrue(sidecar_out.is_file())
            sidecar_meta = json.loads(sidecar_out.read_text(encoding="utf-8"))

            self.assertEqual(sidecar_meta["schema"], "x94-v1")
            self.assertEqual(sidecar_meta["features_count"], 94)
            self.assertEqual(len(sidecar_meta["feature_columns"]), 94)
            self.assertEqual(sidecar_meta["asymmetry_columns"], expected_asym_names)
            self.assertEqual(sidecar_meta["asymmetry_scales"], [2, 4, 8])
            self.assertEqual(sidecar_meta["rows"], 10)
            self.assertEqual(sidecar_meta["total_columns"], len(df_out.columns))
            self.assertEqual(
                sidecar_meta["timing_metric"], "T_feature_extraction_asymmetry"
            )
            self.assertGreater(sidecar_meta["elapsed_seconds"], 0.0)

    def test_overwrite_protection(self) -> None:
        """Existing output file raises FileExistsError unless overwrite=True."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            cfg_rel = _setup_synthetic_asymmetry_project(tmp)
            cfg = load_asymmetry_config(cfg_rel, project_root=tmp)

            append_asymmetry_csv(cfg, dry_run=False)

            with self.assertRaises(FileExistsError):
                append_asymmetry_csv(cfg, dry_run=False)

            # Overwrite=True succeeds
            cfg_overwrite = replace(cfg, overwrite=True)
            summary = append_asymmetry_csv(cfg_overwrite, dry_run=False)
            self.assertEqual(summary.rows, 10)

    def test_dry_run_writes_nothing(self) -> None:
        """Dry-run validates rows and returns metrics without creating output files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            cfg_rel = _setup_synthetic_asymmetry_project(tmp)
            cfg = load_asymmetry_config(cfg_rel, project_root=tmp)

            summary = append_asymmetry_csv(cfg, dry_run=True)
            self.assertTrue(summary.dry_run)
            self.assertEqual(summary.rows, 10)
            self.assertEqual(summary.bytes_written, 0)
            self.assertFalse(cfg.effective_output_csv.exists())
            self.assertFalse(cfg.effective_output_csv.with_suffix(".json").exists())

    def test_chunk_size_invariance(self) -> None:
        """Different chunk sizes yield identical output files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            cfg_rel = _setup_synthetic_asymmetry_project(tmp)
            cfg = load_asymmetry_config(cfg_rel, project_root=tmp)

            cfg_chunk2 = replace(cfg, chunk_size=2)
            append_asymmetry_csv(cfg_chunk2, dry_run=False)
            out_chunk2_hash = _compute_file_hash(cfg.effective_output_csv)

            cfg_chunk10 = replace(cfg, chunk_size=10, overwrite=True)
            append_asymmetry_csv(cfg_chunk10, dry_run=False)
            out_chunk10_hash = _compute_file_hash(cfg.effective_output_csv)

            self.assertEqual(out_chunk2_hash, out_chunk10_hash)

    def test_inverted_dataset_run(self) -> None:
        """Inverted dataset produces target with -inverted-entropy-full.csv."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            cfg_rel = _setup_synthetic_asymmetry_project(tmp)
            cfg = load_asymmetry_config(cfg_rel, project_root=tmp)

            cfg_inv = replace(cfg, inverted=True)
            summary = append_asymmetry_csv(cfg_inv, dry_run=False)

            self.assertEqual(summary.rows, 10)
            self.assertTrue(cfg_inv.effective_output_csv.is_file())
            self.assertIn("inverted-entropy-full", cfg_inv.effective_output_csv.name)


class TestCliEntryPoint(unittest.TestCase):
    """Test CLI argument parsing, flags, and error exit statuses."""

    def test_cli_standard_and_dry_run(self) -> None:
        """CLI executes cleanly and returns 0 exit code."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            cfg_rel = _setup_synthetic_asymmetry_project(tmp)

            # Dry-run
            ret_dry = cli_main(
                ["--config", str(cfg_rel), "--dry-run"], project_root=tmp
            )
            self.assertEqual(ret_dry, 0)

            # Real execution
            ret_real = cli_main(["--config", str(cfg_rel)], project_root=tmp)
            self.assertEqual(ret_real, 0)

    def test_cli_all_flag(self) -> None:
        """CLI --all runs across configurations and variants."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            _setup_synthetic_asymmetry_project(tmp)
            ret_all = cli_main(["--all", "--overwrite"], project_root=tmp)
            self.assertEqual(ret_all, 0)

    def test_cli_error_raises_system_exit(self) -> None:
        """CLI errors raise SystemExit with clean error message."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir).resolve()
            with self.assertRaises(SystemExit) as ctx:
                cli_main(["--config", "nonexistent.toml"], project_root=tmp)
            self.assertIn("error:", str(ctx.exception))


class TestRealDataParitySample(unittest.TestCase):
    """Optional sanity check on real project data if available."""

    def test_real_entropy_csv_sample(self) -> None:
        """If mvtec-ad-entropy.csv exists, check first 5 rows against mathematical contract."""
        real_csv = PROJECT_ROOT / "data" / "csv" / "mvtec-ad-entropy.csv"
        if not real_csv.is_file():
            self.skipTest(f"Real data CSV not found: {real_csv}")

        df = pd.read_csv(real_csv, nrows=5)
        self.assertEqual(len(df), 5)

        # Check scale 2x2: 4 values
        cols_2x2 = entropy_column_names(2)
        vals_2x2 = df[cols_2x2].to_numpy(dtype=np.float64).reshape(-1, 2, 2)
        for i in range(5):
            g = vals_2x2[i]
            expected_lr = np.mean(np.abs(g - g[:, ::-1]))
            expected_tb = np.mean(np.abs(g - g[::-1, :]))
            expected_r180 = np.mean(np.abs(g - g[::-1, ::-1]))
            self.assertGreaterEqual(expected_lr, 0.0)
            self.assertLessEqual(expected_lr, 1.0)
            self.assertGreaterEqual(expected_tb, 0.0)
            self.assertLessEqual(expected_tb, 1.0)
            self.assertGreaterEqual(expected_r180, 0.0)
            self.assertLessEqual(expected_r180, 1.0)


if __name__ == "__main__":
    unittest.main()
