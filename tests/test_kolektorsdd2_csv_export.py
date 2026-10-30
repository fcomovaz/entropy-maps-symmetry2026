"""Unit and integration tests for KolektorSDD2 dataset CSV export.

Manual execution commands:
    .venv\\Scripts\\python.exe -m unittest tests/test_kolektorsdd2_csv_export.py
or:
    .venv\\Scripts\\python.exe -m pytest tests/test_kolektorsdd2_csv_export.py -q
"""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.export_kolektorsdd2_csv import main as cli_main
from src.datasets.kolektorsdd2_csv import (
    METADATA_COLUMNS,
    SIZE_COLUMNS,
    KolektorCsvConfig,
    KolektorRecord,
    build_header,
    classify_type,
    export_kolektor_csv,
    iter_records,
    load_kolektor_csv_config,
    record_to_row,
)
from src.datasets.mvtec_ad_csv import (
    load_pixels,
    pixel_column_names,
)


def _write_png(path: Path, array: np.ndarray, mode: str = "L") -> None:
    """Helper to write a 2D numpy array to PNG with specified mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path)


def _make_kolektor_dataset(root: Path, size: int, invert_offset: int = 0) -> None:
    """Populate a synthetic KolektorSDD2 directory structure under root."""
    base_arr = np.arange(size * size, dtype=np.int32).reshape(size, size)
    if invert_offset > 0:
        base_arr = 255 - base_arr

    # train
    _write_png(root / "train/good/10001.png", ((base_arr + 1) % 256).astype(np.uint8))
    _write_png(root / "train/defect/10002.png", ((base_arr + 2) % 256).astype(np.uint8))

    # test
    _write_png(root / "test/good/10003.png", ((base_arr + 3) % 256).astype(np.uint8))
    _write_png(root / "test/defect/10004.png", ((base_arr + 4) % 256).astype(np.uint8))


def _make_project(tmp: Path, size: int = 4, **csv_overrides: object) -> Path:
    """Build complete temporary repository config and dataset hierarchy.

    Returns the relative path to configs/csv/kolektorsdd2.toml.
    """
    configs_dir = tmp / "configs"
    (configs_dir / "csv").mkdir(parents=True, exist_ok=True)
    (configs_dir / "preprocessing").mkdir(parents=True, exist_ok=True)

    base_toml_content = """[configs]
path = "configs/base.toml"

[datasets.kolektorsdd2]
name = "KolektorSDD2"
input_path = "data/raw/KolektorSDD2"
csv_path = "data/csv/kolektorsdd2.csv"
csv_path_inverted = "data/csv/kolektorsdd2-inverted.csv"
output_path = "data/preprocessed/KolektorSDD2"
output_path_inverted = "data/preprocessed/KolektorSDD2-inverted"
splits = ["train", "test"]
"""
    (configs_dir / "base.toml").write_text(base_toml_content, encoding="utf-8")

    prep_toml_content = f"""[preprocessing]
base_config = "configs/base.toml"
dataset_key = "kolektorsdd2"
target_size = {size}
pad_value = 0
padding_anchor = "center"
overwrite = false
"""
    (configs_dir / "preprocessing/kolektorsdd2.toml").write_text(
        prep_toml_content, encoding="utf-8"
    )

    digits = csv_overrides.get("index_digits", 5)
    inverted = "true" if csv_overrides.get("inverted", False) else "false"
    overwrite = "true" if csv_overrides.get("overwrite", False) else "false"
    splits = csv_overrides.get("splits", '["train", "test"]')

    csv_toml_content = f"""[csv]
base_config = "configs/base.toml"
dataset_key = "kolektorsdd2"
preprocessing_config = "configs/preprocessing/kolektorsdd2.toml"
inverted = {inverted}
overwrite = {overwrite}
image_extensions = [".png"]
splits = {splits}

[csv.labels]
good_folder = "good"
good = "good"
defect = "defect"

[csv.pixels]
column_prefix = "px_"
index_digits = {digits}
"""
    csv_toml_path = configs_dir / "csv/kolektorsdd2.toml"
    csv_toml_path.write_text(csv_toml_content, encoding="utf-8")

    _make_kolektor_dataset(
        tmp / "data/preprocessed/KolektorSDD2", size=size, invert_offset=0
    )
    _make_kolektor_dataset(
        tmp / "data/preprocessed/KolektorSDD2-inverted", size=size, invert_offset=1
    )

    return Path("configs/csv/kolektorsdd2.toml")


class TestKolektorCsvConfigAndLoader(unittest.TestCase):
    """Test suite for KolektorSDD2 CSV export configuration loading and validation."""

    def test_load_config_resolves_paths_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg = load_kolektor_csv_config(rel_toml, tmp)

            self.assertEqual(cfg.image_size, 4)
            self.assertEqual(
                cfg.effective_source_root,
                (tmp / "data/preprocessed/KolektorSDD2").resolve(),
            )
            self.assertEqual(
                cfg.effective_output_csv,
                (tmp / "data/csv/kolektorsdd2.csv").resolve(),
            )
            self.assertEqual(cfg.splits, ("train", "test"))

    def test_inverted_switches_source_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg = replace(load_kolektor_csv_config(rel_toml, tmp), inverted=True)

            self.assertTrue(str(cfg.effective_source_root).endswith("KolektorSDD2-inverted"))
            self.assertEqual(cfg.effective_output_csv.name, "kolektorsdd2-inverted.csv")
            self.assertNotEqual(cfg.output_csv, cfg.output_csv_inverted)

    def test_missing_required_key_raises_keyerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            toml_file = tmp / rel_toml
            # Remove index_digits line
            lines = [
                line
                for line in toml_file.read_text(encoding="utf-8").splitlines()
                if "index_digits" not in line
            ]
            toml_file.write_text("\n".join(lines), encoding="utf-8")

            with self.assertRaises(KeyError):
                load_kolektor_csv_config(rel_toml, tmp)

    def test_digits_too_small_raises_valueerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            # size=4 requires at least 2 digits (indices 0..15)
            rel_toml = _make_project(tmp, size=4, index_digits=1)
            with self.assertRaises(ValueError):
                load_kolektor_csv_config(rel_toml, tmp)

    def test_missing_source_root_raises_filenotfound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            shutil.rmtree(tmp / "data/preprocessed/KolektorSDD2")

            with self.assertRaises(FileNotFoundError):
                load_kolektor_csv_config(rel_toml, tmp)

    def test_missing_split_directory_raises_filenotfound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            shutil.rmtree(tmp / "data/preprocessed/KolektorSDD2/train")

            with self.assertRaises(FileNotFoundError):
                load_kolektor_csv_config(rel_toml, tmp)

    def test_empty_splits_raises_valueerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4, splits="[]")
            with self.assertRaises(ValueError):
                load_kolektor_csv_config(rel_toml, tmp)

    def test_same_output_paths_raise_valueerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            base_file = tmp / "configs/base.toml"
            content = base_file.read_text(encoding="utf-8")
            content = content.replace(
                'csv_path_inverted = "data/csv/kolektorsdd2-inverted.csv"',
                'csv_path_inverted = "data/csv/kolektorsdd2.csv"',
            )
            base_file.write_text(content, encoding="utf-8")

            with self.assertRaises(ValueError):
                load_kolektor_csv_config(rel_toml, tmp)

    def test_non_csv_extension_raises_valueerror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            base_file = tmp / "configs/base.toml"
            content = base_file.read_text(encoding="utf-8")
            content = content.replace(
                'csv_path = "data/csv/kolektorsdd2.csv"',
                'csv_path = "data/csv/kolektorsdd2.txt"',
            )
            base_file.write_text(content, encoding="utf-8")

            with self.assertRaises(ValueError):
                load_kolektor_csv_config(rel_toml, tmp)


class TestDiscoveryAndLabeling(unittest.TestCase):
    """Test suite for KolektorSDD2 image discovery, ordering, and classification."""

    def test_classify_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg = load_kolektor_csv_config(rel_toml, tmp)

            self.assertEqual(classify_type("good", cfg), "good")
            self.assertEqual(classify_type("defect", cfg), "defect")
            self.assertEqual(classify_type("unknown", cfg), "defect")

    def test_iter_records_order_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg = load_kolektor_csv_config(rel_toml, tmp)

            got = [
                (r.split, r.type, r.file_name)
                for r in iter_records(cfg)
            ]
            self.assertEqual(
                got,
                [
                    ("train", "defect", "10002.png"),
                    ("train", "good", "10001.png"),
                    ("test", "defect", "10004.png"),
                    ("test", "good", "10003.png"),
                ],
            )

    def test_non_image_files_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg = load_kolektor_csv_config(rel_toml, tmp)

            # Insert stray files in split and class directories
            stray1 = tmp / "data/preprocessed/KolektorSDD2/train/good/Thumbs.db"
            stray2 = tmp / "data/preprocessed/KolektorSDD2/train/good/notes.txt"
            stray3 = tmp / "data/preprocessed/KolektorSDD2/train/stray_file.txt"
            stray1.write_bytes(b"stray")
            stray2.write_text("text", encoding="utf-8")
            stray3.write_text("root_stray", encoding="utf-8")

            records = list(iter_records(cfg))
            self.assertEqual(len(records), 4)

    def test_missing_split_folder_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg = load_kolektor_csv_config(rel_toml, tmp)

            shutil.rmtree(tmp / "data/preprocessed/KolektorSDD2/test")

            with self.assertRaises(FileNotFoundError):
                list(iter_records(cfg))


class TestHeaderAndRowBuilding(unittest.TestCase):
    """Test suite for column names, pixel loading, and row composition."""

    def test_pixel_column_names_128(self) -> None:
        names = pixel_column_names(128 * 128, "px_", 5)
        self.assertEqual(
            (len(names), names[0], names[-1]),
            (16384, "px_00000", "px_16383"),
        )

    def test_header_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg = load_kolektor_csv_config(rel_toml, tmp)

            h = build_header(cfg)
            self.assertEqual(h[:3], ["split", "type", "file_name"])
            self.assertEqual(h[3], "px_00000")
            self.assertEqual(h[18], "px_00015")
            self.assertEqual(h[-2:], ["width", "height"])
            self.assertEqual(len(h), 3 + 16 + 2)

    def test_row_is_row_major(self) -> None:
        px = np.arange(16, dtype=np.uint8).reshape(4, 4)
        px[1, 2] = 200
        rec = KolektorRecord(
            split="train",
            type="defect",
            file_name="10002.png",
            path=Path("dummy"),
        )
        row = record_to_row(rec, px)
        # 3 metadata cols, then row 1, col 2 => index 3 + 1*4 + 2 = 9
        self.assertEqual(row[3 + 1 * 4 + 2], 200)
        self.assertEqual(row[:3], ["train", "defect", "10002.png"])
        self.assertEqual(row[-2:], [4, 4])

    def test_load_pixels_rejects_wrong_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "wrong_size.png"
            _write_png(img_path, np.zeros((5, 5), dtype=np.uint8))
            with self.assertRaises(ValueError) as ctx:
                load_pixels(img_path, image_size=4)
            self.assertIn(str(img_path), str(ctx.exception))

    def test_load_pixels_rejects_non_L_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "rgb.png"
            rgb = np.zeros((4, 4, 3), dtype=np.uint8)
            _write_png(img_path, rgb, mode="RGB")
            with self.assertRaises(ValueError):
                load_pixels(img_path, image_size=4)


class TestStreamingWriterAndExport(unittest.TestCase):
    """Test suite for export_kolektor_csv atomic writes, overwrite rules, and dry-run."""

    def test_export_writes_expected_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg = load_kolektor_csv_config(rel_toml, tmp)

            s = export_kolektor_csv(cfg)
            self.assertEqual(s.rows, 4)
            self.assertEqual(s.counts[("train", "good")], 1)
            self.assertEqual(s.counts[("train", "defect")], 1)
            self.assertEqual(s.counts[("test", "good")], 1)
            self.assertEqual(s.counts[("test", "defect")], 1)
            self.assertTrue(s.output_csv.is_file())
            self.assertFalse(s.output_csv.with_suffix(".csv.tmp").exists())

            with s.output_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

            self.assertEqual(rows[0], build_header(cfg))
            self.assertEqual(len(rows) - 1, 4)

            # First row in discovery is train/defect/10002.png
            src_img = tmp / "data/preprocessed/KolektorSDD2/train/defect/10002.png"
            expected_px = np.asarray(Image.open(src_img)).ravel().tolist()
            self.assertEqual(rows[1][:3], ["train", "defect", "10002.png"])
            self.assertEqual([int(v) for v in rows[1][3:-2]], expected_px)
            self.assertEqual([int(v) for v in rows[1][-2:]], [4, 4])

    def test_inverted_export_reads_inverted_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg_inv = replace(load_kolektor_csv_config(rel_toml, tmp), inverted=True)

            s_inv = export_kolektor_csv(cfg_inv)
            self.assertTrue(s_inv.output_csv.name.endswith("-inverted.csv"))

            with s_inv.output_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))

            src_inv_img = (
                tmp / "data/preprocessed/KolektorSDD2-inverted/train/defect/10002.png"
            )
            expected_inv = np.asarray(Image.open(src_inv_img)).ravel().tolist()
            self.assertEqual([int(v) for v in rows[1][3:-2]], expected_inv)

    def test_existing_output_without_overwrite_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg = load_kolektor_csv_config(rel_toml, tmp)

            cfg.effective_output_csv.parent.mkdir(parents=True, exist_ok=True)
            cfg.effective_output_csv.write_text("keep_content", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                export_kolektor_csv(cfg)

            self.assertEqual(
                cfg.effective_output_csv.read_text(encoding="utf-8"), "keep_content"
            )

    def test_overwrite_replaces_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg = replace(load_kolektor_csv_config(rel_toml, tmp), overwrite=True)

            cfg.effective_output_csv.parent.mkdir(parents=True, exist_ok=True)
            cfg.effective_output_csv.write_text("old", encoding="utf-8")

            s = export_kolektor_csv(cfg)
            self.assertEqual(s.rows, 4)
            self.assertNotEqual(
                cfg.effective_output_csv.read_text(encoding="utf-8"), "old"
            )

    def test_bad_image_aborts_without_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg = load_kolektor_csv_config(rel_toml, tmp)

            bad_img = tmp / "data/preprocessed/KolektorSDD2/train/good/10001.png"
            _write_png(bad_img, np.zeros((5, 5), dtype=np.uint8))

            with self.assertRaises(ValueError):
                export_kolektor_csv(cfg)

            self.assertFalse(cfg.effective_output_csv.exists())
            self.assertFalse(cfg.effective_output_csv.with_suffix(".csv.tmp").exists())

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            cfg = load_kolektor_csv_config(rel_toml, tmp)

            s = export_kolektor_csv(cfg, dry_run=True)
            self.assertEqual(s.rows, 4)
            self.assertEqual(s.bytes_written, 0)
            self.assertFalse(cfg.effective_output_csv.exists())


class TestCliEntryPoint(unittest.TestCase):
    """Test suite for experiments/export_kolektorsdd2_csv.py CLI behavior."""

    def test_cli_dry_run_returns_zero_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            exit_code = cli_main(
                ["--config", str(rel_toml), "--dry-run"], project_root=tmp
            )
            self.assertEqual(exit_code, 0)
            self.assertFalse((tmp / "data/csv/kolektorsdd2.csv").exists())

    def test_cli_inverted_writes_inverted_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            exit_code = cli_main(
                ["--config", str(rel_toml), "--inverted"], project_root=tmp
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((tmp / "data/csv/kolektorsdd2-inverted.csv").is_file())

    def test_cli_existing_output_exits_with_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            # First export succeeds
            self.assertEqual(cli_main(["--config", str(rel_toml)], project_root=tmp), 0)
            # Second export without --overwrite raises SystemExit
            with self.assertRaises(SystemExit) as ctx:
                cli_main(["--config", str(rel_toml)], project_root=tmp)
            self.assertIn("error:", str(ctx.exception))

    def test_cli_overwrite_replaces_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rel_toml = _make_project(tmp, size=4)
            self.assertEqual(cli_main(["--config", str(rel_toml)], project_root=tmp), 0)
            exit_code = cli_main(
                ["--config", str(rel_toml), "--overwrite"], project_root=tmp
            )
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
