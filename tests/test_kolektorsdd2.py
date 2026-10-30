"""Unit and integration tests for KolektorSDD2 preprocessing pipeline.

Manual execution command:
    .venv\\Scripts\\python.exe -m unittest tests/test_kolektorsdd2.py
or (if pytest is installed):
    .venv\\Scripts\\python.exe -m pytest tests/test_kolektorsdd2.py -q
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.preprocess_kolektorsdd2 import main as cli_main
from src.preprocessing.kolektorsdd2 import (
    KolektorConfig,
    SamplePair,
    discover_pairs,
    is_defective,
    iter_pairs,
    load_config,
    process_pair,
    run,
)


def _setup_mock_environment(root: Path) -> tuple[Path, Path]:
    """Create a minimal synthetic KolektorSDD2 dataset and configuration files.

    Returns:
        Tuple of (base_toml_path, task_toml_path).
    """
    raw_dir = root / "data" / "raw" / "KolektorSDD2"
    train_dir = raw_dir / "train"
    test_dir = raw_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    configs_dir = root / "configs"
    prep_dir = configs_dir / "preprocessing"
    prep_dir.mkdir(parents=True, exist_ok=True)

    base_toml = configs_dir / "base.toml"
    base_toml.write_text(
        f"""[configs]
path = "configs/base.toml"

[datasets.kolektorsdd2]
name = "KolektorSDD2"
input_path = "data/raw/KolektorSDD2"
output_path = "data/processed/KolektorSDD2"
output_path_inverted = "data/processed/KolektorSDD2-inverted"
splits = ["train", "test"]
""",
        encoding="utf-8",
    )

    task_toml = prep_dir / "kolektorsdd2.toml"
    task_toml.write_text(
        """[preprocessing]
base_config = "configs/base.toml"
dataset_key = "kolektorsdd2"
target_size = 128
pad_value = 0
padding_anchor = "center"
overwrite = false

[preprocessing.images]
extension = ".png"
resampling = "LANCZOS"
convert_to_grayscale = true
invert = false

[preprocessing.masks]
suffix = "_GT"
defect_threshold = 0
good_folder = "good"
defect_folder = "defect"

[preprocessing.pairing]
stem_pattern = '^\\d+$'
""",
        encoding="utf-8",
    )

    return base_toml, task_toml


class TestKolektorSDD2Preprocessing(unittest.TestCase):
    """Test suite for KolektorSDD2 pairing, classification, letterboxing, and CLI."""

    def test_is_defective_behavior(self) -> None:
        """Verify is_defective distinguishes zero masks from defect masks."""
        zeros_mask = np.zeros((32, 32), dtype=np.uint8)
        self.assertFalse(is_defective(zeros_mask, threshold=0))

        defect_mask = np.zeros((32, 32), dtype=np.uint8)
        defect_mask[16, 16] = 255
        self.assertTrue(is_defective(defect_mask, threshold=0))

        # Test custom threshold
        defect_mask[16, 16] = 50
        self.assertFalse(is_defective(defect_mask, threshold=50))
        self.assertTrue(is_defective(defect_mask, threshold=49))

        # Test error handling
        with self.assertRaises(ValueError):
            is_defective(np.empty((0, 0), dtype=np.uint8))
        with self.assertRaises(ValueError):
            is_defective(np.zeros((10,), dtype=np.uint8))

    def test_pairing_ignores_orphans_and_invalid_stems(self) -> None:
        """Verify discover_pairs filters out invalid stems and unpaired files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_toml, task_toml = _setup_mock_environment(tmp_path)
            cfg = load_config(task_toml, tmp_path)
            train_dir = cfg.input_root / "train"

            # Valid pair
            Image.new("RGB", (64, 64), color=(100, 100, 100)).save(train_dir / "20000.png")
            Image.new("L", (64, 64), color=0).save(train_dir / "20000_GT.png")

            # Invalid stem copy artifacts (as found in raw dataset)
            Image.new("RGB", (64, 64)).save(train_dir / "10301 (copy).png")
            Image.new("L", (64, 64)).save(train_dir / "10301_GT (copy).png")

            # Orphan image (no mask)
            Image.new("RGB", (64, 64)).save(train_dir / "20001.png")

            # Orphan mask (no image)
            Image.new("L", (64, 64)).save(train_dir / "20002_GT.png")

            pairs, invalid_count = discover_pairs(train_dir, cfg)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].sample_id, "20000")
            # 2 invalid stem files + 1 orphan image + 1 orphan mask = 4 invalid/orphan files
            self.assertEqual(invalid_count, 4)

            iter_list = list(iter_pairs(train_dir, cfg))
            self.assertEqual(len(iter_list), 1)
            self.assertEqual(iter_list[0].sample_id, "20000")

    def test_rgb_converts_to_l_and_l_passes_unchanged(self) -> None:
        """Verify RGB images convert to grayscale and mode 'L' passes without conversion."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, task_toml = _setup_mock_environment(tmp_path)
            cfg = load_config(task_toml, tmp_path)
            train_dir = cfg.input_root / "train"

            # Sample 1: RGB image
            rgb_path = train_dir / "10001.png"
            mask1_path = train_dir / "10001_GT.png"
            Image.new("RGB", (60, 60), color=(150, 100, 50)).save(rgb_path)
            Image.new("L", (60, 60), color=0).save(mask1_path)
            pair_rgb = SamplePair("10001", rgb_path, mask1_path)

            res_rgb = process_pair(pair_rgb, "train", cfg)
            self.assertTrue(res_rgb.converted_to_gray)
            out_rgb_path = cfg.output_root / "train" / "good" / "10001.png"
            with Image.open(out_rgb_path) as out_img:
                self.assertEqual(out_img.mode, "L")
                self.assertEqual(out_img.size, (128, 128))

            # Sample 2: Mode L image
            l_path = train_dir / "10002.png"
            mask2_path = train_dir / "10002_GT.png"
            Image.new("L", (60, 60), color=120).save(l_path)
            Image.new("L", (60, 60), color=0).save(mask2_path)
            pair_l = SamplePair("10002", l_path, mask2_path)

            res_l = process_pair(pair_l, "train", cfg)
            self.assertFalse(res_l.converted_to_gray)
            out_l_path = cfg.output_root / "train" / "good" / "10002.png"
            with Image.open(out_l_path) as out_img:
                self.assertEqual(out_img.mode, "L")
                self.assertEqual(out_img.size, (128, 128))

    def test_non_square_letterbox_padding(self) -> None:
        """Verify non-square portrait image centers content and pads remaining canvas with 0."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, task_toml = _setup_mock_environment(tmp_path)
            cfg = load_config(task_toml, tmp_path)
            train_dir = cfg.input_root / "train"

            # Portrait 230 x 636 image with solid intensity 180
            img_path = train_dir / "30001.png"
            mask_path = train_dir / "30001_GT.png"
            Image.new("L", (230, 636), color=180).save(img_path)
            Image.new("L", (230, 636), color=0).save(mask_path)

            pair = SamplePair("30001", img_path, mask_path)
            process_pair(pair, "train", cfg)

            out_path = cfg.output_root / "train" / "good" / "30001.png"
            with Image.open(out_path) as out_img:
                arr = np.asarray(out_img)

            self.assertEqual(arr.shape, (128, 128))
            # Height scales to 128, width scales to round(230 * 128 / 636) = 46.
            # Horizontal padding is (128 - 46) // 2 = 41 on each side.
            self.assertTrue((arr[:, :41] == 0).all())
            self.assertTrue((arr[:, 41 : 41 + 46] > 0).all())
            self.assertTrue((arr[:, 41 + 46 :] == 0).all())

    def test_no_mask_written_to_output(self) -> None:
        """Verify ground-truth masks are never written into the destination hierarchy."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, task_toml = _setup_mock_environment(tmp_path)
            cfg = load_config(task_toml, tmp_path)
            train_dir = cfg.input_root / "train"

            Image.new("RGB", (64, 64), color=(100, 100, 100)).save(train_dir / "20000.png")
            Image.new("L", (64, 64), color=255).save(train_dir / "20000_GT.png")

            summary = run(cfg)
            self.assertEqual(summary.failed, 0)

            all_outputs = list(cfg.output_root.rglob("*"))
            for out_file in all_outputs:
                if out_file.is_file():
                    self.assertFalse(
                        out_file.name.endswith("_GT.png"),
                        f"Found mask file in output directory: {out_file}",
                    )
                    self.assertEqual(out_file.name, "20000.png")

    def test_good_and_defect_folder_classification(self) -> None:
        """Verify clean samples land in 'good' and defect samples land in 'defect'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, task_toml = _setup_mock_environment(tmp_path)
            cfg = load_config(task_toml, tmp_path)
            train_dir = cfg.input_root / "train"

            # Sample 1: good (mask all zeros)
            Image.new("RGB", (64, 64), color=(10, 10, 10)).save(train_dir / "20001.png")
            Image.new("L", (64, 64), color=0).save(train_dir / "20001_GT.png")

            # Sample 2: defect (mask has 255)
            Image.new("RGB", (64, 64), color=(20, 20, 20)).save(train_dir / "20002.png")
            defect_mask = Image.new("L", (64, 64), color=0)
            defect_mask.putpixel((32, 32), 255)
            defect_mask.save(train_dir / "20002_GT.png")

            summary = run(cfg)
            self.assertEqual(summary.good, 1)
            self.assertEqual(summary.defect, 1)

            good_file = cfg.output_root / "train" / "good" / "20001.png"
            defect_file = cfg.output_root / "train" / "defect" / "20002.png"
            self.assertTrue(good_file.is_file())
            self.assertTrue(defect_file.is_file())

    def test_inverted_writes_to_inverted_root_and_inverts_content(self) -> None:
        """Verify --inverted writes to separate root and inverts content intensity (255 - x)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, task_toml = _setup_mock_environment(tmp_path)
            cfg_normal = load_config(task_toml, tmp_path)
            train_dir = cfg_normal.input_root / "train"

            # Create portrait image with solid intensity 100
            Image.new("L", (230, 636), color=100).save(train_dir / "20000.png")
            Image.new("L", (230, 636), color=0).save(train_dir / "20000_GT.png")

            run(cfg_normal)

            from dataclasses import replace
            cfg_inverted = replace(cfg_normal, invert=True)
            run(cfg_inverted)

            normal_out = cfg_normal.output_root / "train" / "good" / "20000.png"
            inverted_out = cfg_inverted.effective_output_root / "train" / "good" / "20000.png"

            self.assertTrue(normal_out.is_file())
            self.assertTrue(inverted_out.is_file())
            self.assertNotEqual(cfg_normal.effective_output_root, cfg_inverted.effective_output_root)

            with Image.open(normal_out) as img_norm, Image.open(inverted_out) as img_inv:
                arr_norm = np.asarray(img_norm)
                arr_inv = np.asarray(img_inv)

            # Center content area should be inverted: 100 -> 155
            center_x = 64
            center_y = 64
            self.assertEqual(arr_norm[center_y, center_x], 100)
            self.assertEqual(arr_inv[center_y, center_x], 155)

            # Padding areas should remain 0 for both
            self.assertEqual(arr_norm[0, 0], 0)
            self.assertEqual(arr_inv[0, 0], 0)

    def test_dry_run_and_overwrite(self) -> None:
        """Verify dry-run creates no files and overwrite controls skipping of existing files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _, task_toml = _setup_mock_environment(tmp_path)
            cfg = load_config(task_toml, tmp_path)
            train_dir = cfg.input_root / "train"

            Image.new("RGB", (64, 64)).save(train_dir / "20000.png")
            Image.new("L", (64, 64), color=0).save(train_dir / "20000_GT.png")

            # 1. Dry run
            summary_dry = run(cfg, dry_run=True)
            self.assertEqual(summary_dry.pairs, 1)
            self.assertEqual(summary_dry.good, 1)
            self.assertFalse(cfg.output_root.exists())

            # 2. First normal run
            summary_run1 = run(cfg, dry_run=False)
            self.assertEqual(summary_run1.pairs, 1)
            self.assertEqual(summary_run1.skipped_existing, 0)
            target_file = cfg.output_root / "train" / "good" / "20000.png"
            self.assertTrue(target_file.is_file())

            # 3. Second run with overwrite=False
            summary_run2 = run(cfg, dry_run=False)
            self.assertEqual(summary_run2.skipped_existing, 1)

            # 4. Third run with overwrite=True
            from dataclasses import replace
            cfg_overwrite = replace(cfg, overwrite=True)
            summary_run3 = run(cfg_overwrite, dry_run=False)
            self.assertEqual(summary_run3.skipped_existing, 0)

    def test_bad_configuration_raises_informative_errors(self) -> None:
        """Verify fail-fast validation on illegal config values and missing paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            configs_dir = tmp_path / "configs"
            prep_dir = configs_dir / "preprocessing"
            prep_dir.mkdir(parents=True, exist_ok=True)

            base_toml = configs_dir / "base.toml"
            base_toml.write_text(
                """[configs]
path = "configs/base.toml"

[datasets.kolektorsdd2]
name = "KolektorSDD2"
input_path = "non_existent_raw"
output_path = "data/processed/KolektorSDD2"
output_path_inverted = "data/processed/KolektorSDD2-inverted"
splits = ["train"]
""",
                encoding="utf-8",
            )

            task_toml = prep_dir / "kolektorsdd2.toml"
            task_toml.write_text(
                """[preprocessing]
base_config = "configs/base.toml"
dataset_key = "kolektorsdd2"
target_size = 128
pad_value = 0
padding_anchor = "center"
overwrite = false

[preprocessing.images]
extension = ".png"
resampling = "LANCZOS"
convert_to_grayscale = true
invert = false

[preprocessing.masks]
suffix = "_GT"
defect_threshold = 0
good_folder = "good"
defect_folder = "defect"

[preprocessing.pairing]
stem_pattern = '^\\d+$'
""",
                encoding="utf-8",
            )

            # Non-existent input path
            with self.assertRaises(FileNotFoundError):
                load_config(task_toml, tmp_path)


if __name__ == "__main__":
    unittest.main()
