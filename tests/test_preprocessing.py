"""Unit and integration tests for image preprocessing, letterboxing, and intensity inversion.

Manual execution command:
    .venv\\Scripts\\python.exe -m unittest tests/test_preprocessing.py
or (if pytest is installed):
    .venv\\Scripts\\python.exe -m pytest tests/test_preprocessing.py -q
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

from experiments.preprocess_mvtec_ad import main as cli_main
from src.preprocessing.letterbox import (
    invert_intensity,
    letterbox_to_square,
    to_grayscale,
)
from src.preprocessing.mvtec_ad import (
    PreprocessConfig,
    is_mask,
    load_config,
    preprocess_image,
    run,
)


class TestLetterboxAndGrayscale(unittest.TestCase):
    """Test suite for pure image transformation functions."""

    def test_to_grayscale_from_rgb(self) -> None:
        """Verify RGB image converts to 2D uint8 array."""
        rgb_img = Image.new("RGB", (64, 64), color=(120, 200, 50))
        gray = to_grayscale(rgb_img)
        self.assertEqual(gray.shape, (64, 64))
        self.assertEqual(gray.dtype, np.uint8)

    def test_to_grayscale_from_l(self) -> None:
        """Verify single-channel 'L' image preserves values."""
        l_img = Image.new("L", (32, 32), color=180)
        gray = to_grayscale(l_img)
        self.assertEqual(gray.shape, (32, 32))
        self.assertEqual(gray.dtype, np.uint8)
        self.assertTrue((gray == 180).all())

    def test_square_300x300_becomes_128x128_no_padding(self) -> None:
        """Square 300x300 RGB becomes [128, 128] without padding."""
        img = np.full((300, 300), fill_value=128, dtype=np.uint8)
        result = letterbox_to_square(img, size=128, pad_value=0)
        self.assertEqual(result.shape, (128, 128))
        self.assertTrue((result == 128).all())

    def test_landscape_200x100_pads_rows_centered(self) -> None:
        """200x100 (W=200, H=100) scales to [64, 128] and pads 32 rows top and bottom."""
        # Note: shape is [H=100, W=200]
        img = np.full((100, 200), fill_value=255, dtype=np.uint8)
        result = letterbox_to_square(img, size=128, pad_value=0)
        self.assertEqual(result.shape, (128, 128))

        # Top 32 rows should be padding (0)
        self.assertTrue((result[:32, :] == 0).all())
        # Center 64 rows (indices 32 to 96) should be content (255)
        self.assertTrue((result[32:96, :] == 255).all())
        # Bottom 32 rows should be padding (0)
        self.assertTrue((result[96:, :] == 0).all())

    def test_portrait_100x200_pads_columns_centered(self) -> None:
        """100x200 (W=100, H=200) scales to [128, 64] and pads 32 columns left and right."""
        # shape is [H=200, W=100]
        img = np.full((200, 100), fill_value=200, dtype=np.uint8)
        result = letterbox_to_square(img, size=128, pad_value=0)
        self.assertEqual(result.shape, (128, 128))

        # Left 32 columns should be padding (0)
        self.assertTrue((result[:, :32] == 0).all())
        # Center 64 columns should be content (200)
        self.assertTrue((result[:, 32:96] == 200).all())
        # Right 32 columns should be padding (0)
        self.assertTrue((result[:, 96:] == 0).all())

    def test_upscaling_smaller_than_target_size(self) -> None:
        """Image smaller than 128x128 is enlarged to fill target size."""
        img = np.full((50, 50), fill_value=75, dtype=np.uint8)
        result = letterbox_to_square(img, size=128, pad_value=0)
        self.assertEqual(result.shape, (128, 128))
        self.assertTrue((result == 75).all())

    def test_mask_nearest_preserves_binary_values_and_alignment(self) -> None:
        """Mask with {0, 255} stays {0, 255} with NEAREST and shares geometric alignment."""
        mask = np.zeros((100, 200), dtype=np.uint8)
        mask[25:75, 50:150] = 255  # center rectangle

        result = letterbox_to_square(
            mask,
            size=128,
            pad_value=0,
            interpolation=Image.Resampling.NEAREST,
        )
        unique_vals = set(np.unique(result))
        self.assertTrue(unique_vals.issubset({0, 255}))

        # Scaled mask rectangle:
        # H: 50 -> 32 rows, centered in 64 rows + 32 pad -> row 48 to 80
        # W: 100 -> 64 cols, centered in 128 cols -> col 32 to 96
        self.assertEqual(result[48:80, 32:96].mean(), 255.0)

    def test_invalid_parameters_raise_value_error(self) -> None:
        """Ensure invalid size, anchor, ndim, or pad_value raise ValueError."""
        valid_img = np.zeros((64, 64), dtype=np.uint8)
        with self.assertRaises(ValueError):
            letterbox_to_square(valid_img, size=0)
        with self.assertRaises(ValueError):
            letterbox_to_square(valid_img, size=128, pad_value=300)
        with self.assertRaises(ValueError):
            letterbox_to_square(valid_img, size=128, anchor="top-left")
        with self.assertRaises(ValueError):
            letterbox_to_square(np.zeros((64, 64, 3), dtype=np.uint8), size=128)

    def test_invert_intensity_values_and_type_check(self) -> None:
        """Verify invert_intensity maps [0, 255] -> [255, 0] and rejects invalid inputs."""
        arr = np.array([[0, 100], [255, 200]], dtype=np.uint8)
        inv = invert_intensity(arr)
        expected = np.array([[255, 155], [0, 55]], dtype=np.uint8)
        np.testing.assert_array_equal(inv, expected)

        # Rejects 3D input
        with self.assertRaises(ValueError):
            invert_intensity(np.zeros((10, 10, 3), dtype=np.uint8))
        # Rejects non-uint8 input
        with self.assertRaises(ValueError):
            invert_intensity(np.zeros((10, 10), dtype=np.float32))

    def test_white_image_inverted_padding_behavior(self) -> None:
        """White 200x100 RGB image with invert=True produces black content and black padding."""
        white_rgb = Image.new("RGB", (200, 100), color=(255, 255, 255))
        gray = to_grayscale(white_rgb)

        # Standard (non-inverted): content 255, pad 0
        normal_box = letterbox_to_square(gray, size=128, pad_value=0)
        self.assertTrue((normal_box[32:96, :] == 255).all())
        self.assertTrue((normal_box[:32, :] == 0).all())
        self.assertTrue((normal_box[96:, :] == 0).all())

        # Inverted: invert before letterbox
        inv_gray = invert_intensity(gray)
        inv_box = letterbox_to_square(inv_gray, size=128, pad_value=0)
        # Content was 255 -> now 0. Padding is 0. Entire canvas is 0.
        self.assertTrue((inv_box == 0).all())


class TestPipelineOrchestration(unittest.TestCase):
    """Test suite for pipeline configuration, traversal, and execution."""

    def test_is_mask_detection(self) -> None:
        """Verify is_mask identifies category/ground_truth paths correctly."""
        root = Path("C:/data/raw/mvtec-ad")
        mask_path = root / "bottle" / "ground_truth" / "broken_large" / "000_mask.png"
        img_path = root / "bottle" / "test" / "broken_large" / "000.png"
        train_path = root / "bottle" / "train" / "good" / "000.png"

        self.assertTrue(is_mask(mask_path, root, "ground_truth"))
        self.assertFalse(is_mask(img_path, root, "ground_truth"))
        self.assertFalse(is_mask(train_path, root, "ground_truth"))

    def test_output_inside_input_refused(self) -> None:
        """Ensure configuring output directory inside raw data raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cfg_file = temp_path / "task.toml"
            base_file = temp_path / "base.toml"

            raw_dir = temp_path / "raw"
            raw_dir.mkdir()
            (raw_dir / "cat1").mkdir()

            base_file.write_text(
                f"""
                [datasets.mvtecad]
                input_path = "raw"
                output_path = "raw/preprocessed"
                output_path_inverted = "data/preprocessed-inv"
                subdatasets = ["cat1"]
                """
            )

            cfg_file.write_text(
                f"""
                [preprocessing]
                base_config = "{base_file.relative_to(temp_path).as_posix()}"
                dataset_key = "mvtecad"
                target_size = 128
                pad_value = 0
                padding_anchor = "center"
                image_extensions = [".png"]
                overwrite = false
                """
            )

            with self.assertRaises(ValueError):
                load_config(cfg_file, temp_path)

    def test_pipeline_execution_and_skip_logic(self) -> None:
        """Test full pipeline run, mirroring structure and skip on existing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_dir = temp_path / "data" / "raw" / "mvtec-ad"
            out_dir = temp_path / "data" / "preprocessed" / "mvtec-ad"
            out_inv_dir = temp_path / "data" / "preprocessed" / "mvtec-ad-inverted"

            category_dir = raw_dir / "bottle"
            train_dir = category_dir / "train" / "good"
            gt_dir = category_dir / "ground_truth" / "broken_small"
            train_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)

            # Create sample train RGB image [200, 100]
            img_sample = Image.new("RGB", (200, 100), color=(100, 150, 200))
            img_path = train_dir / "000.png"
            img_sample.save(img_path)

            # Create sample mask [200, 100]
            mask_sample = Image.new("L", (200, 100), color=0)
            mask_path = gt_dir / "000_mask.png"
            mask_sample.save(mask_path)

            cfg = PreprocessConfig(
                input_root=raw_dir,
                output_root=out_dir,
                output_root_inverted=out_inv_dir,
                subdatasets=("bottle",),
                target_size=128,
                pad_value=0,
                image_extensions=(".png",),
                image_resampling=Image.Resampling.LANCZOS,
                mask_resampling=Image.Resampling.NEAREST,
                mask_folder="ground_truth",
                padding_anchor="center",
                overwrite=False,
                invert=False,
            )

            # 1. Run standard pipeline
            summary1 = run(cfg)
            self.assertEqual(summary1.total_discovered, 2)
            self.assertEqual(summary1.processed, 2)
            self.assertEqual(summary1.skipped, 0)
            self.assertEqual(summary1.failed, 0)
            self.assertEqual(summary1.masks, 1)
            self.assertEqual(summary1.padded, 2)  # both were non-square

            out_img = out_dir / "bottle" / "train" / "good" / "000.png"
            out_mask = out_dir / "bottle" / "ground_truth" / "broken_small" / "000_mask.png"
            self.assertTrue(out_img.exists())
            self.assertTrue(out_mask.exists())

            # Check format and size
            with Image.open(out_img) as pil_out:
                self.assertEqual(pil_out.mode, "L")
                self.assertEqual(pil_out.size, (128, 128))

            with Image.open(out_mask) as pil_mask:
                self.assertEqual(pil_mask.mode, "L")
                self.assertEqual(pil_mask.size, (128, 128))

            # 2. Re-run without overwrite: should skip both
            summary2 = run(cfg)
            self.assertEqual(summary2.processed, 0)
            self.assertEqual(summary2.skipped, 2)

            # 3. Re-run with overwrite: should process both
            cfg_overwrite = PreprocessConfig(
                input_root=cfg.input_root,
                output_root=cfg.output_root,
                output_root_inverted=cfg.output_root_inverted,
                subdatasets=cfg.subdatasets,
                target_size=cfg.target_size,
                pad_value=cfg.pad_value,
                image_extensions=cfg.image_extensions,
                image_resampling=cfg.image_resampling,
                mask_resampling=cfg.mask_resampling,
                mask_folder=cfg.mask_folder,
                padding_anchor=cfg.padding_anchor,
                overwrite=True,
                invert=False,
            )
            summary3 = run(cfg_overwrite)
            self.assertEqual(summary3.processed, 2)
            self.assertEqual(summary3.skipped, 0)

            # 4. Inverted pipeline run: writes to out_inv_dir
            cfg_inverted = PreprocessConfig(
                input_root=cfg.input_root,
                output_root=cfg.output_root,
                output_root_inverted=cfg.output_root_inverted,
                subdatasets=cfg.subdatasets,
                target_size=cfg.target_size,
                pad_value=cfg.pad_value,
                image_extensions=cfg.image_extensions,
                image_resampling=cfg.image_resampling,
                mask_resampling=cfg.mask_resampling,
                mask_folder=cfg.mask_folder,
                padding_anchor=cfg.padding_anchor,
                overwrite=False,
                invert=True,
            )
            summary_inv = run(cfg_inverted)
            self.assertEqual(summary_inv.processed, 2)

            inv_img_path = out_inv_dir / "bottle" / "train" / "good" / "000.png"
            inv_mask_path = out_inv_dir / "bottle" / "ground_truth" / "broken_small" / "000_mask.png"
            self.assertTrue(inv_img_path.exists())
            self.assertTrue(inv_mask_path.exists())

            # Mask must be identical in both normal and inverted folders
            with Image.open(out_mask) as norm_m, Image.open(inv_mask_path) as inv_m:
                np.testing.assert_array_equal(np.asarray(norm_m), np.asarray(inv_m))

            # Image content inverted relationship: inverted content == 255 - normal content
            with Image.open(out_img) as norm_im, Image.open(inv_img_path) as inv_im:
                arr_norm = np.asarray(norm_im)
                arr_inv = np.asarray(inv_im)
                # In center 64 rows (32:96), arr_inv should be 255 - arr_norm
                np.testing.assert_array_equal(
                    arr_inv[32:96, :], (255 - arr_norm[32:96, :]).astype(np.uint8)
                )
                # In padding rows (top and bottom 32), both must be 0
                self.assertTrue((arr_norm[:32, :] == 0).all())
                self.assertTrue((arr_inv[:32, :] == 0).all())

    def test_cli_inverted_flag(self) -> None:
        """Test invoking CLI main with --dry-run and --inverted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            base_file = temp_path / "base.toml"
            task_file = temp_path / "task.toml"

            raw_dir = temp_path / "raw"
            raw_dir.mkdir()
            (raw_dir / "cat1").mkdir()

            base_file.write_text(
                f"""
                [datasets.mvtecad]
                input_path = "{raw_dir.relative_to(temp_path).as_posix()}"
                output_path = "preprocessed"
                output_path_inverted = "preprocessed-inv"
                subdatasets = ["cat1"]
                """
            )

            task_file.write_text(
                f"""
                [preprocessing]
                base_config = "{base_file.relative_to(temp_path).as_posix()}"
                dataset_key = "mvtecad"
                target_size = 128
                pad_value = 0
                padding_anchor = "center"
                image_extensions = [".png"]
                overwrite = false

                [preprocessing.images]
                resampling = "LANCZOS"
                convert_to_grayscale = true
                invert = false

                [preprocessing.masks]
                folder = "ground_truth"
                resampling = "NEAREST"
                convert_to_grayscale = false
                """
            )

            exit_code = cli_main(
                [
                    "--config",
                    str(task_file),
                    "--inverted",
                    "--dry-run",
                ]
            )
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
