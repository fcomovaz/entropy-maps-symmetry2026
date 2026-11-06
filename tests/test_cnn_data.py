"""Unit and integration tests for CNN data loading, metadata indexing, and split parity.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_cnn_data.py
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_cnn_data.py -v
"""

from __future__ import annotations

import csv
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn.config import (
    AugmentationConfig,
    BenchmarkConfig,
    CnnConfig,
    DataConfig,
    OutputConfig,
    TrainingConfig,
)
from src.models.cnn.data import (
    CnnDatasetPartition,
    InMemCnnDataset,
    load_cnn_partition,
)


class TestCnnData(unittest.TestCase):
    """Test suite for validating CNN image loading, metadata indexing, and split parity."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_cnn_data_"))
        self.img_dir = self.temp_dir / "preprocessed"
        self.csv_path = self.temp_dir / "metadata.csv"

        # Create mock MVTec structure: object/split/defect_name/file.png
        # 10 good images, 10 defect images, plus 2 ground_truth mask rows
        self.img_dir.mkdir(parents=True)
        self.rows_meta = []

        # 10 good images in train and test
        for i in range(10):
            split = "train" if i < 6 else "test"
            fname = f"{i:03d}.png"
            subpath = Path("bottle") / split / "good" / fname
            full_path = self.img_dir / subpath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            # Create valid 128x128 PNG
            img = Image.fromarray(np.full((128, 128), fill_value=120, dtype=np.uint8))
            img.save(full_path)
            self.rows_meta.append(["bottle", split, "good", "good", fname])

        # 10 defect images in train and test
        for i in range(10):
            split = "train" if i < 6 else "test"
            fname = f"def_{i:03d}.png"
            subpath = Path("bottle") / split / "broken" / fname
            full_path = self.img_dir / subpath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            img = Image.fromarray(np.full((128, 128), fill_value=200, dtype=np.uint8))
            img.save(full_path)
            self.rows_meta.append(["bottle", split, "defect", "broken", fname])

        # 2 ground_truth rows (masks) that MUST be ignored
        self.rows_meta.append(["bottle", "ground_truth", "defect", "broken", "mask_0.png"])
        self.rows_meta.append(["bottle", "ground_truth", "defect", "broken", "mask_1.png"])

        # Write metadata CSV
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["object_name", "split", "type", "defect_name", "file_name"])
            writer.writerows(self.rows_meta)

        self.cfg = CnnConfig(
            dataset_key="mvtecad",
            dataset_name="mvtec-ad",
            preprocessed_dir=self.img_dir,
            preprocessed_dir_inverted=self.img_dir,
            metadata_csv_path=self.csv_path,
            metadata_csv_path_inverted=self.csv_path,
            results_dir=self.temp_dir / "results",
            inverted=False,
            overwrite=True,
            seed=42,
            seeds=(42,),
            default_model="ds_cnn",
            models=("ds_cnn",),
            default_object="bottle",
            subdatasets=("bottle",),
            data=DataConfig(
                split_column="split",
                label_column="type",
                object_column="object_name",
                positive_label="defect",
                negative_label="good",
                split_mode="stratified",
                train_split_name="train",
                test_split_name="test",
                test_size=0.20,
                val_size=0.15,
                image_size=128,
                normalize_mean=0.5,
                normalize_std=0.5,
                use_metadata_index=True,
                metadata_csv_key="full_csv_path",
                metadata_csv_key_inverted="full_csv_path_inverted",
                allowed_splits=("train", "test"),
            ),
            training=TrainingConfig(
                epochs=2,
                batch_size=4,
                learning_rate=1e-3,
                weight_decay=1e-4,
                optimizer="adamw",
                scheduler="cosine",
                early_stopping_patience=2,
                class_weighting="balanced",
                num_workers=0,
                device="cpu",
                amp=False,
                deterministic=True,
                threshold_mode="val_f1",
            ),
            augmentation=AugmentationConfig(
                hflip=True,
                vflip=False,
                rotate_deg=0,
                brightness=0.0,
            ),
            benchmark=BenchmarkConfig(
                warmup_iters=1,
                timed_iters=2,
                batch_sizes=(1,),
                devices=("cpu",),
                threads=1,
                measure_flops=False,
            ),
            output=OutputConfig(
                results_dir=self.temp_dir / "results",
                dpi=100,
                float_precision=6,
                save_checkpoints=False,
            ),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_partition_shapes_and_classes(self) -> None:
        """Verify partition loads correct shapes, excludes ground_truth, and preserves counts."""
        partition = load_cnn_partition(self.cfg, object_name="bottle", seed=42)
        self.assertIsInstance(partition, CnnDatasetPartition)

        # 20 total images (10 good + 10 defect); 20% test = 4 images (2 good, 2 defect)
        # Train full = 16 images; 15% val ≈ 2 images (1 good, 1 defect) -> Train = 14 images
        self.assertEqual(len(partition.y_test), 4)
        self.assertEqual(partition.test_counts["good"], 2)
        self.assertEqual(partition.test_counts["defect"], 2)

        # Shapes must be [N, 1, 128, 128]
        self.assertEqual(tuple(partition.X_train.shape[1:]), (1, 128, 128))
        self.assertEqual(tuple(partition.X_test.shape[1:]), (1, 128, 128))
        self.assertEqual(tuple(partition.X_val.shape[1:]), (1, 128, 128))

    def test_split_parity_with_sklearn_reference(self) -> None:
        """Verify that stratified test file names exactly match sklearn train_test_split."""
        partition = load_cnn_partition(self.cfg, object_name="bottle", seed=42)

        # Filtered rows: 20 rows (excluding 2 ground_truth)
        y_ref = np.array([0] * 10 + [1] * 10)
        file_names_ref = np.array([r[4] for r in self.rows_meta[:20]])

        _, test_idx = train_test_split(
            np.arange(20), test_size=0.20, stratify=y_ref, random_state=42
        )
        expected_test_files = tuple(file_names_ref[test_idx])
        self.assertEqual(partition.test_file_names, expected_test_files)

    def test_invalid_image_resolution_fails_fast(self) -> None:
        """Verify non-128x128 image triggers ValueError."""
        bad_img_path = self.img_dir / "bottle" / "train" / "good" / "000.png"
        bad_img = Image.fromarray(np.full((64, 64), fill_value=128, dtype=np.uint8))
        bad_img.save(bad_img_path)

        with self.assertRaises(ValueError) as ctx:
            load_cnn_partition(self.cfg, object_name="bottle")
        self.assertIn("does not match expected", str(ctx.exception))

    def test_dataset_augmentation_hflip(self) -> None:
        """Verify InMemCnnDataset applies horizontal flip without altering labels."""
        img = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
        lbl = torch.tensor(1, dtype=torch.long)
        ds = InMemCnnDataset(img.unsqueeze(0), lbl.unsqueeze(0), hflip=True)

        found_flipped = False
        for _ in range(50):
            x_out, y_out = ds[0]
            self.assertEqual(y_out.item(), 1)
            if torch.equal(x_out, torch.flip(img, dims=[-1])):
                found_flipped = True
                break
        self.assertTrue(found_flipped, "Horizontal flip should be triggered probabilistically")


if __name__ == "__main__":
    unittest.main(verbosity=2)
