"""Unit and smoke tests for CNN model training, scheduling, and determinism.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_cnn_train.py
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_cnn_train.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn.architectures import create_model
from src.models.cnn.architectures.common import seed_everything
from src.models.cnn.config import (
    AugmentationConfig,
    BenchmarkConfig,
    CnnConfig,
    DataConfig,
    OutputConfig,
    TrainingConfig,
)
from src.models.cnn.data import CnnDatasetPartition
from src.models.cnn.train import TrainResult, train_model


def _create_synthetic_partition(
    n_train: int = 32, n_val: int = 8, n_test: int = 8, size: int = 32, seed: int = 42
) -> CnnDatasetPartition:
    """Create lightweight synthetic partition for smoke testing."""
    g = torch.Generator().manual_seed(seed)

    def _make_tensors(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(n, 1, size, size, generator=g)
        y = (torch.arange(n) % 2).long()
        # Add a clear signal for class 1 to ensure learning
        for i in range(n):
            if y[i] == 1:
                x[i, 0, : size // 2, : size // 2] += 2.0
        return x, y

    X_train, y_train = _make_tensors(n_train)
    X_val, y_val = _make_tensors(n_val)
    X_test, y_test = _make_tensors(n_test)

    return CnnDatasetPartition(
        dataset_name="synthetic",
        object_name="synth_obj",
        inverted=False,
        split_mode="presplit",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_counts={"good": int((y_train == 0).sum().item()), "defect": int((y_train == 1).sum().item())},
        val_counts={"good": int((y_val == 0).sum().item()), "defect": int((y_val == 1).sum().item())},
        test_counts={"good": int((y_test == 0).sum().item()), "defect": int((y_test == 1).sum().item())},
        test_file_names=tuple(f"{i}.png" for i in range(n_test)),
        load_elapsed_s=0.01,
    )


class TestCnnTrain(unittest.TestCase):
    """Test suite for training dynamics, convergence smoke test, and determinism."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.cfg = CnnConfig(
            dataset_key="synthetic",
            dataset_name="synthetic",
            preprocessed_dir=Path("."),
            preprocessed_dir_inverted=Path("."),
            metadata_csv_path=Path("."),
            metadata_csv_path_inverted=Path("."),
            results_dir=Path("."),
            inverted=False,
            overwrite=True,
            seed=42,
            seeds=(42,),
            default_model="ultra_tiny",
            models=("ultra_tiny",),
            default_object="synth_obj",
            subdatasets=("synth_obj",),
            data=DataConfig(
                split_column="split",
                label_column="type",
                object_column="",
                positive_label="defect",
                negative_label="good",
                split_mode="presplit",
                train_split_name="train",
                test_split_name="test",
                test_size=0.20,
                val_size=0.20,
                image_size=32,
                normalize_mean=0.5,
                normalize_std=0.5,
                use_metadata_index=False,
                metadata_csv_key="full_csv_path",
                metadata_csv_key_inverted="full_csv_path_inverted",
                allowed_splits=("train", "test"),
            ),
            training=TrainingConfig(
                epochs=4,
                batch_size=8,
                learning_rate=0.01,
                weight_decay=1e-4,
                optimizer="adamw",
                scheduler="cosine",
                early_stopping_patience=3,
                class_weighting="balanced",
                num_workers=0,
                device="cpu",
                amp=False,
                deterministic=True,
                threshold_mode="val_f1",
            ),
            augmentation=AugmentationConfig(
                hflip=False,
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
                results_dir=Path("."),
                dpi=100,
                float_precision=6,
                save_checkpoints=False,
            ),
        )

    def test_training_smoke_and_loss_decrease(self) -> None:
        """Verify 4 epochs of ultra_tiny training lowers loss and populates TrainResult."""
        partition = _create_synthetic_partition(seed=42)
        model = create_model("ultra_tiny", num_classes=2)

        res = train_model(
            model=model,
            partition=partition,
            cfg=self.cfg,
            seed=42,
            device=self.device,
        )

        self.assertIsInstance(res, TrainResult)
        self.assertGreater(res.train_elapsed_s, 0.0)
        self.assertEqual(len(res.history), 4)

        first_loss = res.history[0]["train_loss"]
        last_loss = res.history[-1]["train_loss"]
        self.assertLess(last_loss, first_loss, "Training should reduce empirical loss")

    def test_training_determinism_with_same_seed(self) -> None:
        """Verify that training with the same seed produces identical parameter weights."""
        partition = _create_synthetic_partition(seed=42)

        seed_everything(123)
        model1 = create_model("ultra_tiny", num_classes=2)
        res1 = train_model(
            model=model1,
            partition=partition,
            cfg=self.cfg,
            seed=123,
            device=self.device,
        )

        seed_everything(123)
        model2 = create_model("ultra_tiny", num_classes=2)
        res2 = train_model(
            model=model2,
            partition=partition,
            cfg=self.cfg,
            seed=123,
            device=self.device,
        )

        for k in res1.best_state_dict:
            self.assertTrue(
                torch.equal(res1.best_state_dict[k], res2.best_state_dict[k]),
                f"Weight mismatch for layer {k} under identical seeds",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
