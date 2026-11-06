"""Unit tests for CNN evaluation metrics computation and probability outputs.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_cnn_evaluate.py
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_cnn_evaluate.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

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
from src.models.cnn.data import CnnDatasetPartition
from src.models.cnn.evaluate import (
    ThresholdCalibrationInfo,
    evaluate_cnn,
    find_optimal_threshold,
    predict_proba,
)
from src.models.descriptor_classifier import ClassificationMetrics


class MockLinearModel(nn.Module):
    """Deterministic mock model for verifying exact metrics calculation."""

    def __init__(self, logits_matrix: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits_matrix
        self.idx = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        out = self.logits[self.idx : self.idx + b]
        self.idx = (self.idx + b) % len(self.logits)
        return out


class TestCnnEvaluate(unittest.TestCase):
    """Test suite for verifying CNN evaluation metrics and probability outputs."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.cfg = CnnConfig(
            dataset_key="test_ds",
            dataset_name="TestDataset",
            preprocessed_dir=Path("."),
            preprocessed_dir_inverted=Path("."),
            metadata_csv_path=Path("."),
            metadata_csv_path_inverted=Path("."),
            results_dir=Path("."),
            inverted=False,
            overwrite=True,
            seed=42,
            seeds=(42,),
            default_model="mock",
            models=("mock",),
            default_object=None,
            subdatasets=(),
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
                val_size=0.15,
                image_size=32,
                normalize_mean=0.5,
                normalize_std=0.5,
                use_metadata_index=False,
                metadata_csv_key="full_csv_path",
                metadata_csv_key_inverted="full_csv_path_inverted",
                allowed_splits=("train", "test"),
            ),
            training=TrainingConfig(
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                weight_decay=1e-4,
                optimizer="adamw",
                scheduler="cosine",
                early_stopping_patience=1,
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
                timed_iters=1,
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

    def test_predict_proba_normalization(self) -> None:
        """Verify predicted probabilities are within [0, 1] and sum to 1.0 per row."""
        # Logits for 4 samples: [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 3.0]]
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [-1.0, 3.0]])
        model = MockLinearModel(logits)
        dummy_x = torch.zeros(4, 1, 32, 32)

        probs = predict_proba(model, dummy_x, self.device)
        self.assertEqual(probs.shape, (4, 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(4), atol=1e-6)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_metrics_calculation_and_type_compatibility(self) -> None:
        """Verify evaluate_cnn produces valid ClassificationMetrics with expected math."""
        # 4 samples:
        # y_true: [0, 0, 1, 1]
        # Logits predicting:
        # Sample 0: good (p=0.88 -> class 0) -> TN
        # Sample 1: defect (p=0.88 -> class 1) -> FP
        # Sample 2: defect (p=0.88 -> class 1) -> TP
        # Sample 3: defect (p=0.88 -> class 1) -> TP
        # Confusion matrix: TN=1, FP=1, FN=0, TP=2
        # Accuracy: 3 / 4 = 0.75
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
        model = MockLinearModel(logits)

        partition = CnnDatasetPartition(
            dataset_name="test",
            object_name=None,
            inverted=False,
            split_mode="presplit",
            X_train=torch.empty(0),
            y_train=torch.empty(0),
            X_val=torch.empty(0),
            y_val=torch.empty(0),
            X_test=torch.zeros(4, 1, 32, 32),
            y_test=torch.tensor([0, 0, 1, 1], dtype=torch.long),
            train_counts={"good": 0, "defect": 0},
            val_counts={"good": 0, "defect": 0},
            test_counts={"good": 2, "defect": 2},
            test_file_names=("0.png", "1.png", "2.png", "3.png"),
            load_elapsed_s=0.01,
        )

        metrics, y_pred, y_prob, th_info = evaluate_cnn(
            model=model,
            partition=partition,
            cfg=self.cfg,
            model_name="mock_cnn",
            train_elapsed_s=1.23,
            device=self.device,
            threshold_mode="fixed_05",
        )

        self.assertIsInstance(metrics, ClassificationMetrics)
        self.assertIsInstance(th_info, ThresholdCalibrationInfo)
        self.assertEqual(metrics.accuracy, 0.75)
        self.assertEqual(metrics.confusion_matrix, ((1, 1), (0, 2)))
        self.assertGreater(metrics.f1, 0.0)
        self.assertEqual(metrics.train_elapsed_s, 1.23)
        self.assertGreater(metrics.inference_elapsed_s, 0.0)
        self.assertGreater(metrics.inference_ms_per_sample, 0.0)
        self.assertEqual(th_info.selected_threshold, 0.50)

    def test_find_optimal_threshold_calculation(self) -> None:
        """Verify find_optimal_threshold identifies the cutoff that maximizes F1."""
        # Ground truth: 2 negatives, 2 positives
        y_true = np.array([0, 0, 1, 1])
        # Probabilities:
        # Negative samples have probabilities 0.10 and 0.55
        # Positive samples have probabilities 0.60 and 0.90
        # If threshold = 0.50:
        #   preds = [0, 1, 1, 1] -> 1 FP, 2 TP, 0 FN -> Prec = 2/3 = 0.67, Rec = 2/2 = 1.0, F1 = 0.80
        # If threshold = 0.58:
        #   preds = [0, 0, 1, 1] -> 0 FP, 2 TP, 0 FN -> Prec = 1.0, Rec = 1.0, F1 = 1.0!
        y_prob = np.array([0.10, 0.55, 0.60, 0.90])

        best_th, best_f1 = find_optimal_threshold(y_true, y_prob, metric="f1")
        self.assertAlmostEqual(best_th, 0.56, delta=0.01)
        self.assertLessEqual(best_th, 0.60)
        self.assertAlmostEqual(best_f1, 1.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
