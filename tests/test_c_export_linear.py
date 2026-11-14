"""Unit tests for formal MlpDotCoreClassifier parity, scaler folding, and linear/MLP quantization."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from src.c_export.fixed_point import quantize
from src.c_export.linear_export import (
    _crosscheck_saved_metrics,
    _quantize_lr_dot,
    _quantize_mlp_16_dot,
    eval_lr_dot_q15,
    eval_mlp_16_dot_q15,
)
from src.models.descriptor_classifier import ClassificationMetrics, LinearDotCoreClassifier
from src.models.mlp_dot import MlpDotCoreClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_lab14_models_module():
    mod_path = PROJECT_ROOT / "understanding" / "14_ml_classifiers_mcu" / "models.py"
    lab_dir = str(mod_path.parent)
    added = False
    if lab_dir not in sys.path:
        sys.path.insert(0, lab_dir)
        added = True
    try:
        spec = importlib.util.spec_from_file_location("lab14_models_parity_ref", mod_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        if added and lab_dir in sys.path:
            sys.path.remove(lab_dir)


class TestCExportLinear(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        x0 = rng.normal(loc=0.25, scale=0.08, size=(32, 94))
        x1 = rng.normal(loc=0.55, scale=0.10, size=(24, 94))
        self.X = np.clip(np.vstack([x0, x1]), 0.0, 1.0)
        self.y = np.concatenate([np.zeros(32, dtype=np.int64), np.ones(24, dtype=np.int64)])

    def test_mlp_dot_matches_lab14_reference(self) -> None:
        from sklearn.preprocessing import StandardScaler

        lab14 = _load_lab14_models_module()
        scaler = StandardScaler().fit(self.X)
        d_in = int(self.X.shape[1])
        net = lab14._train_torch_binary_mlp(
            net_factory=lambda: lab14._TorchMlp1Hidden(
                d_in=d_in, d_hidden=16, dropout=0.1, use_skip=False
            ),
            scaler=scaler,
            X_train=self.X,
            y_train=self.y,
            epochs=25,
            lr=0.002,
            weight_decay=0.05,
            seed=42,
            train_threads=1,
        )
        ref = lab14.FoldedMlp1HiddenModel(name="mlp_16_dot", net=net, scaler=scaler)
        formal = MlpDotCoreClassifier(
            hidden_dims=(16,),
            epochs=25,
            learning_rate=0.002,
            weight_decay=0.05,
            dropout=0.1,
            positive_weight_mode="balanced",
            train_threads=1,
            random_state=42,
        ).fit(self.X, self.y)

        np.testing.assert_allclose(formal.w1_eff_, ref.w1_64, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(formal.b1_eff_, ref.b1_64, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(formal.w2_eff_, ref.w2_64, atol=1e-12, rtol=0.0)
        self.assertAlmostEqual(formal.b2_eff_, float(ref.b2_64), places=12)
        np.testing.assert_allclose(
            formal.decision_function(self.X), ref.predict_scores_f64(self.X), atol=1e-11
        )

    def test_lr_dot_and_mlp_quantization_zero_label_mismatch(self) -> None:
        lr_model = LinearDotCoreClassifier(random_state=42).fit(self.X, self.y)
        mlp_model = MlpDotCoreClassifier(epochs=30, random_state=42).fit(self.X, self.y)
        x_q15, n_sat = quantize(self.X, 15, bits=16)
        self.assertEqual(n_sat, 0)

        lr_q = _quantize_lr_dot(lr_model, self.X, x_q15, in_frac=15)
        _, lr_labels_q = eval_lr_dot_q15(x_q15, lr_q)
        lr_labels_f32 = (lr_model.decision_function(self.X) > 0.0).astype(np.int32)
        self.assertEqual(int(np.sum(lr_labels_q != lr_labels_f32)), 0)

        mlp_q = _quantize_mlp_16_dot(mlp_model, self.X, self.X, x_q15, in_frac=15)
        _, mlp_labels_q = eval_mlp_16_dot_q15(x_q15, mlp_q)
        mlp_labels_f32 = (mlp_model.decision_function(self.X) > 0.0).astype(np.int32)
        self.assertEqual(int(np.sum(mlp_labels_q != mlp_labels_f32)), 0)

    def test_crosscheck_detects_metric_drift(self) -> None:
        # Use existing KolektorSDD2 lr_dot metrics_summary.json if present; skip cleanly if absent.
        summary_dir = PROJECT_ROOT / "results" / "classification"
        target_file = summary_dir / "kolektorsdd2" / "full_lr_dot" / "metrics_summary.json"
        if not target_file.exists():
            self.skipTest("KolektorSDD2 lr_dot metrics_summary.json not generated yet.")

        fake_cfg = SimpleNamespace(
            classification_cfg=SimpleNamespace(
                effective_dataset_tag="kolektorsdd2",
                float_precision=4,
            ),
            object_name=None,
            classifier=SimpleNamespace(
                metrics_crosscheck_dir=summary_dir,
                metrics_crosscheck_abs_tol=1e-9,
            ),
        )
        drifted_metrics = ClassificationMetrics(
            classifier_name="lr_dot",
            subset_name="x94_full",
            n_features=94,
            accuracy=0.1234,
            balanced_accuracy=0.1234,
            precision=0.1234,
            recall=0.1234,
            f1=0.1234,
            roc_auc=0.1234,
            pr_auc=0.1234,
            confusion_matrix=((1, 0), (0, 1)),
            fpr=np.zeros(2),
            tpr=np.zeros(2),
            pr_precision=np.zeros(2),
            pr_recall=np.zeros(2),
            train_elapsed_s=0.0,
            inference_elapsed_s=0.0,
            inference_ms_per_sample=0.0,
        )
        with self.assertRaises(ValueError):
            _crosscheck_saved_metrics(fake_cfg, "lr_dot", drifted_metrics)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
