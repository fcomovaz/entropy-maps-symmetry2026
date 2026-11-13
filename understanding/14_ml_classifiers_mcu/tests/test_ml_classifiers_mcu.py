"""Unit tests for Lab 14 MCU classifier folding, memory/ops accounting, and C99 parity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
LAB_ROOT = TEST_DIR.parent
PROJECT_ROOT = LAB_ROOT.parent.parent
for _p in (PROJECT_ROOT, LAB_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config_loader import load_lab_config
from models import (
    FoldedSparseLinearModel,
    train_and_build_mcu_model,
    verify_model_parity,
)
from native_bridge import try_load_native_library
from src.models.descriptor_classifier import DatasetPartition, load_classification_config


def _make_synthetic_partition(seed: int = 42, include_constant_col: bool = True) -> DatasetPartition:
    """Construct an imbalanced nonlinear binary classification fixture shaped like x94."""
    rng = np.random.default_rng(seed)
    n_train, n_test, d = 160, 60, 94
    X_tr = rng.uniform(0.0, 1.0, size=(n_train, d))
    X_te = rng.uniform(0.0, 1.0, size=(n_test, d))
    if include_constant_col:
        X_tr[:, 0] = 0.35
        X_te[:, 0] = 0.35
        X_tr[:, 93] = 0.0
        X_te[:, 93] = 0.0

    # Nonlinear XOR-like signal on features 1..4 plus monotonic shift on feature 5
    def _labels(X: np.ndarray) -> np.ndarray:
        score = (
            1.8 * (X[:, 1] - 0.5) * (X[:, 2] - 0.5)
            + 0.7 * (X[:, 3] - X[:, 4])
            + 0.4 * ((X[:, 5] - 0.5) ** 2)
        )
        thresh = np.quantile(score, 0.78)
        return (score > thresh).astype(np.int64)

    y_tr = _labels(X_tr)
    y_te = _labels(X_te)
    cols = tuple(f"f_{i}" for i in range(d))
    return DatasetPartition(
        dataset_name="synthetic",
        object_name="fixture",
        inverted=False,
        split_mode="stratified",
        subset_name="x94_full",
        feature_columns=cols,
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        train_counts={"good": int(np.sum(y_tr == 0)), "defect": int(np.sum(y_tr == 1))},
        test_counts={"good": int(np.sum(y_te == 0)), "defect": int(np.sum(y_te == 1))},
        load_elapsed_s=0.0,
    )


class TestMcuClassifiersLab14(unittest.TestCase):
    """Verify exact folding math, C99 parity, budgets, and determinism."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_lab_config("configs/understanding/ml_classifiers_mcu.toml", PROJECT_ROOT)
        cls.src_cfg = load_classification_config(
            Path("configs/classification/mvtec_ad.toml"), PROJECT_ROOT
        )
        cls.partition = _make_synthetic_partition(seed=cls.cfg.experiment.seed)
        cls.native_lib = try_load_native_library(cls.cfg.native)

    def test_config_loads_without_missing_keys(self) -> None:
        self.assertEqual(self.cfg.experiment.seed, 42)
        self.assertGreater(len(self.cfg.models.active), 5)
        self.assertIn("mlp_16_dot", self.cfg.models.active)
        self.assertIn("gbdt_tiny_core", self.cfg.models.active)

    def test_missing_config_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_lab_config("configs/understanding/does_not_exist_12345.toml", PROJECT_ROOT)

    def test_all_new_models_algebraic_and_c99_parity(self) -> None:
        candidate_models = [
            "lr_l1_dot",
            "lda_shrink_dot",
            "lsvc_dot",
            "gnb_quad",
            "dt_core",
            "gbdt_tiny_core",
            "mlp_16_dot",
            "mlp_16_skip_dot",
            "mlp_32_16_dot",
        ]
        # Reduce epochs for fast unit test while preserving exact layer folding geometry
        fast_params = dict(self.cfg.models.params_by_model)
        for mlp_key in ("mlp_16_dot", "mlp_16_skip_dot", "mlp_32_16_dot"):
            fast_params[mlp_key] = {**fast_params[mlp_key], "epochs": 25}
        fast_models_cfg = replace(self.cfg.models, params_by_model=fast_params)

        for name in candidate_models:
            with self.subTest(model=name):
                model = train_and_build_mcu_model(
                    model_name=name,
                    partition=self.partition,
                    src_cfg=self.src_cfg,
                    models_cfg=fast_models_cfg,
                    thresholds=self.cfg.thresholds,
                    seed=42,
                )
                parity = verify_model_parity(
                    model=model,
                    X_test=self.partition.X_test,
                    native_lib=self.native_lib,
                    thresholds=self.cfg.thresholds,
                )
                self.assertIsNotNone(parity.max_abs_diff_f64)
                c_snippet = model.export_c99_snippet(f"test_{name}")
                self.assertIn("static const", c_snippet)
                if name in ("dt_core", "gbdt_tiny_core"):
                    self.assertEqual(model.cost_profile.macs, 0)
                if self.native_lib is not None:
                    self.assertIsNotNone(parity.max_abs_diff_c99)
                    self.assertLessEqual(parity.max_abs_diff_c99, 1e-5)

    def test_exact_mcu_memory_and_mac_accounting(self) -> None:
        fast_params = {
            **self.cfg.models.params_by_model,
            "mlp_16_dot": {**self.cfg.models.params_by_model["mlp_16_dot"], "epochs": 15},
            "mlp_16_skip_dot": {**self.cfg.models.params_by_model["mlp_16_skip_dot"], "epochs": 15},
            "mlp_32_16_dot": {**self.cfg.models.params_by_model["mlp_32_16_dot"], "epochs": 15},
        }
        fast_cfg = replace(self.cfg.models, params_by_model=fast_params)

        lda = train_and_build_mcu_model(
            "lda_shrink_dot", self.partition, self.src_cfg, fast_cfg, self.cfg.thresholds, 42
        )
        self.assertEqual(lda.cost_profile.n_params, 95)
        self.assertEqual(lda.cost_profile.flash_bytes_f32, 380)
        self.assertEqual(lda.cost_profile.sram_scratch_bytes_f32, 0)
        self.assertEqual(lda.cost_profile.macs, 94)

        gnb = train_and_build_mcu_model(
            "gnb_quad", self.partition, self.src_cfg, fast_cfg, self.cfg.thresholds, 42
        )
        self.assertEqual(gnb.cost_profile.n_params, 189)
        self.assertEqual(gnb.cost_profile.flash_bytes_f32, 756)
        self.assertEqual(gnb.cost_profile.macs, 188)
        self.assertEqual(gnb.cost_profile.math_calls_at_threshold, 0)

        mlp16 = train_and_build_mcu_model(
            "mlp_16_dot", self.partition, self.src_cfg, fast_cfg, self.cfg.thresholds, 42
        )
        self.assertEqual(mlp16.cost_profile.n_params, 1537)
        self.assertEqual(mlp16.cost_profile.flash_bytes_f32, 1537 * 4)
        self.assertEqual(mlp16.cost_profile.sram_scratch_bytes_f32, 16 * 4)
        self.assertEqual(mlp16.cost_profile.macs, 94 * 16 + 16)

        mlp32 = train_and_build_mcu_model(
            "mlp_32_16_dot", self.partition, self.src_cfg, fast_cfg, self.cfg.thresholds, 42
        )
        expected_p = 94 * 32 + 32 + 32 * 16 + 16 + 16 + 1
        self.assertEqual(mlp32.cost_profile.n_params, expected_p)
        self.assertEqual(mlp32.cost_profile.flash_bytes_f32, expected_p * 4)
        self.assertEqual(mlp32.cost_profile.sram_scratch_bytes_f32, (32 + 16) * 4)

    def test_mlp_training_and_folding_are_deterministic(self) -> None:
        fast_params = {
            **self.cfg.models.params_by_model,
            "mlp_16_skip_dot": {**self.cfg.models.params_by_model["mlp_16_skip_dot"], "epochs": 20},
        }
        fast_cfg = replace(self.cfg.models, params_by_model=fast_params)
        m_a = train_and_build_mcu_model(
            "mlp_16_skip_dot", self.partition, self.src_cfg, fast_cfg, self.cfg.thresholds, 42
        )
        m_b = train_and_build_mcu_model(
            "mlp_16_skip_dot", self.partition, self.src_cfg, fast_cfg, self.cfg.thresholds, 42
        )
        s_a = m_a.predict_scores_f32(self.partition.X_test)
        s_b = m_b.predict_scores_f32(self.partition.X_test)
        self.assertTrue(np.array_equal(s_a, s_b))

    def test_sparse_l1_zero_nnz_edge_case(self) -> None:
        class _DummyLinear:
            def decision_function(self, X: np.ndarray) -> np.ndarray:
                return np.full(X.shape[0], -0.25, dtype=np.float64)

        zero_model = FoldedSparseLinearModel(
            name="lr_l1_empty",
            reference_model=_DummyLinear(),
            w_eff=np.zeros(94, dtype=np.float64),
            b_eff=-0.25,
            zero_weight_eps=1e-12,
        )
        self.assertEqual(zero_model.cost_profile.n_active_features, 0)
        self.assertEqual(zero_model.cost_profile.macs, 0)
        s = zero_model.predict_scores_f32(self.partition.X_test)
        self.assertTrue(np.allclose(s, -0.25))
        if self.native_lib is not None:
            sc = zero_model.predict_scores_c99(
                self.native_lib, np.ascontiguousarray(self.partition.X_test, dtype=np.float32)
            )
            self.assertTrue(np.allclose(sc, -0.25))


if __name__ == "__main__":
    unittest.main()
