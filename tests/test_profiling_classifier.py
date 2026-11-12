"""Unit tests for classical ML classifier profiling including LinearDotCoreClassifier (lr_dot).

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_profiling_classifier.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.descriptor_classifier import LinearDotCoreClassifier
from src.profiling.classifier_profile import (
    count_classical_model_params,
    estimate_classical_inference_ops,
    profile_classifier,
)


class TestProfilingClassifier(unittest.TestCase):
    """Test parameter counting, analytical FLOP estimates, and latency profiling for lr vs lr_dot."""

    def test_linear_dot_core_parity_params_ops_and_profile(self) -> None:
        rng = np.random.default_rng(42)
        x_train = rng.normal(loc=0.4, scale=0.15, size=(60, 94)).astype(np.float64)
        y_train = (x_train[:, 0] + 0.5 * x_train[:, 1] > 0.6).astype(np.int64)
        x_test = rng.normal(loc=0.4, scale=0.15, size=(20, 94)).astype(np.float64)

        sk_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=1.0,
                        penalty="l2",
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
        sk_pipe.fit(x_train, y_train)

        dot_clf = LinearDotCoreClassifier(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            scale_features=True,
            random_state=42,
        )
        dot_clf.fit(x_train, y_train)

        # Verify sklearn clone & get_params contract
        cloned = clone(dot_clf)
        self.assertIsInstance(cloned, LinearDotCoreClassifier)
        self.assertEqual(cloned.get_params()["max_iter"], 1000)

        # Verify folded weights achieve <=1e-12 numerical parity
        np.testing.assert_allclose(
            dot_clf.decision_function(x_test),
            sk_pipe.decision_function(x_test),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            dot_clf.predict_proba(x_test),
            sk_pipe.predict_proba(x_test),
            atol=1e-12,
        )
        np.testing.assert_array_equal(dot_clf.predict(x_test), sk_pipe.predict(x_test))

        # Verify param and op accounting: scaler folded => scaler_params=0, 188 FLOPs (vs 376)
        sk_params = count_classical_model_params(sk_pipe)
        dot_params = count_classical_model_params(dot_clf)
        self.assertEqual(sk_params.params_model, 95)
        self.assertEqual(sk_params.scaler_params, 188)
        self.assertEqual(dot_params.params_model, 95)
        self.assertEqual(dot_params.scaler_params, 0)

        sk_ops = estimate_classical_inference_ops(sk_pipe, n_features=94)
        dot_ops = estimate_classical_inference_ops(dot_clf, n_features=94)
        self.assertEqual(sk_ops.analytical_macs, 94)
        self.assertEqual(sk_ops.analytical_flops, 376)
        self.assertEqual(dot_ops.analytical_macs, 94)
        self.assertEqual(dot_ops.analytical_flops, 188)

        # Verify profile_classifier runs cleanly and records direct_core_stats + overhead
        rec_dot = profile_classifier(
            model=dot_clf,
            classifier_name="lr_dot",
            X_test=x_test,
            n_train=60,
            threads=1,
            batch_size=1,
            warmup_iters=2,
            timed_iters=20,
            max_real_samples=20,
            bootstrap_resamples=30,
            seed=42,
        )
        self.assertEqual(rec_dot.classifier_name, "lr_dot")
        self.assertIsNone(rec_dot.scaler_stats)
        self.assertIsNone(rec_dot.estimator_only_stats)
        self.assertIsNotNone(rec_dot.direct_core_stats)
        self.assertIsNotNone(rec_dot.sklearn_dispatch_overhead_ms)
        self.assertGreater(rec_dot.predict_stats.median_ms, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
