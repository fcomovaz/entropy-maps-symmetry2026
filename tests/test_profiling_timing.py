"""Unit tests for nanosecond timing, GC/thread guards, memory snapshots, and statistical tests.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_profiling_timing.py -v
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_profiling_timing.py -v
"""

from __future__ import annotations

import gc
from pathlib import Path
import sys
import unittest

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.profiling.timing import (
    LatencyStats,
    MemorySnapshot,
    MultiRunStats,
    StatisticalComparisonResult,
    compare_latency_distributions,
    disabled_gc,
    measure_callable_latencies,
    measure_indexed_latencies,
    measure_memory_alloc,
    summarize_latencies,
    summarize_multi_run,
    thread_limits,
)


class TestProfilingTiming(unittest.TestCase):
    """Test suite verifying timing accuracy, statistical helpers, and isolation guards."""

    def test_summarize_latencies_known_values_and_divisor(self) -> None:
        """Verify quantiles, median, IQR, CI bounds, and batch divisor math."""
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = summarize_latencies(
            samples,
            per_sample_divisor=1,
            bootstrap_resamples=200,
            confidence_level=0.95,
            seed=42,
        )

        self.assertIsInstance(stats, LatencyStats)
        self.assertEqual(stats.n_samples, 5)
        self.assertAlmostEqual(stats.median_ms, 3.0, places=9)
        self.assertAlmostEqual(stats.q25_ms, 2.0, places=9)
        self.assertAlmostEqual(stats.q75_ms, 4.0, places=9)
        self.assertAlmostEqual(stats.iqr_ms, 2.0, places=9)
        self.assertAlmostEqual(stats.mean_ms, 3.0, places=9)
        self.assertAlmostEqual(stats.min_ms, 1.0, places=9)
        self.assertAlmostEqual(stats.max_ms, 5.0, places=9)
        self.assertLessEqual(stats.ci95_low_ms, stats.median_ms)
        self.assertGreaterEqual(stats.ci95_high_ms, stats.median_ms)
        self.assertAlmostEqual(stats.throughput_samples_s, 1000.0 / 3.0, places=6)

        amortized = summarize_latencies(
            samples,
            per_sample_divisor=2,
            bootstrap_resamples=200,
            seed=42,
        )
        self.assertAlmostEqual(amortized.median_ms, 1.5, places=9)
        self.assertAlmostEqual(amortized.iqr_ms, 1.0, places=9)
        self.assertAlmostEqual(amortized.throughput_samples_s, 2000.0 / 3.0, places=6)

    def test_measure_latencies_exact_count_and_non_negative(self) -> None:
        """Verify indexed and callable latency measurements execute N+warmup times and return >= 0."""
        calls: list[int] = []

        def _work(idx: int) -> int:
            calls.append(idx)
            return idx * 2

        res_idx = measure_indexed_latencies(_work, warmup_iters=3, timed_iters=7, disable_gc=True)
        self.assertEqual(len(res_idx), 7)
        self.assertEqual(calls, [0, 1, 2, 0, 1, 2, 3, 4, 5, 6])
        self.assertTrue(all(v >= 0.0 for v in res_idx))

        zero_arg_count = 0

        def _zero_work() -> int:
            nonlocal zero_arg_count
            zero_arg_count += 1
            return zero_arg_count

        res_call = measure_callable_latencies(
            _zero_work, warmup_iters=2, timed_iters=5, disable_gc=True
        )
        self.assertEqual(len(res_call), 5)
        self.assertEqual(zero_arg_count, 7)
        self.assertTrue(all(v >= 0.0 for v in res_call))

    def test_disabled_gc_restores_previous_state(self) -> None:
        """Verify GC guard restores enabled state normally, on exceptions, and when already off."""
        gc_initially_enabled = gc.isenabled()
        gc.enable()
        try:
            with disabled_gc(enabled=True):
                self.assertFalse(gc.isenabled())
            self.assertTrue(gc.isenabled())

            with self.assertRaises(RuntimeError):
                with disabled_gc(enabled=True):
                    self.assertFalse(gc.isenabled())
                    raise RuntimeError("boom inside gc guard")
            self.assertTrue(gc.isenabled())

            gc.disable()
            with disabled_gc(enabled=True):
                self.assertFalse(gc.isenabled())
            self.assertFalse(gc.isenabled())
        finally:
            if gc_initially_enabled:
                gc.enable()
            else:
                gc.disable()

    def test_thread_limits_sets_and_restores_torch_threads(self) -> None:
        """Verify PyTorch thread count is pinned inside context and restored cleanly on exit."""
        orig_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            with thread_limits(1):
                self.assertEqual(torch.get_num_threads(), 1)
            self.assertEqual(torch.get_num_threads(), 2)
        finally:
            torch.set_num_threads(orig_threads)

        with self.assertRaises(ValueError):
            with thread_limits(0):
                pass

    def test_mann_whitney_comparison_and_multi_run_cv(self) -> None:
        """Verify non-parametric significance test and multi-run repeatability stats."""
        fast = [1.0, 1.05, 0.98, 1.02, 1.01, 0.99, 1.03, 1.04]
        slow = [3.0, 3.10, 2.95, 3.05, 3.02, 2.99, 3.08, 3.01]

        comp = compare_latency_distributions(fast, slow, group_a="dt", group_b="cnn", alpha=0.05)
        self.assertIsInstance(comp, StatisticalComparisonResult)
        self.assertTrue(comp.significant)
        self.assertLess(comp.p_value, 0.01)
        self.assertLess(comp.ratio_a_over_b, 0.5)

        same = compare_latency_distributions(fast, fast, group_a="a", group_b="b", alpha=0.05)
        self.assertFalse(same.significant)
        self.assertAlmostEqual(same.ratio_a_over_b, 1.0, places=9)

        multi = summarize_multi_run([2.0, 2.0, 2.0])
        self.assertIsInstance(multi, MultiRunStats)
        self.assertEqual(multi.repeats, 3)
        self.assertEqual(multi.overall_median_ms, 2.0)
        self.assertEqual(multi.inter_run_cv_pct, 0.0)

    def test_measure_memory_alloc_returns_valid_snapshot(self) -> None:
        """Verify psutil RSS and tracemalloc heap telemetry return positive metrics."""
        payload, snap = measure_memory_alloc(lambda: [float(i) for i in range(1000)])
        self.assertEqual(len(payload), 1000)
        self.assertIsInstance(snap, MemorySnapshot)
        self.assertGreater(snap.rss_before_mb, 0.0)
        self.assertGreater(snap.rss_after_mb, 0.0)
        self.assertGreaterEqual(snap.tracemalloc_peak_kb, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
