"""Integration tests for the evaluate_cnn and compare_baseline_vs_cnn CLI interfaces.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_cnn_cli.py
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_cnn_cli.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.compare_baseline_vs_cnn import main as compare_main
from experiments.evaluate_cnn import main as evaluate_main


class TestCnnCli(unittest.TestCase):
    """Test suite for validating CLI execution, argument parsing, and fail-fast triggers."""

    def test_evaluate_cnn_dry_run_kolektorsdd2(self) -> None:
        """Verify dry-run execution on KolektorSDD2 returns exit code 0."""
        argv = [
            "--config",
            "configs/cnn/kolektorsdd2.toml",
            "--model",
            "ds_cnn",
            "--dry-run",
        ]
        exit_code = evaluate_main(argv, project_root=PROJECT_ROOT)
        self.assertEqual(exit_code, 0)

    def test_evaluate_cnn_dry_run_mvtec_ad(self) -> None:
        """Verify dry-run execution on MVTec-AD bottle returns exit code 0."""
        argv = [
            "--config",
            "configs/cnn/mvtec_ad.toml",
            "--object",
            "bottle",
            "--model",
            "ultra_tiny",
            "--dry-run",
        ]
        exit_code = evaluate_main(argv, project_root=PROJECT_ROOT)
        self.assertEqual(exit_code, 0)

    def test_unknown_model_raises_system_exit(self) -> None:
        """Verify requesting an unregistered model architecture triggers SystemExit."""
        argv = [
            "--config",
            "configs/cnn/kolektorsdd2.toml",
            "--model",
            "unknown_architecture_xyz",
            "--dry-run",
        ]
        with self.assertRaises(SystemExit) as ctx:
            evaluate_main(argv, project_root=PROJECT_ROOT)
        self.assertIn("error", str(ctx.exception).lower())

    def test_unknown_mvtec_object_raises_system_exit(self) -> None:
        """Verify requesting an invalid MVTec-AD object category triggers SystemExit."""
        argv = [
            "--config",
            "configs/cnn/mvtec_ad.toml",
            "--object",
            "non_existent_object_category",
            "--dry-run",
        ]
        with self.assertRaises(SystemExit) as ctx:
            evaluate_main(argv, project_root=PROJECT_ROOT)
        self.assertIn("error", str(ctx.exception).lower())

    def test_compare_baseline_vs_cnn_cli_help(self) -> None:
        """Verify compare_baseline_vs_cnn executes cleanly with valid dataset flag."""
        argv = [
            "--dataset",
            "kolektorsdd2",
            "--output-dir",
            "results/comparison",
        ]
        exit_code = compare_main(argv, project_root=PROJECT_ROOT)
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
