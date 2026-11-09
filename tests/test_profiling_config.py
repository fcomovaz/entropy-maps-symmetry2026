"""Unit tests for profiling TOML configuration validation and project-root path anchoring.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_profiling_config.py -v
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_profiling_config.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.profiling.config import (
    ProfilingConfig,
    effective_variant_tag,
    load_profiling_config,
    resolve_target_objects,
)


class TestProfilingConfig(unittest.TestCase):
    """Test suite for profiling TOML parsing, fail-fast checks, and path resolution."""

    def test_load_real_profiling_configs(self) -> None:
        """Verify both production profiling TOMLs load cleanly with anchored paths and CPU device."""
        kol_cfg = load_profiling_config(Path("configs/profiling/kolektorsdd2.toml"), PROJECT_ROOT)
        self.assertIsInstance(kol_cfg, ProfilingConfig)
        self.assertEqual(kol_cfg.dataset_key, "kolektorsdd2")
        self.assertEqual(kol_cfg.protocol.device, "cpu")
        self.assertEqual(kol_cfg.protocol.primary_threads, 1)
        self.assertEqual(kol_cfg.protocol.primary_batch_size, 1)
        self.assertIn(1, kol_cfg.protocol.effective_threads_list())
        self.assertTrue(kol_cfg.base_config_path.is_absolute())
        self.assertTrue(kol_cfg.classification_config_path.is_file())
        self.assertTrue(kol_cfg.cnn_config_path.is_file())
        self.assertEqual(resolve_target_objects(kol_cfg), (None,))

        mv_cfg = load_profiling_config(Path("configs/profiling/mvtec_ad.toml"), PROJECT_ROOT)
        self.assertEqual(mv_cfg.dataset_key, "mvtecad")
        self.assertEqual(mv_cfg.protocol.device, "cpu")
        self.assertEqual(resolve_target_objects(mv_cfg), ("bottle",))
        self.assertEqual(
            resolve_target_objects(mv_cfg, objects=["bottle", "tile"]),
            ("bottle", "tile"),
        )
        all_objs = resolve_target_objects(mv_cfg, all_objects=True)
        self.assertEqual(len(all_objs), 15)
        with self.assertRaises(ValueError):
            resolve_target_objects(mv_cfg, object_name="nonexistent_object_xyz")

    def test_effective_variant_tag_formatting(self) -> None:
        """Verify deterministic folder/filename tags for normal and inverted variants."""
        self.assertEqual(effective_variant_tag("KolektorSDD2", None, False), "kolektorsdd2")
        self.assertEqual(effective_variant_tag("KolektorSDD2", None, True), "kolektorsdd2-inverted")
        self.assertEqual(effective_variant_tag("mvtec-ad", "bottle", False), "mvtec-ad_bottle")
        self.assertEqual(
            effective_variant_tag("mvtec-ad", "bottle", True),
            "mvtec-ad-inverted_bottle",
        )

    def test_missing_file_and_missing_key_fail_fast(self) -> None:
        """Verify FileNotFoundError and KeyError trigger immediately on invalid config paths/schemas."""
        with self.assertRaises(FileNotFoundError):
            load_profiling_config(Path("configs/profiling/does_not_exist.toml"), PROJECT_ROOT)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "bad_missing.toml"
            tmp_path.write_text(
                """
[profiling]
base_config = "configs/base.toml"
classification_config = "configs/classification/kolektorsdd2.toml"
cnn_config = "configs/cnn/kolektorsdd2.toml"
entropy_config = "configs/entropy/kolektorsdd2.toml"
asymmetry_config = "configs/asymmetry/kolektorsdd2.toml"
dataset_key = "kolektorsdd2"
                """,
                encoding="utf-8",
            )
            with self.assertRaises(KeyError):
                load_profiling_config(tmp_path, PROJECT_ROOT)

    def test_cuda_device_in_protocol_rejected_fail_fast(self) -> None:
        """Verify primary protocol rejects device='cuda' at config load time."""
        real_toml = (PROJECT_ROOT / "configs/profiling/kolektorsdd2.toml").read_text(encoding="utf-8")
        bad_toml = real_toml.replace('device = "cpu"', 'device = "cuda"')

        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "bad_cuda.toml"
            bad_path.write_text(bad_toml, encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_profiling_config(bad_path, PROJECT_ROOT)
            self.assertIn("cpu", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
