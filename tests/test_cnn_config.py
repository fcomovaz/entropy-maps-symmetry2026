"""Unit tests for CNN TOML configuration loader and schema validation.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_cnn_config.py
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_cnn_config.py -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn.config import CnnConfig, load_cnn_config


class TestCnnConfig(unittest.TestCase):
    """Test suite for validating CNN TOML parsing, schema enforcement, and path resolution."""

    def test_load_kolektorsdd2_config_valid(self) -> None:
        """Verify standard KolektorSDD2 TOML loads correctly with presplit."""
        cfg_path = Path("configs/cnn/kolektorsdd2.toml")
        cfg = load_cnn_config(cfg_path, PROJECT_ROOT)

        self.assertIsInstance(cfg, CnnConfig)
        self.assertEqual(cfg.dataset_key, "kolektorsdd2")
        self.assertEqual(cfg.data.split_mode, "presplit")
        self.assertEqual(cfg.data.image_size, 128)
        self.assertEqual(cfg.training.epochs, 40)
        self.assertTrue(cfg.preprocessed_dir.is_dir())
        self.assertTrue(cfg.metadata_csv_path.is_file())

    def test_load_mvtec_ad_config_valid(self) -> None:
        """Verify standard MVTec-AD TOML loads correctly with stratified splitting."""
        cfg_path = Path("configs/cnn/mvtec_ad.toml")
        cfg = load_cnn_config(cfg_path, PROJECT_ROOT)

        self.assertIsInstance(cfg, CnnConfig)
        self.assertEqual(cfg.dataset_key, "mvtecad")
        self.assertEqual(cfg.data.split_mode, "stratified")
        self.assertEqual(cfg.default_object, "bottle")
        self.assertIn("bottle", cfg.subdatasets)
        self.assertTrue(cfg.preprocessed_dir.is_dir())
        self.assertTrue(cfg.metadata_csv_path.is_file())

    def test_missing_file_raises_not_found(self) -> None:
        """Verify loading non-existent config file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_cnn_config(Path("configs/cnn/non_existent.toml"), PROJECT_ROOT)

    def test_missing_required_key_raises_key_error(self) -> None:
        """Verify missing mandatory key triggers fail-fast KeyError."""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(
                """
[cnn]
base_config = "configs/base.toml"
dataset_key = "kolektorsdd2"
# Missing seed and models
                """
            )
            temp_path = Path(f.name)

        try:
            with self.assertRaises(KeyError):
                load_cnn_config(temp_path, PROJECT_ROOT)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_invalid_split_mode_raises_value_error(self) -> None:
        """Verify unsupported split_mode triggers ValueError."""
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write(
                """
[cnn]
base_config = "configs/base.toml"
dataset_key = "kolektorsdd2"
seed = 42
seeds = [42]
inverted = false
overwrite = true
default_model = "ds_cnn"
models = ["ds_cnn"]

[cnn.data]
split_column = "split"
label_column = "type"
object_column = ""
positive_label = "defect"
negative_label = "good"
split_mode = "invalid_mode"
train_split_name = "train"
test_split_name = "test"
test_size = 0.20
val_size = 0.15
image_size = 128
normalize_mean = 0.5
normalize_std = 0.5
use_metadata_index = true
metadata_csv_key = "full_csv_path"
metadata_csv_key_inverted = "full_csv_path_inverted"
allowed_splits = ["train", "test"]

[cnn.training]
epochs = 1
batch_size = 8
learning_rate = 1e-3
weight_decay = 1e-4
optimizer = "adamw"
scheduler = "cosine"
early_stopping_patience = 2
class_weighting = "balanced"
num_workers = 0
device = "cpu"
amp = false
deterministic = true

[cnn.augmentation]
hflip = false
vflip = false
rotate_deg = 0
brightness = 0.0

[cnn.benchmark]
warmup_iters = 1
timed_iters = 2
batch_sizes = [1]
devices = ["cpu"]
threads = 1
measure_flops = false

[cnn.output]
results_dir = "results/cnn"
dpi = 160
float_precision = 6
save_checkpoints = false
                """
            )
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_cnn_config(temp_path, PROJECT_ROOT)
            self.assertIn("split_mode", str(ctx.exception))
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
