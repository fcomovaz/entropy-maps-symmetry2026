"""Integration tests for `experiments.benchmark_cpu_fair` and `experiments.profile_pipelines` CLIs.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_benchmark_cpu_fair_cli.py -v
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_benchmark_cpu_fair_cli.py -v
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.benchmark_cpu_fair import main as benchmark_cpu_main
from experiments.profile_pipelines import main as profile_pipelines_main
from src.features.asymmetry_maps import asymmetry_column_names
from src.features.entropy_maps import feature_names
from src.models.cnn import (
    create_model,
    evaluate_cnn,
    load_cnn_config,
    load_cnn_partition,
    save_checkpoint,
    save_metrics_summary,
)
from src.models.cnn.train import TrainResult
from src.profiling.descriptor_stages import extract_x94_end_to_end


def _write_mini_project(root: Path) -> None:
    """Create a self-contained temporary project layout matching production TOML contracts."""
    # 1. Config files
    (root / "configs" / "profiling").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "classification").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "cnn").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "entropy").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "asymmetry").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "preprocessing").mkdir(parents=True, exist_ok=True)

    (root / "configs" / "base.toml").write_text(
        """
[configs]
path = "configs/base.toml"

[datasets.kolektorsdd2]
name = "KolektorSDD2"
input_path = "data/raw/KolektorSDD2"
csv_path = "data/csv/kolektorsdd2.csv"
csv_path_inverted = "data/csv/kolektorsdd2-inverted.csv"
entropy_csv_path = "data/csv/kolektorsdd2-entropy.csv"
entropy_csv_path_inverted = "data/csv/kolektorsdd2-inverted-entropy.csv"
full_csv_path = "data/csv/kolektorsdd2-entropy-full.csv"
full_csv_path_inverted = "data/csv/kolektorsdd2-inverted-entropy-full.csv"
output_path = "data/preprocessed/KolektorSDD2"
output_path_inverted = "data/preprocessed/KolektorSDD2-inverted"
splits = ["train", "test"]
""",
        encoding="utf-8",
    )

    (root / "configs" / "preprocessing" / "kolektorsdd2.toml").write_text(
        """
[preprocessing]
target_size = 32
""",
        encoding="utf-8",
    )

    (root / "configs" / "entropy" / "kolektorsdd2.toml").write_text(
        """
[entropy_maps]
base_config = "configs/base.toml"
dataset_key = "kolektorsdd2"
preprocessing_config = "configs/preprocessing/kolektorsdd2.toml"
inverted = false
overwrite = false

[entropy_maps.regions]
grid_levels = [1, 2, 4, 8]
bins = 16

[entropy_maps.source]
pixel_column_prefix = "px_"
metadata_columns = ["split", "type", "file_name"]
size_columns = ["width", "height"]

[entropy_maps.output]
feature_name_template = "E_{g}x{g}_r{r}_c{c}"
float_precision = 8
write_sidecar = false
bins_column = "bins"
levels_column = "grid_levels"
levels_separator = "|"

[entropy_maps.runtime]
chunk_size = 64
""",
        encoding="utf-8",
    )

    (root / "configs" / "asymmetry" / "kolektorsdd2.toml").write_text(
        """
[asymmetry_features]
base_config = "configs/base.toml"
entropy_config = "configs/entropy/kolektorsdd2.toml"
dataset_key = "kolektorsdd2"
inverted = false
overwrite = false
schema = "x94-v1"

[asymmetry_features.regions]
scales = [2, 4, 8]
kinds = ["lr", "tb", "rot180"]

[asymmetry_features.source]
feature_name_template = "E_{g}x{g}_r{r}_c{c}"
global_scale = 1
bins_column = "bins"
levels_column = "grid_levels"
levels_separator = "|"

[asymmetry_features.output]
suffix = "-full"
column_templates = { lr = "asym_lr_{g}x{g}", tb = "asym_tb_{g}x{g}", rot180 = "asym_rot180_{g}x{g}" }
float_precision = 8
value_tolerance = 1e-6
write_sidecar = false
sidecar_timing_metric = "T_feature_extraction_asymmetry"

[asymmetry_features.runtime]
chunk_size = 64
""",
        encoding="utf-8",
    )

    (root / "configs" / "classification" / "kolektorsdd2.toml").write_text(
        """
[classification]
base_config = "configs/base.toml"
asymmetry_config = "configs/asymmetry/kolektorsdd2.toml"
dataset_key = "kolektorsdd2"
inverted = false
overwrite = true
seed = 42
default_classifier = "dt"

[classification.data]
split_column = "split"
label_column = "type"
object_column = ""
positive_label = "defect"
negative_label = "good"
allowed_splits = ["train", "test"]
split_mode = "presplit"
train_split_name = "train"
test_split_name = "test"
test_size = 0.20

[classification.subsets]
active = "x94_full"
compare_subsets = ["x85_only", "x9_only", "x94_full"]

[classification.models.rf]
n_estimators = 5
max_depth = 4
min_samples_leaf = 1
class_weight = "balanced"
n_jobs = 1

[classification.models.knn]
n_neighbors = 3
weights = "distance"
metric = "minkowski"
p = 2
scale_features = true

[classification.models.dt]
max_depth = 4
min_samples_leaf = 1
class_weight = "balanced"

[classification.models.lr]
C = 1.0
penalty = "l2"
solver = "liblinear"
class_weight = "balanced"
max_iter = 500
scale_features = true

[classification.explainability]
enabled = true
permutation_repeats = 2
top_k_features = 5

[classification.output]
results_dir = "results/classification"
dpi = 80
float_precision = 6
""",
        encoding="utf-8",
    )

    (root / "configs" / "cnn" / "kolektorsdd2.toml").write_text(
        """
[cnn]
base_config = "configs/base.toml"
dataset_key = "kolektorsdd2"
seed = 42
seeds = [42]
inverted = false
overwrite = true
default_model = "ultra_tiny"
models = ["ultra_tiny"]

[cnn.data]
split_column = "split"
label_column = "type"
object_column = ""
positive_label = "defect"
negative_label = "good"
split_mode = "presplit"
train_split_name = "train"
test_split_name = "test"
test_size = 0.20
val_size = 0.20
image_size = 32
normalize_mean = 0.5
normalize_std = 0.5
use_metadata_index = true
metadata_csv_key = "full_csv_path"
metadata_csv_key_inverted = "full_csv_path_inverted"
allowed_splits = ["train", "test"]

[cnn.training]
epochs = 1
batch_size = 4
learning_rate = 1e-3
weight_decay = 1e-4
optimizer = "adamw"
scheduler = "cosine"
early_stopping_patience = 2
class_weighting = "balanced"
threshold_mode = "fixed_05"
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
warmup_iters = 2
timed_iters = 4
batch_sizes = [1, 4]
devices = ["cpu"]
threads = 1
measure_flops = false

[cnn.output]
results_dir = "results/cnn"
dpi = 80
float_precision = 6
save_checkpoints = true
""",
        encoding="utf-8",
    )

    (root / "configs" / "profiling" / "kolektorsdd2.toml").write_text(
        """
[profiling]
base_config = "configs/base.toml"
classification_config = "configs/classification/kolektorsdd2.toml"
cnn_config = "configs/cnn/kolektorsdd2.toml"
entropy_config = "configs/entropy/kolektorsdd2.toml"
asymmetry_config = "configs/asymmetry/kolektorsdd2.toml"
dataset_key = "kolektorsdd2"
inverted = false
overwrite = true
seed = 42
objects = []
default_object = ""
classifiers = ["dt", "lr"]
models = ["ultra_tiny"]

[profiling.protocol]
device = "cpu"
include_cuda_reference = false
primary_threads = 1
threads_list = [1]
include_physical_cores = false
primary_batch_size = 1
batch_sizes = [1, 4]
warmup_iters = 2
timed_iters = 6
batch_timed_iters = 4
repeats = 2
n_real_samples = 6
disable_gc = true

[profiling.descriptor]
image_size = 32
grid_levels = [1, 2, 4, 8]
bins = 16
asymmetry_scales = [2, 4, 8]
asymmetry_kinds = ["lr", "tb", "rot180"]
scalability_image_sizes = [32, 64]
scalability_bins = [8, 16]
scalability_grid_levels = [[1, 2], [1, 2, 4]]
cprofile_top_n = 15

[profiling.cnn]
torch_profiler_warmup = 1
torch_profiler_active = 3
hook_overhead_warn_pct = 25.0
metrics_tolerance = 1e-5
include_checkpoint_load_and_cold_start = true

[profiling.statistics]
bootstrap_resamples = 25
confidence_level = 0.95
mann_whitney_alpha = 0.05
ml_baseline_models = ["dt", "lr"]
cnn_reference_models = ["ultra_tiny"]
ablation_subsets = ["x85_only", "x9_only", "x94_full"]

[profiling.output]
comparison_cpu_dir = "results/comparison_cpu"
profiling_dir = "results/profiling"
dpi = 80
float_precision = 6
run_full_profiling_on_inverted = false
""",
        encoding="utf-8",
    )

    # 2. Generate synthetic PNGs and x94 CSVs (normal & inverted)
    rng = np.random.default_rng(42)
    records = [
        *[("train", "good", f"tr_g_{i}.png") for i in range(12)],
        *[("train", "defect", f"tr_d_{i}.png") for i in range(4)],
        *[("test", "good", f"te_g_{i}.png") for i in range(7)],
        *[("test", "defect", f"te_d_{i}.png") for i in range(3)],
    ]

    e_cols = list(feature_names((1, 2, 4, 8), template="E_{g}x{g}_r{r}_c{c}"))
    a_cols = list(
        asymmetry_column_names(
            (2, 4, 8),
            ("lr", "tb", "rot180"),
            {"lr": "asym_lr_{g}x{g}", "tb": "asym_tb_{g}x{g}", "rot180": "asym_rot180_{g}x{g}"},
        )
    )
    header = ["split", "type", "file_name", "bins", "grid_levels", *e_cols, *a_cols]

    (root / "data" / "csv").mkdir(parents=True, exist_ok=True)
    for inv in (False, True):
        img_root = root / "data" / "preprocessed" / ("KolektorSDD2-inverted" if inv else "KolektorSDD2")
        csv_path = root / "data" / "csv" / (
            "kolektorsdd2-inverted-entropy-full.csv" if inv else "kolektorsdd2-entropy-full.csv"
        )
        rows = []
        for split, lbl, fname in records:
            base_arr = rng.integers(10, 90, size=(32, 32), dtype=np.uint8)
            if lbl == "defect":
                base_arr[4:12, 12:24] = rng.integers(180, 250, size=(8, 12), dtype=np.uint8)
            if inv:
                base_arr = (255 - base_arr).astype(np.uint8)

            out_img = img_root / split / lbl / fname
            out_img.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(base_arr, mode="L").save(out_img)

            x94 = extract_x94_end_to_end(base_arr)[0]
            rows.append([split, lbl, fname, 16, "1|2|4|8", *[f"{v:.8f}" for v in x94.tolist()]])

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    # 3. Create saved CNN checkpoint + matching metrics_summary.json for normal variant
    cnn_cfg = load_cnn_config(Path("configs/cnn/kolektorsdd2.toml"), root)
    part = load_cnn_partition(cnn_cfg, object_name=None, seed=42)
    torch.manual_seed(42)
    mdl = create_model("ultra_tiny", num_classes=2).cpu()
    metrics, _, _, th_info = evaluate_cnn(
        model=mdl,
        partition=part,
        cfg=cnn_cfg,
        model_name="ultra_tiny",
        train_elapsed_s=0.1,
        device=torch.device("cpu"),
        batch_size=4,
        threshold_mode="fixed_05",
    )
    res_dir = cnn_cfg.effective_results_dir(None, "ultra_tiny")
    save_checkpoint(res_dir, mdl.state_dict())
    save_metrics_summary(
        output_dir=res_dir,
        cfg=cnn_cfg,
        model_name="ultra_tiny",
        metrics=metrics,
        train_res=TrainResult(
            history=[], best_epoch=1, best_val_pr_auc=0.5, train_elapsed_s=0.1, best_state_dict=mdl.state_dict()
        ),
        benchmarks=[],
        partition=part,
        seed=42,
        th_info=th_info,
    )


class TestBenchmarkCpuFairCli(unittest.TestCase):
    """End-to-end integration tests for Phase A and Phase B profiling CLIs."""

    def test_benchmark_cpu_fair_dry_run_cuda_rejection_and_missing_checkpoint(self) -> None:
        """Verify dry-run pass, fail-fast CUDA rejection, and fail-fast missing checkpoint error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            _write_mini_project(tmp_root)

            # 1. Dry-run succeeds
            code = benchmark_cpu_main(
                ["--config", "configs/profiling/kolektorsdd2.toml", "--dry-run"],
                project_root=tmp_root,
            )
            self.assertEqual(code, 0)

            # 2. --device cuda strictly rejected
            with self.assertRaises(SystemExit) as ctx_cuda:
                benchmark_cpu_main(
                    ["--config", "configs/profiling/kolektorsdd2.toml", "--device", "cuda"],
                    project_root=tmp_root,
                )
            self.assertIn("cpu", str(ctx_cuda.exception).lower())

            # 3. Missing checkpoint fails fast with actionable evaluate_cnn hint
            ckpt = tmp_root / "results" / "cnn" / "kolektorsdd2" / "full" / "ultra_tiny" / "checkpoint.pt"
            ckpt.unlink()
            with self.assertRaises(SystemExit) as ctx_ckpt:
                benchmark_cpu_main(
                    ["--config", "configs/profiling/kolektorsdd2.toml", "--dry-run"],
                    project_root=tmp_root,
                )
            msg = str(ctx_ckpt.exception).lower()
            self.assertIn("error", msg)
            self.assertIn("evaluate_cnn", msg)

    def test_benchmark_cpu_fair_writes_strict_cpu_csv_summary_and_pareto(self) -> None:
        """Verify Phase A generates strictly CPU comparison CSV, summary CSV, and Pareto figure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            _write_mini_project(tmp_root)

            code = benchmark_cpu_main(
                [
                    "--config",
                    "configs/profiling/kolektorsdd2.toml",
                    "--threads",
                    "1",
                    "--batch-sizes",
                    "1",
                    "4",
                    "--no-cuda-reference",
                ],
                project_root=tmp_root,
            )
            self.assertEqual(code, 0)

            out_dir = tmp_root / "results" / "comparison_cpu"
            cpu_csv = out_dir / "kolektorsdd2_baseline_vs_cnn_cpu.csv"
            sum_csv = out_dir / "summary_all_datasets.csv"
            png_1 = out_dir / "kolektorsdd2_pareto_threads1.png"

            self.assertTrue(cpu_csv.is_file())
            self.assertTrue(sum_csv.is_file())
            self.assertTrue(png_1.is_file())

            with cpu_csv.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            # (2 ML + 1 CNN) * 2 batch sizes = 6 rows
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(r["device"] == "cpu" for r in rows))
            self.assertEqual({r["family"] for r in rows}, {"ml", "cnn"})

            for r in rows:
                if r["family"] == "ml":
                    self.assertGreater(float(r["t_descriptor_ms_median"]), 0.0)
                    self.assertEqual(float(r["t_preprocess_ms_median"]), 0.0)
                else:
                    self.assertEqual(float(r["t_descriptor_ms_median"]), 0.0)
                    self.assertGreaterEqual(float(r["t_preprocess_ms_median"]), 0.0)

    def test_profile_pipelines_dry_run_full_and_short_inverted(self) -> None:
        """Verify Phase B CLI dry-run, full artifact suite, and automatic short inverted check."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            _write_mini_project(tmp_root)

            self.assertEqual(
                profile_pipelines_main(
                    ["--config", "configs/profiling/kolektorsdd2.toml", "--dry-run"],
                    project_root=tmp_root,
                ),
                0,
            )

            # Full Phase B run on normal variant
            self.assertEqual(
                profile_pipelines_main(
                    [
                        "--config",
                        "configs/profiling/kolektorsdd2.toml",
                        "--threads",
                        "1",
                        "--no-cuda-reference",
                    ],
                    project_root=tmp_root,
                ),
                0,
            )
            prof_dir = tmp_root / "results" / "profiling" / "kolektorsdd2"
            expected_files = (
                "environment.json",
                "comparison_cpu.csv",
                "descriptor_stage_times.csv",
                "cprofile_descriptor.txt",
                "descriptor_scalability.csv",
                "classifier_times.csv",
                "classifier_thread_scaling.csv",
                "ablation_quality_vs_cost.csv",
                "cnn_layer_times_ultra_tiny.csv",
                "cnn_aten_ops_ultra_tiny.csv",
                "cnn_aten_table_ultra_tiny.txt",
                "cnn_architectures_summary.csv",
                "cnn_chrome_trace_ultra_tiny.json",
                "cnn_thread_scaling.csv",
                "statistical_tests.csv",
                "fig1_x94_stage_breakdown.png",
                "fig2_cnn_layer_heatmap.png",
                "fig3_pareto_cpu_1t_b1.png",
                "fig4_thread_scaling.png",
                "fig5_descriptor_scalability.png",
                "fig6_throughput_batch32.png",
                "README.md",
            )
            for fname in expected_files:
                self.assertTrue((prof_dir / fname).is_file(), f"Missing expected artifact: {fname}")

            env_data = json.loads((prof_dir / "environment.json").read_text(encoding="utf-8"))
            self.assertIn("torch_version", env_data)
            self.assertEqual(env_data["seed"], 42)

            # Short descriptor verification on inverted variant (no inverted checkpoints needed)
            self.assertEqual(
                profile_pipelines_main(
                    ["--config", "configs/profiling/kolektorsdd2.toml", "--inverted", "--threads", "1"],
                    project_root=tmp_root,
                ),
                0,
            )
            inv_dir = tmp_root / "results" / "profiling" / "kolektorsdd2-inverted"
            self.assertTrue((inv_dir / "descriptor_stage_times.csv").is_file())
            self.assertTrue((inv_dir / "README.md").is_file())
            self.assertIn("Short Descriptor Check", (inv_dir / "README.md").read_text(encoding="utf-8"))

    def test_real_workspace_dry_run_if_data_and_checkpoints_present(self) -> None:
        """Smoke-test dry-run on real repository KolektorSDD2 artifacts when available locally."""
        csv_ok = (PROJECT_ROOT / "data/csv/kolektorsdd2-entropy-full.csv").is_file()
        ckpt_ok = (PROJECT_ROOT / "results/cnn/kolektorsdd2/full/ds_cnn/checkpoint.pt").is_file()
        if not (csv_ok and ckpt_ok):
            self.skipTest("Real KolektorSDD2 CSV or checkpoints not present on disk")

        self.assertEqual(
            benchmark_cpu_main(
                ["--config", "configs/profiling/kolektorsdd2.toml", "--dry-run"],
                project_root=PROJECT_ROOT,
            ),
            0,
        )
        self.assertEqual(
            profile_pipelines_main(
                ["--config", "configs/profiling/kolektorsdd2.toml", "--dry-run"],
                project_root=PROJECT_ROOT,
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
