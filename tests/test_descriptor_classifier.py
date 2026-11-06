"""Unit and integration tests for x94 descriptor classification and explainability pipeline.

Manual execution commands (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe -m unittest tests/test_descriptor_classifier.py
    .\\.venv\\Scripts\\python.exe -m pytest tests/test_descriptor_classifier.py -v
"""
from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.asymmetry_maps import asymmetry_column_names, entropy_column_names
from src.models.descriptor_classifier import (
    ClassificationConfig,
    ClassificationMetrics,
    DatasetPartition,
    ExplainabilityReport,
    GroupImportance,
    build_classifier,
    compute_explainability,
    load_and_split_descriptors,
    load_classification_config,
    resolve_feature_groups,
    run_subset_ablation,
    train_and_evaluate,
)


def _compute_file_hash(path: Path) -> str:
    """Calculate SHA-256 hash of a file's contents."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _create_synthetic_x94_csv(
    path: Path,
    metadata_cols: list[str],
    rows_meta: list[list[str]],
    labels: list[int],
    seed: int = 42,
) -> None:
    """Create a synthetic x94 CSV where class 1 ('defect') has distinct signal in E_4x4_r1_c2 and asym_lr_2x2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scales = (1, 2, 4, 8)
    asym_scales = (2, 4, 8)
    kinds = ("lr", "tb", "rot180")
    templates = {
        "lr": "asym_lr_{g}x{g}",
        "tb": "asym_tb_{g}x{g}",
        "rot180": "asym_rot180_{g}x{g}",
    }

    entropy_cols: list[str] = []
    for g in scales:
        entropy_cols.extend(entropy_column_names(g, "E_{g}x{g}_r{r}_c{c}"))
    asym_cols = asymmetry_column_names(asym_scales, kinds, templates)
    feature_cols = entropy_cols + asym_cols

    header = metadata_cols + ["bins", "grid_levels"] + feature_cols
    rng = np.random.default_rng(seed)

    target_e_idx = feature_cols.index("E_4x4_r1_c2")
    target_a_idx = feature_cols.index("asym_lr_2x2")
    e4_indices = [i for i, c in enumerate(feature_cols) if c.startswith("E_4x4_")]
    asym_indices = [i for i, c in enumerate(feature_cols) if c.startswith("asym_")]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for meta, label in zip(rows_meta, labels):
            vals = rng.uniform(0.18, 0.25, size=len(feature_cols))
            if label == 1:
                for idx in e4_indices:
                    vals[idx] = rng.uniform(0.38, 0.62)
                for idx in asym_indices:
                    vals[idx] = rng.uniform(0.32, 0.55)
                # Dominant peak features (strictly separable) for explainability verification
                vals[target_e_idx] = rng.uniform(0.90, 0.98)
                vals[target_a_idx] = rng.uniform(0.85, 0.95)
            else:
                for idx in e4_indices:
                    vals[idx] = rng.uniform(0.25, 0.48)
                for idx in asym_indices:
                    vals[idx] = rng.uniform(0.18, 0.40)
                vals[target_e_idx] = rng.uniform(0.02, 0.08)
                vals[target_a_idx] = rng.uniform(0.01, 0.05)
            formatted = [f"{v:.8f}" for v in vals]
            writer.writerow(meta + ["16", "1|2|4|8"] + formatted)


def _setup_synthetic_classification_project(tmp: Path) -> tuple[Path, Path]:
    """Create a self-contained project hierarchy in tmp with base, asymmetry, and classification TOMLs."""
    (tmp / "configs" / "asymmetry").mkdir(parents=True, exist_ok=True)
    (tmp / "configs" / "classification").mkdir(parents=True, exist_ok=True)
    (tmp / "data" / "csv").mkdir(parents=True, exist_ok=True)

    base_toml = """[configs]
path = "configs/base.toml"

[datasets.mvtecad]
name = "mvtec-ad"
entropy_csv_path = "data/csv/mvtec-ad-entropy.csv"
entropy_csv_path_inverted = "data/csv/mvtec-ad-inverted-entropy.csv"
full_csv_path = "data/csv/mvtec-ad-entropy-full.csv"
full_csv_path_inverted = "data/csv/mvtec-ad-inverted-entropy-full.csv"
subdatasets = ["bottle", "cable", "tile"]

[datasets.kolektorsdd2]
name = "KolektorSDD2"
entropy_csv_path = "data/csv/kolektorsdd2-entropy.csv"
entropy_csv_path_inverted = "data/csv/kolektorsdd2-inverted-entropy.csv"
full_csv_path = "data/csv/kolektorsdd2-entropy-full.csv"
full_csv_path_inverted = "data/csv/kolektorsdd2-inverted-entropy-full.csv"
splits = ["train", "test"]
"""
    (tmp / "configs" / "base.toml").write_text(base_toml, encoding="utf-8")

    for ds_key, fname in [("mvtecad", "mvtec_ad.toml"), ("kolektorsdd2", "kolektorsdd2.toml")]:
        asym_toml = f"""[asymmetry_features]
base_config = "configs/base.toml"
dataset_key = "{ds_key}"
inverted = false
overwrite = false
schema = "x94-v1"

[asymmetry_features.regions]
scales = [2, 4, 8]
kinds = ["lr", "tb", "rot180"]

[asymmetry_features.source]
feature_name_template = "E_{{g}}x{{g}}_r{{r}}_c{{c}}"
global_scale = 1
bins_column = "bins"
levels_column = "grid_levels"
levels_separator = "|"

[asymmetry_features.output]
suffix = "-full"
column_templates = {{ lr = "asym_lr_{{g}}x{{g}}", tb = "asym_tb_{{g}}x{{g}}", rot180 = "asym_rot180_{{g}}x{{g}}" }}
float_precision = 8
value_tolerance = 1e-6
write_sidecar = true
sidecar_timing_metric = "T_feature_extraction_asymmetry"

[asymmetry_features.runtime]
chunk_size = 128
"""
        (tmp / "configs" / "asymmetry" / fname).write_text(asym_toml, encoding="utf-8")

    kolektor_cls_toml = """[classification]
base_config = "configs/base.toml"
asymmetry_config = "configs/asymmetry/kolektorsdd2.toml"
dataset_key = "kolektorsdd2"
inverted = false
overwrite = true
seed = 42
default_classifier = "rf"

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
compare_subsets = ["x85_only", "x9_only", "x94_full", "E_1x1", "E_2x2", "E_4x4", "E_8x8"]

[classification.models.rf]
n_estimators = 50
max_depth = 6
min_samples_leaf = 1
class_weight = "balanced"
n_jobs = 1

[classification.models.knn]
n_neighbors = 3
weights = "distance"
metric = "minkowski"
p = 2
scale_features = true

[classification.explainability]
enabled = true
permutation_repeats = 5
top_k_features = 10

[classification.output]
results_dir = "results/classification"
dpi = 100
float_precision = 6
"""
    kolektor_cfg_path = Path("configs/classification/kolektorsdd2.toml")
    (tmp / kolektor_cfg_path).write_text(kolektor_cls_toml, encoding="utf-8")

    mvtec_cls_toml = """[classification]
base_config = "configs/base.toml"
asymmetry_config = "configs/asymmetry/mvtec_ad.toml"
dataset_key = "mvtecad"
inverted = false
overwrite = true
seed = 42
default_classifier = "rf"
default_object = "bottle"

[classification.data]
split_column = "split"
label_column = "type"
object_column = "object_name"
positive_label = "defect"
negative_label = "good"
allowed_splits = ["train", "test"]
split_mode = "stratified"
train_split_name = "train"
test_split_name = "test"
test_size = 0.25

[classification.subsets]
active = "x94_full"
compare_subsets = ["x85_only", "x9_only", "x94_full", "E_1x1", "E_2x2", "E_4x4", "E_8x8"]

[classification.models.rf]
n_estimators = 50
max_depth = 6
min_samples_leaf = 1
class_weight = "balanced"
n_jobs = 1

[classification.models.knn]
n_neighbors = 3
weights = "distance"
metric = "minkowski"
p = 2
scale_features = true

[classification.explainability]
enabled = true
permutation_repeats = 5
top_k_features = 10

[classification.output]
results_dir = "results/classification"
dpi = 100
float_precision = 6
"""
    mvtec_cfg_path = Path("configs/classification/mvtec_ad.toml")
    (tmp / mvtec_cfg_path).write_text(mvtec_cls_toml, encoding="utf-8")

    # Populate synthetic KolektorSDD2 CSVs (train and test both have good and defect)
    k_meta: list[list[str]] = []
    k_labels: list[int] = []
    for i in range(24):
        lbl = 1 if i >= 16 else 0
        k_meta.append(["train", "defect" if lbl == 1 else "good", f"train_{i:03d}.png"])
        k_labels.append(lbl)
    for i in range(12):
        lbl = 1 if i >= 8 else 0
        k_meta.append(["test", "defect" if lbl == 1 else "good", f"test_{i:03d}.png"])
        k_labels.append(lbl)

    _create_synthetic_x94_csv(
        tmp / "data" / "csv" / "kolektorsdd2-entropy-full.csv",
        ["split", "type", "file_name"],
        k_meta,
        k_labels,
        seed=42,
    )
    _create_synthetic_x94_csv(
        tmp / "data" / "csv" / "kolektorsdd2-inverted-entropy-full.csv",
        ["split", "type", "file_name"],
        k_meta,
        k_labels,
        seed=99,
    )

    # Populate synthetic MVTec-AD CSVs:
    # train has ONLY 'good' (0), test has 'good' (0) and 'defect' (1),
    # and ground_truth has binary masks labeled 'defect' (which must be excluded!).
    m_meta: list[list[str]] = []
    m_labels: list[int] = []
    for obj_idx, obj_name in enumerate(["bottle", "cable"]):
        for i in range(16):
            m_meta.append([obj_name, "train", "good", "good", f"{i:03d}.png"])
            m_labels.append(0)
        for i in range(8):
            m_meta.append([obj_name, "test", "good", "good", f"t_good_{i:03d}.png"])
            m_labels.append(0)
        for i in range(16):
            m_meta.append([obj_name, "test", "defect", "broken", f"t_def_{i:03d}.png"])
            m_labels.append(1)
        for i in range(16):
            # ground_truth masks: should be filtered out completely!
            m_meta.append([obj_name, "ground_truth", "defect", "broken", f"t_def_{i:03d}_mask.png"])
            m_labels.append(1)

    _create_synthetic_x94_csv(
        tmp / "data" / "csv" / "mvtec-ad-entropy-full.csv",
        ["object_name", "split", "type", "defect_name", "file_name"],
        m_meta,
        m_labels,
        seed=42 + obj_idx,
    )
    _create_synthetic_x94_csv(
        tmp / "data" / "csv" / "mvtec-ad-inverted-entropy-full.csv",
        ["object_name", "split", "type", "defect_name", "file_name"],
        m_meta,
        m_labels,
        seed=142,
    )

    return kolektor_cfg_path, mvtec_cfg_path


class TestDescriptorClassifierConfig(unittest.TestCase):
    """Task 1 tests: TOML configuration loading and defensive validation."""

    def test_load_classification_config_resolves_paths_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            kolektor_cfg_rel, mvtec_cfg_rel = _setup_synthetic_classification_project(tmp)

            cfg_k = load_classification_config(kolektor_cfg_rel, tmp)
            self.assertIsInstance(cfg_k, ClassificationConfig)
            self.assertEqual(cfg_k.dataset_key, "kolektorsdd2")
            self.assertEqual(cfg_k.split_mode, "presplit")
            self.assertFalse(cfg_k.inverted)
            self.assertEqual(
                cfg_k.effective_input_csv,
                (tmp / "data" / "csv" / "kolektorsdd2-entropy-full.csv").resolve(),
            )

            cfg_k_inv = replace(cfg_k, inverted=True)
            self.assertEqual(
                cfg_k_inv.effective_input_csv,
                (tmp / "data" / "csv" / "kolektorsdd2-inverted-entropy-full.csv").resolve(),
            )

            cfg_m = load_classification_config(mvtec_cfg_rel, tmp)
            self.assertEqual(cfg_m.dataset_key, "mvtecad")
            self.assertEqual(cfg_m.split_mode, "stratified")
            self.assertEqual(cfg_m.default_object, "bottle")
            self.assertEqual(cfg_m.subdatasets, ("bottle", "cable", "tile"))

            out_dir = cfg_m.effective_results_dir(object_name="bottle", classifier_name="rf")
            self.assertTrue(str(out_dir).endswith("mvtec-ad\\bottle_rf") or str(out_dir).endswith("mvtec-ad/bottle_rf"))

    def test_load_classification_config_rejects_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            kolektor_cfg_rel, _ = _setup_synthetic_classification_project(tmp)
            cfg_file = tmp / kolektor_cfg_rel
            original_text = cfg_file.read_text(encoding="utf-8")

            # Invalid split_mode
            cfg_file.write_text(
                original_text.replace('split_mode = "presplit"', 'split_mode = "invalid_mode"'),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_classification_config(kolektor_cfg_rel, tmp)

            # Invalid test_size
            cfg_file.write_text(
                original_text.replace("test_size = 0.20", "test_size = 1.50"),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_classification_config(kolektor_cfg_rel, tmp)

            # Missing required key
            cfg_file.write_text(
                original_text.replace("seed = 42\n", ""),
                encoding="utf-8",
            )
            with self.assertRaises(KeyError):
                load_classification_config(kolektor_cfg_rel, tmp)


class TestDescriptorLoadingAndSplitting(unittest.TestCase):
    """Task 2 tests: in-memory filtering, ground_truth exclusion, presplit vs stratified, and subsets."""

    def test_kolektor_presplit_and_feature_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            kolektor_cfg_rel, _ = _setup_synthetic_classification_project(tmp)
            cfg = load_classification_config(kolektor_cfg_rel, tmp)

            part_full = load_and_split_descriptors(cfg, subset_name="x94_full")
            self.assertIsInstance(part_full, DatasetPartition)
            self.assertEqual(part_full.X_train.shape, (24, 94))
            self.assertEqual(part_full.X_test.shape, (12, 94))
            self.assertEqual(part_full.train_counts, {"good": 16, "defect": 8})
            self.assertEqual(part_full.test_counts, {"good": 8, "defect": 4})

            part_x85 = load_and_split_descriptors(cfg, subset_name="x85_only")
            self.assertEqual(part_x85.X_train.shape, (24, 85))
            part_x9 = load_and_split_descriptors(cfg, subset_name="x9_only")
            self.assertEqual(part_x9.X_train.shape, (24, 9))
            part_e4 = load_and_split_descriptors(cfg, subset_name="E_4x4")
            self.assertEqual(part_e4.X_train.shape, (24, 16))

    def test_mvtec_excludes_ground_truth_and_stratifies_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            _, mvtec_cfg_rel = _setup_synthetic_classification_project(tmp)
            cfg = load_classification_config(mvtec_cfg_rel, tmp)
            csv_path = cfg.effective_input_csv
            hash_before = _compute_file_hash(csv_path)

            # For 'bottle': 16 train (good), 8 test (good), 16 test (defect), plus 16 ground_truth (masks).
            # After dropping ground_truth: 24 good + 16 defect = 40 total samples.
            # Stratified split with test_size=0.25 -> 30 train (18 good, 12 defect) and 10 test (6 good, 4 defect).
            part = load_and_split_descriptors(cfg, object_name="bottle")
            self.assertEqual(part.object_name, "bottle")
            self.assertEqual(part.X_train.shape, (30, 94))
            self.assertEqual(part.X_test.shape, (10, 94))
            self.assertEqual(part.train_counts, {"good": 18, "defect": 12})
            self.assertEqual(part.test_counts, {"good": 6, "defect": 4})

            # Verify CSV on disk is 100% untouched
            self.assertEqual(_compute_file_hash(csv_path), hash_before)

            # Verify presplit on MVTec-AD raises ValueError because native train has only 'good' (1 class)
            cfg_presplit = replace(cfg, split_mode="presplit")
            with self.assertRaisesRegex(ValueError, "fewer than 2 unique classes"):
                load_and_split_descriptors(cfg_presplit, object_name="bottle")

            # Verify unknown object raises ValueError listing valid subdatasets
            with self.assertRaisesRegex(ValueError, "Unknown object_name"):
                load_and_split_descriptors(cfg, object_name="unknown_gadget")


class TestClassifierAndAblation(unittest.TestCase):
    """Task 3 tests: RandomForest & k-NN training, metrics, timing separation, and subset ablation."""

    def test_train_and_evaluate_rf_and_knn_and_subset_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            kolektor_cfg_rel, mvtec_cfg_rel = _setup_synthetic_classification_project(tmp)
            cfg_k = load_classification_config(kolektor_cfg_rel, tmp)
            part_k = load_and_split_descriptors(cfg_k)

            # Random Forest
            model_rf, metrics_rf = train_and_evaluate(part_k, "rf", cfg_k)
            self.assertIsInstance(metrics_rf, ClassificationMetrics)
            self.assertEqual(metrics_rf.classifier_name, "rf")
            self.assertEqual(metrics_rf.subset_name, "x94_full")
            self.assertEqual(metrics_rf.n_features, 94)
            self.assertGreaterEqual(metrics_rf.f1, 0.85)
            self.assertGreaterEqual(metrics_rf.roc_auc, 0.85)
            self.assertGreater(metrics_rf.train_elapsed_s, 0.0)
            self.assertGreater(metrics_rf.inference_elapsed_s, 0.0)
            self.assertGreater(metrics_rf.inference_ms_per_sample, 0.0)
            cm_sum = sum(sum(row) for row in metrics_rf.confusion_matrix)
            self.assertEqual(cm_sum, len(part_k.y_test))

            # k-NN (with StandardScaler inside Pipeline)
            model_knn, metrics_knn = train_and_evaluate(part_k, "knn", cfg_k)
            self.assertEqual(metrics_knn.classifier_name, "knn")
            self.assertTrue(hasattr(model_knn, "named_steps"))
            self.assertIn("scaler", model_knn.named_steps)
            self.assertIn("knn", model_knn.named_steps)
            self.assertGreaterEqual(metrics_knn.accuracy, 0.5)

            # Decision Tree
            model_dt, metrics_dt = train_and_evaluate(part_k, "dt", cfg_k)
            self.assertEqual(metrics_dt.classifier_name, "dt")
            self.assertGreaterEqual(metrics_dt.accuracy, 0.5)

            # Logistic Regression (with StandardScaler inside Pipeline)
            model_lr, metrics_lr = train_and_evaluate(part_k, "lr", cfg_k)
            self.assertEqual(metrics_lr.classifier_name, "lr")
            self.assertTrue(hasattr(model_lr, "named_steps"))
            self.assertIn("scaler", model_lr.named_steps)
            self.assertIn("lr", model_lr.named_steps)
            self.assertGreaterEqual(metrics_lr.accuracy, 0.5)

            # Invalid classifier name
            with self.assertRaises(ValueError):
                build_classifier("svm_unknown", cfg_k)

            # Subset ablation on MVTec-AD ('bottle')
            cfg_m = load_classification_config(mvtec_cfg_rel, tmp)
            ablation_results = run_subset_ablation(cfg_m, object_name="bottle", classifier_name="rf")
            subset_names = [m.subset_name for m in ablation_results]
            self.assertEqual(
                subset_names,
                ["x85_only", "x9_only", "x94_full", "E_1x1", "E_2x2", "E_4x4", "E_8x8"],
            )
            by_name = {m.subset_name: m for m in ablation_results}
            self.assertEqual(by_name["x85_only"].n_features, 85)
            self.assertEqual(by_name["x9_only"].n_features, 9)
            self.assertEqual(by_name["x94_full"].n_features, 94)
            self.assertEqual(by_name["E_4x4"].n_features, 16)
            # In synthetic data, signal was planted in E_4x4_r1_c2 and asym_lr_2x2,
            # so E_4x4 and x9_only and x94_full should all achieve high F1
            self.assertGreaterEqual(by_name["x94_full"].f1, 0.80)
            self.assertGreaterEqual(by_name["E_4x4"].f1, 0.80)
            self.assertGreaterEqual(by_name["x9_only"].f1, 0.80)


class TestExplainabilityEngine(unittest.TestCase):
    """Task 4 tests: x85 vs x9 family breakdown, scale breakdown, and 2D spatial importance heatmaps."""

    def test_compute_explainability_family_scale_and_spatial_grids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            _, mvtec_cfg_rel = _setup_synthetic_classification_project(tmp)
            cfg = load_classification_config(mvtec_cfg_rel, tmp)
            part = load_and_split_descriptors(cfg, object_name="bottle")

            model_rf, _ = train_and_evaluate(part, "rf", cfg)
            report_rf = compute_explainability(model_rf, part, cfg)
            self.assertIsInstance(report_rf, ExplainabilityReport)
            self.assertEqual(len(report_rf.feature_names), 94)
            self.assertEqual(report_rf.importances.shape, (94,))
            self.assertEqual(len(report_rf.top_k_indices), cfg.top_k_features)

            top_names = [report_rf.feature_names[i] for i in report_rf.top_k_indices]
            self.assertIn("E_4x4_r1_c2", top_names)
            self.assertIn("asym_lr_2x2", top_names)

            # Family breakdown: x85_entropy (85) + x9_asymmetry (9) -> shares sum to ~100%
            fam_map = {g.group_name: g for g in report_rf.family_breakdown}
            self.assertEqual(set(fam_map.keys()), {"x85_entropy", "x9_asymmetry"})
            self.assertEqual(fam_map["x85_entropy"].n_features, 85)
            self.assertEqual(fam_map["x9_asymmetry"].n_features, 9)
            self.assertAlmostEqual(
                fam_map["x85_entropy"].share_pct + fam_map["x9_asymmetry"].share_pct,
                100.0,
                places=4,
            )
            # Mean importance per feature must equal total_importance / n_features
            for g in report_rf.family_breakdown:
                self.assertIsInstance(g, GroupImportance)
                self.assertAlmostEqual(
                    g.mean_importance_per_feature,
                    g.total_importance / g.n_features,
                    places=10,
                )

            # Scale breakdown: E_1x1, E_2x2, E_4x4, E_8x8, asym_2x2, asym_4x4, asym_8x8
            scale_map = {g.group_name: g for g in report_rf.scale_breakdown}
            self.assertEqual(
                set(scale_map.keys()),
                {"E_1x1", "E_2x2", "E_4x4", "E_8x8", "asym_2x2", "asym_4x4", "asym_8x8"},
            )
            self.assertAlmostEqual(
                sum(g.share_pct for g in report_rf.scale_breakdown),
                100.0,
                places=4,
            )
            # Planted signal is in E_4x4 and asym_*, so E_4x4 should outperform E_8x8 per feature
            self.assertGreater(
                scale_map["E_4x4"].mean_importance_per_feature,
                scale_map["E_8x8"].mean_importance_per_feature,
            )

            # Asymmetry kind breakdown: asym_lr, asym_tb, asym_rot180
            kind_map = {g.group_name: g for g in report_rf.asymmetry_kind_breakdown}
            self.assertEqual(set(kind_map.keys()), {"asym_lr", "asym_tb", "asym_rot180"})

            # Spatial 2D heatmaps: shapes (2,2), (4,4), (8,8), with peak of 4x4 at (1, 2)
            self.assertEqual(set(report_rf.spatial_heatmaps.keys()), {2, 4, 8})
            self.assertEqual(report_rf.spatial_heatmaps[2].shape, (2, 2))
            self.assertEqual(report_rf.spatial_heatmaps[4].shape, (4, 4))
            self.assertEqual(report_rf.spatial_heatmaps[8].shape, (8, 8))
            peak_r, peak_c = np.unravel_index(
                np.argmax(report_rf.spatial_heatmaps[4]),
                report_rf.spatial_heatmaps[4].shape,
            )
            self.assertEqual((int(peak_r), int(peak_c)), (1, 2))

            # Verify k-NN explainability produces non-zero normalized importances and surfaces planted features
            model_knn, _ = train_and_evaluate(part, "knn", cfg)
            report_knn = compute_explainability(model_knn, part, cfg)
            self.assertIn(report_knn.method, {"permutation_importance", "knn_fisher_distance"})
            self.assertEqual(report_knn.importances.shape, (94,))
            self.assertAlmostEqual(float(np.sum(report_knn.importances)), 1.0, places=5)
            knn_top_names = [report_knn.feature_names[i] for i in report_knn.top_k_indices]
            self.assertIn("E_4x4_r1_c2", knn_top_names)
            self.assertIn("asym_lr_2x2", knn_top_names)


class TestPlotsAndCli(unittest.TestCase):
    """Task 5 tests: figure generation, JSON/CSV artifact export, and CLI execution."""

    def test_cli_generates_metrics_and_four_plots_and_supports_dry_run_and_all_objects(
        self,
    ) -> None:
        from experiments.evaluate_descriptors import main as cli_main

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            kolektor_cfg_rel, mvtec_cfg_rel = _setup_synthetic_classification_project(tmp)

            # 1. Dry-run writes nothing
            rc_dry = cli_main(
                ["--config", str(mvtec_cfg_rel), "--object", "bottle", "--dry-run"],
                project_root=tmp,
            )
            self.assertEqual(rc_dry, 0)
            self.assertFalse((tmp / "results" / "classification").exists())

            # 2. Full run on MVTec-AD ('bottle', rf)
            rc_run = cli_main(
                ["--config", str(mvtec_cfg_rel), "--object", "bottle", "--classifier", "rf"],
                project_root=tmp,
            )
            self.assertEqual(rc_run, 0)
            out_dir = tmp / "results" / "classification" / "mvtec-ad" / "bottle_rf"
            self.assertTrue(out_dir.is_dir())

            expected_files = [
                "metrics_summary.json",
                "ablation_table.csv",
                "feature_importance.csv",
                "scale_and_family_breakdown.csv",
                "confusion_matrix_and_curves.png",
                "descriptor_ablation_x85_x9_x94.png",
                "explainability_importance_breakdown.png",
                "spatial_importance_heatmaps.png",
            ]
            for fname in expected_files:
                fpath = out_dir / fname
                self.assertTrue(fpath.is_file(), f"Missing expected artifact: {fpath}")
                self.assertGreater(fpath.stat().st_size, 0, f"Empty artifact: {fpath}")

            # 3. Inverted run on KolektorSDD2 with k-NN
            rc_k = cli_main(
                ["--config", str(kolektor_cfg_rel), "--classifier", "knn", "--inverted"],
                project_root=tmp,
            )
            self.assertEqual(rc_k, 0)
            out_dir_k = tmp / "results" / "classification" / "kolektorsdd2-inverted" / "full_knn"
            for fname in expected_files:
                self.assertTrue((out_dir_k / fname).is_file())

            # 4. --all-objects on MVTec-AD (synthetic has 'bottle' and 'cable' populated; adjust subdatasets to ['bottle', 'cable'])
            base_path = tmp / "configs" / "base.toml"
            base_path.write_text(
                base_path.read_text(encoding="utf-8").replace(
                    'subdatasets = ["bottle", "cable", "tile"]',
                    'subdatasets = ["bottle", "cable"]',
                ),
                encoding="utf-8",
            )
            rc_all = cli_main(
                ["--config", str(mvtec_cfg_rel), "--all-objects", "--classifier", "rf"],
                project_root=tmp,
            )
            self.assertEqual(rc_all, 0)
            summary_csv = (
                tmp / "results" / "classification" / "mvtec-ad" / "mvtec_all_objects_rf_summary.csv"
            )
            summary_png = (
                tmp
                / "results"
                / "classification"
                / "mvtec-ad"
                / "mvtec_all_objects_rf_comparison.png"
            )
            self.assertTrue(summary_csv.is_file())
            self.assertTrue(summary_png.is_file())

            # 5. Error handling: unknown object exits with SystemExit("error: ...")
            with self.assertRaises(SystemExit) as ctx:
                cli_main(
                    ["--config", str(mvtec_cfg_rel), "--object", "nonexistent_obj"],
                    project_root=tmp,
                )
            self.assertTrue(str(ctx.exception).startswith("error:"))

    def test_real_repository_configs_load_and_dry_run(self) -> None:
        from experiments.evaluate_descriptors import main as cli_main

        for rel_cfg in [
            Path("configs/classification/kolektorsdd2.toml"),
            Path("configs/classification/mvtec_ad.toml"),
        ]:
            cfg = load_classification_config(rel_cfg, PROJECT_ROOT)
            if not cfg.effective_input_csv.is_file():
                self.skipTest(f"Real CSV not present: {cfg.effective_input_csv}")
            rc = cli_main(["--config", str(rel_cfg), "--dry-run"], project_root=PROJECT_ROOT)
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
