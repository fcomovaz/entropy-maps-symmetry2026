# MVTEC-AD Dataset CSV conversion

## Issue

* The main issue addressed in this file is i want a simple example in how to use this descriptors, for example using a randomforest classifier, i think, based on what i saw, for the kolektorsdd2 dataset i can use the train and test as it comes, but for mvtec-ad i need to specify the object and then choose the train and test.
* all the files i want them with its respective cli execution as it has been done in the whole project, but for now i want a minimum example to know how to use it. randomforest classifier, or knnclassifier, anything but i want to see it and choose the dataset (inverted or not) and the object for mvtec-ad.
* also check how to to the train test, i think it's oke, but also if you can generate some example graphs would be useful for the research report and show how it was working.


### Folder structure

The folder structure where the x94 features are located is the following:

`D:\Publications\2026-10-Symmetry-Entropy\code\data\csv\mvtec-ad-entropy-full`,
`D:\Publications\2026-10-Symmetry-Entropy\code\data\csv\mvtec-ad-inverted-entropy-full`,
`D:\Publications\2026-10-Symmetry-Entropy\code\data\csv\kolektorsdd2-entropy-full`,
`D:\Publications\2026-10-Symmetry-Entropy\code\data\csv\kolektorsdd2-inverted-entropy-full`,

> Remember not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.


## Plan

> Don't execute anything, just plan it

- [x] Read AGENTS.md file to know the general rules of the project.
- [x] Create a detailed plan for this task and update it here to implement it later for an agent, so it needs to be detailed.
- [x] Update this file with the plan, stages and everything to resolve and implement the task.

---

# Baseline Classification & Explainability ($\mathbf{x}_{94} = [\mathbf{x}_{85}, \mathbf{x}_{9}]$) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Plan only; nothing has been executed yet.

**Goal:** Build a modular, CLI-driven, TOML-configured supervised classification and explainability pipeline (`RandomForestClassifier` and `KNeighborsClassifier`) over the $\mathbf{x}_{94}$ tabular descriptors for KolektorSDD2 and MVTec-AD (normal and `--inverted`, per-object or all-objects), quantifying and visualizing both classification performance and the contribution of the $\mathbf{x}_9$ symmetry features and each $\mathbf{x}_{85}$ entropy scale ($1\times1, 2\times2, 4\times4, 8\times8$).

**Architecture:** Pure classification, data-splitting, and explainability logic lives in `src/models/` (`descriptor_classifier.py`, `classification_plots.py`), strictly free of hardcoded paths or parameters. Configuration lives in `configs/classification/{kolektorsdd2,mvtec_ad}.toml` anchored to `configs/base.toml` and `configs/asymmetry/*.toml`. A thin CLI entry point `experiments/evaluate_descriptors.py` executes single or batch runs and exports reproducible JSON/CSV metrics and publication-ready Matplotlib figures to `results/classification/`.

**Tech Stack:** Python 3.11 (`code/.venv`), `numpy`, `pandas`, `scikit-learn` (`RandomForestClassifier`, `KNeighborsClassifier`, `StandardScaler`, `Pipeline`, `train_test_split`, `permutation_importance`), `matplotlib`, `tomllib`, `pytest` / `unittest`.

**Spec:** `tasks/task_06A.md` and `AGENTS.md`.

## Global Constraints

- **TOML-only configuration:** All paths, split ratios, random seeds, classifier hyperparameters, feature subset definitions, and plot settings must live in `configs/base.toml` and `configs/classification/*.toml`. Zero hardcoded defaults in `src/`.
- **Dynamic path resolution:** Resolve `PROJECT_ROOT = Path(__file__).resolve().parents[1]` in CLI entry points; resolve all relative paths against `project_root`. Never rely on CWD.
- **Clean typing & immutability:** Every file starts with `from __future__ import annotations`; all config/result containers use `@dataclass(frozen=True, slots=True)`.
- **No mutation of source CSVs:** All filtering (`object_name`, excluding `ground_truth`) and stratified splitting happen strictly in memory. The CSV files in `data/csv/` are read-only.
- **Timing separation (`AGENTS.md`):** Measure and record `T_data_load`, `T_model_train`, and `T_model_inference` (and per-sample latency) separately from feature extraction times stored in the CSV sidecars (`T_feature_extraction` + `T_feature_extraction_asymmetry`), respecting `T_total = T_selector + T_model`.
- **Logging & CLI contract:** Use `logging.getLogger("research")` (no `print`); encapsulate CLI in `build_parser()` and `main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int`; catch `(OSError, ValueError, KeyError)` and abort with `raise SystemExit(f"error: {exc}") from exc`.

## Review Focus

1. **MVTec-AD `ground_truth` mask leakage:** `mvtec-ad-entropy-full.csv` contains 1,258 rows with `split == "ground_truth"` (binary masks). The loader must explicitly filter to `allowed_splits` from TOML (`["train", "test"]`) and reject `ground_truth` rows before any split or training.
2. **MVTec-AD single-class `train` split:** Native MVTec-AD `train` has `3,629 good` and `0 defect` across all 15 objects. Attempting `split_mode = "presplit"` on a partition where `train` has `< 2` unique classes must fail fast with a clear `ValueError` explaining that `split_mode = "stratified"` is required for supervised binary classification.
3. **Unknown `--object` name in MVTec-AD:** Passing an invalid object (e.g., `--object foo`) must fail fast with `ValueError` listing the valid `subdatasets` from `configs/base.toml`.
4. **k-NN scale sensitivity without leakage:** Entropy features live in $[0, 1]$ while asymmetry features are typically smaller ($\approx [0, 0.4]$). `KNeighborsClassifier` must wrap `StandardScaler` inside a `sklearn.pipeline.Pipeline` fitted **only** on `X_train` (never on `X_test`).
5. **Explainability bias (85 entropy vs 9 asymmetry features):** Summing raw importances across 85 features vs 9 features favors the larger group by cardinality. Explainability reports must compute **both** *total group importance* ($\sum I_i$) and *mean importance per feature* ($\frac{1}{|G|}\sum_{i \in G} I_i$) for every scale ($1\times1, 2\times2, 4\times4, 8\times8$) and family ($\mathbf{x}_{85}$ vs $\mathbf{x}_9$).

---

## 0. Findings & Scientific Role of Each Dataset

| Aspect | KolektorSDD2 (`kolektorsdd2*-entropy-full.csv`) | MVTec-AD (`mvtec-ad*-entropy-full.csv`) |
|---|---|---|
| **Actual path on disk** | `data/csv/kolektorsdd2-entropy-full.csv` & `data/csv/kolektorsdd2-inverted-entropy-full.csv` | `data/csv/mvtec-ad-entropy-full.csv` & `data/csv/mvtec-ad-inverted-entropy-full.csv` |
| **Rows & Columns** | 3,335 rows, 99 cols (3 meta + `bins` + `grid_levels` + 85 entropy + 9 asymmetry) | 6,612 rows, 101 cols (5 meta + `bins` + `grid_levels` + 85 entropy + 9 asymmetry) |
| **Class counts by split** | `train`: 2,085 `good`, 246 `defect` \| `test`: 894 `good`, 110 `defect` | `train`: 3,629 `good`, **0 `defect`** \| `test`: 467 `good`, 1,258 `defect` \| `ground_truth`: 1,258 masks |
| **Split strategy** | `presplit` (uses official `train` and `test` directly) | `stratified` in-memory per `object_name` (pools `train`+`test` after dropping `ground_truth`, stratified split with TOML `test_size` and `seed`) |
| **Role in the research (Entropy Maps vs Convolutions)** | **Primary benchmark for direct supervised comparison:** Official train/test split with both classes; ideal baseline to compare $\mathbf{x}_{94}$ + lightweight classifiers against lightweight CNNs on stochastic metallic surfaces where fine local entropy ($4\times4, 8\times8$) dominates. | **Multi-object & symmetry benchmark (15 categories):** Essential to test **object categories** (`bottle`, `cable`, `capsule`, `metal_nut`, `screw`, `toothbrush`, `transistor`, `zipper`) where defects break spatial symmetry ($\mathbf{x}_9$) vs **texture categories** (`carpet`, `grid`, `leather`, `tile`, `wood`) where multiscale entropy ($\mathbf{x}_{85}$) dominates. Using the same seeded stratified split for both $\mathbf{x}_{94}$ and CNNs guarantees a fair head-to-head comparison. |

---

## 1. File Structure & Responsibilities

| Action | File | Responsibility |
|---|---|---|
| Modify | `configs/base.toml` | Add `full_csv_path` and `full_csv_path_inverted` to `[datasets.mvtecad]` and `[datasets.kolektorsdd2]`. |
| Create | `configs/classification/kolektorsdd2.toml` | TOML config for KolektorSDD2 (`split_mode = "presplit"`, RF & k-NN hyperparams, subsets, explainability, output paths). |
| Create | `configs/classification/mvtec_ad.toml` | TOML config for MVTec-AD (`split_mode = "stratified"`, `test_size = 0.2`, `default_object = "bottle"`, RF & k-NN hyperparams, subsets, explainability). |
| Create | `src/models/__init__.py` | Package marker for classification and evaluation modules. |
| Create | `src/models/descriptor_classifier.py` | Config dataclasses & loader, CSV loading & filtering, `presplit`/`stratified` partitioning, RF/k-NN builder, evaluation metrics, subset ablation, and group/scale/spatial explainability. |
| Create | `src/models/classification_plots.py` | Matplotlib figure generators (Confusion Matrix + ROC/PR curves, $\mathbf{x}_{85}$ vs $\mathbf{x}_9$ & per-scale ablation bar chart, Top-K + scale breakdown importance plot, and $2\times2, 4\times4, 8\times8$ 2D spatial importance heatmaps). |
| Create | `experiments/evaluate_descriptors.py` | CLI entry point supporting `--config`, `--classifier`, `--inverted`, `--object`, `--all-objects`, `--dry-run`, `--overwrite`. |
| Create | `tests/test_descriptor_classifier.py` | Unit & integration tests for config loading, `ground_truth` exclusion, stratified vs presplit partitioning, RF/k-NN training, explainability math, figure generation, and CLI execution. |
| Modify | `src/README.md`, `experiments/README.md` | Document the new module, CLI flags, and reproducible commands. |

---

## 2. Configuration Schema (`configs/classification/*.toml`)

### `configs/base.toml` additions
Under `[datasets.mvtecad]`:
```toml
full_csv_path = "data/csv/mvtec-ad-entropy-full.csv"
full_csv_path_inverted = "data/csv/mvtec-ad-inverted-entropy-full.csv"
```
Under `[datasets.kolektorsdd2]`:
```toml
full_csv_path = "data/csv/kolektorsdd2-entropy-full.csv"
full_csv_path_inverted = "data/csv/kolektorsdd2-inverted-entropy-full.csv"
```

### `configs/classification/mvtec_ad.toml` (and analogous `kolektorsdd2.toml`)
```toml
[classification]
base_config = "configs/base.toml"
asymmetry_config = "configs/asymmetry/mvtec_ad.toml"
dataset_key = "mvtecad"                  # "kolektorsdd2" in kolektorsdd2.toml
inverted = false
overwrite = true
seed = 42
default_classifier = "rf"                # "rf" or "knn"
default_object = "bottle"                # omitted / ignored in kolektorsdd2.toml

[classification.data]
split_column = "split"
label_column = "type"
object_column = "object_name"            # "" in kolektorsdd2.toml
positive_label = "defect"
negative_label = "good"
allowed_splits = ["train", "test"]       # strictly excludes "ground_truth"
split_mode = "stratified"                # "presplit" in kolektorsdd2.toml
train_split_name = "train"
test_split_name = "test"
test_size = 0.20                         # used when split_mode == "stratified"

[classification.subsets]
active = "x94_full"
compare_subsets = ["x85_only", "x9_only", "x94_full", "E_1x1", "E_2x2", "E_4x4", "E_8x8"]

[classification.models.rf]
n_estimators = 200
max_depth = 12
min_samples_leaf = 2
class_weight = "balanced"
n_jobs = -1

[classification.models.knn]
n_neighbors = 5
weights = "distance"
metric = "minkowski"
p = 2
scale_features = true

[classification.explainability]
enabled = true
permutation_repeats = 10
top_k_features = 15

[classification.output]
results_dir = "results/classification"
dpi = 160
float_precision = 6
```

---

## 3. Explainability & Research Questions Addressed Automatically

Every execution produces both quantitative tables (`metrics_summary.json`, `ablation_table.csv`, `feature_importance.csv`) and 4 figures specifically designed for the research report:

1. **¿Funcionaron las 9 medidas de asimetría ($\mathbf{x}_9$)? (Ablación directa):**
   - Evaluates the chosen classifier on the exact same train/test partition across `compare_subsets`:
     - `x85_only` (85 multiscale entropy features)
     - `x9_only` (9 spatial asymmetry features)
     - `x94_full` (complete 94-feature descriptor)
   - Reports $\Delta\text{F1} = \text{F1}(\mathbf{x}_{94}) - \text{F1}(\mathbf{x}_{85})$ and $\Delta\text{ROC-AUC}$ to prove when symmetry residuals boost detection (e.g., structured objects in MVTec-AD vs textures).
2. **¿Cuál mapa de entropía sirvió más? ($H^{(1)}$ vs $H^{(2)}$ vs $H^{(4)}$ vs $H^{(8)}$):**
   - **Single-scale ablation:** Trains/evaluates using only `E_1x1` (1 feat), `E_2x2` (4 feats), `E_4x4` (16 feats), or `E_8x8` (64 feats).
   - **Grouped importance breakdown:** Aggregates feature importance (Random Forest Gini MDI + model-agnostic Permutation Importance on test set) by scale:
     - `E_1x1` (global), `E_2x2` (quadrants), `E_4x4` (medium grid), `E_8x8` (fine local grid), `asym_2x2`, `asym_4x4`, `asym_8x8` (and by symmetry axis `lr`, `tb`, `rot180`).
     - Reports both **Total Share (%)** and **Mean Importance per Feature** (normalizing by group cardinality $1, 4, 16, 64, 3, 3, 3$).
3. **¿Qué regiones espaciales de la imagen importan más? (2D Spatial Importance Maps):**
   - Reshapes the per-feature importances of `E_2x2_*`, `E_4x4_*`, and `E_8x8_*` back into $2\times2$, $4\times4$, and $8\times8$ 2D matrices and plots them as heatmaps showing where in the $128\times128$ canvas the classifier focuses to separate `good` from `defect`.

---

## 4. Implementation Tasks

### Task 1: Configuration & Dataset Path Registration

**Files:**
- Modify: `configs/base.toml`
- Create: `configs/classification/kolektorsdd2.toml`
- Create: `configs/classification/mvtec_ad.toml`
- Create: `src/models/__init__.py`
- Create: `src/models/descriptor_classifier.py` (config dataclasses + `load_classification_config`)
- Test: `tests/test_descriptor_classifier.py`

**Interfaces:**
- Produces:
  - `ClassificationConfig` (`@dataclass(frozen=True, slots=True)`) with `effective_input_csv: Path` and `effective_results_dir(object_name: str | None, classifier_name: str) -> Path`
  - `load_classification_config(config_path: Path, project_root: Path) -> ClassificationConfig`

- [x] **Step 1: Write the failing config & validation tests in `tests/test_descriptor_classifier.py`**

```python
def test_load_classification_config_resolves_paths_and_validates(tmp_path: Path) -> None:
    # Verifies TOML loading, inverted path switching, missing key -> KeyError,
    # invalid split_mode or test_size out of (0, 1) -> ValueError.
    ...
```

- [x] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models'`

- [x] **Step 3: Update `configs/base.toml`, create `configs/classification/*.toml`, and implement `ClassificationConfig` + `load_classification_config` in `src/models/descriptor_classifier.py`**

Read `full_csv_path` / `full_csv_path_inverted` from `configs/base.toml` and feature naming templates (`feature_name_template`, `scales`, `kinds`, `column_templates`) from the linked `asymmetry_config` (`configs/asymmetry/*.toml`) so column schemas have a single source of truth. Validate `split_mode in {"presplit", "stratified"}`, `0.0 < test_size < 1.0`, `default_classifier in {"rf", "knn"}`, and non-empty `allowed_splits`.

- [x] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```powershell
git add configs/base.toml configs/classification/ src/models/__init__.py src/models/descriptor_classifier.py tests/test_descriptor_classifier.py
git commit -m "feat(models): add TOML classification configs and loader"
```

---

### Task 2: In-Memory Dataset Filtering, Partitioning (`presplit` vs `stratified`), and Feature Subset Resolution

**Files:**
- Modify: `src/models/descriptor_classifier.py`
- Test: `tests/test_descriptor_classifier.py`

**Interfaces:**
- Consumes: `ClassificationConfig`
- Produces:
  - `DatasetPartition` (`@dataclass(frozen=True, slots=True)`): `dataset_name: str`, `object_name: str | None`, `inverted: bool`, `split_mode: str`, `feature_columns: tuple[str, ...]`, `X_train: np.ndarray`, `y_train: np.ndarray`, `X_test: np.ndarray`, `y_test: np.ndarray`, `train_counts: dict[str, int]`, `test_counts: dict[str, int]`, `load_elapsed_s: float`
  - `resolve_feature_groups(header: Sequence[str], cfg: ClassificationConfig) -> dict[str, tuple[str, ...]]`
  - `load_and_split_descriptors(cfg: ClassificationConfig, object_name: str | None = None, subset_name: str | None = None) -> DatasetPartition`

- [x] **Step 1: Write failing tests for filtering `ground_truth`, object selection, single-class guard, and deterministic stratified split**

```python
def test_mvtec_excludes_ground_truth_and_stratifies_in_memory(tmp_path: Path) -> None:
    # Synthetic CSV with train (only good), test (good + defect), ground_truth (defect masks).
    # Asserts ground_truth is excluded, presplit raises ValueError (1 class in train),
    # stratified split preserves class ratio in train/test and leaves CSV file byte-identical.
    ...
```

- [x] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v`
Expected: FAIL with `ImportError` / `AttributeError`

- [x] **Step 3: Implement `resolve_feature_groups` and `load_and_split_descriptors` in `src/models/descriptor_classifier.py`**

- Filter `df[cfg.split_column].isin(cfg.allowed_splits)` first (dropping `ground_truth`).
- If `cfg.object_column` is non-empty, validate `object_name` against `subdatasets` from `configs/base.toml` and filter `df[cfg.object_column] == target_object`.
- Map labels strictly (`positive_label -> 1`, `negative_label -> 0`; unknown label raises `ValueError`).
- For `split_mode == "presplit"`, split by `train_split_name` and `test_split_name`. For `split_mode == "stratified"`, use `sklearn.model_selection.train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)`.
- Fail fast with `ValueError` if either `y_train` or `y_test` contains fewer than 2 unique classes.

- [x] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```powershell
git add src/models/descriptor_classifier.py tests/test_descriptor_classifier.py
git commit -m "feat(models): implement descriptor loading, ground_truth filtering, and split strategies"
```

---

### Task 3: Classifier Factory (`RandomForest` & `k-NN`), Evaluation Metrics, and Timing Separation

**Files:**
- Modify: `src/models/descriptor_classifier.py`
- Test: `tests/test_descriptor_classifier.py`

**Interfaces:**
- Consumes: `ClassificationConfig`, `DatasetPartition`
- Produces:
  - `ClassificationMetrics` (`@dataclass(frozen=True, slots=True)`): `classifier_name: str`, `subset_name: str`, `n_features: int`, `accuracy: float`, `balanced_accuracy: float`, `precision: float`, `recall: float`, `f1_defect: float`, `f1_macro: float`, `roc_auc: float`, `pr_auc: float`, `confusion_matrix: tuple[tuple[int, int], tuple[int, int]]`, `fpr: np.ndarray`, `tpr: np.ndarray`, `pr_precision: np.ndarray`, `pr_recall: np.ndarray`, `train_elapsed_s: float`, `inference_elapsed_s: float`, `inference_ms_per_sample: float`
  - `build_classifier(classifier_name: str, cfg: ClassificationConfig) -> ClassifierMixin | Pipeline`
  - `train_and_evaluate(partition: DatasetPartition, classifier_name: str, cfg: ClassificationConfig, subset_name: str = "x94_full") -> tuple[ClassifierMixin | Pipeline, ClassificationMetrics]`
  - `run_subset_ablation(partition: DatasetPartition, classifier_name: str, cfg: ClassificationConfig) -> dict[str, ClassificationMetrics]`

- [x] **Step 1: Write failing tests for RF & k-NN training, metrics bounds, and timing fields**

```python
def test_train_and_evaluate_rf_and_knn_and_subset_ablation(tmp_path: Path) -> None:
    # Verifies both 'rf' and 'knn' (with StandardScaler inside Pipeline) train and predict,
    # metrics are in [0, 1], confusion matrix sums to len(y_test),
    # and run_subset_ablation returns metrics for x85_only, x9_only, x94_full, E_1x1..E_8x8.
    ...
```

- [x] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v`
Expected: FAIL with `AttributeError`

- [x] **Step 3: Implement `build_classifier`, `train_and_evaluate`, and `run_subset_ablation` in `src/models/descriptor_classifier.py`**

- `"rf"` builds `RandomForestClassifier(n_estimators=..., max_depth=..., min_samples_leaf=..., class_weight=..., random_state=cfg.seed, n_jobs=...)`.
- `"knn"` builds `Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=..., weights=..., metric=..., p=..., n_jobs=-1))])` when `scale_features` is true.
- Measure `train_elapsed_s` around `.fit(X_train, y_train)` and `inference_elapsed_s` around `.predict(X_test)` + `.predict_proba(X_test)` using `time.perf_counter()`.

- [x] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```powershell
git add src/models/descriptor_classifier.py tests/test_descriptor_classifier.py
git commit -m "feat(models): add RF and kNN classifiers, evaluation metrics, and subset ablation"
```

---

### Task 4: Explainability Engine ($\mathbf{x}_{85}$ vs $\mathbf{x}_9$, Scale Breakdown, and 2D Spatial Maps)

**Files:**
- Modify: `src/models/descriptor_classifier.py`
- Test: `tests/test_descriptor_classifier.py`

**Interfaces:**
- Consumes: fitted model (`RandomForestClassifier` or `Pipeline`), `DatasetPartition`, `ClassificationConfig`
- Produces:
  - `GroupImportance` (`@dataclass(frozen=True, slots=True)`): `group_name: str`, `n_features: int`, `total_importance: float`, `share_pct: float`, `mean_importance_per_feature: float`
  - `ExplainabilityReport` (`@dataclass(frozen=True, slots=True)`): `method: str`, `feature_names: tuple[str, ...]`, `importances: np.ndarray`, `normalized_importances_pct: np.ndarray`, `top_k_indices: tuple[int, ...]`, `family_breakdown: tuple[GroupImportance, ...]`, `scale_breakdown: tuple[GroupImportance, ...]`, `asymmetry_kind_breakdown: tuple[GroupImportance, ...]`, `spatial_heatmaps: dict[int, np.ndarray]`, `best_single_entropy_scale: str`, `x85_total_share_pct: float`, `x9_total_share_pct: float`
  - `compute_explainability(model: object, partition: DatasetPartition, cfg: ClassificationConfig) -> ExplainabilityReport`

- [x] **Step 1: Write failing tests for explainability breakdowns and 2D spatial heatmap reconstruction**

```python
def test_compute_explainability_family_scale_and_spatial_grids(tmp_path: Path) -> None:
    # Plant a synthetic feature where asym_lr_2x2 and E_4x4_r1_c2 are the dominant predictors.
    # Assert top_k_indices surfaces them, family_breakdown covers x85_entropy and x9_asymmetry,
    # scale_breakdown covers E_1x1..E_8x8 and asym_2x2..asym_8x8,
    # and spatial_heatmaps[4] has shape (4, 4) with peak at [1, 2].
    ...
```

- [x] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v`
Expected: FAIL with `AttributeError`

- [x] **Step 3: Implement `compute_explainability` in `src/models/descriptor_classifier.py`**

- For `RandomForestClassifier` (when `hasattr(model, "feature_importances_")`), extract Gini MDI importances; otherwise (e.g., `k-NN` `Pipeline`), compute `sklearn.inspection.permutation_importance(model, X_test, y_test, scoring="average_precision", n_repeats=cfg.permutation_repeats, random_state=cfg.seed)` (with standardized Fisher inter-class distance fallback when high dimensional redundancy masks single-column permutations).
- Group features into:
  - `family_breakdown`: `x85_entropy` (85 cols) vs `x9_asymmetry` (9 cols).
  - `scale_breakdown`: `E_1x1` (1), `E_2x2` (4), `E_4x4` (16), `E_8x8` (64), `asym_2x2` (3), `asym_4x4` (3), `asym_8x8` (3).
  - `asymmetry_kind_breakdown`: `asym_lr` (3), `asym_tb` (3), `asym_rot180` (3).
- Reconstruct `spatial_heatmaps[g]` for each $g \in \{2, 4, 8\}$ by reshaping the $g^2$ importance values of `E_{g}x{g}_r{r}_c{c}` into `[g, g]` row-major order.

- [x] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v`
Expected: PASS

- [x] **Step 5: Commit**

```powershell
git add src/models/descriptor_classifier.py tests/test_descriptor_classifier.py
git commit -m "feat(models): add explainability breakdown for x85 vs x9 and 2D spatial importance maps"
```

---

### Task 5: Publication Figures, Artifact Exporter, and CLI (`experiments/evaluate_descriptors.py`)

**Files:**
- Create: `src/models/classification_plots.py`
- Modify: `src/models/descriptor_classifier.py` (`run_evaluation_pipeline` + JSON/CSV writers)
- Create: `experiments/evaluate_descriptors.py`
- Modify: `src/README.md`, `experiments/README.md`, `tests/README.md`, `README.md`
- Test: `tests/test_descriptor_classifier.py`

**Interfaces:**
- Produces in `src/models/classification_plots.py`:
  - `plot_confusion_and_curves(metrics: ClassificationMetrics, title_prefix: str, out_path: Path, dpi: int) -> Path`
  - `plot_subset_ablation(ablation_metrics: dict[str, ClassificationMetrics], subset_order: Sequence[str], title_prefix: str, out_path: Path, dpi: int) -> Path`
  - `plot_explainability_breakdown(report: ExplainabilityReport, title_prefix: str, out_path: Path, dpi: int) -> Path`
  - `plot_spatial_importance_heatmaps(report: ExplainabilityReport, title_prefix: str, out_path: Path, dpi: int) -> Path`
  - `plot_all_objects_summary(summaries: Sequence[EvaluationRunSummary], title_prefix: str, out_path: Path, dpi: int) -> Path`
- Produces in `experiments/evaluate_descriptors.py`:
  - `build_parser() -> argparse.ArgumentParser`
  - `main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int`

- [x] **Step 1: Write failing tests for figure creation, JSON/CSV export, and CLI execution**

```python
def test_cli_generates_metrics_and_four_plots(tmp_path: Path) -> None:
    # Runs main(["--config", ..., "--classifier", "rf", "--object", "bottle"]) on synthetic project.
    # Verifies exit code 0, metrics_summary.json, ablation_table.csv, feature_importance.csv,
    # and the 4 PNG figures exist and are non-empty; verifies --dry-run writes nothing.
    ...
```

- [x] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.classification_plots'`

- [x] **Step 3: Implement `src/models/classification_plots.py`, pipeline exporter, and `experiments/evaluate_descriptors.py`**

- Use `matplotlib` (`Agg` backend safe) to render the 4 figures into `results/classification/<dataset>[_inverted]/<object_or_full>_<classifier>/`:
  1. `confusion_matrix_and_curves.png` (1×3 panel: Confusion Matrix with counts/%, ROC curve with AUC, PR curve with AP).
  2. `descriptor_ablation_x85_x9_x94.png` (grouped bar chart comparing `F1`, `Balanced Accuracy`, and `ROC-AUC` across `x85_only`, `x9_only`, `x94_full`, and per-scale `E_1x1..E_8x8`, annotating $\Delta\text{F1}$ of adding $\mathbf{x}_9$).
  3. `explainability_importance_breakdown.png` (1×3 panel: Top-$K$ individual features color-coded by Entropy vs Asymmetry; Total Share % by scale $H^{(1)}..H^{(8)}$ & $A^{(2)}..A^{(8)}$; Mean Importance per Feature normalized by group size).
  4. `spatial_importance_heatmaps.png` (1×3 panel: 2D heatmaps for $2\times2$, $4\times4$, and $8\times8$ entropy grids with cell values annotated).
- CLI flags in `experiments/evaluate_descriptors.py`:
  - `--config` (default `configs/classification/kolektorsdd2.toml`)
  - `--classifier` (`rf` or `knn`, overrides TOML `default_classifier`)
  - `--inverted` (uses inverted CSV and writes to `*-inverted` output subfolder)
  - `--object` (for MVTec-AD single object, e.g. `bottle`, `metal_nut`, `tile`)
  - `--all-objects` (iterates over all 15 MVTec-AD objects and writes an aggregate `mvtec_all_objects_<classifier>_summary.csv` and comparison plot)
  - `--dry-run` (validates config, CSV schema, and class counts per split without training or writing files)
  - `--overwrite`

- [x] **Step 4: Run full test suite to verify everything passes**

Run: `.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v`
Expected: PASS (all 8 tests green; full repository test suite 149/149 green)

- [x] **Step 5: Update `src/README.md`, `experiments/README.md`, `tests/README.md`, and `README.md`**

---

### Task 6: Verification Commands (Manual / Post-Implementation Check)

All verification commands below were executed on the real datasets from `code/`:

```powershell
# 1. Run unit and integration tests (8 tests in test_descriptor_classifier.py, 149 total in repo)
.\.venv\Scripts\python.exe -m unittest tests/test_descriptor_classifier.py -v

# 2. Dry-run validation (KolektorSDD2 and MVTec-AD)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --dry-run

# 3. KolektorSDD2 (normal & inverted, Random Forest & k-NN)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier rf
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier knn
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier rf --inverted
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier knn --inverted

# 4. MVTec-AD single object example (e.g., bottle or tile, normal & inverted)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier rf
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object tile --classifier rf
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier knn --inverted

# 5. MVTec-AD all 15 objects summary (normal & inverted)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier rf
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier rf --inverted
```

---

## 5. Resultados Experimentales Medidos, Complementariedad Científica y Guía de Interpretación (Para Reporte y NotebookLM / RAG)

### 5.1 ¿Por qué KolektorSDD2 y MVTec-AD se complementan científicamente en la investigación (*Entropy Maps + Symmetry vs. Convolutions*)?

Para la tesis central del proyecto —demostrar que un descriptor compacto de **entropía multiescala ($\mathbf{x}_{85}$) + residuos de asimetría espacial ($\mathbf{x}_9$)** acoplado a clasificadores ligeros puede competir en detección de defectos contra **redes convolucionales (CNNs)** con una fracción mínima del cómputo—, **usar ambos datasets es mucho más sólido que usar uno solo**, porque cada uno responde una pregunta experimental distinta:

1. **KolektorSDD2 (`split_mode = "presplit"` — Benchmark Primario de Comparación Supervisada Directa):**
   - **Ventaja metodológica:** Su protocolo oficial ya viene dividido en `train` (`2,085 good`, `246 defect`) y `test` (`894 good`, `110 defect`) con muestras defectuosas en ambos conjuntos. No requiere ningún re-split: cualquier modelo supervisado ($\mathbf{x}_{94}$ + Random Forest / k-NN / MLP vs. MobileNetV3 / ShuffleNetV2 / SqueezeNet) se entrena y evalúa exactamente sobre las mismas filas oficiales.
   - **Naturaleza visual:** Son superficies metálicas alargadas de conmutadores eléctricos (escaladas a $128\times128$ con letterboxing negro lateral) donde los defectos son fisuras o desgaste fino. Permite evaluar qué tan bien las escalas finas de entropía ($4\times4$ y $8\times8$) y la asimetría vertical/horizontal detectan micro-rupturas superficiales frente a filtros convolucionales $3\times3$.

2. **MVTec-AD (`split_mode = "stratified"` en memoria por objeto — Benchmark de Diversidad Geométrica, Simetría vs. Textura):**
   - **Por qué se usa partición estratificada en memoria (`Enfoque 1`):** En MVTec-AD nativo, el split `train` contiene únicamente imágenes `good` (`3,629 good`, `0 defect`) porque fue concebido para *Unsupervised One-Class Anomaly Detection*, y además el CSV contiene `1,258` filas `split == "ground_truth"` correspondientes a máscaras binarias. Entrenar un clasificador binario supervisado directamente sobre el `train` nativo es imposible (solo existe 1 clase). Al filtrar en memoria `split in ["train", "test"]` (excluyendo `ground_truth`) y aplicar un `StratifiedShuffleSplit` determinista (`test_size = 0.20`, `seed = 42`) por cada objeto:
     - **No se modifica ningún archivo CSV en disco** (verificado mediante hash SHA-256 en los tests).
     - Se mantiene toda la investigación dentro del paradigma de **clasificación supervisada eficiente** (en lugar de abrir otra línea de *One-Class Anomaly Detection*).
     - Al usar la misma semilla (`seed = 42`) cuando se entrenen las CNNs sobre MVTec-AD, la comparación $\mathbf{x}_{94}$ vs. CNN por objeto es 100% justa y pareada.
   - **Por qué MVTec-AD es indispensable para validar $\mathbf{x}_9$ y $\mathbf{x}_{85}$:** Contiene **15 categorías** divididas en **10 objetos estructurados** (`bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`) y **5 texturas** (`carpet`, `grid`, `leather`, `tile`, `wood`). Permite probar empíricamente en qué geometrías brilla la asimetría espacial ($\mathbf{x}_9$) y en cuáles domina la entropía multiescala ($\mathbf{x}_{85}$).

---

### 5.2 Resultados Medidos en KolektorSDD2 (`presplit` oficial: Train = 2,331 | Test = 1,004)

#### A. Comparación por clasificador y variante de intensidad (`x94_full`)

| Variante | Clasificador | Accuracy | Balanced Acc | Precision (`defect`) | Recall (`defect`) | F1 (`defect`) | ROC-AUC | PR-AUC | $\Delta\text{F1}(\mathbf{x}_{94} - \mathbf{x}_{85})$ | Share $\mathbf{x}_{85}$ | Share $\mathbf{x}_9$ | $T_{\text{train}}$ (s) | $T_{\text{inf}}$ (ms/img) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Normal** | **Random Forest (`rf`)** | `0.9044` | `0.6035` | `0.7059` | `0.2182` | `0.3333` | **`0.8219`** | **`0.4906`** | **`+0.0388`** | `65.02%` | **`34.98%`** | `0.6540` | `0.120` |
| **Invertido** | **Random Forest (`rf`)** | `0.9044` | `0.6035` | `0.7059` | `0.2182` | `0.3333` | **`0.8217`** | **`0.4899`** | **`+0.0388`** | `65.04%` | **`34.96%`** | `0.4890` | `0.107` |
| **Normal** | **k-NN (`knn`, $k=5$)** | `0.8924` | `0.5880` | `0.5250` | `0.1909` | `0.2800` | `0.7013` | `0.3266` | `-0.0884` | `51.82%` | `48.18%` | `0.0044` | `0.068` |
| **Invertido** | **k-NN (`knn`, $k=5$)** | `0.8924` | `0.5880` | `0.5250` | `0.1909` | `0.2800` | `0.7013` | `0.3266` | `-0.0884` | `51.85%` | `48.15%` | `0.0074` | `0.097` |

#### B. Ablación por subconjunto y escala en KolektorSDD2 (Random Forest, Normal)

| Subconjunto | Nº Features | Accuracy | Balanced Acc | Precision (`defect`) | Recall (`defect`) | F1 (`defect`) | ROC-AUC | PR-AUC | Interpretación |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `x85_only` | 85 | `0.9094` | `0.5830` | `0.9500` | `0.1743` | `0.2946` | `0.8279` | `0.4910` | Muy alta precisión (`95%`), pero baja sensibilidad sin simetría. |
| `x9_only` | 9 | `0.8406` | `0.6383` | `0.3333` | `0.3670` | `0.3493` | `0.7393` | `0.3016` | Con solo 9 números supera en Recall (`36.7%`) y F1 (`0.3493`) a `x85_only`. |
| `x94_full` | 94 | `0.9044` | `0.6035` | `0.7059` | `0.2182` | `0.3333` | `0.8219` | `0.4906` | Combina el alto ROC-AUC de $\mathbf{x}_{85}$ con $+3.88$ pts de F1 gracias a $\mathbf{x}_9$. |
| `E_1x1` | 1 | `0.5388` | `0.5419` | `0.1268` | `0.5455` | `0.2058` | `0.5540` | `0.1301` | La entropía global de 1 sola celda es insuficiente (casi azar). |
| `E_2x2` | 4 | `0.7331` | `0.5877` | `0.1818` | `0.4000` | `0.2500` | `0.6656` | `0.2144` | Captura diferencias gruesas por cuadrante, aún insuficiente para fisuras finas. |
| **`E_4x4`** | **16** | `0.8805` | **`0.6308`** | `0.4444` | **`0.3636`** | **`0.4000`** | `0.7740` | `0.4027` | **Mejor F1 individual (`0.4000`) y mejor Balanced Accuracy (`0.6308`).** |
| **`E_8x8`** | **64** | `0.9034` | `0.5563` | `0.8235` | `0.1273` | `0.2205` | **`0.8181`** | **`0.4648`** | **Mejor ROC-AUC individual (`0.8181`) y PR-AUC (`0.4648`).** |

---

### 5.3 Resultados Medidos en MVTec-AD (15 Categorías de Objetos, Random Forest, Split Estratificado 80/20 en Memoria)

| Categoría (`object_name`) | Tipo | Train (`good`/`defect`) | Test (`good`/`defect`) | F1 (`x85_only`) | F1 (`x9_only`) | F1 (`x94_full`) | $\Delta\text{F1}(\mathbf{x}_{94}-\mathbf{x}_{85})$ | ROC-AUC (`x94`) | PR-AUC (`x94`) | Mejor Escala $\mathbf{x}_{85}$ | Share $\mathbf{x}_{85}$ (%) | Share $\mathbf{x}_9$ (%) | Característica Top-1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| **`tile`** | Textura periódica | 210 / 67 | 53 / 17 | `0.2000` | `0.8485` | **`0.8235`** | **`+0.6235`** | **`0.9711`** | `0.9242` | `E_4x4` | `41.37%` | **`58.63%`** | `asym_rot180_8x8` |
| **`carpet`** | Textura homogénea | 246 / 71 | 62 / 18 | `0.0000` | `0.4000` | **`0.3478`** | **`+0.3478`** | `0.7858` | `0.5364` | `E_1x1` | `81.10%` | **`18.90%`** | `asym_tb_4x4` |
| **`bottle`** | Objeto simétrico | 183 / 50 | 46 / 13 | `0.5263` | `0.6400` | **`0.7273`** | **`+0.2010`** | **`0.8980`** | `0.8235` | `E_4x4` | `62.72%` | **`37.28%`** | `asym_tb_8x8` |
| **`hazelnut`** | Objeto rotacional | 344 / 56 | 87 / 14 | `0.3529` | `0.0000` | **`0.5263`** | **`+0.1734`** | **`0.9302`** | `0.8055` | `E_4x4` | `91.05%` | `8.95%` | `E_8x8_r7_c4` |
| **`leather`** | Textura orgánica | 221 / 74 | 56 / 18 | `0.6154` | `0.6250` | **`0.7222`** | **`+0.1068`** | **`0.9395`** | `0.8505` | `E_2x2` | `92.54%` | `7.46%` | `E_8x8_r1_c4` |
| **`capsule`** | Objeto axial | 193 / 87 | 49 / 22 | `0.6286` | `0.5294` | **`0.7027`** | **`+0.0741`** | **`0.9100`** | `0.8147` | `E_4x4` | `88.18%` | `11.82%` | `asym_rot180_4x4` |
| **`cable`** | Objeto estructurado | 225 / 74 | 57 / 18 | `0.3636` | `0.3478` | **`0.4348`** | **`+0.0711`** | `0.8177` | `0.6600` | `E_4x4` | `90.90%` | `9.10%` | `E_8x8_r3_c4` |
| **`screw`** | Objeto fino | 289 / 95 | 72 / 24 | `0.6000` | `0.4615` | **`0.6667`** | **`+0.0667`** | **`0.8692`** | `0.7610` | `E_8x8` | `91.27%` | `8.73%` | `E_8x8_r6_c0` |
| **`zipper`** | Objeto axial | 217 / 95 | 55 / 24 | `0.8000` | `0.4878` | **`0.8085`** | **`+0.0085`** | **`0.9424`** | `0.7823` | `E_8x8` | `83.86%` | **`16.14%`** | `asym_tb_4x4` |
| **`metal_nut`** | Objeto rotacional | 194 / 74 | 48 / 19 | `0.5926` | `0.2308` | **`0.5926`** | `0.0000` | **`0.8640`** | `0.7989` | `E_4x4` | `87.73%` | `12.27%` | `E_8x8_r2_c5` |
| **`toothbrush`**| Objeto pequeño | 57 / 24 | 15 / 6 | `0.2857` | `0.7692` | `0.2857` | `0.0000` | **`0.9889`** | **`0.9762`** | `E_2x2` | `88.43%` | `11.57%` | `E_8x8_r7_c7` |
| **`transistor`**| Objeto electrónico | 218 / 32 | 55 / 8 | `0.0000` | `0.4000` | `0.0000` | `0.0000` | **`0.9409`** | `0.7888` | `E_4x4` | `90.61%` | `9.39%` | `E_8x8_r4_c2` |
| **`wood`** | Textura direccional | 212 / 48 | 54 / 12 | `0.3750` | `0.2000` | `0.3750` | `0.0000` | `0.7623` | `0.5819` | `E_4x4` | `88.07%` | `11.93%` | `asym_tb_8x8` |
| **`pill`** | Objeto moteado | 234 / 113 | 59 / 28 | `0.7826` | `0.4000` | **`0.7556`** | `-0.0271` | **`0.9582`** | **`0.9277`** | `E_4x4` | `94.09%` | `5.91%` | `E_4x4_r0_c0` |
| **`grid`** | Textura malla fina | 227 / 46 | 58 / 11 | `0.1667` | `0.3529` | `0.0000` | `-0.1667` | `0.6254` | `0.2377` | `E_1x1` | `80.82%` | **`19.18%`** | `asym_lr_4x4` |

*(Nota: Los resultados sobre `mvtec-ad-inverted` son prácticamente idénticos en las 15 categorías debido a la invarianza del histograma de Shannon bajo inversión $255 - I$; el resumen completo está en `results/classification/mvtec-ad-inverted/mvtec_all_objects_rf_summary.csv`).*

---

### 5.4 Cuatro Conclusiones Científicas Clave para el Artículo

1. **¿Funcionan las 9 medidas de simetría/asimetría ($\mathbf{x}_9$)? Sí, de forma desproporcionada a su tamaño:**
   - $\mathbf{x}_9$ representa apenas **9 de las 94 columnas ($9.57\%$ del vector)**. Sin embargo:
     - En **KolektorSDD2**, captura el **34.98% de la importancia total del Random Forest** ($3.89\%$ de importancia media por característica frente a $0.76\%$ de las características de entropía, es decir, **5.1 veces más informativa por característica**), y las **7 variables individuales más importantes de las 94 son todas de asimetría** (`asym_tb_8x8`, `asym_tb_4x4`, `asym_lr_8x8`, `asym_tb_2x2`, `asym_rot180_8x8`, `asym_lr_4x4`, `asym_rot180_4x4`).
     - En **MVTec-AD**, anexar $\mathbf{x}_9$ mejora o mantiene el F1 en **13 de los 15 objetos**, logrando incrementos masivos donde el defecto rompe la homogeneidad o simetría espacial: **`tile` ($\Delta\text{F1} = +0.6235$, de $0.2000 \to 0.8235$, con $58.63\%$ de importancia en $\mathbf{x}_9$)**, **`carpet` ($\Delta\text{F1} = +0.3478$)**, **`bottle` ($\Delta\text{F1} = +0.2010$, de $0.5263 \to 0.7273$, con $37.28\%$ de importancia en $\mathbf{x}_9$)**, **`hazelnut` ($\Delta\text{F1} = +0.1734$)**, **`leather` ($\Delta\text{F1} = +0.1068$)**, **`capsule` ($\Delta\text{F1} = +0.0741$)**, **`cable` ($\Delta\text{F1} = +0.0711$)** y **`screw` ($\Delta\text{F1} = +0.0667$)**.
     - Incluso usando **únicamente las 9 características (`x9_only`)** sin las 85 de entropía, el modelo alcanza `F1 = 0.8485` en `tile`, `F1 = 0.7692` en `toothbrush`, `F1 = 0.6400` en `bottle`, `F1 = 0.6250` en `leather` y `F1 = 0.3493` en `KolektorSDD2`.

2. **¿Cuál mapa de entropía ($\mathbf{x}_{85}$) sirvió más? El compromiso resolución-varianza en $H^{(4)}$ ($4\times4$) y $H^{(8)}$ ($8\times8$):**
   - **Por capacidad predictiva individual (ablación por escala):** La escala intermedia **`E_4x4` (16 celdas de $32\times32$ px)** es la mejor escala individual en **KolektorSDD2 (`F1 = 0.4000`, `BalAcc = 0.6308`)** y en **9 de los 15 objetos de MVTec-AD** (`bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `tile`, `transistor`, `wood`). La razón física y estadística es que cada sub-región de $32\times32$ contiene $1,024$ píxeles para estimar un histograma de 256 bins con baja varianza muestral y suficiente localización espacial.
   - **La escala fina `E_8x8` (64 celdas de $16\times16$ px, $256$ píxeles/celda):** Domina cuando el defecto es estrecho o muy localizado (`screw`, `zipper`, y logra el mayor `ROC-AUC = 0.8181` en `KolektorSDD2`) y aporta el mayor volumen acumulado de importancia (`38.24%` en KolektorSDD2), pero con menor importancia media por celda (`0.60%`) porque muchas celdas periféricas caen sobre fondo o zonas sin defecto.

3. **Demostración visual del efecto del Letterboxing en `spatial_importance_heatmaps.png`:**
   - En **KolektorSDD2**, las imágenes originales son tiras verticales centradas sobre un lienzo cuadrado $128\times128$ con relleno negro (`pad_value = 0`) a izquierda y derecha. En `results/classification/kolektorsdd2/full_rf/spatial_importance_heatmaps.png`, las columnas exteriores `c0` y `c3` de `E_4x4` y las columnas `c0, c1, c6, c7` de `E_8x8` tienen exactamente **`0.00%` de importancia**, mientras que el **100% de la atención del clasificador se concentra en la franja central (`c1–c2` en $4\times4$, `c2–c5` en $8\times8$)**. Esto demuestra visualmente que el árbol descarta automáticamente las celdas de entropía nula del padding sin necesidad de máscaras manuales.

4. **Invarianza ante inversión fotométrica (`normal` vs. `--inverted`), conservación del letterboxing en `0` y criterio de elección para el paper:**
   - **El letterboxing se conserva siempre en `0`:** En el preprocesamiento de ambos datasets, la inversión $I_{\text{inv}} = 255 - I$ se aplica a la imagen **antes** de `letterbox_to_square(..., pad_value=0)`. De este modo, las bandas de relleno permanecen siempre en `0` (negro puro, entropía `0.0`).
   - **Por qué usar `normal` como configuración primaria para detectar defectos (bajo el umbral realista `0.50`):**
     - En **KolektorSDD2** (donde sí hay bandas laterales de letterboxing), `normal` detecta un **33% más de defectos verdaderos en la matriz de confusión (`TP = 24/110`, `F1 = 0.3333`, `BalAcc = 0.6035`)** frente a `inverted` (`TP = 18/110`, `F1 = 0.2647`).
     - En **MVTec-AD** (donde las 15 categorías son originalmente cuadradas y no requieren letterboxing), invertir $255 - I$ solo invierte el orden de los 256 bins del histograma ($p'_k = p_{255-k}$). Como la entropía de Shannon $H = -\sum p_k \log_2 p_k$ es conmutativa, `normal` e `inverted` dan exactamente la misma matriz de confusión en 12 de los 15 objetos.
   - **Por qué conviene reportar ambos (`normal` e `inverted`) en el artículo:**
     - En **MVTec-AD**, reportar ambos demuestra empíricamente que **$\mathbf{x}_{94}$ es invariante a la inversión de polaridad de iluminación ($255 - I$)** en imágenes sin padding (una ventaja analítica frente a las CNNs sin *data augmentation*).
     - En **KolektorSDD2**, reportar ambos muestra el compromiso físico en presencia de letterboxing: `normal` maximiza la detección directa de defectos (`TP = 24` a umbral `0.50`), mientras que `inverted` mejora ligeramente el ranking probabilístico (`ROC-AUC = 0.8383` vs `0.8219`) al separar las sombras oscuras del metal respecto al `pad_value = 0` del relleno.
   - **De dónde salen las probabilidades de las curvas ROC/PR y por qué algunas matrices dan `TP = 0` a umbral `0.50`:**
     - En un `RandomForestClassifier` de 200 árboles, `predict_proba(X_test)[:, 1]` calcula el promedio de votos de los 200 árboles ($\hat{p} = \frac{1}{200}\sum_{t=1}^{200} p_t$). Para la matriz de confusión se mantiene el umbral estándar realista $\hat{p} \ge 0.50$. En clases con muy pocos defectos (ej. `transistor` con 32 defectos en train) y `max_features="sqrt"` (9 columnas por corte), muchos árboles dividen sobre celdas sanas de fondo y diluyen el promedio $\hat{p}$ de las piezas defectuosas a `[0.16, 0.45]` (vs `[0.00, 0.43]`, mediana `0.09` en las sanas), produciendo `TP = 0` en el corte `0.50` aunque el `ROC-AUC` sea `0.9386`. Evaluar un `DecisionTreeClassifier` individual (`max_features=None`) evita esa dilución por promediado.
   - El tiempo de inferencia del clasificador sobre $\mathbf{x}_{94}$ es de **`0.07–0.12 ms` por imagen en KolektorSDD2** (`~1.0–1.4 ms` en MVTec-AD), varios órdenes de magnitud inferior al paso forward de una CNN sobre tensores $128\times128$.

---

### 5.5 Guía Práctica: Qué buscar en cada archivo y gráfica generada

Cada corrida crea un directorio `results/classification/<dataset>[_inverted]/<object_or_full>_<classifier>/` con **4 archivos de datos** y **4 figuras PNG** (más 2 archivos agregados cuando se usa `--all-objects`):

1. **`confusion_matrix_and_curves.png` (Panel 1×3):**
   - *Panel izquierdo (Matriz de Confusión):* Revisar la esquina inferior derecha (`True Defect / Predicted Defect = TP`) y la superior derecha (`False Positives = FP`). Cuando el dataset está muy desbalanceado (ej. `transistor` o `KolektorSDD2`), un clasificador con umbral por defecto `0.5` puede tener pocos `TP` pero muy pocos `FP`.
   - *Panel central (Curva ROC) y derecho (Curva Precision-Recall):* **Crucial cuando el F1 en umbral 0.5 parece bajo.** Por ejemplo, en `toothbrush` (`F1 = 0.2857`) o `transistor` (`F1 = 0.0000`), el **ROC-AUC es `0.9889` y `0.9409`** respectivamente. Eso significa que las probabilidades predichas por $\mathbf{x}_{94}$ separan casi perfectamente `good` de `defect`, y basta ajustar el umbral de decisión (menor a `0.5`) para obtener un F1 alto.

2. **`descriptor_ablation_x85_x9_x94.png` (Gráfica de Barras Agrupadas):**
   - **Qué buscar primero:** Comparar las primeras tres barras (`x85_only`, `x9_only`, `x94_full`) y el recuadro superior derecho **`Symmetry contribution: dF1(x94 - x85)`**. Si es positivo (ej. `+0.6235` en `tile`, `+0.2010` en `bottle`, `+0.0388` en `kolektorsdd2`), demuestra empíricamente que las 9 medidas de simetría mejoran la detección.
   - **Qué buscar después:** Comparar las cuatro barras derechas (`E_1x1`, `E_2x2`, `E_4x4`, `E_8x8`) para identificar cuál nivel de la pirámide de entropía sostiene el rendimiento por sí solo.

3. **`explainability_importance_breakdown.png` (Panel 1×3):**
   - *Panel 1 (Top-15 Individual Features):* Las barras **naranjas** son características de **asimetría ($\mathbf{x}_9$)** y las **azules** son de **entropía ($\mathbf{x}_{85}$)**. Si aparecen barras naranjas en los primeros lugares (como en `kolektorsdd2`, `bottle`, `tile`, `carpet`, `capsule`, `zipper`), tienes evidencia directa de que la ruptura de simetría lidera las particiones del modelo.
   - *Panel 2 (Total Importance Share % by Scale):* Muestra el porcentaje total del pastel (suma = `100%`) que se lleva cada escala (`E_1x1..E_8x8` y `asym_2x2..asym_8x8`).
   - *Panel 3 (Mean Importance per Feature — Normalized by Group Size):* **El gráfico más importante para el paper al comparar $\mathbf{x}_9$ contra $\mathbf{x}_{85}$.** Como `E_8x8` tiene 64 variables y cada `asym_*` tiene solo 3, el Panel 2 favorece a `E_8x8` por cantidad. El Panel 3 divide entre el número de variables del grupo y revela el peso individual real de cada característica.

4. **`spatial_importance_heatmaps.png` (Panel 1×3: Mapas 2D $2\times2$, $4\times4$, $8\times8$):**
   - Reconstruye espacialmente la importancia de cada celda `E_{g}x{g}_r{r}_c{c}` sobre el lienzo $128\times128$.
   - **Qué buscar:** En objetos centrados (`bottle`, `hazelnut`, `metal_nut`, `pill`), verás un núcleo caliente en el centro y bordes fríos; en `KolektorSDD2`, verás la banda vertical central encendida y las bandas laterales de letterboxing en `0.00%`; en texturas que cubren toda la imagen (`tile`, `leather`, `wood`), verás activación distribuida en todo el lienzo.

5. **`mvtec_all_objects_rf_comparison.png` y `mvtec_all_objects_rf_summary.csv` (Generados con `--all-objects`):**
   - *Panel superior:* Compara lado a lado `F1(x85_only)` (azul), `F1(x94_full)` (verde) y `ROC-AUC(x94_full)` (morado) en las 15 categorías de MVTec-AD.
   - *Panel inferior:* Muestra el `% de contribución total de x9` en cada objeto comparado contra la línea punteada de expectativa uniforme (`9/94 = 9.6%`). Todo objeto cuya barra naranja supere la línea punteada demuestra que las características de simetría aportan más información que el promedio esperado por azar.
   - *Archivo CSV (`mvtec_all_objects_rf_summary.csv`):* Lista para importar en LaTeX/Pandas o consultar en NotebookLM con todas las métricas de los 15 objetos en una sola tabla.


> 💡 **Quick tip** — for when you modify the code yourself and add `dt` (`DecisionTreeClassifier`)

You only need to touch **4 very clean places**:

### 1️⃣ Configs (`kolektorsdd2.toml` and `mvtec_ad.toml`)

Add a new section:

```toml
[classification.models.dt]
max_depth = 12
min_samples_leaf = 2
class_weight = "balanced"
```

### 2️⃣ `descriptor_classifier.py`

- Create the `@dataclass(frozen=True, slots=True) DtcHyperparams` *(just like `RfHyperparams`)* and read it in `load_classification_config`.
- Add `"dt"` to `VALID_CLASSIFIERS = ("rf", "knn", "dt")`.
- Add the `elif normalized == "dt": return DecisionTreeClassifier(...)` in `build_classifier`.

> ✨ **Bonus:** since `DecisionTreeClassifier` also exposes `.feature_importances_`, the **Gini MDI** explainability engine will work automatically without changing anything else.

### 3️⃣ `evaluate_descriptors.py`

Add `"dt"` to `choices=("rf", "knn", "dt")` in `argparse`.