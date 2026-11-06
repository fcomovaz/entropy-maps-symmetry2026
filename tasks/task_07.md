# CNN Comparison

## Issue

* In the task_06A.md the main pipeline was implemented to train and evaluate a lightweight ML model. Now, this is our reference, but we need to secure our proposal is stronger than it. So we need to compare the performance of this with a high level architecture, in this case a CNN.
* The proposed architectures for CNN, you can check them, they are under the folder `./code/understanding/04_lightweight_cnns/models/`. Please take a look at them and review if they are the best option or you can improve them.
```powershell
Device: cpu; float32; MB = 1,000,000 bytes (parameters only)
Model                     Parameters   Trainable          Input shape   Output shape    Size MB
tiny_cnn                      23,538      23,538     (4, 1, 128, 128)         (4, 2)     0.0942
small_cnn                     97,650      97,650     (4, 1, 128, 128)         (4, 2)     0.3906
ds_cnn                        12,866      12,866     (4, 1, 128, 128)         (4, 2)     0.0515
mobilenet_like                70,786      70,786     (4, 1, 128, 128)         (4, 2)     0.2831
tiny_resnet                   77,234      77,234     (4, 1, 128, 128)         (4, 2)     0.3089
ultra_tiny                     1,154       1,154     (4, 1, 128, 128)         (4, 2)     0.0046
torchvision_mobilenet      1,519,618   1,519,618     (4, 1, 128, 128)         (4, 2)     6.0785
```
* Maybe they are caped in possibilities, so if that's the case and you think my laptop will struggle, please create a notebook and i can run it connecting directly to a colab env and make use of a gpu/tpu for this. The setup in my local machine is as follows:
```json
  "python": "3.11.9",
  "platform": "Windows-10-10.0.19045-SP0",
  "processor": "Intel64 Family 6 Model 158 Stepping 9, GenuineIntel",
  "torch": "2.14.0+cu126",
  "cuda_runtime": "12.6",
  "gpu": "NVIDIA GeForce GTX 1050",
  "cpu": "Intel(R) Core(TM) i7-10510U CPU @ 1.80GHz"
```
* Actually from the understanding 04 if everything is well done, then the models can be copypasted and not waist more time in improve, but please check it first.
* Also please consider the output from my ml models i was running was below,what i want you to notice is the metrics which will be the main comparision point, also the train and inference time, actually more important the inference time because is what matters at the end:
```powershell
PS D:\Publications\2026-10-Symmetry-Entropy\code> .\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier lr
2026-11-06T08:50:46.477Z | INFO    | evaluate_descriptors.py:142  | Completed KolektorSDD2 [LR]: Acc=0.7968 | BalAcc=0.7424 | F1=0.4205 | ROC-AUC=0.8073 | PR-AUC=0.4380 | dF1(x94-x85)=+0.1439 | Share(x85=72.4%, x9=27.6%) | Top-5=asym_lr_8x8 (12.8%), E_2x2_r1_c0 (12.2%), E_4x4_r1_c1 (10.8%), E_1x1_r0_c0 (10.0%), asym_tb_8x8 (9.4%) | T_train=0.1936s, T_inf=0.0016s (0.002 ms/sample) -> D:\Publications\2026-10-Symmetry-Entropy\code\results\classification\kolektorsdd2\full_lr
```


### Folder structure

The folder structure where the images are located is the following:

`D:\Publications\2026-10-Symmetry-Entropy\code\data\preprocessed\mvtec-ad` or
`D:\Publications\2026-10-Symmetry-Entropy\code\data\preprocessed\mvtec-ad-inverted` or
`D:\Publications\2026-10-Symmetry-Entropy\code\data\preprocessed\kolektorsdd2` or
`D:\Publications\2026-10-Symmetry-Entropy\code\data\preprocessed\kolektorsdd2-inverted`

the inverted and not inverted are important since in the ml models, the inverted image doesn't affect at all to the entropy metric.

> Remember not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.


## Plan

> Don't execute anything, just plan it

- [x] Read AGENTS.md file to know the general rules of the project.
- [x] Create a detailed plan for this task and update it here to implement it later for an agent, so it needs to be detailed.
- [x] Update this file with the plan, stages and everything to resolve and implement the task.

### 0. Findings from inspection (facts, not assumptions)

* Data: preprocessed images are **128×128 grayscale PNG**, so the CNNs from `understanding/04_lightweight_cnns` (input `[B,1,128,128]`) fit with no resize. CSVs (`data/csv/*.csv`) store pixels as `px_00000…` columns (16384 px) plus metadata. MVTec CSV columns: `object_name, split, type, defect_name, file_name`; `type` includes `ground_truth` rows, which must be excluded (as the baseline does).
* KolektorSDD2 presplit: train = 2085 good / 246 defect; test = 894 good / 110 defect (~11 % positives, so imbalance handling is required).
* MVTec-AD: the baseline uses an **in-memory stratified split per object** (`split_mode = "stratified"`, `test_size = 0.20`, `seed = 42`). The CNN MUST reuse the same split logic and seed so results are comparable.
* Baseline pipeline: `src/models/descriptor_classifier.py` (`load_classification_config`, `load_and_split_descriptors`, `train_and_evaluate`, `ClassificationMetrics`, `run_evaluation_pipeline`), CLI `experiments/evaluate_descriptors.py`, configs `configs/classification/*.toml`. Metrics: Acc, BalAcc, F1, ROC-AUC, PR-AUC, `T_train`, `T_inf`, ms/sample. Reuse this metric code; don't rewrite it.
* `understanding/04_lightweight_cnns/models/` has 7 models plus a registry (`create_model`). The sampled `ds_cnn` is correct (depthwise-separable blocks, BN/ReLU, `bias=False` before BN, global avg pool, linear head). Stage 1 confirms the rest before copying.
* Hardware (GTX 1050 4 GB, i7-10510U, torch 2.14+cu126) is enough for **all 7 models at 128×128** (~2.3k train images for KSDD2, a few hundred per MVTec object; the largest model is 1.5 M params). **Colab is optional**; a notebook is still provided (Stage 8) for the long sweeps (15 objects × 7 models × 2 variants × 3 seeds).

### 1. Model review (first, small)

1. Read each file in `understanding/04_lightweight_cnns/models/` and `common.py`. Check: output `[B,2]` logits; no hidden hardcoded sizes (global pooling → any `H,W`); `weights=None` in torchvision (no downloads); `num_classes` parametrised; BN/ReLU correct.
2. Rule: copy as-is unless a concrete defect is found. Record the verdict (OK / changed + reason) in `results/cnn/model_review.md`. No time spent on improvements.

### 2. Configuration (TOML only; nothing hardcoded in `src/`)

Create `configs/cnn/kolektorsdd2.toml` and `configs/cnn/mvtec_ad.toml` (same layout as `configs/classification/*.toml`; reuse `base_config = "configs/base.toml"` and `dataset_key`):

```toml
[cnn]
base_config = "configs/base.toml"
dataset_key = "kolektorsdd2"          # "mvtecad" in the other file
seed = 42
seeds = [42, 43, 44]                  # repeated runs -> mean ± std
inverted = false
overwrite = true
default_model = "ds_cnn"
models = ["ultra_tiny","ds_cnn","tiny_cnn","mobilenet_like","tiny_resnet","small_cnn","torchvision_mobilenet"]
default_object = "bottle"             # mvtec only

[cnn.data]                            # mirrors [classification.data]
split_column = "split"
label_column = "type"
object_column = ""                    # "object_name" for mvtec
positive_label = "defect"
negative_label = "good"
split_mode = "presplit"               # "stratified" for mvtec
test_size = 0.20
val_size = 0.15                       # carved from train only (stratified) for early stopping
image_size = 128
normalize_mean = 0.5
normalize_std = 0.5

[cnn.training]
epochs = 40
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-4
optimizer = "adamw"
scheduler = "cosine"
early_stopping_patience = 8
class_weighting = "balanced"
num_workers = 0                       # Windows-safe
device = "auto"                       # cuda if available else cpu
amp = false                           # float32 keeps timing comparable
deterministic = true

[cnn.augmentation]                    # label-preserving only; never inversion
hflip = true
vflip = false
rotate_deg = 0
brightness = 0.0

[cnn.benchmark]                       # inference timing protocol
warmup_iters = 10
timed_iters = 50
batch_sizes = [1, 32]
devices = ["cpu", "cuda"]
threads = 1                           # CPU latency, embedded-like
measure_flops = true

[cnn.output]
results_dir = "results/cnn"
dpi = 160
float_precision = 6
save_checkpoints = true
```

The values above are proposed defaults, to be tuned only on validation data. `src/` reads them through `@dataclass(frozen=True, slots=True)` classes with fail-fast validation of required keys (same `_require` pattern as the baseline).

### 3. Source code (`src/`, small modules, `from __future__ import annotations`, type hints, docstrings with tensor shapes)

* `src/models/cnn/__init__.py`
* `src/models/cnn/architectures/`: copy the 7 model files and `registry.py` **unchanged** (fix relative imports only). Keep the `understanding` originals (AGENTS.md: no automatic moves). Stage 6 test 1 proves the copy is faithful.
* `src/models/cnn/config.py`: `CnnConfig`, `TrainingConfig`, `BenchmarkConfig`, … and `load_cnn_config(path, project_root)`.
* `src/models/cnn/data.py`: `load_cnn_partition(cfg, object_name)` returns train/val/test tensors `[N,1,128,128]` plus labels `[N]`. Import or extract (minimal refactor, covered by existing tests) the split helper from `descriptor_classifier.py` so the **test set is identical to the baseline** (same `train_test_split(..., random_state=seed, stratify=y)`). **SUPERSEDED by section 12: pixels come from the preprocessed PNG files, NOT the CSV.** Memory is trivial (KSDD2 ≈ 55 MB). The validation split is taken from **train only**; test is never used for early stopping or thresholds.
* `src/models/cnn/train.py`: `train_model(model, data, cfg, seed) -> TrainResult` (history, best epoch, `T_train` with `torch.cuda.synchronize()`), weighted CrossEntropy, AdamW, cosine schedule, early stopping on val PR-AUC, full seeding (`torch.manual_seed`, cudnn deterministic).
* `src/models/cnn/evaluate.py`: `predict_proba(...)` and metrics through the **same `ClassificationMetrics`** as the baseline. Primary threshold 0.5 (comparable). Optionally add a clearly labelled extra column with the val-chosen threshold F1.
* `src/models/cnn/benchmark.py`: inference timing separate from metrics: warm-up, `perf_counter` plus `cuda.synchronize`, median/IQR over `timed_iters`, batch 1 and batch N, CPU single-thread and GPU. Reports ms/sample, throughput, params, size MB, MACs/FLOPs (`torchinfo` already in requirements or `torch.utils.flop_counter.FlopCounterMode`; no new dependency), peak VRAM.
* `src/models/cnn/report.py`: writes `metrics_summary.json`, `training_history.csv`, and matplotlib plots (ROC/PR curves, F1 vs ms/sample, accuracy-vs-latency Pareto including the baseline points).

### 4. Experiment CLIs (`experiments/`)

* `experiments/evaluate_cnn.py`: run as `python -m experiments.evaluate_cnn`; `build_parser()`, `main(argv: list[str] | None = None) -> int`, `if __name__ == "__main__": raise SystemExit(main())`; catch `OSError/ValueError/KeyError` → `SystemExit(f"error: {exc}")`; `logging` not `print`. Args: `--config`, `--model` (registry name or `all`), `--inverted`, `--object`, `--all-objects`, `--seeds`, `--device`, `--overwrite`, `--dry-run` (validates config, CSV, split counts and model shapes, no training). The final log line mirrors the baseline format (Acc | BalAcc | F1 | ROC-AUC | PR-AUC | T_train | T_inf ms/sample).
  Output: `results/cnn/<dataset>/<[object]>/<model>[_inverted]/` with `metrics_summary.json` (params, seed, hyperparameters, split sizes, device, torch/cuda versions, metrics, timings), `training_history.csv`, `checkpoint.pt` (gitignored), plots.
* `experiments/compare_baseline_vs_cnn.py`: reads `results/classification/**/metrics_summary.json` and `results/cnn/**/metrics_summary.json`; writes `results/comparison/<dataset>_baseline_vs_cnn.csv` and a figure. Columns: `model, family(ml|cnn), params, MACs, Acc, BalAcc, F1, ROC-AUC, PR-AUC, T_train, T_inf_ms_per_sample, T_descriptor, T_total`.
  **Fairness (AGENTS.md performance rule):** the baseline `T_inf = 0.002 ms/sample` covers only the classifier on precomputed x94 features. The comparison script must also measure x94 extraction time per image (existing `src/features/entropy_maps.py` and `asymmetry_maps.py`) so that `T_total = T_descriptor + T_model` is compared against the CNN, which takes raw pixels. Report both numbers; never claim the ML wins on classifier time alone.

### 5. Evaluation protocol (only MEASURED numbers go into reports)

* Datasets × variants: KolektorSDD2 {normal, inverted}; MVTec-AD {normal, inverted}, starting with `bottle`, `metal_nut`, `screw`, `tile`, then `--all-objects`.
* Models: the 7 CNNs; seeds {42,43,44}; mean ± std.
* Hypothesis to test, not assume: x94 + classical ML is invariant to inversion (entropy is symmetric under 255−x) while CNNs may degrade. Report Δ = metric(normal) − metric(inverted) per model, whatever the outcome.
* Metric priority: F1, PR-AUC, ROC-AUC, BalAcc (imbalanced; accuracy secondary); then **inference time** (most important), train time, params, MACs, VRAM.

### 6. Tests (`tests/test_cnn_*.py`, pytest)

1. Every registry architecture: output `[B,2]`, finite, parameter counts equal to the table in this issue (23,538 / 97,650 / 12,866 / 70,786 / 77,234 / 1,154 / 1,519,618).
2. Config loader: missing key → error; paths resolved from project root; no absolute paths.
3. Data: on a tiny synthetic CSV fixture, the test partition equals the baseline split with the same seed; no train/val/test overlap; stratification preserved; inverted path selection.
4. Training smoke: 2 epochs on 32 synthetic images with `ultra_tiny` lowers the loss and is deterministic for the same seed.
5. Metrics equal the baseline function on a fixed prediction vector.
6. CLI: `main([... "--dry-run"])` returns 0; unknown model → `SystemExit`.
7. Full suite: `.\.venv\Scripts\python.exe -m pytest tests -q` (existing tests must still pass).

### 7. Execution order for the implementing agent

1. Stage 1 review, write the verdict.
2. Stage 2 TOMLs, then `--dry-run` for both datasets.
3. Stage 3 modules and Stage 6 tests until green.
4. Stage 4 CLI: smoke (`ultra_tiny`, 2 epochs, KSDD2), then real runs: KSDD2 all models (normal + `--inverted`), then MVTec `bottle`/`metal_nut`/`screw`/`tile`.
5. Comparison script → table/figures; short measured-only analysis in `results/comparison/README.md` (measured vs expected kept separate).
6. Update `requirements.txt` only if a package is added (none planned); update docs: `experiments/README.md`, `src/README.md`, `tests/README.md`, and a pointer in `understanding/04_lightweight_cnns/README.md`.
7. Definition of done: implementation → execution → tests → expected output; state explicitly anything not verified (e.g., MVTec all-objects if not run).

### 8. Colab notebook (optional, heavy sweeps)

Create `notebooks/cnn_colab.ipynb`: (1) mount Drive / clone or upload repo, (2) install requirements without the cu126 pins (Colab ships torch), (3) use `configs/cnn/colab.toml` (the only place for Colab paths), (4) call `python -m experiments.evaluate_cnn ...` so the **same code** produces results, (5) zip `results/cnn` for download. Colab timings are NOT comparable with the local i7/GTX 1050; paper latency numbers must be re-measured locally, and the notebook logs the device name next to each timing.

### 9. Risks / open points

* Overfitting on small MVTec objects (~200 train images): augmentation, early stopping, 3 seeds, report variance.
* Fair timing: same batch size and threads on CPU; GPU timing includes `synchronize`; data loading excluded from `T_inf` but reported separately.
* The baseline timing excludes descriptor extraction; fix in the comparison script, not by editing baseline results.
* KSDD2 imbalance (~11 % positives): class-weighted loss; PR-AUC as the main threshold-free metric.

### 10. Addendum: MVTec-AD is NOT used as anomaly detection (from task_06A)

Facts from `task_06A.md`: native MVTec `train` = 3,629 good / **0 defect**; `test` = 467 good / 1,258 defect; `split == "ground_truth"` = 1,258 mask rows. The baseline therefore (a) drops `ground_truth` via `allowed_splits = ["train","test"]`, (b) selects one `object_name`, (c) **pools native train+test** and does a supervised stratified split (`test_size = 0.20`, `seed`), (d) fails fast if `presplit` is requested with a single-class train, and (e) fails fast on unknown `--object` (lists valid `subdatasets` from `configs/base.toml`). The CNN pipeline MUST do exactly the same:

* Do NOT train only on good images, and do NOT use the native train/test split or masks for MVTec. The supervised scaffold is the only protocol.
* Implement it by **calling the baseline's filtering/splitting function** (`load_and_split_descriptors` helper, refactored minimally) rather than copying logic, so train/test indices are identical for RF/LR and CNN under the same seed. Test: assert equal test file lists.
* The CNN validation subset (`val_size`) is taken from the pooled-train part only, after the baseline split.
* Same guards: `ground_truth` excluded, unknown object → `ValueError`, single-class partitions → `ValueError`.
* Per-object sets are small (about 100–400 images): keep multiple seeds, report variance, and note that results are NOT comparable to published anomaly-detection MVTec numbers (state this in the paper/report).
* Correction to section 0: the `ground_truth` marker is in the `split` column, not `type`.

### 11. Model review verdict (static code read of all 7 files; not yet executed)

All 7 are correct and need no reimplementation: valid shapes, `bias=False` before BN, global average pooling (any H×W), `num_classes` parametrised, no pretrained weights. Hand-checked `tiny_cnn` parameters = 23,538 (matches your table). Stage 6 test 1 will confirm all counts by execution.
For citing in the paper, describe them accurately:
* `tiny_cnn`, `small_cnn`: standard Conv-BN-ReLU-pool CNNs (own design, not a published architecture).
* `ds_cnn`, `ultra_tiny`: own designs built on depthwise-separable convolutions (cite the technique: MobileNetV1 / Howard et al. 2017, not an exact model).
* `mobilenet_like`: own small design; its docstring states that it is NOT MobileNetV2/V3 (no inverted residuals/expansion).
* `tiny_resnet`: ResNet-style basic residual blocks (He et al. 2016), reduced; not ResNet-18.
* `torchvision_mobilenet`: official torchvision MobileNetV3-Small, random init, 1-channel stem, 2 classes; the only one that can be cited as a standard architecture (still not ImageNet-pretrained).
Publish the exact code and parameter counts in the repo so they are reproducible. Verify the references with the real papers before citing (none were checked here).

### 12. Addendum (overrides sections 0, 3 and 6 where they mention the CSV): images come from the preprocessed PNGs

The CNN experiment reads the **PNG files** under `data/preprocessed/` (paths from `configs/base.toml`: `output_path` / `output_path_inverted` of `[datasets.mvtecad]` and `[datasets.kolektorsdd2]`). It does NOT read the pixel CSV (`px_*`).

* **Directory layouts (verified on disk):** KolektorSDD2: `<root>/{train,test}/{good,defect}/*.png`; MVTec-AD: `<root>/<object>/{train,test,ground_truth}/<defect_name>/*.png` (`train/good`, `test/good`, `test/<defect_name>`). Label: `good` → 0, any other defect folder under `train`/`test` → 1 (`defect`). `ground_truth` folders are never listed.
* **Split parity with the baseline without the CSV:** build the sample list (relative path, label, native split) by scanning the folders, sorted deterministically. For MVTec, pool `train`+`test` of the chosen object and apply the same `train_test_split(test_size, stratify=y, random_state=seed)`. Because the image order of the CSV may differ from a folder scan, parity is NOT automatic. Required: order the scanned list the same way as the descriptor CSV (read only the metadata columns `object_name, split, type, defect_name, file_name` of `mvtec-ad-entropy-full.csv`, no pixels, and use them as the ordered index to locate each PNG), then call the baseline's split helper on that index. Test: the resulting test `file_name` lists are identical for baseline and CNN. For KolektorSDD2 the native `train`/`test` folders are used directly (`presplit`).
* **Loader (`src/models/cnn/data.py`):** `PIL` → grayscale `uint8` → tensor `[N,1,128,128]` in `[0,1]` → normalised with TOML mean/std; the whole set is cached in memory once per run. Fail fast if an image is not 128×128 (`image_size` in TOML) or a PNG is missing.
* **Inverted:** same code with `output_path_inverted`; it is a separate dataset, not an on-the-fly transform (augmentation must never invert).
* **TOML:** remove any CSV reliance from `[cnn.data]`; add `use_metadata_index = true`, `metadata_csv_key = "full_csv_path"` (and the inverted key) for the ordered index only. Paths resolved from the project root via `pathlib`.
* **Tests:** replace the CSV fixture of section 6 test 3 with a tiny fixture of PNG folders plus a metadata-only CSV; assert split parity, `ground_truth` exclusion, label mapping and a 128×128 check.
* **Colab:** PNG folders must be uploaded or mounted (≈ tens of thousands of small files for MVTec); a zipped copy of `data/preprocessed` is recommended.
