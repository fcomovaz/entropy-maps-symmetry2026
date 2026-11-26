# INSTRUCTIONS.md — Reproducing the pipeline from raw datasets

Minimal formal runs to go from **raw MVTec-AD + KolektorSDD2** to every result reported across the entire research (or run end-to-end sequentially via [reproduce_paper_pipeline.ps1](reproduce_paper_pipeline.ps1), which generates the consolidated report [results/FINAL_PAPER_REPORT.md](results/FINAL_PAPER_REPORT.md)). No dry-runs, no unit tests, no lab detours. Each stage explains **what it does**, **the governing equations**, **the exact PowerShell commands**, and **what artifacts to expect**.

> [!IMPORTANT]
> - All commands run from `code/` in PowerShell with `.\.venv\Scripts\python.exe`.
> - Reference numbers were measured on: Windows 10, Python 3.11.9, PyTorch 2.14.0+cu126, GTX 1050, Intel i7-7700HQ, `seed = 42`. Classical-ML metrics should match closely; **CNN metrics and all latencies are hardware/driver dependent** (compare shapes, rankings, and orders of magnitude, not exact digits).
> - Numbers below come from the repository's own documents ([results/FINAL_PAPER_REPORT.md](results/FINAL_PAPER_REPORT.md), [HYPOTHESIS.md](HYPOTHESIS.md), [experiments/README.md](experiments/README.md), [RESUME.md](RESUME.md), [results/c_imp/README.md](results/c_imp/README.md), [results/hw_test/README.md](results/hw_test/README.md)).
> - `DecisionTreeClassifier` is built without `random_state`, so `dt` metrics can vary slightly between runs (e.g. `bottle_dt` 0.6364 vs 0.5714).

---

## 0. Inputs and layout

Place the raw datasets (not provided by this repo) at:

```text
code/data/raw/mvtec-ad/<category>/{train/good, test/<defect>, ground_truth/<defect>}   # 15 categories
code/data/raw/KolektorSDD2/{train,test}/N.png + N_GT.png                              # numeric stems; "(copy)" files are ignored
```

Paths, category list and output locations live in [configs/base.toml](configs/base.toml). Everything is derived dynamically; do not hardcode paths.

---

## 1. Environment and native C99 backend (Tasks 09 & 10)

**What it does.** Creates the pinned venv (PyTorch cu126 first, then `requirements.txt`) and compiles the single-pass C99 entropy kernel (`src/features/native/entropy_pyramid.c`, MSVC x64).

**Why it matters for reproducibility.** Entropy backend `"auto"` (default in every TOML) resolves to **`c_single_pass` if the DLL exists, otherwise `numpy_finest_pyramid`**. Both are numerically equivalent (parity $\le 1.89 \times 10^{-15}$), but latency claims require the DLL. `numpy_fused_entropy` is never chosen by `auto`.

```powershell
# from the project root D:\Publications\2026-10-Symmetry-Entropy
py -3.11 -m venv code\.venv
.\code\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\code\.venv\Scripts\python.exe -m pip install --no-cache-dir --default-timeout=1000 torch==2.14.0+cu126 torchvision==0.29.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.\code\.venv\Scripts\python.exe -m pip install -r code\requirements.txt

# from code/
Set-Location code
.\.venv\Scripts\python.exe -m src.features.build_native --config configs/features/native_entropy.toml --clean
```

**Expect:** `src/features/native/build/entropy_pyramid.dll` exists. (Requires MSVC build tools and an NVIDIA GPU; torch wheel ≈ 2.6 GB, takes several minutes. Use cu12.x, never cu13 on Pascal).

---

## 2. Preprocessing: raw images → 128×128 grayscale (Tasks 01 & 03)

**What it does.**
1. Convert to grayscale (ITU-R 601 luma): $Y = 0.299R + 0.587G + 0.114B$.
2. *Inverted variant only:* $I' = 255 - I$, applied **before** padding so padding stays 0.
3. Aspect-preserving resize so the longest side is 128 (images: `LANCZOS`; MVTec masks: `NEAREST`, keeps values in $\{0, 255\}$).
4. Center letterbox onto a 128×128 canvas with `pad_value = 0`.

Order is fixed: **gray → invert → resize → letterbox.** KolektorSDD2 labels come from the native-resolution mask: $\text{defect} \iff \max(\text{mask}) > 0$; masks are discarded from output.

```powershell
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --inverted
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2 --inverted
```

**Expect:** `data/preprocessed/{mvtec-ad, mvtec-ad-inverted, KolektorSDD2, KolektorSDD2-inverted}/`.
KolektorSDD2: **3335 pairs** → train 2085 good / 246 defect, test 894 good / 110 defect, 2 orphan/copy files skipped, no mask files in output. MVTec: every image is 128×128 mode `L`, masks only contain $\{0, 255\}$.

Optional integrity checks: `understanding/08_verify_mvtec_preprocessing/verify.py`, `understanding/10_verify_kolektorsdd2/verify.py` (+ `--inverted`).

---

## 3. Flat pixel CSVs (Tasks 02 & 04)

**What it does.** One image per row, pixels row-major: `px_{row*128+col}` (`px_00000 … px_16383`), plus metadata and `width, height`.

```powershell
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --inverted
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv
.\.venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --inverted
```

| File (`data/csv/`) | Columns | Rows | Metadata columns |
|---|---|---|---|
| `mvtec-ad[-inverted].csv` | 16391 | 6612 (includes `ground_truth` mask rows) | `object_name, split, type, defect_name, file_name` |
| `kolektorsdd2[-inverted].csv` | 16389 | 3335 | `split, type, file_name` |

Round-trip checks (should report diff = 0): `understanding/09_verify_mvtec_csv/verify.py`, `understanding/11_verify_kolektorsdd2_csv/verify.py` (+ `--inverted`).

---

## 4. Multiscale entropy descriptor $\mathbf{x}_{85}$ (Task 05)

**What it does.** For each grid level $g \in \{1, 2, 4, 8\}$ split the image into $g \times g$ cells ($1 + 4 + 16 + 64 = \mathbf{85\text{ cells}}$). Per cell with $N$ pixels:

- Quantize: $b = \lfloor I \cdot B / 256 \rfloor$, with $B = 16$ bins.
- Probabilities: $p_b = n_b / N$.
- Normalized Shannon entropy:
$$E = \frac{-\sum_{b} p_b \log_2 p_b}{\log_2 B} \in [0, 1]$$

Implementation: one fine $8\times8$ histogram, then coarser levels by bottom-up summation; entropy evaluated through precomputed $c\log_2 c$ lookup tables in C99.

```powershell
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml --inverted
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml --inverted
```

**Expect:** `data/csv/*-entropy.csv` + `.json` sidecar. Columns: original metadata, `bins`, `grid_levels` (`1|2|4|8`), then `E_1x1_r0_c0 … E_8x8_r7_c7` (85 values in $[0, 1]$, 8 decimals). Padding-only cells have entropy 0.

---

## 5. Asymmetry features $\rightarrow \mathbf{x}_{94}$ (schema `x94-v1`) (Task 06)

**What it does.** Reshape the entropy columns of level $g \in \{2, 4, 8\}$ into a matrix $E^{(g)} \in \mathbb{R}^{g\times g}$ and measure its deviation from three spatial symmetries $T \in \{\text{LR}, \text{TB}, \text{R180}\}$:

$$A_T^{(g)} = \frac{1}{g^2}\sum_{r,c}\left|E^{(g)}_{r,c} - T(E^{(g)})_{r,c}\right|$$

with $T_{LR}: c \to g-1-c$, $T_{TB}: r \to g-1-r$, $T_{R180}: (r,c)\to(g-1-r,\,g-1-c)$. Gives 9 columns appended in the order `asym_{lr,tb,rot180}_{2x2,4x4,8x8}`, yielding $\mathbf{x}_{94} = [\mathbf{x}_{85}, \mathbf{x}_9]$. The $1\times1$ global level has no asymmetry.

```powershell
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --all
```

**Expect:** four `data/csv/*-entropy-full.csv` (+ `.json`): `mvtec-ad`, `mvtec-ad-inverted`, `kolektorsdd2`, `kolektorsdd2-inverted`. Metadata and the 85 entropy columns are byte-identical to the `-entropy.csv` inputs; 94 feature columns in total.

---

## 6. Classical classifiers, ablation and explainability (Tasks 06A & 10)

**What it does.** Trains `dt`, `knn`, `lr`, `lr_dot`, `mlp_16_dot`, and `rf` on $\mathbf{x}_{94}$ (label good=0, defect=1) and automatically ablates feature subsets `x85_only`, `x9_only`, `x94_full`, `E_1x1`, `E_2x2`, `E_4x4`, `E_8x8`.

- **Splits.** KolektorSDD2: the official `train`/`test`. MVTec-AD: `ground_truth` rows discarded, then an in-memory stratified 80/20 split (`seed = 42`); CSVs are never modified.
- **`lr_dot`** folds `StandardScaler` + `LogisticRegression` into one dot product: $w'_i = w_i/\sigma_i,\; b' = b - \sum_i w_i\mu_i/\sigma_i$ $\rightarrow$ numerically identical to `lr` ($< 10^{-12}$), 95 parameters; **`mlp_16_dot`** (`MlpDotCoreClassifier`) similarly folds `StandardScaler` into its 16-ReLU hidden layer (1,537 parameters).
- **Symmetry gain:** $\Delta F_1 = F_1(\mathbf{x}_{94}) - F_1(\mathbf{x}_{85})$. **Chance share** of $\mathbf{x}_9$ importance is $9/94 = 9.57\,\%$.

```powershell
# KolektorSDD2
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all --inverted --overwrite

# MVTec-AD: canonical object (bottle)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier all --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier all --inverted --overwrite

# MVTec-AD: all 15 objects (all 6 classifiers)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier all --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier all --inverted --overwrite
```

**Expect** in `results/classification/<dataset>[_inverted]/<object|full>_<clf>/`:
- `metrics_summary.json` (hyperparameters, metrics, timings).
- `confusion_matrix_and_curves.png` (CM + ROC + PR curves).
- `descriptor_ablation_x85_x9_x94.png`, `ablation_table.csv` ($\Delta F_1$ symmetry gain).
- `explainability_importance_breakdown.png`, `feature_importance.csv`, `scale_and_family_breakdown.csv`.
- `spatial_importance_heatmaps.png`.

**Reference values (`seed = 42`; from HYPOTHESIS.md §2.4):**

| Dataset | Clf | `x85` F1/AUC | `x9` F1/AUC | `x94` F1/AUC | $\Delta F_1$ |
|---|---|---|---|---|---|
| kolektorsdd2 | lr / lr_dot | 0.2687 / 0.6179 | 0.3591 / 0.7371 | 0.4265 / 0.8061 | +0.1579 |
| kolektorsdd2 | rf (200 trees) | 0.4286 / 0.8255 | 0.3765 / 0.7429 | 0.4298 / 0.7904 | +0.0013 |
| mvtec bottle | lr / lr_dot | 0.7333 / 0.9181 | 0.7333 / 0.9130 | 0.8000 / 0.9231 | +0.0667 |
| mvtec bottle | rf (100 trees) | 0.5263 / 0.8696 | 0.6923 / 0.8980 | 0.7273 / 0.9030 | +0.2010 |

---

## 7. Lightweight CNN zoo (Task 07)

**What it does.** Trains 7 CNNs on 128×128 grayscale PNGs: `ultra_tiny` (1.1k), `ds_cnn` (12.8k), `tiny_cnn` (23.5k), `mobilenet_like` (70.7k), `tiny_resnet` (77.2k), `small_cnn` (97.6k), `torchvision_mobilenet` (1.5M, random init). Config: AdamW, lr 1e-3, cosine schedule, 40 epochs, early stopping 8, balanced class weights, `seed = 42`.

**Threshold calibration:** `val_f1` mode picks $\tau^* = \arg\max_\tau F_1^{\text{val}}$ on validation only (avoiding test leakage).

```powershell
# KolektorSDD2: 7 CNNs, normal and inverted
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model all --seeds 42 --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model all --seeds 42 --inverted --overwrite

# MVTec-AD bottle: 7 CNNs, normal and inverted
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object bottle --model all --seeds 42 --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object bottle --model all --seeds 42 --inverted --overwrite

# MVTec-AD all 15 objects (long run)
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --all-objects --model all --seeds 42 --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --all-objects --model all --seeds 42 --inverted --overwrite
```

**Expect** under `results/cnn/...`: `checkpoint.pt`, `metrics_summary.json` (metrics at calibrated $\tau^*$, at 0.5, and test-oracle), curves.
Reference (HYPOTHESIS.md §2.1):
- `kolektorsdd2`: best `tiny_resnet` F1 0.7130 / ROC-AUC 0.9213 (CNNs outperform `lr_dot` on micro-cracks).
- `bottle`: only `mobilenet_like` does not collapse (F1 0.7586, AUC 0.9415 vs `lr_dot` F1 0.8000).

---

## 8. 100% CPU-fair benchmark and deep profiling (Tasks 08 & 10)

**What it does.** Times the full pipeline on CPU, `threads = 1`, with **200 real test images** (seed 42), `gc` disabled, 20 warmups, 200 timed iterations, 3 repeats, bootstrap 95% CI (1000 resamples):

$$T_{total} = T_{descriptor} + T_{model}\quad(\text{ML}),\qquad T_{total} = T_{preprocess} + T_{model}\quad(\text{CNN})$$

Checkpoints from Stage 7 are re-evaluated on CPU and verified to match within $10^{-5}$.

```powershell
# Phase A: 100% CPU Benchmark (Pareto frontier, F1 vs latency)
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml --inverted
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle --inverted

# Phase B: Deep Pipeline Profiling (stage/layer dissection, ATen ops, Mann-Whitney U)
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml --inverted
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/mvtec_ad.toml --objects bottle
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/mvtec_ad.toml --objects bottle --inverted
```

**Expect:**
- `results/comparison_cpu/<tag>_baseline_vs_cnn_cpu.csv`, Pareto PNGs, `summary_all_datasets.csv`.
- `results/profiling/<tag>/`: `environment.json`, `descriptor_stage_times.csv`, `classifier_times.csv`, `cnn_layer_times_<model>.csv`, `statistical_tests.csv`, 6 figures, `README.md`.

**Reference (i7-7700HQ, `threads = 1`, `B = 1`; HYPOTHESIS.md §2.1):**
- `kolektorsdd2`: `lr_dot (x94)` = **0.3799 ms** vs `ultra_tiny` = 0.9919 ms vs `mobilenet_like` = 4.0077 ms.
- `bottle`: `lr_dot (x94)` = **0.3002 ms** vs `ultra_tiny` = 0.9171 ms vs `mobilenet_like` = 4.1313 ms.

---

## 9. Pure-Python pre-MCU benchmark (Optional / Reference)

Isolates what is lost without the C99 DLL (`numpy_finest_pyramid` vs `c_single_pass`). Output is Git-ignored.

```powershell
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml --entropy-backend numpy_finest_pyramid --output-dir results/comparison_cpu_python --no-cuda-reference
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle --entropy-backend numpy_finest_pyramid --output-dir results/comparison_cpu_python --no-cuda-reference
```

---

## 10. Embedded C99 pipeline export, parity & visualization (Task 11)

**What it does.** Exports self-contained C99 code, weights, Look-Up Tables, and test vectors (`f32` and `q15` fixed-point) into `c_imp/generated/`, compiles the desktop C test bench (`cimp_desktop.exe`), verifies bit-exact integer parity against Python, and produces diagnostic visualization figures.

```powershell
# 1. KolektorSDD2
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.verify_c_pipeline --config configs/c_imp/kolektorsdd2.toml --benchmark
.\.venv\Scripts\python.exe -m experiments.visualize_c_pipeline --config configs/c_imp/kolektorsdd2.toml

# 2. MVTec-AD (e.g. bottle; repeat --object <category> across all 15 categories for 16/16 targets)
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/mvtec_ad.toml --object bottle
.\.venv\Scripts\python.exe -m experiments.verify_c_pipeline --config configs/c_imp/mvtec_ad.toml --object bottle --benchmark
.\.venv\Scripts\python.exe -m experiments.visualize_c_pipeline --config configs/c_imp/mvtec_ad.toml --object bottle
```

**Expect:**
- `c_imp/generated/<target>/`: C headers, lookup tables, model weights (`lr_dot`, `mlp_16_dot`, and all 7 CNNs), and golden vector CSVs.
- `results/c_imp/<target>/`: `summary_report.md`, `summary_models_metrics.csv`, `memory_report.json`, and 6 diagnostic figures (`fig1..fig6`).
- Parity: 100% bit-exact integer matching on `q15` logits across all `16/16` targets (`0` raw diff).

---

## 11. Microcontroller hardware test bench (`PC → MCU → PC`) (Task 12)

**What it does.** Discovers connected serial ports, stages 8 deterministic KolektorSDD2 test images (`hw_test/img2send/`) with binary framing (`CRC-16/CCITT-FALSE`), evaluates cross-board Flash/SRAM/PSRAM fit (`nano`, `uno`, `nodemcu`, `esp32cam`), executes live UART inference on connected MCUs, compiles the 53-row hardware fit report, and renders 5 MDPI 300-DPI hardware figures.

```powershell
# 1. Discover boards and serial COM ports
.\.venv\Scripts\python.exe -m experiments.hw_test boards --config configs/hw_test/boards.toml

# 2. Stage 8 deterministic test images (seed 42)
.\.venv\Scripts\python.exe -m experiments.hw_test images --config configs/hw_test/boards.toml --all

# 3. Compile-only fit sweeps (unattended, no hardware required)
.\.venv\Scripts\python.exe -m experiments.hw_test bench --config configs/hw_test/boards.toml --board nano --compile-only
.\.venv\Scripts\python.exe -m experiments.hw_test bench --config configs/hw_test/boards.toml --board uno --compile-only
.\.venv\Scripts\python.exe -m experiments.hw_test bench --config configs/hw_test/boards.toml --board nodemcu --compile-only

# 4. Generate consolidated 53-row hardware fit report & 5 MDPI 300-DPI figures
.\.venv\Scripts\python.exe -m experiments.hw_test report --config configs/hw_test/boards.toml
.\.venv\Scripts\python.exe -m experiments.visualize_hw_test --config configs/visualization/hw_test.toml --overwrite

# 5. (Interactive) Flash and benchmark physically connected MCUs (e.g. ESP32-CAM on COM5, Nano on COM4, Uno on COM6):
# .\.venv\Scripts\python.exe -m experiments.hw_test flash --config configs/hw_test/boards.toml --board esp32cam --model lr_dot --mode q15 --port COM5
# .\.venv\Scripts\python.exe -m experiments.hw_test send --config configs/hw_test/boards.toml --board esp32cam --model lr_dot --mode q15 --transport uart --port COM5 --all --dump-descriptor
```

**Expect:**
- `results/hw_test/fit_report.csv` and `fit_report.md` (53 rows across `nano`, `uno`, `nodemcu`, `esp32cam`).
- `results/hw_test/figures/fig1_cross_board_all_mcus.png` .. `fig5_board_nodemcu.png` (5 MDPI 300-DPI figures).
- Live UART measurements in `results/FINAL_PAPER_REPORT.md §6D–6D-bis` (`esp32cam` `COM5`: 12 runs, 96/96 parity; `nano` `COM4` & `uno` `COM6`: 16/16 parity each).

---

## 12. Methodology figures, paper report & where to read the results

```powershell
.\.venv\Scripts\python.exe -m experiments.visualize_methodology --config configs/visualization/methodology.toml --overwrite
.\.venv\Scripts\python.exe -m experiments.generate_paper_report --config configs/reporting/paper_report.toml --overwrite
```

| Topic | Document / Path |
|---|---|
| Consolidated paper-ready report (16 targets, ESP32-CAM/Nano/Uno, captions) | [results/FINAL_PAPER_REPORT.md](results/FINAL_PAPER_REPORT.md) |
| End-to-end sequential reproduction script | [reproduce_paper_pipeline.ps1](reproduce_paper_pipeline.ps1) |
| Hypothesis verdicts & `486c111` baseline tables | [HYPOTHESIS.md](HYPOTHESIS.md) |
| Classification figures and ablations | [results/classification/README.md](results/classification/README.md) |
| 100% CPU benchmark & profiling findings | `results/profiling/<tag>/README.md` |
| C99 export, memory & parity audit (`16/16` targets) | [results/c_imp/README.md](results/c_imp/README.md) |
| Physical microcontroller test bench findings | [results/hw_test/README.md](results/hw_test/README.md) |
| Complete CLI flags per experiment | [experiments/README.md](experiments/README.md) |
| Binding project rules & conventions | [AGENTS.md](AGENTS.md) |
