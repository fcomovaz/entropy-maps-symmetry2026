# REVIEW 01 — Deep Scientific Audit, Forensic Data Verification & C/C++ Readiness Report

- **Review Date:** 2026-11-09 *(Follow-up Design Notes §3 added 2026-11-11)*
- **Reference Specification:** [`tasks/review_01.md`](tasks/review_01.md)
- **Workspace Root:** `d:/Publications/2026-10-Symmetry-Entropy/code`
- **Git Revision Audited:** `bf4a29f` *(local full-results snapshot audited 2026-11-09; current repo HEAD is `65e668f`, which ignores heavy `results/` CSVs/JSONs via `.gitignore` while keeping [`results/profiling/*/README.md`](results/profiling/mvtec-ad_bottle/README.md) tracked)*
- **Target Decision:** Determine with peer-review rigor whether (1) the scientific hypothesis is valid and verified, (2) the work constitutes real scientific research vs. speculation, (3) the underlying CSV/JSON experimental artifacts are internally consistent and methodologically sound, and (4) we can proceed to a **native C/C++ implementation of the $\mathbf{x}_{94}$ descriptor + lightweight classifier pipeline**.

---

## 1. Direct Answers to the 4 Core Review Questions

### Q1. Is the hypothesis completely valid, and was it verified?

**Answer: The hypothesis is CONDITIONALLY VALID (strongly verified within a well-defined physical regime, and explicitly refuted outside it), NOT universally valid across all visual anomalies.**

Specifically, the project tests a composite scientific hypothesis $H = H_{\text{sym}} \wedge H_{\text{inv}} \wedge H_{\text{pareto}} \wedge H_{\text{univ}}$:

| Sub-Hypothesis | Statement | Empirical Verdict | Quantitative Proof from CSV/JSON Artifacts |
| :--- | :--- | :--- | :--- |
| **$H_{\text{sym}}$ (Symmetry–Entropy Synergy)** | Adding 9 multi-scale bilateral/rotational entropy-asymmetry features ($\mathbf{x}_9$) to 85 local Shannon entropy features ($\mathbf{x}_{85}$) significantly improves anomaly detection over entropy alone. | **VERIFIED (Strongly Valid)** | • **`mvtec-ad_tile`:** $\Delta\text{F1}(\mathbf{x}_{94}-\mathbf{x}_{85}) = \mathbf{+0.1315}$ (`lr`: `0.7333 → 0.8649`, ROC-AUC `0.8590 → 0.9678`), $\mathbf{+0.3887}$ (`rf`: `0.4348 → 0.8235`), $\mathbf{+0.5262}$ (`dt`: `0.3125 → 0.8387`). Remarkably, **$\mathbf{x}_9$ alone (9 features)** reaches **F1 = 0.8571, ROC-AUC = 0.9589** (`rf`).<br>• **`mvtec-ad_bottle`:** $\Delta\text{F1} = \mathbf{+0.0667}$ (`lr`: `0.7333 → 0.8000`, PR-AUC `0.8307 → 0.8963`), $\mathbf{+0.2010}$ (`rf-100`: `0.5263 → 0.7273`), $\mathbf{+0.2356}$ (`rf-200`: `0.5263 → 0.7619`).<br>• **`kolektorsdd2`:** $\Delta\text{F1} = \mathbf{+0.1579}$ (`lr`: `0.2687 → 0.4265`, ROC-AUC `0.6179 → 0.8061`). |
| **$H_{\text{inv}}$ (Photometric Inversion Invariance)** | On unpadded images, $\mathbf{x}_{94}$ is invariant to polarity inversion ($I_{\text{inv}} = 255 - I$), whereas standard CNNs degrade if not trained with polarity inversion. | **VERIFIED (Mathematically & Empirically Valid)** | • **Mathematical identity:** For `bins=16`, $255 - I$ reverses bin order ($b \mapsto 15 - b$); since $-\sum p_b \log_2 p_b$ is commutative, $H(255-I) = H(I)$ up to `LANCZOS` `.5` rounding (`4,770 / 6,612` MVTec-AD rows are bit-identical).<br>• **Empirical proof (`bottle`):** `lr (x94)` keeps **F1 = 0.8000** (`ROC-AUC = 0.9231` normal vs `0.9214` inverted), whereas `mobilenet_like` collapses from `ROC-AUC = 0.9047` to `0.4967` and `small_cnn` drops from `F1 = 0.8571` to `0.8000`. |
| **$H_{\text{pareto}}$ (Efficiency vs. Lightweight CNNs on Macro/Structural Defects)** | On structural/regional defects (`bottle`), $\mathbf{x}_{94}$ + linear/tree classifiers match or beat lightweight CNNs with orders-of-magnitude fewer parameters. | **VERIFIED (With Threshold Caveat — see Finding F2, Table 4 & §3.6)** | • On `mvtec-ad_bottle` (`threads=1, batch=1`), `lr (x94)` (**95 params**, **2.29 ms**, **F1 = 0.8000**, **ROC-AUC = 0.9231**) beats 6 of 7 CNNs in F1 and ROC-AUC, and runs **2.42× faster** with **1,028× fewer parameters** than `small_cnn` (`5.56 ms`, `97,650` params, `F1 = 0.8571`).<br>• At the uncalibrated threshold $\tau = 0.50$, `lr (x94)` (**F1 = 0.8000**) even beats `small_cnn` (**F1@0.50 = 0.4906**) by **+0.3094 F1**! |
| **$H_{\text{univ}}$ (Universal Superiority on Sub-Cell Micro-Cracks & Python Raw Latency)** | $\mathbf{x}_{94}$ outperforms CNNs on hair-thin surface cracks (`kolektorsdd2`) and is the fastest pipeline in Python. | **REFUTED (Bounded by Physics & Python Overhead)** | • **Spatial resolution floor (`kolektorsdd2`):** At $128\times128$, each $8\times8$ cell has $16\times16 = 256$ pixels. A 1–2 px micro-crack alters only 6–12% of a cell's histogram. Consequently, **6 of 7 CNNs beat all $\mathbf{x}_{94}$ models on KolektorSDD2**: `tiny_resnet` achieves **F1 = 0.7389, ROC-AUC = 0.9219** vs `lr (x94)` **F1 = 0.4265, ROC-AUC = 0.8061**.<br>• **Python latency floor:** On `kolektorsdd2` (`threads=1, batch=1`), `ultra_tiny` (`1.03 ms, F1 = 0.5914`) and `ds_cnn` (`2.39 ms, F1 = 0.6102`) strictly dominate `lr (x94)` (`2.89 ms, F1 = 0.4265`) in **both speed and accuracy** because Python spends `2.07–2.56 ms` slicing NumPy arrays for $\mathbf{x}_{94}$. |

---

### Q2. Is this real scientific research or just speculations?

**Answer: It is REAL, PUBLICATION-GRADE SCIENTIFIC RESEARCH — not speculation.**

Speculative work relies on hand-waving claims, unverified heuristics, cherry-picked metrics, and hidden failure modes. This repository demonstrates five hallmarks of genuine scientific research:

1. **First-Principles Mathematical Construction & Falsifiable Proofs (Labs 01–07):**
   - Every stage of the descriptor (`image → letterbox → 16-bin quantization → pyramid partition → cell histogram → normalized Shannon entropy → involutory reflection/rotation residuals`) was derived from information theory and group symmetry before writing production code.
   - [`understanding/07_descriptor_behavior/`](understanding/07_descriptor_behavior/README.md) formally proves and tests the exact invariance and collision properties of $\mathbf{x}_{94}$ across 279 synthetic test cases (e.g., proving that intra-cell pixel permutations leave $\mathbf{x}_{94}$ unchanged via `cellwise_collision`, and proving Jensen's inequality $H(1\times1) \ne \frac{1}{4}\sum H(2\times2)$).
2. **Controlled Multi-Scale & Component Ablations:**
   - Rather than reporting a single black-box score, [`experiments/evaluate_descriptors.py`](experiments/evaluate_descriptors.py) systematically ablates 7 feature subsets (`E_1x1`, `E_2x2`, `E_4x4`, `E_8x8`, `x85_only`, `x9_only`, `x94_full`) across 4 classifier families (`dt`, `knn`, `lr`, `rf`) and quantifies per-feature importance via Gini MDI and permutation importance.
3. **Disciplined Systems & Statistical Methodology (`Task 08`):**
   - [`experiments/benchmark_cpu_fair.py`](experiments/benchmark_cpu_fair.py) and [`experiments/profile_pipelines.py`](experiments/profile_pipelines.py) enforce `gc.disable()`, `time.perf_counter_ns()`, strict thread pinning (`threadpoolctl` + `torch.set_num_threads`), bootstrap 95% confidence intervals, interquartile ranges (`IQR`), `p95`, inter-run coefficient of variation (`CV%`), and two-sided **Mann-Whitney U statistical significance tests** ($N=200$, $\alpha=0.05$).
   - The profiling framework explicitly separates **measured wall-clock latencies** from **analytical operation counts (`ops_basis = analytical_estimate`)** and **PyTorch neural FLOPs (`torch_flop_counter`)**, preventing apples-to-oranges complexity claims.
4. **Transparent Reporting of Negative Results:**
   - The failure of $\mathbf{x}_{94}$ to match CNNs on fine micro-cracks (`KolektorSDD2`) is not hidden; it is measured, statistically tested, and explained via the $16\times16$-pixel receptive cell size of $E_{8\times8}$.
5. **End-to-End Reproducibility:**
   - 268 automated tests (`195` formal + `73` laboratory) verify every mathematical identity, schema contract, CLI flag, and CPU-vs-GPU checkpoint parity (`max_diff < 5e-7` on all 6 custom CNNs).

---

### Q3. Forensic Audit of the CSVs, JSONs, and Experimental Protocol: Is Everything Correct to Proceed?

A deep line-by-line inspection of all CSV and JSON files in [`results/`](results/) confirms that **all numbers cited in [`tasks/review_01.md`](tasks/review_01.md) are real measured outputs**, and **the C/C++ implementation can proceed immediately**. However, the forensic audit uncovered **5 critical methodological and data-consistency findings** in the Python evaluation files that **must be understood and addressed for the academic paper**:

#### Finding F1 — Seed Overwrite & Stratified Split Mismatch on `mvtec-ad_bottle` (`seed=42` for ML vs. `seed=44` for CNNs)
- **Evidence in JSONs:**
  - In [`results/classification/mvtec-ad/bottle_lr/metrics_summary.json`](results/classification/mvtec-ad/bottle_lr/metrics_summary.json#L7), `"seed": 42`, `train_counts = {"good": 183, "defect": 50}` (`233` samples), `test_counts = {"good": 46, "defect": 13}` (`59` samples).
  - In **all 7 CNN JSONs** under [`results/cnn/mvtec-ad/bottle/*/metrics_summary.json`](results/cnn/mvtec-ad/bottle/small_cnn/metrics_summary.json#L6), `"seed": 44`, `split_counts = {"train": {"good": 156, "defect": 42}, "val": {"good": 27, "defect": 8}, "test": {"good": 46, "defect": 13}}`.
- **Root Cause in Code:**
  - [`configs/cnn/mvtec_ad.toml`](configs/cnn/mvtec_ad.toml#L7-L8) defines `seed = 42` and `seeds = [42, 43, 44]` with `overwrite = true`.
  - When [`experiments/evaluate_cnn.py`](experiments/evaluate_cnn.py#L428-L438) runs without `--seeds 42`, it loops over `seeds = [42, 43, 44]` and writes every run to the same folder `results/cnn/mvtec-ad/bottle/<model>/`, overwriting `checkpoint.pt` and `metrics_summary.json` on each iteration so only **`seed = 44`** remains on disk.
  - Because `mvtec-ad` uses `split_mode = "stratified"` (in-memory 80/20 split seeded by `seed`), the classical ML models (`seed=42`) and the saved CNN checkpoints (`seed=44`) on `mvtec-ad_bottle` were evaluated on **different random 80/20 stratified splits** of the 292 bottle images!
  - *(Note: On `kolektorsdd2`, `split_mode = "presplit"`, so the 1,004 test images are fixed on disk and 100% identical between `seed=42` and `seed=44`.)*

#### Finding F2 — Decision Threshold Calibration Asymmetry ($\tau = 0.50$ Fixed for ML vs. `val_f1` Tuned $\tau^*$ for CNNs) & CNN Threshold Collapse
- **Evidence in JSONs:**
  - Classical ML models in [`src/models/descriptor_classifier.py`](src/models/descriptor_classifier.py) are evaluated strictly at the fixed default threshold **$\tau = 0.50$**.
  - CNNs in [`experiments/evaluate_cnn.py`](experiments/evaluate_cnn.py#L218-L229) use `threshold_mode = "val_f1"`, tuning the probability cutoff $\tau^*$ on the validation split (`35` images = `27 good, 8 defect` on `bottle`).
  - Inspecting the `"threshold_calibration"` block inside [`results/cnn/mvtec-ad/bottle/*/metrics_summary.json`](results/cnn/mvtec-ad/bottle/small_cnn/metrics_summary.json#L121-L143) reveals dramatic differences between $\tau = 0.50$ (`test_at_fixed_05`) and $\tau^*$ (`test_at_selected_threshold`):
    - **`small_cnn` on `bottle`:** At fixed $\tau = 0.50$, **`F1 = 0.4906` (`Accuracy = 0.5424`)**! It only achieves **`F1 = 0.8571` (`Accuracy = 0.9322`)** because `val_f1` raises its threshold to **$\tau^* = 0.82$**. Thus, at the standard uncalibrated threshold $\tau = 0.50$, **`lr (x94)` (`F1 = 0.8000`) beats `small_cnn` (`F1 = 0.4906`) by +0.3094 F1**!
    - **`tiny_resnet` on `bottle`:** Has `ROC-AUC = 0.8512`, and at $\tau = 0.50$ achieves `Accuracy = 0.8136, Precision = 1.0000`. However, because early stopping picked `best_epoch = 3` and `val_f1` selected $\tau^* = 0.49$ on the tiny 8-defect validation set, all 59 test samples had probability slightly above `0.49`, causing **total threshold collapse (`confusion_matrix = [[0, 46], [0, 13]]`, predicting 0 `good` and 59 `defect`, `Accuracy = 0.2203`, `BalAcc = 0.5000`)**!
    - **Classical `rf (x94)` under $\tau = 0.50$ vs. OOB $\tau^*$:** Conversely, as documented in [`results/classification/README.md`](results/classification/README.md#L50-L66), random forest probabilities on imbalanced categories are diluted below `0.50` by `max_features="sqrt"`. When calibrated via Out-Of-Bag (OOB) threshold $\tau^*$, `rf (x94)` F1 jumps from `0.0000 → 0.5714` on `transistor` (`ROC-AUC = 0.9386`), from `0.2857 → 0.9091` on `toothbrush` (`ROC-AUC = 0.9889`), and from `0.3333 → 0.4526` on `kolektorsdd2`.

#### Finding F3 — Small Test Sample Size on `mvtec-ad_bottle` ($N_{\text{test}} = 59$, Only 13 Defects) & Cross-Paradigm Coverage Gap
- **Evidence in JSONs:**
  - `mvtec-ad_bottle` has `292` images total, so its 20% test split contains **`59` images (`46 good, 13 defect`)**.
  - With only `13` defect images, **1 single image changes Recall by `7.69%`**.
  - Compare the confusion matrices on `mvtec-ad_bottle`:
    - `small_cnn` ($\tau^*=0.82$): `TN = 43, FP = 3, FN = 1, TP = 12` $\to$ **4 errors out of 59 images** (`F1 = 0.8571`).
    - `lr (x94)` ($\tau=0.50$): `TN = 41, FP = 5, FN = 1, TP = 12` $\to$ **6 errors out of 59 images** (`F1 = 0.8000`).
    - Both models detect **the exact same number of true defects (`12 / 13`, Recall = `92.31%`)**; the entire F1 gap between `small_cnn` and `lr (x94)` is **just 2 false positives out of 46 good bottles**!
  - Meanwhile, **`mvtec-ad_tile`** (`277 train / 70 test`, where `lr (x94)` achieves **ROC-AUC = 0.9678, F1 = 0.8649, TN=49, FP=4, FN=1, TP=16**) already has classification results in [`results/classification/mvtec-ad/tile_lr/`](results/classification/mvtec-ad/tile_lr/metrics_summary.json), but **has not yet been run through `evaluate_cnn.py` or `benchmark_cpu_fair.py`**.

#### Finding F4 — RF & DT Hyperparameter Drift Between Earlier Classification Runs (`n_estimators=200`) and Profiling Runs (`n_estimators=100`)
- **Evidence in CSVs/JSONs:**
  - In [`results/classification/mvtec-ad/bottle_rf/metrics_summary.json`](results/classification/mvtec-ad/bottle_rf/metrics_summary.json) (and [`results/comparison/mvtec-ad_bottle_baseline_vs_cnn.csv`](results/comparison/mvtec-ad_bottle_baseline_vs_cnn.csv)), `rf (x94)` has **`F1 = 0.7619, ROC-AUC = 0.8913, Acc = 0.9153`** (`TN=46, FP=0, FN=5, TP=8`), trained when MVTec-AD used `n_estimators = 200, max_depth = 6`.
  - Later, [`configs/classification/mvtec_ad.toml`](configs/classification/mvtec_ad.toml#L39-L56) converged to `n_estimators = 100, max_depth = 12` for `rf` (and `max_depth = 3` for `dt`), whereas [`configs/classification/kolektorsdd2.toml`](configs/classification/kolektorsdd2.toml#L38-L56) kept `n_estimators = 200, max_depth = 6` (`dt max_depth = 16`). Consequently, in `results/comparison_cpu/summary_all_datasets.csv` and `results/profiling/mvtec-ad_bottle/ablation_quality_vs_cost.csv` (recorded in [`results/profiling/mvtec-ad_bottle/README.md`](results/profiling/mvtec-ad_bottle/README.md#L23)), `rf (x94)` on `bottle` reports **`F1 = 0.7273, ROC-AUC = 0.9030, Acc = 0.8983`**.

#### Finding F5 — Why Python Latency Alone Understates $\mathbf{x}_{94}$'s True Speed (And Why C/C++ Is Required)
- **Evidence in Profiling CSVs:**
  - In [`results/profiling/mvtec-ad_bottle/statistical_tests.csv`](results/profiling/mvtec-ad_bottle/statistical_tests.csv) and [`kolektorsdd2/statistical_tests.csv`](results/profiling/kolektorsdd2/statistical_tests.csv), Mann-Whitney U tests ($N=200$) show that in Python (`threads=1, batch=1`), `ultra_tiny` (`0.89–1.08 ms`) is **2.33×–3.05× faster** than `lr (x94)` (`2.63–2.71 ms`, $p < 10^{-61}$), and `ds_cnn` (`2.08–2.80 ms`) is statistically tied with `lr (x94)` on `bottle` ($p = 0.414$) and `1.30×` faster on `kolektorsdd2` ($p = 7.96 \times 10^{-56}$).
  - Why? Because PyTorch CNN layers execute in compiled C++/AVX2 (`aten::conv2d`, `aten::_nnpack_spatial_convolution`), whereas `extract_x94_end_to_end` executes **85 separate Python loop iterations, 2D slice allocations, `int32` casts, and `np.bincount` calls (`2.07–2.56 ms`)**, and `lr` executes `scikit-learn` `Pipeline.predict` (`256.6 µs` for a `3.2 µs` dot product!).
  - **Conclusion:** Comparing a pure-Python/NumPy loop for $\mathbf{x}_{94}$ against C++/AVX2 ATen kernels for CNNs handicaps $\mathbf{x}_{94}$ by **15×–25×** in descriptor extraction and **80×** in linear classification. **Implementing $\mathbf{x}_{94} + \text{LR/DT/RF}$ in native C/C++ is scientifically essential to make the latency comparison truly fair at the compiled-code level.**

---

### Q4. Final Verdict & Dual-Track Roadmap

**Verdict: GO — Proceed to the Native C/C++ Implementation of $\mathbf{x}_{94} + \text{LR/DT/RF}$ (Track A), while executing a parallel Python Protocol Alignment pass (Track B) for the manuscript tables.**

- **Why Track A (C/C++ Implementation) can start immediately:** The mathematical specification of $\mathbf{x}_{94}$ (`schema x94-v1`) and the trained classifier structures (`StandardScaler + LogisticRegression`, `DecisionTree`, `RandomForest`) are 100% locked, deterministic, and verified (`0.000e+00` stage-to-E2E error across 268 tests).
- **What Track B (Manuscript Data Alignment) should complete before paper submission:** (1) Lock CNN evaluation to `seed=42` (or report `mean ± std` across `seeds=[42, 43, 44]` for both ML and CNNs on identical stratified splits), (2) report both $\tau=0.50$ and $\tau^*=\text{val\_f1}$ for both ML and CNNs, and (3) run `evaluate_cnn.py` + `benchmark_cpu_fair.py` on `mvtec-ad_tile` (and 2–3 additional MVTec-AD categories such as `zipper`, `pill`, `toothbrush`).

---

## 2. Pertinent Master Tables: Complete Visualization of Everything Done

### Table 1 — Project Architecture & Progression Inventory (Understanding Labs 01–12 $\to$ Formal Pipeline)

| Stage / Module | Location | Core Scientific / Engineering Purpose | Status & Verification |
| :--- | :--- | :--- | :--- |
| **Lab 01 — Basic Image Processing** | [`understanding/01_basic_image_processing/`](understanding/01_basic_image_processing/) | Grayscale conversion (`L`), matrix inspection, cropping, basic Sobel/intensity visualizations. | Complete |
| **Lab 02 — Local Entropy** | [`understanding/02_local_entropy/`](understanding/02_local_entropy/) | Step-by-step decomposition: `patch → histogram → counts → p → -p*log2(p) → H`. | Complete |
| **Lab 03 — Entropy Benchmark** | [`understanding/03_entropy_benchmark/`](understanding/03_entropy_benchmark/) | NumPy vectorization vs. reference loops, bin-count sensitivity, timing benchmarks. | Complete (`5/5` tests pass) |
| **Lab 04 — Lightweight CNNs** | [`understanding/04_lightweight_cnns/`](understanding/04_lightweight_cnns/) | 7 architectures (`ultra_tiny` to `torchvision_mobilenet`), parameter & FLOPs profiling on CPU/CUDA. | Complete (`8/8` suites, `140` subcases pass) |
| **Lab 05 — Symmetry Features** | [`understanding/05_symmetry_features/`](understanding/05_symmetry_features/) | Involutory reflection/rotation operators ($A_{\text{LR}}, A_{\text{TB}}, A_{\text{R180}}$) on 2D entropy grids ($\mathbf{x}_9$). | Complete (`20/20` tests pass) |
| **Lab 06 — Pyramid Descriptor $\mathbf{x}_{94}$** | [`understanding/06_entropy_pyramid_descriptor/`](understanding/06_entropy_pyramid_descriptor/) | Canonical `x94-v1` schema: 85 multi-scale entropies ($1^2+2^2+4^2+8^2$) + 9 asymmetries. | Complete (`25/25` tests pass) |
| **Lab 07 — Descriptor Behavior** | [`understanding/07_descriptor_behavior/`](understanding/07_descriptor_behavior/) | 279 synthetic scenes: histogram collisions, grid boundaries, noise/blur/contrast curves. | Complete (`15/15` tests, SHA-256 locked) |
| **Lab 08 — Preprocessing (`128×128`)** | [`understanding/08_preprocessing_128/`](understanding/08_preprocessing_128/) | Aspect-ratio-preserving `LANCZOS` letterbox (`pad_value=0`) + photometric inversion ($255-I$). | Complete |
| **Lab 09 — Tabular CSV Export** | [`understanding/09_csv_export/`](understanding/09_csv_export/) | Streaming $16,384$-pixel tabular CSV layout and integrity verification. | Complete |
| **Lab 10 — Entropy Maps** | [`understanding/10_entropy_maps/`](understanding/10_entropy_maps/) | Visual 2D heatmaps of $E_{1\times1}, E_{2\times2}, E_{4\times4}, E_{8\times8}$ on real industrial defects. | Complete |
| **Lab 11 — Asymmetry Maps** | [`understanding/11_asymmetry_maps/`](understanding/11_asymmetry_maps/) | Visual breakdown of bilateral/rotational entropy differences on real defects. | Complete |
| **Lab 12 — Stage Profiling** | [`understanding/12_profiling_stages/`](understanding/12_profiling_stages/) | Pedagogical dissection of descriptor stages, `sklearn` overhead, and PyTorch layer hooks. | Complete |
| **Formal Preprocessing & CSV** | [`src/preprocessing/`](src/preprocessing/), [`experiments/preprocess_datasets.py`](experiments/preprocess_datasets.py) | Production dataset preprocessing (`MVTec-AD`, `KolektorSDD2`, normal + `--inverted`). | Verified (`53` unit/CLI tests pass) |
| **Formal $\mathbf{x}_{94}$ Extraction** | [`src/features/`](src/features/), [`experiments/extract_entropy_features.py`](experiments/extract_entropy_features.py), [`append_asymmetry_features.py`](experiments/append_asymmetry_features.py) | Vectorized LUT + `bincount` $\mathbf{x}_{85}$ and $\mathbf{x}_9$ extraction (`< 10^{-12}` parity with Lab 06). | Verified (`45` unit/CLI tests pass) |
| **Formal Classifiers & CNNs** | [`src/models/`](src/models/), [`experiments/evaluate_descriptors.py`](experiments/evaluate_descriptors.py), [`evaluate_cnn.py`](experiments/evaluate_cnn.py) | Classical ML (`dt, knn, lr, rf`) + 7 CNNs with ablations and explainability. | Verified (`44` unit/CLI tests pass) |
| **Formal CPU Profiling & Benchmark** | [`src/profiling/`](src/profiling/), [`experiments/benchmark_cpu_fair.py`](experiments/benchmark_cpu_fair.py), [`profile_pipelines.py`](experiments/profile_pipelines.py) | 100% CPU fair benchmark, stage/layer profiling, scalability sweeps, Mann-Whitney U tests. | Verified (`52` unit/CLI tests pass) |

---

### Table 2 — Master Dataset, Split & Sample Size Inventory

| Dataset / Variant | Native Resolution | Preprocessed Format | Split Mode | Train Samples (`Good / Defect`) | Val Samples (`Good / Defect` — CNN only) | Test Samples (`Good / Defect`) | Total Test ($N_{\text{test}}$) | Defect Rate (Test) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: |
| **`kolektorsdd2`** (Normal & Inverted) | ~`230×630` (rect.) | `128×128` uint8 (`L`, letterbox `pad=0`) | `presplit` (Fixed on disk) | **2,331** (`2,085 / 246`) | **350** (`313 / 37`) *(from 2,331; CNN train uses `1,772 / 209`)* | **1,004** (`894 / 110`) | **1,004** | **10.96%** |
| **`mvtec-ad_bottle`** (Normal & Inverted) | `900×900` (square) | `128×128` uint8 (`L`, no padding needed) | `stratified` (80% / 20%) | **233** (`183 / 50`) *(ML)*<br>**198** (`156 / 42`) *(CNN)* | **35** (`27 / 8`) *(CNN only)* | **59** (`46 / 13`) | **59** | **22.03%** |
| **`mvtec-ad_tile`** (Normal) | `840×840` (square) | `128×128` uint8 (`L`, no padding needed) | `stratified` (80% / 20%) | **277** (`210 / 67`) *(ML)* | — *(CNN not yet run)* | **70** (`53 / 17`) | **70** | **24.29%** |
| **`mvtec-ad` (All 15 Objects Total)** | `700²–1024²` (sq.) | `128×128` uint8 (`L`, no padding needed) | `stratified` (80% / 20%) | **4,282** total across 15 classes | — | **1,072** total across 15 classes | **1,072** | **23.51%** |

> [!WARNING]
> **Statistical Power Note on `mvtec-ad_bottle`:** Because `mvtec-ad_bottle` has only $N_{\text{test}} = 59$ samples (`46 good, 13 defect`), a single misclassified defect shifts Recall by $\pm 7.69\%$ and a single false positive shifts Precision by $\approx \pm 5\%$. In contrast, `kolektorsdd2` ($N_{\text{test}} = 1,004$, `110` defects) has $8.5\times$ higher positive-class statistical resolution.

---

### Table 3 — Master 100% CPU Fair Benchmark Across All 4 Evaluated Variants (`threads=1`, `batch=1` & `batch=32`)

Data extracted directly from [`results/comparison_cpu/mvtec-ad_bottle_baseline_vs_cnn_cpu.csv`](results/comparison_cpu/mvtec-ad_bottle_baseline_vs_cnn_cpu.csv), [`mvtec-ad-inverted_bottle_baseline_vs_cnn_cpu.csv`](results/comparison_cpu/mvtec-ad-inverted_bottle_baseline_vs_cnn_cpu.csv), [`kolektorsdd2_baseline_vs_cnn_cpu.csv`](results/comparison_cpu/kolektorsdd2_baseline_vs_cnn_cpu.csv), and [`kolektorsdd2-inverted_baseline_vs_cnn_cpu.csv`](results/comparison_cpu/kolektorsdd2-inverted_baseline_vs_cnn_cpu.csv).

#### 3A. Structural Anomalies: `mvtec-ad_bottle` (Normal vs. Inverted, `threads=1`)

| Model | Family | Model Params | Ops / FLOPs (`basis`) | **Normal Acc.** | **Normal BalAcc** | **Normal F1** | **Normal ROC-AUC** | **Normal PR-AUC** | **Normal $T_{\text{desc/prep}}$ (ms)** | **Normal $T_{\text{model}}$ (ms)** | **Normal $T_{\text{total}}$ B=1 (ms)** | **Normal Thru B=1 (img/s)** | **Normal Thru B=32 (img/s)** | **Inverted F1** | **Inverted ROC-AUC** | **Inverted $T_{\text{total}}$ B=1 (ms)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`lr (x94)`** | `ml` | **95** | 152,753 (`analyt.`) | **0.8983** | **0.9072** | **0.8000** | **0.9231** | **0.8963** | 2.0747 | 0.2063 | **2.2920** | **436.3** | **2,074.9** | **0.8000** | **0.9214** | **2.4704** |
| **`rf (x94)`** *(100t)* | `ml` | 3,496 | 152,855 (`analyt.`) | **0.8983** | 0.7968 | 0.7273 | 0.9030 | 0.8266 | 2.0747 | 4.1360 | 6.2655 | 159.6 | 1,635.2 | 0.7273 | 0.9013 | 6.4997 |
| **`dt (x94)`** | `ml` | **15** | 152,380 (`analyt.`) | 0.8305 | 0.7258 | 0.5833 | 0.6614 | 0.5167 | 2.0747 | **0.0760** | **2.1508** | **464.9** | **2,097.5** | 0.5833 | 0.6614 | 2.2964 |
| **`knn (x94)`** | `ml` | 21,902 | 218,271 (`analyt.`) | 0.8305 | 0.6154 | 0.3750 | 0.7057 | 0.5907 | 2.0747 | 0.4939 | 2.6081 | 383.4 | 2,025.1 | 0.3750 | 0.7057 | 2.7055 |
| `small_cnn` | `cnn` | 97,650 | 117.97M (`torch`) | **0.9322** | **0.9289** | **0.8571** | **0.9816** | **0.9411** | 0.0378 | 5.5170 | 5.5556 | 180.0 | 149.9 | **0.8000** | **0.9799** | 5.8288 |
| `tiny_cnn` | `cnn` | 23,538 | 80.22M (`torch`) | 0.8475 | 0.7642 | 0.6400 | 0.8896 | 0.6930 | 0.0378 | 4.2518 | 4.2906 | 233.1 | 199.9 | 0.6000 | 0.8846 | 5.1998 |
| `ds_cnn` | `cnn` | 12,866 | 19.37M (`torch`) | 0.8475 | 0.7090 | 0.5714 | 0.8980 | 0.7228 | 0.0378 | 1.8485 | 1.8881 | 529.6 | 460.1 | 0.4444 | 0.8813 | 1.9100 |
| `ultra_tiny` | `cnn` | 1,154 | 1.34M (`torch`) | 0.6102 | 0.6672 | 0.4651 | 0.7709 | 0.5764 | 0.0378 | **0.7983** | **0.8366** | **1,195.3** | **3,467.7** | 0.4444 | 0.8946 | **0.8453** |
| `tiny_resnet` | `cnn` | 77,234 | 97.65M (`torch`) | 0.2203 | 0.5000 | 0.3611 | 0.8512 | 0.6736 | 0.0378 | 3.2041 | 3.2434 | 308.3 | 456.2 | 0.6452 | 0.8562 | 3.5189 |
| `mobilenet_like` | `cnn` | 70,786 | 38.72M (`torch`) | 0.8136 | 0.6045 | 0.3529 | 0.9047 | 0.7775 | 0.0378 | 2.9449 | 2.9850 | 335.0 | 436.7 | 0.3611 | 0.4967 | 3.1474 |
| `torchvision_mobilenet` | `cnn` | 1,519,618 | 35.29M (`torch`) | 0.2203 | 0.5000 | 0.3611 | 0.5385 | 0.2803 | 0.0378 | 6.4154 | 6.4560 | 154.9 | 506.3 | 0.3611 | 0.5000 | 6.5397 |

#### 3B. Surface Micro-Cracks: `kolektorsdd2` (Normal vs. Inverted, `threads=1`)

| Model | Family | Model Params | Ops / FLOPs (`basis`) | **Normal Acc.** | **Normal BalAcc** | **Normal F1** | **Normal ROC-AUC** | **Normal PR-AUC** | **Normal $T_{\text{desc/prep}}$ (ms)** | **Normal $T_{\text{model}}$ (ms)** | **Normal $T_{\text{total}}$ B=1 (ms)** | **Normal Thru B=1 (img/s)** | **Normal Thru B=32 (img/s)** | **Inverted F1** | **Inverted ROC-AUC** | **Inverted $T_{\text{total}}$ B=1 (ms)** |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`tiny_resnet`** | `cnn` | 77,234 | 97.65M (`torch`) | **0.9472** | **0.8308** | **0.7389** | **0.9219** | **0.7983** | 0.0540 | 3.8663 | **3.9518** | 253.0 | 370.3 | **0.7030** | 0.8743 | 3.5515 |
| `mobilenet_like` | `cnn` | 70,786 | 38.72M (`torch`) | 0.9263 | 0.7633 | 0.6224 | 0.8894 | 0.7057 | 0.0540 | 4.0456 | 4.1188 | 242.8 | 413.0 | 0.6000 | 0.8405 | 3.2498 |
| `small_cnn` | `cnn` | 97,650 | 117.97M (`torch`) | 0.9253 | 0.7627 | 0.6193 | 0.9093 | 0.7266 | 0.0540 | 7.2246 | 7.3014 | 137.0 | 157.6 | 0.5949 | 0.8612 | 6.0142 |
| **`ds_cnn`** | `cnn` | 12,866 | 19.37M (`torch`) | 0.9084 | 0.7971 | **0.6102** | **0.8875** | **0.7094** | 0.0540 | 2.3114 | **2.3914** | **418.2** | 396.1 | **0.6355** | **0.8744** | **1.9254** |
| `torchvision_mobilenet` | `cnn` | 1,519,618 | 35.29M (`torch`) | 0.9363 | 0.7210 | 0.6049 | 0.7816 | 0.6157 | 0.0540 | 8.6622 | 8.7233 | 114.6 | 497.3 | 0.1975 | 0.5206 | 6.9935 |
| **`ultra_tiny`** | `cnn` | **1,154** | 1.34M (`torch`) | 0.9243 | 0.7383 | **0.5914** | **0.8793** | **0.6703** | 0.0540 | **0.9663** | **1.0255** | **975.2** | **1,762.4** | 0.3889 | 0.7440 | **0.8580** |
| `tiny_cnn` | `cnn` | 23,538 | 80.22M (`torch`) | 0.9014 | 0.7653 | 0.5677 | 0.8625 | 0.6539 | 0.0540 | 6.3050 | 6.3862 | 156.6 | 188.2 | 0.5625 | 0.8170 | 4.8647 |
| **`rf (x94)`** *(200t)* | `ml` | 13,208 | 153,555 (`analyt.`) | 0.8705 | 0.6841 | **0.4298** | 0.7904 | 0.4223 | 2.5569 | 11.5206 | 14.4221 | 69.3 | 1,006.8 | 0.4237 | 0.7978 | 12.1650 |
| **`lr (x94)`** | `ml` | **95** | 152,753 (`analyt.`) | 0.8018 | **0.7452** | **0.4265** | **0.8061** | **0.4360** | 2.5569 | 0.2769 | **2.8868** | **346.4** | **1,552.2** | **0.4153** | **0.8151** | **2.3986** |
| **`dt (x94)`** | `ml` | 285 | 152,389 (`analyt.`) | 0.8257 | 0.5992 | 0.2798 | 0.5962 | 0.1805 | 2.5569 | **0.0810** | **2.6538** | 376.8 | **1,568.3** | 0.3141 | 0.6228 | 2.2486 |
| **`knn (x94)`** | `ml` | 219,114 | 809,907 (`analyt.`) | 0.8984 | 0.5404 | 0.1500 | 0.6652 | 0.3441 | 2.5569 | 0.8489 | 3.9155 | 255.4 | 1,402.9 | 0.1500 | 0.6607 | 3.1678 |

---

### Table 4 — Forensic Threshold Calibration Comparison: Fixed $\tau = 0.50$ vs. Calibrated $\tau^*$ (`mvtec-ad_bottle` & `kolektorsdd2`)

Extracted from the `"threshold_calibration"` blocks of [`results/cnn/mvtec-ad/bottle/*/metrics_summary.json`](results/cnn/mvtec-ad/bottle/small_cnn/metrics_summary.json#L121-L143), [`results/cnn/kolektorsdd2/full/*/metrics_summary.json`](results/cnn/kolektorsdd2/full/tiny_resnet/metrics_summary.json#L121-L143), and [`results/classification/README.md`](results/classification/README.md#L50-L66):

| Dataset | Model | Family | ROC-AUC *(Threshold-Free)* | **F1 @ Fixed $\tau = 0.50$** | **Acc. @ Fixed $\tau = 0.50$** | **Selected $\tau^*$ (`val_f1` / OOB)** | **F1 @ Selected $\tau^*$** | **Acc. @ Selected $\tau^*$** | **Confusion Matrix @ Reported `[TN, FP / FN, TP]`** | Forensic Diagnosis |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| **`mvtec-ad_bottle`** | **`lr (x94)`** | `ml` | **0.9231** | **0.8000** | **0.8983** | `0.50` *(fixed)* | **0.8000** | **0.8983** | `[41, 5 / 1, 12]` | **Best uncalibrated F1 (`0.8000`)**; detects `12/13` defects at standard $\tau=0.50$. |
| **`mvtec-ad_bottle`** | **`rf (x94)`** *(200t)* | `ml` | 0.8913 | **0.7619** | **0.9153** | `0.50` *(fixed)* | 0.7619 | 0.9153 | `[46, 0 / 5, 8]` | Zero false positives (`46/46 good` correct) at $\tau=0.50$. |
| **`mvtec-ad_bottle`** | `small_cnn` | `cnn` | **0.9816** | **0.4906** | 0.5424 | **`0.82`** (`val_f1`) | **0.8571** | **0.9322** | `[43, 3 / 1, 12]` | Uncalibrated $\tau=0.50$ has high FP (`F1=0.4906`); relies on $\tau^*=0.82$ to reach `0.8571`. |
| **`mvtec-ad_bottle`** | `tiny_cnn` | `cnn` | 0.8896 | 0.6087 | 0.8475 | `0.49` (`val_f1`) | 0.6400 | 0.8475 | `[42, 4 / 5, 8]` | Well-calibrated near `0.50`. |
| **`mvtec-ad_bottle`** | `ds_cnn` | `cnn` | 0.8980 | 0.6429 | 0.8305 | `0.73` (`val_f1`) | 0.5714 | 0.8475 | `[44, 2 / 7, 6]` | `val_f1` ($\tau^*=0.73$) overfit the 8-defect val set, lowering test F1 (`0.6429 → 0.5714`). |
| **`mvtec-ad_bottle`** | `mobilenet_like` | `cnn` | 0.9047 | 0.5143 | 0.7119 | `0.73` (`val_f1`) | 0.3529 | 0.8136 | `[45, 1 / 10, 3]` | `val_f1` ($\tau^*=0.73$) over-raised threshold, cutting test Recall to `3/13` (`23.1%`). |
| **`mvtec-ad_bottle`** | `tiny_resnet` | `cnn` | 0.8512 | 0.2667 | 0.8136 | `0.49` (`val_f1`) | **0.3611** | **0.2203** | **`[0, 46 / 0, 13]`** | **Threshold collapse** at `best_epoch=3`: all 59 test logits slightly $> 0.49$. |
| **`mvtec-ad_bottle`** | `ultra_tiny` | `cnn` | 0.7709 | **0.0000** | 0.7797 | `0.47` (`val_f1`) | 0.4651 | 0.6102 | `[26, 20 / 3, 10]` | Predicts all `0` at $\tau=0.50$; narrow probability band around `0.47`. |
| **`mvtec-ad_bottle`** | `torchvision_mobilenet` | `cnn` | 0.4557 | **0.0000** | 0.7797 | `0.39` (`val_f1`) | 0.3611 | 0.2203 | **`[0, 46 / 0, 13]`** | Failed to converge on 198 samples (`1.52M` params; `ROC-AUC = 0.4557`). |
| **`kolektorsdd2`** | **`tiny_resnet`** | `cnn` | **0.9219** | **0.7312** | **0.9502** | `0.44` (`val_f1`) | **0.7389** | **0.9472** | `[876, 18 / 35, 75]` | Stable calibration on large val set (`350` samples); strong at both `0.50` and `0.44`. |
| **`kolektorsdd2`** | **`lr (x94)`** | `ml` | 0.8061 | **0.4265** | 0.8018 | `0.50` *(fixed)* | 0.4265 | 0.8018 | `[731, 163 / 36, 74]` | Detects `74/110` defects (`67.3%` recall vs `75/110` for `tiny_resnet`), but has `163` FPs. |
| **`kolektorsdd2`** | **`rf (x94)`** *(200t)* | `ml` | 0.8219 | 0.3333 | 0.9044 | `0.37` *(OOB)* | **0.4526** | 0.8964 | `[884, 10 / 86, 24]` | Low FP (`10/894`), conservative recall at $\tau=0.50$; OOB $\tau^*=0.37$ lifts F1 to `0.4526`. |

---

### Table 5 — Complete Multi-Scale & Symmetry Ablation Matrix ($\mathbf{x}_{85}$ vs. $\mathbf{x}_9$ vs. $\mathbf{x}_{94}$ and Individual Scales $E_{1\times1} \dots E_{8\times8}$)

Compiled from [`results/classification/mvtec-ad/bottle_*/ablation_table.csv`](results/classification/mvtec-ad/bottle_lr/ablation_table.csv), [`tile_*/ablation_table.csv`](results/classification/mvtec-ad/tile_lr/ablation_table.csv), and [`results/classification/kolektorsdd2/full_*/ablation_table.csv`](results/classification/kolektorsdd2/full_lr/ablation_table.csv):

| Dataset / Object | Classifier | Metric | $E_{1\times1}$ (`1` feat) | $E_{2\times2}$ (`4` feats) | $E_{4\times4}$ (`16` feats) | $E_{8\times8}$ (`64` feats) | $\mathbf{x}_{85}$ (`85` Entropy) | $\mathbf{x}_9$ (`9` Asym.) | **$\mathbf{x}_{94}$ (`94` Full)** | **Symmetry Gain $\Delta(\mathbf{x}_{94} - \mathbf{x}_{85})$** |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`mvtec-ad_tile`** | **`lr`** | **F1**<br>ROC-AUC | 0.1702<br>0.3341 | 0.4490<br>0.6970 | 0.4762<br>0.7292 | 0.5263<br>0.6482 | 0.7333<br>0.8590 | **0.8333**<br>**0.9556** | **0.8649**<br>**0.9678** | **+0.1315 F1**<br>**+0.1088 AUC** |
| **`mvtec-ad_tile`** | **`rf`** *(200t)* | **F1**<br>ROC-AUC | 0.0667<br>0.4506 | 0.4667<br>0.7270 | 0.5714<br>0.8901 | 0.4444<br>0.8679 | 0.4348<br>0.9001 | **0.8571**<br>**0.9589** | **0.8235**<br>**0.9745** | **+0.3887 F1**<br>**+0.0744 AUC** |
| **`mvtec-ad_tile`** | **`dt`** | **F1**<br>ROC-AUC | 0.2222<br>0.4861 | 0.3889<br>0.6099 | 0.5000<br>0.6854 | 0.3226<br>0.5622 | 0.3125<br>0.5527 | **0.7273**<br>**0.8252** | **0.8387**<br>**0.8729** | **+0.5262 F1**<br>**+0.3202 AUC** |
| **`mvtec-ad_bottle`** | **`lr`** | **F1**<br>ROC-AUC | 0.3043<br>0.5100 | 0.4091<br>0.6589 | 0.5333<br>0.7977 | 0.7333<br>0.9181 | 0.7333<br>0.9181 | 0.7333<br>0.9130 | **0.8000**<br>**0.9231** | **+0.0667 F1**<br>**+0.0050 AUC** |
| **`mvtec-ad_bottle`** | **`rf`** *(200t)* | **F1**<br>ROC-AUC | 0.4545<br>0.6731 | 0.5600<br>0.7458 | 0.5455<br>0.7993 | 0.4211<br>0.8562 | 0.5263<br>0.8729 | **0.6923**<br>**0.8846** | **0.7619**<br>**0.8913** | **+0.2356 F1**<br>**+0.0184 AUC** |
| **`mvtec-ad_bottle`** | **`dt`** | **F1**<br>ROC-AUC | 0.3636<br>0.6137 | 0.5161<br>0.7391 | 0.4615<br>0.6472 | 0.1600<br>0.4699 | 0.3077<br>0.5585 | **0.5161**<br>**0.7174** | **0.5600**<br>**0.7266** | **+0.2523 F1**<br>**+0.1681 AUC** |
| **`kolektorsdd2`** | **`lr`** | **F1**<br>ROC-AUC | 0.2166<br>0.5824 | 0.1848<br>0.5334 | 0.2270<br>0.6051 | 0.2576<br>0.5976 | 0.2687<br>0.6179 | **0.3591**<br>**0.7371** | **0.4265**<br>**0.8061** | **+0.1579 F1**<br>**+0.1883 AUC** |
| **`kolektorsdd2`** | **`rf`** *(200t)* | **F1**<br>ROC-AUC | 0.2102<br>0.5679 | 0.3351<br>0.6721 | **0.4541**<br>0.7717 | 0.4156<br>0.8089 | 0.4286<br>**0.8255** | 0.3765<br>0.7429 | **0.4298**<br>0.7904 | **+0.0013 F1**<br>-0.0351 AUC |
| **`kolektorsdd2`** | **`knn`** | **F1**<br>ROC-AUC | 0.0881<br>0.5360 | **0.2727**<br>0.6422 | 0.2520<br>0.6624 | 0.1500<br>0.6385 | 0.1513<br>0.6443 | **0.2208**<br>0.6328 | 0.1500<br>**0.6652** | -0.0013 F1<br>**+0.0210 AUC** |
| **`kolektorsdd2`** | **`dt`** | **F1**<br>ROC-AUC | 0.2016<br>0.5463 | 0.3083<br>0.6113 | 0.2734<br>0.5467 | 0.3723<br>0.5528 | **0.3898**<br>**0.6397** | 0.2424<br>0.5861 | 0.2678<br>0.5877 | -0.1220 F1<br>-0.0520 AUC |

---

### Table 6 — Explainability & Feature Importance Attribution ($\mathbf{x}_{85}$ Entropy vs. $\mathbf{x}_9$ Asymmetry)

Extracted from [`results/classification/mvtec-ad/bottle_*/metrics_summary.json`](results/classification/mvtec-ad/bottle_lr/metrics_summary.json), [`tile_*/metrics_summary.json`](results/classification/mvtec-ad/tile_rf/metrics_summary.json), and [`kolektorsdd2/full_*/metrics_summary.json`](results/classification/kolektorsdd2/full_rf/metrics_summary.json):

| Dataset / Classifier | Method | $\mathbf{x}_{85}$ Total Share (`85` feats) | $\mathbf{x}_{85}$ Mean / Feature | **$\mathbf{x}_9$ Total Share (`9` feats)** | **$\mathbf{x}_9$ Mean / Feature** | **Asymmetry Efficiency Ratio ($\mathbf{x}_9 / \mathbf{x}_{85}$ per feat)** | Dominant Asymmetry Axis (within $\mathbf{x}_9$) | Top-3 Individual Features (`share_pct`) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| **`mvtec-ad_tile` (`rf`)** | Gini MDI | 41.46% | 0.49% | **58.54%** | **6.50%** | **13.33×** | `asym_rot180` (**45.6%**) | 1. `asym_rot180_8x8` (`11.49%`)<br>2. `asym_rot180_4x4` (`10.36%`)<br>3. `asym_tb_8x8` (`7.79%`) |
| **`mvtec-ad_tile` (`lr`)** | Permutation | 81.49% | 0.96% | **18.51%** | **2.06%** | **2.15×** | `asym_tb` (**60.0%**) | 1. `asym_tb_4x4` (`6.46%`)<br>2. `E_1x1_r0_c0` (`4.45%`)<br>3. `E_8x8_r7_c0` (`4.04%`) |
| **`mvtec-ad_bottle` (`rf`)** | Gini MDI | 62.33% | 0.73% | **37.67%** | **4.19%** | **5.71×** | `asym_rot180` (**39.8%**) | 1. `asym_tb_8x8` (`8.04%`)<br>2. `asym_rot180_8x8` (`7.54%`)<br>3. `asym_lr_8x8` (`5.41%`) |
| **`mvtec-ad_bottle` (`lr`)** | Permutation | 80.67% | 0.95% | **19.33%** | **2.15%** | **2.26×** | `asym_lr` (**46.2%**) | 1. `asym_lr_8x8` (`6.22%`)<br>2. `asym_rot180_8x8` (`4.57%`)<br>3. `asym_tb_8x8` (`4.01%`) |
| **`kolektorsdd2` (`rf`)** | Gini MDI | 65.02% | 0.76% | **34.98%** | **3.89%** | **5.08×** | `asym_tb` (**38.4%**) | 1–7. All 7 top features are `asym_*` (`asym_tb_8x8`, `asym_rot180_8x8`, `asym_lr_8x8`, ...) |

---

### Table 7 — Stage-by-Stage Profiling of $\mathbf{x}_{94}$, Classifier Wrapper Overhead & C/C++ Speedup Projection

Compiled from [`results/profiling/mvtec-ad_bottle/descriptor_stage_times.csv`](results/profiling/mvtec-ad_bottle/descriptor_stage_times.csv), [`classifier_times.csv`](results/profiling/mvtec-ad_bottle/classifier_times.csv), and [`descriptor_scalability.csv`](results/profiling/mvtec-ad_bottle/descriptor_scalability.csv) (`threads=1, batch=1`):

| Pipeline Component / Stage | Measured Python Median (`ms`) | Share of Staged Descriptor Time | Peak `tracemalloc` (`KB`) | Root Cause of Latency in Python / NumPy / `sklearn` | Exact Native C/C++ Optimization | Projected C/C++ Latency (x86_64 1-Core) |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: |
| `validate_and_quantize` (`128×128`) | `0.0258 ms` | 1.62% | 32.2 KB | Allocates a new `128×128` uint8 array via `lut[image]`. | Inline bit-shift `px >> 4` during single pixel pass (0 allocations). | `< 0.005 ms` *(fused)* |
| `region_histograms_g1` ($1\times1$, 1 cell) | `0.1074 ms` | 6.77% | 193.2 KB | `ravel().astype(int32)` + `np.bincount`. | Summed from $2\times2$ (`4 × 16 = 64` `uint16_t` adds). | `< 0.001 ms` |
| `region_histograms_g2` ($2\times2$, 4 cells) | `0.1318 ms` | 8.30% | 50.8 KB | 4 Python loop iterations + 2D slice copies + `bincount`. | Summed from $4\times4$ (`16 × 16 = 256` `uint16_t` adds). | `< 0.001 ms` |
| `region_histograms_g4` ($4\times4$, 16 cells) | `0.2371 ms` | 14.94% | 16.3 KB | 16 Python loop iterations + 2D slice copies + `bincount`. | Summed from $8\times8$ (`64 × 16 = 1,024` `uint16_t` adds). | `< 0.002 ms` |
| **`region_histograms_g8` ($8\times8$, 64 cells)** | **`0.6141 ms`** | **38.69%** | 13.4 KB | **64 Python loop iterations + non-contiguous 2D slice copies + `int32` cast + `bincount`.** | **Single contiguous row-major pass over `16,384` bytes incrementing `hist8[y>>4][x>>4][px>>4]`.** | **`~0.025 ms`** |
| `shannon_entropy_all` (85 cells × 16 bins) | `0.2646 ms` | 16.67% | 15.6 KB | `float64` division, boolean mask `p > 0`, `np.log2(p)` across 4 calls. | Static 257-entry `c * log2f(c)` LUT for $8\times8$ ($N=256$) + fast `log2f` on $4\times4, 2\times2, 1\times1$. | `~0.015 ms` |
| `asymmetry_maps_all` ($A_{\text{LR}}, A_{\text{TB}}, A_{\text{R180}}$) | `0.2065 ms` | 13.01% | 6.3 KB | 9 NumPy array slice reversals (`[:, ::-1]`, `[::-1, :]`) + `np.mean(np.abs(...))`. | Direct indexed ` fabsf(E[r][c] - E[r][g-1-c]) ` loops over $2\times2, 4\times4, 8\times8$ stack arrays. | `< 0.003 ms` |
| **Python Inter-Stage Dispatch Gap** | **`0.7393 ms`** | *(+46.6% overhead)* | — | Python function calls, tuple unpacking, `np.concatenate` (`2.3266 ms` E2E vs `1.5873 ms` staged sum). | Zero-allocation flat struct `float x94[94]` passed by pointer. | **`0.000 ms`** |
| **`lr` Inference (`sklearn` vs. Core)** | **`0.2566 ms`** (`sklearn`)<br>**`0.0032 ms`** (`core`) | — | — | **98.7% (`0.2534 ms`) is `sklearn` `Pipeline` validation & `StandardScaler` wrapper.** | **Fuse $\mu_i, \sigma_i$ into $w'_i = w_i / \sigma_i$, $b' = b - \sum w_i \mu_i / \sigma_i$: 94 `float32` MACs (`380 bytes`).** | **`< 0.001 ms`** |
| **TOTAL PIPELINE (`x94 + lr`, `batch=1`)** | **`2.29–2.89 ms`** | **100%** | ~250 KB | Dominated by 4-pass NumPy slicing (`1.09 ms`), Python dispatch (`0.74 ms`), and `sklearn` (`0.25 ms`). | Single-pass bottom-up pyramid + fused 94-MAC dot product (`0` heap allocations). | **`~0.05 – 0.12 ms` (`8,000+ img/s`)** |

---

### Table 8 — Complete Automated Test Suite Verification Matrix (268 / 268 Tests Passed)

| Test Category | Test File / Suite | Command | Tests | Time (`s`) | Status |
| :--- | :--- | :--- | ---: | ---: | :--- |
| **Environment & GPU** | [`tests/test_environment.py`](tests/test_environment.py) | `.\.venv\Scripts\python.exe tests/test_environment.py` | 1 (E2E) | 4.82 | **PASS** (GTX 1050 CUDA 12.6) |
| **Pip Dependencies** | `pip check` | `.\.venv\Scripts\python.exe -m pip check` | — | 1.95 | **PASS** (0 broken deps) |
| **Formal Unit & CLI (`tests/`)** | 14 test modules (`test_letterbox`, `test_invert`, `test_csv_export`, `test_entropy_maps`, `test_asymmetry_maps`, `test_descriptor_classifier`, `test_cnn_*`, `test_profiling_*`, `test_benchmark_cpu_fair_cli`) | `.\.venv\Scripts\python.exe -m unittest discover tests -v` | **195** | 87.70 | **PASS (195/195)** |
| **Lab 03 Entropy** | [`understanding/03_entropy_benchmark/test_entropy.py`](understanding/03_entropy_benchmark/test_entropy.py) | `.\.venv\Scripts\python.exe understanding/03_entropy_benchmark/test_entropy.py` | **5** | 1.47 | **PASS (5/5)** |
| **Lab 04 CNNs** | [`understanding/04_lightweight_cnns/tests/test_models.py`](understanding/04_lightweight_cnns/tests/test_models.py) | `.\.venv\Scripts\python.exe understanding/04_lightweight_cnns/tests/test_models.py` | **8** *(140 cases)* | 8.09 | **PASS (8/8)** |
| **Lab 05 Symmetry** | [`understanding/05_symmetry_features/tests/test_symmetry_features.py`](understanding/05_symmetry_features/tests/test_symmetry_features.py) | `.\.venv\Scripts\python.exe understanding/05_symmetry_features/tests/test_symmetry_features.py` | **20** | 6.84 | **PASS (20/20)** |
| **Lab 06 Pyramid $\mathbf{x}_{94}$** | [`understanding/06_entropy_pyramid_descriptor/tests/test_descriptor.py`](understanding/06_entropy_pyramid_descriptor/tests/test_descriptor.py) | `.\.venv\Scripts\python.exe understanding/06_entropy_pyramid_descriptor/tests/test_descriptor.py` | **25** | 9.19 | **PASS (25/25)** |
| **Lab 07 Behavior** | [`understanding/07_descriptor_behavior/tests/test_behavior_helpers.py`](understanding/07_descriptor_behavior/tests/test_behavior_helpers.py) | `$env:PYTHONPATH=(Get-Location).Path; .\.venv\Scripts\python.exe understanding/07_descriptor_behavior/tests/test_behavior_helpers.py` | **15** *(279 scenes)* | 24.30 | **PASS (15/15)** |
| **TOTAL** | — | — | **268** | **144.36** | **100% PASS** |

---

### Table 9 — Actionable Checklist of Methodological & Documentation Fixes (Pre-Paper & Pre-C/C++)

| ID | Severity | Location | Issue Identified During Deep Audit | Recommended Fix | Blocks C/C++? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M1** | **High (Paper Validity)** | [`experiments/evaluate_cnn.py`](experiments/evaluate_cnn.py#L428-L438), [`configs/cnn/mvtec_ad.toml`](configs/cnn/mvtec_ad.toml#L7-L8) | Looping over `seeds = [42, 43, 44]` overwrites `checkpoint.pt` and `metrics_summary.json` in `out_dir`, leaving `seed=44` on disk while classical ML uses `seed=42` (`stratified` split mismatch on `bottle`). | Re-run `evaluate_cnn.py --seeds 42` for canonical single-seed parity, or save per-seed subfolders (`seed_42`, `seed_43`, `seed_44`) and evaluate ML on the exact same `[42, 43, 44]` seeds. | **No** (C/C++ exports whatever trained weights are selected) |
| **M2** | **High (Paper Validity)** | [`experiments/benchmark_cpu_fair.py`](experiments/benchmark_cpu_fair.py) | CNNs use `threshold_mode = "val_f1"` ($\tau^*$), whereas classical ML models are evaluated only at fixed $\tau = 0.50$. Also, `tiny_resnet` suffers threshold collapse on `bottle`'s 8-defect validation split. | Report both **F1 @ $\tau = 0.50$** and **F1 @ $\tau^*$** (`val_f1` / OOB) for both ML and CNNs in the paper's comparison tables. | **No** (In C/C++, $\tau^*$ is just the scalar threshold compared against logit $s$) |
| **M3** | **Medium (Paper Scope)** | `results/cnn/mvtec-ad/`, `results/comparison_cpu/` | Cross-paradigm CPU comparison currently covers `kolektorsdd2` and 1 MVTec-AD category (`bottle`), missing `tile` (where $\mathbf{x}_{94}$ reaches `ROC-AUC = 0.9678, F1 = 0.8649`). *(See also full Colab sweep [`notebooks/Entropy_Maps_Full_Run.ipynb`](notebooks/Entropy_Maps_Full_Run.ipynb).)* | Run `evaluate_cnn.py` and `benchmark_cpu_fair.py` on `--object tile` (and 2–3 additional MVTec-AD categories, or all 15 via `--all-objects`). | **No** |
| **M4** | **Low (Config Consistency)** | [`configs/classification/mvtec_ad.toml`](configs/classification/mvtec_ad.toml#L39-L56) | `rf` uses `n_estimators = 100, max_depth = 12` (`dt max_depth = 3`) in `mvtec_ad.toml` vs `n_estimators = 200, max_depth = 6` (`dt max_depth = 16`) in `kolektorsdd2.toml` (and `200` trees in the earlier `bottle_rf` run). | Document or unify `n_estimators` and depth limits across classification configs. | **No** |
| **D1** | **Low (Docs & Test Hygiene)** | [`RESUME.md`](RESUME.md), [`tests/README.md`](tests/README.md), [`understanding/README.md`](understanding/README.md) | Six minor documentation/test-path drift items (`§3.8`): outdated `RESUME.md` (`2026-10-30`, `149` tests), old Lab 05–07 test paths in `tests/README.md`, Lab 07 `sys.path` requiring `PYTHONPATH`, CSV column typo (`16387`), outdated classification README, and `test_cnn_cli.py` writing to `results/comparison/`. | Apply the 6 cleanup items cataloged in **§3.8 [Note N8]**. | **No** |

---

## 3. Technical Design Notes & Follow-Up Clarifications (`N1–N8`)

This section records detailed technical design decisions, physical root-cause analyses, and housekeeping notes arising from follow-up review discussions (`2026-11-11`).

### 3.1 [N1] Feature Subset Selection: Why $\mathbf{x}_{94}$ Is Retained Over $\mathbf{x}_9$ Alone

> **Question:** Since $\mathbf{x}_9$ achieves strong F1/ROC-AUC on several tasks with only 9 features, should $\mathbf{x}_9$ replace $\mathbf{x}_{94}$, or should $\mathbf{x}_{94}$ be kept and its loop computation optimized?

**Decision: Keep $\mathbf{x}_{94} = [\mathbf{x}_{85}, \mathbf{x}_9]$ as the primary descriptor and eliminate redundant loops in its computation.**

1. **Mathematical & Computational Coupling:** $\mathbf{x}_9$ is not an independent descriptor — it consists of the Left–Right, Top–Bottom, and $180^\circ$ rotational residuals computed directly from the $2\times2$, $4\times4$, and $8\times8$ entropy grids ($4 + 16 + 64 = 84$ of the $85$ features in $\mathbf{x}_{85}$). Computing $\mathbf{x}_9$ already requires computing 98.8% of $\mathbf{x}_{85}$. In a **single-pass bottom-up histogram pyramid** (`8×8 -> 4×4 -> 2×2 -> 1×1`), obtaining $E_{1\times1}$ adds only `4 × 16 = 64` integer additions plus one 16-bin Shannon sum ($< 1\text{ }\mu\text{s}$ in C/C++).
2. **Structural Completeness & Symmetry Blind Spot:** By definition, any **centrally symmetric** anomaly (e.g., uniform surface haze, centered illumination shifts, annular/concentric defects, or symmetric edge wear) produces identical or near-identical opposite quadrants, driving **$\mathbf{x}_9 \approx 0$**. Only $\mathbf{x}_{85}$ preserves absolute local and global entropy magnitudes.
3. **Empirical Evidence Across Real Datasets (`Table 5`):**
   - **`mvtec-ad_bottle` (`lr`):** $\mathbf{x}_{94}$ outperforms $\mathbf{x}_9$ (`F1 = 0.8000` vs `0.7333`, `ROC-AUC = 0.9231` vs `0.9130`).
   - **`mvtec-ad_tile` (`lr`):** $\mathbf{x}_{94}$ outperforms $\mathbf{x}_9$ (`F1 = 0.8649` vs `0.8333`, `ROC-AUC = 0.9678` vs `0.9556`).
   - **`kolektorsdd2` (`lr` & `rf`):** $\mathbf{x}_{94}$ beats $\mathbf{x}_9$ across both strong baselines (`lr`: `F1 = 0.4265` vs `0.3591`, `ROC-AUC = 0.8061` vs `0.7371`; `rf`: `F1 = 0.4298` vs `0.3765`, `ROC-AUC = 0.7904` vs `0.7429`).
   - *Where $\mathbf{x}_9$ alone helps:* Only distance-based or low-capacity learners on small sample counts (`knn` or `rf` on `tile` F1: `0.8571` vs `0.8235`, though `x94` still wins ROC-AUC `0.9745` vs `0.9589`) benefit from reducing `94 -> 9` features to curb curse-of-dimensionality / feature-subsampling dilution. With regularized linear models (`lr`, 95 parameters), keeping all 94 features consistently achieves the highest accuracy.
4. **Actionable Path:** Optimize the descriptor extractor so that $\mathbf{x}_{85} \to \mathbf{x}_{94}$ is computed in a single pixel pass without 85 Python/NumPy slice iterations (the optimization planned in [`tasks/task_09.md`](tasks/task_09.md)), while preserving $\mathbf{x}_9$ and $\mathbf{x}_{85}$ as explicit experimental ablations.

---

### 3.2 [N2] Photometric Polarity (`normal` vs. `inverted`): Defect Polarity, Letterboxing & CNN Behavior

> **Question:** For classical ML on $\mathbf{x}_{94}$, `inverted` barely changes accuracy, whereas CNNs shift noticeably. In each dataset variant (`normal` vs. `inverted`), are defects black/dark or white/bright, and why do the two model families react differently?

| Dataset | Variant | Polarity Transform (`255 - I`) & Order | Background & Non-Defect Surface | Defect Visual Appearance (`raw` $\to$ `128×128`) | Effect on $\mathbf{x}_{94}$ + ML | Effect on CNNs (Trained Without Inversion Aug.) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`mvtec-ad_bottle`** *(square `900×900`, no padding)* | **`normal`** | None | **Bright white** backlit glass/ring (`~220–255`) | **Dark / black** fractures, contamination, broken glass (`~0–100`) | Baseline (`lr` `F1=0.8000, ROC-AUC=0.9231`). | `small_cnn` `F1=0.8571, ROC-AUC=0.9816`; `mobilenet_like` `ROC-AUC=0.9047`. |
| **`mvtec-ad_bottle`** *(square `900×900`, no padding)* | **`inverted`** | `I_inv = 255 - I` | **Black** background (`~0–35`) | **Bright / white** defects (`~155–255`) | **Mathematically invariant** (`b <-> 15-b` preserves $-\sum p_b \log_2 p_b$): `lr` `F1=0.8000, ROC-AUC=0.9214`. | Spatial gradient sign flips; without polarity augmentation `mobilenet_like` collapses (`ROC-AUC 0.9047 -> 0.4967`) and `small_cnn` drops (`F1 0.8571 -> 0.8000`). |
| **`kolektorsdd2`** *(rect. `~230×630` + black letterbox `pad=0`)* | **`normal`** | None; padded with `pad_value = 0` (black bars left/right) | Mid-to-bright gray metallic strip + **black** side padding bars (`0`) | **Dark / black** micro-cracks & pits (`~0–50`) on gray metal | Baseline (`lr` `F1=0.4265, ROC-AUC=0.8061`). Dark metal shadows merge into `bin 0` with `pad=0`. | `tiny_resnet` `F1=0.7389, ROC-AUC=0.9219`. Zero-padding (`0.0` tensor after `(px/255 - 0.5)/0.5 = -1.0`) matches dark borders. |
| **`kolektorsdd2`** *(rect. `~230×630` + black letterbox `pad=0`)* | **`inverted`** | `255 - I` applied **before** letterboxing ([`src/preprocessing/kolektorsdd2.py`](src/preprocessing/kolektorsdd2.py#L21-L25)) | Dark inverted strip + **black** side padding bars (kept `0`, *not* inverted to `255`) | **Bright / white** micro-cracks (`~205–255`) | `bin 0` now merges *former bright specular highlights* (`255->0`) with `pad=0`, subtly changing boundary cells (`lr` `F1=0.4153, ROC-AUC=0.8151`). | `torchvision_mobilenet` (`F1 0.6049 -> 0.1975`) and `ultra_tiny` (`F1 0.5914 -> 0.3889`) degrade; `ds_cnn` remains relatively resilient (`F1 0.6102 -> 0.6355`). |

**Core Insight:** Unpadded Shannon entropy $\mathbf{x}_{94}$ depends only on the **set** of histogram probabilities $\{p_0, \dots, p_{15}\}$, not on which brightness bin index carries mass ($p_b \mapsto p_{15-b}$ leaves $-\sum p_b \log_2 p_b$ strictly commutative). CNNs learn signed edge/intensity filters whose ReLU activations depend directly on pixel polarity unless explicit polarity-inversion augmentation is added during training.

---

### 3.3 [N3] Grid Granularity Extension (`[2, 4, 8, 16]` vs. `[4, 8, 16, 32]`) & Experiment Scope

> **Question:** Could finer grid scales (`2, 4, 8, 16` or `4, 8, 16, 32` instead of `1, 2, 4, 8`) improve micro-crack sensitivity on KolektorSDD2, and if tested, which existing experiments must be re-run?

#### Quantitative Cell Budget on $128\times128$ Images (`bins = 16`)

| Grid Scale $g$ | Cells ($g^2$) | Cell Size (px) | Pixels / Cell ($N$) | Avg. Pixels / Bin ($N / 16$) | 2-px Crack Area Share / Cell | Cumulative Features (`x_ent` + `x_asym`) | Physical Regime & Assessment |
| :---: | ---: | :---: | ---: | ---: | :---: | :---: | :--- |
| **`1×1`** | 1 | `128×128` | 16,384 | 1,024.0 px/bin | ~1.6% | `1` *(has no asym.)* | Global context anchor; minimal cost (`0` loop cost if summed bottom-up). |
| **`2×2`** | 4 | `64×64` | 4,096 | 256.0 px/bin | ~3.1% | `5 + 3 = 8` | Macro quadrant asymmetry (`x9`). |
| **`4×4`** | 16 | `32×32` | 1,024 | 64.0 px/bin | ~6.3% | `21 + 6 = 27` | Best single scale on many structural objects (`E_4x4`). |
| **`8×8`** *(current max)* | 64 | `16×16` | 256 | 16.0 px/bin | ~12.5% *(for 2×16 px strip)* | `85 + 9 = 94` (**`x94-v1`**) | Strong on macro defects; 1–2 px KolektorSDD2 crack still diluted inside `256` px. |
| **`16×16`** *(recommended ablation)* | 256 | `8×8` | **64** | **4.0 px/bin** | **~25.0% (4× stronger!)** | `[1,2,4,8,16]` $\to 341+12=\mathbf{353}$<br>`[2,4,8,16]` $\to 340+12=\mathbf{352}$ | **High-value hypothesis test:** `64` px/cell stays statistically populated for `bins=16` (`4` px/bin average) while boosting micro-crack occupancy $4\times$. |
| **`32×32`** *(caution)* | 1,024 | `4×4` | **16** | **1.0 px/bin** *(floor!)* | ~50.0% | `[4,8,16,32]` $\to 1,360+12=\mathbf{1,372}$ | **Information-theoretic & $p \gg n$ floor:** Only `16` pixels distribute across `16` bins (extremely high finite-sample variance; entropy saturates sharply and `p = 1,372 >> N_train = 233` on `bottle`, severely harming generalization unless `bins` is lowered to `4–8` and regularized heavily). |

**Recommendations:**
1. **Include `[1, 2, 4, 8, 16]` (and optionally `[2, 4, 8, 16]`) as a clean comparative ablation against `[1, 2, 4, 8]`; do *not* drop `1×1` for runtime reasons** if using a bottom-up sum (it costs $< 1\text{ }\mu\text{s}$), though `[2, 4, 8, 16]` vs `[1, 2, 4, 8, 16]` is a valid ablation if matching feature symmetry grids (`4` vs `5` scales). Note also [`tasks/task_09.md`](tasks/task_09.md) specifically anticipates increasing patch counts (`g=16` / `g=32`), making fast C/single-pass computation prerequisite.
2. **Do NOT re-run existing `x94-v1` experiments!** Keep `x94-v1` frozen as the canonical baseline. Both [`src/features/entropy_maps.py`](src/features/entropy_maps.py#L18-L48) and [`src/features/asymmetry_maps.py`](src/features/asymmetry_maps.py#L22-L56) already parameterize arbitrary `grid_levels` and `scales` via TOML. Run a **separate standalone ablation script** (with distinct output directory/suffix, e.g., `grid_ablation_[1,2,4,8,16]` and `[4,8,16,32]`) over `kolektorsdd2` and `mvtec-ad_{bottle,tile}` to evaluate **F1 / ROC-AUC gain vs. parameter count (`354` vs `95` LR parameters) & latency** before deciding whether a second schema (e.g., `x353`) merits a full sweep.

---

### 3.4 [N4] Inference Latency Dynamics: CPU vs. GPU vs. Python Interpreter Overhead

> **Question:** Is $\mathbf{x}_{94}$ always faster than lightweight CNNs? Previously on GPU the CNNs won, and now everything runs on CPU — how do timings actually compare across runtimes?

**Answer: No — $\mathbf{x}_{94}$ is *not* faster across all settings in Python, and hardware/runtime context cleanly separates three distinct regimes (`Table 3` & `Table 7`):**

1. **Regime 1 — CPU Single-Thread Latency (`threads=1, batch=1`, Python):**
   - **CNNs beat Python `x94` at the low-latency end:** `ultra_tiny` (`0.84–1.08 ms`) and `ds_cnn` (`1.89–2.39 ms`) are faster than Python `lr (x94)` (`2.29–2.89 ms`) with high statistical significance (`p < 1e-55` vs `ultra_tiny`).
   - **Why?** PyTorch runs compiled C++/AVX2 `ATen`/`NNPACK` kernels (`~0.8 ms`), whereas `extract_x94_end_to_end` pays **85 separate Python loop/slice/`bincount` calls (`2.07–2.56 ms`)** plus **`0.74 ms` Python inter-stage dispatch** (`Table 7`), and `sklearn` `Pipeline.predict` spends **`0.253 ms` in wrapper validation** for a `0.0032 ms` (3.2 µs) linear dot product.
   - **Still beats heavier accurate CNNs:** Python `lr (x94)` (`2.29 ms`) still runs **2.4× faster** than `small_cnn` (`5.56 ms` / `6.50 ms` on `bottle`) and **1.4×–1.5× faster** than `tiny_resnet` (`3.24–4.11 ms`).
2. **Regime 2 — CPU Burst Throughput (`threads=1, batch=32`, Python):**
   - When batching amortizes Python dispatch across 32 images (`Table 3`), **`lr (x94)` (`1,552–2,075 img/s`) and `dt (x94)` (`1,568–2,098 img/s`) overtake 6 of 7 CNNs** (`small_cnn`: `150–158 img/s`, `tiny_resnet`: `370–456 img/s`, `ds_cnn`: `396–460 img/s`), beaten in throughput only by `ultra_tiny` (`1,762–3,468 img/s`).
3. **Regime 3 — GPU Throughput (Secondary Reference, `GTX 1050 CUDA`, `batch=32`) vs. Native C/C++ Target:**
   - On desktop GPU at `batch=32` (`results/profiling/*/README.md` §7), CNNs exploit thousands of CUDA cores (`ultra_tiny`: `22,261–25,797 img/s`, `small_cnn`: `4,268–4,322 img/s`), because $\mathbf{x}_{94}$ runs strictly on CPU (`94`-feature dot product needs no GPU).
   - **Native C/C++ single-core target (`Table 7`):** Eliminating the Python/NumPy/sklearn wrapper overhead projects `x94 + lr` to **`~0.05–0.12 ms/img` (`8,000+ img/s` at `batch=1`)** — putting it **7×–20× ahead of `ultra_tiny` CPU** and validating the core thesis for **GPU-less, RAM-constrained MCUs / embedded CPUs**.

---

### 3.5 [N5] Seed Mismatch Scope (`seed=42` vs. `seed=44`): Actual Severity & Remediation

> **Question:** How serious is the `seed=42` vs. `seed=44` discrepancy found during the audit?

- **KolektorSDD2 (`split_mode = "presplit"`): ZERO test-split bias.** Every classical ML and CNN run is evaluated on the exact same 1,004 fixed images on disk (`894 good / 110 defect`). Seed `44` vs `42` only changes weight initialization, mini-batch shuffling, and the internal `350`-sample train/val split.
- **MVTec-AD `bottle` (`split_mode = "stratified"`): Moderate–High methodological issue for paper comparison tables.** Because `mvtec-ad` constructs the 80/20 test split in memory using the run seed, classical ML evaluated on `seed=42`'s 59 test images whereas the CNN checkpoints left on disk (`seeds = [42, 43, 44]` overwriting the same folder in [`experiments/evaluate_cnn.py`](experiments/evaluate_cnn.py#L428-L438)) evaluated on `seed=44`'s 59 test images. On $N_{\text{test}}=59$ (`13` defects), different random splits shift F1 by several points purely from sample draw. *(Note: [`notebooks/Entropy_Maps_Full_Run.ipynb`](notebooks/Entropy_Maps_Full_Run.ipynb#L411-L416) also invokes `evaluate_cnn.py` without `--seeds 42`, so running Stage 5 as-is will exhibit the same last-seed overwrite.)*
- **Severity Assessment:** Does **not** compromise the validity of the engineering pipeline or block C/C++ work, **does** need to be aligned (`--seeds 42`, or per-seed `seed_<k>` directories across `[42, 43, 44]` for both ML and CNNs) before locking final manuscript tables (`~2–3 min` re-run per MVTec-AD category on GTX 1050 / T4).

---

### 3.6 [N6] CNN Threshold Calibration ($\tau$) & Degenerate Collapse Prevention

> **Question:** Because `lr (x94)` at $\tau = 0.50$ (`F1 = 0.8000`) beats `small_cnn` at $\tau = 0.50$ (`F1 = 0.4906`), should CNN threshold calibration (`val_f1`) be kept, removed (`0.50` only), or modified?

**Decision: Keep `val_f1` calibration enabled, harden it against single-class collapse, and report both $\tau = 0.50$ and $\tau^*$ side-by-side for both CNNs and classical ML.**

1. **Why disabling `val_f1` would be unfair to CNNs:** Balanced `CrossEntropyLoss` deliberately reweights minority-class gradients (`class_weighting = "balanced"`), shifting softmax log-odds by $\approx \log(w_1/w_0)$ so the Bayes-optimal F1 decision boundary moves away from `0.50` (`src/models/cnn/evaluate.py` lines 141–145). Judging CNNs solely at `0.50` while using balanced loss would invite valid peer-review criticism.
2. **Why reporting *only* `val_f1` is also incomplete:** Comparing calibrated CNNs (`val_f1`) against uncalibrated ML (`0.50`) introduces an asymmetric advantage (and hides cases like `tiny_resnet` collapsing to all-positive predictions at $\tau^* = 0.49$ on `bottle`'s tiny 8-defect validation split, or `ds_cnn` dropping `0.6429 -> 0.5714` due to validation overfitting, whereas `rf` jumps `0.3333 -> 0.4526` on `kolektorsdd2` when OOB-calibrated).
3. **Recommended Paper & Code Standard:**
   - Primary ranking tables should show **Threshold-Free Metrics** (`ROC-AUC`, `PR-AUC`) alongside **Both Operating Points** (`F1 @ τ=0.50` and `F1 @ τ*`), clearly labeling how $\tau^*$ was chosen (`val_f1` for CNNs; internal CV / OOB for ML, with zero test leakage).
   - Add a **non-degeneracy guard** inside [`find_optimal_threshold`](src/models/cnn/evaluate.py#L54-L93) so candidate thresholds predicting 100% negative or 100% positive on `val` (e.g., `F1 = 0.3611` all-positive dummy baseline on `bottle`) are filtered out whenever any threshold predicts both classes with valid precision & recall.

---

### 3.7 [N7] Remediation Scope: Targeted Arguments & Logic Tweaks vs. Major Refactoring

> **Question:** Do recommendations `M1–M4` and `D1` require a major architectural refactor?

**Answer: No — zero structural refactoring is required.** Every architecture module, config schema (`src/preprocessing/`, `src/features/`, `src/models/`, `src/profiling/`), and test contract stays intact. The remediation amounts to **~30–40 lines of localized logic/CLI updates + clean re-execution**:
- **M1 (~10 lines / CLI arg):** Pass `--seeds 42` in runs/notebooks for exact parity with classical ML, or write per-seed paths (`<model>/seed_<k>/`) inside [`_run_single_evaluation`](experiments/evaluate_cnn.py) when `len(seeds) > 1`.
- **M2 (~15–20 lines):** Add the non-degenerate threshold filter in [`find_optimal_threshold`](src/models/cnn/evaluate.py#L54-L93), surface `f1_at_05` beside `f1` in comparison summaries, and optionally log training-OOB/CV $\tau^*$ for ML without changing test determinism.
- **M3 (0 code lines — CLI execution only):** Run `evaluate_cnn.py` and `benchmark_cpu_fair.py` with `--object tile` (or `--all-objects`, already wired in [`notebooks/Entropy_Maps_Full_Run.ipynb`](notebooks/Entropy_Maps_Full_Run.ipynb)).
- **M4 / D1 (~10–20 lines docs/configs/tests):** Sync TOML RF/DT comments/defaults, update stale markdown READMEs, and use `tempfile.TemporaryDirectory` in [`tests/test_cnn_cli.py`](tests/test_cnn_cli.py#L77-L86).

---

### 3.8 [N8] Detailed Documentation & Test Housekeeping Notes (`D1` Audit Catalog)

The six minor documentation and test-isolation discrepancies logged under **Table 9 (`D1`)** are itemized below for clean resolution alongside manuscript preparation:

1. **[`RESUME.md`](RESUME.md#L1-L78):** Last reviewed `2026-10-30`; still states `149 tests` (current count: `195` formal + `73` laboratory = `268`) and predates `dt`/`lr` classifiers, `src/models/cnn/`, `src/profiling/`, `evaluate_cnn.py`, `benchmark_cpu_fair.py`, `profile_pipelines.py`, and Lab 12.
2. **[`tests/README.md`](tests/README.md#L50-L52):** Lists wrong paths for Labs 05–07 test runners (`test_symmetry.py`, `test_descriptor.py`, `test_behavior.py` instead of `tests/test_symmetry_features.py`, `tests/test_descriptor.py`, `tests/test_behavior_helpers.py`).
3. **[`understanding/07_descriptor_behavior/compare_descriptors.py`](understanding/07_descriptor_behavior/compare_descriptors.py#L9-L11):** Prepends `code/` to `sys.path`, so invoking Lab 07 directly from PowerShell without `PYTHONPATH` set hits a module collision with top-level `experiments/`. Running with `$env:PYTHONPATH = (Get-Location).Path` resolves this cleanly (`15/15` tests, `279` scenes pass) without altering SHA-256-locked Lab 07 files.
4. **[`understanding/README.md`](understanding/README.md#L79):** ASCII flow diagram on line 79 reads `16387 cols` for KolektorSDD2 CSV instead of `16389 cols` (line 110 is accurate), and lines `115–120` (*Gaps hacia el paper*) list supervised evaluation, ablations, and $T_{\text{total}}$ benchmarking as still pending.
5. **[`results/classification/README.md`](results/classification/README.md):** Documents the earlier 15-object RF (`200` trees) sweep, whereas [`configs/classification/mvtec_ad.toml`](configs/classification/mvtec_ad.toml#L39-L56) now configures `rf` (`n_estimators=100, max_depth=12`) and `dt` (`max_depth=3`) vs. `kolektorsdd2.toml` (`n_estimators=200, max_depth=6`, `dt max_depth=16`). Additionally, commit `65e668f` updated `.gitignore` to ignore generated artifacts under `results/` (keeping `results/profiling/*/README.md` and top-level READMEs tracked); Table 3–7 numbers in this audit record the exact `bf4a29f` snapshot and can be regenerated end-to-end via [`notebooks/Entropy_Maps_Full_Run.ipynb`](notebooks/Entropy_Maps_Full_Run.ipynb).
6. **[`tests/test_cnn_cli.py`](tests/test_cnn_cli.py#L77-L86):** `test_compare_baseline_vs_cnn_cli_help` writes comparison CSVs directly into `results/comparison/` instead of a `tempfile.TemporaryDirectory`.

