# RESUME.md — project state for any agent

- **Last reviewed:** 2026-11-13
- **Provisional project name:** *Entropy-Aware Region Selection for Efficient Embedded Vision* (`2026-10-Symmetry-Entropy`)
- **Repository root:** `code/` (`D:\Publications\2026-10-Symmetry-Entropy\code`)

Read [AGENTS.md](AGENTS.md) first: it holds the binding operating rules (Windows 10, PowerShell, Python 3.11, venv at `.venv`, PyTorch `cu126`, no global installs, no TensorFlow, TOML-only configuration, scientific-integrity and performance-fairness rules). Then read [README.md](README.md) (environment setup, verification, and full end-to-end reproduction commands), [HYPOTHESIS.md](HYPOTHESIS.md) (current hypothesis verdicts and pre-MCU Python vs. C99 vs. CNNs synthesis), and [understanding/README.md](understanding/README.md) (pedagogical lab map).

---

## 1. Research goal

Classify defects/errors in industrial images using compact hand-crafted entropy and spatial-asymmetry descriptors ($\mathbf{x}_{94}$) as a lightweight alternative (or complement) to CNNs, aiming toward efficient region selection and embedded execution on microcontrollers (MCUs).

Core claim evaluated: describing images / selecting informative regions via multiscale Shannon entropy + spatial asymmetry can match or beat lightweight CNNs on structural defects at orders-of-magnitude lower parameter count and CPU latency, while quantifying where spatial-resolution limits require finer grids or CNNs.

---

## 2. Current state in one paragraph

Fourteen pedagogical laboratories exist under `understanding/` (`01`–`14`, plus `07_entropy_maps_vectorized/`). Labs `01`–`04` cover primitives (patch partitioning, histograms, Shannon entropy benchmarks, and 7 lightweight CNNs). Labs `05`–`07` define and stress-test the canonical 94-feature descriptor `image -> x94` (`schema x94-v1`: 85 multiscale entropies + 9 spatial asymmetry residuals across 279 synthetic scenes). Labs `08`–`12` validate dataset standardization, pixel CSV round-trips, spatial visualizations, and stage/layer CPU bottlenecks, Lab `13` proves single-pass bottom-up histogram summation and Shannon LUTs in pure NumPy (`9.40x`) and heap-free C99 (`92.65x`, `0.024 ms` for `x85`), and Lab `14` benchmarks MCU-portable folded C99 classifiers (`mlp_16_dot`, `mlp_16_skip_dot`, `mlp_32_16_dot`, `lr_l1_dot`, `lda_shrink_dot`, `lsvc_dot`, `gnb_quad`, `dt_core`, `gbdt_tiny_core`) against unmodified `src/` baselines (`lr_dot`, `dt`, `rf`). Formal production modules live in `src/` (`preprocessing/`, `datasets/`, `features/` with 5 entropy backends including `c_single_pass` + `numpy_finest_pyramid`, `models/` with `dt`, `knn`, `lr`, algebraically folded `lr_dot`, `rf`, ablation, explainability, and 7 CNNs, and `profiling/`). CLI entry points in `experiments/` (plus `notebooks/Entropy_Maps_Full_Run.ipynb` for Colab runs) execute dataset preprocessing, CSV/feature export, `seed=42` aligned ML/CNN evaluation on `kolektorsdd2` and `mvtec-ad_bottle` (normal + inverted), 100% CPU-fair comparisons (`results/comparison_cpu/`), isolated pure-Python CPU comparisons (`results/comparison_cpu_python/`), and deep statistical profiling (`results/profiling/`). Automated verification comprises **300 unit/integration tests + real CUDA environment check** (`208` formal tests across 22 suites in `tests/` + `92` lab tests in `understanding/`).

---

## 3. Repository layout (relative to `code/`)

```text
code/
├── AGENTS.md                        binding engineering & scientific rules (read first)
├── README.md                        environment, CUDA check, end-to-end reproduction CLI & replay guide
├── HYPOTHESIS.md                    verified hypothesis verdicts & Python vs. C99 pre-MCU synthesis
├── RESUME.md                        this file (project architecture & state summary)
├── requirements.txt                 pinned Python dependencies (install cu126 torch/torchvision first)
├── .gitignore                       excludes data/, heavy results/*, .venv/, native build artifacts, memory/
├── configs/                         declarative TOML configs (no hardcoded defaults in src/)
│   ├── base.toml                    dataset roots, object lists, splits, CSV paths
│   ├── logging.py                   centralized rotating logging setup
│   ├── features/native_entropy.toml MSVC C99 DLL build flags & paths
│   ├── optimization/entropy_maps.toml Lab 13 benchmark & sweep configuration
│   ├── understanding/ml_classifiers_mcu.toml Lab 14 exploratory MCU classifiers benchmark config
│   ├── preprocessing/               mvtec_ad.toml, kolektorsdd2.toml
│   ├── csv/                         mvtec_ad.toml (16391 cols), kolektorsdd2.toml (16389 cols)
│   ├── entropy/                     mvtec_ad.toml, kolektorsdd2.toml (x85 + backend selection)
│   ├── asymmetry/                   mvtec_ad.toml, kolektorsdd2.toml (x85 -> x94)
│   ├── classification/              mvtec_ad.toml, kolektorsdd2.toml (dt, knn, lr, lr_dot, rf)
│   ├── cnn/                         mvtec_ad.toml, kolektorsdd2.toml, colab.toml (7 CNNs, AdamW, val_f1)
│   └── profiling/                   mvtec_ad.toml, kolektorsdd2.toml (100% CPU-fair & deep profiling)
├── src/                             formal reusable pipeline modules
│   ├── preprocessing/               letterbox.py, mvtec_ad.py, kolektorsdd2.py
│   ├── datasets/                    mvtec_ad_csv.py, kolektorsdd2_csv.py
│   ├── features/
│   │   ├── entropy_maps.py          5 backends (auto, c_single_pass, numpy_finest_pyramid, numpy_fused_entropy, numpy_baseline)
│   │   ├── native_entropy.py        ctypes wrapper & zero-alloc NativeEntropyPlan
│   │   ├── build_native.py          MSVC C99 DLL builder -> src/features/native/build/entropy_pyramid.dll
│   │   ├── native/                  entropy_pyramid.h, entropy_pyramid.c (pure C99, stack-only on B=1)
│   │   ├── entropy_csv.py           streaming CSV extractor for x85 entropy maps
│   │   ├── asymmetry_maps.py        2D reconstruction & 9 spatial asymmetry residuals (LR, TB, R180)
│   │   └── asymmetry_csv.py         streaming CSV appender (x85 -> x94, schema x94-v1)
│   ├── models/
│   │   ├── descriptor_classifier.py dt, knn, lr, LinearDotCoreClassifier (lr_dot), rf, ablation, explainability
│   │   ├── classification_plots.py  CM+ROC/PR, ablation, importance breakdown, 2D heatmaps
│   │   └── cnn/                     architectures/ (7 CNNs + common/registry), config.py, data.py, train.py, evaluate.py, benchmark.py, report.py
│   └── profiling/                   config.py, timing.py, environment.py, descriptor_stages.py, classifier_profile.py, cnn_layers.py, report.py
├── experiments/                     CLI entry points (`python -m experiments.<name>`)
│   ├── preprocess_{mvtec_ad,kolektorsdd2}.py
│   ├── export_{mvtec_ad,kolektorsdd2}_csv.py
│   ├── export_entropy_maps.py
│   ├── append_asymmetry_features.py
│   ├── evaluate_descriptors.py
│   ├── evaluate_cnn.py
│   ├── benchmark_cpu_fair.py        Phase A: 100% CPU-fair ML vs. CNN benchmark (--entropy-backend)
│   ├── profile_pipelines.py         Phase B: stage/layer/ATen/Mann-Whitney U & ablation profiling
│   └── compare_baseline_vs_cnn.py   legacy mixed-device generator (deprecated for CPU latency claims)
├── notebooks/
│   └── Entropy_Maps_Full_Run.ipynb  full Colab execution notebook (from preprocessed datasets)
├── tests/                           23 test files: test_environment.py (CUDA) + 22 unittest modules (208 cases)
│   ├── test_{preprocessing,kolektorsdd2,csv_export,kolektorsdd2_csv_export}.py
│   ├── test_{entropy_maps,native_entropy,entropy_csv,asymmetry_maps,asymmetry_csv,descriptor_classifier}.py
│   ├── test_cnn_{architectures,config,data,train,evaluate,cli}.py
│   └── test_profiling_{config,timing,descriptor_stages,classifier,cnn_layers}.py, test_benchmark_cpu_fair_cli.py
├── data/                            raw/, preprocessed/, csv/ (datasets gitignored; .gitkeep & README.md tracked)
├── results/                         environment_debug.md, classification/, cnn/, comparison_cpu/, profiling/, comparison/ (deprecated)
├── memory/                          NotebookLM README exporter (export_readmes.ps1, run_canonical_experiments.ps1, archive_and_reset.ps1)
├── tasks/                           historical planning specifications (task_01..task_10, review_01)
└── understanding/                   14 pedagogical laboratories (`01`..`14`) + `07_entropy_maps_vectorized/`
```

---

## 4. Dependency graph (imports)

```text
Understanding laboratories:
  01_patch_partition  --data-->  02_histogram_benchmark, 03_entropy_benchmark
  03_entropy_benchmark.e1  --->  05_symmetry_features, 06_entropy_pyramid_descriptor
  05_symmetry_features     --->  06_entropy_pyramid_descriptor
  06_entropy_pyramid       --->  07_descriptor_behavior (via importlib)
  04_lightweight_cnns      (standalone; shares 7 architectures design with src/models/cnn)
  08..11                   (QA verification scripts over data/preprocessed/ and data/csv/)
  12_profiling_stages      --->  src.features, src.models.cnn
  13_entropy_maps_opt      --->  src.features, src.models
  14_ml_classifiers_mcu    --->  src.models.descriptor_classifier, src.profiling, src.features.build_native (read-only over src/)

Formal pipeline (`src/` & `experiments/`):
  src.preprocessing.letterbox             --->  src.preprocessing.{mvtec_ad, kolektorsdd2}
  src.datasets.mvtec_ad_csv               --->  src.datasets.kolektorsdd2_csv
  src.features.native_entropy             --->  src.features.{entropy_maps, entropy_csv, build_native}
  src.features.asymmetry_maps             --->  src.features.asymmetry_csv, src.models.descriptor_classifier
  src.models.descriptor_classifier       <--->  src.models.classification_plots
  src.models.descriptor_classifier        --->  src.models.cnn.{evaluate, report}
  src.features.* + src.models.*           --->  src.profiling.{config, descriptor_stages, classifier_profile, cnn_layers}
  configs.logging + src.*                 --->  experiments/*
```

Cross-lab imports in `understanding/` use `importlib.import_module("understanding.<name>...")` after inserting `code/` into `sys.path`.

---

## 5. Canonical descriptor (`x94-v1`)

Frozen reference specification in `understanding/06_entropy_pyramid_descriptor/feature_names.py`; vectorized & C99 production implementation in `src/features/`:

```text
input: uint8 grayscale image (Pillow mode 'L'), width,height >= 8 (128x128 in formal pipeline)
per level g in (1,2,4,8): partition into g*g half-open boxes (floor division)
per cell: counts[b] = #{pixels with pixel*B//256 == b}, B in [2,256], default 16
          p = counts/N,  H = -sum(p*log2(p)),  E = H / log2(B) in [0,1] (no clipping)
feature 0:      E of 1x1 (global)
features 1-84:  E of 2x2 (4), 4x4 (16), 8x8 (64), row-major
features 85-93: A_T(g) = sum |E[i,j] - T(E)[i,j]| / g^2  for T in {LR, TB, R180}, g in {2,4,8}
```

---

## 6. What each understanding lab established (evidence, not claims)

* **01** — Patch geometry works: coordinates, three edge policies (`DROP`/`PARTIAL`/`PAD`), pixel-exact reconstruction.
* **02** — Exact 256-bin histogram is $\Omega(N)$; direct-address indexing is optimal ($\Theta(N+256)$).
* **03** — Shannon reference `e1` + `e2` + LUTs (`e3`/`e4`) + sparse `e5`. Equivalence $\le 10^{-12}$; Q30 preserves top-k ranking.
* **04** — Seven lightweight CNNs (`ultra_tiny` to `torchvision_mobilenet`; 1.15k–1.52M params) and isolated CPU vs. GTX 1050 benchmarks.
* **05** — Nine spatial asymmetry residuals defined and bounded ($A \in [0, 1]$, triangle inequalities); same-histogram/different-structure collisions yield $A = 0$.
* **06** — Canonical extractor + `x94-v1` schema, SHA-256 pixel tracking, 25 tests.
* **07** — Controlled behavior study (`15` experiments, `279` rows): defect response localizes to affected cells, additive noise inflates $E$ and flattens $A$, intra-cell collisions produce exact identical $\mathbf{x}_{94}$, peak sensitivity occurs at cell boundaries (`07_entropy_maps_vectorized/` also visualizes vectorized $\mathbf{x}_{85}$).
* **08** — MVTec-AD QA: `128x128` `'L'`, `{0, 255}` mask preservation via `NEAREST`, visual matrix.
* **09** — MVTec-AD CSV QA: `16,391` columns, `6,612` rows, deterministic row-major order, bit-exact round-trip (`diff = 0`).
* **10** — KolektorSDD2 QA: binary labeling via native-resolution mask (`mask.max() > 0`), exact counts (`2,085/246` train, `894/110` test), zero masks in output.
* **11** — KolektorSDD2 CSV QA: `16,389` columns, `3,335` rows, bit-exact round-trip (`diff = 0`).
* **12** — Stage-by-stage CPU dissection: reveals Python 85-region slice/`bincount` loop, inter-stage dispatch gap, and `sklearn` wrapper overhead vs. PyTorch `ATen`.
* **13** — Single-pass bottom-up histogram pyramid + Shannon LUTs: `9.40x–10.49x` speedup in pure NumPy (`V1/V2`) and **`92.65x` in C99 (`V3`, `0.024 ms` for `x85`)** with $\le 1.89\times 10^{-15}$ parity (`13` tests).
* **14** — MCU-portable folded MLPs & alternative classifiers (`mlp_16_dot`, `mlp_16_skip_dot`, `mlp_32_16_dot`, `lr_l1_dot`, `lda_shrink_dot`, `lsvc_dot`, `gnb_quad`, `dt_core`, `gbdt_tiny_core`) in pure `float32` C99 (`0 malloc`, `0 <math.h>` at threshold) without touching `src/`: folded MLPs reach `0.6905` mean F1 across 15 MVTec-AD objects (vs `0.6486` `lr_dot`, `0.5178` `rf`) and `0.5000` F1 / `0.5936` PR-AUC on `kolektorsdd2` (`< 6.4 KB` Flash, `0.019–4.02 us` C99, `100%` prediction parity, `6` tests).

---

## 7. Completed vs. open gaps (priority order; see `HYPOTHESIS.md §4`)

**Completed on `main` (verified 2026-11-11):**
* Supervised ML (`dt`, `knn`, `lr`, folded `lr_dot`, `rf`) and 7-subset ablations (`x85_only`, `x9_only`, `x94_full`, `E_1x1..E_8x8`) on `kolektorsdd2` and `mvtec-ad_bottle` (normal + inverted).
* All 7 lightweight CNNs evaluated on identical `seed=42` splits (`kolektorsdd2` & `mvtec-ad_bottle`, normal + inverted).
* Strictly fair 100% CPU end-to-end benchmarking (`T_total = T_descriptor + T_model`, `threads=1`, `B=1/32`) with C99 `x85` backend (`results/comparison_cpu/`), granular profiling (`results/profiling/`), and isolated pure-Python pre-MCU CPU comparison (`results/comparison_cpu_python/`).

**Remaining gaps (in priority order):**
1. **Complete C99 port + real MCU benchmark:** Port `asymmetry_g2/4/8` (currently 74% of desktop descriptor latency), `lr_dot` weights (`94 + 1` floats), and compact `257`-entry `float32` Flash LUTs into standalone heap-free C99 alongside `entropy_pyramid.c`; measure Flash, peak SRAM, cycles/latency, and energy on real embedded hardware.
2. **CNN threshold non-degeneracy guard & dual reporting (`M2`):** Add a guard in `find_optimal_threshold` (`src/models/cnn/evaluate.py`) against all-positive collapse on small validation splits (`bottle` `N_val=35`) and report both $\tau=0.50$ and calibrated $\tau^*$ side-by-side for ML and CNNs.
3. **`DecisionTreeClassifier` determinism + multi-category/multi-seed sweeps (`M3`):** Pass `random_state=cfg.seed` in `build_classifier` (`src/models/descriptor_classifier.py`) to stop tie-breaking drift on `dt`, and complete `seed=42` (plus per-seed folders for multi-seed) CNN + CPU benchmarks on `mvtec-ad_tile` and additional categories.
4. **Finer grids & dynamic region selection:** Ablate `[1, 2, 4, 8, 16]` vs. `x94-v1` on micro-cracks (`kolektorsdd2`), and implement entropy-aware top-$k$/threshold region selectors feeding downstream cropped classifiers (`T_total = T_selector + T_model`).

---

## 8. Reproducibility rules that bite (see `README.md` for all CLI commands)

* **Seeds & overwrites:** `20260928` (labs `02`/`03`), `20261001` (labs `05`/`06`/`07`), `42` (canonical ML/CNN/profiling). Always pass `--seeds 42` in `evaluate_cnn.py` when targeting canonical single-seed folders because multi-seed loops overwrite the same output folder.
* **`dt` tie-breaking:** `DecisionTreeClassifier` currently omits `random_state` in `build_classifier`, causing small variance across retrains (e.g. `bottle_dt` `0.6364` vs. `0.5714`).
* **Lab locks:** Never edit files inside `understanding/07_descriptor_behavior/` (SHA-256 locked; run with `$env:PYTHONPATH = (Get-Location).Path`) or edit hashes/`run1/` in `02`/`03`. For re-running Labs `01–03`, follow [README.md](README.md#repetir-01-03-sin-sobrescribir-evidencia).
* **Legacy deprecation:** Never cite `results/comparison/` (`compare_baseline_vs_cnn.py`) for latency claims — it mixes GPU CNN latency with CPU ML latency. Use `results/comparison_cpu/` and `results/profiling/` only.
