# RESUME.md — project state for any agent

Last reviewed: 2026-11-11. Provisional name: *Entropy-Aware Region Selection for Efficient Embedded Vision* (folder `2026-10-Symmetry-Entropy`).

Read `AGENTS.md` first: it holds the operating rules (Windows 10, PowerShell, Python 3.11, venv at `code/.venv`, PyTorch cu126, no global installs, no TensorFlow, scientific-integrity rules). Then read this file, [HYPOTHESIS.md](HYPOTHESIS.md) (verified hypotheses & pre-MCU pure-Python vs C99 synthesis), [REVIEW_01.md](REVIEW_01.md) (forensic audit), and [understanding/README.md](understanding/README.md) (knowledge map).

## 1. Research goal

Classify defects/errors in industrial images using hand-crafted image entropy features as a lightweight alternative (or complement) to a CNN, plus asymmetry metrics of the entropy layout. Long-term: efficient region selection before running a model, possibly on embedded microcontrollers (MCU).

Stated claim to test: selecting informative regions / describing images with entropy + asymmetry can replace or cheapen a CNN while keeping acceptable quality.

## 2. Current state in one paragraph

Thirteen understanding labs exist under `code/understanding/` (01–13). Labs 01–04 cover exploratory primitives (patches, histograms, entropy benchmark, lightweight CNNs). Labs 05–07 establish the canonical 94-dimension descriptor `image -> x94` (schema `x94-v1`: 85 pyramid Shannon entropies + 9 asymmetry residuals) and its 15 behavior experiments (279 scenes). Labs 08–12 validate preprocessing, CSV round-trips, spatial heatmaps, and pipeline stage profiling, while Lab 13 establishes single-pass C99 entropy optimization (`92.65x` over pure NumPy `V0` for `x85`). Formal pipeline modules live in `code/src/` (`src/preprocessing/` for letterbox, MVTec-AD and KolektorSDD2; `src/datasets/` for tabular CSV export; `src/features/` for NumPy and C99 `entropy_pyramid` multiscale entropy plus spatial asymmetry extraction; `src/models/` for `dt`, `knn`, `lr`, folded `lr_dot`, `rf`, subset ablation, spatial explainability, and 7 lightweight CNNs under `src/models/cnn/`; `src/profiling/` for 100% CPU-fair benchmarking and stage/layer profiling). CLI entry points in `code/experiments/` run dataset preprocessing, CSV/feature export, ML/CNN evaluation (`seed=42` aligned on `kolektorsdd2` and `mvtec-ad_bottle` normal/inverted), 100% CPU-fair comparisons (`results/comparison_cpu/`), pre-MCU pure-Python CPU runs (`results/comparison_cpu_python/`), and deep profiling (`results/profiling/`). Automated verification spans **294 tests + CUDA environment check** (`208` formal tests in `code/tests/` + `86` lab tests in `understanding/`).

## 3. Directory map

```text
AGENTS.md                     binding work rules (read first)
README.md                     environment, install, replay protocol, CLI quickstarts
HYPOTHESIS.md                 verified hypothesis verdicts + Python vs C99 pre-MCU synthesis
REVIEW_01.md                  forensic audit of historical bf4a29f snapshot & design notes
RESUME.md                     this file
code/
├── configs/                  declarative TOML configurations (no hardcoding in src/)
│   ├── base.toml             dataset roots, categories, splits, and full CSV paths
│   ├── logging.py            centralized rotating logging setup
│   ├── features/             native_entropy.toml (MSVC C99 DLL build flags & paths)
│   ├── preprocessing/        mvtec_ad.toml, kolektorsdd2.toml
│   ├── csv/                  mvtec_ad.toml (16391 cols), kolektorsdd2.toml (16389 cols)
│   ├── entropy/              mvtec_ad.toml, kolektorsdd2.toml (multiscale x85 + backend)
│   ├── asymmetry/            mvtec_ad.toml, kolektorsdd2.toml (x85 -> x94)
│   ├── classification/       mvtec_ad.toml, kolektorsdd2.toml (dt, knn, lr, lr_dot, rf)
│   ├── cnn/                  mvtec_ad.toml, kolektorsdd2.toml (7 CNNs, AdamW, val_f1)
│   └── profiling/            mvtec_ad.toml, kolektorsdd2.toml (100% CPU fair & profiling)
├── src/                      formal pipeline modules
│   ├── preprocessing/        letterbox.py, mvtec_ad.py, kolektorsdd2.py
│   ├── datasets/             mvtec_ad_csv.py, kolektorsdd2_csv.py
│   ├── features/
│   │   ├── entropy_maps.py   5 backends (auto, c_single_pass, numpy_finest_pyramid, numpy_fused_entropy, numpy_baseline)
│   │   ├── native_entropy.py ctypes wrapper & zero-alloc NativeEntropyPlan
│   │   ├── build_native.py   MSVC C99 DLL builder -> src/features/native/build/entropy_pyramid.dll
│   │   ├── native/           entropy_pyramid.h, entropy_pyramid.c (pure C99, zero heap allocs)
│   │   ├── entropy_csv.py    streaming CSV extractor for x85 entropy descriptors
│   │   ├── asymmetry_maps.py 2D reconstruction and 9 spatial asymmetry residuals (LR, TB, R180)
│   │   └── asymmetry_csv.py  streaming CSV appender (x85 -> x94, schema x94-v1)
│   ├── models/
│   │   ├── descriptor_classifier.py dt, knn, lr, LinearDotCoreClassifier (lr_dot), rf, ablation, explainability
│   │   ├── classification_plots.py  publication figures (CM+ROC/PR, ablation, importance, 2D heatmaps)
│   │   └── cnn/              architectures/ (7 CNNs), config.py, data.py, train.py, evaluate.py, benchmark.py, report.py
│   └── profiling/            config.py, timing.py, environment.py, descriptor_stages.py, classifier_profile.py, cnn_layers.py, report.py
├── experiments/              CLI experiment entry points
│   ├── preprocess_{mvtec_ad,kolektorsdd2}.py
│   ├── export_{mvtec_ad,kolektorsdd2}_csv.py
│   ├── export_entropy_maps.py
│   ├── append_asymmetry_features.py
│   ├── evaluate_descriptors.py
│   ├── evaluate_cnn.py
│   ├── compare_baseline_vs_cnn.py    legacy mixed-device CSV/plot generator (deprecated for CPU latency)
│   ├── benchmark_cpu_fair.py         100% CPU-fair ML vs CNN benchmark (supports --entropy-backend)
│   └── profile_pipelines.py          granular stage/layer/ATen/Mann-Whitney U profiling
├── tests/                    formal test suites (208 unittest cases + test_environment.py)
├── results/                  diagnostics, logs, classification, cnn, comparison_cpu, profiling, comparison (deprecated)
└── understanding/            pedagogical laboratories (01_patch_partition through 13_entropy_maps_optimization)
```

## 4. Dependency graph (imports)

```text
01_patch_partition  --data-->  02_histogram_benchmark, 03_entropy_benchmark
03_entropy_benchmark.e1  <-->  05 (imports e1), 06 (imports e1)
05_symmetry_features.symmetry_features  <-->  06 (imports compute_nine_asymmetry_features)
06_entropy_pyramid_descriptor           <-->  07 (importlib: understanding.06_...)
04_lightweight_cnns                     (independent; ready to consume standardized datasets)
src.preprocessing.letterbox             <-->  src.preprocessing.mvtec_ad, src.preprocessing.kolektorsdd2
src.preprocessing                       <-->  experiments/
src.datasets                            <-->  experiments/
```

Cross-lab imports in `understanding/` use `importlib.import_module("understanding.<name>...")` after inserting `code/` into `sys.path`.

## 5. The descriptor (core scientific artifact)

`code/understanding/06_entropy_pyramid_descriptor/` — schema `x94-v1`, frozen order in `feature_names.py`:

```text
input: uint8 grayscale image (Pillow convert('L')), width,height >= 8, any size
per level g in (1,2,4,8): partition into g*g half-open boxes (floor division), no padding/resize/DROP
per cell: counts[b] = #{pixels with pixel*B//256 == b}, B in [2,256], default 16
          p = counts/N,  H = -sum(p*log2(p))  (reference E1 from lab 03)
          E = H / log2(B)   in [0,1], no clipping
feature 0:      E of 1x1 (global)
features 1-84:  E of 2x2 (4), 4x4 (16), 8x8 (64), row-major
features 85-93: A_T(g) = sum |E[i,j] - T(E)[i,j]| / g^2  for T in {LR, TB, R180}, g in {2,4,8}
```

Key API (06): `extract_descriptor(image, bins) -> DescriptorResult` (`.vector`, `.feature_names`, `.entropy_features`, `.asymmetry_features`, `.entropy_grids[g]`, dims, bins), plus `extract_descriptor_vector`.

## 6. What each lab established (evidence, not claims)

- **01** — Patch geometry works: coordinates, three edge policies (DROP/PARTIAL/PAD), pixel-exact reconstruction.
- **02** — Exact 256-bin histogram is $\Omega(N)$; direct-address list is already the right algorithm ($\Theta(N+256)$).
- **03** — Shannon reference `e1` (literal probabilities) + e2 (counts) + e3/e4 (LUTs) + sparse e5. Equivalence $\le 10^{-12}$ verified. Q30 quantization avoids ranking inversions.
- **04** — Seven models (tiny_cnn, small_cnn, ds_cnn, mobilenet_like, tiny_resnet, ultra_tiny, torchvision_mobilenet; 1.1k–1.5M params), AdaptiveAvgPool2d(1). Inference benchmark: CPU vs GTX 1050.
- **05** — The 9 asymmetry features defined and bounded ($A \in [0, 1]$, triangular relations). Documented failure modes: equal-histogram/different-structure collisions give all $A=0$.
- **06** — Canonical extractor + schema `x94-v1`. Six synthetic demo cases, JSON/CSV export, SHA256 pixel tracking. 25 tests.
- **07** — Controlled behavior study of $f(\text{image}) \to x94$, 15 experiments, 279 rows. Key findings: response localizes to cells containing defects; noise inflates $E$ and flattens $A$; exact collisions occur for identical histograms with different spatial arrangements; peak sensitivity occurs when defect front crosses cell boundaries.
- **08** — Quality control of MVTec-AD standardization: dimensional integrity (`128x128`, `'L'`), strict mask binary preservation (`{0, 255}` via `NEAREST`), disk usage reduction, visual validation matrix.
- **09** — Quality control of MVTec-AD CSV serialization: validates 16391 columns header schema, 6612 total records, deterministic row-major order, and bit-exact reconstruction (absolute difference = 0).
- **10** — Quality control of KolektorSDD2: confirms binary defect classification via full-resolution mask inspection (`mask.max() > 0`), verifies exact category counts (train: 2085 good / 246 defect; test: 894 good / 110 defect), confirms zero mask files in output tree, generates visual comparison matrix.
- **11** — Quality control of KolektorSDD2 CSV serialization: validates 16389 columns header schema (`split, type, file_name, px_00000...px_16383, width, height`), 3335 total records, deterministic row-major order, and bit-exact reconstruction (absolute difference = 0).
- **12** — Stage-by-stage profiling & CNN layer dissection (`12_profiling_stages/`): identifies Python/NumPy per-cell histogram loops and `sklearn` `Pipeline.predict` wrapper overhead vs PyTorch `ATen` kernels.
- **13** — Single-pass entropy pyramid optimization (`13_entropy_maps_optimization/`): proves bottom-up histogram summation + Shannon LUTs in NumPy (`9.4x–10.5x`) and pure C99 `entropy_pyramid.dll` (`92.65x`, `0.024 ms` for `x85`) with exact $\le 10^{-12}$ parity (`13` unit tests).

## 7. Completed vs. actual remaining gaps (pre-MCU & paper)

**Completed on `main` (verified 2026-11-11):**
- Supervised real-data evaluation (`dt`, `knn`, `lr`, folded `lr_dot`, `rf`) and full 7-subset ablations (`x85_only`, `x9_only`, `x94_full`, `E_1x1..E_8x8`) on KolektorSDD2 and MVTec-AD (`bottle` normal + inverted).
- All 7 lightweight CNNs evaluated on identical `seed=42` splits (`kolektorsdd2` & `mvtec-ad_bottle`, normal + inverted).
- Strictly fair 100% CPU end-to-end benchmarking (`T_total = T_descriptor + T_model`, `threads=1`, `batch=1/32`) with C99 `x85` backend (`results/comparison_cpu/`), deep stage/layer profiling (`results/profiling/`), and isolated pure-Python pre-MCU evaluation (`results/comparison_cpu_python/`).

**Remaining gaps:**
1. **Pure C99 MCU pipeline port:** Port `asymmetry_g2/4/8` and folded `lr_dot` (94 `float32`/fixed-point MACs + Flash LUTs) into C99 alongside `entropy_pyramid.c`, eliminating all Python/NumPy calls, and benchmark directly on target embedded hardware (Flash/SRAM/FPU telemetry).
2. **Threshold & multi-category CNN/ML hardening:** Add a non-degeneracy guard in `find_optimal_threshold` (`src/models/cnn/evaluate.py`, per `REVIEW_01.md` `M2`) to prevent `val_f1` single-class collapse on small validation sets, run multi-seed sweeps (`seeds=[42, 43, 44]` per-seed folders), and extend CPU-fair CNN comparisons to additional MVTec-AD categories (e.g., `tile`).
3. **Sub-cell micro-crack grid ablation:** Evaluate `[1, 2, 4, 8, 16]` grids against frozen `x94-v1` on KolektorSDD2 to test finer spatial resolution without dropping `x94-v1` parity.
4. **Dynamic region selection pipeline:** Implement top-k / entropy-threshold patch selectors feeding downstream cropped classifiers / lightweight CNNs on selected informative patches.

## 8. Environment and commands

Interpreter: `.\code\.venv\Scripts\python.exe` from repo root `D:\Publications\2026-10-Symmetry-Entropy` (or `.\.venv\Scripts\python.exe` from `code/`). Verified: Python 3.11.9, PyTorch 2.14.0+cu126, torchvision 0.29.0+cu126, CUDA 12.6 runtime, GTX 1050 (sm_61), 4 GiB VRAM.

```powershell
Set-Location D:\Publications\2026-10-Symmetry-Entropy\code

# 1. Health check, native C99 build & full formal unit test suite (208 tests)
.\.venv\Scripts\python.exe tests/test_environment.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m src.features.build_native --config configs/features/native_entropy.toml
.\.venv\Scripts\python.exe -m unittest discover tests -v

# 2. Pedagogical lab tests (86 tests total)
.\.venv\Scripts\python.exe understanding/03_entropy_benchmark/test_entropy.py
.\.venv\Scripts\python.exe understanding/04_lightweight_cnns/tests/test_models.py
.\.venv\Scripts\python.exe understanding/05_symmetry_features/tests/test_symmetry_features.py
.\.venv\Scripts\python.exe understanding/06_entropy_pyramid_descriptor/tests/test_descriptor.py
$env:PYTHONPATH = (Get-Location).Path; .\.venv\Scripts\python.exe understanding/07_descriptor_behavior/tests/test_behavior_helpers.py
.\.venv\Scripts\python.exe -m unittest understanding/13_entropy_maps_optimization/tests/test_entropy_variants.py -v

# 3. Formal descriptor, CNN, CPU-fair & profiling experiments (see experiments/README.md for flags)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all --overwrite
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml
```

## 9. Reproducibility rules that bite

- Seeds: 20260928 (labs 02/03), 20261001 (labs 05/06/07), 42 (CNN/ML canonical comparisons, CSV/dataset verification). When evaluating multiple seeds in `evaluate_cnn.py`, pass `--seeds 42` for single-seed parity because multi-seed loops overwrite the same target folder.
- Labs 01–03 and 05/06/07 default outputs overwrite same-named files on rerun; use `--output` or a replay copy. Lab 07 (`understanding/07_descriptor_behavior/`) is SHA-256 locked (`compare_descriptors.py`); pass `$env:PYTHONPATH = (Get-Location).Path` rather than editing Lab 07.
- `analyze_results.py` fails if script hash $\ne$ recorded hash: never edit hashes or delete `run1/` to make it pass.
- Dry-run flags (`--dry-run`) exist across preprocessing, CSV/feature export, evaluation, and CPU profiling CLIs.

## 10. Suggested next steps (priority order)

1. Port `asymmetry_g2/4/8` + folded `lr_dot` (`w_eff_[94]`, `b_eff_`) + `float32`/fixed-point LUTs into pure standalone C99 for the MCU target (zero Python, zero heap allocation).
2. Add non-degeneracy guard in `find_optimal_threshold` (`src/models/cnn/evaluate.py`) and run multi-seed + additional MVTec-AD categories (`tile`, etc.) for paper completeness.
3. Run `[1, 2, 4, 8, 16]` vs `x94-v1` grid ablation on KolektorSDD2 to test sub-cell micro-crack sensitivity.
4. Implement and profile dynamic entropy-based region selection before downstream classification (`T_total = T_selector + T_model`).

## 11. Quick file index (code only, no outputs)

```text
configs/base.toml
configs/{features/native_entropy,preprocessing/*,csv/*,entropy/*,asymmetry/*,classification/*,cnn/*,profiling/*}.toml
configs/logging.py

src/preprocessing/{letterbox.py, mvtec_ad.py, kolektorsdd2.py}
src/datasets/{mvtec_ad_csv.py, kolektorsdd2_csv.py}
src/features/{entropy_maps.py, native_entropy.py, build_native.py, entropy_csv.py, asymmetry_maps.py, asymmetry_csv.py}
src/features/native/{entropy_pyramid.h, entropy_pyramid.c}
src/models/{descriptor_classifier.py, classification_plots.py}
src/models/cnn/{config.py, data.py, train.py, evaluate.py, benchmark.py, report.py, architectures/*.py}
src/profiling/{config.py, timing.py, environment.py, descriptor_stages.py, classifier_profile.py, cnn_layers.py, report.py}

experiments/{preprocess_mvtec_ad.py, export_mvtec_ad_csv.py, preprocess_kolektorsdd2.py, export_kolektorsdd2_csv.py,
             export_entropy_maps.py, append_asymmetry_features.py, evaluate_descriptors.py, evaluate_cnn.py,
             compare_baseline_vs_cnn.py, benchmark_cpu_fair.py, profile_pipelines.py}

tests/{test_environment.py, test_preprocessing.py, test_kolektorsdd2.py, test_csv_export.py,
       test_kolektorsdd2_csv_export.py, test_entropy_maps.py, test_native_entropy.py, test_entropy_csv.py,
       test_asymmetry_maps.py, test_asymmetry_csv.py, test_descriptor_classifier.py,
       test_cnn_architectures.py, test_cnn_config.py, test_cnn_data.py, test_cnn_train.py,
       test_cnn_evaluate.py, test_cnn_cli.py, test_profiling_config.py, test_profiling_timing.py,
       test_profiling_descriptor_stages.py, test_profiling_classifier.py, test_profiling_cnn_layers.py,
       test_benchmark_cpu_fair_cli.py}

understanding/{01..13}_*/  (labs 01-13; see understanding/README.md for exact file tree)
```
