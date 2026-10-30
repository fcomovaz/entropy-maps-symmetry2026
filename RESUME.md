# RESUME.md — project state for any agent

Last reviewed: 2026-10-30. Provisional name: *Entropy-Aware Region Selection for Efficient Embedded Vision* (folder `2026-10-Symmetry-Entropy`).

Read `AGENTS.md` first: it holds the operating rules (Windows 10, PowerShell, Python 3.11, venv at `code/.venv`, PyTorch cu126, no global installs, no TensorFlow, scientific-integrity rules). Then this file. Then `code/understanding/README.md` (knowledge map).

## 1. Research goal

Classify defects/errors in industrial images using hand-crafted image entropy features as a lightweight alternative (or complement) to a CNN, plus asymmetry metrics of the entropy layout. Long-term: efficient region selection before running a model, possibly on embedded hardware.

Stated claim to test: selecting informative regions / describing images with entropy + asymmetry can replace or cheapen a CNN while keeping acceptable quality.

## 2. Current state in one paragraph

Eleven understanding labs exist under `code/understanding/` (01–11). Labs 01–04 cover exploratory primitives (patches, histograms, entropy benchmark, lightweight CNNs). Labs 05–07 establish the canonical 94-dimension descriptor `image -> x94` (schema `x94-v1`: 85 pyramid Shannon entropies + 9 asymmetry residuals) and its 15 behavior experiments. Formal pipeline modules live in `code/src/` (`src/preprocessing/` for letterbox, MVTec-AD and KolektorSDD2; `src/datasets/` for tabular CSV export). CLI experiment entry points in `code/experiments/` automate dataset standardization (`preprocess_mvtec_ad.py`, `export_mvtec_ad_csv.py`, `preprocess_kolektorsdd2.py`, `export_kolektorsdd2_csv.py`). Labs 08–11 provide visual validation and metric integrity checks for MVTec-AD (standard and inverted), its flattened CSV representation, KolektorSDD2 binary classification, and KolektorSDD2 flattened CSV representation. Formal unit tests in `code/tests/` verify environment, preprocessing, KolektorSDD2 pairing, and CSV export suites.

## 3. Directory map

```text
AGENTS.md                     binding work rules (read first)
README.md                     environment, install, replay protocol, dataset pipelines
RESUME.md                     this file
code/
├── configs/                  declarative configurations (no hardcoding in src/)
│   ├── base.toml             dataset roots, categories, and splits
│   ├── logging.py            centralized rotating logging setup
│   ├── preprocessing/
│   │   ├── mvtec_ad.toml     letterbox, 128x128, LANCZOS, NEAREST for masks
│   │   └── kolektorsdd2.toml binary mask threshold, letterbox, regex stem pairing
│   └── csv/
│       ├── mvtec_ad.toml     16391 columns tabular export specification
│       └── kolektorsdd2.toml 16387 columns tabular export specification
├── src/                      formal pipeline modules
│   ├── preprocessing/
│   │   ├── letterbox.py      pure routines: letterbox_to_square, to_grayscale, invert_intensity
│   │   ├── mvtec_ad.py       batch traversal, LANCZOS images, NEAREST masks, inverted variant
│   │   └── kolektorsdd2.py   stem pairing, native mask defect classification, letterbox, no mask output
│   └── datasets/
│       ├── mvtec_ad_csv.py   streaming CSV writer (.tmp -> atomic replace), 16391 columns
│       └── kolektorsdd2_csv.py streaming CSV writer (.tmp -> atomic replace), 16387 columns
├── experiments/              CLI experiment entry points
│   ├── preprocess_mvtec_ad.py       MVTec-AD standardized letterboxing
│   ├── export_mvtec_ad_csv.py       MVTec-AD row-major pixel flattening
│   ├── preprocess_kolektorsdd2.py   KolektorSDD2 pairing, classification, letterboxing
│   └── export_kolektorsdd2_csv.py   KolektorSDD2 row-major pixel flattening
├── tests/                    formal test suites (runnable via unittest)
│   ├── test_environment.py          real CUDA execution, matmul, small net forward on GTX 1050
│   ├── test_preprocessing.py        letterbox geometry, masks preservation, inversion, CLI
│   ├── test_kolektorsdd2.py         mask classification, copy/orphan filtering, letterbox, CLI
│   ├── test_csv_export.py           MVTec-AD CSV schema, pixel round-trip, streaming, CLI
│   └── test_kolektorsdd2_csv_export.py KolektorSDD2 CSV schema, pixel round-trip, streaming, CLI
├── results/                  env diagnostics + review logs (results/README.md = review registry)
└── understanding/            eleven pedagogical laboratories
    ├── 01_patch_partition/          patch geometry, DROP/PARTIAL/PAD edge policies, exact reconstruction
    ├── 02_histogram_benchmark/      8 exact histogram implementations, 35 inputs, cost study
    ├── 03_entropy_benchmark/        Shannon E1-E5, LUTs, quantization, ranking; reference `e1`
    ├── 04_lightweight_cnns/         7 CNN architectures, demo training/inference/benchmark (CPU/CUDA)
    ├── 05_symmetry_features/        9 asymmetry features of entropy layouts; theory + tests
    ├── 06_entropy_pyramid_descriptor/  CANONICAL image -> x94 extractor (schema x94-v1)
    ├── 07_descriptor_behavior/      15 controlled behavior experiments on x94 (279 comparison rows)
    ├── 08_verify_mvtec_preprocessing/  verification and visual comparison for letterboxed 128x128 dataset
    ├── 09_verify_mvtec_csv/         verification and visual round-trip comparison for flattened CSV dataset
    ├── 10_verify_kolektorsdd2/      verification and visual comparison for KolektorSDD2 128x128 good/defect
    └── 11_verify_kolektorsdd2_csv/  verification and visual round-trip comparison for KolektorSDD2 CSV
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

## 7. What does NOT exist yet (the actual remaining gaps)

1. **Classification models on real data:** Train and evaluate classifiers (Logistic Regression, Random Forest, SVM, MLP) on `x94` features extracted from MVTec-AD and KolektorSDD2.
2. **Ablation study on real datasets:** Measure performance of 85 entropies alone vs 9 asymmetries alone vs combined 94.
3. **Region selection pipeline:** Implement top-k / threshold region selector feeding CNN models from lab 04.
4. **End-to-end cost accounting:** Measure actual timings: $T_{\text{selector}}$, $T_{\text{model}}$, $T_{\text{total}} = T_{\text{selector}} + T_{\text{model}}$, RAM, VRAM, and throughput to test if filtering patches saves total time.
5. **Embedded implementation:** Pure C or embedded inference on microcontrollers/embedded boards.

## 8. Environment and commands

Interpreter: `.\code\.venv\Scripts\python.exe` from repo root `D:\Publications\2026-10-Symmetry-Entropy` (or `.\.venv\Scripts\python.exe` from `code/`). Verified: Python 3.11.9, PyTorch 2.14.0+cu126, torchvision 0.29.0+cu126, CUDA 12.6 runtime, GTX 1050 (sm_61), 4 GiB VRAM.

```powershell
Set-Location D:\Publications\2026-10-Symmetry-Entropy\code

# 1. Health check & formal unit test suites
.\.venv\Scripts\python.exe tests/test_environment.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest tests/test_preprocessing.py
.\.venv\Scripts\python.exe -m unittest tests/test_kolektorsdd2.py
.\.venv\Scripts\python.exe -m unittest tests/test_csv_export.py

# 2. Dataset preprocessing & export experiments
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad
.\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --inverted
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv
.\.venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --inverted
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2
.\.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2 --inverted

# 3. Post-run visual validations
.\.venv\Scripts\python.exe understanding/08_verify_mvtec_preprocessing/verify.py
.\.venv\Scripts\python.exe understanding/09_verify_mvtec_csv/verify.py
.\.venv\Scripts\python.exe understanding/10_verify_kolektorsdd2/verify.py

# 4. Pedagogical lab tests
.\.venv\Scripts\python.exe understanding/03_entropy_benchmark/test_entropy.py
.\.venv\Scripts\python.exe understanding/04_lightweight_cnns/tests/test_models.py
.\.venv\Scripts\python.exe understanding/05_symmetry_features/tests/test_symmetry_features.py
.\.venv\Scripts\python.exe understanding/06_entropy_pyramid_descriptor/tests/test_descriptor.py
.\.venv\Scripts\python.exe understanding/07_descriptor_behavior/tests/test_behavior_helpers.py
```

## 9. Reproducibility rules that bite

- Seeds: 20260928 (labs 02/03), 20261001 (labs 05/06/07), 42 (CNN demo, CSV verification, KolektorSDD2 verification).
- Labs 01–03 and 05/06/07 default outputs overwrite same-named files on rerun; use `--output` or a replay copy.
- `analyze_results.py` fails if script hash $\ne$ recorded hash: never edit hashes or delete `run1/` to make it pass.
- Dry-run flags (`--dry-run`) exist across all preprocessing and export CLIs to verify dataset scans before writing to disk.

## 10. Suggested next steps (priority order)

1. Implement feature extraction pipeline: batch extract `x94` descriptor on preprocessed MVTec-AD and KolektorSDD2 datasets.
2. Train lightweight baseline classifiers (SVM, Random Forest, Logistic Regression, MLP) on `x94` vectors.
3. Execute ablation study: 85 entropy features vs 9 asymmetry features vs 94 combined.
4. Train baseline lightweight CNNs (from lab 04) on preprocessed 128x128 images.
5. Develop region selection pipeline (top-k / threshold patches) feeding CNNs and evaluate total latency: $T_{\text{total}} = T_{\text{selector}} + T_{\text{model}}$.

## 11. Quick file index (code only, no outputs)

```text
configs/base.toml
configs/preprocessing/{mvtec_ad.toml, mvtec_ad_csv.toml, kolektorsdd2.toml}
configs/logging.py

src/preprocessing/{letterbox.py, mvtec_ad.py, kolektorsdd2.py}
src/datasets/mvtec_ad_csv.py

experiments/{preprocess_mvtec_ad.py, export_mvtec_ad_csv.py, preprocess_kolektorsdd2.py}

tests/{test_environment.py, test_preprocessing.py, test_kolektorsdd2.py, test_csv_export.py}

understanding/01_patch_partition/patch_partition.py
understanding/02_histogram_benchmark/histogram_benchmark.py
understanding/03_entropy_benchmark/{entropy_benchmark.py, analyze_results.py, test_entropy.py}
understanding/04_lightweight_cnns/{common.py, train_demo.py, predict_image.py, benchmark_inference.py,
        inspect_models.py, inspect_architecture.py, models/*.py, tests/test_models.py}
understanding/05_symmetry_features/{symmetry_features.py, entropy_pyramid.py, synthetic_images.py,
        synthetic_matrices.py, run_demo.py, tests/test_symmetry_features.py}
understanding/06_entropy_pyramid_descriptor/{descriptor.py, entropy.py, partition.py, feature_names.py,
        synthetic_examples.py, run_demo.py, __init__.py, tests/test_descriptor.py}
understanding/07_descriptor_behavior/{run_all.py, experiments.py, compare_descriptors.py,
        perturbations.py, synthetic_scenes.py, tests/test_behavior_helpers.py}
understanding/08_verify_mvtec_preprocessing/verify.py
understanding/09_verify_mvtec_csv/verify.py
understanding/10_verify_kolektorsdd2/verify.py
```
