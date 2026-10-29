# RESUME.md — project state for any agent

Last reviewed: 2026-10-22. Provisional name: *Entropy-Aware Region Selection for Efficient Embedded Vision* (folder `2026-10-Symmetry-Entropy`).

Read `AGENTS.md` first: it holds the operating rules (Windows 10, PowerShell, Python 3.11, venv at `code/.venv`, PyTorch cu126, no global installs, no TensorFlow, scientific-integrity rules). Then this file. Then `code/understanding/README.md` (knowledge map).

## 1. Research goal

Classify defects/errors in industrial images using hand-crafted image entropy features as a lightweight alternative (or complement) to a CNN, plus asymmetry metrics of the entropy layout. Long-term: efficient region selection before running a model, possibly on embedded hardware.

Stated claim to test: selecting informative regions / describing images with entropy + asymmetry can replace or cheapen a CNN while keeping acceptable quality. **No experiment has tested this claim yet.** Everything so far is component-level understanding: patches, histograms, Shannon entropy, lightweight CNNs, one94-d descriptor, and behavior tests of that descriptor. `code/src/` and `code/experiments/` are still empty reservations.

## 2. Current state in one paragraph

Seven understanding labs exist under `code/understanding/` (01–07). Labs 01–04 are the original four (patches, histograms, entropy benchmark, lightweight CNNs). Labs 05–07 were added later (asymmetry features, canonical x94 descriptor, descriptor behavior tests) and form a dependency chain: 07 imports 06, which imports 05 and the entropy reference `e1` from 03. The canonical deliverable so far is a fixed 94-dimension descriptor `image -> x94` (85 normalized Shannon entropies over a 1/2/4/8 spatial pyramid + 9 asymmetry residuals LR/TB/R180 at scales 2/4/8), fully tested (25 tests in 06, 20 in 05, 15 in 07, 5 in 03, 8 in 04, 1 environment test). Root and `code/` READMEs are stale: they only list labs 01–04 and still describe 05–07 as missing (lab 07's own README §2 acknowledges this). No real dataset, no classifier on x94, no selector pipeline, no embedded deployment.

## 3. Directory map

```text
AGENTS.md                     binding work rules (read first)
README.md                     project entry (stale: lists only labs 01-04)
RESUME.md                     this file
code/
├── .venv/                    Python 3.11.9, PyTorch 2.14.0+cu126, CUDA verified on GTX 1050
├── requirements.txt          direct pins (torch+cu126 first, then the rest)
├── README.md                 environment, install, replay protocol (stale lab list)
├── src/                      formal pipeline modules (src/preprocessing/: letterbox, mvtec_ad)
├── experiments/              CLI experiments (experiments/preprocess_mvtec_ad.py)
├── tests/                    test_environment.py (CUDA), test_preprocessing.py (letterbox & dataset tests)
├── results/                  env diagnostics + review logs (results/README.md = review registry)
└── understanding/            the eight labs (01–07 exploratory labs + 08 verification lab)
    ├── 01_patch_partition/       patch geometry, DROP/PARTIAL/PAD edge policies, exact reconstruction
    ├── 02_histogram_benchmark/   8 exact histogram implementations, 35 inputs, cost study
    ├── 03_entropy_benchmark/     Shannon E1-E5, LUTs, quantization, ranking; reference `e1`
    ├── 04_lightweight_cnns/      7 CNN architectures, demo training/inference/benchmark (CPU/CUDA)
    ├── 05_symmetry_features/     9 asymmetry features of entropy layouts; theory + tests
    ├── 06_entropy_pyramid_descriptor/  CANONICAL image -> x94 extractor (schema x94-v1)
    └── 07_descriptor_behavior/   15 controlled behavior experiments on x94 (279 comparison rows)
```

Each lab keeps its data/results next to its own code (`assets/`, `outputs/`, `results/`). Labs 01–03 write to fixed paths: repeat them only via the replay-copy procedure in `code/README.md` ("Repetir 01-03 sin sobrescribir evidencia") or you overwrite canonical evidence.

## 4. Dependency graph (imports)

```text
01_patch_partition  --data-->  02_histogram_benchmark, 03_entropy_benchmark
03_entropy_benchmark.e1  <-->  05 (imports e1), 06 (imports e1)
05_symmetry_features.symmetry_features  <-->  06 (imports compute_nine_asymmetry_features)
06_entropy_pyramid_descriptor           <-->  07 (importlib: understanding.06_...)
04_lightweight_cnns                     (independent; never consumes selected patches)
```

Cross-lab imports use `importlib.import_module("understanding.<name>...")` after inserting `code/` into `sys.path`; numbered folder names are not valid Python identifiers. Keep 03, 05, 06 as siblings. **07 must import 06's API, never copy the descriptor math.**

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

Key API (06): `extract_descriptor(image, bins) -> DescriptorResult` (`.vector`, `.feature_names`, `.entropy_features`, `.asymmetry_features`, `.entropy_grids[g]`, dims, bins), plus `extract_descriptor_vector`. Entry points: `descriptor.py`, `entropy.py` (traceable counts→p→terms→H), `partition.py`, `feature_names.py`.

Properties proven by tests (not assumed): exact94 names/order independent of dict iteration; each level recomputes pooled histograms (coarse entropy is never the average of children); equiprobable counts use analytic `H=log2(B)` to dodge float overshoot; reflections permute the 85 E exactly at 256×256 and leave the 9 A invariant (measured error 0), but NOT at 257×257 (floor borders break mirror geometry; max error ~0.009).

## 6. What each lab established (evidence, not claims)

- **01** — Patch geometry works: coordinates, three edge policies (DROP/PARTIAL/PAD), pixel-exact reconstruction. `outputs/run_summary.json`, cached Lorem Picsum JPEG + derived 500×333 PNG (SHA256 in `assets/source_metadata.json`).
- **02** — Exact 256-bin histogram is Ω(N); direct-address list is already the right algorithm (Θ(N+256)). 8 methods × 35 inputs, medians in `results/summary_table.md` (H1≈53.7 ns/pixel). No algorithmic win from fancy structures; CPython timings do not transfer to embedded C.
- **03** — Shannon reference `e1` (literal probabilities) + e2 (counts) + e3/e4 (LUTs) + sparse e5. Equivalence ≤1e-12 verified (`PASS: 7668 comparisons`). Q16 quantization can invert patch rankings; Q30 showed no strict inversions in measured cases. LUT memory table per patch size. Historical runs in `results/run1/`, validator `analyze_results.py` (hash-checks code vs metadata; do not edit hashes to pass).
- **04** — Seven models (tiny_cnn, small_cnn, ds_cnn, mobilenet_like, tiny_resnet, ultra_tiny, torchvision_mobilenet; 1.1k–1.5M params at 2 classes), logits `[B,num_classes]`, `AdaptiveAvgPool2d(1)`. Demo training on synthetic data only; fixture `outputs/folder_fixture` has classes `defective=0`, `normal=1` (folder-reading test, not a dataset). Benchmark = fresh forwards, batch 1, input 128. This is the CNN baseline the entropy approach must later be compared against; it currently has no link to entropy selection.
- **05** — The 9 asymmetry features defined and bounded (A in [0,1], triangular relations `|A_LR−A_TB| ≤ A_R180 ≤ A_LR+A_TB`, two exact symmetries imply the third). 85+9=94 table. Worked 2×2 example gives (.4,.2,.4). Nine synthetic images with measured A per scale. Documented failure modes: equal-histogram/different-structure collisions give all A=0 exactly; A averages away localized defects by g²; crop/alignment/grid/B dependence. 20 tests.
- **06** — Canonical extractor + schema (section 5 above). Six synthetic demo cases (uniform, iid_noise, localized_texture, symmetric, asymmetric, worked 16×16; seed 20261001). JSON/CSV export, SHA256 of pixels, `x94-v1`. 25 tests.
- **07** — Controlled behavior study of `f(image)→x94`, 15 experiments, 279 rows, seed 20261001, 256×256, B=16, tol 1e-6. Files: `run_all.py` (CLI/export), `experiments.py` (protocols), `synthetic_scenes.py` (analytic textured scene + checkerboard/noise/collision fixtures), `perturbations.py` (offset/contrast/noise/blur/defect/translate with explicit uint8 rounding), `compare_descriptors.py` (metrics L1/MAD/L2/max/changed per block, top-10, locality, curve shape). Results: 15 CSVs + `response_curves.csv`, `top_features.csv`, `multiscale_response.csv`, `observations.md`, plots, `outputs/check_128` and `outputs/check_257` companion runs. 15 tests (regenerate-from-seed exactness).

### 07 headline findings (measured, one seed, synthetic only)

| Experiment | Result |
|---|---|
| Noise σ=0→30 | mean E rises 0.661→0.830, **mean A falls** 0.0154→0.0095: noise looks like "more info" and flatter symmetry |
| Blur r=0→8 | mean E 0.661→0.097; **A non-monotonic** (min at r=4) |
| Defect size/strength | L1/L2/max grow monotonically (in these fixtures); response localizes to intersected cells |
| Defect position x=0..240 | 32 consecutive steps with zero change above tolerance: movement inside a cell can be invisible |
| Grid boundary | response peaks when defect *front* crosses a cell border, before its center; centered at x=32 distance drops ~19% vs inside-cell, same defect |
| Translation (zero-pad) | L2 non-monotonic horizontal, monotonic vertical: not translation-invariant |
| Reflections/rot180 @256 | E permutes exactly, 9 A invariant exactly; @257 invariance breaks (floor partitions) |
| Same histogram, different structure | **exact x94 collision** (L1=L2=0) at every B tested (8..256): Shannon ignores intra-cell arrangement |
| Bins 8..256 | more bins do not monotonicly increase discrimination; collision persists |
| Contrast/offset | small but measurable; ±40 offset gives identical descriptor (bin relabeling without clipping) |

Implication for the paper: the descriptor has demonstrable blind spots (collisions, noise inflates E, grid/alignment sensitivity, A=0 ≠ no defect) and demonstrable desirable behavior (monotone response to defect size/strength, cell localization, explicit symmetry relations). Predictive value on real defects is **entirely untested**.

## 7. What does NOT exist yet (the actual gaps)

1. Task + dataset definition (industrial defect dataset, train/val/test splits). Everything is synthetic or one Picsum image.
2. Any classifier on x94 (logistic/RF/SVM/MLP) or ablation entropy-only(85) / asymmetry-only(9) / combined(94) — 05 §18 names this ablation, never ran it.
3. Selector pipeline: top-k / threshold selection, tie-breaking, aggregation, patch budget.
4. Baseline comparison under equal budget: full image / all patches / random selection vs entropy selection; quality + cost.
5. Cost accounting: `T_selector`, `T_model`, `T_total = T_selector + T_model`, params, FLOPs/MACs, RAM/VRAM, latency, throughput (mandated by AGENTS.md; less patches ≠ less total time).
6. Sobel/edge features, MLP studies, real CNN-vs-descriptor experiment (04 never consumes entropy-selected patches).
7. Anything in `code/src/`, `code/experiments/`; embedded/C implementation; figures/tables for the paper.

## 8. Environment and commands

Interpreter: `.\code\.venv\Scripts\python.exe` from repo root `D:\Publications\2026-10-Symmetry-Entropy`. Do not activate; do not install globally; one pip at a time. Verified: Python 3.11.9, torch 2.14.0+cu126, torchvision 0.29.0+cu126, CUDA 12.6 runtime, GTX 1050 (sm_61), 4 GiB VRAM. Never CUDA 13.x builds. GPU test must actually run CUDA and fail otherwise.

```powershell
# health check
.\code\.venv\Scripts\python.exe .\code\tests\test_environment.py      # expects "Environment verification: PASS"
.\code\.venv\Scripts\python.exe -m pip check                          # "No broken requirements found."

# lab tests (invoke by path; no global pytest discovery)
.\code\.venv\Scripts\python.exe .\code\understanding\03_entropy_benchmark\test_entropy.py          # 5 tests, needs lab 01 outputs
.\code\.venv\Scripts\python.exe .\code\understanding\04_lightweight_cnns\tests\test_models.py      # 8 tests, CPU+CUDA
.\code\.venv\Scripts\python.exe .\code\understanding\05_symmetry_features\tests\test_symmetry_features.py   # 20
.\code\.venv\Scripts\python.exe .\code\understanding\06_entropy_pyramid_descriptor\tests\test_descriptor.py # 25
.\code\.venv\Scripts\python.exe .\code\understanding\07_descriptor_behavior\tests\test_behavior_helpers.py  # 15 (needs default 07 run)

# demos / experiments
.\code\.venv\Scripts\python.exe .\code\understanding\06_entropy_pyramid_descriptor\run_demo.py
.\code\.venv\Scripts\python.exe .\code\understanding\07_descriptor_behavior\run_all.py
.\code\.venv\Scripts\python.exe .\code\understanding\07_descriptor_behavior\run_all.py --experiment grid_boundary
.\code\.venv\Scripts\python.exe .\code\understanding\05_symmetry_features\run_demo.py
```

`pytest` is not installed; tests are plain `unittest`/script-style, use `assert` (never run Python with `-O`). matplotlib is optional in 07 (plots skipped if absent; it is installed per requirements). Replay of 01–03 without overwriting evidence: copy procedure in `code/README.md`.

## 9. Reproducibility rules that bite

- Seeds: 20260928 (labs 02/03), 20261001 (labs 05/06/07), 42 (CNN demo). Recorded per run in metadata JSON/manifests with SHA256 of code and pixels.
- Labs 01–03 and 05/06/07 default outputs overwrite same-named files on rerun; use `--output` (04/05/06/07) or a replay copy (01–03).
- `analyze_results.py` fails if script hash ≠ recorded hash: never edit hashes or delete `run1/` to make it pass.
- Short benchmark runs (`--target-pixels-per-run 10000`, `--rounds 1`) validate execution only; their timings never replace canonical tables (AGENTS.md integrity rule).
- Do not edit CSV/results to fit a hypothesis.

## 10. Suggested next steps (priority order)

1. Update stale entry docs (root `README.md`, `code/README.md`, `code/understanding/README.md`, `code/src/README.md`, `code/experiments/README.md`): they still say "four labs" and "no descriptor exists". Lab 07 README §2 already notes this.
2. Define the task and dataset (industrial defect images; e.g. a public anomaly dataset), splits, and metrics (accuracy/F1/AUROC) before any modeling.
3. First real experiment in `code/experiments/`: x94 → simple classifier, ablation 85 vs 9 vs 94, with seeds and a held-out test set.
4. Add baseline CNN from lab 04 on the same data; report params/latency honestly.
5. Then selector study: entropy top-k vs random vs all-patches, measuring `T_selector`, `T_model`, `T_total`, patches processed, quality.
6. Only after 2–5: promote proven modules into `code/src/`, then embedded/C work.

## 11. Known documentation inconsistencies (verified2026-10-22)

- Root `README.md` and `code/README.md` and `code/understanding/README.md` list only labs 01–04; labs 05/06/07 exist with full code, tests and results.
- `code/src/README.md` / `code/experiments/README.md` correctly say "empty", but `code/understanding/README.md` "Qué falta" list omits that the x94 descriptor and its behavior study now exist.
- `code/results/README.md` says "no Git repository initialized at root" — the root is now a git repo; verify before trusting that line.
- Lab 05 reuses `e1` from 03 via `sys.path` insertion; lab 06 does it via `importlib`. Both work only while03/05/06 stay siblings under `code/understanding/`.

## 12. Quick file index (code only, no outputs)

```text
code/tests/test_environment.py
code/understanding/01_patch_partition/patch_partition.py
code/understanding/02_histogram_benchmark/histogram_benchmark.py
code/understanding/03_entropy_benchmark/{entropy_benchmark.py, analyze_results.py, test_entropy.py}
code/understanding/04_lightweight_cnns/{common.py, train_demo.py, predict_image.py, benchmark_inference.py,
        inspect_models.py, inspect_architecture.py, models/*.py (7 archs + registry), tests/test_models.py}
code/understanding/05_symmetry_features/{symmetry_features.py, entropy_pyramid.py, synthetic_images.py,
        synthetic_matrices.py, run_demo.py, tests/test_symmetry_features.py}
code/understanding/06_entropy_pyramid_descriptor/{descriptor.py, entropy.py, partition.py, feature_names.py,
        synthetic_examples.py, run_demo.py, __init__.py, tests/test_descriptor.py}
code/understanding/07_descriptor_behavior/{run_all.py, experiments.py, compare_descriptors.py,
        perturbations.py, synthetic_scenes.py, tests/test_behavior_helpers.py,
        results/observations.md + 15 CSVs + manifests}
```

Every lab README (01–07) is long, precise and contains the measured numbers; consult the specific one before changing anything in that lab. When in doubt: understanding-first — new concepts go to `code/understanding/`, not `code/src/`.
