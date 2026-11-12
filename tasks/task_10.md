# Task 10: Formal Integration of Single-Pass Entropy Backends & Folded Linear Classifier

## 1. Objective

Promote the validated optimizations from Lab 13 (`understanding/13_entropy_maps_optimization/`) into the formal `src/`, `configs/`, `experiments/`, and `tests/` pipeline on branch `feat/fast-entropy-opt`:

1. **Multi-backend entropy pyramid extraction (`src/features/entropy_maps.py`, `src/features/native/`, `src/features/native_entropy.py`, `src/features/build_native.py`):**
   - `numpy_baseline`: general-purpose fallback for arbitrary non-nested grid levels (`[3, 5]`, etc.).
   - `numpy_finest_pyramid`: single global `np.bincount` pass over the finest nested grid + hierarchical child-sum pyramid + per-level Shannon LUT.
   - `numpy_fused_entropy`: finest-grid pyramid + single fused `[N, total_features, bins]` `c * log2(c)` LUT evaluation.
   - `c_single_pass`: zero-heap-allocation C99 DLL (`entropy_pyramid_u8`) loaded via `ctypes` with preallocated buffers and compiled MSVC build helper.
   - `auto`: selects `c_single_pass` when grid levels nest cleanly and the compiled DLL passes structural limits; otherwise selects `numpy_finest_pyramid` for nested grids or `numpy_baseline` for general non-nested grids. Explicit backends fail fast on incompatible grids or missing DLLs.
2. **Folded linear decision-core classifier (`LinearDotCoreClassifier`, key `lr_dot`):**
   - Fits optional `StandardScaler` + `LogisticRegression` and mathematically folds scaling (`w_eff = w / scale`, `b_eff = b - dot(w_eff, mean)`) into a single 94-D dot product + bias scalar (`w_eff @ x + b_eff`), eliminating per-call sklearn dispatch and scaler overhead while preserving numerical parity (`atol <= 1e-12`).
3. **Profiling, CPU-fair comparison & downstream experiments:**
   - Update `src/profiling/` (`config.py`, `descriptor_stages.py`, `classifier_profile.py`, `report.py`) and `experiments/` (`export_entropy_maps.py`, `evaluate_descriptors.py`, `benchmark_cpu_fair.py`, `profile_pipelines.py`, `compare_baseline_vs_cnn.py`) to expose `--backend` / `--linear-mode`, profile C/NumPy stages and `lr_dot` accurately, and regenerate downstream results without altering existing `data/`.

## 2. Verification Commands (PowerShell from `code/`)

```powershell
# 1. Environment & CUDA check
.\.venv\Scripts\python.exe tests/test_environment.py

# 2. Build native C99 DLL into src/features/native/build/
.\.venv\Scripts\python.exe -m src.features.build_native --config configs/features/native_entropy.toml

# 3. Full unit & integration test suite (+ Lab 13 regression tests)
.\.venv\Scripts\python.exe -m unittest discover tests -v
.\.venv\Scripts\python.exe -m unittest understanding/13_entropy_maps_optimization/tests/test_entropy_variants.py -v

# 4. Dry-run entropy CSV export validation (do not overwrite existing data/csv/)
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/mvtec_ad.toml --dry-run

# 5. Re-evaluate classical descriptors (including lr_dot)
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all --inverted --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier all --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --object bottle --classifier all --inverted --overwrite

# 6. Train lightweight CNN checkpoints on GTX 1050 CUDA (seed=42)
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model all --seeds 42 --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model all --seeds 42 --inverted --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object bottle --model all --seeds 42 --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --object bottle --model all --seeds 42 --inverted --overwrite

# 7. Phase A: CPU-fair cross-paradigm benchmark
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml --inverted
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --objects bottle --inverted

# 8. Phase B: Deep profiling & figures
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/mvtec_ad.toml --objects bottle

# 9. Cross-paradigm baseline-vs-CNN comparison
.\.venv\Scripts\python.exe -m experiments.compare_baseline_vs_cnn --dataset kolektorsdd2
.\.venv\Scripts\python.exe -m experiments.compare_baseline_vs_cnn --dataset kolektorsdd2 --inverted
.\.venv\Scripts\python.exe -m experiments.compare_baseline_vs_cnn --dataset mvtecad --object bottle
.\.venv\Scripts\python.exe -m experiments.compare_baseline_vs_cnn --dataset mvtecad --object bottle --inverted
```
