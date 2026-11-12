# Optimization Entropy Maps


## Issue

- The task is optimize the entropy maps, right now the enropy maps are calculated by iterating N times over the image, where N is the number of patches in the image. One of the solutions you provided was to iterate over the patches in a single pass, what i did understand was to calculate only the "coordinates" of the pathes and then iterate over them in a single pass.
- i want to check if this is true, also i would like another solution type using a c function and call it (at the end this will be migrated to c but not yet). i want a big optimization to reduce the velocity in batch 1 (review REVIEW_01.md in the root of this code folder).
- also if you have another solution to overpass this overhead please tell me, i want a solution to overpass the pytorch time but with less parameters, and also maybe i will increase the patches so overpass this time is necessary.
- you can start by creating another laboratory in understanding doing this optimization, in this way you will never touch the src/ files yet.
- i really want to optimize this stage and this critical operation use any computer science resource you know and remember this will be migrated to c but not yet so if is easy to implement in that language it will be helpful. but the optimization is requiered, in the understanding for this also ceate a script with scientific graphs to see the comparison if there is a real improvement.



## Plan

> Don't execute anything, just plan it

- [x] Read AGENTS.md file to know the general rules of the project.
- [x] Create a detailed plan for this task and update it here to implement it later for an agent, so it needs to be detailed.
- [x] Update this file with the plan, stages and everything to resolve and implement the task.

> Remember not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.




---

# DETAILED IMPLEMENTATION PLAN (for the executing agent)

> Status: **planning only**. Nothing below has been executed. Do not touch `src/` in this task (laboratory work only, in `understanding/`). Follow `AGENTS.md`: parameters only in TOML, `pathlib` with root computed from `__file__`, `from __future__ import annotations`, type hints, `@dataclass(frozen=True, slots=True)`, CLI with `argparse` + `main(argv) -> int` + `if __name__ == "__main__": raise SystemExit(main())`, `logging` (not `print`, except CLI output), `.venv\Scripts\python.exe`, explicit seeds, no fabricated results, tests before declaring done.

## 0. Diagnosis (findings from the inspection)

| Finding | Evidence | Consequence |
|---|---|---|
| `compute_entropy_features` in `src/features/entropy_maps.py` **already** precomputes coordinates (`EntropyPlan.regions`, 85 `RegionSpec`) and loops over the 85 regions in Python. Each iteration does a non-contiguous slice + `astype(int32)` + `np.bincount` + `term_lut[counts]` + `sum`. | L331-399 | The idea "compute only the coordinates and iterate over them in a single pass" is **already implemented**, and it is exactly what remains slow (`region_histograms_g8` = 38.7 % of the descriptor, REVIEW_01 Table 7). Precomputing coordinates does not remove the 85 iterations nor the 85 reads of the image. |
| Cost is spread over ~85 NumPy calls of a few µs each, plus Python dispatch gap (0.74 ms). | REVIEW_01 Table 7, F5 | At batch=1 the fixed per-call overhead dominates. |
| Every pixel is read **4 times** (once per level 1/2/4/8). | `plan.regions` covers 4 levels | Redundant: histograms are **additive**. |
| No compiler on PATH (`gcc/cl/clang/zig` not found). `cmake`, `ninja` and **MSVC Build Tools** exist (`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` and `...\18\BuildTools`). `numba`/`cffi` are not installed; `setuptools 84` is. | environment check | Build the DLL with `cl.exe` through `vcvars64.bat` (located with `vswhere.exe`) and load it with `ctypes` (stdlib, **no new dependencies**). |

### Key idea: the finest grid, single pass

Histograms are integer counts and therefore additive. Compute **only** the histograms of the finest grid (8x8 = 64 cells) in one pass over the pixels, then derive the 4x4, 2x2 and 1x1 histograms by **summing child cells**. Each pixel is read once; cost is O(H*W + cells*bins). The sums are exact integers, so the resulting entropy is identical to the baseline.

Nesting condition: `floor(k*size/g)` nests levels only if `size % g_finest == 0` and `g_finest % g_coarser == 0` (true for 128 with [1,2,4,8]). The fast variants must **validate** this (fail fast with `ValueError`); no silent fallback.

## 1. Variants to compare (all must produce the same vector)

Discarded on purpose (do NOT implement): coordinate-loop variant (adds nothing over the existing plan), PyTorch variant (CNN inference is already benchmarked on CPU with `device = "cpu"` per task_08, so descriptor-in-torch adds no fairness), integral-histogram variant (we will not increase the number of patches; it is scientifically unjustified), and the fused C + asymmetry + LR pipeline (moved to a **separate future task**).

| ID | Variant | Idea | Expected (to be verified, NOT a result) |
|---|---|---|---|
| `V0_baseline` | `src.features.entropy_maps.compute_entropy_features` (with plan) | reference/oracle | n/a |
| `V1_numpy_finest_pyramid` | One global `bincount` with index `((n*cells + cell)*bins + q)` over the finest grid (via `reshape(N,g,h,g,w)`) giving `[N,64,bins]`; coarser levels via `reshape(N,g,k,g,k,bins).sum(...)`; entropy per level with the Shannon LUT | removes the 85-region Python loop; reads each pixel once | clear gain, most promising pure-NumPy option |
| `V2_numpy_fused_entropy` | V1 plus a refinement of the **entropy step only**: concatenate the cells of all levels into one `[N,85,bins]` array so the LUT lookup, the sum and the equiprobable rule run once (1 call) instead of once per level (4 calls) | minimizes NumPy call count (relevant at batch=1) | small extra gain over V1 at batch=1; measure whether it justifies the added complexity |
| `V3_c_single_pass` | C DLL (ctypes): single pass over the finest grid + pyramid sums + LUT + equiprobable rule, `double` output | main solution | largest gain, especially at batch=1 |

V1 is the most promising pure-Python/NumPy candidate; V3 (native) is expected to be the largest overall gain because it removes both the NumPy call overhead and Python dispatch. This is a hypothesis to be confirmed by measurement. If V2 does not beat V1 meaningfully, report it as a negative result and keep V1.

Ideas to document in the lab README (evaluate with evidence, do not implement all):
- **Fusing quantization + histogram** in C (`px >> 4` inline when `bins` is a power of two; otherwise a 256-byte LUT) -> zero allocations.
- **`uint16` counters** (N=256 <= 65535) to stay in L1: 64 cells x 16 bins x 2 B = 2 KB.
- **Entropy via `c*log2(c)` LUT**: `H = log2(N) - (1/N) * sum(c*log2(c))` -> one `log2` per cell and a static table (257 entries for N=256). Keep the **equiprobable rule** (all counts equal => H = 1.0) and `N = 0 => 0.0` to match the oracle.
- **SIMD (AVX2)**: only for the pyramid sums and LUT reduction; scatter-style histogramming does not vectorize well, so do not promise SIMD speedup there.
- **Multithreading**: only for large batches (`#pragma omp parallel for` over images); batch=1 stays single-thread (CPU-fair protocol of task_08). Measure, but do not base conclusions on it.
- **Numba**: not installed and not allowed without an explicit request; mention only as an alternative.
- **Fewer parameters than PyTorch**: descriptor + LR has ~95 parameters vs 1.1k-97k for the CNNs; the lab reports `T_desc`, `T_model` and `T_total` separately (Performance claims rule).

## 2. Deliverables (all new; do not modify `src/`)

```
configs/optimization/entropy_maps.toml            # NEW: the only place with hardcoded values
understanding/13_entropy_maps_optimization/
    README.md                                      # what it is, how to run, MEASURED findings (filled after running)
    native/entropy_pyramid.c                       # C kernel (C99, no dependencies, no malloc in the hot path)
    native/entropy_pyramid.h
    build_native.py                                # locates vswhere/vcvars64, compiles cl.exe /O2 /LD -> native/build/*.dll
    variants.py                                    # V0..V2 behind a common interface (Strategy)
    native_binding.py                              # ctypes loading, fixed argtypes, preallocated buffers (V3)
    explain_single_pass.py                         # step-by-step lab: image -> fine hist -> pyramid sum -> H, with matplotlib
    benchmark_entropy_variants.py                  # CLI: correctness + latency + sweeps -> CSV/JSON
    plot_entropy_optimization.py                   # CLI: scientific plots from the CSVs
    tests/test_entropy_variants.py                 # unittest: parity against V0 and the oracle
    outputs/                                       # CSV, JSON, PNG (DLL goes in .gitignore)
```

`configs/optimization/entropy_maps.toml` (proposed schema; concrete values live in the TOML, not in code):

```toml
[entropy_optimization]
entropy_config = "configs/entropy/mvtec_ad.toml"     # reuses existing grid_levels and bins
base_config    = "configs/base.toml"
seed = 42
variants = ["V0_baseline", "V1_numpy_finest_pyramid", "V2_numpy_fused_entropy", "V3_c_single_pass"]
reference_variant = "V0_baseline"

[entropy_optimization.data]
source = "real"            # "real" (preprocessed images) | "synthetic"
n_real_samples = 200
image_size = 128

[entropy_optimization.native]
source_dir = "understanding/13_entropy_maps_optimization/native"
build_dir  = "understanding/13_entropy_maps_optimization/native/build"
library_name = "entropy_pyramid"
compiler_flags = ["/O2", "/LD", "/arch:AVX2", "/fp:precise"]
vswhere_hint = "%ProgramFiles(x86)%/Microsoft Visual Studio/Installer/vswhere.exe"

[entropy_optimization.protocol]
threads = 1
batch_sizes = [1, 8, 32, 256]
warmup_iters = 20
timed_iters = 200
repeats = 3
disable_gc = true
tolerance_abs = 1e-12

[entropy_optimization.sweeps]
bins_list = [8, 16, 32, 64]
image_sizes = [128, 256, 512]        # sizes must be divisible by the finest grid

[entropy_optimization.plots]
output_dir = "understanding/13_entropy_maps_optimization/outputs"
dpi = 200
formats = ["png"]
```

The grid levels stay fixed (taken from `configs/entropy/*.toml`); the number of patches is NOT swept.

Reuse (import, do not copy) `src/profiling/timing.py` (robust timer + median/IQR/p95/bootstrap CI), `src/profiling/environment.py` (HW/SW capture) and `src.features.entropy_maps` as the oracle. Load TOML with `tomllib` (Python 3.11) and paths with `PROJECT_ROOT = Path(__file__).resolve().parents[2]`.

## 3. Execution stages (each verifiable)

**Stage 1 - Configuration and skeleton.** Create the TOML, the lab `README.md` with hypotheses H1-H2 below, module skeletons and the common interface:
`class EntropyVariant(Protocol): name: str; def prepare(self, image_size, grid_levels, bins) -> None; def run(self, images_u8: np.ndarray) -> np.ndarray  # [N,H,W] uint8 -> [N, total_features] float64`.
`prepare` isolates amortizable cost (plan/LUT/buffers) from per-image cost. Validate inputs and TOML at startup (fail fast, `SystemExit("error: ...")`).
- H1: reading each pixel once on the finest grid and summing children gives a large gain over the baseline.
- H2: a native C kernel adds a further large gain, mainly at batch=1.

**Stage 2 - Pedagogical lab (`explain_single_pass.py`, `understanding > optimization`).** With matplotlib: image -> quantized -> histogram of each 8x8 cell -> sum of 2x2 children -> 4x4 -> 2x2 -> 1x1 -> LUT -> entropy map per level. Assert that `hist_4x4 == sum(hist_8x8 children)` and that entropy matches the oracle. Log how many times each pixel is read: baseline = 4, single pass = 1.

**Stage 3 - NumPy variants (V0-V2)** in `variants.py` with parity tests. Document dimensions (`[N,g,h,g,w]`, `[N,cells,bins]`).

**Stage 4 - C kernel (V3).** Suggested signature:
`void entropy_pyramid_u8(const uint8_t* img, int n_images, int size, const int* levels, int n_levels, int bins, const uint8_t* qlut, const double* nlogn_lut, double* out_features);`
- Single pass: per image, `hist[finest_cell][bin]` in `uint16` on the stack (64x16 -> 2 KB; max size validated in `prepare`, no `malloc` in the hot path).
- Coarser levels by child sums; entropy with `H = (log2N - S/N)/log2(bins)` where `S = sum(c*log2(c))` (table), with the equiprobable rule and `N = 0`.
- Output in the same order as `feature_names` (level, row, column).
- Keep it **portable to embedded C**: C99, `stdint` types, no STL, no recursion, no dynamic allocation, tables passed in or built by an `init` function.
- `build_native.py`: locate MSVC with `vswhere`, run `vcvars64.bat`, compile `cl /O2 /LD`; if no compiler is found, **fail with a clear message** (C tests use `skipTest`, never pass falsely).
- `native_binding.py`: `ctypes.CDLL`, `argtypes/restype` set once, pointers via `arr.ctypes.data_as`, preallocated reusable output buffer for batch=1 (ctypes overhead, ~1-3 us, measured separately with an "empty call" test).

**Stage 5 - Correctness (before timing).** `tests/test_entropy_variants.py` (unittest, `.venv`): for each variant, `max|V - V0| <= tolerance_abs`, also against the canonical oracle in `src`. Cases: constant image (H=0), uniform image (H=1.0, equiprobable), a real image, `N=1`, `N=0` (empty), different `bins`, different `grid_levels`, `size` not divisible by the finest grid (must raise `ValueError` in the fast variants), invalid inputs. Every variant must pass before appearing in any plot.

**Stage 6 - Benchmark (`benchmark_entropy_variants.py`).** CPU-fair protocol from task_08: `gc.disable()`, `perf_counter_ns`, pinned threads (`threadpoolctl` + `torch.set_num_threads`), warmup, median/IQR/p95/bootstrap 95% CI, `repeats`, real data (`n_real_samples`) plus optional seeded synthetic data. Separate `T_prepare` from per-image descriptor time. Measurements:
1. Batch=1 latency and throughput at batch 8/32/256 per variant.
2. Sweeps: `bins` and `image_size` (grid levels fixed).
3. Speedup vs V0 with CI and Mann-Whitney U test (alpha=0.05) as in task_08.
4. Reference: CPU forward of `ds_cnn` and `ultra_tiny` (batch=1, 1 thread) via `src/models/cnn/benchmark.py` (`device = "cpu"`) to place the descriptor against PyTorch. Report `T_total = T_selector + T_model`, parameters and analytical FLOPs/MACs (explicit basis).
5. Peak memory (`tracemalloc`/`psutil`) per variant.
Outputs: `outputs/benchmark_latency.csv`, `benchmark_sweeps.csv`, `correctness.csv`, `environment.json`, `run_manifest.json` (seed, config, git hash, versions, compiler).

**Stage 7 - Scientific plots (`plot_entropy_optimization.py`, matplotlib).** From the CSVs, without recomputing:
1. Batch=1 latency bars (log scale) with 95% CI and speedup label vs V0.
2. Throughput (img/s) vs batch size.
3. Stacked stage breakdown (quantization / histogram / pyramid sum / entropy) for V0 vs V1/V2 vs V3.
4. Speedup vs `bins` and vs `image_size`.
5. Max absolute error vs oracle per variant (log axis; must be <= tolerance).
6. Violin/box of the latency distribution (200 samples) at batch=1.
7. Descriptor vs CPU CNN forward (`ds_cnn`, `ultra_tiny`): `T_desc` + `T_model` bars.
Style: labeled axes with units, legends, consistent scales, DPI from TOML, `matplotlib.use("Agg")`.

**Stage 8 - Honest conclusions.** Fill the lab `README.md` **only with measured numbers**: are H1/H2 confirmed? How much does batch=1 really improve? Does V2 beat V1? Is the PyTorch CPU forward beaten? Record negative results. State anything unverified (e.g. `cl.exe` failed to build, no AVX2).

**Stage 9 - Migration prep (no migration).** List in the README which pieces would move to `src/` (winning variant + binding) and which part of the C code is portable to embedded firmware. Do **not** move code to `src/` until validated and explicitly approved by the user.

## 4. Acceptance criteria (Definition of done)

- [x] `configs/optimization/entropy_maps.toml` is the only place with constants; no magic numbers or absolute paths in the new `.py`/`.c` files (in C, only structural constants derived from parameters).
- [x] DLL builds with MSVC via `build_native.py`; clear failure if no compiler.
- [x] `.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/tests/test_entropy_variants.py` passes; existing tests (`tests/test_entropy_maps.py`, full suite) still pass (`src/` untouched).
- [x] All variants match V0 within `tolerance_abs` before benchmarking.
- [x] CSV/JSON/PNG generated by the scripts with a reproducible manifest (seed, versions, flags, git hash).
- [x] README explicitly answers, with measured data: (a) does the finest-grid single pass improve over the baseline? (b) how much does C add? (c) is the PyTorch CPU time beaten with fewer parameters?
- [x] `requirements.txt` unchanged (no new dependencies); document it if one is ever added.

## 5. Verification commands (PowerShell from `code/`)

```powershell
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/build_native.py
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/explain_single_pass.py
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/tests/test_entropy_variants.py
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/benchmark_entropy_variants.py --config configs/optimization/entropy_maps.toml
.\.venv\Scripts\python.exe understanding/13_entropy_maps_optimization/plot_entropy_optimization.py --config configs/optimization/entropy_maps.toml
.\.venv\Scripts\python.exe -m unittest discover tests -v
```

> Existing labs are run by script path (the directory starts with digits, so it is not importable as a module); keep that convention, but each script still exposes `main(argv: list[str] | None = None) -> int`.

## 6. Risks

| Risk | Mitigation |
|---|---|
| `cl.exe` not found or fails to initialize | `vswhere` + `vcvars64.bat`; two BuildTools exist (2022 and 18); if it fails, report it and mark V3 as unverified; continue with V0-V2. |
| Rounding differences C vs NumPy | `double` in C, same LUT, same equiprobable rule; tolerance 1e-12. |
| `ctypes` overhead hides the gain at batch=1 | Measure the empty call; preallocated buffers and `argtypes`; report pure kernel and full call separately. |
| "No improvement" result (e.g. V2 over V1) | A valid result; report it as is. |
| Noisy measurements (Windows) | task_08 protocol (warmup, repeats, IQR, pinned threads, gc off); report CV%. |

## 7. Out of scope (other tasks)

- Fused C pipeline with asymmetry (x9) and LR dot product.
- Increasing the number of patches / integral histograms / sliding windows.
- PyTorch implementation of the descriptor.
