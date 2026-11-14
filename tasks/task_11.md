# C Migration

## Issue

- The main issue now is the way we need to migrate the entire pipeline into C code, for which  I need to start from getting the image, but i think this step we can do it in python and only consuming the image from C, because this is not the main focus of the research.
- I create an initial folder, or the working folder where you will put all the files is c_imp/, i placed there some files that i think are important, but not sure.
- so i want to have all the pipeline, in and understandable way, separate files or headers, i don't care, i want to understand everything.
- also i think a big issue is how i'm going to import the weights or the cnn pytorchs models, i think we need to create a whole infrastructure for this and for support the migration.
- I was thinking since this will be migrated to an arduino or esp32 that will be helpful to use q15 or q31:
```c
typedef int16_t q15_t;

q15_t q15_mul_round(q15_t a, q15_t b)
{
    int32_t prod = ((int32_t)a * b) + (1 << 14);
    if (prod >= 0x40000000) return 32767;
    return (q15_t)(prod >> 15);
}
```


> Remember do not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.


## Plan

> Don't execute anything, just plan it

- [x] Read AGENTS.md file to know the general rules of the project.
- [x] Create a detailed plan for this task and update it here to implement it later for an agent, so it needs to be detailed.
- [x] Update this file with the plan, stages and everything to resolve and implement the task.

---

# Implementation plan (for the implementing agent)

> Read first: `AGENTS.md`, `RESUME.md`, `src/README.md`, `understanding/13_entropy_maps_optimization/README.md`,
> `understanding/14_ml_classifiers_mcu/README.md`. This plan was written on 2026-11-13 from the state of
> `main` at commit `ba41189`. Work on a new branch `feat/c-migration`.
> Every stage ends with a **Definition of done** that must be verified by execution, not by inspection.
> If a stage cannot be verified (e.g. no MSVC found), state it explicitly in the final report.

## 0. Analysis of the current state (what already exists and must be reused)

| Piece of the pipeline | Python (formal, `src/`) | Existing C | Gap for this task |
|---|---|---|---|
| Image acquisition / preprocessing | `src/preprocessing/` → 128×128 `uint8` `'L'` PNG in `data/preprocessed/` | none | **Stays in Python** (decided by the issue). Python only *serializes* images for C. |
| `x85` entropy pyramid | `src/features/entropy_maps.py` (5 backends) | `src/features/native/entropy_pyramid.{h,c}` (C99, `float64`, MSVC DLL via `ctypes`) | Desktop-only DLL, `double` LUT of length `size²+1` (16 385 × 8 B = 128 KB), not MCU-friendly, no fixed point. |
| `x9` asymmetry | `src/features/asymmetry_maps.py` (NumPy) | none | Must be written in C (RESUME §7: 74 % of desktop descriptor latency). |
| Classical classifier | `LinearDotCoreClassifier` (`lr_dot`, `w_eff @ x + b_eff`, predict = `margin > 0`) in `src/models/descriptor_classifier.py` | Lab 14 kernels (`understanding/14_ml_classifiers_mcu/c_kernels/`, exploratory, `float32`) | **No model persistence**: classical models are re-fitted each run (no `.joblib`). Exporter must re-fit deterministically and cross-check against `results/classification/*/metrics_summary.json`. |
| CNNs | 7 architectures in `src/models/cnn/architectures/`, `state_dict` saved as `results/cnn/<dataset>/<variant>/<model>/checkpoint.pt`, calibrated threshold in `metrics_summary.json → threshold_calibration.selected_threshold` | none | Full weight-import infrastructure needed (graph extraction, BN folding, quantization, code generation, C kernels). |
| Build | `src/features/build_native.py` (MSVC via `vswhere` + `vcvars64`) | — | Toolchain discovery helpers are private (`_find_vswhere`, …). Reuse by extracting them into a shared module (Stage 9). |

`c_imp/` currently contains 4 **empty** placeholder files (`infra_pytorch.h`, `infra_sklearn.h`, `lut_bin_quant.h`,
`lut_shannon.h`, all 0 bytes, untracked). Their names are kept and given a precise role (see §2). Because they are
empty, relocating them loses nothing.

Operations used by the 6 exportable CNNs (read from the architecture files):
`Conv2d` (3×3 full, 3×3 depthwise `groups=Cin`, 1×1 pointwise/shortcut; stride 1/2; padding 1/0; `bias=False`),
`BatchNorm2d`, `ReLU`, `MaxPool2d(2)`, `AdaptiveAvgPool2d(1)`, `flatten`, `Linear`, residual `add` (`tiny_resnet`).
`torchvision_mobilenet` (MobileNetV3-Small) is **in scope** (revised 2026-11-13). Extra ops needed: Hardswish
(`x·relu6(x+3)/6`, piecewise, trivial in q15 with shifts), Hardsigmoid (`relu6(x+3)/6`), Squeeze-Excitation (global avg
pool → 1×1 conv → ReLU → 1×1 conv → Hardsigmoid → per-channel multiply, i.e. only one new kernel `cimp_scale_channels`
plus existing ones), depthwise 5×5 kernels, inverted-residual `add`, Dropout/Identity (removed at export, eval mode).
Real difficulty is not the ops but **size**: ~1.5 M params → ~3 MB in q15 (too much for the 4 MB flash with firmware
and OTA partitions) → this model requires `int8` weights (≈1.5 MB) with q15 activations (see §1 decision 11).
Models whose weights do not fit the flash budget in the TOML are rejected by the exporter with a measured size.

## 1. Key design decisions (justify them in `c_imp/README.md`)

1. **Python ↔ C boundary.** Python owns: reading PNGs, preprocessing, splits, training, exporting weights/LUTs/config,
   writing test vectors, and verifying parity. C owns: `uint8 image[H*W] → x94 → classifier → decision` and
   `uint8 image[H*W] → CNN → decision`. On desktop, C reads raw `.bin` images (exactly `H*W` bytes, row-major) listed in
   a manifest; on MCU the same function receives a pointer to a camera/flash buffer. C never parses PNG/TOML/JSON.
2. **No hardcoded numbers in C.** Every numeric constant (image size, grid levels, bins, feature count, Q formats,
   buffer sizes, thresholds, weights) is emitted by the Python exporter into generated headers
   (`cimp_config.h`, `lut_*.h`, `model_*.h`) from the TOML configuration. Hand-written runtime C files only contain
   algorithms that take sizes as arguments or `#include` the generated config. Paths reach the C executable only as
   CLI arguments supplied by the Python runner, which reads them from TOML.
3. **Single source of truth.** The new TOML must *reference* the existing configs (`configs/entropy/*.toml`,
   `configs/asymmetry/*.toml`, `configs/classification/*.toml`, `configs/cnn/*.toml`) instead of duplicating
   `image_size`, `bins`, `grid_levels`, `scales`, `kinds`, `normalize_mean/std`, `seed`. Reuse the existing loaders
   (`load_entropy_config`, `load_asymmetry_config`, `load_classification_config`, `load_cnn_config`).
4. **Two numeric paths, same code structure.**
   - `float32` reference path: proves the C port is algorithmically correct (parity vs. Python `float64`/PyTorch).
   - `q15` fixed-point path (`int16` storage, `int32`/`int64` accumulators, `q31` where range demands it):
     the MCU target. Selected at compile time with `CIMP_NUMERIC_Q15` / `CIMP_NUMERIC_F32`.
   A **bit-exact Python model of the q15 path** (`src/c_export/fixed_point.py`) is the golden reference for C
   (C q15 output must be *identical*), while the float64 `src/` pipeline measures the *accuracy cost* of quantization.
5. **Power-of-two (Qm.n) scales only** (shifts, no float at runtime), chosen per tensor by the exporter from observed
   ranges. This matches the issue's `q15_mul_round` idea and keeps the C readable. (Affine int8 / CMSIS-NN / ESP-NN
   are documented as future optimized backends, not implemented now.)
6. **Readable over clever.** CNN code generation emits a straight-line `model_<name>_forward()` that calls one kernel
   per layer with named buffers, so a reader can follow the network line by line against the PyTorch class.
7. **Memory model.** C99, no `malloc`, no VLAs (MSVC also lacks them), no `<math.h>` in the q15 runtime, static
   arena buffers whose sizes are computed by the exporter (liveness of activations), `const` weights (flash on MCU).
8. **Thresholds without `exp`.** For 2-class softmax, `p1 ≥ τ ⇔ z1 − z0 ≥ logit(τ)`. The exporter precomputes
   `logit(τ)` (CNN, from `selected_threshold`) or `0` (`lr_dot`, `margin > 0`) in the logits' Q format.
10. **No DLLs, no `ctypes`, no OS dependency in the runtime (embedded-first).** `c_imp/runtime/` is plain C99 source:
   no `malloc`, no `<stdio.h>`/file I/O, no OS headers except inside `cimp_platform.c`, no `__declspec`. The legacy
   `src/features/native/*.dll` stays untouched as a *desktop benchmark backend* only; it is not part of this migration.
   The desktop `.exe` is merely a test harness that links the same runtime sources statically; a future ESP32 firmware (separate task) will compile the very same files.
11. **Target = ESP32-CAM (AI-Thinker): ESP32 Xtensa LX6 dual core 240 MHz, 520 KB SRAM (~320 KB usable as static
   DRAM), 4 MB flash, 4 MB PSRAM (8 MB on some boards), OV2640.** Consequences that drive the numeric design:
   - The classic ESP32 has a **single-precision hardware FPU** and **no SIMD** (SIMD exists only on ESP32-S3).
     Therefore `float32` may be as fast as, or faster than, `q15` for compute; fixed point mostly buys **memory**
     (2× smaller weights/activations). We do not guess: both paths are kept and **measured on the device**.
   - `int64` accumulation is multi-instruction on LX6. q15×q15 products need up to 30 bits, so a plain `int32`
     accumulator overflows after 2 terms. Policy: `int32` accumulator with **pre-shifted products** only where the exporter
     proves (worst-case bound from actual weights + calibrated ranges) that overflow is impossible; otherwise `int64`.
     The exporter records which layers use which accumulator.
   - **Mixed precision policy (answer to "q15, q31 or mixed")**: activations and features `q15` (`int16`); weights `q15`
     by default, `int8` per-output-channel for layers/models that exceed the flash budget; `q31`/`int32` only for
     bias, LUT entries needing >16 bits (e.g. `c·log2 c`) and requantization multipliers. Selection is automatic per tensor
     from observed ranges and reported; the TOML only sets the allowed formats and budgets.
   - PSRAM (external, SPI, slow, not DMA-capable for some paths) is allowed for big activation buffers via a
     `CIMP_ARENA_IN_PSRAM` placement macro; weights are `const` in flash (memory-mapped, cached). Budgets
     (`flash_budget_bytes`, `sram_budget_bytes`, `psram_budget_bytes`) live in the TOML and are enforced by the exporter.
9. **Timing honesty.** C reports `T_descriptor` (selector/feature stage), `T_model`, `T_total = T_descriptor + T_model`
   per image, labelled **desktop x86** — never presented as MCU latency.

### 1.1 Numerical design of the q15 descriptor (to be validated in Stage 1)

- Quantization of pixels: `q = qlut[p]` with `qlut[p] = (p * B) / 256` (256 B, `lut_bin_quant.h`). Same formula as `src`.
- Histograms: counts are exact integers. Finest cell (8×8 grid on 128 px → 16×16 = 256 px) needs `uint16`
  (256 does not fit `uint8`); the exporter selects the count type from `max cell pixels` (≤ 65 535 → `uint16`, else `uint32`).
- Normalized entropy per cell: `E = (log2 N − (1/N) Σ c·log2 c) / log2 B`. Three LUT strategies, chosen in TOML
  (`[c_imp.descriptor].shannon_lut_mode`), all generated into `lut_shannon.h`:
  - `nlogn_full`: `c·log2 c` for `c ∈ [0, N_max]` (port of current C kernel; 16 385 entries; Q format chosen by exporter,
    e.g. `uint32` with `31 − ceil(log2(max))` fractional bits). Reference, desktop-friendly, ~64 KB.
  - `term_per_level`: `−(c/N)·log2(c/N)/log2 B` per level in q15 (sum of `N+1` over levels, ~44 KB for `[1,2,4,8]`).
  - `log2_mantissa`: `c·log2 c = c·(e + log2(1 + m))`, `e` from leading-zero count, `log2(1+m)` from a
    `2^k + 1` entry LUT (`k` in TOML; `k = 8` → 257 entries ≈ 0.5–1 KB). MCU default. Error must be **measured**.
- Representation of `E ∈ [0, 1]` in q15: `1.0` is not representable (max `32767/32768`). Decision: saturate to `32767`
  (error 3.05e‑5 for exactly equiprobable cells) and document; alternative `Q1.14` exposed as TOML option
  `feature_frac_bits` (default 15).
- Asymmetry `A_T(g) = Σ|E − T(E)| / g²`: q15 differences, `int32` sum, division by `g²` as a shift when `g²` is a power
  of two, else multiplication by an exporter-computed q15 reciprocal. Stays in `[0, 1]`.

### 1.2 Memory budget (back-of-envelope, **must be recomputed and reported by the exporter**)

Descriptor: image 16 384 B; finest histograms 64 × 16 × 2 B = 2 048 B; coarse 21 × 16 × 2 B = 672 B; `x94` 188 B (q15).

CNN peak activations (input + output of the largest layer, elements → q15 bytes), arithmetic from the architectures:

| Model | Largest live tensors | ≈ q15 peak | Comment |
|---|---|---|---|
| `ultra_tiny` | 1×128×128 + 8×64×64 | ~96 KB | fits ESP32 internal SRAM (520 KB) |
| `mobilenet_like` | 16×64×64 + 32×64×64 | ~384 KB | tight on ESP32 |
| `tiny_resnet` | 3 × 16×64×64 (input, residual, branch) | ~384 KB | tight on ESP32 |
| `ds_cnn`, `tiny_cnn`, `small_cnn` | 16×128×128 + next | ~576–640 KB | needs PSRAM, or conv+ReLU+MaxPool fusion (`tiny_cnn`, `small_cnn`) |

Target fixed by the user: **ESP32-CAM (4 MB flash, 520 KB SRAM, 4 MB PSRAM)**. Models above ~250 KB peak activations go
to PSRAM (slower) or use fused `conv+relu(+maxpool)`; exporter reports the real numbers. `torchvision_mobilenet`
needs int8 weights (~1.5 MB flash). OV2640 delivers GRAYSCALE up to its configured frame size (e.g. QVGA 320×240),
not 128×128: on-device preprocessing is deferred to the future firmware task and its
difference vs. the Python `LANCZOS` letterbox must be measured, not assumed.

## 2. Target file layout

```text
code/
├── configs/c_imp/
│   ├── kolektorsdd2.toml            NEW: C-export + verification config (references existing configs)
│   └── mvtec_ad.toml                NEW: same for MVTec-AD (object selectable via CLI)
├── src/c_export/                    NEW formal Python package (export + verification infrastructure)
│   ├── __init__.py
│   ├── config.py                    load/validate configs/c_imp/*.toml → frozen dataclasses
│   ├── c_writer.py                  deterministic C header/source emitter (arrays, guards, provenance)
│   ├── fixed_point.py               Qm.n quantization + bit-exact Python mirror of every C q15 op
│   ├── luts.py                      bin-quant LUT, 3 Shannon LUT modes, CNN input-normalization LUT
│   ├── descriptor_reference.py      bit-exact q15 x94 reference (Python mirror of c_imp/runtime/cimp_descriptor.c)
│   ├── linear_export.py             re-fit lr_dot via src API, cross-check metrics, quantize, emit model_lr_dot.h
│   ├── cnn_graph.py                 checkpoint → torch.fx graph → validated layer IR (op whitelist) + BN folding
│   ├── cnn_quantize.py              calibration of activation ranges, frac-bit selection, weight/bias quantization
│   ├── cnn_codegen.py               IR → model_<name>.h (weights) + model_<name>.c (straight-line forward)
│   ├── memory_plan.py               activation liveness → arena sizes; flash/SRAM estimate report
│   ├── test_vectors.py              PNG → .bin + manifest.csv + golden outputs (float64 & bit-exact q15)
│   ├── toolchain.py                 shared MSVC discovery (extracted from src/features/build_native.py)
│   ├── build_c.py                   compile c_imp runtime + generated + app into an executable
│   └── parity.py                    run executable, compare outputs, metrics, timings → results/c_imp/
├── experiments/
│   ├── export_c_pipeline.py         NEW CLI: export config/LUTs/models/test vectors for one target
│   └── verify_c_pipeline.py         NEW CLI: build + run + parity + timing report
├── c_imp/                           C side (the user's working folder)
│   ├── README.md                    full explanation of the C pipeline (data flow, Q formats, how to build/run)
│   ├── runtime/                     HAND-WRITTEN portable C99 (never generated)
│   │   ├── cimp_fixed_point.h       q15_t/q31_t, q15_mul_round (issue snippet), sat add/sub, round-shift, clamp
│   │   ├── cimp_tensor.h            tensor views {data, c, h, w, frac_bits} in [C, H, W] layout
│   │   ├── cimp_platform.h/.c       timer abstraction (Windows QPC / POSIX clock_gettime / MCU cycle counter hook)
│   │   ├── cimp_descriptor.h/.c     quantize → finest hist → pyramid sums → entropy (x85) → asymmetry (x9) → x94
│   │   ├── infra_sklearn.h/.c       classical classifier kernels: dense dot (lr_dot) in f32 and q15
│   │   ├── infra_pytorch.h/.c       CNN layer kernels (conv, dwconv, pwconv, relu, maxpool, gap, linear, add)
│   │   └── cimp_pipeline.h/.c       public API: cimp_run_descriptor_classifier(), cimp_run_cnn()
│   ├── generated/<target_tag>/      PRODUCED by exporter (gitignored, reproducible by one command)
│   │   ├── cimp_config.h            all sizes/levels/bins/frac bits/thresholds from TOML + calibration
│   │   ├── lut_bin_quant.h          256-entry pixel → bin LUT
│   │   ├── lut_shannon.h            Shannon LUT (mode from TOML)
│   │   ├── lut_cnn_input.h          256-entry pixel → normalized q15 CNN input
│   │   ├── model_lr_dot.h           w_eff[94], b_eff, frac bits, decision threshold
│   │   ├── model_<cnn>.h            folded + quantized weights/biases per layer
│   │   ├── model_<cnn>.c            generated straight-line forward pass
│   │   └── export_manifest.json     provenance: configs, SHA-256 of checkpoint, git commit, Q formats, sizes
│   ├── app/
│   │   └── desktop_main.c           CLI: --manifest --mode {descriptor,classifier,cnn} --model --out
│   └── tests/
│       └── test_runtime.c           C unit tests (q15 ops, crafted histograms, tiny conv vs known values)
├── tests/
│   ├── test_c_export_config.py
│   ├── test_c_export_fixed_point.py
│   ├── test_c_export_luts.py
│   ├── test_c_export_linear.py
│   ├── test_c_export_cnn_graph.py
│   └── test_c_pipeline_parity.py    builds + runs C (skipped with explicit message if MSVC is unavailable)
├── understanding/15_c_migration_fixed_point/   NEW lab (understanding-first rule)
└── results/c_imp/<target_tag>/       parity, accuracy, timing, memory reports (gitignored)
```

Placeholders: `c_imp/infra_pytorch.h` → `c_imp/runtime/infra_pytorch.h`, `c_imp/infra_sklearn.h` →
`c_imp/runtime/infra_sklearn.h`; `lut_bin_quant.h` and `lut_shannon.h` become **generated** files in
`c_imp/generated/<target_tag>/` (remove the empty top-level placeholders when the generator exists).

`.gitignore` additions: `c_imp/build/`, `*.exe`, `*.pdb`. (`results/*` is already ignored.)

C conventions (add a short section to `c_imp/README.md`): C99, prefix `cimp_`, fixed-width integer types,
doc comment on every public function with shapes (`[C, H, W]`, `[Cout, Cin/groups, kH, kW]`, `[N_FEATURES]`) and Q
formats, return `int` status codes (`0` ok, negative = error) for public entry points, no globals except `const`
tables and the static arena in generated code.

## 3. Configuration (`configs/c_imp/kolektorsdd2.toml`, sketch)

```toml
# C99 pipeline export & verification for KolektorSDD2. All C numeric constants derive from this file
# or from the configs it references. Relative paths are anchored to the repository 'code/' root.

[c_imp]
target_tag = "kolektorsdd2"
seed = 42
inverted = false
entropy_config = "configs/entropy/kolektorsdd2.toml"          # image_size (via preprocessing), bins, grid_levels
asymmetry_config = "configs/asymmetry/kolektorsdd2.toml"      # scales, kinds, schema
classification_config = "configs/classification/kolektorsdd2.toml"
cnn_config = "configs/cnn/kolektorsdd2.toml"                  # normalize_mean/std, splits, results_dir

[c_imp.paths]
runtime_dir = "c_imp/runtime"
app_dir = "c_imp/app"
c_tests_dir = "c_imp/tests"
generated_dir = "c_imp/generated"     # + /<target_tag>
build_dir = "c_imp/build"
results_dir = "results/c_imp"         # + /<target_tag>[-inverted]

[c_imp.numeric]
modes = ["f32", "q15"]                # builds/verifies both
feature_frac_bits = 15                # E, A in [0,1]; 1.0 saturates to 32767
accumulator_bits = 64                 # int64 accumulators in reference kernels
weight_storage_bits = 16
rounding = "half_up"                  # matches q15_mul_round (+ 1 << (shift-1))

[c_imp.descriptor]
shannon_lut_mode = "log2_mantissa"    # "nlogn_full" | "term_per_level" | "log2_mantissa"
log2_mantissa_bits = 8                # 2^8 + 1 = 257 entries

[c_imp.classifier]
models = ["lr_dot", "mlp_16_dot"]    # mlp_16_dot promoted to src/ in Stage 7A
metrics_crosscheck_dir = "results/classification"
metrics_crosscheck_abs_tol = 1e-9

[c_imp.cnn]
models = ["ultra_tiny", "ds_cnn", "tiny_cnn", "mobilenet_like", "tiny_resnet", "small_cnn", "torchvision_mobilenet"]
allowed_weight_formats = ["q15", "int8_per_channel"]   # per-model choice automatic from flash budget
flash_budget_bytes = 2097152          # weights budget on a 4 MB ESP32-CAM; provisional, set from partition table
sram_budget_bytes = 200000            # static DRAM available for arena; provisional
psram_budget_bytes = 3000000          # provisional

checkpoint_variant = "full"           # results/cnn/<dataset>/<variant>/<model>/checkpoint.pt
calibration_split = "train"
calibration_max_images = 256
activation_range_percentile = 100.0   # 100 = max-abs (no clipping)
weight_granularity = "per_tensor"     # "per_tensor" | "per_output_channel"
input_frac_bits = 15                  # normalized input in [-1, 1)

[c_imp.test_vectors]
split = "test"
max_images = 0                        # 0 = whole split
embed_headers_count = 4               # images also emitted as C headers for MCU smoke tests

[c_imp.build]
compiler = "msvc"
c_standard_flag = "/std:c11"
compiler_flags = ["/O2", "/W4", "/fp:precise", "/nologo"]
vswhere_hint = "%ProgramFiles(x86)%/Microsoft Visual Studio/Installer/vswhere.exe"
executable_name = "cimp_pipeline"

[c_imp.verification]
f32_descriptor_abs_tol = 1e-5
f32_logit_abs_tol = 1e-4
q15_must_be_bit_exact_vs_python_model = true
q15_descriptor_report_only = true     # accuracy cost is reported, never forced
timing_warmup_iters = 50
timing_timed_iters = 300
```

`configs/c_imp/mvtec_ad.toml`: same structure with the MVTec configs and `object_name = "bottle"` default
(overridable by `--object`). Every value above that does not come from a referenced config is the only place where it
is written. Values marked as tolerances are **provisional acceptance gates**; if they fail, report the measured value
and discuss — do not silently loosen them.

## 4. Stages

### Stage 0 — Branch, baseline and conventions
1. `git checkout -b feat/c-migration`.
2. Run the baseline: `.\.venv\Scripts\python.exe -m unittest discover tests -v` and
   `.\.venv\Scripts\python.exe tests/test_environment.py`; record pass counts (expected 208 + CUDA check, per RESUME).
3. Confirm that canonical artifacts exist: `results/cnn/kolektorsdd2/full/<model>/checkpoint.pt` + `metrics_summary.json`,
   `results/classification/kolektorsdd2/full_lr_dot/metrics_summary.json`, `data/preprocessed/KolektorSDD2/`.
   If missing, stop and report the commands from `tasks/task_10.md` §2 needed to regenerate them.

**Done when:** baseline test counts recorded; artifact check documented.

### Stage 1 — Understanding lab `understanding/15_c_migration_fixed_point/` (understanding-first rule)
Small, didactic scripts with matplotlib figures and `tests/`:
1. `01_q_formats.py`: Qm.n representation, `q15_mul_round` from the issue (including the `-32768 × -32768` saturation
   case and the rounding of negative numbers), saturation vs. wrap-around, quantization error histograms.
2. `02_entropy_fixed_point.py`: walk `image → patch → histogram → counts → p → −p·log2 p → E` in float64 and in the
   three LUT modes of §1.1; plot error per cell and per level; check top-k ranking preservation (Lab 03 precedent).
3. `03_bn_folding.py`: fold `BatchNorm2d` into the preceding conv (`w' = w·γ/√(σ²+ε)`, `b' = β − γ·μ/√(σ²+ε)`),
   verify on a real checkpoint that the folded conv equals conv+BN in eval mode (tolerance documented).
4. `04_requantization.py`: one conv layer in float vs. q15 with int64 accumulator and output shift; visualize
   error and overflow headroom as a function of `Cin·k²`.
5. Config: add a section to a new `configs/understanding/c_migration_fixed_point.toml` (seeds, image samples, paths).

**Done when:** lab tests pass, figures saved under the lab's `outputs/`, lab `README.md` states measured errors and
which LUT mode/frac bits are recommended for Stage 4 (this decision then goes into `configs/c_imp/*.toml`).

### Stage 2 — Configuration and loader (`configs/c_imp/*.toml`, `src/c_export/config.py`)
1. Create both TOML files (§3).
2. `config.py`: `@dataclass(frozen=True, slots=True)` classes `CImpPaths`, `CImpNumeric`, `CImpDescriptor`,
   `CImpClassifier`, `CImpCnn`, `CImpTestVectors`, `CImpBuild`, `CImpVerification`, `CImpConfig`;
   `load_c_imp_config(config_path: Path, project_root: Path, object_name: str | None = None, inverted: bool | None = None) -> CImpConfig`.
   - Resolve referenced configs with existing loaders and expose a derived `DescriptorSpec(image_size, bins,
     grid_levels, asym_scales, asym_kinds, schema)`.
   - Fail fast: missing keys, unknown LUT mode, non-nested grid levels (C kernel requires nesting, like `c_single_pass`),
     `image_size % max(levels) != 0`, `bins ∉ [2,256]`, unsupported CNN listed in `models`, frac bits out of range.
3. Tests `tests/test_c_export_config.py`: valid load, each fail-fast branch, path anchoring independent of CWD.

**Done when:** tests pass; loader used by every later stage (no other place reads these TOMLs).

### Stage 3 — C core runtime and C writer
1. `c_imp/runtime/cimp_fixed_point.h` (header-only `static inline`): `q15_t`, `q31_t`, `cimp_q15_mul_round`
   (the issue's function, renamed with prefix, comment explaining each line), `cimp_sat16(int64)`,
   `cimp_sat32(int64)`, `cimp_round_shift_right_i64(x, s)`, `cimp_q15_add_sat`, `cimp_clz32` (portable fallback +
   MSVC `_BitScanReverse` / GCC `__builtin_clz` when available).
2. `cimp_tensor.h`: `typedef struct { int16_t *data; int32_t c, h, w; int8_t frac_bits; } cimp_tensor_q15;` and the
   `float` variant; index helper `CIMP_IDX(c, y, x, h, w)`.
3. `cimp_platform.h/.c`: `uint64_t cimp_time_ns(void)` with `_WIN32` / POSIX branches and a weak MCU hook.
4. `src/c_export/c_writer.py`: `write_c_array(name, values, c_type, per_line)`, `write_header(path, guard,
   includes, body, provenance)`, deterministic formatting (no timestamps in content; provenance = config paths,
   SHA-256 of inputs, git commit), atomic write (`.tmp` → final), mirroring `src` conventions.
5. `src/c_export/fixed_point.py`: `choose_frac_bits(max_abs, storage_bits, headroom_bits=0)`, `quantize(x, frac, bits)
   -> (ints, n_saturated)`, `dequantize`, and NumPy mirrors of every C op (`q15_mul_round`, `round_shift`, `sat16`)
   that are **bit-exact** with the C semantics (document arithmetic right shift of negatives).
6. Tests `tests/test_c_export_fixed_point.py` (exhaustive over all `int16` pairs for `q15_mul_round` against a pure
   Python integer reference; edge cases) and C tests in `c_imp/tests/test_runtime.c` for the same cases.

**Done when:** Python tests pass; `test_runtime.c` compiled and run in Stage 9 (mark as pending until then).

### Stage 4 — LUT generation (`src/c_export/luts.py` → `lut_bin_quant.h`, `lut_shannon.h`, `lut_cnn_input.h`)
1. `build_bin_quant_lut(bins)` must reuse `src.features.entropy_maps.build_quantization_lut` and assert equality.
2. `build_shannon_lut(spec, mode, numeric)` for the three modes; returns arrays + chosen Q formats + byte size.
3. `build_cnn_input_lut(mean, std, frac_bits)`: `round(((p/255) − mean)/std · 2^frac)` saturated; mean/std from
   `configs/cnn/*.toml` (`normalize_mean`, `normalize_std`) via `load_cnn_config`.
4. Emit `cimp_config.h` with: `CIMP_IMAGE_SIZE`, `CIMP_N_LEVELS`, `CIMP_LEVELS[]`, `CIMP_BINS`, `CIMP_FINEST_GRID`,
   `CIMP_N_X85`, `CIMP_N_X9`, `CIMP_N_X94`, asymmetry scales/kinds tables, count type, Shannon mode macro,
   feature frac bits, reciprocal constants for non-power-of-two divisions, schema string `"x94-v1"`.
5. Tests `tests/test_c_export_luts.py`: equality with `src` LUTs, monotonicity/limits, deterministic bytes (generate
   twice → identical SHA-256).

**Done when:** headers are generated deterministically for both targets and tests pass.

### Stage 5 — C descriptor `x94` (`c_imp/runtime/cimp_descriptor.{h,c}` + Python q15 mirror)
Implement as small functions, each documented with shapes, in both `f32` and `q15`:
1. `cimp_quantize_and_fine_hist(const uint8_t *img, int32_t size, int32_t g_max, int32_t bins, const uint8_t *qlut,
   uint16_t *fine_hist /* [g_max*g_max, bins] */)` — single pass over pixels (port of `entropy_pyramid_stage_fine_hist_u8`).
2. `cimp_pyramid_sum(...)` — child summation to coarser levels in caller level order (port of `entropy_pyramid_stage_sum`).
3. `cimp_entropy_cells_f32/q15(...)` — per-cell normalized entropy with the selected LUT mode, equiprobable rule
   identical to `src` (`E = 1` exactly in f32; saturated in q15).
4. `cimp_asymmetry_f32/q15(const E_by_level, scales, kinds, out_x9)` — `lr`, `tb`, `rot180` exactly as
   `src/features/asymmetry_maps.py` (reconstruct row-major grids, mean absolute residual, output order of `x94-v1`).
5. `cimp_descriptor_x94_f32/q15(const uint8_t *img, workspace*, out_x94)` — orchestrator; workspace is a struct with
   arrays sized by `cimp_config.h` (stack/static, zero heap).
6. `src/c_export/descriptor_reference.py`: integer-exact Python mirror of steps 1–5 in q15 (same LUTs, same shifts).
   Validate it first against `src.features.entropy_maps.compute_entropy_features` + `compute_asymmetry_features`
   (accuracy report), then C must equal it bit for bit.

Notes: do **not** modify `src/features/native/entropy_pyramid.c` (it remains the desktop `ctypes` backend used by
the formal benchmarks); the new C file is a portable re-implementation with comments linking each block to its
counterpart.

**Done when (verified in Stage 9):** f32 C vs. `src` float64 max |Δ| ≤ `f32_descriptor_abs_tol`; q15 C ≡ Python q15
mirror (bit-exact); q15 accuracy cost (max/mean |Δ| per feature group) reported.

### Stage 6 — Image ingestion & test vectors (`src/c_export/test_vectors.py`)
1. Load the evaluation split with the **same** logic as the baselines: use `src.models.cnn.data.load_cnn_partition`
   (already aligned with the classical split via metadata index) for images + labels + file names.
2. Write `results/c_imp/<target>/vectors/images/<index:05d>.bin` (raw `H*W` uint8, row-major) and `manifest.csv`
   (`index, file_name, split, label, bin_relpath`), SHA-256 per image.
3. Golden outputs: `golden_x94_f64.csv` (from `src`), `golden_x94_q15.csv` (from the mirror), later
   `golden_lr_dot.csv` and `golden_cnn_<model>.csv` (Stages 7–8).
4. Emit `embed_headers_count` images as `c_imp/generated/<target>/test_image_<k>.h` for future MCU smoke tests.

**Done when:** manifest row count equals the split counts reported in the existing `metrics_summary.json`
(e.g. KolektorSDD2 test: 894 good / 110 defect); round-trip `.bin` → array equals PNG pixels.

### Stage 7 — Classical classifier export (`src/c_export/linear_export.py`, `runtime/infra_sklearn.{h,c}`)
1. Re-fit `lr_dot` exactly as the formal pipeline does: `load_classification_config` → `load_and_split_descriptors`
   → `build_classifier("lr_dot", cfg)` → `fit` (seed 42). Read `w_eff_`, `b_eff_`.
2. Cross-check: recompute test metrics and compare with `results/classification/<dataset>[-inverted]/full_lr_dot/
   metrics_summary.json` (`metrics_crosscheck_abs_tol`); abort on mismatch (fail fast, scientific integrity).
3. Quantize: q15 inputs (`feature_frac_bits`), weights with exporter-chosen frac bits (`w_eff` can be large because of
   the folded `1/scale`: choose bits from `max|w_eff|`; if q15 precision is insufficient, store `q31` and document),
   bias aligned to `feature_frac + weight_frac`, decision threshold `0`.
4. Emit `model_lr_dot.h`; kernels `cimp_dense_dot_f32` and `cimp_dense_dot_q15` (int64 accumulator) in
   `infra_sklearn.c` with comments relating to `LinearDotCoreClassifier.decision_function`.
5. Golden margins/predictions to `golden_lr_dot.csv`.
6. **Stage 7A — promote `mlp_16_dot` to `src/models/`** (new module, e.g. `src/models/mlp_dot.py`, reimplemented from
   `understanding/14_ml_classifiers_mcu/models.py` — do not import from `understanding/`): scaler-folded 1-hidden-layer
   (16, ReLU) MLP, hyperparameters from `configs/classification/*.toml` (new `[models.mlp_16_dot]` section copied from
   `configs/understanding/ml_classifiers_mcu.toml`), registered in `build_classifier`, with tests and a parity test against the
   Lab 14 implementation (same seed → same predictions). Then export it like `lr_dot` (`model_mlp_16_dot.h`, kernel
   `cimp_mlp_1hidden_*` in `infra_sklearn.c`, port of `mcu_mlp_1hidden_score`), threshold from validation as in Lab 14.

**Done when:** tests `tests/test_c_export_linear.py` pass (fold parity, quantization error bound, cross-check
failure path); in Stage 9 f32 C predictions agree 100 % with Python and q15 agreement/F1 are reported.

### Stage 8 — CNN weight-import infrastructure (the core of the issue)
Pipeline: `checkpoint.pt → nn.Module → torch.fx graph → layer IR (JSON) → BN folding → calibration → quantization →
memory plan → C code generation`.

1. **`cnn_graph.py`**
   - `load_trained_model(cfg, model_name) -> nn.Module`: `create_model(name, num_classes)` from the registry,
     `load_state_dict(torch.load(path, map_location="cpu", weights_only=True))`, `eval()`; record SHA-256 of the file.
   - `trace_to_ir(model, input_shape) -> list[LayerSpec]` using `torch.fx.symbolic_trace` + `ShapeProp`.
     `LayerSpec` (frozen dataclass): `name, op, inputs, output, in_shape, out_shape, stride, padding, groups,
     kernel, weight_key, bias_key`. Whitelist: `conv2d` (groups 1 or depthwise, kernel 1/3/5), `batchnorm2d`, `relu`, `hardswish`, `hardsigmoid`, `maxpool2d(k=2,s=2)`, `adaptive_avgpool2d(1)`, `flatten`, `linear`, `add`, channel-wise `mul` (SE), `dropout` (removed). Anything else -> `UnsupportedLayerError` naming the module. `torchvision_mobilenet` requires tracing through torchvision `SqueezeExcitation`/`InvertedResidual` (custom fx leaf rules); verify IR vs `model(x)`. New kernels: `cimp_hardswish`, `cimp_hardsigmoid`, `cimp_scale_channels`, 5x5 depthwise. Bring this model up last.
   - `fold_batchnorm(ir, state) -> ir'`: every `conv → bn` pair becomes one conv with bias; assert BN follows a conv
     and that the conv output is used only by that BN.
   - Validation: run the folded IR with a small NumPy/PyTorch float interpreter and compare with `model(x)` on real
     images (tolerance `f32_logit_abs_tol`).
2. **`cnn_quantize.py`**
   - Calibration on `calibration_split` (`calibration_max_images`, seeded): record max-abs (or percentile) per
     activation tensor using the float interpreter.
   - Frac bits per activation (`choose_frac_bits`), per weight tensor (or per output channel), bias in
     `int32/int64` at `w_frac + x_frac`, output shift `w_frac + x_frac − y_frac` per layer; residual `add` aligns
     both inputs to a common frac.
   - Bit-exact Python q15 interpreter of the IR (mirror of C kernels) → golden q15 logits.
   - Threshold: `selected_threshold` from `metrics_summary.json` → `logit(τ)` in the logits' Q format.
3. **`memory_plan.py`**: liveness analysis over the IR; assign ping-pong buffers + extra buffer for residual
   branches; output arena sizes, total weight bytes (flash), peak activation bytes (SRAM), per-layer MACs. Write
   `memory_report.json` (this replaces the estimates in §1.2 with computed values).
4. **`cnn_codegen.py`**: emit
   - `model_<name>.h`: `static const int16_t L03_conv_w[Cout*Cin/g*kH*kW]` (layout `[Cout, Cin/groups, kH, kW]`,
     same as PyTorch OIHW), biases, per-layer shifts, input/output frac bits, arena sizes, `CIMP_<NAME>_LOGIT_THRESHOLD`.
   - `model_<name>.c`: `int cimp_<name>_forward_q15(const uint8_t *img, int32_t *out_logits)` and the `f32` twin,
     one kernel call per IR node with the PyTorch module path in a comment (e.g. `/* block1.depthwise + depthwise_bn */`).
5. **C kernels `runtime/infra_pytorch.{h,c}`** (f32 + q15, reference loops, int64 accumulators, rounding + saturation):
   `cimp_conv2d`, `cimp_dwconv2d`, `cimp_relu`, `cimp_maxpool2d_2x2`, `cimp_global_avgpool`, `cimp_linear`,
   `cimp_add`, `cimp_input_from_u8` (uses `lut_cnn_input.h`). Optional fused `conv+relu(+maxpool)` to lower peak
   memory, behind a TOML flag, only after the unfused version is verified.
6. Order of bring-up: `ultra_tiny` → `ds_cnn` → `tiny_cnn` → `small_cnn` → `mobilenet_like` → `tiny_resnet` (add).
7. Tests `tests/test_c_export_cnn_graph.py`: tracing each supported model, rejection of `torchvision_mobilenet`,
   BN folding parity, IR interpreter vs. PyTorch, q15 mirror vs. float error bound, deterministic codegen bytes.

**Done when:** for each supported model, f32 C logits vs. PyTorch ≤ `f32_logit_abs_tol` and 100 % prediction
agreement; q15 C logits ≡ Python q15 mirror; q15 agreement, F1/PR-AUC delta vs. PyTorch, flash/SRAM reported.

### Stage 9 — Build, desktop app, parity and timing
1. `src/c_export/toolchain.py`: move MSVC discovery (`_find_vswhere`, `vcvars64` environment capture, compiler
   version) out of `src/features/build_native.py`; make `build_native.py` import it; re-run
   `tests/test_native_entropy.py` and the native build to prove no regression.
2. `build_c.py`: compile `runtime/*.c` + `generated/<target>/*.c` + `app/desktop_main.c` with
   `/I runtime /I generated/<target>` and `/D CIMP_NUMERIC_F32` or `/D CIMP_NUMERIC_Q15` → two executables in
   `c_imp/build/<target>/`; compile + run `tests/test_runtime.c`; write `build_info.json` (flags, compiler, hashes).
3. `app/desktop_main.c`: arguments `--manifest`, `--images-root`, `--mode {descriptor,classifier,cnn}`, `--model`,
   `--out`, `--warmup`, `--iters` (all supplied by the Python runner from TOML). Outputs one CSV row per image:
   `index, x94..., margin|logits..., pred, T_descriptor_ns, T_model_ns, T_total_ns`.
4. `parity.py` + `experiments/verify_c_pipeline.py`: build → run → compare against golden files → compute
   classification metrics with the same functions as `src` → write to `results/c_imp/<target>/`:
   `parity_descriptor.csv`, `parity_classifier.csv`, `parity_cnn_<model>.csv`, `accuracy_summary.csv`
   (Python float vs. C f32 vs. C q15), `timing_desktop.csv` (median/IQR/p95, labelled desktop x86, single thread),
   `memory_report.json`, `environment.json` (reuse `src.profiling.environment.capture_environment`), `README.md`.
5. `experiments/export_c_pipeline.py`: `main(argv)`, `argparse` (`--config`, `--object`, `--inverted`, `--only
   {luts,descriptor,classifier,cnn,vectors}`, `--dry-run`), logging via `configs.logging`, clean `SystemExit` messages.

**Done when:** both CLIs run end-to-end for `kolektorsdd2` and `mvtec-ad bottle` (normal; inverted optional), all
parity gates pass or deviations are reported with measured values.

### Stage 10 — Tests
- New Python suites listed in §2; `tests/test_c_pipeline_parity.py` uses `unittest.skipUnless(msvc_available(), ...)`
  with a clear skip reason and runs a small subset (e.g. 8 images) to stay fast.
- Full regression: `.\.venv\Scripts\python.exe -m unittest discover tests -v` (all previous 208 tests must still pass)
  and the new lab tests.

### Stage 11 — Documentation
- `c_imp/README.md`: data-flow diagram, each file's role, Q-format table per tensor, how weights get from
  `checkpoint.pt` to `model_<name>.h`, how to build/run, how to port to MCU, limitations.
- Update `src/README.md` (new `src/c_export/`), `experiments/README.md` (2 CLIs), `tests/README.md`, `README.md`
  (commands), `RESUME.md` (state + gap 1 progress), `understanding/README.md` (Lab 15), `.gitignore`.
- `requirements.txt`: no new dependency expected (`torch.fx`, `tomllib`, `hashlib` are already available); update
  only if that changes.

### Stage 12 - ESP32-CAM firmware (OUT OF SCOPE for this task, planned for a later task)
**Scope decision (user, 2026-11-13): in this task nothing is tested on hardware; only C on the local PC.** Stages 0-11 are the whole deliverable. The runtime stays embedded-friendly (no DLL, no malloc, no OS deps, budgets enforced by the exporter) so the later port is mechanical. Deferred to a future task: PlatformIO/ESP-IDF project compiling `c_imp/runtime` + `c_imp/generated/<target>` unchanged, cycle/flash/SRAM/PSRAM measurements on the ESP32-CAM, on-device OV2640 capture + C preprocessing (crop/letterbox + resize) and its accuracy delta vs. the Python LANCZOS pipeline, and the on-device f32 vs q15 vs int8 comparison that settles the numeric format. Until then, f32/q15 choices rest on desktop parity/accuracy and exporter-computed flash/SRAM budgets only; do not claim ESP32 speed.

## 5. Verification commands (PowerShell from `code/`, after implementation)

```powershell
.\.venv\Scripts\python.exe tests/test_environment.py
.\.venv\Scripts\python.exe -m unittest discover tests -v
.\.venv\Scripts\python.exe -m unittest discover understanding/15_c_migration_fixed_point/tests -v
.\.venv\Scripts\python.exe -m src.features.build_native --config configs/features/native_entropy.toml   # regression after toolchain refactor

.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/kolektorsdd2.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.verify_c_pipeline --config configs/c_imp/kolektorsdd2.toml

.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/mvtec_ad.toml --object bottle
.\.venv\Scripts\python.exe -m experiments.verify_c_pipeline --config configs/c_imp/mvtec_ad.toml --object bottle
```

Expected outputs: `c_imp/generated/<target>/*`, `c_imp/build/<target>/cimp_pipeline_{f32,q15}.exe`,
`results/c_imp/<target>/{parity_*.csv, accuracy_summary.csv, timing_desktop.csv, memory_report.json, README.md}`.

## 6. Decisions taken (user, 2026-11-13)

1. **MCU**: ESP32-CAM AI-Thinker (see §1 decision 11). Arduino AVR out of scope.
2. **Numeric format**: whatever is most efficient, mixed allowed → mixed policy of §1 decision 11, decided by measurement.
3. **Classifiers**: v1 = `lr_dot` **and** `mlp_16_dot`. Stage 7A promotes `mlp_16_dot` from Lab 14 to `src/models/`
   (without deleting the lab; lab keeps working and a parity test compares both implementations).
   Note on the name: "dot" = the scaler (`StandardScaler`) is algebraically **folded into the first dense layer**, so
   the model consumes raw `x94` and is just dot products (`w·x + b`), no per-call scaling/sklearn dispatch. It comes
   from `lr_dot` (`w_eff @ x + b_eff`). Document this in `src/README.md`.
4. **Generated C is committed to git**: `c_imp/generated/<target>/` is NOT gitignored (remove it from the planned
   `.gitignore` additions; keep `c_imp/build/`, `*.exe`, `*.pdb` ignored). Generation must be deterministic (no timestamps)
   so diffs are meaningful; large weight headers (e.g. MobileNetV3, ~1.5 MB binary → several MB of text) should be
   emitted as compact hex arrays and their size reported before committing; if >~10 MB total, warn the user.
5. **`torchvision_mobilenet` is supported** (Hardswish, Hardsigmoid, SE, 5×5 depthwise, Dropout removed).

## 7. Commit plan (Conventional Commits, small and verifiable)

1. `feat(understanding): add Lab 15 fixed-point and BN-folding study`
2. `feat(config,c_export): add c_imp TOML configs and loader`
3. `feat(c_imp): add fixed-point core, tensor and platform runtime`
4. `feat(c_export): generate config and LUT headers`
5. `feat(c_imp): portable x94 descriptor in f32 and q15 with Python mirror`
6. `feat(c_export): export test vectors and golden outputs`
7. `feat(c_export,c_imp): export lr_dot and dense-dot kernels`
8. `feat(c_export,c_imp): CNN graph IR, BN folding, q15 quantization and codegen`
9. `refactor(features): share MSVC toolchain discovery` + `feat(experiments): build, run and verify C pipeline`
10. `docs: document C migration pipeline and results`
