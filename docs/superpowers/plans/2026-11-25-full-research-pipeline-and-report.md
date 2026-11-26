# Full Research Pipeline, MDPI Visualization Upgrade & Paper-Ready Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task as a **single sequential autonomous agent** (do **NOT** use `superpowers:subagent-driven-development` or spawn any subagents; sequential single-process execution is mandatory per user spec for uncontaminated CPU/GPU latency measurements). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `src/visualization/` to a clean, headless-safe MDPI publication plotting package integrated across all pipeline plotters, execute a clean sequential end-to-end rerun of classical $\mathbf{x}_{94}$ classifiers, 7 lightweight CNNs, 100% CPU-fair benchmarks, deep profiling, and C99 fixed-point verification across `KolektorSDD2` and all 15 `MVTec-AD` categories, and compile a purely data-driven `results/FINAL_PAPER_REPORT.md`.

**Architecture:** Refine `src/visualization/sci_plots.py` into an importable `Agg`-safe package with deterministic bounding-box label placement, wire MDPI styling (`300 DPI`, editorial colorblind-safe palette, collision-free Pareto labels) into existing `src/models/`, `src/profiling/`, and `src/c_export/` plotting modules plus TOML-configured methodology/reporting CLIs (`experiments/visualize_methodology.py`, `experiments/generate_paper_report.py`), validate with the full `unittest` suite, safely archive/clean software `results/` while preserving irreplaceable physical MCU logs (`results/hw_test/`), and execute every experiment stage sequentially via `.\.venv\Scripts\python.exe`.

**Tech Stack:** Windows 10 PowerShell, Python 3.11 (`.\.venv\Scripts\python.exe`), PyTorch `2.14.0+cu126` (GTX 1050 Pascal `sm_61`), NumPy, scikit-learn, Matplotlib (`Agg` backend, Type-42 PDF/PNG `300 DPI`), MSVC x64 C99/C11 (`cl.exe`), declarative TOML (`tomllib`).

**Spec:** User request in conversation `cf69c7fa-8370-421a-8fcb-8cb9dc1617b9` + binding repository contracts in `AGENTS.md`, `README.md`, `RESUME.md`, and `INSTRUCTIONS.md`.

## Global Constraints

- **Single autonomous agent & strictly sequential execution:** Never call `invoke_subagent`, `swarm`, or `run_workflow`; never run training, benchmarking, or profiling commands in background/parallel. Wait for each command to finish and verify `$LASTEXITCODE -eq 0` before starting the next.
- **Interpreter & environment:** Execute all commands from `D:\Publications\2026-10-Symmetry-Entropy\code` using `.\.venv\Scripts\python.exe` (Python 3.11, PyTorch `2.14.0+cu126`); never install packages globally or introduce TensorFlow.
- **TOML-only configuration & dynamic paths:** All external parameters, paths, palettes, and DPI values live in `configs/**/*.toml`; `src/` contains zero hardcoded filesystem paths or CLI configuration defaults; resolve `PROJECT_ROOT = Path(__file__).resolve().parents[...]` dynamically.
- **Canonical seed & overwrite policy:** Keep `seed = 42` across all TOMLs and pass `--seeds 42` to `experiments.evaluate_cnn`. Pass `--overwrite` to every CLI that defines the flag (`evaluate_descriptors`, `evaluate_cnn`, `visualize_methodology`, `generate_paper_report`). Note that `benchmark_cpu_fair`, `profile_pipelines`, `export_c_pipeline`, `verify_c_pipeline`, and `visualize_c_pipeline` overwrite deterministically without an `--overwrite` CLI flag.
- **Hard Boundary — LaTeX untouched:** Never read, list, inspect, or modify any `.tex`, `.bib`, `.sty`, or `.cls` file anywhere in `code/` or `paper/`.
- **Hard Boundary — Precomputed datasets untouched:** Verify existence of `data/raw/`, `data/preprocessed/{mvtec-ad,mvtec-ad-inverted,KolektorSDD2,KolektorSDD2-inverted}/`, and `data/csv/*-entropy-full.csv` (+ `.json`); do **not** run `preprocess_mvtec_ad`, `preprocess_kolektorsdd2`, `export_mvtec_ad_csv`, `export_kolektorsdd2_csv`, `export_entropy_maps`, or `append_asymmetry_features` unless one of the 4 `*-entropy-full.csv` files is genuinely missing.
- **Hard Boundary — Locked pedagogical labs untouched:** Never edit `understanding/07_descriptor_behavior/` or `run1/` / SHA-256 hash files in `understanding/02_histogram_benchmark/` and `understanding/03_entropy_benchmark/`.
- **Hard Boundary — Deprecated benchmark excluded:** Never run or cite `experiments/compare_baseline_vs_cnn.py` or `results/comparison/` for latency claims; use exclusively `results/comparison_cpu/` (`T_total = T_descriptor + T_model` vs. `T_preprocess + T_model` on 100% single-thread CPU) and `results/profiling/`.
- **Scientific integrity:** Every number in `results/FINAL_PAPER_REPORT.md` must be read programmatically from fresh artifacts generated during this run—never estimated, fabricated, or copied from historical Markdown docs.

## Review Focus

1. **Root `cleanup.ps1 -AllArtifacts` wiping physical MCU UART logs (`results/hw_test/esp32cam/*_uart/`):** Running `cleanup.ps1 -AllArtifacts` on the root `results/` directory deletes `summary.json` and `per_image.csv` captured on physical ESP32-CAM hardware (`COM3`), which cannot be regenerated without attached hardware. Task 1 pins a scoped cleanup targeting only software-regenerable subdirectories (`classification`, `cnn`, `comparison_cpu`, `comparison_cpu_python`, `profiling`, `c_imp`, `comparison`) and verifies `results/hw_test/` file count before and after.
2. **`mlp_16_dot` missing from `ALLOWED_PROFILING_CLASSIFIERS` and `configs/profiling/*.toml`:** `src/profiling/classifier_profile.py` already implements parameter, FLOP, and direct-kernel profiling for `MlpDotCoreClassifier`, but `src/profiling/config.py` (`ALLOWED_PROFILING_CLASSIFIERS`), `experiments/benchmark_cpu_fair.py` (`resolve_selected_classifiers`), and `configs/profiling/{kolektorsdd2,mvtec_ad}.toml` omit `"mlp_16_dot"`, which would leave Deliverable §4.3 without 100% CPU `T_desc / T_model / T_total` numbers for `mlp_16_dot`. Task 3 adds `"mlp_16_dot"` and pins it in `tests/test_profiling_config.py` and `tests/test_profiling_classifier.py`.
3. **Dense scatter/Pareto label collisions across 13 models (`6 ML + 7 CNN`):** Fixed `(6, 4)` or `(5, 4)` text offsets collide when `lr_dot` and `mlp_16_dot` sit microseconds apart on log-latency axes, when `tiny_cnn` and `small_cnn` neighbour each other on `kolektorsdd2`, or when multiple CNNs tie at `F1 ≈ 0.3611` on small MVTec validation splits. Task 2 adds `annotate_points_non_overlapping` with bounding-box, marker, and leader-segment repulsion and pins zero-overlap behavior on tied/clustered coordinates in `tests/test_sci_plots.py`.
4. **Tabular weight-format tag hardcoded as `"q15"` when `weight_bits == 32` (`q31`):** On `kolektorsdd2`, `_quantize_lr_dot` and `_quantize_mlp_16_dot` auto-upgrade weights to 32-bit (`int32_t`, recorded as `"weight_bits": 32` in `export_manifest.json`), while `src/c_export/visualize_data.py` unconditionally labels tabular `weight_format = "q15"`. Task 3 maps `weight_bits == 32 -> "q31"` (preserving `"q15"` for `16` or omitted legacy fixtures) and pins both cases in `tests/test_c_export_visualize.py`.
5. **Headless Matplotlib / layout warnings & `dpi` test assumption (`dpi = 160` -> `300`):** `src/visualization/sci_plots.py` currently lacks `matplotlib.use("Agg")`, mixes `constrained_layout=True` (`create_grid`) with `fig.tight_layout()`, includes interactive `plt.show()` demo code, and `tests/test_c_export_visualize.py` hardcodes `assertEqual(cfg.visualization.dpi, 160)` + `.replace("dpi = 160", "dpi = 0")`. Tasks 2 and 3 fix backend/layout safety and update the test to validate positive TOML DPI cleanly at `300 DPI`.

---

## File Structure Map

- **Create:**
  - `src/visualization/__init__.py` — Public package exports for `sci_plots`.
  - `configs/visualization/methodology.toml` — Declarative config for methodology figures (sample selection, grid levels, palette, journal preset, DPI, output directory `results/methodology`).
  - `experiments/visualize_methodology.py` — CLI entry point generating publication-grade multiscale entropy pyramid ($1\times 1, 2\times 2, 4\times 4, 8\times 8$) + spatial asymmetry ($\mathbf{x}_9$) figures on real `KolektorSDD2` and `MVTec-AD` normal/defect/inverted samples.
  - `configs/reporting/paper_report.toml` — Declarative paths, MVTec object-vs-texture category partition, and schema rules for the final paper report generator.
  - `experiments/generate_paper_report.py` — Read-only, fail-fast report compiler producing `results/FINAL_PAPER_REPORT.md` strictly from fresh JSON/CSV outputs on disk.
  - `tests/test_sci_plots.py` — Unit tests for `src/visualization/sci_plots.py` (Agg safety, style presets, `create_grid` + subplot helpers without layout warnings, zero-overlap Pareto label placement, PDF/PNG export).
  - `tests/test_visualize_methodology.py` — Unit/CLI tests for `experiments/visualize_methodology.py` (`--dry-run`, output creation, `--overwrite`).
  - `tests/test_generate_paper_report.py` — Unit tests verifying report assembly, object/texture macro-averages, polarity deltas, timestamp freshness check, and zero fabrication on synthetic fixtures.
- **Modify:**
  - `src/visualization/sci_plots.py` — Enforce `Agg` safety when headless, remove `__main__` and commented `plt.show()` blocks, guard `tight_layout()` when `constrained_layout` is active, add `annotate_points_non_overlapping(...)`, and wire it into `plot_scatter(..., annotate_points=True)`.
  - `src/models/classification_plots.py` — Apply `sci_plots.apply_style("mdpi")` and editorial palette/typography across all 5 classification/ablation/explainability/heatmap/all-objects figures while preserving exact function signatures and output filenames.
  - `src/models/cnn/report.py` — Apply `sci_plots.apply_style("mdpi")` and crisp editorial colors/typography in `plot_confusion_and_curves` and `plot_training_curves`.
  - `src/profiling/config.py` & `experiments/benchmark_cpu_fair.py` — Add `"mlp_16_dot"` to `ALLOWED_PROFILING_CLASSIFIERS` and `resolve_selected_classifiers`.
  - `src/profiling/report.py` — Apply `sci_plots.apply_style("mdpi")` and `annotate_points_non_overlapping` in `plot_cpu_comparison_pareto` and `generate_profiling_figures`.
  - `src/c_export/visualize_data.py` — Resolve tabular `weight_format` from `export_manifest.json` (`"q31"` when `weight_bits == 32`, `"q15"` when `16` or omitted).
  - `src/c_export/visualize_plots.py` — Apply `sci_plots.apply_style("mdpi")` and upgrade `_annotate_pareto_labels` / `fig1`–`fig6` styling to eliminate `tiny_cnn`/`small_cnn` leader-line crossover.
  - `configs/classification/{kolektorsdd2,mvtec_ad}.toml`, `configs/cnn/{kolektorsdd2,mvtec_ad}.toml`, `configs/c_imp/{kolektorsdd2,mvtec_ad}.toml` — Set `dpi = 300`.
  - `configs/profiling/{kolektorsdd2,mvtec_ad}.toml` — Set `dpi = 300` and include `"mlp_16_dot"` in `classifiers` and `ml_baseline_models`.
  - `tests/test_c_export_visualize.py`, `tests/test_profiling_config.py`, `tests/test_profiling_classifier.py` — Add unit tests for `weight_bits == 32 -> "q31"`, `dpi = 300` config loading, and `mlp_16_dot` profiling.

---

### Task 1: Environment Verification, Dataset Integrity Audit & Safe Scoped Reset (Step 0)

**Files:**
- Read-only inspect: `cleanup.ps1`, `pipeline.ps1`, `memory/export_readmes.ps1`, `memory/export_results.ps1`, `configs/base.toml`, `configs/features/native_entropy.toml`
- Build output: `src/features/native/build/entropy_pyramid.dll`
- Archive output: `memory/archive_pre_run_<timestamp>.zip`
- Scoped clean targets: `results/classification`, `results/cnn`, `results/comparison_cpu`, `results/comparison_cpu_python`, `results/profiling`, `results/c_imp`, `results/comparison`, `c_imp/generated/*`, `c_imp/build/*` (strictly preserving `results/hw_test/`)

**Interfaces:**
- Consumes: Precomputed `data/raw/`, `data/preprocessed/`, `data/csv/*-entropy-full.csv` (+ `.json`), `.venv` Python 3.11 + CUDA 12.6, MSVC `cl.exe`.
- Produces: Verified `entropy_pyramid.dll` (`c_single_pass`), archived pre-run software results zip, clean software `results/` subdirectories, intact `results/hw_test/` directory, recorded `RUN_START_UTC` timestamp.

- [ ] **Step 1: Verify Python virtual environment, dependency health, and real CUDA execution**

Run sequentially in PowerShell from `D:\Publications\2026-10-Symmetry-Entropy\code`:
```powershell
.\.venv\Scripts\python.exe tests/test_environment.py
.\.venv\Scripts\python.exe -m pip check
```
Expected: `Environment verification: PASS` (confirming NVIDIA GeForce GTX 1050 `sm_61` real tensor op on CUDA) and `No broken requirements found.`

- [ ] **Step 2: Clean-build the native C99 single-pass entropy DLL and run its unit tests**

Run sequentially:
```powershell
.\.venv\Scripts\python.exe -m src.features.build_native --config configs/features/native_entropy.toml --clean
.\.venv\Scripts\python.exe -m unittest tests/test_native_entropy.py -v
```
Expected: `src/features/native/build/entropy_pyramid.dll` compiled with exit code `0` and all 5 tests in `test_native_entropy.py` passing (`OK`).

- [ ] **Step 3: Audit precomputed dataset directories and `*-entropy-full.csv` files without touching them**

Run a read-only PowerShell verification check confirming:
1. `data/preprocessed/KolektorSDD2`, `data/preprocessed/KolektorSDD2-inverted`, `data/preprocessed/mvtec-ad` (all 15 category folders from `configs/base.toml`), and `data/preprocessed/mvtec-ad-inverted` (all 15 category folders) exist and are non-empty.
2. `data/csv/kolektorsdd2-entropy-full.csv` (`3,335` data rows, `99` columns), `data/csv/kolektorsdd2-inverted-entropy-full.csv` (`3,335` data rows), `data/csv/mvtec-ad-entropy-full.csv` (`6,612` data rows, `101` columns), and `data/csv/mvtec-ad-inverted-entropy-full.csv` (`6,612` data rows) plus their `.json` sidecars exist.
Expected: All checks pass; zero preprocessing or CSV export scripts are executed.

- [ ] **Step 4: Archive existing software results and execute scoped cleanup while protecting `results/hw_test/`**

*(Note: `RESUME.md` references legacy names `memory/archive_and_reset.ps1` and `memory/run_canonical_experiments.ps1`; on disk, the actual utilities are root `cleanup.ps1` and `pipeline.ps1` plus `memory/export_{readmes,results}.ps1`.)*
1. Record file count in `results/hw_test/` before cleanup (`$hwBefore = (Get-ChildItem results/hw_test -Recurse -File).Count`).
2. Compress existing software result folders (`results/classification`, `results/cnn`, `results/comparison_cpu`, `results/comparison_cpu_python`, `results/profiling`, `results/c_imp`, `results/comparison`) if non-empty into `memory/archive_pre_run_$(Get-Date -Format 'yyyyMMddTHHmmss').zip`.
3. Preview with `-DryRun`, then run `cleanup.ps1 -AllArtifacts` **per software subdirectory** so `results/hw_test/` is never touched:
```powershell
$softDirs = @('classification', 'cnn', 'comparison_cpu', 'comparison_cpu_python', 'profiling', 'c_imp', 'comparison')
foreach ($d in $softDirs) {
    $target = Join-Path 'results' $d
    if (Test-Path -LiteralPath $target) {
        .\cleanup.ps1 -AllArtifacts -TargetDirectory $target -Confirm:$false
    }
}
```
4. Remove generated target subdirectories inside `c_imp/generated/` and `c_imp/build/` (preserving any `.gitkeep`), and remove `results/FINAL_PAPER_REPORT.md` if present.
5. Verify `(Get-ChildItem results/hw_test -Recurse -File).Count -eq $hwBefore` and confirm zero `.csv`, `.json`, `.png`, `.pt` artifacts remain under the 7 cleaned software result folders.

---

### Task 2: Refine `src/visualization/sci_plots.py` into a Headless-Safe MDPI Package (Step 1a)

**Files:**
- Create: `src/visualization/__init__.py`
- Modify: `src/visualization/sci_plots.py`
- Create: `tests/test_sci_plots.py`

**Interfaces:**
- Consumes: `matplotlib` (`Agg`-safe), `numpy`.
- Produces (`src.visualization`):
  - Constants: `PALETTES: dict[str, list[str]]`, `MARKERS: list[str]`, `LINESTYLES: list[str]`, `HATCHES: list[str]`, `FIG_SIZES: dict[str, tuple[float, float]]`
  - Style & layout helpers:
    - `apply_style(journal: str = "mdpi", palette: str = "editorial", font_size: float | None = None, use_tex: bool = False) -> None`
    - `get_colors(n: int | None = None, palette: str | Sequence[str] | None = None) -> list[str]`
    - `save_figure(fig: Figure, path: str | Path, dpi: int = 300, transparent: bool = False) -> Path`
    - `create_grid(nrows: int, ncols: int, *, figsize: str | tuple[float, float] | None = None, sharex: bool = False, sharey: bool = False, panel_labels: bool = True, label_pos: tuple[float, float] = (-0.12, 1.06)) -> tuple[Figure, np.ndarray]`
    - `annotate_points_non_overlapping(ax: Axes, labels: Sequence[str], xs: Sequence[float], ys: Sequence[float], *, xscale: str = "linear", yscale: str = "linear", fontsize: float | None = None, text_color: str = "#222222", leader_color: str = "#6B7280", reserved_boxes_axes: Sequence[tuple[float, float, float, float]] = (), zorder: int = 5) -> list[tuple[float, float, float, float]]`
  - Plotters: `plot_lines`, `plot_regression`, `plot_scatter`, `plot_bars`, `plot_histogram`, `plot_donut`, `plot_heatmap`.

- [ ] **Step 1: Write failing unit tests in `tests/test_sci_plots.py`**

Cover all 5 requirements and Review Focus items #3 and #5:
1. `test_headless_agg_and_style_presets`: importing `src.visualization` does not require a GUI display; `apply_style("mdpi", palette="editorial")` sets `savefig.dpi == 300`, `pdf.fonttype == 42`, `axes.spines.top == False`, `axes.spines.right == False`; invalid journal or palette raises `ValueError`.
2. `test_create_grid_and_subplot_composition_no_layout_warning`: combining `create_grid(1, 2)` (`constrained_layout=True`) with `plot_scatter(..., ax=axes[0, 0])` and `plot_heatmap(..., ax=axes[0, 1])` emits zero Matplotlib layout `UserWarning`s and writes non-empty `.png` and `.pdf` files via `save_figure`.
3. `test_non_overlapping_annotations_on_clustered_log_pareto`: given 9 tightly clustered/tied points on a log-x axis (including duplicate `(0.30, 0.3611)` and adjacent `(37.9, 0.619)` / `(52.2, 0.596)`), `annotate_points_non_overlapping` returns normalized bounding boxes within `[0.0, 1.0]` with zero pairwise box overlaps when feasible and leader lines enabled for displaced labels.
4. `test_all_master_plotters_smoke`: `plot_lines`, `plot_regression`, `plot_bars` (simple + grouped + horizontal), `plot_histogram`, `plot_donut`, and `plot_heatmap` (sequential + centered diverging) render cleanly to temporary PNGs and close figures cleanly.

- [ ] **Step 2: Run `tests/test_sci_plots.py` to verify failure**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_sci_plots.py -v
```
Expected: FAIL (`ImportError` for `src.visualization` / `annotate_points_non_overlapping`).

- [ ] **Step 3: Refine `src/visualization/sci_plots.py` and create `src/visualization/__init__.py`**

1. Set headless backend safely before `pyplot` import (`if not os.environ.get("MPLBACKEND") and matplotlib.get_backend().lower().startswith(("tk", "qt", "macosx", "gtk", "wx")): matplotlib.use("Agg")`, or unconditional `matplotlib.use("Agg")` consistent with `classification_plots.py` and `profiling/report.py`).
2. Remove the standalone `if __name__ == "__main__":` demo and lines 1315–1464 of commented `plt.show()` code so `src/visualization/sci_plots.py` contains exclusively reusable library logic compliant with `AGENTS.md`.
3. In `_finalize_ax` and `plot_donut`, call `fig.tight_layout()` only when `created_here and not fig.get_constrained_layout()`.
4. Implement `annotate_points_non_overlapping(...)` in normalized axes coordinates (handling both `"linear"` and `"log"` `xscale`/`yscale`), scoring candidate `(dx_pt, dy_pt)` offsets against: (a) axes bounds `[0.015, 0.985]`, (b) already-placed label boxes + `reserved_boxes_axes` (e.g. legend box), (c) all scatter marker positions, and (d) leader-segment-to-other-marker clearance (preventing the `tiny_cnn` -> `small_cnn` crossover), drawing thin `#6B7280` leader lines whenever displacement exceeds `13 pt`.
5. Use `annotate_points_non_overlapping` inside `plot_scatter(..., annotate_points=True)` for both scalar and array series, and re-export the public API in `src/visualization/__init__.py`.

- [ ] **Step 4: Run `tests/test_sci_plots.py` to verify all tests pass**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_sci_plots.py -v
```
Expected: PASS (all tests in `test_sci_plots.py` OK, zero warnings).

---

### Task 3: Integrate MDPI Styling Across Pipeline Plotters, Fix `c_imp` Tags, Enable `mlp_16_dot` CPU Profiling & Add Methodology Plotter (Step 1b)

**Files:**
- Modify: `src/models/classification_plots.py`
- Modify: `src/models/cnn/report.py`
- Modify: `src/profiling/config.py`
- Modify: `experiments/benchmark_cpu_fair.py`
- Modify: `src/profiling/report.py`
- Modify: `src/c_export/visualize_data.py`
- Modify: `src/c_export/visualize_plots.py`
- Modify: `configs/classification/kolektorsdd2.toml`, `configs/classification/mvtec_ad.toml`
- Modify: `configs/cnn/kolektorsdd2.toml`, `configs/cnn/mvtec_ad.toml`
- Modify: `configs/profiling/kolektorsdd2.toml`, `configs/profiling/mvtec_ad.toml`
- Modify: `configs/c_imp/kolektorsdd2.toml`, `configs/c_imp/mvtec_ad.toml`
- Create: `configs/visualization/methodology.toml`
- Create: `experiments/visualize_methodology.py`
- Modify: `tests/test_c_export_visualize.py`, `tests/test_profiling_config.py`, `tests/test_profiling_classifier.py`
- Create: `tests/test_visualize_methodology.py`

**Interfaces:**
- Consumes: `src.visualization` (`apply_style`, `PALETTES`, `annotate_points_non_overlapping`, `save_figure`), `src.features.{entropy_maps,asymmetry_maps}`, `src.models.mlp_dot.MlpDotCoreClassifier`.
- Produces:
  - Updated `ALLOWED_PROFILING_CLASSIFIERS = ("dt", "knn", "lr", "lr_dot", "mlp_16_dot", "rf")` in `src/profiling/config.py` and matching set in `experiments/benchmark_cpu_fair.py::resolve_selected_classifiers`.
  - Accurate tabular `ModelSeriesBundle.weight_format` (`"q31"` when `weight_bits == 32`, `"q15"` otherwise) in `src/c_export/visualize_data.py`.
  - `experiments.visualize_methodology.main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int` producing `results/methodology/{kolektorsdd2,mvtec_ad_bottle}_entropy_pyramid_methodology.png` and `polarity_inversion_entropy_invariance.png` at `300 DPI`.
  - All existing plotting functions in `classification_plots.py`, `cnn/report.py`, `profiling/report.py`, and `c_export/visualize_plots.py` retaining identical call signatures and output filenames while rendering in crisp `300 DPI` MDPI editorial style.

- [ ] **Step 1: Write failing unit tests for `mlp_16_dot` profiling, `q31` weight-format tags, `300 DPI` configs, and methodology CLI**

1. In `tests/test_profiling_config.py`: assert `"mlp_16_dot"` is in `kol_cfg.classifiers`, `mv_cfg.classifiers`, and `kol_cfg.statistics.ml_baseline_models`, and `kol_cfg.output.dpi == 300`.
2. In `tests/test_profiling_classifier.py`: add `test_mlp_16_dot_params_ops_and_profile` asserting `count_classical_model_params` reports `params_model == 94 * 16 + 16 + 16 + 1 == 1537` (`scaler_params == 0`), `estimate_classical_inference_ops` reports `analytical_macs == 1520`, and `profile_classifier` populates `direct_core_stats`.
3. In `tests/test_c_export_visualize.py`:
   - Update `test_visualization_config_validation` to assert `cfg.visualization.dpi == 300` and use `re.sub(r"dpi\s*=\s*\d+", "dpi = 0", base_text)` for the negative test.
   - Add assertion in `test_enriched_post_verify_visualization_generates_all_six_figures` (with `"weight_bits": 32` on a tabular entry vs `"weight_bits": 16` / omitted) verifying `weight_format == "q31"` when `weight_bits == 32` and `"q15"` when `16` or omitted.
4. Create `tests/test_visualize_methodology.py` testing `experiments.visualize_methodology.main` on a synthetic temp project (`--dry-run` writes nothing and returns `0`; normal run writes expected `300 DPI` PNG figures; re-run without `--overwrite` skips cleanly; `--overwrite` regenerates).

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_profiling_config.py tests/test_profiling_classifier.py tests/test_c_export_visualize.py tests/test_visualize_methodology.py -v
```
Expected: FAIL on missing `"mlp_16_dot"`, `dpi` mismatch, `"q31"` mapping, and missing `experiments.visualize_methodology`.

- [ ] **Step 3: Update TOML configs, profiling classifier whitelist, `c_imp` tabular format tag, pipeline plotters, and `experiments/visualize_methodology.py`**

1. **Configs (`dpi = 300` & `mlp_16_dot`):**
   - Set `dpi = 300` in `configs/classification/{kolektorsdd2,mvtec_ad}.toml`, `configs/cnn/{kolektorsdd2,mvtec_ad}.toml`, `configs/profiling/{kolektorsdd2,mvtec_ad}.toml`, and `configs/c_imp/{kolektorsdd2,mvtec_ad}.toml`.
   - In `configs/profiling/{kolektorsdd2,mvtec_ad}.toml`, set `classifiers = ["dt", "knn", "lr", "lr_dot", "mlp_16_dot", "rf"]` and `ml_baseline_models = ["dt", "lr", "lr_dot", "mlp_16_dot", "rf"]`.
2. **Profiling whitelist:** Add `"mlp_16_dot"` to `ALLOWED_PROFILING_CLASSIFIERS` in `src/profiling/config.py` and `allowed` in `experiments/benchmark_cpu_fair.py::resolve_selected_classifiers`.
3. **Tabular weight format (`src/c_export/visualize_data.py`):** In `load_target_bundle`, read `w_bits = int(t_meta.get("weight_bits", 16))` and set `w_fmt = "q31" if w_bits == 32 else "q15"`.
4. **Integrate MDPI style (`apply_style("mdpi", palette="editorial")`) & collision-free labels across all 4 plotting modules:**
   - `src/models/classification_plots.py`: apply `apply_style("mdpi")`, editorial palette (`#1F4E79` entropy, `#B85042` asymmetry, `#2E7D5B` full `x94`, `#6B4C7A` ROC-AUC), clean despined axes, crisp bar annotations, and `cividis`/`YlOrRd` luminance-aware cell text on 2D spatial heatmaps.
   - `src/models/cnn/report.py`: apply `apply_style("mdpi")`, luminance-aware confusion matrix annotations, editorial curve styling, and clean legends in `plot_confusion_and_curves` and `plot_training_curves`.
   - `src/profiling/report.py`: apply `apply_style("mdpi")` and replace fixed `xytext=(5, 4)` loops in `plot_cpu_comparison_pareto` (`ax1`, `ax2`, `ax3`) with `annotate_points_non_overlapping(...)` so all 13 models (`6 ML + 7 CNN`) never overlap; harmonize colors/legends across `fig1`–`fig6`.
   - `src/c_export/visualize_plots.py`: apply `apply_style("mdpi")`, editorial palette, and use `annotate_points_non_overlapping(...)` (with legend bounding box reserved) in `_plot_fig5_memory_and_latency` so `tiny_cnn` and `small_cnn` annotations/leaders never collide.
5. **Create `configs/visualization/methodology.toml` and `experiments/visualize_methodology.py`:**
   - Load paths strictly from `configs/base.toml`, `configs/entropy/*.toml`, and `configs/asymmetry/*.toml`.
   - Deterministically select representative `good` vs. `defect` test images (`seed = 42`) for `KolektorSDD2` and `MVTec-AD` (`bottle` + a texture category `tile`) and render:
     1. `kolektorsdd2_x94_methodology.png` — `128×128` input, $B=16$ quantized map, multiscale entropy maps ($2\times 2, 4\times 4, 8\times 8$) for `good` vs. `defect`, and 9-bar spatial asymmetry residual comparison ($\mathbf{x}_9$).
     2. `mvtec_ad_bottle_x94_methodology.png` — same multiscale decomposition highlighting radial symmetry breaking on `bottle`.
     3. `polarity_inversion_invariance_comparison.png` — side-by-side standard vs. inverted (`255 - I` before zero-pad letterbox) showing exact histogram/entropy invariance on unpadded `MVTec-AD` cells vs. zero-padded boundary cells on `KolektorSDD2`.

- [ ] **Step 4: Run the full formal unit test suite and unlocked pedagogical lab test suites**

Run sequentially:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
.\.venv\Scripts\python.exe understanding/03_entropy_benchmark/test_entropy.py
.\.venv\Scripts\python.exe understanding/04_lightweight_cnns/tests/test_models.py
.\.venv\Scripts\python.exe understanding/05_symmetry_features/tests/test_symmetry_features.py
.\.venv\Scripts\python.exe understanding/06_entropy_pyramid_descriptor/tests/test_descriptor.py
$env:PYTHONPATH = (Get-Location).Path; .\.venv\Scripts\python.exe understanding/07_descriptor_behavior/tests/test_behavior_helpers.py
.\.venv\Scripts\python.exe -m unittest understanding/13_entropy_maps_optimization/tests/test_entropy_variants.py -v
.\.venv\Scripts\python.exe -m unittest understanding/14_ml_classifiers_mcu/tests/test_ml_classifiers_mcu.py -v
.\.venv\Scripts\python.exe -m unittest understanding/15_c_migration_fixed_point/tests/test_c_migration_fixed_point.py -v
```
Expected: 100% PASS across all formal `tests/` modules (existing 247 tests + new `test_sci_plots` & `test_visualize_methodology` cases) and all 96 lab tests, with zero locked lab files modified (`git status` clean on `understanding/`).

---

### Task 4: Classical Descriptor Evaluation (`x85`, `x9`, `x94`) — Full Coverage (Step 2)

**Files:**
- Outputs in `results/classification/kolektorsdd2/` (`6` classifier dirs: `full_{lr,lr_dot,mlp_16_dot,dt,knn,rf}`)
- Outputs in `results/classification/kolektorsdd2-inverted/` (`6` classifier dirs)
- Outputs in `results/classification/mvtec-ad/` (`15 × 6 = 90` `<object>_<clf>` dirs + `6` `mvtec_all_objects_<clf>_summary.csv` + `6` `mvtec_all_objects_<clf>_comparison.png`)
- Outputs in `results/classification/mvtec-ad-inverted/` (`15 × 6 = 90` `<object>_<clf>` dirs + `6` summary CSVs + `6` comparison PNGs)

**Interfaces:**
- Consumes: `data/csv/{kolektorsdd2,kolektorsdd2-inverted,mvtec-ad,mvtec-ad-inverted}-entropy-full.csv`, `configs/classification/{kolektorsdd2,mvtec_ad}.toml`.
- Produces: `192` `metrics_summary.json` + `ablation_table.csv` + `feature_importance.csv` + `scale_and_family_breakdown.csv` + `4` `300 DPI` PNGs per directory, plus `12` multi-object summary CSVs and `12` multi-object comparison PNGs.

- [ ] **Step 1: Run `KolektorSDD2` classical evaluation (standard, then `--inverted`) sequentially**

Run one command at a time, verifying `$LASTEXITCODE -eq 0` after each:
```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/kolektorsdd2.toml --classifier all --inverted --overwrite
```
Expected: Both exit `0`; `12` directories created under `results/classification/kolektorsdd2[-inverted]/full_<clf>/`, each containing `metrics_summary.json` and 4 PNG figures.

- [ ] **Step 2: Run `MVTec-AD` all-15-objects classical evaluation (standard, then `--inverted`) sequentially**

Run one command at a time, verifying `$LASTEXITCODE -eq 0` after each:
```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier all --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_descriptors --config configs/classification/mvtec_ad.toml --all-objects --classifier all --inverted --overwrite
```
Expected: Both exit `0`; all 15 categories (`bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`) evaluated across all 6 classifiers (`lr`, `lr_dot`, `mlp_16_dot`, `dt`, `knn`, `rf`) for standard and inverted (`180` object-classifier directories + `12` `mvtec_all_objects_<clf>_summary.csv` and `12` `mvtec_all_objects_<clf>_comparison.png`).

- [ ] **Step 3: Verify Step 2 artifact completeness**

Run a PowerShell count check:
```powershell
(Get-ChildItem results/classification -Recurse -Filter metrics_summary.json).Count
(Get-ChildItem results/classification -Recurse -Filter mvtec_all_objects_*_summary.csv).Count
```
Expected: `192` `metrics_summary.json` files and `12` `mvtec_all_objects_*_summary.csv` files.

---

### Task 5: Lightweight CNN Zoo Training & Evaluation — Full Coverage with Fresh Checkpoints (Step 3)

**Files:**
- Outputs in `results/cnn/kolektorsdd2/full/<model>/` (`7` models)
- Outputs in `results/cnn/kolektorsdd2-inverted/full/<model>/` (`7` models)
- Outputs in `results/cnn/mvtec-ad/<object>/<model>/` (`15 × 7 = 105` model dirs + `7` `mvtec_all_objects_<model>_summary.csv`)
- Outputs in `results/cnn/mvtec-ad-inverted/<object>/<model>/` (`15 × 7 = 105` model dirs + `7` `mvtec_all_objects_<model>_summary.csv`)

**Interfaces:**
- Consumes: `data/preprocessed/{KolektorSDD2,KolektorSDD2-inverted,mvtec-ad,mvtec-ad-inverted}/`, `data/csv/*-entropy-full.csv` (metadata index), `configs/cnn/{kolektorsdd2,mvtec_ad}.toml`, CUDA GPU (`GTX 1050`).
- Produces: `224` fresh `checkpoint.pt`, `metrics_summary.json`, `training_history.csv`, `confusion_matrix_and_curves.png`, and `training_curves.png` (`seed = 42`), plus `14` `mvtec_all_objects_<model>_summary.csv` tables.

- [ ] **Step 1: Train and evaluate all 7 CNNs on `KolektorSDD2` (standard, then `--inverted`) sequentially**

Run sequentially with `--seeds 42 --overwrite`:
```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model all --seeds 42 --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/kolektorsdd2.toml --model all --seeds 42 --inverted --overwrite
```
Expected: Both exit `0`; `14` fresh `checkpoint.pt` + `metrics_summary.json` files written under `results/cnn/kolektorsdd2[-inverted]/full/<model>/`.

- [ ] **Step 2: Train and evaluate all 7 CNNs across all 15 `MVTec-AD` categories (standard, then `--inverted`) sequentially**

*(Verified in `experiments/evaluate_cnn.py`: `--all-objects --model all --seeds 42 --overwrite` loops sequentially over each of the 7 architectures and each of the 15 `base.toml` categories, writing per-object checkpoints and `mvtec_all_objects_<model>_summary.csv`.)*
Run sequentially (allow full GPU training time; set generous command timeout and do not run anything else concurrently):
```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --all-objects --model all --seeds 42 --overwrite
.\.venv\Scripts\python.exe -m experiments.evaluate_cnn --config configs/cnn/mvtec_ad.toml --all-objects --model all --seeds 42 --inverted --overwrite
```
Expected: Both exit `0`; `210` fresh `checkpoint.pt` + `metrics_summary.json` files written across all 15 categories (standard + inverted) plus `14` `mvtec_all_objects_<model>_summary.csv` files.

- [ ] **Step 3: Verify Step 3 checkpoint & summary completeness**

Run a PowerShell count check:
```powershell
(Get-ChildItem results/cnn -Recurse -Filter checkpoint.pt).Count
(Get-ChildItem results/cnn -Recurse -Filter metrics_summary.json).Count
(Get-ChildItem results/cnn -Recurse -Filter mvtec_all_objects_*_summary.csv).Count
```
Expected: `224` `checkpoint.pt` files, `224` `metrics_summary.json` files, and `14` `mvtec_all_objects_*_summary.csv` files.

---

### Task 6: 100% CPU-Fair Benchmarking (Phase A) & Deep Pipeline Profiling (Phase B) (Step 4)

**Files:**
- Outputs in `results/comparison_cpu/`: `<tag>_baseline_vs_cnn_cpu.csv`, `<tag>_cuda_reference.csv` (isolated), `<tag>_pareto_threads*.png` across `32` dataset variants (`kolektorsdd2`, `kolektorsdd2-inverted`, `15` `mvtec-ad_<obj>`, `15` `mvtec-ad-inverted_<obj>`), plus consolidated `summary_all_datasets.csv`.
- Outputs in `results/profiling/<tag>/`: full Phase B suite (`environment.json`, `comparison_cpu.csv`, `descriptor_stage_times.csv`, `descriptor_scalability.csv`, `classifier_times.csv`, `cnn_layer_times_*.csv`, `cnn_aten_ops_*.csv`, `statistical_tests.csv`, `ablation_quality_vs_cost.csv`, `fig1`–`fig6`, `README.md`) for standard variants (`kolektorsdd2` + 15 `mvtec-ad_<obj>`), and inverted short descriptor verification suites for `--inverted` variants (`kolektorsdd2-inverted` + 15 `mvtec-ad-inverted_<obj>`).

**Interfaces:**
- Consumes: Fresh Step 3 CNN checkpoints (`results/cnn/**/checkpoint.pt`, verified on CPU within `1e-5`), precomputed `*-entropy-full.csv` and `128×128` PNGs, compiled `entropy_pyramid.dll` (`auto -> c_single_pass`), `configs/profiling/{kolektorsdd2,mvtec_ad}.toml`.
- Produces: Uncontaminated 100% CPU latency measurements (`threads=1, batch=1` primary + thread/batch sweeps) for all `6` classical classifiers (`dt`, `knn`, `lr`, `lr_dot`, `mlp_16_dot`, `rf`) and all `7` CNNs across `KolektorSDD2` and all `15` `MVTec-AD` objects.

- [ ] **Step 1: Run Phase A (`experiments.benchmark_cpu_fair`) sequentially with zero background load**

*(Note: `benchmark_cpu_fair` has no `--overwrite` flag; it overwrites per-tag CSVs/PNGs and upserts `summary_all_datasets.csv` deterministically.)*
Run one command at a time, waiting for completion and verifying `$LASTEXITCODE -eq 0`:
```powershell
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/kolektorsdd2.toml --inverted
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --all-objects
.\.venv\Scripts\python.exe -m experiments.benchmark_cpu_fair --config configs/profiling/mvtec_ad.toml --all-objects --inverted
```
Expected: All 4 commands exit `0` with zero CPU parity tolerance failures; `32` `*_baseline_vs_cnn_cpu.csv` files and `results/comparison_cpu/summary_all_datasets.csv` written; log confirms `backend=c_single_pass`.

- [ ] **Step 2: Run Phase B (`experiments.profile_pipelines`) sequentially with zero background load**

Run one command at a time, waiting for completion and verifying `$LASTEXITCODE -eq 0`:
```powershell
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/kolektorsdd2.toml --inverted
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/mvtec_ad.toml --all-objects
.\.venv\Scripts\python.exe -m experiments.profile_pipelines --config configs/profiling/mvtec_ad.toml --all-objects --inverted
```
Expected: All 4 commands exit `0`; `32` subdirectories under `results/profiling/<tag>/` populated with `environment.json`, stage/layer CSVs, `300 DPI` MDPI figures, and data-driven `README.md`.

- [ ] **Step 3: Verify Phase A & Phase B artifact completeness**

Run a PowerShell verification check:
```powershell
(Get-ChildItem results/comparison_cpu -Filter *_baseline_vs_cnn_cpu.csv).Count
(Get-ChildItem results/profiling -Recurse -Filter README.md).Count
```
Expected: `32` `*_baseline_vs_cnn_cpu.csv` files and `32` `results/profiling/<tag>/README.md` files (`16` full standard + `16` inverted checks).

---

### Task 7: C99 Fixed-Point Pipeline Export, Parity Verification, Timing, Visualization & Methodology Figures (Step 5)

**Files:**
- Generated C99 sources/headers: `c_imp/generated/<target>/` (`kolektorsdd2` + all `15` `mvtec_ad_<obj>` targets; ~0.6 GB total on `D:` which has >340 GB free)
- Compiled MSVC binaries: `c_imp/build/<target>/cimp_pipeline.exe`
- Parity, memory, timing & figure outputs: `results/c_imp/<target>/` (`memory_report.json`, `parity_report.json`, `timing_benchmark.csv`, `figures/fig1..fig6.png`, `figures/summary_models_metrics.csv`, `figures/summary_report.md`)
- Synchronized MCU fit report: `results/hw_test/fit_report.{csv,md}`
- Methodology visual figures: `results/methodology/*.png`

**Interfaces:**
- Consumes: Fresh Step 2 `metrics_summary.json` (cross-check) and Step 3 `checkpoint.pt` + `metrics_summary.json`, `configs/c_imp/{kolektorsdd2,mvtec_ad}.toml`, `configs/hw_test/boards.toml`, `configs/visualization/methodology.toml`, MSVC `cl.exe`.
- Produces: Verified C99 `f32` and bit-exact `Q15` (`0` LSB raw diff) parity + single-thread desktop C timings + ESP32-CAM Flash/SRAM/PSRAM footprints + `6` `300 DPI` figures across `kolektorsdd2` and all `15` `MVTec-AD` objects, refreshed `hw_test/fit_report.{csv,md}`, and `300 DPI` methodology figures in `results/methodology/`.

- [ ] **Step 1: Export, compile/verify (`--benchmark`), and visualize C99 pipeline on `KolektorSDD2` sequentially**

Run sequentially:
```powershell
.\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.verify_c_pipeline --config configs/c_imp/kolektorsdd2.toml --benchmark
.\.venv\Scripts\python.exe -m experiments.visualize_c_pipeline --config configs/c_imp/kolektorsdd2.toml
```
Expected: Exit `0` on all 3 commands; `results/c_imp/kolektorsdd2/parity_report.json` shows `"all_gates_passed": true` and `q15_vs_py_q15_max_raw_diff == 0` across `descriptor_x94`, `lr_dot`, `mlp_16_dot`, and all 7 CNNs; `results/c_imp/kolektorsdd2/figures/` has `fig1`–`fig6` (`300 DPI`) and `summary_models_metrics.csv` (with `q31` vs `q15` vs `int8_per_channel` accurately labeled).

- [ ] **Step 2: Export, compile/verify (`--benchmark`), and visualize C99 pipeline sequentially across all 15 `MVTec-AD` objects**

Run a sequential PowerShell loop over the exact 15 categories in `configs/base.toml`, aborting immediately on any non-zero exit code:
```powershell
$mvtecObjects = @(
    'bottle','cable','capsule','carpet','grid',
    'hazelnut','leather','metal_nut','pill','screw',
    'tile','toothbrush','transistor','wood','zipper'
)
foreach ($obj in $mvtecObjects) {
    .\.venv\Scripts\python.exe -m experiments.export_c_pipeline --config configs/c_imp/mvtec_ad.toml --object $obj
    if ($LASTEXITCODE -ne 0) { throw "export_c_pipeline failed on $obj" }
    .\.venv\Scripts\python.exe -m experiments.verify_c_pipeline --config configs/c_imp/mvtec_ad.toml --object $obj --benchmark
    if ($LASTEXITCODE -ne 0) { throw "verify_c_pipeline failed on $obj" }
    .\.venv\Scripts\python.exe -m experiments.visualize_c_pipeline --config configs/c_imp/mvtec_ad.toml --object $obj
    if ($LASTEXITCODE -ne 0) { throw "visualize_c_pipeline failed on $obj" }
}
```
Expected: All `15 × 3 = 45` sequential invocations exit `0`; all 15 `results/c_imp/mvtec_ad_<obj>/parity_report.json` files report `"all_gates_passed": true` and `q15` raw diff `0`, and each `figures/` folder contains `fig1`–`fig6` + `summary_models_metrics.csv` + `summary_report.md`.

- [ ] **Step 3: Refresh `hw_test` offline accuracy cross-reference and generate `results/methodology/` figures**

Run sequentially:
```powershell
.\.venv\Scripts\python.exe -m experiments.hw_test report --config configs/hw_test/boards.toml
.\.venv\Scripts\python.exe -m experiments.visualize_methodology --config configs/visualization/methodology.toml --overwrite
```
Expected: `results/hw_test/fit_report.{csv,md}` refreshed cleanly (combining preserved physical ESP32-CAM UART timings with freshly verified `kolektorsdd2` offline metrics) and all `300 DPI` methodology figures saved in `results/methodology/`.

---

### Task 8: Compile Factual Paper-Ready Research Report (`results/FINAL_PAPER_REPORT.md`) (Deliverable §4)

**Files:**
- Create: `configs/reporting/paper_report.toml`
- Create: `experiments/generate_paper_report.py`
- Create: `tests/test_generate_paper_report.py`
- Output: `results/FINAL_PAPER_REPORT.md`

**Interfaces:**
- Consumes (read-only from disk):
  - `results/profiling/kolektorsdd2/environment.json` & `results/c_imp/kolektorsdd2/parity_report.json` (compiler version, CPU/GPU/OS/Python/PyTorch specs)
  - `results/classification/**/metrics_summary.json`, `ablation_table.csv`, `scale_and_family_breakdown.csv`, `mvtec_all_objects_*_summary.csv`
  - `results/cnn/**/metrics_summary.json`, `mvtec_all_objects_*_summary.csv`
  - `results/comparison_cpu/*_baseline_vs_cnn_cpu.csv`, `summary_all_datasets.csv`
  - `results/profiling/*/descriptor_stage_times.csv`, `statistical_tests.csv`, `ablation_quality_vs_cost.csv`
  - `results/c_imp/*/figures/summary_models_metrics.csv`, `parity_report.json`, `memory_report.json`, `timing_benchmark.csv`
  - All `.png` paths under `results/{methodology,classification,cnn,comparison_cpu,profiling,c_imp}/`
- Produces:
  - `experiments.generate_paper_report.main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int`
  - Verified `results/FINAL_PAPER_REPORT.md` containing exclusively measured values from this run across all 6 required sections (§4.1–§4.6), echoed in full/structured form in the final response without touching any `.tex` or `.bib` file.

- [ ] **Step 1: Write failing unit tests in `tests/test_generate_paper_report.py`**

Test `experiments.generate_paper_report` on synthetic CSV/JSON fixtures covering:
1. Object (`10` categories: `bottle`, `cable`, `capsule`, `hazelnut`, `metal_nut`, `pill`, `screw`, `toothbrush`, `transistor`, `zipper`) vs. Texture (`5` categories: `carpet`, `grid`, `leather`, `tile`, `wood`) macro-average computation (F1, Balanced Accuracy, ROC-AUC, PR-AUC, $\Delta\text{F1}$ ablation, and $\mathbf{x}_9$ share %).
2. Polarity-inversion delta calculation ($\Delta\text{F1}_{\text{inv}} = \text{F1}_{\text{inverted}} - \text{F1}_{\text{standard}}$) for all 6 classical classifiers and 7 CNNs.
3. Fail-fast `FileNotFoundError` if any required dataset/object result file is missing, and timestamp audit confirming every consumed artifact is newer than a supplied `min_mtime_utc` threshold.

- [ ] **Step 2: Run `tests/test_generate_paper_report.py` to verify failure**

Run:
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_generate_paper_report.py -v
```
Expected: FAIL (`ModuleNotFoundError: No module named 'experiments.generate_paper_report'`).

- [ ] **Step 3: Implement `configs/reporting/paper_report.toml` and `experiments/generate_paper_report.py`**

Build a clean, modular, zero-hardcoding report generator compliant with `AGENTS.md` that parses all freshly generated JSON/CSV artifacts, verifies file modification timestamps against the Step 0 reset time, and emits `results/FINAL_PAPER_REPORT.md` with the exact 6 sections requested:
1. **Execution Audit & Environment:** CPU, GPU (`GTX 1050`), RAM, OS, Python/PyTorch/CUDA/NumPy/scikit-learn/MSVC versions, `seed = 42`, exact sequential CLI commands executed, artifact counts & timestamp confirmation of clean overwrite, and documented methodological observations (e.g. `dt` tie-breaking note, small-validation `val_f1` threshold collapse vs. $\tau=0.50$ also recorded from `metrics_summary.json`).
2. **Dataset & Split Summary Table:** Exact train/val/test `good` and `defect` image counts for `KolektorSDD2` (`3,335` total) and all `15` `MVTec-AD` categories explicitly labeled `Object` (`10`) vs. `Texture` (`5`), plus subtotal rows.
3. **`KolektorSDD2` Complete Results (Standard vs. Inverted):** Complete table of all `6` classical classifiers (`dt`, `knn`, `lr`, `lr_dot`, `mlp_16_dot`, `rf`) and all `7` CNNs reporting F1 (plus CNN F1@0.50 reference), Balanced Accuracy, ROC-AUC, PR-AUC, `params_model`, 100% CPU single-thread (`threads=1, batch=1`) $T_{\text{desc}}$ / $T_{\text{prep}}$, $T_{\text{model}}$, $T_{\text{total}}$ (median + IQR, ms), and polarity-inversion delta $\Delta\text{F1}_{\text{inv}}$.
4. **`MVTec-AD` All-15-Objects Results (Standard vs. Inverted):**
   - Per-object comparison tables across all 15 categories (all 6 $\mathbf{x}_{94}$ classifiers & best $\mathbf{x}_{94}$ vs. lightweight CNNs, standard & inverted, with 100% CPU `threads=1, batch=1` latencies).
   - Macro-average summary table stratified by **All 15 Categories**, **10 Object Categories**, and **5 Texture Categories**.
   - Multiscale ablation table ($\mathbf{x}_{85}$ vs. $\mathbf{x}_9$ vs. $\mathbf{x}_{94}$, $\Delta\text{F1} = \text{F1}(\mathbf{x}_{94}) - \text{F1}(\mathbf{x}_{85})$) and explainability importance share table ($\mathbf{x}_{85}$ vs. $\mathbf{x}_9$ share % compared against the $9/94 = 9.57\%$ uniform chance baseline).
5. **C99 `f32` / `Q15` Parity, Latency & Memory Summary:** Bit-exact `Q15` parity (`0` raw LSB diff) and `f32` gate audit across `kolektorsdd2` and all `15` `MVTec-AD` objects, Flash/internal-SRAM/PSRAM footprints (`q15` / `q31` / `int8_per_channel`), desktop C99 single-thread `T_selector`, `T_model`, `T_total` (`f32` & `Q15`) vs. Python CPU timings (plus clearly labeled cross-reference to preserved physical ESP32-CAM `COM3` measurements in `results/hw_test/fit_report.csv`).
6. **Generated Figures Inventory:** Complete categorized table of every generated `300 DPI` publication figure (`results/methodology/`, `results/classification/`, `results/cnn/`, `results/comparison_cpu/`, `results/profiling/`, `results/c_imp/`) with exact paths and one-line LaTeX-ready caption descriptions.

- [ ] **Step 4: Run unit tests, generate `results/FINAL_PAPER_REPORT.md`, and run final verification**

Run sequentially:
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
.\.venv\Scripts\python.exe -m experiments.generate_paper_report --config configs/reporting/paper_report.toml --overwrite
```
Verify:
1. `results/FINAL_PAPER_REPORT.md` exists, is non-empty, contains zero `NaN`/`TBD`/placeholder tokens, and every cited figure path exists on disk.
2. `git status --porcelain ../paper` (or read-only check that zero `.tex`/`.bib` files were modified or untracked) confirms Hard Boundary #1 (`0` LaTeX files touched), `data/` timestamps are untouched (Hard Boundary #2), `understanding/` is untouched (Hard Boundary #3), and `results/comparison/` was never generated or referenced (Hard Boundary #4).
3. Present the complete measured report in the final response and link `results/FINAL_PAPER_REPORT.md`.
