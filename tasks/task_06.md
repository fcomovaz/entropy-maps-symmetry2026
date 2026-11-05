# Asymmetry Features Appender ($\mathbf{x}_{85} \rightarrow \mathbf{x}_{94}$ CSV Pipeline)

$$
\mathbf{x}_{94} = \left[ \underbrace{H^{(1)}, H^{(2)}, H^{(4)}, H^{(8)}}_{85\text{ entropy features}}, \underbrace{\mathbf{A}^{(2)}, \mathbf{A}^{(4)}, \mathbf{A}^{(8)}}_{9\text{ asymmetry features}} \right]
$$

## Issue

We need to compute the 9 multiscale spatial entropy asymmetry measurements ($\mathbf{x}_9$) directly from the existing $\mathbf{x}_{85}$ entropy CSV files and append them as new columns to produce the complete $\mathbf{x}_{94}$ descriptor tables.

- **Input:** Existing tabular CSV files containing metadata and the 85 normalized Shannon entropy features ($\mathbf{x}_{85}$). Do NOT recompute histograms or read raw images; operate strictly on the tabular entropy values.
- **Output:** For each processed input file `<filename>.csv`, generate a new file named `<filename>-full.csv` (or configurable suffix via TOML) containing all original columns (metadata + $\mathbf{x}_{85}$) followed by the 9 appended asymmetry columns ($\mathbf{x}_9$).

### Mathematical & Canonical Contract

1. **Grid Reconstruction:**
   - Exclude the $1\times1$ global entropy feature (its asymmetry residuals are identically zero).
   - For each spatial scale $g \in \{2, 4, 8\}$ (configurable via TOML), reconstruct the 2D row-major matrix $E^{(g)} \in \mathbb{R}^{g \times g}$ from the corresponding entropy columns of each row.

2. **Asymmetry Residuals ($A_{\mathrm{LR}}, A_{\mathrm{TB}}, A_{\mathrm{R180}}$):**
   - Must match the exact mathematical definition and numerical output of `code/understanding/05_symmetry_features/symmetry_features.py` and `code/understanding/06_entropy_pyramid_descriptor`:
     - $A_{\mathrm{LR}}^{(g)} = \frac{1}{g^2} \sum_{r=0}^{g-1}\sum_{c=0}^{g-1} \left| E^{(g)}_{r,c} - E^{(g)}_{r,\,g-1-c} \right|$
     - $A_{\mathrm{TB}}^{(g)} = \frac{1}{g^2} \sum_{r=0}^{g-1}\sum_{c=0}^{g-1} \left| E^{(g)}_{r,c} - E^{(g)}_{g-1-r,\,c} \right|$
     - $A_{\mathrm{R180}}^{(g)} = \frac{1}{g^2} \sum_{r=0}^{g-1}\sum_{c=0}^{g-1} \left| E^{(g)}_{r,c} - E^{(g)}_{g-1-r,\,g-1-c} \right|$

3. **Canonical Column Naming & Ordering (`schema x94-v1`):**
   - Append the 9 columns in this exact deterministic order:
     `asym_lr_2x2`, `asym_tb_2x2`, `asym_rot180_2x2`, `asym_lr_4x4`, `asym_tb_4x4`, `asym_rot180_4x4`, `asym_lr_8x8`, `asym_tb_8x8`, `asym_rot180_8x8`.

### Folder structure

The folder structure where the csv tables are located is the following:

`D:\Publications\2026-10-Symmetry-Entropy\code\data\csv\kolektorsdd2-entropy`,
`D:\Publications\2026-10-Symmetry-Entropy\code\data\csv\kolektorsdd2-inverted-entropy`
`D:\Publications\2026-10-Symmetry-Entropy\code\data\csv\mvtec-ad-entropy`,
`D:\Publications\2026-10-Symmetry-Entropy\code\data\csv\mvtec-ad-inverted-entropy`

each files contain an attached json file with metadata that needs to be updated,
each CSV file contains dataset metadata columns (such as `split`, `type`, `file_name`, `width`, `height`, etc.) and the 85 flattened multiscale entropy columns (`1x1`, `2x2`, `4x4`, `8x8` in row-major order). Inspect the exact header schema of these CSV files before designing the parser.

> Remember not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.



## Plan

> Planning only. Nothing has been executed or implemented. Facts below were verified by inspecting the repo on 2026-11-04.

- [x] Read `AGENTS.md` (TOML-only config, `pathlib` + root from entry point, `src/` has no hardcoded values, frozen/slots dataclasses, `from __future__ import annotations`, `argparse` + `main(argv)->int`, `logging` not `print`, fail fast, tests before done, no fabricated results).
- [x] Inspect existing code/data and write this plan.
- [x] Implement Stages A–E (Configs, `asymmetry_maps.py`, `asymmetry_csv.py`, CLI `append_asymmetry_features.py`, test suites `test_asymmetry_maps.py` and `test_asymmetry_csv.py`).
- [x] Manual execution (Stage F verified: all 4 `-full.csv` and `.json` files generated and validated).

### 0. Findings that shape the design

1. **Real file layout differs from the "Folder structure" section.** The files are flat in `data/csv/` (no `*-entropy/` sub-folders):
   `kolektorsdd2-entropy.csv`, `kolektorsdd2-inverted-entropy.csv`, `mvtec-ad-entropy.csv`, `mvtec-ad-inverted-entropy.csv`, each with a sibling `.json` sidecar. These four paths already exist in `configs/base.toml` as `entropy_csv_path` / `entropy_csv_path_inverted` under `[datasets.mvtecad]` and `[datasets.kolektorsdd2]`. **Reuse them. Do not create the sub-folders.**
2. **Header schema (verified).**
   - MVTec-AD: `object_name,split,type,defect_name,file_name,bins,grid_levels,E_1x1_r0_c0,…,E_8x8_r7_c7` → 92 columns (5 meta + `bins` + `grid_levels` + 85). 6612 rows.
   - KolektorSDD2: `split,type,file_name,bins,grid_levels,E_*` → 90 columns. 3335 rows.
   - Feature names follow `E_{g}x{g}_r{r}_c{c}` (TOML `feature_name_template`), row-major, levels in order 1, 2, 4, 8. Values are formatted with 8 decimals.
   - Sidecar JSON keys: `source_csv, output_csv, image_size, grid_levels, bins, rows, features_count, feature_columns, total_columns, float_precision, elapsed_seconds, numpy_version, timing_metric`.
3. **Math oracle**: `understanding/05_symmetry_features/symmetry_features.py` (pure Python, `fsum`, validates finite and in `[0,1]`, no clipping). `understanding/06_entropy_pyramid_descriptor` composes the x94 layout. `src/` must not import from `understanding/`. The oracle is used only in tests via `importlib`, as in task_05.
4. **Reuse patterns, do not duplicate**: `src/features/entropy_csv.py` already has `_require`, TOML loading, base-config resolution, chunked `pandas.read_csv`, a `.tmp` + atomic `replace` write, and the `--overwrite`, `--inverted`, `--dry-run` flags. Mirror its structure. `_require` is private, so either import it, which is acceptable inside `src/features`, or move it to a tiny shared helper. Prefer a small import and avoid a refactor.
5. **Input size is small** (3 to 7 MB), but keep chunked reading so the design scales and stays consistent with task_05.

### 1. Design decisions

| Topic | Decision |
|---|---|
| Config | **New** file per dataset: `configs/asymmetry/mvtec_ad.toml` and `configs/asymmetry/kolektorsdd2.toml`, same style as `configs/entropy/*.toml`. Paths are **not repeated**: they come from `configs/base.toml` via `dataset_key`. The only new path keys in `base.toml` are the output paths (see Stage A). |
| Source of truth for the suffix | `[asymmetry_features.output].suffix = "-full"` in TOML. The output is `<input stem><suffix>.csv`, built with `Path.with_name(f"{stem}{suffix}{ext}")`. Do not store four extra full paths. Alternative: add `*_full` keys to `base.toml`. **Choose the suffix approach**, because the task says "or configurable suffix via TOML". |
| Scales | `[asymmetry_features.regions].scales = [2, 4, 8]`, validated against the input `grid_levels` column and header. `1` is rejected (residuals are identically 0). |
| Column names | `[asymmetry_features.output].column_templates = { lr = "asym_lr_{g}x{g}", tb = "asym_tb_{g}x{g}", rot180 = "asym_rot180_{g}x{g}" }`. Order of kinds is `["lr","tb","rot180"]`, also in TOML, so the output is 2x2 lr,tb,rot180, then 4x4, then 8x8 (`schema x94-v1`). |
| Schema tag | `schema = "x94-v1"`, written to the sidecar. |
| Input column discovery | Entropy columns are located **by name** from `feature_name_template` and the scales (`E_{g}x{g}_r{r}_c{c}`), never by position or by the literal 7/92/85. Metadata columns are "everything in the input header that is not a generated entropy column", so the schema is not hardcoded per dataset. |
| Numerics | Vectorized NumPy on `float64`, reshape `[N, g, g]`. `A_LR = mean(abs(E - E[:, :, ::-1]))`, `A_TB = mean(abs(E - E[:, ::-1, :]))`, `A_R180 = mean(abs(E - E[:, ::-1, ::-1]))` over axes `(1,2)`. Output is formatted with the configured `float_precision`, same as the entropy CSV (8). |
| Validation | Fail fast, before the full read: (a) all required entropy columns exist for the configured scales and `1x1` exists, (b) no output column already exists in the input (prevents double append), (c) the `grid_levels` column, if present, contains the configured scales, (d) per chunk, values are finite and in `[0,1+eps]`, where `eps` is a TOML key `value_tolerance`, so rounding at the 8th decimal never fails. Out-of-range values raise `ValueError`, no clipping. |
| Row order | Identical to the input. 1:1 rows. |
| Overwrite | Same rule as task_05: error if the `-full.csv` exists unless `--overwrite`. Never modify the input file. |
| Sidecar | `<name>-full.json` written next to the output. It is a **new** JSON, derived from the input sidecar, with the fields updated: `source_csv` = the entropy CSV, `output_csv` = the `-full` CSV, `features_count` = 94 (85 + 9, computed), `feature_columns` = 85 + 9 names, `total_columns` = computed, `asymmetry_columns`, `asymmetry_scales`, `schema`, `timing_metric = "T_feature_extraction_asymmetry"`, `elapsed_seconds`, `numpy_version`. Keep the other input keys (`image_size`, `bins`, `float_precision`, ...) by copying them. If the input sidecar is missing, log a warning and build one without the inherited keys. The input sidecar `<filename>.json` stays untouched. This is the safe reading of "metadata that needs to be updated". |
| Unit of work | A function that processes **one** CSV file, plus a runner that loops over the configured dataset and `--inverted` / both variants. |

### 2. Files to create or modify

| File | Action |
|---|---|
| `configs/asymmetry/mvtec_ad.toml`, `configs/asymmetry/kolektorsdd2.toml` | **Create** |
| `src/features/asymmetry_maps.py` | **Create.** Pure math: validation and vectorized residuals, column names. No I/O. |
| `src/features/asymmetry_csv.py` | **Create.** Config dataclass and loader, header validation, chunked pipeline, sidecar. |
| `experiments/append_asymmetry_features.py` | **Create.** CLI with `build_parser()`, `main(argv, project_root)`, docstring of manual commands. |
| `tests/test_asymmetry_maps.py`, `tests/test_asymmetry_csv.py` | **Create** |
| `requirements.txt` | No change expected (NumPy and pandas already pinned). Confirm. |
| `README.md`, `experiments/README.md`, `src/README.md`, `tests/README.md` | Append a short section each: command, schema `x94-v1`, outputs. |
| `configs/base.toml` | No change if the suffix approach is used. |

### 3. Stages

**Stage A: Configuration (`configs/asymmetry/*.toml`)**
```toml
[asymmetry_features]
base_config = "configs/base.toml"
dataset_key = "mvtecad"            # "kolektorsdd2" in the other file
inverted = false
overwrite = false
schema = "x94-v1"

[asymmetry_features.regions]
scales = [2, 4, 8]
kinds = ["lr", "tb", "rot180"]

[asymmetry_features.source]
feature_name_template = "E_{g}x{g}_r{r}_c{c}"   # must match configs/entropy/*.toml
global_scale = 1                                 # required in the input header, excluded from the asymmetry
bins_column = "bins"
levels_column = "grid_levels"
levels_separator = "|"

[asymmetry_features.output]
suffix = "-full"
column_templates = { lr = "asym_lr_{g}x{g}", tb = "asym_tb_{g}x{g}", rot180 = "asym_rot180_{g}x{g}" }
float_precision = 8
value_tolerance = 1e-6
write_sidecar = true
sidecar_timing_metric = "T_feature_extraction_asymmetry"

[asymmetry_features.runtime]
chunk_size = 1024
```
The `feature_name_template` is duplicated from the entropy TOML. If that is unacceptable, read it from `configs/entropy/<dataset>.toml` through an `entropy_config` key. **Preferred: add `entropy_config = "configs/entropy/mvtec_ad.toml"` and read `feature_name_template`, `levels_*` and `bins_column` from it**, so there is a single source of truth. Document the choice in the file header.

**Stage B: Pure math (`src/features/asymmetry_maps.py`)**
- `asymmetry_column_names(scales, kinds, templates) -> list[str]`
- `entropy_column_names(scale, template) -> list[str]`: row-major, same ordering as `entropy_maps.feature_names`. **Reuse `feature_names`** from `entropy_maps.py` if it fits (it takes `grid_levels` and a template).
- `validate_scales(scales, global_scale)`: ints ≥ 2, unique, ascending, global scale not included.
- `reconstruct_grids(values[N, g*g], g) -> [N, g, g]` (row-major reshape).
- `compute_asymmetry(grids[N, g, g]) -> dict[kind, [N]]` (or `[N, 3]`, order lr, tb, rot180).
- `compute_asymmetry_features(entropy_by_scale: dict[int, ndarray], kinds, tolerance) -> ndarray[N, len(scales)*3]` in canonical order.
- Type hints and docstrings with shapes (`[N, g, g]`, `[N, features]`).

**Stage C: CSV pipeline (`src/features/asymmetry_csv.py`)**
- `@dataclass(frozen=True, slots=True) AsymmetryCsvConfig` (paths: input and output for standard and inverted variants, scales, kinds, templates, suffix, tolerance, float_precision, chunk_size, inverted, overwrite, write_sidecar, schema).
- `load_asymmetry_config(config_path, project_root)`: same style as `load_entropy_config` (anchor on `project_root`, `_require`, type checks). Compute `output_csv = input.with_name(stem + suffix + ".csv")` for both variants. Fail if input == output.
- `validate_header(header, cfg)`: checks required columns (including `1x1`), no collisions with output names, ordering sanity.
- `append_asymmetry_csv(cfg, dry_run=False) -> AsymmetrySummary`:
  1. Resolve the input, check it exists, and check the overwrite rule.
  2. Read the header only, then validate.
  3. Chunked `pd.read_csv(..., dtype=str)` for pass-through columns, so the original metadata and the 85 entropy strings are copied **byte-for-byte** (no float re-formatting). Parse only the needed entropy columns as `float64`.
  4. Compute the 9 features, format with `%.{p}f`, write `original columns + 9 new` to `<out>.csv.tmp`, then `replace` atomically. Remove the tmp file on error.
  5. After the pass, count rows, and assert `rows_out == rows_in`.
  6. Write the sidecar (Stage design above).
- `dry_run`: validates the header and the value ranges and counts rows, and writes nothing.
- Use `logging` (`logger = logging.getLogger("research")`), `tqdm` for chunks.

**Stage D: CLI (`experiments/append_asymmetry_features.py`)**
`python -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml [--inverted] [--overwrite] [--dry-run]`. Default config `configs/asymmetry/mvtec_ad.toml`. Optionally `--all`, which runs both variants and both datasets by globbing `configs/asymmetry/*.toml`, to cover the four files in one call. Catch `OSError, ValueError, KeyError` in `main` and raise `SystemExit(f"error: {exc}") from exc`. Include the docstring with the manual PowerShell commands (same style as `export_entropy_maps.py`).

**Stage E: Tests (`tests/`, run with `.venv\Scripts\python.exe -m pytest`)**
`test_asymmetry_maps.py`
- [x] Symmetric grid (`E == E[:, ::-1]`) → `A_LR == 0`. Same for TB and R180. A constant grid → all 0.
- [x] Hand-computed 2x2 example, e.g. `[[0,1],[0,1]]` → `LR = 1`, `TB = 0`, `R180 = 1`.
- [x] **Oracle parity**: random grids in `[0,1]` for `g ∈ {2,4,8}` against `understanding/05_symmetry_features/symmetry_features.py` (`reflection_asymmetry_lr/tb`, `rot180_asymmetry`), `atol=1e-12`. Load by `importlib` inside the test and skip with a clear message if it cannot be imported.
- [x] Row-major reshape: a grid built from `E_4x4_r{r}_c{c}` columns equals the expected matrix (asymmetric, to catch transposition).
- [x] Output order and names are exactly the 9 names listed in the issue. `scales=[2,4]` → 6 columns. `1` in scales → `ValueError`.
- [x] Out-of-range or NaN value → `ValueError` (no clipping). A tolerance edge of `1 + 1e-9` passes.

`test_asymmetry_csv.py` (with `tmp_path` and a synthetic entropy CSV plus `project_root` injection, no real data needed)
- [x] The output header is `original header + 9 columns`. Original columns are byte-identical row by row. Row count is unchanged and the order is preserved.
- [x] Column suffix: `foo.csv → foo-full.csv`. A custom suffix works. `foo-full.csv` is not accepted as input if it already has asymmetry columns (collision → `ValueError`).
- [x] Existing output without `--overwrite` → `FileExistsError`. With it → rewritten. Input file is unchanged (hash before and after).
- [x] Sidecar: `features_count == 94`, `total_columns` correct, `asymmetry_columns` correct, input sidecar untouched.
- [x] `--dry-run` writes nothing. A missing entropy column → `ValueError` mentioning the column. A `chunk_size` smaller than N gives the same result as a large one.
- [x] `main(argv)` returns 0, and a CLI error exits with a `SystemExit("error: ...")`.
- [x] Optional real-data test (skipped if the file is absent): 5 random rows of `mvtec-ad-entropy-full.csv` match the oracle recomputed from the printed 85 values.

**Stage F: Real execution (only after tests pass)**
Order: Kolektor first (smaller), then MVTec, then the inverted variants. From `code/`:
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_asymmetry_maps.py tests/test_asymmetry_csv.py
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/kolektorsdd2.toml --dry-run
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/kolektorsdd2.toml
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/kolektorsdd2.toml --inverted
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml
.\.venv\Scripts\python.exe -m experiments.append_asymmetry_features --config configs/asymmetry/mvtec_ad.toml --inverted
```
Verification for each output (record measured values only): rows equal the input rows (3335 / 3335 / 6612 / 6612 at the time of writing, **read from the input, never hardcoded**), columns = input + 9 (MVTec 101, Kolektor 99), no NaN, all 9 values in `[0,1]`, the header tail equals the canonical order, 5 to 10 random rows recomputed with the `understanding/05` oracle, and the original 92/90 columns equal the input (e.g. `pandas.DataFrame.equals` on strings). Run the full test suite (`pytest tests`) to check for regressions.

**Measured Execution Results (verified from sidecars):**
- `data/csv/kolektorsdd2-entropy-full.csv`: 3,335 rows, 99 columns (94 features), 3,559,332 bytes, `T_feature_extraction_asymmetry = 0.4290s`.
- `data/csv/kolektorsdd2-inverted-entropy-full.csv`: 3,335 rows, 99 columns (94 features), 3,559,332 bytes, `T_feature_extraction_asymmetry = 0.4156s`.
- `data/csv/mvtec-ad-entropy-full.csv`: 6,612 rows, 101 columns (94 features), 7,152,530 bytes, `T_feature_extraction_asymmetry = 0.8570s`.
- `data/csv/mvtec-ad-inverted-entropy-full.csv`: 6,612 rows, 101 columns (94 features), 7,152,530 bytes, `T_feature_extraction_asymmetry = 0.9270s`.

**Stage G: Documentation and close-out**
- Update the READMEs listed above with the command, schema `x94-v1`, and the output naming.
- Tick the checkboxes in this file, and list anything not verified.
- Report `T_feature_extraction_asymmetry` as measured, separate from the entropy extraction time. Do not sum it into pipeline claims without noting it (`T_total = T_selector + T_model` stays untouched).

### 4. Edge cases and risks

- **Rounding**: the inputs are rounded to 8 decimals, so the asymmetry values come from the printed numbers, not from the original floats. This is expected and must be documented. Parity tests use the same printed values.
- **Tolerance vs "no clipping"**: values slightly above 1 from rounding are tolerated up to `value_tolerance` but are never altered.
- **Double append**: running on an already-`-full` file is blocked by the collision check.
- **Inverted datasets** are processed as separate files, with the same schema. No special-casing.
- **Different metadata columns** per dataset are handled by "all non-entropy columns", not by listing names.
- **`grid_levels` column** (e.g. `1|2|4|8`) is used only as a consistency check against the header and the configured scales.
- **Windows**: use `Path.replace` for the atomic write, and `newline=""` on CSV writers (as in task_05).
- **Not in scope**: recomputing histograms, reading raw images, modifying the original entropy CSV or JSON files, and creating new sub-folders.

### 5. Open questions (defaults chosen if no answer)

1. Folder layout: keep the files flat in `data/csv/` as they are. *(Default: yes. The sub-folder names in the issue do not exist.)*
2. Suffix `-full` vs explicit paths in `base.toml`. *(Default: suffix in TOML.)*
3. Single-source `feature_name_template` through `entropy_config`. *(Default: yes.)*
4. Add `--all` to the CLI. *(Default: yes, it is small.)*
