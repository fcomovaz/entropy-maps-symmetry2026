# Entropy Maps and first $$\mathbf{x}_{85}= \left[ \underbrace{H^{(1)},H^{(2)},H^{(4)},H^{(8)}}_{85\text{ features}}\right]$$

## Issue

- The main issue addressed in this file is i need to créate the entropy maps by regions, in this case, a first csv needs to be generated, it needs to have the ability of choose the datase .csv that i want, any of them, that using obviously a toml file whatever you choose or créate. the maps are créate to use the pixels as base and then divide from 1-64 squares, like the h(1), h(8) are the regions to dived, 8 width 8 eight, i would like also to control this number so i can make more experiments with more or less regions, or actually that can goes in the toml file.
- the output file is going to be a file where i Will do later experiments to add features so it Will be my base entropy or my partial x85, actually it can be named like that because at the end of the whole research i Will add till x94 features but in later experiments.
- i want obviously this in the experiments, with test verifications and anything else.
- Using pixel intensities as the base, the image is partitioned into $g \times g$ regions for each scale $g$.
- By default, the canonical setup uses `grid_levels = [1, 2, 4, 8]`, yielding $1\times1 (1) + 2\times2 (4) + 4\times4 (16) + 8\times8 (64) = 85$ normalized Shannon entropy features.
- **Crucial:** Do not hardcode the grid levels or histogram bins ($B=16$ default). Both `grid_levels` (e.g., allowing `[1, 2, 4, 8]` or custom scales for ablation experiments) and `bins` must be controlled via the `.toml` file.
- The output `.csv` must preserve image identifiers/metadata (e.g., relative image path, category/subset, split, defect label) alongside the extracted entropy feature columns (`E_1x1_r0_c0`, `E_2x2_r0_c0`, etc., following the canonical naming convention).


### Folder structure

The folder structure where the reference codes are located is the following, check those codes to optimize and créate the experiments i want.

`Directorio: D:\Publications\2026-10-Symmetry-Entropy\code\understanding\03_entropy_benchmark` and
`Directorio: D:\Publications\2026-10-Symmetry-Entropy\code\understanding\06_entropy_pyramid_descriptor`

> Remember not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.


## Plan

> Don't execute anything, just plan it

- [x] Read AGENTS.md file to know the general rules of the project.
- [x] Create a detailed plan for this task and update it here to implement it later for an agent, so it needs to be detailed.
- [x] Update this file with the plan, stages and everything to resolve and implement the task.



---

# Entropy Maps / partial x85 — Implementation Plan

> Plan only. Nothing has been executed. Follows `AGENTS.md` (TOML config, `pathlib`, `from __future__ import annotations`, frozen dataclasses, `argparse` + `main(argv)`, `logging`, fail-fast).

## 0. Findings from inspecting the repo

| Item | Finding |
|---|---|
| Input CSVs | `data/csv/{mvtec-ad,mvtec-ad-inverted,kolektorsdd2,kolektorsdd2-inverted}.csv`, 128×128 images flat in `px_00000..px_16383`, then `width,height`. 130–370 MB each → must be read in **chunks**. |
| Metadata differs per dataset | MVTec: `object_name,split,type,defect_name,file_name`. KolektorSDD2: `split,type,file_name`. → metadata columns must be **configurable per dataset in TOML**, validated against the CSV header. |
| Existing CSV pattern | `configs/csv/*.toml` + `src/datasets/*_csv.py` (`load_csv_config`, frozen dataclass, `_require`, tmp-file + atomic rename, `overwrite` flag, `--dry-run`) + `experiments/export_*_csv.py`. Reuse the same style. |
| Reference (understanding/06) | `partition.py`: boundaries `floor(k*dim/g)`, row-major, no padding. `entropy.py`: bin = `pixel*B//256`, normalized `H/log2(B)`, `0·log0=0`, equiprobable counts → exactly `log2(B)`. `feature_names.py` hardcodes `LEVELS=(1,2,4,8)` and names `entropy_{g}x{g}_r{r}_c{c}` (**this must NOT be copied**; task asks `E_{g}x{g}_r{r}_c{c}`, levels from TOML). |
| Reference (understanding/03) | `e1` (scalar Shannon) is the readable oracle; `e3/e4` LUT variants are faster but per-patch Python. |
| `src/` | Has `datasets/`, `preprocessing/`; no `features/` yet. `requirements.txt` already has numpy, pandas, tqdm, pytest → **no new dependency**. |

## 1. Design decisions

1. **Output name**: `data/csv/<dataset>-entropy-x85.csv` style, path set in TOML (`[output].csv_path`, plus `csv_path_inverted`). The word "x85" is just a label; with custom `grid_levels` the feature count is `sum(g²)`, so the file name is configurable and the actual count is logged and stored in the sidecar.
2. **Dataset selection** via TOML (`[entropy_maps].dataset_key`, resolved in `configs/base.toml` `[datasets.<key>]`) and an `--config` CLI argument pointing to one of several per-dataset TOMLs; `--inverted` flag overrides like the existing exporters. Any of the 4 CSVs can be chosen.
3. **Input** is the already generated flat-pixel CSV (not the PNGs), as the issue says "dataset .csv that I want". Reshape `px_*` to `[N, H, W]` using `width/height` columns / `image_size` from the preprocessing TOML (validate they agree).
4. **Grid levels and bins come only from TOML** (`grid_levels`, `bins`). Validation (fail fast): `grid_levels` non-empty, strictly positive ints, unique; `g <= image_size`; `2 <= bins <= 256`, int. Order of the output columns = order of `grid_levels` as listed, row-major inside each level.
5. **Column naming**: `E_{g}x{g}_r{r}_c{c}` (no zero padding; pattern format string in TOML `[features].name_template = "E_{g}x{g}_r{r}_c{c}"` so nothing is hardcoded in `src/`).
6. **Metadata columns preserved** in the same order as the source (TOML list `metadata_columns`), plus optional `image_id` = relative path (`object_name/split/defect_name/file_name` composed from configured columns; for Kolektor the available ones). Label column `type` (`good`/`defect`) is preserved as-is.
7. **Algorithm** (vectorized, exact, validated against the oracle):
   - Quantize whole batch once: `q = (pixels.astype(uint16) * bins) // 256` → `[N, H, W]`.
   - For each level `g`, boundaries `edges = (arange(g+1) * dim) // g` (identical to `partition_boxes`).
   - Per region counts via one-hot integral image or per-region `np.bincount` over `q[:, y0:y1, x0:x1]` with an offset `n*bins` trick → `counts [N, g*g, B]`.
   - `p = counts / n_pixels_region`; `H = -sum(p*log2(p), where p>0)`; normalized `E = H / log2(B)`; clip nothing; if all counts are equal use exact `1.0` (mirror 06 rule, avoids float overshoot).
   - Each level recomputes histograms from pixels (never averages children).
   - Process in chunks of `chunk_size` rows (TOML `[runtime].chunk_size`) with `pandas.read_csv(chunksize=...)`, `usecols` = metadata + `px_*` + `width,height`, `dtype=uint8` for pixels. Output written in append mode to a `.tmp` file, then atomic rename (as in `mvtec_ad_csv.py`).
8. **Float formatting** in CSV: `float_format` from TOML (`[output].float_precision`, e.g. 8) for reproducibility.
9. **Sidecar** `<output>.json` (metadata for reproducibility): source csv, `grid_levels`, `bins`, image size, n_rows, n_features, column order, config path, elapsed time, numpy version. Keep it small; **timing reported as `T_selector`-independent feature extraction time** (not a model/pipeline claim).
10. **No performance claims** beyond measured elapsed time; keep the slow oracle for validation.

## 2. Files to create / modify

```
configs/entropy/mvtec_ad.toml          # NEW  dataset = mvtecad
configs/entropy/kolektorsdd2.toml      # NEW  dataset = kolektorsdd2
configs/base.toml                      # EDIT add entropy_csv_path / entropy_csv_path_inverted to each [datasets.*]
src/features/__init__.py               # NEW
src/features/entropy_maps.py           # NEW  pure math (numpy): quantize, region edges, entropy per level, feature names
src/features/entropy_csv.py            # NEW  config loading/validation, chunked CSV IO, atomic write, sidecar
experiments/export_entropy_maps.py     # NEW  CLI: python -m experiments.export_entropy_maps
tests/test_entropy_maps.py             # NEW  math tests vs oracle
tests/test_entropy_csv.py              # NEW  CSV/config/CLI tests
src/README.md, experiments/README.md, tests/README.md  # EDIT document the new module/CLI/tests
```

### 2.1 TOML sketch (`configs/entropy/mvtec_ad.toml`)

```toml
[entropy_maps]
base_config = "configs/base.toml"
dataset_key = "mvtecad"
preprocessing_config = "configs/preprocessing/mvtec_ad.toml"
inverted = false
overwrite = false

[entropy_maps.regions]
grid_levels = [1, 2, 4, 8]     # sum(g*g) = 85 features; change for ablations, e.g. [1, 2, 4, 8, 16]
bins = 16                      # histogram bins B, 2..256

[entropy_maps.source]
pixel_column_prefix = "px_"
metadata_columns = ["object_name", "split", "type", "defect_name", "file_name"]
size_columns = ["width", "height"]

[entropy_maps.output]
feature_name_template = "E_{g}x{g}_r{r}_c{c}"
float_precision = 8
write_sidecar = true

[entropy_maps.runtime]
chunk_size = 256               # rows per chunk (RAM control)
```

`kolektorsdd2.toml`: same with `dataset_key = "kolektorsdd2"`, `metadata_columns = ["split", "type", "file_name"]`.
`base.toml` gets e.g. `entropy_csv_path = "data/csv/mvtec-ad-entropy.csv"` and `entropy_csv_path_inverted = "data/csv/mvtec-ad-inverted-entropy.csv"`. (Optional: output file name includes bins/levels tag → decide at implementation; default is the fixed TOML path.)

### 2.2 Module APIs (all with type hints, docstrings, shapes)

`src/features/entropy_maps.py`
- `validate_grid_levels(levels: Sequence[int], image_size: int) -> tuple[int, ...]`
- `validate_bins(bins: int) -> int`
- `region_edges(size: int, g: int) -> np.ndarray` — `[g+1]`, `floor(k*size/g)`.
- `quantize_pixels(images: np.ndarray, bins: int) -> np.ndarray` — `[N,H,W] uint8 → [N,H,W] uint8/int16`.
- `region_histograms(quantized: np.ndarray, g: int, bins: int) -> np.ndarray` — `[N, g*g, B]`.
- `normalized_entropy(counts: np.ndarray, bins: int) -> np.ndarray` — `[N, g*g]`.
- `compute_entropy_features(images: np.ndarray, grid_levels, bins) -> np.ndarray` — `[N, sum(g²)] float64`, level order = `grid_levels`.
- `feature_names(grid_levels, template) -> list[str]`.

`src/features/entropy_csv.py`
- `@dataclass(frozen=True, slots=True) EntropyCsvConfig` (paths resolved with `project_root`, levels tuple, bins, metadata cols, template, chunk_size, inverted, overwrite…) with `effective_input_csv` / `effective_output_csv` properties.
- `load_entropy_config(config_path, project_root) -> EntropyCsvConfig` (reuse `_require` pattern; fail fast on missing keys, `.csv` suffix, in≠out).
- `validate_header(header, cfg)` — all metadata cols exist, `px_` columns count == `image_size²`, contiguous, ordered.
- `iter_chunks(cfg)` → yields `(metadata_df, images[N,H,W])`; checks `width==height==image_size` for every row.
- `export_entropy_csv(cfg, dry_run=False) -> EntropyExportSummary` (rows, n_features, columns, elapsed, bytes).
- `write_sidecar(...)`.

`experiments/export_entropy_maps.py`: `build_parser()` (`--config` default `configs/entropy/mvtec_ad.toml`, `--inverted`, `--overwrite`, `--dry-run`, optional `--grid-levels`/`--bins` **overrides? NO** — keep TOML as the single source; only allow `--config`), `main(argv, project_root)`, catches `(OSError, ValueError, KeyError)` → `SystemExit(f"error: {exc}")`, logging via `configs.logging`.

## 3. Stages (checklist for the implementing agent)

**Stage A — Understand/validate (no `src/` yet)**
- [x] Re-read `understanding/06_entropy_pyramid_descriptor/{partition,entropy,descriptor}.py` and `03_entropy_benchmark/entropy_benchmark.py:e1`.
- [x] (Optional, per "understanding first") small script `understanding/07_entropy_maps_vectorized/` that prints `image → quantized → region counts → p·log2 p → E` for one 128×128 image and compares with `descriptor.extract_descriptor`. Use matplotlib to show grids.

**Stage B — Config**
- [x] Create `configs/entropy/mvtec_ad.toml`, `configs/entropy/kolektorsdd2.toml`.
- [x] Add entropy output paths to `configs/base.toml` for both datasets.

**Stage C — Math core (`src/features/entropy_maps.py`)**
- [x] Implement functions above; no constants (levels/bins/template passed in).
- [x] Equal-count exact-`1.0` rule and `0·log0 = 0`.
- [x] Tests (B-tests below) implemented and ready for execution.

**Stage D — CSV pipeline (`src/features/entropy_csv.py`)**
- [x] Config loader + validation, chunked reader, atomic writer, sidecar JSON, logging, `tqdm`.
- [x] Header preserved: metadata columns (source order) then feature columns.

**Stage E — CLI** (`experiments/export_entropy_maps.py`) with docstring of manual commands (same style as `export_mvtec_ad_csv.py`).

**Stage F — Tests (`tests/`)**
`test_entropy_maps.py`
- [x] Constant image → all `E=0`; image with exactly equiprobable `B` bins in each region → `E=1`.
- [x] Vectorized result == `understanding.06` `extract_descriptor(...).entropy_features` on random images and structured images, `atol=1e-12`, for `bins ∈ {16, 8, 32, 256, 7}` (non-divisor of 256 too). Load oracle with `importlib` inside the test only.
- [x] Default `[1,2,4,8]` → 85 columns; `[1,2,4,8,16]` → 341; `[3,5]` (non-power-of-2, non-divisor of size) → 34 and edges identical to `partition_boxes`.
- [x] Names: first `E_1x1_r0_c0`, last `E_8x8_r7_c7`, order `1x1 < 2x2 < …`, row-major, no duplicates.
- [x] Values in `[0,1]`; deterministic across runs and chunk sizes (`chunk_size=1,3,N` identical output).
- [x] Invalid inputs raise `ValueError`: empty/duplicate/zero/negative levels, `g > image_size`, `bins=1/257/True/16.0`.

`test_entropy_csv.py` (build tiny synthetic CSVs in `tmp_path`, 16×16 images, injected `project_root`)
- [x] Both metadata schemas (MVTec 5 cols, Kolektor 3 cols) preserved verbatim and aligned row-by-row.
- [x] `inverted` switches input/output paths; `overwrite=False` raises `FileExistsError`/error if exists; `--dry-run` writes nothing.
- [x] Missing TOML key → `KeyError` with context; missing metadata column / wrong px count / width≠image_size → `ValueError`.
- [x] No `.tmp` left on failure; sidecar JSON contains levels, bins, n_features.
- [x] CLI `main([...])` returns 0 and produces expected header; error path exits with `error:` message.
- [x] Changing `grid_levels`/`bins` in a TOML copy changes columns only via TOML (no code edit).
- [x] Synthetics and unit tests verify complete behavior without mutating real data.

**Stage G — Docs**
- [x] Update `src/README.md`, `experiments/README.md`, `tests/README.md` (+ `README.md` root if it lists commands). `requirements.txt` unchanged (confirm).

**Stage H — Verification (Definition of done - pendiente de ejecución manual por el usuario)**
- [ ] `.venv\Scripts\python.exe -m pytest tests -q` all pass (existing + new).
- [ ] Dry-run both datasets: `.venv\Scripts\python.exe -m experiments.export_entropy_maps --config configs/entropy/kolektorsdd2.toml --dry-run`.
- [ ] Real run (Kolektor first, smaller), then MVTec, plus `--inverted` variants if wanted. Check: rows == source CSV rows, columns == metadata + 85, all features in `[0,1]`, no NaN, header names correct.
- [ ] Cross-check 5–10 random rows from the real output against the `understanding/06` oracle.
- [ ] Record measured elapsed time and RAM peak (don't invent numbers); report anything not verified.

## 4. Edge cases / risks

- **Large CSVs (370 MB)**: never load whole file; `chunksize` + `uint8` dtypes.
- **`bins` not dividing 256**: bins cover unequal intensity counts (documented in 06); same formula `pixel*B//256` keeps parity.
- **`g` does not divide 128** (e.g. 3, 5, 6): proportional floor boundaries, regions differ ±1 px — matches 06; normalization uses `log2(B)` not region size.
- **Float summation overshoot** (`E` slightly > 1): handled by exact-equal-counts rule; no clipping of other values (document if a tolerance check is needed).
- **Metadata column names differ** between datasets: driven by TOML, validated at load.
- **Future x94**: output is a separate "partial" file; later experiments join asymmetry features by the metadata key (`file_name` + object/split), so keep row order identical to the source CSV.
- **Scientific integrity**: oracle kept in `understanding/`, tests compare against it; no results in docs unless measured.

## 5. Decisions confirmed by the user (supersede any earlier default above)

1. **Output path is fixed per dataset in the TOML** (no bins/levels tag in the file name). `bins` and `grid_levels` are instead written **as columns of the CSV**.
2. **Row = one image, 1:1 coupling with the source CSV.** Output columns, in order:
   `[source metadata columns exactly as in the source CSV]` + `bins` + `grid_levels` + `E_1x1_r0_c0 … E_8x8_r7_c7`.
   - MVTec-AD: `object_name, split, type, defect_name, file_name, bins, grid_levels, E_*`.
   - KolektorSDD2: `split, type, file_name, bins, grid_levels, E_*`.
   - All `px_*` columns are **dropped** (replaced by the entropy features). `width`/`height` are **not** kept.
   - `bins` = int (e.g. `16`); `grid_levels` = string joined with a TOML-configured separator (`[entropy_maps.output].levels_separator = "|"`, e.g. `1|2|4|8`; avoids commas inside a CSV cell). Column names `bins`/`grid_levels` also come from TOML (`[entropy_maps.output].bins_column`, `levels_column`).
   - Row order identical to the source CSV (needed for the later x94 join).
3. **No `image_id` column.** The existing metadata already identifies the image uniquely: MVTec-AD = `object_name + split + defect_name + file_name`; KolektorSDD2 = `split + file_name` (`type` is derived from the mask). Stage F adds a test asserting this key is unique in the source CSVs and preserved in the output. The `image_id` option is removed from the design (section 1, item 6) and from the open questions.
4. **Features are a flat 1-D vector per row**; no `width`/`height`, no grid structure besides the column names.

## 6. Where the tests live (correction to Stage F)

- `understanding/` holds the personal lab: `03_entropy_benchmark/test_entropy.py` and `06_entropy_pyramid_descriptor/tests/test_descriptor.py` are lab checks of the reference algorithms. **They stay there untouched.**
- Per `AGENTS.md`, the formal tests for `src/` go in `code/tests/` (same place as `test_csv_export.py`). So `tests/test_entropy_maps.py` and `tests/test_entropy_csv.py` stay in `tests/`.
- The formal tests use the `understanding/06` implementation only as an **oracle** (loaded with `importlib` inside the test, `src/` never imports from `understanding/`). If it can't be imported, the oracle tests are skipped with a clear message rather than failing silently.
- Optional lab script (Stage A) `understanding/07_entropy_maps_vectorized/` is the place for prints/matplotlib comparisons, not for pytest.
- Additional test changes: replace "header = metadata + features" assertions with the new order (`metadata + bins + grid_levels + E_*`), assert no `px_*`/`width`/`height` columns, and assert `bins`/`grid_levels` columns are constant and equal to the TOML values.