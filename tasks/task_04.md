# KolektorSDD2 Dataset CSV conversion

## Issue

* The main issue addressed in this task is now that the kolektorsdd2 dataset has been converted to a csv file. This is similar to the `task_02.md`, i want the dataset in the folder data/csv (both inverted and not inverted) into csv files for a more easy consuming. Take as base the implementation of `task_02.md` but now it's a different dataset. The columns that i need in this dataset are:
    - `split`: the split, e.g. [`train`, `test`], one of the two splits in the kolektorsdd2 dataset.
    - `type`: the type, e.g. [`good`, `defect`], one of the two types in the mvtec-ad dataset, in the case of defect.
    - `px_XXXXX`: where XXXXX is the number of pixel from 00000 to (128x128-1), this columns will contain the image as a flat vector, so px_00001 to px_(128x128-1) will have the grayscale value of the pixel.
    - `width`: the width of the image in pixels.
    - `height`: the height of the image in pixels.
* actually since the code is doing made by good practices, a bunch of the code can be reused but maybe a new toml file needs to be created, i don't know, analyze it and plan it.



### Folder structure

The folder structure where the images are located is the following:

`D:\Publications\2026-10-Symmetry-Entropy\code\data\preprocessed\kolektorsdd2` or
`D:\Publications\2026-10-Symmetry-Entropy\code\data\preprocessed\kolektorsdd2-inverted`

both path are valids so i can show if use the inverted or the not inverted ones

> Remember not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.


## Plan

> Don't execute anything, just plan it

- [x] Read AGENTS.md file to know the general rules of the project.
- [x] Create a detailed plan for this task and update it here to implement it later for an agent, so it needs to be detailed.
- [x] Update this file with the plan, stages and everything to resolve and implement the task.



---

# KolektorSDD2 CSV Export Implementation Plan

> **For agentic workers:** use `superpowers:executing-plans` (or subagent-driven-development). Steps use `- [ ]` checkboxes. Each task: failing test → run (FAIL) → implement → run (PASS) → commit. Plan only; nothing has been executed yet.

**Goal:** Flatten every preprocessed KolektorSDD2 PNG (standard or inverted) into one CSV row: `split, type, file_name, px_00000…px_16383, width, height`.

**Architecture:** Reuse the task_02 design. New TOML `configs/csv/kolektorsdd2.toml`; new module `src/datasets/kolektorsdd2_csv.py` (config loading, discovery, streaming writer) that **imports** the dataset-agnostic helpers already in `src/datasets/mvtec_ad_csv.py` (`pixel_column_names`, `load_pixels`, `CsvExportSummary`); thin CLI `experiments/export_kolektorsdd2_csv.py` mirroring `experiments/export_mvtec_ad_csv.py`. No new dependency; PyTorch/CUDA not involved.

## Facts measured during planning (2026-10-30)

- Sources (identical layout for `kolektorsdd2` and `kolektorsdd2-inverted`): `train/good` 2085, `train/defect` 246, `test/good` 894, `test/defect` 110 ⇒ **3335 images**. No files directly under `train/` or `test/`, no non-PNG files.
- Sampled images (standard and inverted): mode `L`, size `(128, 128)`. File names are numeric ids (`10021.png`).
- Layout is `<root>/<split>/<good|defect>/<id>.png`: **no `object_name` level** (KolektorSDD2 is a single object) and **no masks** (masks were only used during preprocessing to decide good/defect). So `type` = class folder name directly.
- `configs/base.toml` `[datasets.kolektorsdd2]` has `input_path`, `output_path = "data/preprocessed/KolektorSDD2"`, `output_path_inverted`, `splits`, but **no `csv_path`/`csv_path_inverted`**. The issue's paths are lowercase (`kolektorsdd2`); Windows is case-insensitive so both work, but see D6.
- `data/csv/` already has `mvtec-ad.csv` and `mvtec-ad-inverted.csv`.
- **Estimate (not measured):** 3335 rows × 16387 columns ≈ 180–200 MB per variant.

## Decisions (proposed; same spirit as task_02 D1–D7, review before implementing)

- **D1 – Columns (user-confirmed 2026-10-30, `file_name` added as in task_02 D2):** `split, type, file_name, px_00000 … px_16383, width, height` (16387 columns). `file_name` = PNG name (`10021.png`) and sits right after `type`. No `object_name` (single object) and no `defect_name` (the class folder is already `type`, so it would be redundant). Row order is deterministic (split in TOML order → class folder sorted → file name sorted). Row ↔ PNG path is `<root>/<split>/<type folder>/<file_name>`.
- **D2 – `type` = class folder name mapped via TOML** (`good_folder = "good"` ⇒ `good`, otherwise `defect`), reusing task_02's rule; no hardcoded vocabulary in `src/`.
- **D3 – Separate outputs:** add to `base.toml [datasets.kolektorsdd2]`: `csv_path = "data/csv/kolektorsdd2.csv"` and `csv_path_inverted = "data/csv/kolektorsdd2-inverted.csv"`. `--inverted` (or `inverted = true`) selects the inverted source and output.
- **D4 – Pixels as stored (0–255 `uint8`)**, row-major (`px_index = row*width + col`, `px_00000` top-left), as in task_02.
- **D5 – `width`/`height` read from each image**, must equal `target_size` from `configs/preprocessing/kolektorsdd2.toml` (single source of truth); mismatch ⇒ abort.
- **D6 – Path casing:** `base.toml` already uses `data/preprocessed/KolektorSDD2`; keep it (do not rename) and make no code depend on casing.
- **D7 – Streaming `csv.writer`, `.tmp` + atomic `Path.replace`**, `overwrite=false` by default, `--dry-run` writes nothing (same as task_02).
- **D8 – Reuse without touching working task_02 behavior:** do **not** refactor `mvtec_ad_csv.py` beyond making reused helpers importable (they already are public, except `_require`). Duplicate the small `_require` (≈4 lines) or import it; if more than discovery/config is duplicated later, extract a `csv_common.py` in a separate task.

## Global constraints

- No path, number or label hardcoded in `src/`; every required TOML key has no code default (missing ⇒ `KeyError`).
- `from __future__ import annotations`, full type hints, `@dataclass(frozen=True, slots=True)`, `pathlib`, `logging` (no `print`), `argparse` + `main(argv=None, project_root=PROJECT_ROOT) -> int`, `SystemExit(f"error: {exc}")` on `OSError/ValueError/KeyError`.
- Run with `.venv\Scripts\python.exe` only; no global installs.

## File map

| Action | File |
|---|---|
| Modify | `configs/base.toml` (add 2 csv keys) |
| Create | `configs/csv/kolektorsdd2.toml` |
| Create | `src/datasets/kolektorsdd2_csv.py` |
| Create | `experiments/export_kolektorsdd2_csv.py` |
| Create | `tests/test_kolektorsdd2_csv_export.py` |
| Create | `understanding/11_verify_kolektorsdd2_csv/{verify.py,README.md}` (check next free lab number first) |
| Modify | `README.md`, `experiments/README.md`, `RESUME.md`, this file |

## Tasks

### Task 1: Configuration and `load_kolektor_csv_config`

- [x] Add to `configs/base.toml` under `[datasets.kolektorsdd2]`: `csv_path`, `csv_path_inverted` (D3).
- [x] Create `configs/csv/kolektorsdd2.toml`:
  ```toml
  [csv]
  base_config = "configs/base.toml"
  dataset_key = "kolektorsdd2"
  preprocessing_config = "configs/preprocessing/kolektorsdd2.toml"
  inverted = false
  overwrite = false
  image_extensions = [".png"]
  splits = ["train", "test"]

  [csv.labels]
  good_folder = "good"
  good = "good"
  defect = "defect"

  [csv.pixels]
  column_prefix = "px_"
  index_digits = 5
  ```
- [x] Tests (tmp project with synthetic TOMLs, size 4): config loads with expected resolved paths; `effective_source_root/effective_output_csv` follow `inverted`; missing key ⇒ `KeyError`; `index_digits` too small ⇒ `ValueError`; equal/non-`.csv` outputs ⇒ `ValueError`; missing source dir ⇒ `FileNotFoundError`.
- [x] Implement `KolektorCsvConfig` (frozen, slots; fields as `CsvExportConfig` minus `subdatasets`) and `load_kolektor_csv_config(config_path, project_root)` following `load_csv_config` (use `target_size` from `[preprocessing]`; `subdatasets` is not read).
- [ ] Commit `task 04: kolektorsdd2 csv config`.

### Task 2: Discovery and labeling

- [x] Tests: records sorted by split (TOML order), class folder, file; `type` mapping good/defect; non-image files ignored; counts on synthetic tree; missing split dir ⇒ `FileNotFoundError`.
- [x] Implement `KolektorRecord(split, type, file_name, path)`, `classify_type`, `iter_records(cfg)` (layout `root/split/class/file`).
- [ ] Commit `task 04: kolektorsdd2 discovery`.

### Task 3: Header and row building

- [x] Tests: `build_header` == `["split","type","file_name"] + px_names + ["width","height"]`; for size 128 length 16387 and last pixel `px_16383`; `record_to_row` (`split, type, file_name` first) row-major flatten of a known `np.arange` array, `width`/`height` last.
- [x] Implement using reused `pixel_column_names` and `load_pixels`. Metadata (`split, type, file_name`) and size column names come from module constants (header schema, like task_02) — not paths or numbers.
- [ ] Commit `task 04: kolektorsdd2 header and rows`.

### Task 4: Streaming writer `export_kolektor_csv`

- [x] Tests: output written with header + rows; `.tmp` removed; existing file without overwrite ⇒ `FileExistsError`; overwrite works; bad image (wrong size/mode) ⇒ `ValueError`, no `.csv`, no `.tmp`; `dry_run` writes nothing and returns counts.
- [x] Implement exactly like `export_csv` in task_02 (validate sources → exists check → records → dry-run → tmp write with tqdm → atomic replace; log rows, columns, MB, elapsed, counts per `(split, type)`). Return the reused `CsvExportSummary`.
- [ ] Commit `task 04: kolektorsdd2 csv writer`.

### Task 5: CLI

- [x] Tests: dry-run returns 0, writes nothing; `--inverted` writes `kolektorsdd2-inverted.csv`; second run without `--overwrite` ⇒ `SystemExit` with `"error:"`.
- [x] Implement `experiments/export_kolektorsdd2_csv.py` copying the structure of `experiments/export_mvtec_ad_csv.py` (default `--config configs/csv/kolektorsdd2.toml`, flags `--inverted --overwrite --dry-run`, docstring with manual commands).
- [ ] Commit `task 04: kolektorsdd2 csv cli`.

### Task 6: Real-data run and verification

- [ ] Dry runs; expected 3335 rows, `train/good 2085, train/defect 246, test/good 894, test/defect 110`:
  ```powershell
  .venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --dry-run
  .venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --inverted --dry-run
  ```
- [ ] Full exports (record elapsed time and size from the log as **measurements**):
  ```powershell
  .venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv
  .venv\Scripts\python.exe -m experiments.export_kolektorsdd2_csv --inverted
  ```
- [x] Write `understanding/11_verify_kolektorsdd2_csv/verify.py` (model on `understanding/09_verify_mvtec_csv/verify.py`; own `build_parser()`/`main(argv)` with `--config --inverted --samples --seed`): stream with `csv.reader`; check header == `build_header(cfg)`, row count == records from `iter_records`, row lengths, `width == height == image_size`, px in `[0,255]`, `split`/`type`/`file_name` match record order; for `--samples` random rows (`numpy.random.default_rng(seed)`) reshape to `[H, W]` and assert exact equality with the PNG; save matplotlib figure (PNG vs reconstructed vs diff); exit 1 on any mismatch.
- [ ] Run verify for both variants → exit 0; write measured results in the lab README.
- [ ] Commit (code + README + figures; `data/` is git-ignored) `task 04: kolektorsdd2 csv verification`.

### Task 7: Documentation

- [x] Update `README.md` (new section after the CSV one), `experiments/README.md` (new experiment entry), `RESUME.md` (tree, commands, file index, state), and tick checkboxes + add measured results to this file. `requirements.txt` needs no change.

## Definition of done

`implementation → execution → tests → expected output`: all new tests and `tests/test_csv_export.py` pass; both CSVs exist with 3335 rows × 16387 columns; verify scripts exit 0. Report anything not executed.
