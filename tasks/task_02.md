# MVTEC-AD Dataset CSV conversion

## Issue

* The main issue addressed in this file is we have the preprocessed images in `data/preprocessed/mvtec-ad` and we need to create a CSV file in `data/csv/mvtec-ad.csv`. The csv must have the following columns:
    - `object_name`: the object name, e.g. [`grid`, `screw`, `zipper`, `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`, `leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`], one of the 15 categories in the mvtec-ad dataset.
    - `split`: the split, e.g. [`train`, `test`, `ground_truth`], one of the three splits in the mvtec-ad dataset.
    - `type`: the type, e.g. [`good`, `defect`], one of the two types in the mvtec-ad dataset, in the case of defect, the test and ground truth splits are always deffects so we only consider these as one type, same as train is always good.
    - `px_XXXXX`: where XXXXX is the number of pixel from 00000 to (128x128-1), this columns will contain the image as a flat vector, so px_00001 to px_(128x128-1) will have the grayscale value of the pixel.
    - `width`: the width of the image in pixels.
    - `height`: the height of the image in pixels.


### Folder structure

The folder structure where the images are located is the following:

`D:\Publications\2026-10-Symmetry-Entropy\code\data\preprocessed\mvtec-ad` or
`D:\Publications\2026-10-Symmetry-Entropy\code\data\preprocessed\mvtec-ad-inverted`

both path are valids so i can show if use the inverted or the not inverted ones

> Remember not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.


## Plan

> Don't execute anything, just plan it

- [x] Read AGENTS.md file to know the general rules of the project.
- [x] Create a detailed plan for this task and update it here to implement it later for an agent, so it needs to be detailed.
- [x] User review of the decisions (confirmed 2026-10-29: D1–D7 approved as proposed; see "Confirmed decisions").
- [x] Implement Tasks 1–7 of the implementation plan below (code implemented; execution pending manual run).

---

# MVTec-AD CSV Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flatten every preprocessed MVTec-AD PNG (standard or inverted) into one row of a CSV with metadata columns + `px_00000…px_16383` + `width`, `height`.

**Architecture:** A new TOML (`configs/csv/mvtec_ad.toml`) holds every export parameter and points to `configs/base.toml` (dataset paths) and `configs/preprocessing/mvtec_ad.toml` (single source of `target_size`). Pure logic lives in `src/datasets/mvtec_ad_csv.py` (config loading, discovery, labeling, row building, streaming writer). A thin CLI `experiments/export_mvtec_ad_csv.py` mirrors `experiments/preprocess_mvtec_ad.py` (`--config`, `--inverted`, `--overwrite`, `--dry-run`).

**Tech Stack:** Python 3.11, stdlib `csv`/`tomllib`/`pathlib`/`logging`, NumPy, Pillow, tqdm, unittest/pytest (all already in `requirements.txt`; no new dependency). PyTorch/CUDA not involved.

**Spec:** this file (`tasks/task_02.md`, "Issue" section) + `AGENTS.md`.

## Facts measured during planning (2026-10-29)

- Both `data/preprocessed/mvtec-ad` and `data/preprocessed/mvtec-ad-inverted` contain 6612 PNGs and no other files:
  `train/good` 3629, `test/good` 467, `test/<defect>` 1258, `ground_truth/<defect>` 1258.
- Every sampled image is mode `L`, shape `(128, 128)`, `uint8`. Masks contain values `{0, 255}`.
- Mask file names are `<stem>_mask.png` (e.g. `bottle/ground_truth/broken_large/000_mask.png` ↔ `bottle/test/broken_large/000.png`).
- `configs/base.toml` has `csv_path = "data/csv/mvtec-ad/"` (a directory, used by no code: `git grep csv_path` only hits the TOML). `data/csv/mvtec-ad/` exists and is empty. `data/` is git-ignored.
- **Estimate (not measured):** 6612 rows × 16391 columns, ~3.5 bytes/pixel ⇒ CSV ≈ 350–400 MB per variant.

## Confirmed decisions (approved by the user on 2026-10-29; binding for implementation)

- **D1 – `type` comes from the folder name, not from the split.** The spec says "test is always defect", but data shows 467 `test/good` images. Rule: folder == `good` ⇒ `good`, anything else ⇒ `defect`. So `train/good`→good, `test/good`→good, `test/<defect>`→defect, `ground_truth/<defect>`→defect. This rule overrides the sentence in the "Issue" section.
- **D2 – Two extra provenance columns `defect_name` and `file_name`**, placed right after `type`. They pair the 1258 mask rows with their 1258 test images and trace each row back to its file. `defect_name` = folder name (`good`, `broken_large`, …); `file_name` = PNG name (`000.png`, `000_mask.png`). Final column order: `object_name, split, type, defect_name, file_name, px_00000 … px_16383, width, height` (16391 columns for 128×128).
- **D3 – Inverted export goes to its own file.** `base.toml`: `csv_path = "data/csv/mvtec-ad.csv"` (as the spec says) and new `csv_path_inverted = "data/csv/mvtec-ad-inverted.csv"`. Selected with `--inverted` (or `inverted = true` in TOML), same convention as task 01.
- **D4 – Mask pixels stored as they are (`0`/`255`)**, not binarized to `0`/`1`.
- **D5 – Flatten order is row-major (NumPy C order):** `px_index = row * width + col`. `px_00000` = top-left, `px_16383` = bottom-right.
- **D6 – `width`/`height` are read from each image** and must equal `target_size` from `configs/preprocessing/mvtec_ad.toml`; any mismatch aborts (a CSV with a fixed number of `px_` columns cannot hold other sizes).
- **D7 – Streaming writer with stdlib `csv`, not pandas.** One row in memory at a time; written to `<output>.tmp` then `Path.replace` (atomic), so a failure never leaves a half CSV.

## Global Constraints

- No path, number or label vocabulary hardcoded in `src/`; everything comes from TOML. Required TOML keys have **no code defaults** (missing key ⇒ `KeyError`).
- `from __future__ import annotations`, full type hints, `@dataclass(frozen=True, slots=True)` for config/records/summary.
- Paths: `pathlib.Path`; `PROJECT_ROOT = Path(__file__).resolve().parents[1]` only in entry points (CLI, tests, understanding script); every TOML path resolved against it.
- CLI: `build_parser()`, `main(argv: list[str] | None = None) -> int`, `raise SystemExit(main())`; catch `OSError`, `ValueError`, `KeyError` in `main()` ⇒ `raise SystemExit(f"error: {exc}") from exc`.
- Logging via `configs.logging.logger` / `logging.getLogger("research")`; no `print` outside understanding plots/CLI queries.
- Use `code/.venv/Scripts/python.exe` explicitly. No new packages; `requirements.txt` stays unchanged.
- Deterministic row order: `subdatasets` order from `base.toml` → `splits` order from TOML → defect folder sorted → file name sorted.

## Review Focus

- Stray non-image files (`Thumbs.db`, `desktop.ini`, `.DS_Store`) inside class folders ⇒ ignored by the extension filter, not a crash (test in Task 2).
- Selected root missing a category or split folder (e.g. inverted set not generated yet) ⇒ `FileNotFoundError` before any file is written (test in Task 1/2).
- An image with wrong size or mode (leftover 256×256 or RGB file) ⇒ `ValueError` naming the file, and no `.csv` nor `.tmp` left behind (test in Task 4).
- Output CSV already exists and `overwrite = false` ⇒ `FileExistsError`, existing file untouched (test in Task 4).
- Standard and inverted exports never write the same file (test in Task 1).

---

## File map

| Action | Path | Responsibility |
|---|---|---|
| Modify | `configs/base.toml` (`[datasets.mvtecad]`) | `csv_path` → file; add `csv_path_inverted` |
| Create | `configs/csv/mvtec_ad.toml` | All CSV export parameters |
| Create | `src/datasets/__init__.py` | Package marker + public exports |
| Create | `src/datasets/mvtec_ad_csv.py` | Config, discovery, labeling, rows, writer |
| Create | `experiments/export_mvtec_ad_csv.py` | CLI entry point |
| Create | `tests/test_csv_export.py` | Unit + integration tests (synthetic tiny dataset) |
| Create | `understanding/09_verify_mvtec_csv/verify.py` + `README.md` | Round-trip check CSV ↔ PNG + figure |
| Modify | `README.md`, `experiments/README.md`, `RESUME.md`, `tasks/task_02.md` | Docs |

---

### Task 1: Configuration and `load_csv_config`

**Files:**
- Modify: `configs/base.toml` lines 7 (`csv_path`)
- Create: `configs/csv/mvtec_ad.toml`
- Create: `src/datasets/__init__.py`, `src/datasets/mvtec_ad_csv.py`
- Test: `tests/test_csv_export.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True, slots=True)
  class CsvExportConfig:
      source_root: Path            # data/preprocessed/mvtec-ad
      source_root_inverted: Path   # data/preprocessed/mvtec-ad-inverted
      output_csv: Path             # data/csv/mvtec-ad.csv
      output_csv_inverted: Path    # data/csv/mvtec-ad-inverted.csv
      subdatasets: tuple[str, ...]
      splits: tuple[str, ...]
      image_extensions: tuple[str, ...]   # lower-case
      good_folder: str
      label_good: str
      label_defect: str
      image_size: int                     # target_size of preprocessing TOML
      pixel_column_prefix: str
      pixel_index_digits: int
      inverted: bool
      overwrite: bool
      @property
      def effective_source_root(self) -> Path: ...
      @property
      def effective_output_csv(self) -> Path: ...

  def load_csv_config(config_path: Path, project_root: Path) -> CsvExportConfig
  ```

- [x] **Step 1: Edit `configs/base.toml`** `[datasets.mvtecad]`:
  ```toml
  csv_path = "data/csv/mvtec-ad.csv"
  csv_path_inverted = "data/csv/mvtec-ad-inverted.csv"
  ```

- [x] **Step 2: Create `configs/csv/mvtec_ad.toml`** (exact content; comments in English like the preprocessing TOML):
  ```toml
  # Export of preprocessed MVTec-AD PNGs to one flat CSV (one row per image).
  # Dataset paths live in configs/base.toml [datasets.mvtecad];
  # image size comes from configs/preprocessing/mvtec_ad.toml [preprocessing].target_size.

  [csv]
  base_config = "configs/base.toml"
  dataset_key = "mvtecad"
  preprocessing_config = "configs/preprocessing/mvtec_ad.toml"
  inverted = false                 # true -> read *-inverted root, write csv_path_inverted
  overwrite = false                # false -> abort if the output CSV exists
  image_extensions = [".png"]
  splits = ["train", "test", "ground_truth"]

  [csv.labels]
  good_folder = "good"             # class folder name meaning "no defect"
  good = "good"                    # value written in column `type`
  defect = "defect"

  [csv.pixels]
  column_prefix = "px_"
  index_digits = 5                 # px_00000 ... px_16383 for 128x128
  ```

- [x] **Step 3: Write failing tests** in `tests/test_csv_export.py` (unittest style, same header/`PROJECT_ROOT` bootstrap as `tests/test_preprocessing.py`). Add module helpers used by all tasks:
  - `_write_png(path: Path, array: np.ndarray) -> None` (mode `L`).
  - `_make_dataset(root: Path, size: int) -> None` creating, under `root`:
    `objA/train/good/000.png`, `objA/test/good/000.png`, `objA/test/crack/000.png`, `objA/ground_truth/crack/000_mask.png`, `objB/train/good/000.png`, `objB/test/good/000.png`, `objB/ground_truth/` (empty dir). Pixel content: `np.arange(size*size, dtype=np.uint8).reshape(size, size)` (+ distinct offset per file so rows differ; mask = values in `{0,255}`).
  - `_make_project(tmp: Path, size: int = 4, **csv_overrides) -> Path` writing tmp `configs/base.toml` (subdatasets `["objA","objB"]`, paths under tmp `data/…`), `configs/preprocessing/mvtec_ad.toml` (`[preprocessing] target_size = size`), `configs/csv/mvtec_ad.toml` (Step 2 content with overrides), building both dataset roots via `_make_dataset`; returns the CSV TOML relative path.

  Tests:
  ```python
  def test_load_config_resolves_paths_and_size(self):
      cfg = load_csv_config(rel_toml, tmp)
      self.assertEqual(cfg.image_size, 4)
      self.assertEqual(cfg.effective_source_root, (tmp / "data/preprocessed/mvtec-ad").resolve())
      self.assertEqual(cfg.effective_output_csv, (tmp / "data/csv/mvtec-ad.csv").resolve())
      self.assertEqual(cfg.splits, ("train", "test", "ground_truth"))

  def test_inverted_switches_source_and_output(self):
      cfg = replace(load_csv_config(rel_toml, tmp), inverted=True)
      self.assertTrue(str(cfg.effective_source_root).endswith("mvtec-ad-inverted"))
      self.assertTrue(cfg.effective_output_csv.name == "mvtec-ad-inverted.csv")
      self.assertNotEqual(cfg.output_csv, cfg.output_csv_inverted)

  def test_missing_required_key_raises_keyerror(self):   # e.g. delete [csv.pixels].index_digits
  def test_digits_too_small_raises_valueerror(self):     # size=4 (16 px) with index_digits=1
  def test_missing_source_category_raises_filenotfound(self):  # remove objB dir from standard root
  def test_same_output_paths_raise_valueerror(self):     # csv_path == csv_path_inverted
  ```

- [ ] **Step 4: Run** `.venv\Scripts\python.exe -m pytest tests/test_csv_export.py -q` → FAIL (`ModuleNotFoundError: src.datasets`).

- [x] **Step 5: Implement `load_csv_config`.** Read CSV TOML → `[csv]`; read `base_config` → `datasets[dataset_key]` keys `output_path`, `output_path_inverted` (these are the preprocessing outputs = CSV sources), `csv_path`, `csv_path_inverted`, `subdatasets`; read `preprocessing_config` → `preprocessing.target_size`. Use a small helper `_require(table: dict, key: str, where: str) -> object` that raises `KeyError(f"Missing '{key}' in {where}")`. Validations (fail fast): `image_size >= 1`; `index_digits >= len(str(image_size*image_size - 1))`; `inverted`/`overwrite` are `bool`; `splits` non-empty; output CSVs differ; output suffix is `.csv`; the **effective** source root exists and contains every subdataset folder (only the selected variant is checked, so standard export works even if inverted was never generated — check it lazily in a `validate_sources(cfg) -> None` called by the writer, and also in the loader for the TOML's own `inverted` value). Export public names in `src/datasets/__init__.py`.

- [ ] **Step 6: Run tests** → PASS. Also run `tests/test_preprocessing.py` to confirm the `base.toml` edit broke nothing.

- [ ] **Step 7: Commit** `git add configs src/datasets tests/test_csv_export.py && git commit -m "task 02: csv export config"`

---

### Task 2: Discovery and labeling

**Files:** Modify `src/datasets/mvtec_ad_csv.py`; Test `tests/test_csv_export.py`

**Interfaces:**
- Consumes: `CsvExportConfig`.
- Produces:
  ```python
  @dataclass(frozen=True, slots=True)
  class ImageRecord:
      object_name: str
      split: str
      type: str          # cfg.label_good | cfg.label_defect
      defect_name: str   # class folder name, e.g. "good", "crack"
      file_name: str
      path: Path

  def classify_type(defect_name: str, cfg: CsvExportConfig) -> str
  def iter_records(cfg: CsvExportConfig) -> Iterator[ImageRecord]
  ```

- [x] **Step 1: Failing tests**
  ```python
  def test_classify_type(self):
      self.assertEqual(classify_type("good", cfg), "good")
      self.assertEqual(classify_type("crack", cfg), "defect")

  def test_iter_records_order_and_labels(self):
      got = [(r.object_name, r.split, r.type, r.defect_name, r.file_name) for r in iter_records(cfg)]
      self.assertEqual(got, [
          ("objA", "train", "good", "good", "000.png"),
          ("objA", "test", "defect", "crack", "000.png"),
          ("objA", "test", "good", "good", "000.png"),
          ("objA", "ground_truth", "defect", "crack", "000_mask.png"),
          ("objB", "train", "good", "good", "000.png"),
          ("objB", "test", "good", "good", "000.png"),
      ])

  def test_non_image_files_are_ignored(self):   # add Thumbs.db and notes.txt in objA/train/good
  def test_missing_split_folder_raises(self):   # delete objB/test -> FileNotFoundError
  ```
  (Empty `objB/ground_truth` is valid and yields no rows.)

- [ ] **Step 2: Run** → FAIL (`ImportError`).
- [x] **Step 3: Implement.** For each `object_name` in `cfg.subdatasets`, each `split` in `cfg.splits`: split dir must exist (else `FileNotFoundError`); iterate `sorted(d for d in split_dir.iterdir() if d.is_dir())`, then `sorted(files)` filtered by `suffix.lower() in cfg.image_extensions`. Do not recurse deeper than `<object>/<split>/<class>/<file>`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `"task 02: record discovery and labeling"`

---

### Task 3: Header and row building

**Files:** Modify `src/datasets/mvtec_ad_csv.py`; Test `tests/test_csv_export.py`

**Interfaces:**
- Consumes: `CsvExportConfig`, `ImageRecord`.
- Produces:
  ```python
  METADATA_COLUMNS: tuple[str, ...] = ("object_name", "split", "type", "defect_name", "file_name")
  SIZE_COLUMNS: tuple[str, ...] = ("width", "height")
  def pixel_column_names(n_pixels: int, prefix: str, digits: int) -> list[str]
  def build_header(cfg: CsvExportConfig) -> list[str]
  def load_pixels(path: Path, image_size: int) -> np.ndarray   # [H, W] uint8
  def record_to_row(record: ImageRecord, pixels: np.ndarray) -> list[str | int]
  ```
  (Column *names* are the CSV schema, defined once as module tuples matching `ImageRecord` fields; they are not tunable parameters.)

- [x] **Step 1: Failing tests**
  ```python
  def test_pixel_column_names_128(self):
      names = pixel_column_names(128 * 128, "px_", 5)
      self.assertEqual((len(names), names[0], names[-1]), (16384, "px_00000", "px_16383"))

  def test_header_layout(self):   # size=4
      h = build_header(cfg)
      self.assertEqual(h[:5], ["object_name", "split", "type", "defect_name", "file_name"])
      self.assertEqual(h[5], "px_00000"); self.assertEqual(h[20], "px_00015")
      self.assertEqual(h[-2:], ["width", "height"]); self.assertEqual(len(h), 5 + 16 + 2)

  def test_row_is_row_major(self):
      px = np.arange(16, dtype=np.uint8).reshape(4, 4); px[1, 2] = 200
      row = record_to_row(rec, px)
      self.assertEqual(row[5 + 1 * 4 + 2], 200)          # px_index = row*width + col
      self.assertEqual(row[-2:], [4, 4])

  def test_load_pixels_rejects_wrong_size(self):  # 5x5 PNG with image_size=4 -> ValueError mentioning path
  def test_load_pixels_rejects_non_L_mode(self):  # RGB PNG -> ValueError
  def test_mask_values_preserved(self):           # mask PNG -> set(row px) == {0, 255}
  ```
- [ ] **Step 2: Run** → FAIL.
- [x] **Step 3: Implement.** `load_pixels`: open with PIL, require `mode == "L"` and `size == (image_size, image_size)` (no silent conversion — preprocessing already guaranteed this; a mismatch means stale data), return `np.asarray(img, dtype=np.uint8)`. `record_to_row`: metadata values + `pixels.ravel(order="C").tolist()` + `[width, height]` with `height, width = pixels.shape`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `"task 02: csv header and rows"`

---

### Task 4: Streaming writer `export_csv`

**Files:** Modify `src/datasets/mvtec_ad_csv.py`; Test `tests/test_csv_export.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  ```python
  @dataclass(frozen=True, slots=True)
  class CsvExportSummary:
      output_csv: Path
      rows: int
      counts: dict[tuple[str, str], int]   # (split, type) -> rows; frozen dataclass, dict built once
      columns: int
      bytes_written: int                   # 0 on dry run
      elapsed_s: float
      dry_run: bool

  def export_csv(cfg: CsvExportConfig, dry_run: bool = False) -> CsvExportSummary
  ```

- [x] **Step 1: Failing tests**
  ```python
  def test_export_writes_expected_csv(self):
      s = export_csv(cfg)
      with s.output_csv.open(newline="") as f: rows = list(csv.reader(f))
      self.assertEqual(rows[0], build_header(cfg))
      self.assertEqual(len(rows) - 1, 6); self.assertEqual(s.rows, 6)
      self.assertEqual(s.counts[("test", "good")], 2)
      # first data row equals the flattened objA/train/good/000.png
      expected = np.asarray(Image.open(src_root / "objA/train/good/000.png")).ravel().tolist()
      self.assertEqual([int(v) for v in rows[1][5:-2]], expected)
      self.assertFalse(s.output_csv.with_suffix(".csv.tmp").exists())

  def test_inverted_export_reads_inverted_root(self):  # inverted root pixels = 255 - standard; compare row 1
  def test_existing_output_without_overwrite_raises(self):  # pre-create file "keep"; FileExistsError; content still "keep"
  def test_overwrite_replaces_output(self):
  def test_bad_image_aborts_without_partial_files(self):    # one 5x5 PNG -> ValueError; no .csv, no .tmp
  def test_dry_run_writes_nothing(self):                    # s.rows == 6, file absent, bytes_written == 0
  ```
- [ ] **Step 2: Run** → FAIL.
- [x] **Step 3: Implement.** Order: `validate_sources(cfg)` → if output exists and not `overwrite` ⇒ `FileExistsError` → `records = list(iter_records(cfg))` (paths only, cheap) → if `dry_run` count and return. Otherwise `output.parent.mkdir(parents=True, exist_ok=True)`, open `tmp = output.with_suffix(output.suffix + ".tmp")` with `newline=""`, `encoding="utf-8"`, `csv.writer`; write header, then for each record (tqdm, `unit="img"`) `writer.writerow(record_to_row(r, load_pixels(r.path, cfg.image_size)))`. On any exception: close, `tmp.unlink(missing_ok=True)`, re-raise. On success `tmp.replace(output)`. Log totals, per `(split, type)` counts, columns, MB and elapsed via `logger.info`.
- [ ] **Step 4: Run** → PASS (whole file).
- [ ] **Step 5: Commit** `"task 02: streaming csv writer"`

---

### Task 5: CLI `experiments/export_mvtec_ad_csv.py`

**Files:** Create `experiments/export_mvtec_ad_csv.py`; Test `tests/test_csv_export.py`

**Interfaces:**
- Consumes: `load_csv_config`, `export_csv`, `configs.logging.configure_logging/logger`.
- Produces: `build_parser() -> argparse.ArgumentParser`, `main(argv: list[str] | None = None, project_root: Path = PROJECT_ROOT) -> int`. (`project_root` parameter is dependency injection so tests can point at the tmp project.)

- [x] **Step 1: Failing tests**
  ```python
  def test_cli_dry_run_returns_zero_and_writes_nothing(self):
      self.assertEqual(cli_main(["--config", rel_toml, "--dry-run"], project_root=tmp), 0)
  def test_cli_inverted_writes_inverted_csv(self):
      self.assertEqual(cli_main(["--config", rel_toml, "--inverted"], project_root=tmp), 0)
      self.assertTrue((tmp / "data/csv/mvtec-ad-inverted.csv").is_file())
  def test_cli_existing_output_exits_with_error(self):
      with self.assertRaises(SystemExit) as ctx: cli_main([...], project_root=tmp)   # second run, no --overwrite
      self.assertIn("error:", str(ctx.exception.code))
  ```
- [ ] **Step 2: Run** → FAIL.
- [x] **Step 3: Implement** following `experiments/preprocess_mvtec_ad.py` exactly (module docstring with manual commands, `sys.path` bootstrap, `--config` default `Path("configs/csv/mvtec_ad.toml")`, flags `--inverted`, `--overwrite`, `--dry-run` applied with `dataclasses.replace`, log effective source/output/inverted/overwrite/image size, catch `(OSError, ValueError, KeyError)` ⇒ `SystemExit(f"error: {exc}")`).
- [ ] **Step 4: Run** `.venv\Scripts\python.exe -m pytest tests/test_csv_export.py tests/test_preprocessing.py -q` → all PASS.
- [ ] **Step 5: Commit** `"task 02: csv export cli"`

---

### Task 6: Real-data run and round-trip verification

**Files:** Create `understanding/09_verify_mvtec_csv/verify.py`, `understanding/09_verify_mvtec_csv/README.md`

**Interfaces:**
- Consumes: `load_csv_config`, `build_header`, generated CSVs. `verify.py` has its own `build_parser()`/`main(argv)` with `--config`, `--inverted`, `--samples` (int, default from argparse), `--seed` (int).

- [ ] **Step 1: Dry runs** (expected: 6612 rows; counts train/good 3629, test/good 467, test/defect 1258, ground_truth/defect 1258):
  ```powershell
  .venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --dry-run
  .venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --inverted --dry-run
  ```
- [ ] **Step 2: Full exports** (record elapsed time and file size from the log in the README; these are measurements):
  ```powershell
  .venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv
  .venv\Scripts\python.exe -m experiments.export_mvtec_ad_csv --inverted
  ```
- [x] **Step 3: Write `verify.py`.** Stream the CSV with `csv.reader` (do not load 380 MB into pandas): check header == `build_header(cfg)`, row count == 6612, every row has the header length, `width == height == image_size`, all px in `[0, 255]`, mask rows px ⊂ `{0, 255}`; for `--samples` rows chosen with `numpy.random.default_rng(seed)` reshape px to `[H, W]` and assert exact equality with the PNG at `source_root/object/split/defect_name/file_name`. Save a matplotlib figure `csv_roundtrip[_inverted].png` (PNG vs reconstructed vs abs diff, plus one image/mask pair). Log a summary; exit 1 on any mismatch.
- [ ] **Step 4: Run** `verify.py` for both variants → exit 0, 0 mismatches. Write measured results in `understanding/09_verify_mvtec_csv/README.md`.
- [ ] **Step 5: Commit** (code + README + figures only; `data/` is git-ignored) `"task 02: csv round-trip verification"`

---

### Task 7: Documentation

**Files:** Modify `README.md` (new section after "Preprocesamiento de datasets (MVTec-AD)"), `experiments/README.md` (new "Experimento 02: Exportación CSV de MVTec-AD"), `RESUME.md` (directory map, commands, file index, current state), `tasks/task_02.md` (tick checkboxes, add measured results).

- [x] **Step 1:** Document commands (tests, dry-run, standard, inverted, overwrite, verify), the column schema (D2, D5), the `type` rule (D1), output paths (D3), and the measured sizes/times from Task 6.
- [ ] **Step 2:** Run final check `.venv\Scripts\python.exe -m pytest tests/test_csv_export.py tests/test_preprocessing.py -q` → PASS. `requirements.txt` unchanged (no new deps) — state this explicitly.
- [ ] **Step 3: Commit** `"task 02 (done) - mvtec-ad csv export"`
