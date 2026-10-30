# KolektorSDD2 preparation

## Issue

* The main issue addressed in this file is that the dataset KolektorSDD2 is not ready for the analysis. It contains 2 folders train and test but there is not clear which are the good ones and which are the bad ones.
* To prevent this, the creators of the dataset put the code as follows `20000_GT.png` is the image masked (most likely in binary) and `20000.png` is the image unmasked (most likely in RGB or gray scale, i don't really know). So the convencion is _GT is masked and the just number is the original image.
* I would like you to create in both train and test folders, some folders that can be labeled as train/good and traind/defect. we will know that there is a defect if the mask has values greater than 0 (i guess, not sure). we should create a script for this.
* also if the images are not in grayscale, we need to move them to grayscale in the separation of the folders. actually we need to check the gray scale, do it if they are not in grayscale and resized the same as the mvtec-ad dataset, with a letterboxing and do them 128x128, also implement an --inverted parameter into the scripts.
* basically separate by good and defects -> check/convert to gray scale -> resize with letterboxing/padding same as in the task_01.md


### Folder structure

The folder structure where the images are located is the following:

`D:\Publications\2026-10-Symmetry-Entropy\code\data\raw\KolektorSDD2`
and for the resized and transformed the path will be
`D:\Publications\2026-10-Symmetry-Entropy\code\data\processed\KolektorSDD2`


> Remember not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.


## Plan

> Don't execute anything, just plan it

- [x] Read AGENTS.md file to know the general rules of the project.
- [x] Create a detailed plan for this task and update it here to implement it later for an agent, so it needs to be detailed.
- [x] Update this file with the plan, stages and everything to resolve and implement the task.

## Findings (measured on `data/raw/KolektorSDD2`)

| Split | Files | Pairs `N.png` + `N_GT.png` | Defective (mask max > 0) | Good |
|-------|-------|----------------------------|--------------------------|------|
| train | 4664  | 2331                       | 246                      | 2085 |
| test  | 2008  | 1004                       | 110                      | 894  |

- Images are mode `RGB`; masks are mode `L` with values exactly `{0, 255}` (binary). A defect is therefore `mask.max() > 0`.
- Every image has the same size as its mask. Sizes are **not constant** (~229-231 x ~634-639, portrait, aspect ~1:2.8).
- **Orphan files in `train/`**: `10301 (copy).png` and `10301_GT (copy).png` (OS copy artifacts). They do not match `^\d+(_GT)?\.png$`, so the pairing step must ignore them (log a warning, never process them).
- Letterboxing a ~230x636 image to 128x128 yields content of ~46x128 px (about 64% of the canvas is padding). This is the same rule as task_01 and is requested, but it is a methodological cost: defects shrink about 5x. Recorded as a decision, not changed.
- Existing reusable code: `src/preprocessing/letterbox.py` (`to_grayscale`, `letterbox_to_square`, `invert_intensity`), config pattern in `configs/preprocessing/mvtec_ad.toml`, CLI pattern in `experiments/preprocess_mvtec_ad.py`.

## Plan (agent reviewed and updated)

Follows AGENTS.md: no hardcoded values in `src/`, `pathlib`, `from __future__ import annotations`, frozen slotted dataclasses, `main(argv)`, `argparse`, `logging`.

### Decisions

1. **Single pass, raw -> processed.** Classification (good/defect), grayscale check/convert and letterbox happen in one pass, writing directly to `data/processed/KolektorSDD2`. The raw folder is never modified (no "move"; the raw data is kept as the source of truth).
2. **Output layout** (paths from the task):
   ```
   data/processed/KolektorSDD2/
     train/good/20000.png     train/defect/20003.png
     test/good/...            test/defect/...
   ```
   **Masks are NOT saved** (user decision): they are only read, at full resolution, to decide good vs defect.
3. **Inverted** (`--inverted`): same as task_01. It inverts image intensity only (`255 - x`) and never masks. The output goes to a separate root (`data/processed/KolektorSDD2-inverted`), so the two variants do not mix.
4. **Defect rule**: `mask.max() > 0`, read from config as `defect_threshold = 0` (strictly greater than).
5. **Grayscale**: the code checks `image.mode`. `L` passes through, otherwise converts with ITU-R 601 (reuses `to_grayscale`). The summary counts how many images were converted.
6. **Reuse `src/preprocessing/letterbox.py` unchanged.** New code only handles dataset layout, pairing and classification.
7. Masks are never resized nor written, so no mask resampling or alignment logic is needed. Images use LANCZOS and pad value 0, same as task_01 (confirmed by user).

### Stage 1: Configuration (no hardcoding outside TOML)

- `configs/base.toml`: add `[datasets.kolektorsdd2]`:
  ```toml
  name = "KolektorSDD2"
  input_path = "data/raw/KolektorSDD2"
  output_path = "data/processed/KolektorSDD2"
  output_path_inverted = "data/processed/KolektorSDD2-inverted"
  splits = ["train", "test"]
  ```
- New `configs/preprocessing/kolektorsdd2.toml`:
  ```toml
  [preprocessing]
  base_config = "configs/base.toml"
  dataset_key = "kolektorsdd2"
  target_size = 128
  pad_value = 0
  padding_anchor = "center"
  overwrite = false

  [preprocessing.images]
  extension = ".png"
  resampling = "LANCZOS"
  convert_to_grayscale = true
  invert = false

  [preprocessing.masks]
  suffix = "_GT"
  defect_threshold = 0            # defect if mask.max() > threshold
  good_folder = "good"
  defect_folder = "defect"

  [preprocessing.pairing]
  stem_pattern = '^\d+$'          # valid sample ids; anything else (e.g. "10301 (copy)") is skipped
  ```
- Fail fast in the loader on missing keys, a non-existent `input_path`, an invalid resampling name, `target_size < 1`, or an empty `splits`.

### Stage 2: Pure logic in `src/preprocessing/kolektorsdd2.py`

- `KolektorConfig` (`@dataclass(frozen=True, slots=True)`) with `effective_output_root` as in `PreprocessConfig`.
- `KolektorSummary` (frozen dataclass): `pairs`, `good`, `defect`, `converted_to_gray`, `skipped_existing`, `skipped_invalid`, `failed`, `elapsed_s`.
- `load_config(config_path, project_root)`: reads both TOMLs and resolves paths against `project_root`.
- `iter_pairs(split_dir, cfg) -> Iterator[SamplePair]`: pairs `<id>.png` with `<id>_GT.png`, ignores non-matching stems, and logs a warning for orphans and for images without a mask.
- `is_defective(mask: np.ndarray, threshold: int) -> bool`: pure function.
- `process_pair(pair, split, cfg) -> PairResult`: read the mask to classify, then open the image, check mode, grayscale, letterbox, optional invert, save into `good`/`defect`. Returns the label and flags for the summary.
- `run(cfg, dry_run) -> KolektorSummary`: iterates over `cfg.splits` with `tqdm`. A per-file `OSError` is counted as `failed`, logged and does not stop the run.
- No literal `"train"`, `128`, `"_GT"` and so on in the code.

### Stage 3: CLI `experiments/preprocess_kolektorsdd2.py`

- Run as `.venv\Scripts\python.exe -m experiments.preprocess_kolektorsdd2`.
- Args: `--config` (default via `PROJECT_ROOT`), `--overwrite`, `--inverted`, `--dry-run`.
- `main(argv: list[str] | None = None) -> int`, catches `OSError, ValueError, KeyError` and raises `SystemExit(f"error: {exc}")`. Exit code 1 if `failed > 0`.
- At the end, logs the summary.

### Stage 4: Tests `tests/test_kolektorsdd2.py` (synthetic data in `tmp_path`, no real dataset needed)

1. `is_defective`: zeros -> False; one pixel at 255 -> True; threshold respected.
2. Pairing ignores `10301 (copy).png` and `10301_GT (copy).png`, and warns on an image without a mask.
3. RGB input -> output mode `L`, shape `[128, 128]`; `L` input passes unchanged (no conversion counted).
4. Non-square (e.g. 230x636): content centered, padding equal to `pad_value`, aspect ratio preserved.
5. No mask file is written anywhere in the output tree.
6. Good/defect folders get the right files.
7. `--inverted`: images equal `255 - normal` in the content area, writes to the inverted root.
8. `--dry-run` writes nothing; a second run without `--overwrite` skips existing files.
9. A bad config raises `ValueError`/`KeyError` with a clear message.

### Stage 5: Run and verify (post-implementation)

1. `.venv\Scripts\python.exe -m pytest tests/test_kolektorsdd2.py tests/test_preprocessing.py -q`
2. `... -m experiments.preprocess_kolektorsdd2 --dry-run`, then the real run, then `--inverted`.
3. Expected counts (measured above): train good 2085 / defect 246, test good 894 / defect 110, and 2 orphans skipped. Check that the summary matches, otherwise investigate before claiming success.
4. Check that all outputs are `128x128`, mode `L`, and that no `_GT` file exists in the output.
5. `understanding/10_verify_kolektorsdd2/verify.py` (matplotlib, like `08_verify_mvtec_preprocessing`): random good/defect samples, and the histogram of the original vs processed, saved as PNG.

### Stage 6: Documentation

- Update `README.md`, `RESUME.md` and `experiments/README.md` with the new command and layout. Update `requirements.txt` only if a dependency is added (none expected: PIL, NumPy and tqdm already used).
- Check that `data/processed/` is ignored in `.gitignore` (processed images must not be versioned).
- Tick the checklist below as each stage is completed.

### Out of scope

Normalization, CSV export for KolektorSDD2 (a follow-up like `export_mvtec_ad_csv`), changing the official train/test split, and cropping the dataset to avoid the large letterbox padding.

### Resolved questions (user)

- Masks are not saved; they only identify good vs defect.
- Letterbox padding with 0 and LANCZOS, as in task_01, is accepted.

## Implementation checklist

- [x] Stage 1: configs (`base.toml`, `kolektorsdd2.toml`)
- [x] Stage 2: `src/preprocessing/kolektorsdd2.py`
- [x] Stage 3: `experiments/preprocess_kolektorsdd2.py`
- [x] Stage 4: `tests/test_kolektorsdd2.py`
- [x] Stage 5: verification script created (`understanding/10_verify_kolektorsdd2/verify.py`), execution ready for user
- [x] Stage 6: documentation and `.gitignore`
