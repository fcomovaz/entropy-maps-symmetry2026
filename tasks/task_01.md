# MVTEC-AD Daaset cleaning and converting

## Issue

* The main issue with this file is that all images are in RGB format and they have diferent sizes. This is an issue to consume in an easy way the dataset. so We need to convert to grayscale.
* Second issue is some images can be have diferent sizes, and some have a big resolution or a low resolution. Through a previous analysis we noticed minimum size as 128x128 to apply to all images. In this case can be possible or there is a possibility that the image is not a perfect square, it can be a rectangle, in this case we won't resize to square 128px due to image ratio change, so we need to do letterboxing/padding and make the larger side 128px and fill the rest with 0's (black), in this way we preserve the image ratio.

### Folder structure

The folder structure where the images are located is the following:

`D:\Publications\2026-10-Symmetry-Entropy\code\data\raw\mvtec-ad`

> Remember not hardcode anything (since paths to numbers in the files), the only place to hardcode is in the toml files, create a nuew one, or use an existing one if it can keep the AGENTS.md rules, so analyse it and plan it.

## Proceeding

- [x] Read AGENTS.md file to know the general rules of the project.
- [x] Analyze the request and create a plan
- [x] Update the configuration file needed (`configs/base.toml` and `configs/preprocessing/mvtec_ad.toml`)
- [x] Update the `## Plan (AI reviewed and updated)` section to implement post user review
- [x] Implement pure transformations in `src/preprocessing/letterbox.py`
- [x] Implement dataset orchestration in `src/preprocessing/mvtec_ad.py`
- [x] Implement CLI command in `experiments/preprocess_mvtec_ad.py`
- [x] Implement automated test suite in `tests/test_preprocessing.py`
- [x] Implement post-run verification and visual validation in `understanding/08_verify_mvtec_preprocessing/verify.py`
- [x] Update documentation in `README.md`, `RESUME.md`, and dependencies in `requirements.txt`
- [x] Implement Plan Addendum (`--inverted` option and separate destination dataset)

## Plan (user provisional)

- [x] Convert all images to grayscale (ITU-R 601 luma to mode 'L')
- [x] Resize all images and do letterboxing/padding at 128px (longer side scaled to 128px, centered, padded with 0)
- [x] Create a new folder with the new structure under the `data\preprocessed\mvtec-ad` folder
- [x] The preprocessed folder preserves the same subfolder structure as raw, with smaller footprint and 128x128 single-channel PNGs

### Findings (measured, sample of ~4% of PNGs)

- `data/raw/mvtec-ad` has 15 category folders (listed in `datasets.mvtecad.subdatasets`) plus **non-dataset entries to ignore**: `.cache`, `results` (generated heatmaps/ckpt), `license.txt`, `readme.txt`, `README.md`, `.gitattributes`. Only iterate over `subdatasets`.
- Inside each category: `train/good`, `test/<defect>|good`, `ground_truth/<defect>` (binary masks, mode `L`). All category files are `.png`.
- Observed sizes per category are square (700, 800, 840, 900, 1000, 1024); modes are mostly `RGB`, but `grid`, `screw`, `zipper` (and all masks) are already `L`. Code must still handle non-square generically.
- Total PNG count across the raw folder is 14491 including `results/` and `.cache/` artefacts, so the real dataset count must be computed by scanning only `subdatasets`.

### Config (done)

Split by purpose to keep `base.toml` small (dataset paths only; reused by every experiment):
- `configs/base.toml`: `[datasets.mvtecad]` gains `output_path = "data/preprocessed/mvtec-ad"` and `output_path_inverted = "data/preprocessed/mvtec-ad-inverted"`.
- `configs/preprocessing/mvtec_ad.toml`: `[preprocessing]` (`base_config`, `dataset_key`, `target_size=128`, `pad_value=0`, `padding_anchor`, `image_extensions`, `overwrite`), `[preprocessing.images]` (`resampling="LANCZOS"`, `convert_to_grayscale=true`, `invert=false`), `[preprocessing.masks]` (`folder="ground_truth"`, `resampling="NEAREST"`, `convert_to_grayscale=false`).
- Convention for future work: one TOML per task type under `configs/<area>/` (e.g. `configs/experiments/`), each pointing to `base.toml` for shared dataset paths. Do not duplicate paths.

### Ground-truth masks (answer to review)

Masks are already single channel (`L`), so grayscale conversion is skipped for them. They **cannot be copied as is**: they are full resolution (700 to 1024 px) and must stay pixel-aligned with their 128x128 letterboxed image. So masks get the same scale + padding, with NEAREST resampling so no new gray levels appear. Output mask values must stay a subset of the input values (checked in tests and in the post-run verification). Padding stays 0 (background), which is correct for masks.

## Plan (agent reviewed and updated)

Follow AGENTS.md: no hardcoded values in `src/`, `pathlib`, `from __future__ import annotations`, type hints, `@dataclass(frozen=True, slots=True)`, `argparse` + `main(argv) -> int`, `logging` not `print`, errors caught in `main()` as `SystemExit(f"error: {exc}")`.

### Files created / updated

1. [x] `src/preprocessing/__init__.py`: Package entry exposing public transforms (`to_grayscale`, `letterbox_to_square`, `invert_intensity`).
2. [x] `src/preprocessing/letterbox.py`: Pure logic (`to_grayscale`, `letterbox_to_square`, `invert_intensity`).
3. [x] `src/preprocessing/mvtec_ad.py`: Pipeline orchestration (`load_config`, `iter_images`, `is_mask`, `preprocess_image`, `run`). Supports standard and inverted runs.
4. [x] `experiments/preprocess_mvtec_ad.py`: Modular CLI entry point with `--dry-run`, `--overwrite`, `--inverted`.
5. [x] `tests/test_preprocessing.py`: Comprehensive test suite runnable via unittest or pytest (includes inversion tests).
6. [x] `understanding/08_verify_mvtec_preprocessing/verify.py`: Post-run inspection and visual validation figure (supports standard and `--inverted`).

## Plan addendum: `--inverted` option (implemented, pending user manual execution)

Goal: produce an intensity-inverted grayscale variant (`255 - x`) so later experiments can measure how background polarity affects the network/selectors. Base case (no flag) stays byte-identical to the current output.

### Decisions implemented

- **Order of operations:** grayscale, then invert, then letterbox. Padding stays `pad_value = 0` after inversion. Inverting after letterbox would turn the padding into 255 (white) and create a fake border; inverting before keeps padding black, which matches the inverted background (white 255 becomes 0).
- **Scope:** only images (`train`, `test`). Masks in `ground_truth` are never inverted (labels are not intensity; inverting would swap defect and background).
- **Separate output folder:** `base.toml` defines `output_path_inverted = "data/preprocessed/mvtec-ad-inverted"`. With `--inverted`, this path is used as `output_root`. Masks are processed without inversion into this folder as well.
- **Config:** in `configs/preprocessing/mvtec_ad.toml`, `[preprocessing.images]` has `invert = false` (default). CLI `--inverted` overrides it to `true`.
- **Inversion pure function:** `invert_intensity(image: np.ndarray) -> np.ndarray` in `src/preprocessing/letterbox.py`: validates `[H, W]` uint8, computes `np.subtract(255, image, dtype=np.uint8)`.

### Changes completed

1. [x] `configs/base.toml`: added `output_path_inverted`.
2. [x] `configs/preprocessing/mvtec_ad.toml`: added `invert = false` under `[preprocessing.images]`.
3. [x] `src/preprocessing/letterbox.py`: added `invert_intensity`.
4. [x] `src/preprocessing/mvtec_ad.py`: `PreprocessConfig` with `invert` and `output_root_inverted`, effective output root routing, non-mask inversion before letterboxing.
5. [x] `experiments/preprocess_mvtec_ad.py`: added `--inverted` CLI argument and logging.
6. [x] `tests/test_preprocessing.py`: tests for `invert_intensity`, inverted letterbox padding, mask non-inversion, directory separation, and CLI invocation.
7. [x] `understanding/08_verify_mvtec_preprocessing/verify.py`: added `--inverted` verification support.

### Manual execution commands for user

Execute manually from workspace root (`d:\Publications\2026-10-Symmetry-Entropy\code`):

1. **Unit and integration tests (all test cases including inversion)**:
   ```powershell
   .\.venv\Scripts\python.exe -m unittest tests/test_preprocessing.py
   # or with pytest:
   .\.venv\Scripts\python.exe -m pytest tests/test_preprocessing.py -q
   ```

2. **Dry run (dataset scan and counts without writing)**:
   ```powershell
   # Standard:
   .\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --dry-run
   # Inverted:
   .\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --inverted --dry-run
   ```

3. **Complete dataset preprocessing**:
   ```powershell
   # Standard (writes to data/preprocessed/mvtec-ad):
   .\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad

   # Inverted (writes to data/preprocessed/mvtec-ad-inverted):
   .\.venv\Scripts\python.exe -m experiments.preprocess_mvtec_ad --inverted
   ```

4. **Integrity verification and visual comparison**:
   ```powershell
   # Standard verification:
   .\.venv\Scripts\python.exe understanding/08_verify_mvtec_preprocessing/verify.py

   # Inverted verification:
   .\.venv\Scripts\python.exe understanding/08_verify_mvtec_preprocessing/verify.py --inverted
   ```

### Decisions and caveats for the reviewer

- Grayscale conversion uses PIL luma weights. The lab-style entropy work later depends on intensity, so this is a documented choice.
- Rescale applies to all images including those smaller than 128 (user asked for a 128px longer side). Resampling is LANCZOS for images (anti-aliasing for 1024 to 128 downscale) and NEAREST for masks.
- Known MVTec-AD sizes are all square in the sample, so letterbox will rarely pad on this dataset. Logic is still implemented and tested with synthetic non-square inputs. The run logs the count of images that needed padding.
- Mask downscaling to 128 can drop very small defects (a few pixels at 1024). Flagged in README.md.
- Not every category has a white background. Texture categories fill the whole frame, and `screw`/`zipper` have gray or dark backgrounds. Inversion applies uniformly across categories as a controlled factor.
- Out of scope: normalization, CSV regeneration, train/test split changes.
