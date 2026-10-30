"""Orchestration for KolektorSDD2 dataset preprocessing.

Implements binary classification (good vs defect) via full-resolution ground-truth mask inspection,
grayscale conversion, centered square letterboxing (128x128), and intensity inversion.
Excludes masks from the processed output tree.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import re
import time
import tomllib
from typing import Iterator

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.preprocessing.letterbox import (
    invert_intensity,
    letterbox_to_square,
    to_grayscale,
)

logger = logging.getLogger("research")


@dataclass(frozen=True, slots=True)
class KolektorConfig:
    """Immutable configuration parameters for KolektorSDD2 preprocessing."""

    input_root: Path
    output_root: Path
    output_root_inverted: Path
    splits: tuple[str, ...]
    target_size: int
    pad_value: int
    padding_anchor: str
    overwrite: bool
    invert: bool
    image_extension: str
    image_resampling: Image.Resampling
    convert_to_grayscale: bool
    mask_suffix: str
    defect_threshold: int
    good_folder: str
    defect_folder: str
    stem_pattern: str

    @property
    def effective_output_root(self) -> Path:
        """Return the target output directory depending on inversion setting."""
        return self.output_root_inverted if self.invert else self.output_root


@dataclass(frozen=True, slots=True)
class SamplePair:
    """Paired raw image and ground-truth mask paths for a single sample."""

    sample_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True, slots=True)
class PairResult:
    """Outcome of processing an individual sample pair."""

    sample_id: str
    label: str
    is_defective: bool
    converted_to_gray: bool
    skipped: bool


@dataclass(frozen=True, slots=True)
class KolektorSummary:
    """Execution metrics collected during KolektorSDD2 preprocessing."""

    pairs: int
    good: int
    defect: int
    converted_to_gray: int
    skipped_existing: int
    skipped_invalid: int
    failed: int
    elapsed_s: float


def _resolve_resampling(name: str) -> Image.Resampling:
    """Resolve a string representation to a PIL.Image.Resampling enum value."""
    try:
        return Image.Resampling[name]
    except KeyError as exc:
        valid = ", ".join(m.name for m in Image.Resampling)
        raise ValueError(
            f"Invalid resampling algorithm '{name}'. Supported values: {valid}"
        ) from exc


def load_config(config_path: Path, project_root: Path) -> KolektorConfig:
    """Load and validate KolektorSDD2 preprocessing parameters from TOML files.

    Args:
        config_path: Path to the task TOML file (e.g. configs/preprocessing/kolektorsdd2.toml).
        project_root: Root directory of the repository for resolving relative paths.

    Returns:
        Validated KolektorConfig dataclass.

    Raises:
        ValueError: If configuration values violate operational bounds.
        KeyError: If required keys or sections are missing.
        FileNotFoundError: If configuration files or input folders do not exist.
    """
    resolved_config_path = (project_root / config_path).resolve()
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved_config_path}")

    with resolved_config_path.open("rb") as f:
        task_data = tomllib.load(f)

    if "preprocessing" not in task_data:
        raise KeyError("Missing required section '[preprocessing]' in task config")

    prep_table = task_data["preprocessing"]
    base_config_rel = prep_table.get("base_config")
    dataset_key = prep_table.get("dataset_key")

    if not base_config_rel or not dataset_key:
        raise KeyError("'base_config' and 'dataset_key' are required in [preprocessing]")

    base_config_path = (project_root / base_config_rel).resolve()
    if not base_config_path.is_file():
        alt_base = (resolved_config_path.parent / base_config_rel).resolve()
        if alt_base.is_file():
            base_config_path = alt_base
        else:
            raise FileNotFoundError(f"Base configuration file not found: {base_config_path}")

    with base_config_path.open("rb") as f:
        base_data = tomllib.load(f)

    datasets_table = base_data.get("datasets", {})
    if dataset_key not in datasets_table:
        raise KeyError(f"Dataset key '{dataset_key}' not found in [datasets] table")

    ds_spec = datasets_table[dataset_key]
    input_path_rel = ds_spec.get("input_path")
    output_path_rel = ds_spec.get("output_path")
    output_path_inverted_rel = ds_spec.get("output_path_inverted")
    splits_raw = ds_spec.get("splits")

    if (
        not input_path_rel
        or not output_path_rel
        or not output_path_inverted_rel
        or splits_raw is None
    ):
        raise KeyError(
            "Dataset specification requires 'input_path', 'output_path', 'output_path_inverted', and 'splits'"
        )

    input_root = (project_root / input_path_rel).resolve()
    output_root = (project_root / output_path_rel).resolve()
    output_root_inverted = (project_root / output_path_inverted_rel).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {input_root}")

    if output_root == input_root or input_root in output_root.parents:
        raise ValueError(
            f"output_root ({output_root}) cannot match or reside inside input_root ({input_root})"
        )
    if (
        output_root_inverted == input_root
        or input_root in output_root_inverted.parents
    ):
        raise ValueError(
            f"output_root_inverted ({output_root_inverted}) cannot match or reside inside input_root ({input_root})"
        )
    if output_root_inverted == output_root:
        raise ValueError(
            f"output_root_inverted ({output_root_inverted}) must differ from output_root ({output_root})"
        )

    splits = tuple(str(s) for s in splits_raw)
    if not splits:
        raise ValueError("Configured splits must not be empty")

    for s in splits:
        split_path = input_root / s
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Configured split '{s}' not found at {split_path}"
            )

    target_size = int(prep_table.get("target_size", 128))
    pad_value = int(prep_table.get("pad_value", 0))
    padding_anchor = str(prep_table.get("padding_anchor", "center"))
    overwrite = bool(prep_table.get("overwrite", False))

    if target_size < 1:
        raise ValueError(f"target_size must be >= 1, got {target_size}")
    if not (0 <= pad_value <= 255):
        raise ValueError(f"pad_value must be in range [0, 255], got {pad_value}")
    if padding_anchor != "center":
        raise ValueError(
            f"Unsupported padding_anchor '{padding_anchor}'; only 'center' is currently supported"
        )

    images_table = prep_table.get("images", {})
    image_extension = str(images_table.get("extension", ".png"))
    if not image_extension.startswith("."):
        image_extension = f".{image_extension}"
    image_resampling_str = str(images_table.get("resampling", "LANCZOS"))
    image_resampling = _resolve_resampling(image_resampling_str)
    convert_to_grayscale = bool(images_table.get("convert_to_grayscale", True))
    invert_val = images_table.get("invert", False)
    if not isinstance(invert_val, bool):
        raise ValueError(
            f"Expected boolean for 'invert' in [preprocessing.images], got {type(invert_val).__name__}"
        )

    masks_table = prep_table.get("masks", {})
    mask_suffix = str(masks_table.get("suffix", "_GT"))
    defect_threshold = int(masks_table.get("defect_threshold", 0))
    if defect_threshold < 0:
        raise ValueError(f"defect_threshold must be >= 0, got {defect_threshold}")
    good_folder = str(masks_table.get("good_folder", "good"))
    defect_folder = str(masks_table.get("defect_folder", "defect"))
    if good_folder == defect_folder:
        raise ValueError(
            f"good_folder ('{good_folder}') and defect_folder ('{defect_folder}') must be distinct"
        )

    pairing_table = prep_table.get("pairing", {})
    stem_pattern = str(pairing_table.get("stem_pattern", r"^\d+$"))
    try:
        re.compile(stem_pattern)
    except re.error as exc:
        raise ValueError(f"Invalid stem_pattern regex '{stem_pattern}': {exc}") from exc

    return KolektorConfig(
        input_root=input_root,
        output_root=output_root,
        output_root_inverted=output_root_inverted,
        splits=splits,
        target_size=target_size,
        pad_value=pad_value,
        padding_anchor=padding_anchor,
        overwrite=overwrite,
        invert=invert_val,
        image_extension=image_extension,
        image_resampling=image_resampling,
        convert_to_grayscale=convert_to_grayscale,
        mask_suffix=mask_suffix,
        defect_threshold=defect_threshold,
        good_folder=good_folder,
        defect_folder=defect_folder,
        stem_pattern=stem_pattern,
    )


def is_defective(mask: np.ndarray, threshold: int = 0) -> bool:
    """Determine whether a mask array contains a defect.

    A defect is present if any pixel intensity exceeds threshold (mask.max() > threshold).

    Args:
        mask: 2D or 3D NumPy array of pixel values.
        threshold: Strict threshold above which a pixel denotes a defect.

    Returns:
        True if any pixel value exceeds threshold, False otherwise.

    Raises:
        ValueError: If mask array is empty or has unsupported dimensions.
    """
    if mask.size == 0:
        raise ValueError("Mask array must not be empty")
    if mask.ndim not in (2, 3):
        raise ValueError(f"Mask array must be 2D or 3D, got shape {mask.shape}")
    return bool(np.any(mask > threshold))


def discover_pairs(
    split_dir: Path, cfg: KolektorConfig
) -> tuple[list[SamplePair], int]:
    """Scan split directory and match image files with corresponding mask files.

    Identifies valid sample ids matching cfg.stem_pattern. Filters out orphan images,
    orphan masks, and invalid file stems, logging warnings for each anomaly.

    Args:
        split_dir: Directory containing image and mask pairs (e.g. data/raw/KolektorSDD2/train).
        cfg: Validated KolektorConfig.

    Returns:
        Tuple containing:
            - Sorted list of valid SamplePair objects.
            - Count of skipped invalid or orphan files.
    """
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    pattern = re.compile(cfg.stem_pattern)
    target_ext = cfg.image_extension.lower()
    mask_suffix = cfg.mask_suffix

    valid_images: dict[str, Path] = {}
    valid_masks: dict[str, Path] = {}
    invalid_files: list[Path] = []

    for file_path in sorted(split_dir.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() != target_ext:
            continue

        stem = file_path.stem
        if stem.endswith(mask_suffix):
            base_id = stem[: -len(mask_suffix)]
            if pattern.fullmatch(base_id):
                valid_masks[base_id] = file_path
            else:
                logger.warning(
                    "Skipping file with invalid mask stem pattern in '%s': %s",
                    split_dir.name,
                    file_path.name,
                )
                invalid_files.append(file_path)
        else:
            base_id = stem
            if pattern.fullmatch(base_id):
                valid_images[base_id] = file_path
            else:
                logger.warning(
                    "Skipping file with invalid image stem pattern in '%s': %s",
                    split_dir.name,
                    file_path.name,
                )
                invalid_files.append(file_path)

    pairs: list[SamplePair] = []
    all_sample_ids = sorted(set(valid_images.keys()) | set(valid_masks.keys()))

    for sample_id in all_sample_ids:
        has_img = sample_id in valid_images
        has_mask = sample_id in valid_masks

        if has_img and has_mask:
            pairs.append(
                SamplePair(
                    sample_id=sample_id,
                    image_path=valid_images[sample_id],
                    mask_path=valid_masks[sample_id],
                )
            )
        elif has_img:
            logger.warning(
                "Image '%s' in '%s' has no matching mask '%s%s%s'",
                valid_images[sample_id].name,
                split_dir.name,
                sample_id,
                mask_suffix,
                target_ext,
            )
            invalid_files.append(valid_images[sample_id])
        else:
            logger.warning(
                "Mask '%s' in '%s' has no matching image '%s%s'",
                valid_masks[sample_id].name,
                split_dir.name,
                sample_id,
                target_ext,
            )
            invalid_files.append(valid_masks[sample_id])

    return pairs, len(invalid_files)


def iter_pairs(split_dir: Path, cfg: KolektorConfig) -> Iterator[SamplePair]:
    """Iterate over valid SamplePair instances discovered in split_dir."""
    pairs, _ = discover_pairs(split_dir, cfg)
    yield from pairs


def process_pair(
    pair: SamplePair,
    split: str,
    cfg: KolektorConfig,
    dry_run: bool = False,
) -> PairResult:
    """Process a single image-mask sample pair.

    Evaluates the mask at full resolution to determine defect presence,
    converts the image to grayscale, applies optional intensity inversion,
    letterboxes to a square canvas, and writes to <split>/good or <split>/defect.
    Masks are never saved to the output tree.

    Args:
        pair: SamplePair holding image and mask paths.
        split: Dataset split name (e.g. 'train' or 'test').
        cfg: Validated KolektorConfig.
        dry_run: If True, performs evaluation without disk writes.

    Returns:
        PairResult detailing classification, gray conversion, and skip state.
    """
    with Image.open(pair.mask_path) as mask_img:
        mask_arr = np.asarray(mask_img)
        defective = is_defective(mask_arr, threshold=cfg.defect_threshold)

    label = cfg.defect_folder if defective else cfg.good_folder
    dest_dir = cfg.effective_output_root / split / label
    dest_path = dest_dir / f"{pair.sample_id}{cfg.image_extension}"

    if not cfg.overwrite and dest_path.exists():
        return PairResult(
            sample_id=pair.sample_id,
            label=label,
            is_defective=defective,
            converted_to_gray=False,
            skipped=True,
        )

    with Image.open(pair.image_path) as img:
        was_converted = False
        if cfg.convert_to_grayscale and img.mode != "L":
            was_converted = True
            arr = to_grayscale(img)
        else:
            arr = np.asarray(
                img.convert("L") if img.mode != "L" else img, dtype=np.uint8
            )

        if cfg.invert:
            # Invert intensity before letterbox so padding stays 0 (black background)
            arr = invert_intensity(arr)

        letterboxed = letterbox_to_square(
            image=arr,
            size=cfg.target_size,
            pad_value=cfg.pad_value,
            interpolation=cfg.image_resampling,
            anchor=cfg.padding_anchor,
        )

    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_pil = Image.fromarray(letterboxed, mode="L")
        out_pil.save(dest_path, format="PNG")

    return PairResult(
        sample_id=pair.sample_id,
        label=label,
        is_defective=defective,
        converted_to_gray=was_converted,
        skipped=False,
    )


def run(cfg: KolektorConfig, dry_run: bool = False) -> KolektorSummary:
    """Execute the complete KolektorSDD2 dataset preprocessing pipeline.

    Args:
        cfg: Validated KolektorConfig.
        dry_run: When True, enumerates and classifies without disk writes.

    Returns:
        KolektorSummary containing execution counts and timings.
    """
    start_time = time.perf_counter()
    dest_root = cfg.effective_output_root

    logger.info(
        "Starting KolektorSDD2 preprocessing pipeline (dry_run=%s, invert=%s, destination=%s)",
        dry_run,
        cfg.invert,
        dest_root,
    )

    total_pairs = 0
    total_good = 0
    total_defect = 0
    total_converted = 0
    total_skipped_existing = 0
    total_skipped_invalid = 0
    total_failed = 0

    split_pairs_map: dict[str, list[SamplePair]] = {}

    for split in cfg.splits:
        split_dir = cfg.input_root / split
        pairs, invalid_count = discover_pairs(split_dir, cfg)
        split_pairs_map[split] = pairs
        total_pairs += len(pairs)
        total_skipped_invalid += invalid_count
        logger.info(
            "Split '%s': discovered %d valid pairs, %d invalid/orphan files",
            split,
            len(pairs),
            invalid_count,
        )

    with tqdm(
        total=total_pairs, desc="Processing KolektorSDD2 pairs", unit="pair"
    ) as pbar:
        for split, pairs in split_pairs_map.items():
            for pair in pairs:
                try:
                    result = process_pair(pair, split, cfg, dry_run=dry_run)
                    if result.is_defective:
                        total_defect += 1
                    else:
                        total_good += 1

                    if result.skipped:
                        total_skipped_existing += 1
                    else:
                        if result.converted_to_gray:
                            total_converted += 1
                except Exception as exc:
                    total_failed += 1
                    logger.error(
                        "Failed to process pair '%s' in split '%s': %s",
                        pair.sample_id,
                        split,
                        exc,
                    )
                finally:
                    pbar.update(1)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "KolektorSDD2 preprocessing complete: pairs=%d (good=%d, defect=%d), converted_to_gray=%d, skipped_existing=%d, skipped_invalid=%d, failed=%d in %.2fs",
        total_pairs,
        total_good,
        total_defect,
        total_converted,
        total_skipped_existing,
        total_skipped_invalid,
        total_failed,
        elapsed,
    )

    return KolektorSummary(
        pairs=total_pairs,
        good=total_good,
        defect=total_defect,
        converted_to_gray=total_converted,
        skipped_existing=total_skipped_existing,
        skipped_invalid=total_skipped_invalid,
        failed=total_failed,
        elapsed_s=elapsed,
    )
