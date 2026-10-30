"""Orchestration for MVTec-AD dataset preprocessing: configuration loading, traversal, and batch execution."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import time
import tomllib
from typing import Iterator, Sequence

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
class PreprocessConfig:
    """Immutable configuration parameters for dataset preprocessing."""

    input_root: Path
    output_root: Path
    output_root_inverted: Path
    subdatasets: tuple[str, ...]
    target_size: int
    pad_value: int
    image_extensions: tuple[str, ...]
    image_resampling: Image.Resampling
    mask_resampling: Image.Resampling
    mask_folder: str
    padding_anchor: str
    overwrite: bool
    invert: bool = False

    @property
    def effective_output_root(self) -> Path:
        """Return the target output directory depending on inversion setting."""
        return self.output_root_inverted if self.invert else self.output_root



@dataclass(frozen=True, slots=True)
class PreprocessSummary:
    """Execution metrics collected during preprocessing."""

    total_discovered: int
    processed: int
    skipped: int
    failed: int
    masks: int
    padded: int
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


def load_config(config_path: Path, project_root: Path) -> PreprocessConfig:
    """Load and validate preprocessing parameters from TOML files.

    Args:
        config_path: Path to the task TOML file (e.g. configs/preprocessing/mvtec_ad.toml).
        project_root: Root directory of the repository for resolving relative paths.

    Returns:
        Validated PreprocessConfig dataclass.

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
            project_root = resolved_config_path.parent
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
    subdatasets_raw = ds_spec.get("subdatasets")

    if (
        not input_path_rel
        or not output_path_rel
        or not output_path_inverted_rel
        or subdatasets_raw is None
    ):
        raise KeyError(
            "Dataset specification requires 'input_path', 'output_path', 'output_path_inverted', and 'subdatasets'"
        )

    input_root = (project_root / input_path_rel).resolve()
    output_root = (project_root / output_path_rel).resolve()
    output_root_inverted = (project_root / output_path_inverted_rel).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {input_root}")

    # Guard against accidental directory clobbering or writing into raw data
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

    subdatasets = tuple(str(s) for s in subdatasets_raw)
    for sub in subdatasets:
        sub_path = input_root / sub
        if not sub_path.is_dir():
            raise FileNotFoundError(
                f"Configured subdataset '{sub}' not found at {sub_path}"
            )

    target_size = int(prep_table.get("target_size", 128))
    pad_value = int(prep_table.get("pad_value", 0))
    padding_anchor = str(prep_table.get("padding_anchor", "center"))
    overwrite = bool(prep_table.get("overwrite", False))
    extensions_raw = prep_table.get("image_extensions", [".png"])
    image_extensions = tuple(str(ext).lower() for ext in extensions_raw)

    if target_size < 1:
        raise ValueError(f"target_size must be >= 1, got {target_size}")
    if not (0 <= pad_value <= 255):
        raise ValueError(f"pad_value must be in range [0, 255], got {pad_value}")
    if padding_anchor != "center":
        raise ValueError(
            f"Unsupported padding_anchor '{padding_anchor}'; only 'center' is currently supported"
        )

    images_table = prep_table.get("images", {})
    masks_table = prep_table.get("masks", {})

    image_resampling_str = images_table.get("resampling", "LANCZOS")
    mask_resampling_str = masks_table.get("resampling", "NEAREST")
    mask_folder = str(masks_table.get("folder", "ground_truth"))
    invert_val = images_table.get("invert", False)
    if not isinstance(invert_val, bool):
        raise ValueError(f"Expected boolean for 'invert' in [preprocessing.images], got {type(invert_val).__name__}")

    image_resampling = _resolve_resampling(image_resampling_str)
    mask_resampling = _resolve_resampling(mask_resampling_str)

    return PreprocessConfig(
        input_root=input_root,
        output_root=output_root,
        output_root_inverted=output_root_inverted,
        subdatasets=subdatasets,
        target_size=target_size,
        pad_value=pad_value,
        image_extensions=image_extensions,
        image_resampling=image_resampling,
        mask_resampling=mask_resampling,
        mask_folder=mask_folder,
        padding_anchor=padding_anchor,
        overwrite=overwrite,
        invert=invert_val,
    )


def iter_images(
    root: Path, subdatasets: Sequence[str], extensions: Sequence[str]
) -> Iterator[Path]:
    """Iterate deterministically over image files within explicit subdatasets.

    Filters strictly by category directory to skip non-dataset artifacts (.cache, results, text files).
    """
    valid_exts = set(extensions)
    for category in sorted(subdatasets):
        category_dir = root / category
        if not category_dir.is_dir():
            continue
        for path in sorted(category_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in valid_exts:
                yield path


def is_mask(path: Path, input_root: Path, mask_folder: str) -> bool:
    """Determine if an image path represents a ground-truth defect mask.

    Checks if `mask_folder` matches the category-relative subfolder (e.g. <category>/ground_truth/...).
    """
    rel = path.relative_to(input_root)
    return len(rel.parts) >= 2 and rel.parts[1] == mask_folder


def preprocess_image(src: Path, dst: Path, cfg: PreprocessConfig) -> bool:
    """Process a single image or mask file and save to target path.

    Returns:
        True if the image required padding (aspect ratio != 1.0), False otherwise.
    """
    is_ground_truth = is_mask(src, cfg.input_root, cfg.mask_folder)

    with Image.open(src) as img:
        width, height = img.size
        needs_padding = width != height

        if is_ground_truth:
            # Masks are single channel (mode 'L'). Do not convert to avoid re-weighting values.
            # Never invert masks (defect and background labels must be preserved).
            arr = np.asarray(img.convert("L") if img.mode != "L" else img, dtype=np.uint8)
            resampling = cfg.mask_resampling
        else:
            arr = to_grayscale(img)
            if cfg.invert:
                # Invert intensity before letterbox so padding stays 0 (black background)
                arr = invert_intensity(arr)
            resampling = cfg.image_resampling

    letterboxed = letterbox_to_square(
        image=arr,
        size=cfg.target_size,
        pad_value=cfg.pad_value,
        interpolation=resampling,
        anchor=cfg.padding_anchor,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    out_pil = Image.fromarray(letterboxed, mode="L")
    out_pil.save(dst, format="PNG")
    return needs_padding


def run(cfg: PreprocessConfig, dry_run: bool = False) -> PreprocessSummary:
    """Execute the complete dataset preprocessing pipeline.

    Args:
        cfg: Validated PreprocessConfig specification.
        dry_run: When True, enumerates images without writing outputs.

    Returns:
        PreprocessSummary with execution counts and timings.
    """
    start_time = time.perf_counter()
    image_paths = list(iter_images(cfg.input_root, cfg.subdatasets, cfg.image_extensions))
    total_discovered = len(image_paths)
    dest_root = cfg.effective_output_root

    logger.info(
        "Discovered %d images across %d subdatasets (dry_run=%s, invert=%s, destination=%s)",
        total_discovered,
        len(cfg.subdatasets),
        dry_run,
        cfg.invert,
        dest_root,
    )

    processed = 0
    skipped = 0
    failed = 0
    masks_count = 0
    padded_count = 0

    if dry_run:
        for src in image_paths:
            if is_mask(src, cfg.input_root, cfg.mask_folder):
                masks_count += 1
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Dry run complete: %d total, %d masks, 0 written (destination=%s)",
            total_discovered,
            masks_count,
            dest_root,
        )
        return PreprocessSummary(
            total_discovered=total_discovered,
            processed=0,
            skipped=0,
            failed=0,
            masks=masks_count,
            padded=0,
            elapsed_s=elapsed,
        )

    for src in tqdm(image_paths, desc="Preprocessing images", unit="img"):
        rel_path = src.relative_to(cfg.input_root)
        dst = dest_root / rel_path

        if not cfg.overwrite and dst.exists():
            skipped += 1
            continue

        try:
            padded = preprocess_image(src, dst, cfg)
            processed += 1
            if padded:
                padded_count += 1
            if is_mask(src, cfg.input_root, cfg.mask_folder):
                masks_count += 1
        except Exception as exc:
            failed += 1
            logger.error("Failed to process '%s': %s", src, exc)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Preprocessing summary: processed=%d, skipped=%d, failed=%d, masks=%d, padded=%d in %.2fs",
        processed,
        skipped,
        failed,
        masks_count,
        padded_count,
        elapsed,
    )

    return PreprocessSummary(
        total_discovered=total_discovered,
        processed=processed,
        skipped=skipped,
        failed=failed,
        masks=masks_count,
        padded=padded_count,
        elapsed_s=elapsed,
    )
