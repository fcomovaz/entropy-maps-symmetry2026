"""Strict TOML configuration loader and deterministic image loading for Task 09 lab."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tomllib
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.entropy_maps import validate_bins, validate_grid_levels


def _require(table: Mapping[str, object], key: str, where: str) -> object:
    """Extract required key from TOML mapping or raise KeyError with section context."""
    if key not in table:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return table[key]


def _require_table(table: Mapping[str, object], key: str, where: str) -> dict[str, object]:
    """Extract required sub-table from TOML mapping with validation."""
    obj = _require(table, key, where)
    if not isinstance(obj, dict):
        raise ValueError(f"Section '{key}' in {where} must be a TOML table")
    return dict(obj)


def _require_int(table: Mapping[str, object], key: str, where: str) -> int:
    """Extract strict integer (rejecting booleans) from TOML table."""
    val = _require(table, key, where)
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"Key '{key}' in {where} must be an integer, got {type(val).__name__}")
    return int(val)


def _require_float(table: Mapping[str, object], key: str, where: str) -> float:
    """Extract numeric float/int (rejecting booleans) from TOML table."""
    val = _require(table, key, where)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"Key '{key}' in {where} must be numeric, got {type(val).__name__}")
    return float(val)


def _require_bool(table: Mapping[str, object], key: str, where: str) -> bool:
    """Extract strict boolean from TOML table."""
    val = _require(table, key, where)
    if not isinstance(val, bool):
        raise ValueError(f"Key '{key}' in {where} must be a boolean, got {type(val).__name__}")
    return bool(val)


@dataclass(frozen=True, slots=True)
class DataOptimizationConfig:
    """Input dataset source and dimensions for entropy optimization experiments."""

    source: str
    n_real_samples: int
    image_size: int
    preprocessed_dir: Path
    image_extensions: tuple[str, ...]
    mask_folder: str | None
    mask_suffix: str | None


@dataclass(frozen=True, slots=True)
class NativeBuildConfig:
    """MSVC native C compilation configuration."""

    source_dir: Path
    build_dir: Path
    library_name: str
    c_sources: tuple[str, ...]
    compiler_flags: tuple[str, ...]
    vswhere_hint: str

    @property
    def dll_path(self) -> Path:
        """Resolved path to compiled shared library."""
        suffix = ".dll" if os.name == "nt" else ".so"
        return self.build_dir / f"{self.library_name}{suffix}"


@dataclass(frozen=True, slots=True)
class ProtocolOptimizationConfig:
    """CPU-fair timing, warmup, statistical confidence, and parity tolerance settings."""

    threads: int
    batch_sizes: tuple[int, ...]
    warmup_iters: int
    timed_iters: int
    batch_timed_iters: int
    repeats: int
    disable_gc: bool
    tolerance_abs: float
    bootstrap_resamples: int
    confidence_level: float
    mann_whitney_alpha: float


@dataclass(frozen=True, slots=True)
class SweepsOptimizationConfig:
    """Parameter sweep axes (grid levels remain fixed)."""

    bins_list: tuple[int, ...]
    image_sizes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PlotsOptimizationConfig:
    """Output directory and figure serialization parameters."""

    output_dir: Path
    dpi: int
    formats: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CnnReferenceConfig:
    """Linked CNN preprocessing normalization parameters for fair comparison."""

    normalize_mean: float
    normalize_std: float
    image_size: int
    reference_models: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LrReferenceConfig:
    """Linked LogisticRegression parameters from classification TOML."""

    c_value: float
    penalty: str
    solver: str
    class_weight: str | None
    max_iter: int
    scale_features: bool


@dataclass(frozen=True, slots=True)
class EntropyOptimizationConfig:
    """Validated root configuration for Lab 13 (entropy maps optimization)."""

    project_root: Path
    config_path: Path
    entropy_config_path: Path
    base_config_path: Path
    cnn_config_path: Path
    classification_config_path: Path
    seed: int
    variants: tuple[str, ...]
    reference_variant: str
    grid_levels: tuple[int, ...]
    bins: int
    dataset_key: str
    dataset_name: str
    inverted: bool
    data: DataOptimizationConfig
    native: NativeBuildConfig
    protocol: ProtocolOptimizationConfig
    sweeps: SweepsOptimizationConfig
    plots: PlotsOptimizationConfig
    cnn_ref: CnnReferenceConfig
    lr_ref: LrReferenceConfig


def resolve_project_path(raw_path: Path | str, project_root: Path) -> Path:
    """Resolve a path relative to project_root when not already absolute."""
    p = Path(raw_path)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def load_optimization_config(
    config_path: Path,
    project_root: Path = PROJECT_ROOT,
) -> EntropyOptimizationConfig:
    """Load, anchor paths, and fail-fast validate entropy optimization TOML configuration."""
    resolved_root = project_root.resolve()
    resolved_cfg = resolve_project_path(config_path, resolved_root)
    if not resolved_cfg.is_file():
        raise FileNotFoundError(f"Optimization config file not found: {resolved_cfg}")

    with resolved_cfg.open("rb") as f:
        raw_cfg = tomllib.load(f)

    opt = _require_table(raw_cfg, "entropy_optimization", str(resolved_cfg))

    entropy_cfg_path = resolve_project_path(
        str(_require(opt, "entropy_config", "[entropy_optimization]")), resolved_root
    )
    base_cfg_path = resolve_project_path(
        str(_require(opt, "base_config", "[entropy_optimization]")), resolved_root
    )
    cnn_cfg_path = resolve_project_path(
        str(_require(opt, "cnn_config", "[entropy_optimization]")), resolved_root
    )
    cls_cfg_path = resolve_project_path(
        str(_require(opt, "classification_config", "[entropy_optimization]")), resolved_root
    )

    for linked in (entropy_cfg_path, base_cfg_path, cnn_cfg_path, cls_cfg_path):
        if not linked.is_file():
            raise FileNotFoundError(f"Linked configuration file not found: {linked}")

    seed = _require_int(opt, "seed", "[entropy_optimization]")
    raw_variants = _require(opt, "variants", "[entropy_optimization]")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("'variants' in [entropy_optimization] must be a non-empty list")
    variants = tuple(str(v) for v in raw_variants)
    reference_variant = str(_require(opt, "reference_variant", "[entropy_optimization]"))
    if reference_variant not in variants:
        raise ValueError(
            f"reference_variant '{reference_variant}' must be present in variants {variants}"
        )

    raw_cnn_models = _require(opt, "cnn_reference_models", "[entropy_optimization]")
    if not isinstance(raw_cnn_models, list) or not raw_cnn_models:
        raise ValueError("'cnn_reference_models' in [entropy_optimization] must be a non-empty list")
    cnn_reference_models = tuple(str(m).lower() for m in raw_cnn_models)

    # Read linked entropy TOML for canonical grid_levels, bins, dataset_key, inverted
    with entropy_cfg_path.open("rb") as f_ent:
        ent_raw = tomllib.load(f_ent)
    ent_tbl = _require_table(ent_raw, "entropy_maps", str(entropy_cfg_path))
    dataset_key = str(_require(ent_tbl, "dataset_key", "[entropy_maps]"))
    inverted = _require_bool(ent_tbl, "inverted", "[entropy_maps]")
    preproc_cfg_path = resolve_project_path(
        str(_require(ent_tbl, "preprocessing_config", "[entropy_maps]")), resolved_root
    )
    if not preproc_cfg_path.is_file():
        raise FileNotFoundError(f"Linked preprocessing config not found: {preproc_cfg_path}")
    ent_regions = _require_table(ent_tbl, "regions", "[entropy_maps]")
    raw_levels = _require(ent_regions, "grid_levels", "[entropy_maps.regions]")
    if not isinstance(raw_levels, list) or not raw_levels:
        raise ValueError("'grid_levels' in [entropy_maps.regions] must be a non-empty list")
    bins = validate_bins(_require_int(ent_regions, "bins", "[entropy_maps.regions]"))

    # Read linked base TOML
    with base_cfg_path.open("rb") as f_base:
        base_raw = tomllib.load(f_base)
    datasets_tbl = _require_table(base_raw, "datasets", str(base_cfg_path))
    ds_tbl = _require_table(datasets_tbl, dataset_key, f"[datasets.{dataset_key}]")
    dataset_name = str(_require(ds_tbl, "name", f"[datasets.{dataset_key}]"))
    prep_rel = str(
        _require(
            ds_tbl,
            "output_path_inverted" if inverted else "output_path",
            f"[datasets.{dataset_key}]",
        )
    )
    preprocessed_dir = resolve_project_path(prep_rel, resolved_root)

    # Read linked preprocessing TOML for image extensions and mask filtering
    with preproc_cfg_path.open("rb") as f_prep:
        prep_raw = tomllib.load(f_prep)
    prep_tbl = _require_table(prep_raw, "preprocessing", str(preproc_cfg_path))
    if "image_extensions" in prep_tbl and isinstance(prep_tbl["image_extensions"], list):
        image_extensions = tuple(str(e).lower() for e in prep_tbl["image_extensions"])
    else:
        prep_images_tbl = _require_table(prep_tbl, "images", "[preprocessing]")
        ext = str(_require(prep_images_tbl, "extension", "[preprocessing.images]")).lower()
        image_extensions = (ext,)
    masks_tbl = prep_tbl.get("masks", {})
    mask_folder: str | None = None
    mask_suffix: str | None = None
    if isinstance(masks_tbl, dict):
        if "folder" in masks_tbl and str(masks_tbl["folder"]).strip():
            mask_folder = str(masks_tbl["folder"]).strip()
        if "suffix" in masks_tbl and str(masks_tbl["suffix"]).strip():
            mask_suffix = str(masks_tbl["suffix"]).strip()

    # [entropy_optimization.data]
    data_tbl = _require_table(opt, "data", "[entropy_optimization]")
    source = str(_require(data_tbl, "source", "[entropy_optimization.data]")).lower().strip()
    if source not in {"real", "synthetic"}:
        raise ValueError(f"data.source must be 'real' or 'synthetic', got '{source}'")
    n_real_samples = _require_int(data_tbl, "n_real_samples", "[entropy_optimization.data]")
    image_size = _require_int(data_tbl, "image_size", "[entropy_optimization.data]")
    if n_real_samples < 1:
        raise ValueError(f"n_real_samples must be >= 1, got {n_real_samples}")
    if image_size < 1:
        raise ValueError(f"image_size must be >= 1, got {image_size}")

    grid_levels = validate_grid_levels([int(g) for g in raw_levels], image_size)

    data_cfg = DataOptimizationConfig(
        source=source,
        n_real_samples=n_real_samples,
        image_size=image_size,
        preprocessed_dir=preprocessed_dir,
        image_extensions=image_extensions,
        mask_folder=mask_folder,
        mask_suffix=mask_suffix,
    )

    # [entropy_optimization.native]
    nat_tbl = _require_table(opt, "native", "[entropy_optimization]")
    source_dir = resolve_project_path(
        str(_require(nat_tbl, "source_dir", "[entropy_optimization.native]")), resolved_root
    )
    build_dir = resolve_project_path(
        str(_require(nat_tbl, "build_dir", "[entropy_optimization.native]")), resolved_root
    )
    library_name = str(_require(nat_tbl, "library_name", "[entropy_optimization.native]")).strip()
    if not library_name:
        raise ValueError("native.library_name cannot be empty")
    raw_c_sources = _require(nat_tbl, "c_sources", "[entropy_optimization.native]")
    if not isinstance(raw_c_sources, list) or not raw_c_sources:
        raise ValueError("'c_sources' in [entropy_optimization.native] must be a non-empty list")
    c_sources = tuple(str(s) for s in raw_c_sources)
    raw_flags = _require(nat_tbl, "compiler_flags", "[entropy_optimization.native]")
    if not isinstance(raw_flags, list):
        raise ValueError("'compiler_flags' in [entropy_optimization.native] must be a list")
    compiler_flags = tuple(str(fl) for fl in raw_flags)
    vswhere_hint = str(_require(nat_tbl, "vswhere_hint", "[entropy_optimization.native]"))

    native_cfg = NativeBuildConfig(
        source_dir=source_dir,
        build_dir=build_dir,
        library_name=library_name,
        c_sources=c_sources,
        compiler_flags=compiler_flags,
        vswhere_hint=vswhere_hint,
    )

    # [entropy_optimization.protocol]
    proto_tbl = _require_table(opt, "protocol", "[entropy_optimization]")
    threads = _require_int(proto_tbl, "threads", "[entropy_optimization.protocol]")
    if threads < 1:
        raise ValueError(f"protocol.threads must be >= 1, got {threads}")
    raw_batches = _require(proto_tbl, "batch_sizes", "[entropy_optimization.protocol]")
    if not isinstance(raw_batches, list) or not raw_batches:
        raise ValueError("'batch_sizes' in [entropy_optimization.protocol] must be a non-empty list")
    batch_sizes = tuple(int(b) for b in raw_batches)
    if any(b < 1 for b in batch_sizes):
        raise ValueError(f"All batch_sizes must be >= 1, got {batch_sizes}")
    warmup_iters = _require_int(proto_tbl, "warmup_iters", "[entropy_optimization.protocol]")
    timed_iters = _require_int(proto_tbl, "timed_iters", "[entropy_optimization.protocol]")
    batch_timed_iters = _require_int(proto_tbl, "batch_timed_iters", "[entropy_optimization.protocol]")
    repeats = _require_int(proto_tbl, "repeats", "[entropy_optimization.protocol]")
    disable_gc = _require_bool(proto_tbl, "disable_gc", "[entropy_optimization.protocol]")
    tolerance_abs = _require_float(proto_tbl, "tolerance_abs", "[entropy_optimization.protocol]")
    bootstrap_resamples = _require_int(
        proto_tbl, "bootstrap_resamples", "[entropy_optimization.protocol]"
    )
    confidence_level = _require_float(
        proto_tbl, "confidence_level", "[entropy_optimization.protocol]"
    )
    mann_whitney_alpha = _require_float(
        proto_tbl, "mann_whitney_alpha", "[entropy_optimization.protocol]"
    )

    if warmup_iters < 0 or timed_iters < 1 or batch_timed_iters < 1 or repeats < 1:
        raise ValueError("Invalid iteration/repeat counts in [entropy_optimization.protocol]")
    if tolerance_abs <= 0.0:
        raise ValueError(f"tolerance_abs must be > 0, got {tolerance_abs}")
    if bootstrap_resamples < 10 or not (0.5 < confidence_level < 1.0):
        raise ValueError("Invalid bootstrap parameters in [entropy_optimization.protocol]")
    if not (0.0 < mann_whitney_alpha < 0.5):
        raise ValueError("mann_whitney_alpha must be in (0.0, 0.5)")

    protocol_cfg = ProtocolOptimizationConfig(
        threads=threads,
        batch_sizes=batch_sizes,
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        batch_timed_iters=batch_timed_iters,
        repeats=repeats,
        disable_gc=disable_gc,
        tolerance_abs=tolerance_abs,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        mann_whitney_alpha=mann_whitney_alpha,
    )

    # [entropy_optimization.sweeps]
    sw_tbl = _require_table(opt, "sweeps", "[entropy_optimization]")
    raw_bins_list = _require(sw_tbl, "bins_list", "[entropy_optimization.sweeps]")
    raw_img_sizes = _require(sw_tbl, "image_sizes", "[entropy_optimization.sweeps]")
    if not isinstance(raw_bins_list, list) or not raw_bins_list:
        raise ValueError("'bins_list' in [entropy_optimization.sweeps] must be non-empty")
    if not isinstance(raw_img_sizes, list) or not raw_img_sizes:
        raise ValueError("'image_sizes' in [entropy_optimization.sweeps] must be non-empty")
    bins_list = tuple(validate_bins(int(b)) for b in raw_bins_list)
    image_sizes = tuple(int(s) for s in raw_img_sizes)
    finest_grid = max(grid_levels)
    for sz in image_sizes:
        if sz < finest_grid or sz % finest_grid != 0:
            raise ValueError(
                f"Sweep image_size={sz} must be divisible by finest grid {finest_grid}"
            )
    sweeps_cfg = SweepsOptimizationConfig(bins_list=bins_list, image_sizes=image_sizes)

    # [entropy_optimization.plots]
    pl_tbl = _require_table(opt, "plots", "[entropy_optimization]")
    output_dir = resolve_project_path(
        str(_require(pl_tbl, "output_dir", "[entropy_optimization.plots]")), resolved_root
    )
    dpi = _require_int(pl_tbl, "dpi", "[entropy_optimization.plots]")
    if dpi < 50:
        raise ValueError(f"plots.dpi must be >= 50, got {dpi}")
    raw_fmts = _require(pl_tbl, "formats", "[entropy_optimization.plots]")
    if not isinstance(raw_fmts, list) or not raw_fmts:
        raise ValueError("'formats' in [entropy_optimization.plots] must be a non-empty list")
    formats = tuple(str(fmt).lower().lstrip(".") for fmt in raw_fmts)
    plots_cfg = PlotsOptimizationConfig(output_dir=output_dir, dpi=dpi, formats=formats)

    # Linked CNN reference parameters (no hardcoded 0.5 normalization)
    with cnn_cfg_path.open("rb") as f_cnn:
        cnn_raw = tomllib.load(f_cnn)
    cnn_tbl = _require_table(cnn_raw, "cnn", str(cnn_cfg_path))
    cnn_data_tbl = _require_table(cnn_tbl, "data", "[cnn]")
    cnn_ref = CnnReferenceConfig(
        normalize_mean=_require_float(cnn_data_tbl, "normalize_mean", "[cnn.data]"),
        normalize_std=_require_float(cnn_data_tbl, "normalize_std", "[cnn.data]"),
        image_size=_require_int(cnn_data_tbl, "image_size", "[cnn.data]"),
        reference_models=cnn_reference_models,
    )

    # Linked LR parameters
    with cls_cfg_path.open("rb") as f_cls:
        cls_raw = tomllib.load(f_cls)
    cls_tbl = _require_table(cls_raw, "classification", str(cls_cfg_path))
    cls_models_tbl = _require_table(cls_tbl, "models", "[classification]")
    lr_tbl = _require_table(cls_models_tbl, "lr", "[classification.models]")
    cw_raw = lr_tbl.get("class_weight")
    lr_ref = LrReferenceConfig(
        c_value=_require_float(lr_tbl, "C", "[classification.models.lr]"),
        penalty=str(_require(lr_tbl, "penalty", "[classification.models.lr]")),
        solver=str(_require(lr_tbl, "solver", "[classification.models.lr]")),
        class_weight=str(cw_raw) if cw_raw is not None else None,
        max_iter=_require_int(lr_tbl, "max_iter", "[classification.models.lr]"),
        scale_features=_require_bool(lr_tbl, "scale_features", "[classification.models.lr]"),
    )

    return EntropyOptimizationConfig(
        project_root=resolved_root,
        config_path=resolved_cfg,
        entropy_config_path=entropy_cfg_path,
        base_config_path=base_cfg_path,
        cnn_config_path=cnn_cfg_path,
        classification_config_path=cls_cfg_path,
        seed=seed,
        variants=variants,
        reference_variant=reference_variant,
        grid_levels=grid_levels,
        bins=bins,
        dataset_key=dataset_key,
        dataset_name=dataset_name,
        inverted=inverted,
        data=data_cfg,
        native=native_cfg,
        protocol=protocol_cfg,
        sweeps=sweeps_cfg,
        plots=plots_cfg,
        cnn_ref=cnn_ref,
        lr_ref=lr_ref,
    )


def generate_synthetic_images(n_images: int, image_size: int, seed: int) -> np.ndarray:
    """Create deterministic structured + textured [N, H, W] uint8 images for testing/benchmarking."""
    rng = np.random.default_rng(seed)
    images = np.empty((n_images, image_size, image_size), dtype=np.uint8)
    half = max(1, image_size // 2)
    grad_row = np.linspace(0, 255, image_size, dtype=np.uint8)
    grad = np.tile(grad_row, (image_size, 1))
    block = max(2, image_size // 16)
    checker = (
        (np.indices((image_size // block, image_size // block)).sum(axis=0) % 2) * 255
    ).astype(np.uint8)
    checker_full = np.repeat(np.repeat(checker, block, axis=0), block, axis=1)

    for i in range(n_images):
        mode = i % 4
        if mode == 0:
            base = grad.copy()
            base[:half, :half] = rng.integers(0, 256, size=(half, half), dtype=np.uint8)
            images[i] = base
        elif mode == 1:
            canvas = np.full((image_size, image_size), (i * 29) % 256, dtype=np.uint8)
            y0, y1 = image_size // 4, (3 * image_size) // 4
            canvas[y0:y1, y0:y1] = rng.integers(64, 220, size=(y1 - y0, y1 - y0), dtype=np.uint8)
            images[i] = canvas
        elif mode == 2:
            images[i, : checker_full.shape[0], : checker_full.shape[1]] = checker_full
            if checker_full.shape[0] < image_size:
                images[i, checker_full.shape[0] :, :] = 128
                images[i, :, checker_full.shape[1] :] = 128
            noise = rng.integers(0, 32, size=(image_size, image_size), dtype=np.uint8)
            images[i] = np.bitwise_xor(images[i], noise)
        else:
            images[i] = rng.integers(0, 256, size=(image_size, image_size), dtype=np.uint8)
    return np.ascontiguousarray(images)


def find_preprocessed_image_paths(data_cfg: DataOptimizationConfig) -> list[Path]:
    """List sorted preprocessed non-mask image paths for real sampling."""
    root = data_cfg.preprocessed_dir
    if not root.is_dir():
        return []

    exts = set(data_cfg.image_extensions)
    candidates: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if data_cfg.mask_folder and data_cfg.mask_folder in p.parts:
            continue
        if data_cfg.mask_suffix and p.stem.endswith(data_cfg.mask_suffix):
            continue
        candidates.append(p)
    candidates.sort()
    return candidates


def load_benchmark_images(
    cfg: EntropyOptimizationConfig,
    source_override: str | None = None,
) -> tuple[np.ndarray, str]:
    """Load deterministic real preprocessed images or generate synthetic images per config."""
    active_source = (source_override or cfg.data.source).lower().strip()
    if active_source not in {"real", "synthetic"}:
        raise ValueError(f"Unsupported data source '{active_source}'. Expected 'real' or 'synthetic'.")

    if active_source == "synthetic":
        imgs = generate_synthetic_images(cfg.data.n_real_samples, cfg.data.image_size, cfg.seed)
        return imgs, "synthetic"

    paths = find_preprocessed_image_paths(cfg.data)
    if not paths:
        raise FileNotFoundError(
            f"No preprocessed images found under {cfg.data.preprocessed_dir}. "
            "Run dataset preprocessing first or pass '--source synthetic'."
        )

    n_take = min(cfg.data.n_real_samples, len(paths))
    rng = np.random.default_rng(cfg.seed)
    chosen_idx = np.sort(rng.choice(len(paths), size=n_take, replace=False))
    chosen_paths = [paths[int(idx)] for idx in chosen_idx]

    sz = cfg.data.image_size
    out = np.empty((n_take, sz, sz), dtype=np.uint8)
    for i, path in enumerate(chosen_paths):
        with Image.open(path) as im:
            gray = im.convert("L") if im.mode != "L" else im
            if gray.size != (sz, sz):
                raise ValueError(
                    f"Preprocessed image {path} has size {gray.size}, expected ({sz}, {sz})"
                )
            out[i] = np.asarray(gray, dtype=np.uint8)

    return np.ascontiguousarray(out), f"real:{cfg.dataset_name}"


def resize_u8_batch(images_u8: np.ndarray, target_size: int) -> np.ndarray:
    """Deterministically resize [N, H, W] uint8 batch outside timed loops."""
    if images_u8.shape[1] == target_size and images_u8.shape[2] == target_size:
        return np.ascontiguousarray(images_u8)
    resized = np.empty((images_u8.shape[0], target_size, target_size), dtype=np.uint8)
    for i in range(images_u8.shape[0]):
        with Image.fromarray(images_u8[i], mode="L") as im:
            res_im = im.resize((target_size, target_size), resample=Image.Resampling.BILINEAR)
            resized[i] = np.asarray(res_im, dtype=np.uint8)
    return np.ascontiguousarray(resized)


def build_nlogn_lut(max_pixels: int) -> np.ndarray:
    """Precompute c * log2(c) lookup table for integer counts c in [0, max_pixels]."""
    if max_pixels < 0:
        raise ValueError(f"max_pixels must be >= 0, got {max_pixels}")
    lut = np.zeros(max_pixels + 1, dtype=np.float64)
    if max_pixels >= 1:
        c = np.arange(1, max_pixels + 1, dtype=np.float64)
        lut[1:] = c * np.log2(c)
    return np.ascontiguousarray(lut)


def validate_nested_grid_levels(image_size: int, grid_levels: Sequence[int]) -> tuple[int, ...]:
    """Validate grid_levels and enforce exact finest-grid nesting divisibility."""
    validated = validate_grid_levels(list(grid_levels), image_size)
    g_max = max(validated)
    if image_size % g_max != 0:
        raise ValueError(
            f"image_size={image_size} must be divisible by finest grid level g_max={g_max} "
            "for exact single-pass nesting"
        )
    for g in validated:
        if g_max % g != 0:
            raise ValueError(
                f"finest grid level g_max={g_max} must be divisible by level g={g} "
                "for exact child summation"
            )
    return validated
