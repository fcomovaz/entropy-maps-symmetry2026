"""Pure mathematical and vectorized operations for visual region entropy features.

Computes Shannon entropy across multi-scale spatial grid partitions of grayscale
images without hardcoded scales or bin counts, matching the canonical reference
oracle while providing high-throughput batch vectorization and Look-Up Table (LUT)
optimizations to eliminate redundant calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
import functools
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from src.features.native_entropy import (
    NativePlanBuffers,
    check_native_compatibility,
    load_native_library,
    prepare_native_buffers,
    resolve_default_native_dll_path,
    run_native_entropy,
)

ENTROPY_BACKENDS: tuple[str, ...] = (
    "auto",
    "numpy_baseline",
    "numpy_finest_pyramid",
    "numpy_fused_entropy",
    "c_single_pass",
)


def validate_grid_levels(levels: Sequence[int], image_size: int) -> tuple[int, ...]:
    """Validate spatial grid levels against image dimensions and uniqueness.

    Args:
        levels: Sequence of integer scale factors g representing g x g region partitions.
        image_size: Spatial height/width of square images being partitioned.

    Returns:
        Validated tuple of integer grid levels.

    Raises:
        ValueError: If levels is empty, contains non-integers, non-positive values,
            duplicates, or levels larger than image_size.
    """
    if not levels:
        raise ValueError("grid_levels cannot be empty")

    for g in levels:
        if type(g) is not int:
            raise ValueError(f"grid level '{g}' must be an integer, got {type(g).__name__}")
        if g < 1:
            raise ValueError(f"grid level must be >= 1, got {g}")
        if g > image_size:
            raise ValueError(
                f"grid level {g} exceeds image dimension {image_size}; requires g <= image_size"
            )

    if len(set(levels)) != len(levels):
        raise ValueError(f"grid_levels contains duplicate levels: {levels}")

    return tuple(int(g) for g in levels)


def validate_entropy_backend(backend: str) -> str:
    """Validate and normalize an entropy extraction backend identifier."""
    normalized = str(backend).strip().lower()
    if normalized not in ENTROPY_BACKENDS:
        raise ValueError(
            f"Unsupported entropy backend '{backend}'. Valid options: {ENTROPY_BACKENDS}"
        )
    return normalized


def is_nested_grid_levels(image_size: int, grid_levels: Sequence[int]) -> bool:
    """Return True when image_size and grid_levels satisfy exact finest-grid nesting."""
    if type(image_size) is not int or image_size < 1:
        return False
    try:
        validated = validate_grid_levels(grid_levels, image_size)
    except ValueError:
        return False
    g_max = max(validated)
    if image_size % g_max != 0:
        return False
    return all(g_max % g == 0 for g in validated)


def validate_nested_grid_levels(image_size: int, grid_levels: Sequence[int]) -> tuple[int, ...]:
    """Validate grid_levels and enforce exact finest-grid nesting divisibility."""
    validated = validate_grid_levels(grid_levels, image_size)
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


@functools.lru_cache(maxsize=32)
def build_nlogn_lut(max_pixels: int) -> np.ndarray:
    """Precompute c * log2(c) lookup table for integer counts c in [0, max_pixels]."""
    if max_pixels < 0:
        raise ValueError(f"max_pixels must be >= 0, got {max_pixels}")
    lut = np.zeros(max_pixels + 1, dtype=np.float64)
    if max_pixels >= 1:
        c = np.arange(1, max_pixels + 1, dtype=np.float64)
        lut[1:] = c * np.log2(c)
    return np.ascontiguousarray(lut)


def resolve_entropy_backend(
    backend: str,
    image_size: int,
    grid_levels: Sequence[int],
    bins: int,
    dll_path: Path | None = None,
) -> str:
    """Resolve an entropy backend ('auto' or explicit) while failing fast on explicit invalid choices."""
    normalized = validate_entropy_backend(backend)
    validated_levels = validate_grid_levels(grid_levels, image_size)
    validated_bins = validate_bins(bins)

    if normalized == "numpy_baseline":
        return "numpy_baseline"

    candidate_dll = dll_path if dll_path is not None else resolve_default_native_dll_path()

    if normalized == "auto":
        if not is_nested_grid_levels(image_size, validated_levels):
            return "numpy_baseline"
        if candidate_dll is not None and Path(candidate_dll).is_file():
            ok, _ = check_native_compatibility(
                image_size,
                validated_levels,
                validated_bins,
                dll_path=Path(candidate_dll),
            )
            if ok:
                return "c_single_pass"
        return "numpy_finest_pyramid"

    # Explicit fast backends require exact nesting (no silent fallback).
    validate_nested_grid_levels(image_size, validated_levels)
    if normalized in {"numpy_finest_pyramid", "numpy_fused_entropy"}:
        return normalized

    # Explicit c_single_pass requires built DLL and full structural compatibility.
    if candidate_dll is None:
        raise FileNotFoundError(
            "Native DLL configuration not found for backend 'c_single_pass'. "
            "Build the DLL via '.\\.venv\\Scripts\\python.exe -m src.features.build_native'."
        )
    resolved_dll = Path(candidate_dll).resolve()
    load_native_library(resolved_dll)
    ok, reason = check_native_compatibility(
        image_size,
        validated_levels,
        validated_bins,
        dll_path=resolved_dll,
    )
    if not ok:
        raise ValueError(f"Backend 'c_single_pass' incompatible: {reason}")
    return "c_single_pass"


def validate_bins(bins: int) -> int:
    """Validate histogram quantization bin count.

    Args:
        bins: Number of discrete intensity bins B.

    Returns:
        Validated integer bin count.

    Raises:
        ValueError: If bins is not an integer or is outside the range [2, 256].
    """
    if type(bins) is not int or isinstance(bins, bool):
        raise ValueError(f"bins must be an integer, got {type(bins).__name__}")
    if not 2 <= bins <= 256:
        raise ValueError(f"bins must be an integer between 2 and 256, got {bins}")
    return int(bins)


def build_quantization_lut(bins: int) -> np.ndarray:
    """Precompute a 256-element uint8 Look-Up Table (LUT) for intensity bin quantization.

    Maps pixel in [0, 255] to (pixel * bins) // 256 using uint16 intermediate arithmetic.
    Evaluating quantization via direct table indexing eliminates millions of repetitive
    multiplications and divisions across large batches of images.

    Args:
        bins: Number of histogram bins B in [2, 256].

    Returns:
        1D array of shape [256], dtype uint8, mapping pixel value to bin index in [0, bins - 1].
    """
    validate_bins(bins)
    p = np.arange(256, dtype=np.uint16)
    return ((p * bins) // 256).astype(np.uint8)


@functools.lru_cache(maxsize=128)
def build_shannon_term_lut(n_pixels: int) -> np.ndarray:
    """Precompute Shannon entropy terms -p * log2(p) for integer counts c in [0, n_pixels].

    In a region of N pixels, any histogram bin count c can only assume integer values in [0, N].
    Precomputing - (c/N) * log2(c/N) into an (N + 1)-element float64 table eliminates all
    transcendental log2() evaluations and floating-point divisions during extraction loops.
    By definition, 0 * log2(0) evaluates to 0.0.

    Args:
        n_pixels: Total pixel count N in the region.

    Returns:
        1D array of shape [n_pixels + 1], dtype float64, where entry c is -p * log2(p).
    """
    if n_pixels <= 0:
        return np.zeros(1, dtype=np.float64)
    c = np.arange(n_pixels + 1, dtype=np.float64)
    lut = np.zeros(n_pixels + 1, dtype=np.float64)
    p = c[1:] / n_pixels
    lut[1:] = -p * np.log2(p)
    return lut


@dataclass(frozen=True, slots=True)
class RegionSpec:
    """Precomputed geometric bounds and term LUT for a single spatial partition region."""

    g: int
    r: int
    c: int
    y0: int
    y1: int
    x0: int
    x1: int
    n_pixels: int
    term_lut: np.ndarray


@dataclass(frozen=True, slots=True)
class EntropyPlan:
    """Immutable execution plan caching precomputed geometries, Look-Up Tables, and C buffers."""

    image_size: int
    grid_levels: tuple[int, ...]
    bins: int
    log2_bins: float
    quant_lut: np.ndarray
    regions: tuple[RegionSpec, ...]
    total_features: int
    backend: str = "numpy_baseline"
    resolved_backend: str = "numpy_baseline"
    g_max: int = 1
    cell_side: int = 1
    desc_levels: tuple[int, ...] = ()
    level_offsets: tuple[tuple[int, int, int], ...] = ()
    merge_schedule: tuple[tuple[int, int, int], ...] = ()
    nlogn_lut: np.ndarray | None = None
    npix_int_vec: np.ndarray | None = None
    npix_f64_vec: np.ndarray | None = None
    log2_npix_vec: np.ndarray | None = None
    term_luts_by_level: tuple[tuple[int, np.ndarray], ...] = ()
    native_buffers: NativePlanBuffers | None = None


def build_entropy_plan(
    image_size: int,
    grid_levels: Sequence[int],
    bins: int,
    backend: str = "auto",
    dll_path: Path | None = None,
) -> EntropyPlan:
    """Precompute spatial partition coordinates, LUTs, and optional fast pyramid / C buffers.

    Args:
        image_size: Height/width of square images.
        grid_levels: Sequence of grid scales g (e.g. [1, 2, 4, 8]).
        bins: Histogram bin count B in [2, 256].
        backend: Requested execution backend ('auto', 'numpy_baseline',
            'numpy_finest_pyramid', 'numpy_fused_entropy', or 'c_single_pass').
        dll_path: Optional explicit path to compiled native C99 DLL.

    Returns:
        EntropyPlan holding all precalculated structures for zero-recalculation execution.
    """
    if type(image_size) is not int or image_size < 1:
        raise ValueError(f"image_size must be a positive integer, got {image_size!r}")
    requested_backend = validate_entropy_backend(backend)
    validated_levels = validate_grid_levels(grid_levels, image_size)
    validated_bins = validate_bins(bins)
    resolved_backend = resolve_entropy_backend(
        requested_backend,
        image_size,
        validated_levels,
        validated_bins,
        dll_path=dll_path,
    )
    log2_bins = math.log2(validated_bins)
    quant_lut = build_quantization_lut(validated_bins)

    regions_list: list[RegionSpec] = []
    for g in validated_levels:
        y_edges = region_edges(image_size, g)
        x_edges = region_edges(image_size, g)
        for r in range(g):
            y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
            for c in range(g):
                x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
                n_pix = (y1 - y0) * (x1 - x0)
                term_lut = build_shannon_term_lut(n_pix)
                regions_list.append(
                    RegionSpec(
                        g=g,
                        r=r,
                        c=c,
                        y0=y0,
                        y1=y1,
                        x0=x0,
                        x1=x1,
                        n_pixels=n_pix,
                        term_lut=term_lut,
                    )
                )

    total_features = sum(g * g for g in validated_levels)

    g_max = max(validated_levels)
    cell_side = image_size // g_max if image_size % g_max == 0 else 1
    desc_levels: tuple[int, ...] = ()
    level_offsets: tuple[tuple[int, int, int], ...] = ()
    merge_schedule: tuple[tuple[int, int, int], ...] = ()
    nlogn_lut: np.ndarray | None = None
    npix_int_vec: np.ndarray | None = None
    npix_f64_vec: np.ndarray | None = None
    log2_npix_vec: np.ndarray | None = None
    term_luts_by_level: tuple[tuple[int, np.ndarray], ...] = ()
    native_buffers: NativePlanBuffers | None = None

    if resolved_backend != "numpy_baseline":
        desc_levels = tuple(sorted(validated_levels, reverse=True))
        offsets_list: list[tuple[int, int, int]] = []
        cursor = 0
        for g in validated_levels:
            span = g * g
            offsets_list.append((g, cursor, cursor + span))
            cursor += span
        level_offsets = tuple(offsets_list)

        schedule_list: list[tuple[int, int, int]] = []
        for idx in range(1, len(desc_levels)):
            g_curr = desc_levels[idx]
            g_child = next(
                desc_levels[j]
                for j in range(idx - 1, -1, -1)
                if desc_levels[j] % g_curr == 0
            )
            schedule_list.append((g_curr, g_child, g_child // g_curr))
        merge_schedule = tuple(schedule_list)

        if resolved_backend == "numpy_finest_pyramid":
            term_luts_by_level = tuple(
                (g, build_shannon_term_lut((image_size // g) ** 2))
                for g in validated_levels
            )
        elif resolved_backend == "numpy_fused_entropy":
            g_min = min(validated_levels)
            nlogn_lut = build_nlogn_lut((image_size // g_min) ** 2)
            npix_int = np.empty(total_features, dtype=np.int64)
            for g, start, end in level_offsets:
                side = image_size // g
                npix_int[start:end] = side * side
            npix_int_vec = npix_int
            npix_f64_vec = npix_int.astype(np.float64)
            log2_npix_vec = np.log2(npix_f64_vec)
        elif resolved_backend == "c_single_pass":
            active_dll = dll_path if dll_path is not None else resolve_default_native_dll_path()
            if active_dll is None:
                raise FileNotFoundError("No native DLL path available for backend 'c_single_pass'")
            g_min = min(validated_levels)
            nlogn_lut = build_nlogn_lut((image_size // g_min) ** 2)
            native_buffers = prepare_native_buffers(
                image_size=image_size,
                grid_levels=validated_levels,
                bins=validated_bins,
                dll_path=Path(active_dll),
                qlut=quant_lut,
                nlogn_lut=nlogn_lut,
            )

    return EntropyPlan(
        image_size=image_size,
        grid_levels=validated_levels,
        bins=validated_bins,
        log2_bins=log2_bins,
        quant_lut=quant_lut,
        regions=tuple(regions_list),
        total_features=total_features,
        backend=requested_backend,
        resolved_backend=resolved_backend,
        g_max=g_max,
        cell_side=cell_side,
        desc_levels=desc_levels,
        level_offsets=level_offsets,
        merge_schedule=merge_schedule,
        nlogn_lut=nlogn_lut,
        npix_int_vec=npix_int_vec,
        npix_f64_vec=npix_f64_vec,
        log2_npix_vec=log2_npix_vec,
        term_luts_by_level=term_luts_by_level,
        native_buffers=native_buffers,
    )


def region_edges(size: int, g: int) -> np.ndarray:
    """Compute 1D region boundaries for a dimension of given size divided into g regions.

    Boundaries are calculated as floor(k * size / g) using exact integer division,
    matching partition_boxes from the canonical reference.

    Args:
        size: Dimension length in pixels.
        g: Number of partition segments.

    Returns:
        1D array of shape [g + 1] containing segment boundary indices [0, ..., size].
    """
    k = np.arange(g + 1, dtype=np.int64)
    return (k * size) // g


def quantize_pixels(
    images: np.ndarray,
    bins: int,
    lut: np.ndarray | None = None,
) -> np.ndarray:
    """Quantize uint8 pixel intensities into discrete histogram bins [0, bins - 1].

    Uses a precomputed 256-element byte Look-Up Table (LUT) for O(1) table indexing.

    Args:
        images: Array of grayscale pixel intensities in [0, 255], shape [..., H, W].
        bins: Number of histogram bins B in [2, 256].
        lut: Optional precomputed 256-element uint8 LUT from build_quantization_lut.

    Returns:
        Array of quantized bin indices in [0, bins - 1], dtype uint8, preserving shape.
    """
    if lut is None:
        lut = build_quantization_lut(bins)
    arr = np.asarray(images)
    return lut[arr]


def region_histograms(quantized: np.ndarray, g: int, bins: int) -> np.ndarray:
    """Compute discrete histogram bin counts for all g*g regions across a batch of images.

    Args:
        quantized: Batch of quantized images [N, H, W] with values in [0, bins - 1].
        g: Grid partition scale factor, producing g*g regions in row-major order.
        bins: Number of discrete histogram bins B.

    Returns:
        Array of histogram counts of shape [N, g*g, bins], dtype int64.
    """
    validate_bins(bins)
    if quantized.ndim != 3:
        raise ValueError(f"quantized images must have shape [N, H, W], got {quantized.shape}")

    n_images, height, width = quantized.shape
    total_regions = g * g
    counts = np.zeros((n_images, total_regions, bins), dtype=np.int64)

    if n_images == 0:
        return counts

    y_edges = region_edges(height, g)
    x_edges = region_edges(width, g)

    # Offset trick using int32: reduces memory footprint by 50% vs int64
    image_offsets = np.arange(n_images, dtype=np.int32)[:, None] * bins

    for r in range(g):
        y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
        for c in range(g):
            x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
            reg_idx = r * g + c
            patch = quantized[:, y0:y1, x0:x1]  # shape [N, reg_h, reg_w]
            flat_values = (patch.reshape(n_images, -1).astype(np.int32) + image_offsets).ravel()
            bincounts = np.bincount(flat_values, minlength=n_images * bins).reshape(n_images, bins)
            counts[:, reg_idx, :] = bincounts

    return counts


def normalized_entropy(
    counts: np.ndarray,
    bins: int,
    term_lut: np.ndarray | None = None,
) -> np.ndarray:
    """Compute normalized Shannon entropy H / log2(B) from region histogram counts.

    Calculates H = -sum(p * log2(p)) where p > 0, normalized by log2(bins).
    When term_lut is provided, evaluates terms via direct Look-Up Table indexing,
    completely bypassing floating-point division and transcendental log2 calls.
    When all histogram counts in a region are identical (equiprobable distribution),
    analytical 1.0 is returned directly to avoid floating-point summation overshoot,
    strictly mirroring the reference implementation.

    Args:
        counts: Histogram counts array of shape [..., bins].
        bins: Total number of histogram bins B in [2, 256].
        term_lut: Optional precomputed Shannon term LUT from build_shannon_term_lut.

    Returns:
        Array of normalized entropy values in [0.0, 1.0], shape [...], dtype float64.
    """
    validate_bins(bins)
    n_pixels = counts.sum(axis=-1, keepdims=True)  # shape [..., 1]
    n_pixels_flat = n_pixels.squeeze(-1)
    log2_bins = math.log2(bins)

    if term_lut is not None and np.all(n_pixels_flat == (len(term_lut) - 1)):
        # Fast LUT branch: direct table indexing with zero transcendental function calls
        terms = term_lut[counts]
        entropy_bits = np.sum(terms, axis=-1)
    else:
        # Dynamic fallback branch
        p = np.zeros(counts.shape, dtype=np.float64)
        np.divide(counts, n_pixels, out=p, where=(n_pixels > 0))

        terms = np.zeros_like(p)
        positive_mask = p > 0.0
        terms[positive_mask] = p[positive_mask] * np.log2(p[positive_mask])

        entropy_bits = -np.sum(terms, axis=-1)

    norm_entropy = entropy_bits / log2_bins

    # Analytical equiprobable rule (mirrors reference oracle to avoid float overshoot)
    is_equiprobable = np.all(counts == counts[..., :1], axis=-1) & (n_pixels_flat > 0)
    norm_entropy[is_equiprobable] = 1.0

    # Ensure empty regions evaluate cleanly to 0.0
    norm_entropy[n_pixels_flat == 0] = 0.0

    return norm_entropy


def _fine_histograms_from_quantized(
    quantized: np.ndarray,
    g_max: int,
    cell_side: int,
    bins: int,
) -> np.ndarray:
    """Compute [N, g_max * g_max, bins] histogram counts via one global np.bincount call."""
    n_images = int(quantized.shape[0])
    n_cells = g_max * g_max
    total_slots = n_images * n_cells * bins
    idx_dtype = np.int32 if total_slots <= 2_000_000_000 else np.int64
    cells_view = quantized.reshape(n_images, g_max, cell_side, g_max, cell_side).transpose(
        0, 1, 3, 2, 4
    )
    cell_offsets = (np.arange(n_images * n_cells, dtype=idx_dtype) * bins).reshape(
        n_images, g_max, g_max, 1, 1
    )
    flat_indices = np.add(cells_view, cell_offsets, dtype=idx_dtype).ravel()
    return np.bincount(flat_indices, minlength=total_slots).reshape(n_images, n_cells, bins)


def _compute_pyramid_counts_from_fine(
    fine_counts: np.ndarray,
    plan: EntropyPlan,
) -> dict[int, np.ndarray]:
    """Sum finest-grid child histograms upward across plan.merge_schedule."""
    n_images = int(fine_counts.shape[0])
    counts_by_level: dict[int, np.ndarray] = {plan.g_max: fine_counts}
    for g_curr, g_child, k in plan.merge_schedule:
        child_counts = counts_by_level[g_child]
        counts_by_level[g_curr] = (
            child_counts.reshape(n_images, g_curr, k, g_curr, k, plan.bins)
            .sum(axis=(2, 4))
            .reshape(n_images, g_curr * g_curr, plan.bins)
        )
    return counts_by_level


def _compute_finest_pyramid_features(images_arr: np.ndarray, plan: EntropyPlan) -> np.ndarray:
    """Evaluate single-pass finest-grid NumPy pyramid + per-level Shannon term LUTs."""
    n_images = int(images_arr.shape[0])
    quantized = quantize_pixels(images_arr, plan.bins, lut=plan.quant_lut)
    fine_counts = _fine_histograms_from_quantized(
        quantized, plan.g_max, plan.cell_side, plan.bins
    )
    counts_by_level = _compute_pyramid_counts_from_fine(fine_counts, plan)
    term_lut_map = dict(plan.term_luts_by_level)

    out = np.empty((n_images, plan.total_features), dtype=np.float64)
    for g, start, end in plan.level_offsets:
        counts_g = counts_by_level[g]
        terms = term_lut_map[g][counts_g]
        h_g = np.sum(terms, axis=-1) / plan.log2_bins
        is_equiprobable = np.all(counts_g == counts_g[:, :, :1], axis=-1)
        h_g[is_equiprobable] = 1.0
        out[:, start:end] = h_g
    return out


def _compute_fused_entropy_features(images_arr: np.ndarray, plan: EntropyPlan) -> np.ndarray:
    """Evaluate finest-grid pyramid + single fused [N, total_features, bins] n*log2(n) LUT pass."""
    assert plan.nlogn_lut is not None
    assert plan.npix_int_vec is not None
    assert plan.npix_f64_vec is not None
    assert plan.log2_npix_vec is not None

    n_images = int(images_arr.shape[0])
    quantized = quantize_pixels(images_arr, plan.bins, lut=plan.quant_lut)
    all_counts = np.empty((n_images, plan.total_features, plan.bins), dtype=np.int64)
    offset_map = {g: (start, end) for g, start, end in plan.level_offsets}

    f_start, f_end = offset_map[plan.g_max]
    all_counts[:, f_start:f_end, :] = _fine_histograms_from_quantized(
        quantized, plan.g_max, plan.cell_side, plan.bins
    )

    for g_curr, g_child, k in plan.merge_schedule:
        c_start, c_end = offset_map[g_child]
        t_start, t_end = offset_map[g_curr]
        all_counts[:, t_start:t_end, :] = (
            all_counts[:, c_start:c_end, :]
            .reshape(n_images, g_curr, k, g_curr, k, plan.bins)
            .sum(axis=(2, 4))
            .reshape(n_images, g_curr * g_curr, plan.bins)
        )

    s_vec = plan.nlogn_lut[all_counts].sum(axis=-1)
    h_all = (
        plan.log2_npix_vec[None, :] - (s_vec / plan.npix_f64_vec[None, :])
    ) / plan.log2_bins
    np.maximum(h_all, 0.0, out=h_all)
    is_const = np.max(all_counts, axis=-1) == plan.npix_int_vec[None, :]
    h_all[is_const] = 0.0
    is_equiprobable = np.all(all_counts == all_counts[:, :, :1], axis=-1) & (
        plan.npix_int_vec[None, :] > 0
    )
    h_all[is_equiprobable] = 1.0
    return h_all


def _can_reuse_plan(
    plan: EntropyPlan | None,
    height: int,
    grid_levels: Sequence[int],
    bins: int,
    backend: str | None,
    dll_path: Path | None,
) -> bool:
    """Return True when cached EntropyPlan satisfies dimensions, backend, and DLL identity."""
    if plan is None:
        return False
    if (plan.image_size, plan.grid_levels, plan.bins) != (height, tuple(grid_levels), bins):
        return False
    if backend is not None:
        req = validate_entropy_backend(backend)
        if req not in (plan.backend, plan.resolved_backend):
            return False
    if dll_path is not None and plan.resolved_backend == "c_single_pass":
        if plan.native_buffers is None:
            return False
        if plan.native_buffers.dll_path != Path(dll_path).resolve():
            return False
    return True


def compute_entropy_features(
    images: np.ndarray,
    grid_levels: Sequence[int],
    bins: int,
    plan: EntropyPlan | None = None,
    backend: str | None = None,
    dll_path: Path | None = None,
) -> np.ndarray:
    """Extract concatenated multi-scale region entropy features for a batch of images.

    When an EntropyPlan is supplied with matching dimensions and backend (or `backend=None`),
    cached boundaries, LUTs, and native C buffers are reused directly without re-initialization.

    Args:
        images: Array of square grayscale images of shape [N, H, W] or [H, W] (uint8).
        grid_levels: Sequence of grid partition levels g (e.g. [1, 2, 4, 8]).
        bins: Discrete histogram bin count B (e.g. 16).
        plan: Optional precomputed EntropyPlan caching geometries, LUTs, and native buffers.
        backend: Optional backend override ('auto', 'numpy_baseline', 'numpy_finest_pyramid',
            'numpy_fused_entropy', or 'c_single_pass'). Defaults to plan's backend when a
            compatible plan is passed, or 'auto' otherwise.
        dll_path: Optional explicit path to compiled native C99 DLL.

    Returns:
        Array of normalized entropy features of shape [N, sum(g^2)], dtype float64,
        ordered by grid_levels and row-major within each grid level.
    """
    images_arr = np.asarray(images)
    if images_arr.ndim == 2:
        images_arr = images_arr[None, ...]
    elif images_arr.ndim != 3:
        raise ValueError(
            f"images must have shape [N, H, W] or [H, W], got shape {images_arr.shape}"
        )

    n_images, height, width = images_arr.shape
    if height != width:
        raise ValueError(f"Images must be square, got {height}x{width}")

    if not _can_reuse_plan(plan, height, grid_levels, bins, backend, dll_path):
        active_backend = "auto" if backend is None else backend
        plan = build_entropy_plan(
            height,
            grid_levels,
            bins,
            backend=active_backend,
            dll_path=dll_path,
        )
    assert plan is not None

    if n_images == 0:
        return np.zeros((0, plan.total_features), dtype=np.float64)

    if plan.resolved_backend == "c_single_pass":
        assert plan.native_buffers is not None
        return run_native_entropy(plan.native_buffers, images_arr)

    if plan.resolved_backend == "numpy_finest_pyramid":
        if images_arr.dtype != np.uint8:
            raise ValueError(
                f"numpy_finest_pyramid requires uint8 images, got dtype {images_arr.dtype}"
            )
        return _compute_finest_pyramid_features(images_arr, plan)

    if plan.resolved_backend == "numpy_fused_entropy":
        if images_arr.dtype != np.uint8:
            raise ValueError(
                f"numpy_fused_entropy requires uint8 images, got dtype {images_arr.dtype}"
            )
        return _compute_fused_entropy_features(images_arr, plan)

    quantized = quantize_pixels(images_arr, plan.bins, lut=plan.quant_lut)

    image_offsets = np.arange(n_images, dtype=np.int32)[:, None] * plan.bins
    total_bins_offset = n_images * plan.bins

    out_features = np.zeros((n_images, plan.total_features), dtype=np.float64)

    # Baseline coordinate loop across regions
    for feat_idx, reg in enumerate(plan.regions):
        patch = quantized[:, reg.y0:reg.y1, reg.x0:reg.x1]
        flat_vals = (patch.reshape(n_images, -1).astype(np.int32) + image_offsets).ravel()
        counts = np.bincount(flat_vals, minlength=total_bins_offset).reshape(n_images, plan.bins)

        terms = reg.term_lut[counts]
        h = np.sum(terms, axis=-1) / plan.log2_bins

        is_equiprobable = np.all(counts == counts[:, :1], axis=-1)
        h[is_equiprobable] = 1.0

        out_features[:, feat_idx] = h

    return out_features


def feature_names(
    grid_levels: Sequence[int],
    template: str = "E_{g}x{g}_r{r}_c{c}",
) -> list[str]:
    """Generate ordered feature column names across the specified grid levels.

    Args:
        grid_levels: Sequence of grid partition levels g (e.g. [1, 2, 4, 8]).
        template: Format string pattern supporting '{g}', '{r}', and '{c}'.

    Returns:
        List of formatted feature column names in level and row-major order.
    """
    names: list[str] = []
    for g in grid_levels:
        for r in range(g):
            for c in range(g):
                names.append(template.format(g=g, r=r, c=c))
    return names
