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
from typing import Sequence

import numpy as np


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
    """Immutable execution plan caching precomputed geometries and Look-Up Tables."""

    image_size: int
    grid_levels: tuple[int, ...]
    bins: int
    log2_bins: float
    quant_lut: np.ndarray
    regions: tuple[RegionSpec, ...]
    total_features: int


def build_entropy_plan(
    image_size: int,
    grid_levels: Sequence[int],
    bins: int,
) -> EntropyPlan:
    """Precompute spatial partition coordinates, quantization LUT, and Shannon term LUTs.

    Args:
        image_size: Height/width of square images.
        grid_levels: Sequence of grid scales g (e.g. [1, 2, 4, 8]).
        bins: Histogram bin count B in [2, 256].

    Returns:
        EntropyPlan holding all precalculated structures for zero-recalculation execution.
    """
    validated_levels = validate_grid_levels(grid_levels, image_size)
    validated_bins = validate_bins(bins)
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
    return EntropyPlan(
        image_size=image_size,
        grid_levels=validated_levels,
        bins=validated_bins,
        log2_bins=log2_bins,
        quant_lut=quant_lut,
        regions=tuple(regions_list),
        total_features=total_features,
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


def compute_entropy_features(
    images: np.ndarray,
    grid_levels: Sequence[int],
    bins: int,
    plan: EntropyPlan | None = None,
) -> np.ndarray:
    """Extract concatenated multi-scale region entropy features for a batch of images.

    When an EntropyPlan is supplied, geometry boundaries, quantization tables, and
    Shannon term Look-Up Tables are reused directly, yielding maximum throughput
    with zero repetitive calculations across chunks.

    Args:
        images: Array of square grayscale images of shape [N, H, W] or [H, W] (uint8).
        grid_levels: Sequence of grid partition levels g (e.g. [1, 2, 4, 8]).
        bins: Discrete histogram bin count B (e.g. 16).
        plan: Optional precomputed EntropyPlan caching geometries and LUTs.

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

    if (
        plan is None
        or plan.image_size != height
        or plan.grid_levels != tuple(grid_levels)
        or plan.bins != bins
    ):
        plan = build_entropy_plan(height, grid_levels, bins)

    if n_images == 0:
        return np.zeros((0, plan.total_features), dtype=np.float64)

    quantized = quantize_pixels(images_arr, plan.bins, lut=plan.quant_lut)

    image_offsets = np.arange(n_images, dtype=np.int32)[:, None] * plan.bins
    total_bins_offset = n_images * plan.bins

    out_features = np.zeros((n_images, plan.total_features), dtype=np.float64)

    # Process all regions directly in a single pass using precomputed geometry and Term-LUTs
    for feat_idx, reg in enumerate(plan.regions):
        patch = quantized[:, reg.y0:reg.y1, reg.x0:reg.x1]
        flat_vals = (patch.reshape(n_images, -1).astype(np.int32) + image_offsets).ravel()
        counts = np.bincount(flat_vals, minlength=total_bins_offset).reshape(n_images, plan.bins)

        # Look-Up Table evaluation of Shannon terms: O(1) table indexing
        terms = reg.term_lut[counts]
        h = np.sum(terms, axis=-1) / plan.log2_bins

        # Analytical equiprobable rule (mirrors reference oracle)
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
