"""Baseline and vectorized NumPy single-pass entropy pyramid variants (V0-V2)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
import time
from typing import Protocol, runtime_checkable

import numpy as np

LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, LAB_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config_loader import build_nlogn_lut, validate_nested_grid_levels
from src.features.entropy_maps import (
    EntropyPlan,
    build_entropy_plan,
    build_quantization_lut,
    build_shannon_term_lut,
    compute_entropy_features,
    quantize_pixels,
    validate_bins,
    validate_grid_levels,
)


@runtime_checkable
class EntropyVariant(Protocol):
    """Common Strategy contract shared across V0-V3 feature extractors."""

    name: str

    def prepare(self, image_size: int, grid_levels: Sequence[int], bins: int) -> None:
        """Precompute geometry plans, lookup tables, and reusable buffers."""
        ...

    def run(self, images_u8: np.ndarray) -> np.ndarray:
        """Compute normalized entropy features: [N, H, W] (or [H, W]) uint8 -> [N, total_features] float64."""
        ...


def normalize_and_validate_images_u8(
    images_u8: np.ndarray,
    expected_size: int,
    variant_name: str,
) -> np.ndarray:
    """Validate uint8 grayscale shape ([N, H, W] or [H, W]) and square image dimension."""
    arr = np.asarray(images_u8)
    if arr.dtype != np.uint8:
        raise ValueError(
            f"{variant_name} requires uint8 images, got dtype {arr.dtype}"
        )
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(
            f"{variant_name} requires shape [N, H, W] or [H, W], got shape {arr.shape}"
        )
    _, height, width = arr.shape
    if height != width:
        raise ValueError(
            f"{variant_name} requires square images, got {height}x{width}"
        )
    if height != expected_size:
        raise ValueError(
            f"{variant_name} prepared for image_size={expected_size}, got {height}x{width}"
        )
    return arr


class V0BaselineVariant:
    """Reference V0 variant backed directly by src.features.entropy_maps.compute_entropy_features."""

    name: str = "V0_baseline"

    def __init__(self) -> None:
        self._image_size: int | None = None
        self._grid_levels: tuple[int, ...] = ()
        self._bins: int = 0
        self._plan: EntropyPlan | None = None

    def prepare(self, image_size: int, grid_levels: Sequence[int], bins: int) -> None:
        """Build cached EntropyPlan matching formal src pipeline."""
        if image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {image_size}")
        validated_levels = validate_grid_levels(list(grid_levels), image_size)
        validated_bins = validate_bins(bins)
        self._image_size = int(image_size)
        self._grid_levels = validated_levels
        self._bins = validated_bins
        self._plan = build_entropy_plan(self._image_size, self._grid_levels, self._bins)

    def run(self, images_u8: np.ndarray) -> np.ndarray:
        """Execute baseline per-region coordinate loop over all grid scales."""
        if self._plan is None or self._image_size is None:
            raise RuntimeError(f"{self.name}.run() called before prepare()")
        arr = normalize_and_validate_images_u8(images_u8, self._image_size, self.name)
        return compute_entropy_features(
            arr,
            grid_levels=self._grid_levels,
            bins=self._bins,
            plan=self._plan,
        )

    def measure_stages_once(self, images_u8: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Diagnostic single execution timing quantization, per-region histograms, and entropy."""
        if self._plan is None or self._image_size is None:
            raise RuntimeError(f"{self.name}.measure_stages_once() called before prepare()")
        arr = normalize_and_validate_images_u8(images_u8, self._image_size, self.name)
        n_images = int(arr.shape[0])
        if n_images == 0:
            return np.zeros((0, self._plan.total_features), dtype=np.float64), {
                "quantize_ms": 0.0,
                "histogram_ms": 0.0,
                "pyramid_sum_ms": 0.0,
                "entropy_ms": 0.0,
            }

        t0 = time.perf_counter_ns()
        quantized = quantize_pixels(arr, self._plan.bins, lut=self._plan.quant_lut)
        quantize_ns = time.perf_counter_ns() - t0

        image_offsets = np.arange(n_images, dtype=np.int32)[:, None] * self._plan.bins
        total_bins_offset = n_images * self._plan.bins
        out_features = np.zeros((n_images, self._plan.total_features), dtype=np.float64)

        hist_ns = 0
        entropy_ns = 0
        for feat_idx, reg in enumerate(self._plan.regions):
            th0 = time.perf_counter_ns()
            patch = quantized[:, reg.y0 : reg.y1, reg.x0 : reg.x1]
            flat_vals = (patch.reshape(n_images, -1).astype(np.int32) + image_offsets).ravel()
            counts = np.bincount(flat_vals, minlength=total_bins_offset).reshape(
                n_images, self._plan.bins
            )
            th1 = time.perf_counter_ns()
            hist_ns += th1 - th0

            te0 = time.perf_counter_ns()
            terms = reg.term_lut[counts]
            h = np.sum(terms, axis=-1) / self._plan.log2_bins
            is_equiprobable = np.all(counts == counts[:, :1], axis=-1)
            h[is_equiprobable] = 1.0
            out_features[:, feat_idx] = h
            te1 = time.perf_counter_ns()
            entropy_ns += te1 - te0

        return out_features, {
            "quantize_ms": quantize_ns / 1e6,
            "histogram_ms": hist_ns / 1e6,
            "pyramid_sum_ms": 0.0,
            "entropy_ms": entropy_ns / 1e6,
        }


class _FinestGridPyramidBase:
    """Shared finest-grid geometry and single-pass bincount machinery for V1 and V2."""

    def __init__(self) -> None:
        self._image_size: int | None = None
        self._grid_levels: tuple[int, ...] = ()
        self._bins: int = 0
        self._log2_bins: float = 0.0
        self._total_features: int = 0
        self._quant_lut: np.ndarray | None = None
        self._g_max: int = 0
        self._cell_side: int = 0
        self._n_fine_cells: int = 0
        # Descending order of grid levels: finest first, coarser levels follow
        self._desc_levels: tuple[int, ...] = ()
        # Maps level g -> (feature_start, feature_end) in caller grid_levels order
        self._level_offsets: dict[int, tuple[int, int]] = {}
        # Child summation schedule: (g_target, g_child_source, merge_factor_k)
        self._merge_schedule: tuple[tuple[int, int, int], ...] = ()
        # Cached level -> Shannon term LUT (used by V1)
        self._term_luts: dict[int, np.ndarray] = {}

    def _prepare_common(
        self,
        image_size: int,
        grid_levels: Sequence[int],
        bins: int,
        include_term_luts: bool = True,
    ) -> None:
        """Validate strict finest-grid nesting and precompute geometry + LUTs."""
        if image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {image_size}")
        validated_levels = validate_nested_grid_levels(image_size, grid_levels)
        validated_bins = validate_bins(bins)

        self._image_size = int(image_size)
        self._grid_levels = validated_levels
        self._bins = validated_bins
        self._log2_bins = float(np.log2(validated_bins))
        self._quant_lut = build_quantization_lut(validated_bins)

        self._g_max = max(validated_levels)
        self._cell_side = self._image_size // self._g_max
        self._n_fine_cells = self._g_max * self._g_max
        self._desc_levels = tuple(sorted(validated_levels, reverse=True))

        offsets: dict[int, tuple[int, int]] = {}
        cursor = 0
        for g in validated_levels:
            span = g * g
            offsets[g] = (cursor, cursor + span)
            cursor += span
        self._total_features = cursor
        self._level_offsets = offsets

        schedule: list[tuple[int, int, int]] = []
        for idx in range(1, len(self._desc_levels)):
            g_curr = self._desc_levels[idx]
            g_child = next(
                self._desc_levels[j]
                for j in range(idx - 1, -1, -1)
                if self._desc_levels[j] % g_curr == 0
            )
            schedule.append((g_curr, g_child, g_child // g_curr))
        self._merge_schedule = tuple(schedule)

        if include_term_luts:
            term_luts: dict[int, np.ndarray] = {}
            for g in validated_levels:
                side = self._image_size // g
                term_luts[g] = build_shannon_term_lut(side * side)
            self._term_luts = term_luts
        else:
            self._term_luts = {}

    def _fine_histograms_from_quantized(self, quantized: np.ndarray) -> np.ndarray:
        """Compute [N, g_max * g_max, bins] histogram counts via one global np.bincount call.

        Tensor flow:
            quantized [N, H, W]
            -> reshape [N, g_max, cell_side, g_max, cell_side]
            -> transpose [N, g_max, g_max, cell_side, cell_side]
            -> global flat indices + bincount -> [N, g_max * g_max, bins]
        """
        n_images = int(quantized.shape[0])
        g_max = self._g_max
        side = self._cell_side
        n_cells = self._n_fine_cells
        bins = self._bins
        total_slots = n_images * n_cells * bins

        # Choose int32 when index space fits safely (<2B) to halve temporary memory traffic
        idx_dtype = np.int32 if total_slots <= 2_000_000_000 else np.int64
        cells_view = quantized.reshape(n_images, g_max, side, g_max, side).transpose(0, 1, 3, 2, 4)
        cell_offsets = (
            np.arange(n_images * n_cells, dtype=idx_dtype) * bins
        ).reshape(n_images, g_max, g_max, 1, 1)
        flat_indices = np.add(cells_view, cell_offsets, dtype=idx_dtype).ravel()

        return (
            np.bincount(flat_indices, minlength=total_slots)
            .reshape(n_images, n_cells, bins)
        )


class V1NumpyFinestPyramidVariant(_FinestGridPyramidBase):
    """V1: single finest-grid bincount pass + child histogram pyramid + per-level entropy LUT."""

    name: str = "V1_numpy_finest_pyramid"

    def prepare(self, image_size: int, grid_levels: Sequence[int], bins: int) -> None:
        """Precompute nested finest-grid geometry and Shannon term LUTs."""
        self._prepare_common(image_size, grid_levels, bins)

    def run(self, images_u8: np.ndarray) -> np.ndarray:
        """Extract entropy vector by reading each pixel once into finest-grid histograms."""
        if self._quant_lut is None or self._image_size is None:
            raise RuntimeError(f"{self.name}.run() called before prepare()")
        arr = normalize_and_validate_images_u8(images_u8, self._image_size, self.name)
        n_images = int(arr.shape[0])
        if n_images == 0:
            return np.zeros((0, self._total_features), dtype=np.float64)

        # 1. Quantize entire batch once: [N, H, W] uint8
        quantized = self._quant_lut[arr]

        # 2. Finest grid histograms via single global bincount: [N, g_max^2, bins]
        counts_by_level: dict[int, np.ndarray] = {
            self._g_max: self._fine_histograms_from_quantized(quantized)
        }

        # 3. Coarser grid histograms by summing [N, g, k, g, k, bins] child blocks
        for g_curr, g_child, k in self._merge_schedule:
            child_counts = counts_by_level[g_child]  # [N, g_child^2, bins]
            counts_by_level[g_curr] = (
                child_counts.reshape(n_images, g_curr, k, g_curr, k, self._bins)
                .sum(axis=(2, 4))
                .reshape(n_images, g_curr * g_curr, self._bins)
            )

        # 4. Evaluate normalized Shannon entropy per scale into ordered [N, total_features]
        out = np.empty((n_images, self._total_features), dtype=np.float64)
        for g in self._grid_levels:
            counts_g = counts_by_level[g]  # [N, g*g, bins]
            term_lut = self._term_luts[g]
            terms = term_lut[counts_g]  # [N, g*g, bins] float64
            h_g = np.sum(terms, axis=-1) / self._log2_bins  # [N, g*g]
            is_equiprobable = np.all(counts_g == counts_g[:, :, :1], axis=-1)
            h_g[is_equiprobable] = 1.0
            start, end = self._level_offsets[g]
            out[:, start:end] = h_g

        return out

    def measure_stages_once(self, images_u8: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Diagnostic single execution reporting V1 stage latencies in milliseconds."""
        if self._quant_lut is None or self._image_size is None:
            raise RuntimeError(f"{self.name}.measure_stages_once() called before prepare()")
        arr = normalize_and_validate_images_u8(images_u8, self._image_size, self.name)
        n_images = int(arr.shape[0])
        if n_images == 0:
            return np.zeros((0, self._total_features), dtype=np.float64), {
                "quantize_ms": 0.0,
                "histogram_ms": 0.0,
                "pyramid_sum_ms": 0.0,
                "entropy_ms": 0.0,
            }

        t0 = time.perf_counter_ns()
        quantized = self._quant_lut[arr]
        t1 = time.perf_counter_ns()

        counts_by_level: dict[int, np.ndarray] = {
            self._g_max: self._fine_histograms_from_quantized(quantized)
        }
        t2 = time.perf_counter_ns()

        for g_curr, g_child, k in self._merge_schedule:
            child_counts = counts_by_level[g_child]
            counts_by_level[g_curr] = (
                child_counts.reshape(n_images, g_curr, k, g_curr, k, self._bins)
                .sum(axis=(2, 4))
                .reshape(n_images, g_curr * g_curr, self._bins)
            )
        t3 = time.perf_counter_ns()

        out = np.empty((n_images, self._total_features), dtype=np.float64)
        for g in self._grid_levels:
            counts_g = counts_by_level[g]
            terms = self._term_luts[g][counts_g]
            h_g = np.sum(terms, axis=-1) / self._log2_bins
            is_equiprobable = np.all(counts_g == counts_g[:, :, :1], axis=-1)
            h_g[is_equiprobable] = 1.0
            start, end = self._level_offsets[g]
            out[:, start:end] = h_g
        t4 = time.perf_counter_ns()

        return out, {
            "quantize_ms": (t1 - t0) / 1e6,
            "histogram_ms": (t2 - t1) / 1e6,
            "pyramid_sum_ms": (t3 - t2) / 1e6,
            "entropy_ms": (t4 - t3) / 1e6,
        }


class V2NumpyFusedEntropyVariant(_FinestGridPyramidBase):
    """V2: V1 finest-grid pyramid + single fused [N, total_cells, bins] entropy evaluation."""

    name: str = "V2_numpy_fused_entropy"

    def __init__(self) -> None:
        super().__init__()
        self._nlogn_lut: np.ndarray | None = None
        self._npix_int_vec: np.ndarray | None = None
        self._npix_f64_vec: np.ndarray | None = None
        self._log2_npix_vec: np.ndarray | None = None

    def prepare(self, image_size: int, grid_levels: Sequence[int], bins: int) -> None:
        """Precompute finest-grid geometry, max-scale n*log2(n) LUT, and per-cell N vectors."""
        self._prepare_common(image_size, grid_levels, bins, include_term_luts=False)
        g_min = min(self._grid_levels)
        max_cell_pixels = (self._image_size // g_min) ** 2
        self._nlogn_lut = build_nlogn_lut(max_cell_pixels)

        npix_int = np.empty(self._total_features, dtype=np.int64)
        for g in self._grid_levels:
            side = self._image_size // g
            start, end = self._level_offsets[g]
            npix_int[start:end] = side * side
        self._npix_int_vec = npix_int
        self._npix_f64_vec = npix_int.astype(np.float64)
        self._log2_npix_vec = np.log2(self._npix_f64_vec)

    def _evaluate_fused_entropy(self, all_counts: np.ndarray) -> np.ndarray:
        """Evaluate entropy for all regions [N, total_features, bins] in one vectorized call."""
        assert self._nlogn_lut is not None
        assert self._npix_int_vec is not None
        assert self._npix_f64_vec is not None
        assert self._log2_npix_vec is not None

        # Single LUT lookup and bin sum across all scales simultaneously: [N, total_features]
        s_vec = self._nlogn_lut[all_counts].sum(axis=-1)
        h_all = (
            self._log2_npix_vec[None, :] - (s_vec / self._npix_f64_vec[None, :])
        ) / self._log2_bins

        # Preserve analytical boundaries matching V0: constant -> 0.0, equiprobable -> 1.0
        np.maximum(h_all, 0.0, out=h_all)
        is_const = np.max(all_counts, axis=-1) == self._npix_int_vec[None, :]
        h_all[is_const] = 0.0
        is_equiprobable = np.all(all_counts == all_counts[:, :, :1], axis=-1) & (
            self._npix_int_vec[None, :] > 0
        )
        h_all[is_equiprobable] = 1.0
        return h_all

    def run(self, images_u8: np.ndarray) -> np.ndarray:
        """Compute finest-grid pyramid histograms and fuse entropy into a single array pass."""
        if self._quant_lut is None or self._image_size is None or self._nlogn_lut is None:
            raise RuntimeError(f"{self.name}.run() called before prepare()")
        arr = normalize_and_validate_images_u8(images_u8, self._image_size, self.name)
        n_images = int(arr.shape[0])
        if n_images == 0:
            return np.zeros((0, self._total_features), dtype=np.float64)

        quantized = self._quant_lut[arr]
        all_counts = np.empty((n_images, self._total_features, self._bins), dtype=np.int64)

        fine_counts = self._fine_histograms_from_quantized(quantized)
        f_start, f_end = self._level_offsets[self._g_max]
        all_counts[:, f_start:f_end, :] = fine_counts

        for g_curr, g_child, k in self._merge_schedule:
            c_start, c_end = self._level_offsets[g_child]
            t_start, t_end = self._level_offsets[g_curr]
            all_counts[:, t_start:t_end, :] = (
                all_counts[:, c_start:c_end, :]
                .reshape(n_images, g_curr, k, g_curr, k, self._bins)
                .sum(axis=(2, 4))
                .reshape(n_images, g_curr * g_curr, self._bins)
            )

        return self._evaluate_fused_entropy(all_counts)

    def measure_stages_once(self, images_u8: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Diagnostic single execution reporting V2 stage latencies in milliseconds."""
        if self._quant_lut is None or self._image_size is None or self._nlogn_lut is None:
            raise RuntimeError(f"{self.name}.measure_stages_once() called before prepare()")
        arr = normalize_and_validate_images_u8(images_u8, self._image_size, self.name)
        n_images = int(arr.shape[0])
        if n_images == 0:
            return np.zeros((0, self._total_features), dtype=np.float64), {
                "quantize_ms": 0.0,
                "histogram_ms": 0.0,
                "pyramid_sum_ms": 0.0,
                "entropy_ms": 0.0,
            }

        t0 = time.perf_counter_ns()
        quantized = self._quant_lut[arr]
        t1 = time.perf_counter_ns()

        all_counts = np.empty((n_images, self._total_features, self._bins), dtype=np.int64)
        fine_counts = self._fine_histograms_from_quantized(quantized)
        f_start, f_end = self._level_offsets[self._g_max]
        all_counts[:, f_start:f_end, :] = fine_counts
        t2 = time.perf_counter_ns()

        for g_curr, g_child, k in self._merge_schedule:
            c_start, c_end = self._level_offsets[g_child]
            t_start, t_end = self._level_offsets[g_curr]
            all_counts[:, t_start:t_end, :] = (
                all_counts[:, c_start:c_end, :]
                .reshape(n_images, g_curr, k, g_curr, k, self._bins)
                .sum(axis=(2, 4))
                .reshape(n_images, g_curr * g_curr, self._bins)
            )
        t3 = time.perf_counter_ns()

        out = self._evaluate_fused_entropy(all_counts)
        t4 = time.perf_counter_ns()

        return out, {
            "quantize_ms": (t1 - t0) / 1e6,
            "histogram_ms": (t2 - t1) / 1e6,
            "pyramid_sum_ms": (t3 - t2) / 1e6,
            "entropy_ms": (t4 - t3) / 1e6,
        }
