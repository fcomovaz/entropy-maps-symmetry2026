"""Stage-by-stage, memory, cProfile, and scalability profiling for the x94 descriptor."""

from __future__ import annotations

import cProfile
from dataclasses import dataclass
import io
import logging
import pstats
import time
from typing import Sequence

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

from src.features.asymmetry_maps import (
    compute_asymmetry_features,
    compute_asymmetry_for_grid,
    reconstruct_grids,
)
from src.features.entropy_maps import (
    EntropyPlan,
    _compute_pyramid_counts_from_fine,
    _fine_histograms_from_quantized,
    build_entropy_plan,
    build_quantization_lut,
    build_shannon_term_lut,
    compute_entropy_features,
    normalized_entropy,
    quantize_pixels,
    region_histograms,
    resolve_entropy_backend,
    validate_entropy_backend,
)
from src.features.native_entropy import (
    run_native_stage_entropy,
    run_native_stage_fine_hist,
    run_native_stage_sum,
)
from .timing import (
    LatencyStats,
    disabled_gc,
    measure_indexed_latencies,
    measure_memory_alloc,
    summarize_latencies,
    thread_limits,
)

logger = logging.getLogger("research")


@dataclass(frozen=True, slots=True)
class DescriptorAnalyticalOps:
    """Explicit analytical operation estimate for x94 stages (not directly comparable to CNN MACs)."""

    quantize_ops: int
    histogram_ops: int
    entropy_ops: int
    asymmetry_ops: int
    total_ops: int
    lut_memory_bytes: int


@dataclass(frozen=True, slots=True)
class DescriptorStageRecord:
    """Measured latency, share, memory, and analytical complexity for one descriptor stage."""

    stage: str
    category: str
    scale: str
    batch_size: int
    median_ms: float
    q25_ms: float
    q75_ms: float
    iqr_ms: float
    p95_ms: float
    mean_ms: float
    std_ms: float
    ci95_low_ms: float
    ci95_high_ms: float
    share_of_staged_sum_pct: float
    tracemalloc_peak_kb: float
    rss_delta_mb: float
    expected_complexity: str
    analytical_ops_per_sample: int


@dataclass(frozen=True, slots=True)
class DescriptorProfileReport:
    """Complete stage breakdown and overhead diagnostic for x94 extraction on real images."""

    dataset_tag: str
    image_size: int
    bins: int
    grid_levels: tuple[int, ...]
    asymmetry_scales: tuple[int, ...]
    asymmetry_kinds: tuple[str, ...]
    n_real_samples: int
    threads: int
    batch_n_size: int
    init_quant_lut_ms: float
    init_shannon_lut_ms: float
    init_entropy_plan_ms: float
    e2e_batch1_stats: LatencyStats
    e2e_batch_n_stats: LatencyStats
    staged_sum_batch1_ms: float
    python_call_gap_ms: float
    python_call_gap_pct: float
    max_numerical_diff: float
    analytical_ops: DescriptorAnalyticalOps
    stages_batch1: tuple[DescriptorStageRecord, ...]
    stages_batch_n: tuple[DescriptorStageRecord, ...]
    raw_e2e_batch1_ms: tuple[float, ...]
    cprofile_report_text: str
    requested_backend: str = "auto"
    entropy_backend: str = "numpy_baseline"


@dataclass(frozen=True, slots=True)
class DescriptorScalabilityRecord:
    """Latency measurement across varying image_size, bins, and grid_levels configurations."""

    sweep_dimension: str
    config_label: str
    image_size: int
    bins: int
    grid_levels: str
    n_features: int
    threads: int
    init_plan_ms: float
    median_ms: float
    iqr_ms: float
    p95_ms: float
    ci95_low_ms: float
    ci95_high_ms: float
    throughput_samples_s: float
    analytical_ops_per_sample: int
    expected_complexity_note: str
    entropy_backend: str = "numpy_baseline"


def _normalize_images_u8(images: np.ndarray) -> np.ndarray:
    """Validate and normalize image input into 3D uint8 array [N, H, W]."""
    arr = np.asarray(images)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Expected images shape [H, W] or [N, H, W], got {arr.shape}")
    if arr.shape[1] != arr.shape[2]:
        raise ValueError(f"Expected square images, got {arr.shape[1]}x{arr.shape[2]}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _uniform_term_lut_for_scale(plan: EntropyPlan, g: int) -> np.ndarray | None:
    """Return shared Shannon term LUT for scale g when all regions share equal pixel counts."""
    regs_g = [r for r in plan.regions if r.g == g]
    if not regs_g:
        return None
    n_pix0 = regs_g[0].n_pixels
    if all(r.n_pixels == n_pix0 for r in regs_g):
        return regs_g[0].term_lut
    return None


def _build_or_reuse_plan(
    image_size: int,
    grid_levels: Sequence[int],
    bins: int,
    plan: EntropyPlan | None,
    backend: str | None,
) -> EntropyPlan:
    """Reuse caller plan when geometry/backend match, otherwise construct resolved EntropyPlan."""
    levels_t = tuple(int(g) for g in grid_levels)
    if plan is not None and (plan.image_size, plan.grid_levels, plan.bins) == (
        image_size,
        levels_t,
        bins,
    ):
        if backend is None:
            return plan
        req = validate_entropy_backend(backend)
        if req in (plan.backend, plan.resolved_backend):
            return plan
    active_backend = "auto" if backend is None else validate_entropy_backend(backend)
    return build_entropy_plan(image_size, levels_t, bins, backend=active_backend)


def extract_x94_end_to_end(
    images: np.ndarray,
    grid_levels: Sequence[int] = (1, 2, 4, 8),
    bins: int = 16,
    asymmetry_scales: Sequence[int] = (2, 4, 8),
    asymmetry_kinds: Sequence[str] = ("lr", "tb", "rot180"),
    plan: EntropyPlan | None = None,
    backend: str | None = None,
) -> np.ndarray:
    """Extract x94 descriptors via formal src.features entry points.

    Args:
        images: Square grayscale uint8 images of shape [N, H, W] or [H, W].
        grid_levels: Spatial partition levels (e.g. [1, 2, 4, 8]).
        bins: Histogram bin count (e.g. 16).
        asymmetry_scales: Spatial scales >= 2 used for symmetry residuals.
        asymmetry_kinds: Reflection/rotation kinds ('lr', 'tb', 'rot180').
        plan: Optional precomputed EntropyPlan.
        backend: Optional backend override ('auto', 'numpy_baseline',
            'numpy_finest_pyramid', 'numpy_fused_entropy', or 'c_single_pass').

    Returns:
        Array of shape [N, sum(g^2) + len(asymmetry_scales)*len(asymmetry_kinds)], dtype float64.
    """
    arr = _normalize_images_u8(images)
    if arr.shape[0] == 0:
        n_ent = sum(g * g for g in grid_levels)
        n_asym = len(asymmetry_scales) * len(asymmetry_kinds)
        return np.zeros((0, n_ent + n_asym), dtype=np.float64)

    x85 = compute_entropy_features(
        arr,
        grid_levels=grid_levels,
        bins=bins,
        plan=plan,
        backend=backend,
    )
    e_by_scale: dict[int, np.ndarray] = {}
    offset = 0
    for g in grid_levels:
        cols = g * g
        e_by_scale[g] = x85[:, offset : offset + cols]
        offset += cols
    x9 = compute_asymmetry_features(
        e_by_scale,
        scales=asymmetry_scales,
        kinds=asymmetry_kinds,
    )
    return np.concatenate([x85, x9], axis=1)


def extract_x94_staged(
    images: np.ndarray,
    grid_levels: Sequence[int] = (1, 2, 4, 8),
    bins: int = 16,
    asymmetry_scales: Sequence[int] = (2, 4, 8),
    asymmetry_kinds: Sequence[str] = ("lr", "tb", "rot180"),
    plan: EntropyPlan | None = None,
    backend: str | None = None,
) -> np.ndarray:
    """Replicate x94 extraction step-by-step per resolved backend without altering src/features.

    Args:
        images: Square grayscale uint8 images of shape [N, H, W] or [H, W].
        grid_levels: Spatial partition levels (e.g. [1, 2, 4, 8]).
        bins: Histogram bin count.
        asymmetry_scales: Scales for spatial entropy asymmetry.
        asymmetry_kinds: Symmetry axes.
        plan: Optional precomputed EntropyPlan.
        backend: Optional backend override ('auto', 'numpy_baseline',
            'numpy_finest_pyramid', 'numpy_fused_entropy', or 'c_single_pass').

    Returns:
        Feature matrix [N, n_features], numerically equivalent (<1e-12) to `extract_x94_end_to_end`.
    """
    arr = _normalize_images_u8(images)
    if arr.shape[0] == 0:
        n_ent = sum(g * g for g in grid_levels)
        n_asym = len(asymmetry_scales) * len(asymmetry_kinds)
        return np.zeros((0, n_ent + n_asym), dtype=np.float64)

    height = int(arr.shape[1])
    plan = _build_or_reuse_plan(height, grid_levels, bins, plan=plan, backend=backend)
    n_images = int(arr.shape[0])
    resolved = plan.resolved_backend

    e_by_scale: dict[int, np.ndarray] = {}
    if resolved == "c_single_pass":
        assert plan.native_buffers is not None
        fine_hist = run_native_stage_fine_hist(plan.native_buffers, arr)
        all_hist = run_native_stage_sum(plan.native_buffers, fine_hist)
        x85 = run_native_stage_entropy(plan.native_buffers, all_hist)
        for g, start, end in plan.level_offsets:
            e_by_scale[g] = x85[:, start:end]
    elif resolved == "numpy_finest_pyramid":
        quantized = quantize_pixels(arr, bins=bins, lut=plan.quant_lut)
        fine_counts = _fine_histograms_from_quantized(
            quantized, plan.g_max, plan.cell_side, bins
        )
        counts_by_level = _compute_pyramid_counts_from_fine(fine_counts, plan)
        term_lut_map = dict(plan.term_luts_by_level)
        e_parts: list[np.ndarray] = []
        for g in grid_levels:
            counts_g = counts_by_level[g]
            terms = term_lut_map[g][counts_g]
            ent_g = np.sum(terms, axis=-1) / plan.log2_bins
            ent_g[np.all(counts_g == counts_g[:, :, :1], axis=-1)] = 1.0
            e_parts.append(ent_g)
            e_by_scale[g] = ent_g
        x85 = np.concatenate(e_parts, axis=1)
    elif resolved == "numpy_fused_entropy":
        assert plan.nlogn_lut is not None
        assert plan.npix_int_vec is not None
        assert plan.npix_f64_vec is not None
        assert plan.log2_npix_vec is not None
        quantized = quantize_pixels(arr, bins=bins, lut=plan.quant_lut)
        all_counts = np.empty((n_images, plan.total_features, bins), dtype=np.int64)
        offset_map = {g: (s, e) for g, s, e in plan.level_offsets}
        f_start, f_end = offset_map[plan.g_max]
        all_counts[:, f_start:f_end, :] = _fine_histograms_from_quantized(
            quantized, plan.g_max, plan.cell_side, bins
        )
        for g_curr, g_child, k in plan.merge_schedule:
            c_start, c_end = offset_map[g_child]
            t_start, t_end = offset_map[g_curr]
            all_counts[:, t_start:t_end, :] = (
                all_counts[:, c_start:c_end, :]
                .reshape(n_images, g_curr, k, g_curr, k, bins)
                .sum(axis=(2, 4))
                .reshape(n_images, g_curr * g_curr, bins)
            )
        s_vec = plan.nlogn_lut[all_counts].sum(axis=-1)
        x85 = (
            plan.log2_npix_vec[None, :] - (s_vec / plan.npix_f64_vec[None, :])
        ) / plan.log2_bins
        np.maximum(x85, 0.0, out=x85)
        x85[np.max(all_counts, axis=-1) == plan.npix_int_vec[None, :]] = 0.0
        x85[
            np.all(all_counts == all_counts[:, :, :1], axis=-1)
            & (plan.npix_int_vec[None, :] > 0)
        ] = 1.0
        for g, start, end in plan.level_offsets:
            e_by_scale[g] = x85[:, start:end]
    else:
        quantized = quantize_pixels(arr, bins=bins, lut=plan.quant_lut)
        e_parts = []
        for g in grid_levels:
            counts_g = region_histograms(quantized, g=g, bins=bins)
            term_lut_g = _uniform_term_lut_for_scale(plan, g)
            ent_g = normalized_entropy(counts_g, bins=bins, term_lut=term_lut_g)
            e_parts.append(ent_g)
            e_by_scale[g] = ent_g
        x85 = np.concatenate(e_parts, axis=1)

    asym_cols: list[np.ndarray] = []
    for g in asymmetry_scales:
        grids_g = reconstruct_grids(e_by_scale[g], g)
        asym_dict = compute_asymmetry_for_grid(grids_g, kinds=asymmetry_kinds)
        for kind in asymmetry_kinds:
            asym_cols.append(asym_dict[kind][:, None])
    x9 = np.concatenate(asym_cols, axis=1)

    return np.concatenate([x85, x9], axis=1)


def estimate_descriptor_ops(
    image_size: int,
    grid_levels: Sequence[int],
    bins: int,
    asymmetry_scales: Sequence[int],
    asymmetry_kinds: Sequence[str],
    backend: str = "numpy_baseline",
) -> DescriptorAnalyticalOps:
    """Compute an explicit analytical operation count and LUT memory footprint for x94.

    Marked strictly as an analytical estimate: LUT table indexing and integer histogram
    increments are memory-bound integer ops rather than neural float MACs.
    """
    levels_t = tuple(int(g) for g in grid_levels)
    bins_i = int(bins)
    img_i = int(image_size)
    resolved = resolve_entropy_backend(validate_entropy_backend(backend), img_i, levels_t, bins_i)

    n_pix = img_i * img_i
    n_levels = len(levels_t)
    quant_ops = n_pix  # 1 LUT lookup per pixel
    total_regions = sum(g * g for g in levels_t)
    ent_ops = total_regions * (3 * bins_i + 1)
    asym_regions = sum(int(g) * int(g) for g in asymmetry_scales)
    asym_ops = len(asymmetry_kinds) * asym_regions * 3

    quant_lut_bytes = 256  # uint8[256]
    if resolved == "numpy_baseline":
        hist_ops = n_levels * (2 * n_pix)
        unique_pix_counts = {(img_i // g) * (img_i // g) for g in levels_t}
        shannon_lut_bytes = sum((c + 1) * 8 for c in unique_pix_counts)
    else:
        g_max = max(levels_t)
        coarse_cells = max(0, total_regions - (g_max * g_max))
        if resolved == "c_single_pass":
            hist_ops = n_pix + (coarse_cells * bins_i)
            max_pix = (img_i // min(levels_t)) ** 2
            shannon_lut_bytes = (max_pix + 1) * 8
        elif resolved == "numpy_fused_entropy":
            hist_ops = (2 * n_pix) + (coarse_cells * bins_i)
            max_pix = (img_i // min(levels_t)) ** 2
            shannon_lut_bytes = (max_pix + 1) * 8
        else:  # numpy_finest_pyramid
            hist_ops = (2 * n_pix) + (coarse_cells * bins_i)
            unique_pix_counts = {(img_i // g) * (img_i // g) for g in levels_t}
            shannon_lut_bytes = sum((c + 1) * 8 for c in unique_pix_counts)

    total_ops = quant_ops + hist_ops + ent_ops + asym_ops
    return DescriptorAnalyticalOps(
        quantize_ops=quant_ops,
        histogram_ops=hist_ops,
        entropy_ops=ent_ops,
        asymmetry_ops=asym_ops,
        total_ops=total_ops,
        lut_memory_bytes=quant_lut_bytes + shannon_lut_bytes,
    )


def run_descriptor_cprofile(
    images: np.ndarray,
    grid_levels: Sequence[int],
    bins: int,
    asymmetry_scales: Sequence[int],
    asymmetry_kinds: Sequence[str],
    top_n: int = 30,
    n_calls: int = 100,
    backend: str = "auto",
) -> str:
    """Run Python `cProfile` on repeated single-image x94 extractions to locate call bottlenecks."""
    arr = _normalize_images_u8(images)
    plan = build_entropy_plan(int(arr.shape[1]), grid_levels, bins, backend=backend)
    n = int(arr.shape[0])

    # Warm caches once outside cProfile
    _ = extract_x94_end_to_end(
        arr[:1],
        grid_levels=grid_levels,
        bins=bins,
        asymmetry_scales=asymmetry_scales,
        asymmetry_kinds=asymmetry_kinds,
        plan=plan,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    for i in range(n_calls):
        sample = arr[i % n : (i % n) + 1]
        _ = extract_x94_end_to_end(
            sample,
            grid_levels=grid_levels,
            bins=bins,
            asymmetry_scales=asymmetry_scales,
            asymmetry_kinds=asymmetry_kinds,
            plan=plan,
        )
    profiler.disable()

    buf = io.StringIO()
    buf.write(
        f"# cProfile summary (backend={plan.resolved_backend}, {n_calls} single-image calls; NOTE: trace callback inflates micro-ops)\n"
    )
    stats_obj = pstats.Stats(profiler, stream=buf).sort_stats("cumtime")
    stats_obj.print_stats(top_n)
    return buf.getvalue()


def _measure_stages_for_batch_mode(
    images_u8: np.ndarray,
    plan: EntropyPlan,
    grid_levels: tuple[int, ...],
    bins: int,
    asym_scales: tuple[int, ...],
    asym_kinds: tuple[str, ...],
    batch_size: int,
    warmup_iters: int,
    timed_iters: int,
    disable_gc_flag: bool,
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
    ops: DescriptorAnalyticalOps,
    scaler: StandardScaler | None = None,
) -> tuple[tuple[DescriptorStageRecord, ...], LatencyStats, float]:
    """Measure per-stage and end-to-end x94 distributions for a given batch_size (1 or N)."""
    n_total = int(images_u8.shape[0])
    img_size = int(images_u8.shape[1])

    batches: list[np.ndarray] = []
    if batch_size == 1:
        for i in range(n_total):
            batches.append(np.ascontiguousarray(images_u8[i : i + 1]))
    else:
        n_batches = max(1, (n_total + batch_size - 1) // batch_size)
        for b in range(n_batches):
            idx = (np.arange(b * batch_size, (b + 1) * batch_size) % n_total).astype(np.int64)
            batches.append(np.ascontiguousarray(images_u8[idx]))
    n_batches = len(batches)

    resolved = plan.resolved_backend
    n_ent_feats = sum(g * g for g in grid_levels)
    n_total_feats = n_ent_feats + len(asym_scales) * len(asym_kinds)
    coarse_cells = max(0, n_ent_feats - plan.g_max * plan.g_max)

    stage_meta: list[tuple[str, str, str, str, int]] = []
    if resolved == "c_single_pass":
        stage_meta.extend(
            [
                (
                    "c_fine_hist",
                    "entropy",
                    f"{plan.g_max}x{plan.g_max}",
                    f"O(H*W) [C99 fused quant+hist, H=W={img_size}]",
                    2 * img_size * img_size,
                ),
                (
                    "c_pyramid_sum",
                    "entropy",
                    "coarse",
                    f"O(C_coarse*B) [C99 child sum, C_coarse={coarse_cells}]",
                    coarse_cells * bins,
                ),
                (
                    "c_entropy_lut",
                    "entropy",
                    "all",
                    f"O(F_e*B) [C99 nlogn LUT, F_e={n_ent_feats}]",
                    n_ent_feats * (3 * bins + 1),
                ),
            ]
        )
    elif resolved == "numpy_finest_pyramid":
        stage_meta.extend(
            [
                ("quantize_pixels", "entropy", "all", f"O(H*W) [H=W={img_size}]", ops.quantize_ops),
                (
                    "fine_grid_histograms",
                    "entropy",
                    f"{plan.g_max}x{plan.g_max}",
                    f"O(H*W) [1 global bincount, H=W={img_size}]",
                    2 * img_size * img_size,
                ),
                (
                    "pyramid_sum",
                    "entropy",
                    "coarse",
                    f"O(C_coarse*B) [child sum, C_coarse={coarse_cells}]",
                    coarse_cells * bins,
                ),
            ]
        )
        for g in grid_levels:
            stage_meta.append(
                (
                    f"normalized_entropy_g{g}",
                    "entropy",
                    f"{g}x{g}",
                    f"O(g^2*B) [{g*g}x{bins}]",
                    g * g * (3 * bins + 1),
                )
            )
        stage_meta.append(
            ("concat_x85", "assembly", "x85", f"O(F_e) [F_e={n_ent_feats}]", n_ent_feats)
        )
    elif resolved == "numpy_fused_entropy":
        stage_meta.extend(
            [
                ("quantize_pixels", "entropy", "all", f"O(H*W) [H=W={img_size}]", ops.quantize_ops),
                (
                    "fine_grid_histograms",
                    "entropy",
                    f"{plan.g_max}x{plan.g_max}",
                    f"O(H*W) [1 global bincount, H=W={img_size}]",
                    2 * img_size * img_size,
                ),
                (
                    "pyramid_sum",
                    "entropy",
                    "coarse",
                    f"O(C_coarse*B) [child sum, C_coarse={coarse_cells}]",
                    coarse_cells * bins,
                ),
                (
                    "fused_entropy",
                    "entropy",
                    "all",
                    f"O(F_e*B) [fused nlogn LUT, F_e={n_ent_feats}]",
                    n_ent_feats * (3 * bins + 1),
                ),
            ]
        )
    else:
        stage_meta.append(
            ("quantize_pixels", "entropy", "all", f"O(H*W) [H=W={img_size}]", ops.quantize_ops)
        )
        for g in grid_levels:
            stage_meta.append(
                (
                    f"region_histograms_g{g}",
                    "entropy",
                    f"{g}x{g}",
                    f"O(H*W) [2*H*W={2*img_size*img_size}]",
                    2 * img_size * img_size,
                )
            )
        for g in grid_levels:
            stage_meta.append(
                (
                    f"normalized_entropy_g{g}",
                    "entropy",
                    f"{g}x{g}",
                    f"O(g^2*B) [{g*g}x{bins}]",
                    g * g * (3 * bins + 1),
                )
            )
        stage_meta.append(
            ("concat_x85", "assembly", "x85", f"O(F_e) [F_e={n_ent_feats}]", n_ent_feats)
        )

    for g in asym_scales:
        stage_meta.append(
            (
                f"asymmetry_g{g}",
                "asymmetry",
                f"{g}x{g}",
                f"O(g^2*K) [{g*g}x{len(asym_kinds)}]",
                g * g * len(asym_kinds) * 3,
            )
        )
    stage_meta.append(
        ("concat_x94", "assembly", "x94", f"O(F_total) [F={n_total_feats}]", n_total_feats)
    )
    if scaler is not None:
        stage_meta.append(
            ("standard_scaler", "preprocess", "x94", f"O(F_total) [LR/kNN, F={n_total_feats}]", n_total_feats * 2)
        )

    stage_raw_ms: dict[str, list[float]] = {m[0]: [] for m in stage_meta}

    # Precomputed / reusable per-backend stage buffers outside timed iterations
    term_luts = {g: _uniform_term_lut_for_scale(plan, g) for g in grid_levels}
    term_lut_map = dict(plan.term_luts_by_level)
    offset_map = {g: (s, e) for g, s, e in plan.level_offsets}

    c_fine_buf = (
        np.empty((batch_size, plan.g_max * plan.g_max, bins), dtype=np.uint32)
        if resolved == "c_single_pass"
        else None
    )
    c_all_buf = (
        np.empty((batch_size, plan.total_features, bins), dtype=np.uint32)
        if resolved == "c_single_pass"
        else None
    )
    c_x85_buf = (
        np.empty((batch_size, plan.total_features), dtype=np.float64)
        if resolved == "c_single_pass"
        else None
    )

    def _run_staged_record(batch: np.ndarray, record: bool) -> np.ndarray:
        e_by_scale: dict[int, np.ndarray] = {}

        if resolved == "c_single_pass":
            assert plan.native_buffers is not None
            assert c_fine_buf is not None and c_all_buf is not None and c_x85_buf is not None
            t0 = time.perf_counter_ns()
            _ = run_native_stage_fine_hist(plan.native_buffers, batch, out_fine_hist=c_fine_buf)
            t1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["c_fine_hist"].append((t1 - t0) / 1e6)

            ts0 = time.perf_counter_ns()
            _ = run_native_stage_sum(plan.native_buffers, c_fine_buf, out_all_hist=c_all_buf)
            ts1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["c_pyramid_sum"].append((ts1 - ts0) / 1e6)

            te0 = time.perf_counter_ns()
            x85 = run_native_stage_entropy(plan.native_buffers, c_all_buf, out_features=c_x85_buf)
            te1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["c_entropy_lut"].append((te1 - te0) / 1e6)

            for g in asym_scales:
                s_idx, e_idx = offset_map[g]
                e_by_scale[g] = x85[:, s_idx:e_idx]
        elif resolved == "numpy_finest_pyramid":
            t0 = time.perf_counter_ns()
            q = quantize_pixels(batch, bins=bins, lut=plan.quant_lut)
            t1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["quantize_pixels"].append((t1 - t0) / 1e6)

            tfh0 = time.perf_counter_ns()
            fine_counts = _fine_histograms_from_quantized(q, plan.g_max, plan.cell_side, bins)
            tfh1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["fine_grid_histograms"].append((tfh1 - tfh0) / 1e6)

            tp0 = time.perf_counter_ns()
            counts_by_level = _compute_pyramid_counts_from_fine(fine_counts, plan)
            tp1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["pyramid_sum"].append((tp1 - tp0) / 1e6)

            e_parts: list[np.ndarray] = []
            for g in grid_levels:
                te0 = time.perf_counter_ns()
                counts_g = counts_by_level[g]
                terms = term_lut_map[g][counts_g]
                ent_g = np.sum(terms, axis=-1) / plan.log2_bins
                ent_g[np.all(counts_g == counts_g[:, :, :1], axis=-1)] = 1.0
                te1 = time.perf_counter_ns()
                if record:
                    stage_raw_ms[f"normalized_entropy_g{g}"].append((te1 - te0) / 1e6)
                e_parts.append(ent_g)
                e_by_scale[g] = ent_g

            tc0 = time.perf_counter_ns()
            x85 = np.concatenate(e_parts, axis=1)
            tc1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["concat_x85"].append((tc1 - tc0) / 1e6)
        elif resolved == "numpy_fused_entropy":
            assert plan.nlogn_lut is not None
            assert plan.npix_int_vec is not None
            assert plan.npix_f64_vec is not None
            assert plan.log2_npix_vec is not None
            t0 = time.perf_counter_ns()
            q = quantize_pixels(batch, bins=bins, lut=plan.quant_lut)
            t1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["quantize_pixels"].append((t1 - t0) / 1e6)

            tfh0 = time.perf_counter_ns()
            fine_counts = _fine_histograms_from_quantized(q, plan.g_max, plan.cell_side, bins)
            tfh1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["fine_grid_histograms"].append((tfh1 - tfh0) / 1e6)

            tp0 = time.perf_counter_ns()
            n_b_cur = int(batch.shape[0])
            all_counts = np.empty((n_b_cur, plan.total_features, bins), dtype=np.int64)
            f_s, f_e = offset_map[plan.g_max]
            all_counts[:, f_s:f_e, :] = fine_counts
            for g_curr, g_child, k in plan.merge_schedule:
                cs, ce = offset_map[g_child]
                ts, te = offset_map[g_curr]
                all_counts[:, ts:te, :] = (
                    all_counts[:, cs:ce, :]
                    .reshape(n_b_cur, g_curr, k, g_curr, k, bins)
                    .sum(axis=(2, 4))
                    .reshape(n_b_cur, g_curr * g_curr, bins)
                )
            tp1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["pyramid_sum"].append((tp1 - tp0) / 1e6)

            te0 = time.perf_counter_ns()
            s_vec = plan.nlogn_lut[all_counts].sum(axis=-1)
            x85 = (
                plan.log2_npix_vec[None, :] - (s_vec / plan.npix_f64_vec[None, :])
            ) / plan.log2_bins
            np.maximum(x85, 0.0, out=x85)
            x85[np.max(all_counts, axis=-1) == plan.npix_int_vec[None, :]] = 0.0
            x85[
                np.all(all_counts == all_counts[:, :, :1], axis=-1)
                & (plan.npix_int_vec[None, :] > 0)
            ] = 1.0
            te1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["fused_entropy"].append((te1 - te0) / 1e6)

            for g in asym_scales:
                s_idx, e_idx = offset_map[g]
                e_by_scale[g] = x85[:, s_idx:e_idx]
        else:
            t0 = time.perf_counter_ns()
            q = quantize_pixels(batch, bins=bins, lut=plan.quant_lut)
            t1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["quantize_pixels"].append((t1 - t0) / 1e6)

            e_parts = []
            for g in grid_levels:
                th0 = time.perf_counter_ns()
                counts_g = region_histograms(q, g=g, bins=bins)
                th1 = time.perf_counter_ns()
                if record:
                    stage_raw_ms[f"region_histograms_g{g}"].append((th1 - th0) / 1e6)

                te0 = time.perf_counter_ns()
                ent_g = normalized_entropy(counts_g, bins=bins, term_lut=term_luts[g])
                te1 = time.perf_counter_ns()
                if record:
                    stage_raw_ms[f"normalized_entropy_g{g}"].append((te1 - te0) / 1e6)

                e_parts.append(ent_g)
                e_by_scale[g] = ent_g

            tc0 = time.perf_counter_ns()
            x85 = np.concatenate(e_parts, axis=1)
            tc1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["concat_x85"].append((tc1 - tc0) / 1e6)

        asym_cols: list[np.ndarray] = []
        for g in asym_scales:
            ta0 = time.perf_counter_ns()
            grids_g = reconstruct_grids(e_by_scale[g], g)
            asym_dict = compute_asymmetry_for_grid(grids_g, kinds=asym_kinds)
            for k in asym_kinds:
                asym_cols.append(asym_dict[k][:, None])
            ta1 = time.perf_counter_ns()
            if record:
                stage_raw_ms[f"asymmetry_g{g}"].append((ta1 - ta0) / 1e6)

        tf0 = time.perf_counter_ns()
        x9 = np.concatenate(asym_cols, axis=1)
        x94 = np.concatenate([x85, x9], axis=1)
        tf1 = time.perf_counter_ns()
        if record:
            stage_raw_ms["concat_x94"].append((tf1 - tf0) / 1e6)

        if scaler is not None:
            ts0 = time.perf_counter_ns()
            _ = scaler.transform(x94)
            ts1 = time.perf_counter_ns()
            if record:
                stage_raw_ms["standard_scaler"].append((ts1 - ts0) / 1e6)

        return x94

    # Stage timing loop
    with disabled_gc(enabled=disable_gc_flag):
        for i in range(warmup_iters):
            _ = _run_staged_record(batches[i % n_batches], record=False)
        for i in range(timed_iters):
            _ = _run_staged_record(batches[i % n_batches], record=True)

    # Formal E2E timing loop
    e2e_raw_ms = measure_indexed_latencies(
        fn=lambda idx: extract_x94_end_to_end(
            batches[idx % n_batches],
            grid_levels=grid_levels,
            bins=bins,
            asymmetry_scales=asym_scales,
            asymmetry_kinds=asym_kinds,
            plan=plan,
        ),
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        disable_gc=disable_gc_flag,
    )
    e2e_stats = summarize_latencies(
        e2e_raw_ms,
        per_sample_divisor=batch_size,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )

    # Single-pass memory attribution per stage using sample batch 0
    mem_batch = batches[0]
    stage_mem: dict[str, tuple[float, float]] = {}
    e_by_scale_mem: dict[int, np.ndarray] = {}

    if resolved == "c_single_pass":
        assert plan.native_buffers is not None
        assert c_fine_buf is not None and c_all_buf is not None and c_x85_buf is not None
        fine_m, snap_cf = measure_memory_alloc(
            lambda: run_native_stage_fine_hist(plan.native_buffers, mem_batch, out_fine_hist=c_fine_buf)
        )
        stage_mem["c_fine_hist"] = (snap_cf.tracemalloc_peak_kb, snap_cf.rss_delta_mb)
        all_m, snap_cp = measure_memory_alloc(
            lambda: run_native_stage_sum(plan.native_buffers, fine_m, out_all_hist=c_all_buf)
        )
        stage_mem["c_pyramid_sum"] = (snap_cp.tracemalloc_peak_kb, snap_cp.rss_delta_mb)
        x85_mem, snap_ce = measure_memory_alloc(
            lambda: run_native_stage_entropy(plan.native_buffers, all_m, out_features=c_x85_buf)
        )
        stage_mem["c_entropy_lut"] = (snap_ce.tracemalloc_peak_kb, snap_ce.rss_delta_mb)
        for g in asym_scales:
            s_idx, e_idx = offset_map[g]
            e_by_scale_mem[g] = x85_mem[:, s_idx:e_idx]
    elif resolved == "numpy_finest_pyramid":
        q_mem, snap_q = measure_memory_alloc(
            lambda: quantize_pixels(mem_batch, bins=bins, lut=plan.quant_lut)
        )
        stage_mem["quantize_pixels"] = (snap_q.tracemalloc_peak_kb, snap_q.rss_delta_mb)
        fine_m, snap_fh = measure_memory_alloc(
            lambda: _fine_histograms_from_quantized(q_mem, plan.g_max, plan.cell_side, bins)
        )
        stage_mem["fine_grid_histograms"] = (snap_fh.tracemalloc_peak_kb, snap_fh.rss_delta_mb)
        c_map_m, snap_ps = measure_memory_alloc(
            lambda: _compute_pyramid_counts_from_fine(fine_m, plan)
        )
        stage_mem["pyramid_sum"] = (snap_ps.tracemalloc_peak_kb, snap_ps.rss_delta_mb)
        e_parts_mem: list[np.ndarray] = []
        for g in grid_levels:
            def _ent_finest(gg: int = g) -> np.ndarray:
                cg = c_map_m[gg]
                ent = np.sum(term_lut_map[gg][cg], axis=-1) / plan.log2_bins
                ent[np.all(cg == cg[:, :, :1], axis=-1)] = 1.0
                return ent

            ent_m, snap_e = measure_memory_alloc(_ent_finest)
            stage_mem[f"normalized_entropy_g{g}"] = (snap_e.tracemalloc_peak_kb, snap_e.rss_delta_mb)
            e_parts_mem.append(ent_m)
            e_by_scale_mem[g] = ent_m
        x85_mem, snap_c85 = measure_memory_alloc(lambda: np.concatenate(e_parts_mem, axis=1))
        stage_mem["concat_x85"] = (snap_c85.tracemalloc_peak_kb, snap_c85.rss_delta_mb)
    elif resolved == "numpy_fused_entropy":
        assert plan.nlogn_lut is not None
        assert plan.npix_int_vec is not None
        assert plan.npix_f64_vec is not None
        assert plan.log2_npix_vec is not None
        q_mem, snap_q = measure_memory_alloc(
            lambda: quantize_pixels(mem_batch, bins=bins, lut=plan.quant_lut)
        )
        stage_mem["quantize_pixels"] = (snap_q.tracemalloc_peak_kb, snap_q.rss_delta_mb)
        fine_m, snap_fh = measure_memory_alloc(
            lambda: _fine_histograms_from_quantized(q_mem, plan.g_max, plan.cell_side, bins)
        )
        stage_mem["fine_grid_histograms"] = (snap_fh.tracemalloc_peak_kb, snap_fh.rss_delta_mb)

        def _fused_pyr_mem() -> np.ndarray:
            nb = int(mem_batch.shape[0])
            ac = np.empty((nb, plan.total_features, bins), dtype=np.int64)
            fs, fe = offset_map[plan.g_max]
            ac[:, fs:fe, :] = fine_m
            for g_curr, g_child, k in plan.merge_schedule:
                cs, ce = offset_map[g_child]
                ts, te = offset_map[g_curr]
                ac[:, ts:te, :] = (
                    ac[:, cs:ce, :]
                    .reshape(nb, g_curr, k, g_curr, k, bins)
                    .sum(axis=(2, 4))
                    .reshape(nb, g_curr * g_curr, bins)
                )
            return ac

        all_m, snap_ps = measure_memory_alloc(_fused_pyr_mem)
        stage_mem["pyramid_sum"] = (snap_ps.tracemalloc_peak_kb, snap_ps.rss_delta_mb)

        def _fused_ent_mem() -> np.ndarray:
            assert plan.nlogn_lut is not None
            assert plan.npix_int_vec is not None
            assert plan.npix_f64_vec is not None
            assert plan.log2_npix_vec is not None
            sv = plan.nlogn_lut[all_m].sum(axis=-1)
            ha = (plan.log2_npix_vec[None, :] - (sv / plan.npix_f64_vec[None, :])) / plan.log2_bins
            np.maximum(ha, 0.0, out=ha)
            ha[np.max(all_m, axis=-1) == plan.npix_int_vec[None, :]] = 0.0
            ha[
                np.all(all_m == all_m[:, :, :1], axis=-1) & (plan.npix_int_vec[None, :] > 0)
            ] = 1.0
            return ha

        x85_mem, snap_fe = measure_memory_alloc(_fused_ent_mem)
        stage_mem["fused_entropy"] = (snap_fe.tracemalloc_peak_kb, snap_fe.rss_delta_mb)
        for g in asym_scales:
            s_idx, e_idx = offset_map[g]
            e_by_scale_mem[g] = x85_mem[:, s_idx:e_idx]
    else:
        q_mem, snap_q = measure_memory_alloc(
            lambda: quantize_pixels(mem_batch, bins=bins, lut=plan.quant_lut)
        )
        stage_mem["quantize_pixels"] = (snap_q.tracemalloc_peak_kb, snap_q.rss_delta_mb)
        e_parts_mem = []
        for g in grid_levels:
            counts_m, snap_h = measure_memory_alloc(lambda gg=g: region_histograms(q_mem, g=gg, bins=bins))
            stage_mem[f"region_histograms_g{g}"] = (snap_h.tracemalloc_peak_kb, snap_h.rss_delta_mb)
            ent_m, snap_e = measure_memory_alloc(
                lambda cc=counts_m, gg=g: normalized_entropy(cc, bins=bins, term_lut=term_luts[gg])
            )
            stage_mem[f"normalized_entropy_g{g}"] = (snap_e.tracemalloc_peak_kb, snap_e.rss_delta_mb)
            e_parts_mem.append(ent_m)
            e_by_scale_mem[g] = ent_m

        x85_mem, snap_c85 = measure_memory_alloc(lambda: np.concatenate(e_parts_mem, axis=1))
        stage_mem["concat_x85"] = (snap_c85.tracemalloc_peak_kb, snap_c85.rss_delta_mb)

    asym_cols_mem: list[np.ndarray] = []
    for g in asym_scales:
        def _asym_step(gg: int = g) -> list[np.ndarray]:
            gd = reconstruct_grids(e_by_scale_mem[gg], gg)
            ad = compute_asymmetry_for_grid(gd, kinds=asym_kinds)
            return [ad[k][:, None] for k in asym_kinds]

        cols_m, snap_a = measure_memory_alloc(_asym_step)
        stage_mem[f"asymmetry_g{g}"] = (snap_a.tracemalloc_peak_kb, snap_a.rss_delta_mb)
        asym_cols_mem.extend(cols_m)

    x94_mem, snap_c94 = measure_memory_alloc(
        lambda: np.concatenate([x85_mem, np.concatenate(asym_cols_mem, axis=1)], axis=1)
    )
    stage_mem["concat_x94"] = (snap_c94.tracemalloc_peak_kb, snap_c94.rss_delta_mb)
    if scaler is not None:
        _, snap_sc = measure_memory_alloc(lambda: scaler.transform(x94_mem))
        stage_mem["standard_scaler"] = (snap_sc.tracemalloc_peak_kb, snap_sc.rss_delta_mb)

    # Summarize all stage latencies (per-sample amortized by batch_size)
    summarized: list[tuple[tuple[str, str, str, str, int], LatencyStats]] = []
    core_staged_sum_ms = 0.0
    for meta in stage_meta:
        s_name = meta[0]
        st_stats = summarize_latencies(
            stage_raw_ms[s_name],
            per_sample_divisor=batch_size,
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            seed=seed,
        )
        summarized.append((meta, st_stats))
        if s_name != "standard_scaler":
            core_staged_sum_ms += st_stats.median_ms

    records: list[DescriptorStageRecord] = []
    for (s_name, cat, sc_lbl, comp, op_cnt), st_stats in summarized:
        share = (
            (st_stats.median_ms / core_staged_sum_ms * 100.0)
            if (core_staged_sum_ms > 0.0 and s_name != "standard_scaler")
            else 0.0
        )
        tm_kb, rss_d = stage_mem[s_name]
        records.append(
            DescriptorStageRecord(
                stage=s_name,
                category=cat,
                scale=sc_lbl,
                batch_size=batch_size,
                median_ms=st_stats.median_ms,
                q25_ms=st_stats.q25_ms,
                q75_ms=st_stats.q75_ms,
                iqr_ms=st_stats.iqr_ms,
                p95_ms=st_stats.p95_ms,
                mean_ms=st_stats.mean_ms,
                std_ms=st_stats.std_ms,
                ci95_low_ms=st_stats.ci95_low_ms,
                ci95_high_ms=st_stats.ci95_high_ms,
                share_of_staged_sum_pct=share,
                tracemalloc_peak_kb=tm_kb,
                rss_delta_mb=rss_d,
                expected_complexity=comp,
                analytical_ops_per_sample=op_cnt,
            )
        )

    return tuple(records), e2e_stats, core_staged_sum_ms


def profile_descriptor_stages(
    images_u8: np.ndarray,
    dataset_tag: str,
    grid_levels: Sequence[int] = (1, 2, 4, 8),
    bins: int = 16,
    asymmetry_scales: Sequence[int] = (2, 4, 8),
    asymmetry_kinds: Sequence[str] = ("lr", "tb", "rot180"),
    threads: int = 1,
    batch_n_size: int = 32,
    warmup_iters: int = 20,
    timed_iters: int = 200,
    batch_timed_iters: int = 50,
    disable_gc_flag: bool = True,
    bootstrap_resamples: int = 1000,
    confidence_level: float = 0.95,
    cprofile_top_n: int = 30,
    seed: int = 42,
    scaler: StandardScaler | None = None,
    backend: str = "auto",
) -> DescriptorProfileReport:
    """Profile all x94 descriptor stages, initialization costs, memory, Python gap, and cProfile."""
    arr = _normalize_images_u8(images_u8)
    image_size = int(arr.shape[1])
    levels_t = tuple(int(g) for g in grid_levels)
    asym_s_t = tuple(int(g) for g in asymmetry_scales)
    asym_k_t = tuple(str(k) for k in asymmetry_kinds)
    req_backend = validate_entropy_backend(backend)

    with thread_limits(threads):
        # 1. Measure one-time plan/LUT precomputations (clearing LRU cache first)
        build_shannon_term_lut.cache_clear()
        t_q0 = time.perf_counter_ns()
        _ = build_quantization_lut(bins)
        init_quant_ms = (time.perf_counter_ns() - t_q0) / 1e6

        t_s0 = time.perf_counter_ns()
        _ = build_shannon_term_lut((image_size // 2) * (image_size // 2))
        init_shannon_ms = (time.perf_counter_ns() - t_s0) / 1e6

        build_shannon_term_lut.cache_clear()
        t_p0 = time.perf_counter_ns()
        plan = build_entropy_plan(image_size, levels_t, bins, backend=req_backend)
        init_plan_ms = (time.perf_counter_ns() - t_p0) / 1e6

        # 2. Verify exact numerical equivalence between staged execution and end-to-end
        check_slice = arr[: min(16, arr.shape[0])]
        out_staged = extract_x94_staged(
            check_slice,
            grid_levels=levels_t,
            bins=bins,
            asymmetry_scales=asym_s_t,
            asymmetry_kinds=asym_k_t,
            plan=plan,
        )
        out_e2e = extract_x94_end_to_end(
            check_slice,
            grid_levels=levels_t,
            bins=bins,
            asymmetry_scales=asym_s_t,
            asymmetry_kinds=asym_k_t,
            plan=plan,
        )
        max_diff = float(np.max(np.abs(out_staged - out_e2e)))
        if max_diff > 1e-12:
            raise AssertionError(
                f"Staged x94 output diverged from formal end-to-end by {max_diff:.3e}"
            )

        ops = estimate_descriptor_ops(
            image_size=image_size,
            grid_levels=levels_t,
            bins=bins,
            asymmetry_scales=asym_s_t,
            asymmetry_kinds=asym_k_t,
            backend=plan.resolved_backend,
        )

        # 3. Batch=1 detailed stage profile
        stages_b1, e2e_b1_stats, staged_sum_b1 = _measure_stages_for_batch_mode(
            images_u8=arr,
            plan=plan,
            grid_levels=levels_t,
            bins=bins,
            asym_scales=asym_s_t,
            asym_kinds=asym_k_t,
            batch_size=1,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            disable_gc_flag=disable_gc_flag,
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            seed=seed,
            ops=ops,
            scaler=scaler,
        )

        # Keep raw batch=1 E2E samples for Mann-Whitney comparisons
        n_arr = int(arr.shape[0])
        raw_e2e_b1 = measure_indexed_latencies(
            fn=lambda idx: extract_x94_end_to_end(
                arr[idx % n_arr : (idx % n_arr) + 1],
                grid_levels=levels_t,
                bins=bins,
                asymmetry_scales=asym_s_t,
                asymmetry_kinds=asym_k_t,
                plan=plan,
            ),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            disable_gc=disable_gc_flag,
        )

        # 4. Batch=N amortized stage profile
        stages_bn, e2e_bn_stats, _ = _measure_stages_for_batch_mode(
            images_u8=arr,
            plan=plan,
            grid_levels=levels_t,
            bins=bins,
            asym_scales=asym_s_t,
            asym_kinds=asym_k_t,
            batch_size=batch_n_size,
            warmup_iters=warmup_iters,
            timed_iters=batch_timed_iters,
            disable_gc_flag=disable_gc_flag,
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            seed=seed,
            ops=ops,
            scaler=scaler,
        )

        # 5. cProfile diagnostic dump
        cprof_text = run_descriptor_cprofile(
            images=arr,
            grid_levels=levels_t,
            bins=bins,
            asymmetry_scales=asym_s_t,
            asymmetry_kinds=asym_k_t,
            top_n=cprofile_top_n,
            n_calls=min(100, timed_iters),
            backend=plan.resolved_backend,
        )

    gap_ms = staged_sum_b1 - e2e_b1_stats.median_ms
    gap_pct = (gap_ms / staged_sum_b1 * 100.0) if staged_sum_b1 > 0.0 else 0.0

    return DescriptorProfileReport(
        dataset_tag=dataset_tag,
        image_size=image_size,
        bins=bins,
        grid_levels=levels_t,
        asymmetry_scales=asym_s_t,
        asymmetry_kinds=asym_k_t,
        n_real_samples=n_arr,
        threads=threads,
        batch_n_size=batch_n_size,
        init_quant_lut_ms=init_quant_ms,
        init_shannon_lut_ms=init_shannon_ms,
        init_entropy_plan_ms=init_plan_ms,
        e2e_batch1_stats=e2e_b1_stats,
        e2e_batch_n_stats=e2e_bn_stats,
        staged_sum_batch1_ms=staged_sum_b1,
        python_call_gap_ms=gap_ms,
        python_call_gap_pct=gap_pct,
        max_numerical_diff=max_diff,
        analytical_ops=ops,
        stages_batch1=stages_b1,
        stages_batch_n=stages_bn,
        raw_e2e_batch1_ms=tuple(float(v) for v in raw_e2e_b1),
        cprofile_report_text=cprof_text,
        requested_backend=plan.backend,
        entropy_backend=plan.resolved_backend,
    )


def _resize_u8_batch(images_u8: np.ndarray, target_size: int) -> np.ndarray:
    """Resize [N, H, W] uint8 batch deterministically outside timed loops."""
    if images_u8.shape[1] == target_size and images_u8.shape[2] == target_size:
        return np.ascontiguousarray(images_u8)
    resized = np.empty((images_u8.shape[0], target_size, target_size), dtype=np.uint8)
    for i in range(images_u8.shape[0]):
        with Image.fromarray(images_u8[i], mode="L") as im:
            out_im = im.resize((target_size, target_size), resample=Image.Resampling.BILINEAR)
            resized[i] = np.asarray(out_im, dtype=np.uint8)
    return resized


def profile_descriptor_scalability(
    images_u8: np.ndarray,
    base_image_size: int,
    base_bins: int,
    base_grid_levels: Sequence[int],
    asymmetry_kinds: Sequence[str],
    image_sizes: Sequence[int],
    bins_list: Sequence[int],
    grid_levels_list: Sequence[Sequence[int]],
    threads: int = 1,
    warmup_iters: int = 15,
    timed_iters: int = 100,
    disable_gc_flag: bool = True,
    bootstrap_resamples: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
    backend: str = "auto",
) -> tuple[DescriptorScalabilityRecord, ...]:
    """Sweep image_size, bins, and grid_levels separately on real samples (CPU, batch=1)."""
    arr_base = _normalize_images_u8(images_u8)
    req_backend = validate_entropy_backend(backend)
    records: list[DescriptorScalabilityRecord] = []

    def _bench_cfg(
        sweep_dim: str,
        label: str,
        imgs: np.ndarray,
        img_size: int,
        b_cnt: int,
        g_levels: tuple[int, ...],
        note: str,
    ) -> DescriptorScalabilityRecord:
        asym_s = tuple(g for g in g_levels if g >= 2)
        if not asym_s:
            raise ValueError(f"Grid levels {g_levels} require at least one scale >= 2 for asymmetry")
        t_p0 = time.perf_counter_ns()
        plan = build_entropy_plan(img_size, g_levels, b_cnt, backend=req_backend)
        init_ms = (time.perf_counter_ns() - t_p0) / 1e6

        n_img = int(imgs.shape[0])
        lat_ms = measure_indexed_latencies(
            fn=lambda idx: extract_x94_end_to_end(
                imgs[idx % n_img : (idx % n_img) + 1],
                grid_levels=g_levels,
                bins=b_cnt,
                asymmetry_scales=asym_s,
                asymmetry_kinds=asymmetry_kinds,
                plan=plan,
            ),
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            disable_gc=disable_gc_flag,
        )
        st = summarize_latencies(
            lat_ms,
            per_sample_divisor=1,
            bootstrap_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
            seed=seed,
        )
        ops = estimate_descriptor_ops(
            img_size,
            g_levels,
            b_cnt,
            asym_s,
            asymmetry_kinds,
            backend=plan.resolved_backend,
        )
        n_feats = sum(g * g for g in g_levels) + len(asym_s) * len(asymmetry_kinds)
        return DescriptorScalabilityRecord(
            sweep_dimension=sweep_dim,
            config_label=label,
            image_size=img_size,
            bins=b_cnt,
            grid_levels="|".join(str(g) for g in g_levels),
            n_features=n_feats,
            threads=threads,
            init_plan_ms=init_ms,
            median_ms=st.median_ms,
            iqr_ms=st.iqr_ms,
            p95_ms=st.p95_ms,
            ci95_low_ms=st.ci95_low_ms,
            ci95_high_ms=st.ci95_high_ms,
            throughput_samples_s=st.throughput_samples_s,
            analytical_ops_per_sample=ops.total_ops,
            expected_complexity_note=note,
            entropy_backend=plan.resolved_backend,
        )

    base_levels_t = tuple(int(g) for g in base_grid_levels)
    base_imgs = _resize_u8_batch(arr_base, base_image_size)

    with thread_limits(threads):
        for sz in image_sizes:
            imgs_sz = _resize_u8_batch(arr_base, int(sz))
            records.append(
                _bench_cfg(
                    "image_size",
                    f"{sz}x{sz}",
                    imgs_sz,
                    int(sz),
                    base_bins,
                    base_levels_t,
                    f"Expected O(H*W) per level: {(sz*sz)/(base_image_size*base_image_size):.2f}x pixels vs 128",
                )
            )

        for b_cnt in bins_list:
            records.append(
                _bench_cfg(
                    "bins",
                    f"B={b_cnt}",
                    base_imgs,
                    base_image_size,
                    int(b_cnt),
                    base_levels_t,
                    f"Expected O(g^2*B) LUT terms + bincount bins={b_cnt}",
                )
            )

        for g_lvl in grid_levels_list:
            g_t = tuple(int(g) for g in g_lvl)
            records.append(
                _bench_cfg(
                    "grid_levels",
                    "levels=" + "|".join(str(g) for g in g_t),
                    base_imgs,
                    base_image_size,
                    base_bins,
                    g_t,
                    f"Expected O(|levels|*H*W + sum(g^2)*B), regions={sum(g*g for g in g_t)}",
                )
            )

    return tuple(records)
