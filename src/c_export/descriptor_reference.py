"""Bit-exact Python mirror of C99 `c_imp/runtime/cimp_descriptor.c` in Q15.

Executes the exact integer operations, LUT lookups, rounding shifts, and asymmetry reductions
so C Q15 execution can be checked for bit-identity against Python. Also provides float64 oracle
x94 helper reusing `src.features`.
"""

from __future__ import annotations

import math

import numpy as np

from src.features.asymmetry_maps import compute_asymmetry_for_grid, reconstruct_grids
from src.features.entropy_maps import build_entropy_plan, compute_entropy_features

from .config import CImpConfig
from .fixed_point import clz32, round_shift_right_i64
from .luts import ShannonLutBundle, build_bin_quant_lut, build_shannon_lut


def extract_x94_f64_oracle(images: np.ndarray, cfg: CImpConfig) -> np.ndarray:
    """Extract reference float64 x94 descriptors via formal `src/features` pipeline."""
    spec = cfg.descriptor_spec
    plan = build_entropy_plan(spec.image_size, spec.grid_levels, spec.bins, backend="numpy_baseline")
    arr = np.asarray(images, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    x85 = compute_entropy_features(arr, spec.grid_levels, spec.bins, plan=plan)

    offsets: dict[int, tuple[int, int]] = {}
    cur = 0
    for g in spec.grid_levels:
        offsets[g] = (cur, cur + g * g)
        cur += g * g

    n = arr.shape[0]
    x9 = np.empty((n, spec.n_x9), dtype=np.float64)
    col = 0
    for g in spec.asym_scales:
        s, e = offsets[g]
        grids = reconstruct_grids(x85[:, s:e], g)
        asym = compute_asymmetry_for_grid(grids, kinds=spec.asym_kinds)
        for k in spec.asym_kinds:
            x9[:, col] = asym[k]
            col += 1
    return np.concatenate([x85, x9], axis=1)


def _eval_cell_q15(
    cnt: np.ndarray,
    lvl_idx: int,
    bins: int,
    bundle: ShannonLutBundle,
    log2_b_q15: int,
) -> int:
    """Bit-exact Python equivalent of `cimp_eval_single_cell_q15`."""
    n_pix = int(bundle.level_npix[lvl_idx])
    if n_pix <= 0:
        return 0
    first = int(cnt[0])
    if all(int(cnt[b]) == first for b in range(1, bins)):
        return min(32767, (1 << bundle.feature_frac_bits) - 1)

    q_max = min(32767, (1 << bundle.feature_frac_bits) - 1)
    if bundle.mode == "nlogn_full":
        sum_nlogn = sum(int(bundle.nlogn_table[int(cnt[b])]) for b in range(bins) if cnt[b] > 0)
        sum_q15 = sum_nlogn << (15 - bundle.nlogn_frac_bits)
        total_q15 = n_pix * int(bundle.level_log2_n_q15[lvl_idx])
        numer_q15 = max(0, total_q15 - sum_q15)
        num = numer_q15 << bundle.feature_frac_bits
        den = n_pix * log2_b_q15
        eq = (num + (den >> 1)) // den if den > 0 else 0
        return max(0, min(q_max, int(eq)))

    if bundle.mode == "term_per_level":
        off = int(bundle.term_offsets[lvl_idx])
        acc = sum(int(bundle.term_table[off + int(cnt[b])]) for b in range(bins))
        return max(0, min(q_max, int(acc)))

    # log2_mantissa
    frac_shift = 31 - bundle.mantissa_bits
    frac_mask = (1 << frac_shift) - 1
    sum_clog2c_q15 = 0
    for b in range(bins):
        c = int(cnt[b])
        if c > 1:
            e = 31 - clz32(c)
            rem_shifted = ((c << (31 - e)) & 0x7FFFFFFF)
            idx = rem_shifted >> frac_shift
            frac_rem = rem_shifted & frac_mask
            y0 = int(bundle.mantissa_table[idx])
            y1 = int(bundle.mantissa_table[idx + 1])
            interp_delta = (((y1 - y0) * frac_rem) + (1 << (frac_shift - 1))) >> frac_shift
            log2_c_q15 = (e << 15) + y0 + interp_delta
            sum_clog2c_q15 += c * log2_c_q15
    total_q15 = n_pix * int(bundle.level_log2_n_q15[lvl_idx])
    numer_q15 = max(0, total_q15 - sum_clog2c_q15)
    num = numer_q15 << bundle.feature_frac_bits
    den = n_pix * log2_b_q15
    eq = (num + (den >> 1)) // den if den > 0 else 0
    return max(0, min(q_max, int(eq)))


def extract_x94_q15_single(img: np.ndarray, cfg: CImpConfig) -> np.ndarray:
    """Bit-exact Python mirror of `cimp_descriptor_x94_q15` for one uint8 image."""
    spec = cfg.descriptor_spec
    qlut = build_bin_quant_lut(spec.bins)
    levels_asc = sorted(spec.grid_levels)
    bundle = build_shannon_lut(
        image_size=spec.image_size,
        grid_levels=levels_asc,
        bins=spec.bins,
        mode=cfg.descriptor.shannon_lut_mode,
        feature_frac_bits=cfg.numeric.feature_frac_bits,
        mantissa_bits=cfg.descriptor.log2_mantissa_bits,
    )
    log2_b_q15 = int(math.floor(math.log2(spec.bins) * 32768.0 + 0.5))
    q_max = min(32767, (1 << bundle.feature_frac_bits) - 1)

    g_max = max(levels_asc)
    cell_side = spec.image_size // g_max
    q_img = qlut[np.asarray(img, dtype=np.uint8)]

    hist_by_lvl: dict[int, np.ndarray] = {}
    fine = np.zeros((g_max, g_max, spec.bins), dtype=np.int64)
    for r in range(g_max):
        for c in range(g_max):
            patch = q_img[r * cell_side : (r + 1) * cell_side, c * cell_side : (c + 1) * cell_side]
            fine[r, c] = np.bincount(patch.ravel(), minlength=spec.bins)
    hist_by_lvl[g_max] = fine

    for lvl in range(len(levels_asc) - 2, -1, -1):
        g_curr = levels_asc[lvl]
        g_child = levels_asc[lvl + 1]
        k = g_child // g_curr
        hist_by_lvl[g_curr] = hist_by_lvl[g_child].reshape(g_curr, k, g_curr, k, spec.bins).sum(axis=(1, 3))

    out = np.empty(spec.n_x94, dtype=np.int16)
    cursor = 0
    off_map: dict[int, int] = {}
    for lvl_idx, g in enumerate(levels_asc):
        off_map[g] = cursor
        for r in range(g):
            for c in range(g):
                out[cursor] = np.int16(
                    _eval_cell_q15(hist_by_lvl[g][r, c], lvl_idx, spec.bins, bundle, log2_b_q15)
                )
                cursor += 1

    # Stage 4: asymmetry
    for g in spec.asym_scales:
        grid = out[off_map[g] : off_map[g] + g * g].reshape(g, g).astype(np.int64)
        denom = g * g
        is_pow2 = (denom & (denom - 1)) == 0
        shift = denom.bit_length() - 1 if is_pow2 else -1
        inv_q31 = int(round((1 << 31) / float(denom)))
        for kind in spec.asym_kinds:
            if kind == "lr":
                mirror = grid[:, ::-1]
            elif kind == "tb":
                mirror = grid[::-1, :]
            else:
                mirror = grid[::-1, ::-1]
            acc = int(np.sum(np.abs(grid - mirror)))
            if shift >= 0:
                qval = round_shift_right_i64(acc, shift)
            else:
                qval = round_shift_right_i64(acc * inv_q31, 31)
            out[cursor] = np.int16(max(0, min(q_max, int(qval))))
            cursor += 1

    return out


def extract_x94_q15_batch(images: np.ndarray, cfg: CImpConfig) -> np.ndarray:
    """Evaluate `extract_x94_q15_single` across a batch [N, H, W] -> int16[N, 94]."""
    arr = np.asarray(images, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    return np.stack([extract_x94_q15_single(arr[i], cfg) for i in range(arr.shape[0])], axis=0)
