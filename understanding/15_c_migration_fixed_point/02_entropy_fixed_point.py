"""Lab 15.2: Fixed-point Shannon entropy LUT modes and x94 pyramid accuracy study.

Compares three integer LUT formulations against src.features.entropy_maps & asymmetry_maps:
    1. nlogn_full: 1D uint32 LUT of c * log2(c)
    2. term_per_level: per-level - (c/N) * log2(c/N) / log2(B) in Q0.15
    3. log2_mantissa: clz-based exponent + 2^k+1 mantissa LUT (compact MCU target)

Manual execution (PowerShell from 'code/' directory):
    .\\.venv\\Scripts\\python.exe understanding/15_c_migration_fixed_point/02_entropy_fixed_point.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import configure_logging, logger
from src.features.asymmetry_maps import compute_asymmetry_for_grid, reconstruct_grids
from src.features.entropy_maps import (
    build_entropy_plan,
    build_quantization_lut,
    compute_entropy_features,
)


def clz32(x: int) -> int:
    """Count leading zeros of a 32-bit unsigned integer (exact match to __builtin_clz)."""
    v = int(x) & 0xFFFFFFFF
    if v == 0:
        return 32
    return 32 - v.bit_length()


def build_log2_mantissa_table(k_bits: int = 8) -> np.ndarray:
    """Build (2^k + 1) Q1.15 entries of log2(1 + m / 2^k) rounded half-up."""
    size = 1 << k_bits
    lut = np.empty(size + 1, dtype=np.int32)
    for idx in range(size + 1):
        val = math.log2(1.0 + float(idx) / float(size))
        lut[idx] = int(math.floor(val * 32768.0 + 0.5))
    return lut


def clog2c_mantissa_q16(c: int, k_bits: int, mantissa_lut: np.ndarray) -> int:
    """Compute c * log2(c) in signed/unsigned fixed point with 15 frac bits via clz + mantissa LUT.

    Returns integer scaled by 2^15 (`c * log2(c) * 32768`). Exact power-of-two counts have zero
    mantissa fractional part and yield exact integers `c * e << 15`.
    """
    if c <= 1:
        return 0
    e = 31 - clz32(c)
    # Extract fractional bits immediately below the leading 1-bit of c:
    # norm = (c << (31 - e)) & 0x7FFFFFFF puts bit (e-1) at bit 30 down to 0
    rem_shifted = ((c << (31 - e)) & 0x7FFFFFFF)
    frac_shift = 31 - k_bits
    idx = rem_shifted >> frac_shift
    # Linear interpolation residual for tighter accuracy when e > k_bits
    frac_rem = rem_shifted & ((1 << frac_shift) - 1)
    y0 = int(mantissa_lut[idx])
    y1 = int(mantissa_lut[idx + 1])
    interp = y0 + (((y1 - y0) * frac_rem + (1 << (frac_shift - 1))) >> frac_shift) if frac_shift > 0 else y0
    log2_c_q15 = (e << 15) + interp
    return int(c) * int(log2_c_q15)


def cell_entropy_nlogn_q15(
    counts: np.ndarray,
    bins: int,
    n_pix: int,
    nlogn_q15_lut: np.ndarray,
) -> int:
    """Compute single cell normalized entropy E in Q0.15 [0..32767] using full c*log2(c) LUT."""
    if n_pix <= 0:
        return 0
    # Analytical equiprobable check
    first = int(counts[0])
    if all(int(c) == first for c in counts):
        return 32767
    sum_nlogn = sum(int(nlogn_q15_lut[int(c)]) for c in counts if c > 0)
    log2_n = int(round(math.log2(n_pix)))  # exact for powers of 2
    # E_raw = (log2(N) - sum(c*log2(c)) / N) / log2(B)
    # Using exact arithmetic in scaled integer form (log2_B = 4 for B=16):
    # (N * log2(N) << 15) - sum_nlogn
    numer = (n_pix * log2_n * 32768) - sum_nlogn
    if numer <= 0:
        return 0
    log2_b_den = round(math.log2(bins) * 1024.0)
    if bins & (bins - 1) == 0:
        log2_b = bins.bit_length() - 1
        den = n_pix * log2_b
        e_q15 = (numer + (den >> 1)) // den
    else:
        e_q15 = (numer * 1024 + ((n_pix * log2_b_den) >> 1)) // (n_pix * log2_b_den)
    return max(0, min(32767, int(e_q15)))


def cell_entropy_mantissa_q15(
    counts: np.ndarray,
    bins: int,
    n_pix: int,
    k_bits: int,
    mantissa_lut: np.ndarray,
) -> int:
    """Compute single cell normalized entropy E in Q0.15 using compact clz + mantissa LUT."""
    if n_pix <= 0:
        return 0
    first = int(counts[0])
    if all(int(c) == first for c in counts):
        return 32767
    sum_nlogn = sum(clog2c_mantissa_q16(int(c), k_bits, mantissa_lut) for c in counts if c > 0)
    log2_n = int(round(math.log2(n_pix))) if (n_pix & (n_pix - 1) == 0) else math.log2(n_pix)
    numer = int(round(n_pix * log2_n * 32768)) - sum_nlogn
    if numer <= 0:
        return 0
    if bins & (bins - 1) == 0:
        log2_b = bins.bit_length() - 1
        den = n_pix * log2_b
        e_q15 = (numer + (den >> 1)) // den
    else:
        log2_b = math.log2(bins)
        e_q15 = int(round(numer / (n_pix * log2_b)))
    return max(0, min(32767, int(e_q15)))


def cell_entropy_term_q15(
    counts: np.ndarray,
    term_q15_lut: np.ndarray,
) -> int:
    """Compute single cell normalized entropy E in Q0.15 by summing precomputed per-level terms."""
    first = int(counts[0])
    if all(int(c) == first for c in counts):
        return 32767
    acc = sum(int(term_q15_lut[int(c)]) for c in counts if c > 0)
    return max(0, min(32767, int(acc)))


def compute_x94_fixed_point(
    img: np.ndarray,
    grid_levels: list[int],
    bins: int,
    mode: str,
    mantissa_bits: int = 8,
) -> np.ndarray:
    """Extract x94 descriptor in Q0.15 converted to float64 for comparison with float oracle."""
    h, w = img.shape
    qlut = build_quantization_lut(bins)
    q_img = qlut[img]
    g_max = max(grid_levels)
    cell_side = h // g_max

    # Build tables
    max_n = (h // min(grid_levels)) ** 2
    c_arr = np.arange(max_n + 1, dtype=np.float64)
    nlogn_q15 = np.zeros(max_n + 1, dtype=np.int64)
    nlogn_q15[1:] = np.floor(c_arr[1:] * np.log2(c_arr[1:]) * 32768.0 + 0.5).astype(np.int64)
    mant_lut = build_log2_mantissa_table(mantissa_bits)
    log2_b = math.log2(bins)
    term_luts: dict[int, np.ndarray] = {}
    for g in grid_levels:
        npix = (h // g) ** 2
        t_lut = np.zeros(npix + 1, dtype=np.int32)
        p = np.arange(1, npix + 1, dtype=np.float64) / npix
        t_lut[1:] = np.floor((-p * np.log2(p) / log2_b) * 32768.0 + 0.5).astype(np.int32)
        term_luts[g] = t_lut

    # Fine histograms
    fine_hist = np.zeros((g_max, g_max, bins), dtype=np.int32)
    for r in range(g_max):
        for c in range(g_max):
            patch = q_img[r * cell_side : (r + 1) * cell_side, c * cell_side : (c + 1) * cell_side]
            fine_hist[r, c] = np.bincount(patch.ravel(), minlength=bins)

    # Hierarchical merge and entropy
    hist_by_level: dict[int, np.ndarray] = {g_max: fine_hist}
    for g in sorted(grid_levels, reverse=True)[1:]:
        child_g = g * 2
        if child_g not in hist_by_level:
            child_g = min(k for k in hist_by_level if k > g and k % g == 0)
        k = child_g // g
        hist_by_level[g] = hist_by_level[child_g].reshape(g, k, g, k, bins).sum(axis=(1, 3))

    x85_q15: list[int] = []
    grids_q15: dict[int, np.ndarray] = {}
    for g in grid_levels:
        npix = (h // g) ** 2
        grid_e = np.empty((g, g), dtype=np.int32)
        for r in range(g):
            for c in range(g):
                cnt = hist_by_level[g][r, c]
                if mode == "nlogn_full":
                    val = cell_entropy_nlogn_q15(cnt, bins, npix, nlogn_q15)
                elif mode == "term_per_level":
                    val = cell_entropy_term_q15(cnt, term_luts[g])
                else:
                    val = cell_entropy_mantissa_q15(cnt, bins, npix, mantissa_bits, mant_lut)
                grid_e[r, c] = val
                x85_q15.append(val)
        grids_q15[g] = grid_e

    # Asymmetry x9 on scales [2, 4, 8]
    x9_q15: list[int] = []
    for g in [lvl for lvl in grid_levels if lvl > 1]:
        eg = grids_q15[g]
        denom = g * g
        shift = denom.bit_length() - 1
        sum_lr = int(np.sum(np.abs(eg - eg[:, ::-1])))
        sum_tb = int(np.sum(np.abs(eg - eg[::-1, :])))
        sum_rot = int(np.sum(np.abs(eg - eg[::-1, ::-1])))
        for s in (sum_lr, sum_tb, sum_rot):
            x9_q15.append(min(32767, (s + (1 << (shift - 1))) >> shift))

    return np.asarray(x85_q15 + x9_q15, dtype=np.float64) / 32768.0


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for Lab 15.2."""
    parser = argparse.ArgumentParser(description="Study fixed-point Shannon entropy LUT modes on real images.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/understanding/c_migration_fixed_point.toml"),
        help="Path to Lab 15 TOML config relative to code/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Compare the three Shannon LUT strategies against the float64 oracle."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg_path = (PROJECT_ROOT / args.config).resolve()
        with cfg_path.open("rb") as f:
            cfg = tomllib.load(f)
        out_dir = (PROJECT_ROOT / cfg["experiment"]["output_dir"]).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        dpi = int(cfg["experiment"].get("dpi", 160))

        img_size = int(cfg["descriptor"]["image_size"])
        bins = int(cfg["descriptor"]["num_bins"])
        levels = [int(g) for g in cfg["descriptor"]["grid_levels"]]
        asym_scales = [int(g) for g in cfg["descriptor"]["asymmetry_scales"]]
        asym_kinds = [str(k) for k in cfg["descriptor"]["asymmetry_kinds"]]
        mant_bits = int(cfg["fixed_point"].get("log2_mantissa_bits", 8))
        n_samples = int(cfg["dataset"]["sample_count"])

        pre_dir = (PROJECT_ROOT / cfg["dataset"]["preprocessed_dir"]).resolve() / cfg["dataset"]["split"]
        png_paths = sorted(pre_dir.rglob("*.png"))[:n_samples]
        if not png_paths:
            # Deterministic synthetic fallback if dataset images unavailable
            rng = np.random.default_rng(int(cfg["experiment"]["seed"]))
            images = [rng.integers(0, 256, size=(img_size, img_size), dtype=np.uint8) for _ in range(n_samples)]
        else:
            images = [np.asarray(Image.open(p).convert("L"), dtype=np.uint8) for p in png_paths]

        plan = build_entropy_plan(img_size, levels, bins, backend="numpy_baseline")
        arr = np.stack(images, axis=0)
        x85_ref = compute_entropy_features(arr, levels, bins, plan=plan)

        # Assemble float64 x94 reference
        x94_ref_list = []
        offset_map: dict[int, tuple[int, int]] = {}
        cur = 0
        for g in levels:
            offset_map[g] = (cur, cur + g * g)
            cur += g * g
        for idx in range(len(images)):
            row_feats = list(x85_ref[idx])
            for g in asym_scales:
                s, e = offset_map[g]
                grids = reconstruct_grids(x85_ref[idx, s:e], g)
                ares = compute_asymmetry_for_grid(grids, kinds=asym_kinds)
                for k in asym_kinds:
                    row_feats.append(float(ares[k][0]))
            x94_ref_list.append(row_feats)
        x94_ref = np.asarray(x94_ref_list, dtype=np.float64)

        modes = ["nlogn_full", "term_per_level", "log2_mantissa"]
        errors_by_mode: dict[str, np.ndarray] = {}
        lut_bytes = {
            "nlogn_full": (img_size * img_size + 1) * 4,
            "term_per_level": sum(((img_size // g) ** 2 + 1) * 2 for g in levels),
            "log2_mantissa": ((1 << mant_bits) + 1) * 2,
        }
        csv_rows = []
        for mode in modes:
            preds = np.stack(
                [compute_x94_fixed_point(im, levels, bins, mode, mant_bits) for im in images],
                axis=0,
            )
            diff = np.abs(preds - x94_ref)
            errors_by_mode[mode] = diff
            csv_rows.append(
                [
                    mode,
                    lut_bytes[mode],
                    f"{float(np.max(diff)):.6e}",
                    f"{float(np.mean(diff)):.6e}",
                    f"{float(np.max(diff[:, :85])):.6e}",
                    f"{float(np.max(diff[:, 85:])):.6e}",
                ]
            )

        with (out_dir / "02_entropy_lut_comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["mode", "lut_bytes", "max_abs_err_x94", "mean_abs_err_x94", "max_abs_err_x85", "max_abs_err_x9"])
            writer.writerows(csv_rows)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        max_errs = [float(np.max(errors_by_mode[m])) for m in modes]
        mean_errs = [float(np.mean(errors_by_mode[m])) for m in modes]
        x = np.arange(len(modes))
        axes[0].bar(x - 0.15, max_errs, width=0.3, label="Max |delta|")
        axes[0].bar(x + 0.15, mean_errs, width=0.3, label="Mean |delta|")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(modes)
        axes[0].set_yscale("log")
        axes[0].set_title("x94 Fixed-Point Error by LUT Mode")
        axes[0].legend()

        axes[1].bar(modes, [lut_bytes[m] / 1024.0 for m in modes], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        axes[1].set_title("Shannon LUT Flash Size (KB)")
        axes[1].set_ylabel("KB")
        fig.tight_layout()
        fig.savefig(out_dir / "02_entropy_fixed_point.png", dpi=dpi)
        plt.close(fig)

        logger.info("Lab 15.2 complete: outputs -> %s", out_dir)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
