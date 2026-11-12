"""Pedagogical step-by-step trace of single-pass histogram pyramid and additivity parity."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, LAB_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from configs.logging import configure_logging, logger
from config_loader import (
    EntropyOptimizationConfig,
    build_nlogn_lut,
    find_preprocessed_image_paths,
    generate_synthetic_images,
    load_optimization_config,
    validate_nested_grid_levels,
)
from src.features.entropy_maps import (
    build_entropy_plan,
    build_quantization_lut,
    compute_entropy_features,
    region_histograms,
)


def _load_sample_image(
    cfg: EntropyOptimizationConfig,
    image_path: Path | None,
    force_synthetic: bool,
) -> tuple[np.ndarray, str]:
    """Load one grayscale image of configured size, falling back safely when real data is absent."""
    sz = cfg.data.image_size
    if image_path is not None:
        resolved = (
            image_path.resolve()
            if image_path.is_absolute()
            else (PROJECT_ROOT / image_path).resolve()
        )
        if not resolved.is_file():
            raise FileNotFoundError(f"Requested image file not found: {resolved}")
        with Image.open(resolved) as im:
            gray = im.convert("L")
            if gray.size != (sz, sz):
                gray = gray.resize((sz, sz), resample=Image.Resampling.BILINEAR)
            return np.ascontiguousarray(np.asarray(gray, dtype=np.uint8)), str(resolved)

    if force_synthetic or cfg.data.source == "synthetic":
        img = generate_synthetic_images(1, sz, cfg.seed)[0]
        return img, f"synthetic(seed={cfg.seed})"

    real_paths = find_preprocessed_image_paths(cfg.data)
    if real_paths:
        chosen = real_paths[0]
        with Image.open(chosen) as im:
            gray = im.convert("L")
            if gray.size != (sz, sz):
                gray = gray.resize((sz, sz), resample=Image.Resampling.BILINEAR)
            return np.ascontiguousarray(np.asarray(gray, dtype=np.uint8)), f"real:{chosen.name}"

    logger.warning(
        "No preprocessed images found under %s; generating deterministic synthetic sample.",
        cfg.data.preprocessed_dir,
    )
    img = generate_synthetic_images(1, sz, cfg.seed)[0]
    return img, f"synthetic_fallback(seed={cfg.seed})"


def _compute_single_pass_trace(
    img_u8: np.ndarray,
    grid_levels: tuple[int, ...],
    bins: int,
    nlogn_lut: np.ndarray,
) -> tuple[
    np.ndarray,
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    np.ndarray,
    tuple[tuple[int, int, int], ...],
]:
    """Execute and trace quantization -> finest histograms -> pyramid child sums -> LUT entropy."""
    size = int(img_u8.shape[0])
    validate_nested_grid_levels(size, grid_levels)
    qlut = build_quantization_lut(bins)
    quantized = qlut[img_u8]  # [H, W] uint8, values in [0, bins - 1]

    desc_levels = sorted(grid_levels, reverse=True)
    g_max = desc_levels[0]
    cell_side = size // g_max

    # Single-pass sequential fine-grid histogram fill over every pixel exactly once
    hist_by_level: dict[int, np.ndarray] = {}
    fine_hist = np.zeros((g_max, g_max, bins), dtype=np.int64)
    for br in range(g_max):
        y0 = br * cell_side
        y1 = y0 + cell_side
        for bc in range(g_max):
            x0 = bc * cell_side
            x1 = x0 + cell_side
            patch_q = quantized[y0:y1, x0:x1].ravel()
            fine_hist[br, bc] = np.bincount(patch_q, minlength=bins)
    hist_by_level[g_max] = fine_hist

    # Bottom-up child summation across nested pyramid levels
    merge_edges: list[tuple[int, int, int]] = []
    for idx in range(1, len(desc_levels)):
        g_curr = desc_levels[idx]
        g_child = next(
            desc_levels[j] for j in range(idx - 1, -1, -1) if desc_levels[j] % g_curr == 0
        )
        k = g_child // g_curr
        merge_edges.append((g_curr, g_child, k))
        child_grid = hist_by_level[g_child]  # [g_child, g_child, bins]
        hist_by_level[g_curr] = (
            child_grid.reshape(g_curr, k, g_curr, k, bins)
            .sum(axis=(1, 3))
        )

    # Verify integer identity against canonical per-level region_histograms
    for g in desc_levels:
        direct = region_histograms(quantized[None, ...], g=g, bins=bins)[0].reshape(g, g, bins)
        if not np.array_equal(hist_by_level[g], direct):
            raise AssertionError(
                f"Additivity check failed at level g={g}: pyramid histogram differs from direct region_histograms!"
            )

    # Evaluate normalized entropy via precomputed c * log2(c) LUT
    log2_bins = math.log2(bins)
    entropy_grids: dict[int, np.ndarray] = {}
    parts: list[np.ndarray] = []
    for g in grid_levels:
        counts_g = hist_by_level[g]  # [g, g, bins]
        side = size // g
        n_pix = side * side
        log2_n = math.log2(n_pix)
        s_grid = nlogn_lut[counts_g].sum(axis=-1)
        h_grid = (log2_n - (s_grid / float(n_pix))) / log2_bins
        h_grid = np.maximum(h_grid, 0.0)
        is_const = np.max(counts_g, axis=-1) == n_pix
        h_grid[is_const] = 0.0
        is_equiprobable = np.all(counts_g == counts_g[:, :, :1], axis=-1)
        h_grid[is_equiprobable] = 1.0
        entropy_grids[g] = h_grid
        parts.append(h_grid.ravel())

    feature_vec = np.concatenate(parts, axis=0)
    return quantized, hist_by_level, entropy_grids, feature_vec, tuple(merge_edges)


def _save_pedagogical_figure(
    img_u8: np.ndarray,
    quantized: np.ndarray,
    hist_by_level: dict[int, np.ndarray],
    entropy_grids: dict[int, np.ndarray],
    grid_levels: tuple[int, ...],
    bins: int,
    source_label: str,
    max_abs_err: float,
    output_dir: Path,
    dpi: int,
    formats: tuple[str, ...],
) -> list[Path]:
    """Render two-row pedagogical diagram showing pixel->fine->pyramid->entropy correspondence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    desc_levels = sorted(grid_levels, reverse=True)
    g_max = desc_levels[0]
    size = int(img_u8.shape[0])
    cell_side = size // g_max

    n_scales = len(grid_levels)
    n_cols = max(3, n_scales)
    fig = plt.figure(figsize=(4.2 * n_cols, 7.8), dpi=dpi)
    gs = GridSpec(2, n_cols, figure=fig, height_ratios=[1.1, 1.0], hspace=0.40, wspace=0.48)

    fig.suptitle(
        f"Single-Pass Entropy Pyramid Mechanics ({size}x{size}, bins={bins}, levels={list(grid_levels)}) "
        f"| source={source_label} | max|err|={max_abs_err:.2e}",
        fontsize=11,
        fontweight="bold",
    )

    # Row 0 Col 0: Input image with finest grid + highlighted sample child block
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img_u8, cmap="gray", vmin=0, vmax=255)
    for pos in range(cell_side, size, cell_side):
        ax_img.axhline(pos - 0.5, color="#38bdf8", linewidth=0.55, alpha=0.65)
        ax_img.axvline(pos - 0.5, color="#38bdf8", linewidth=0.55, alpha=0.65)
    # Highlight block merging into top-left parent cell if coarser scale exists
    if len(desc_levels) >= 2:
        k_sample = g_max // desc_levels[1]
        span = k_sample * cell_side
        rect = Rectangle(
            (-0.5, -0.5),
            span,
            span,
            linewidth=1.8,
            edgecolor="#ef4444",
            facecolor="none",
        )
        ax_img.add_patch(rect)
        title_box = f"Input uint8 + {g_max}x{g_max} grid\n(red: {k_sample}x{k_sample} child block)"
    else:
        title_box = f"Input uint8 + {g_max}x{g_max} grid"
    ax_img.set_title(title_box, fontsize=9.5)
    ax_img.set_xlabel("x (px)")
    ax_img.set_ylabel("y (px)")

    # Row 0 Col 1: Quantized image
    ax_q = fig.add_subplot(gs[0, 1])
    im_q = ax_q.imshow(quantized, cmap="viridis", vmin=0, vmax=bins - 1)
    ax_q.set_title(f"Quantized bins [0, {bins - 1}]\n(read pixel 1x)", fontsize=9.5)
    ax_q.set_xlabel("x (px)")
    fig.colorbar(im_q, ax=ax_q, fraction=0.046, pad=0.04)

    # Row 0 Col 2..: Additivity verification bar chart for child->parent cell (0, 0)
    ax_bar = fig.add_subplot(gs[0, 2:])
    bin_idx = np.arange(bins, dtype=np.int32)
    fine_grid = hist_by_level[g_max]
    if len(desc_levels) >= 2:
        g_parent = desc_levels[1]
        k = g_max // g_parent
        bottom = np.zeros(bins, dtype=np.int64)
        child_cmap = plt.get_cmap("tab10")
        c_idx = 0
        for dr in range(k):
            for dc in range(k):
                vals = fine_grid[dr, dc]
                lbl = f"fine({dr},{dc})" if k * k <= 4 else None
                ax_bar.bar(
                    bin_idx,
                    vals,
                    bottom=bottom,
                    width=0.72,
                    alpha=0.75,
                    color=child_cmap(c_idx % 10),
                    label=lbl,
                )
                bottom += vals
                c_idx += 1
        parent_vals = hist_by_level[g_parent][0, 0]
        ax_bar.plot(
            bin_idx,
            parent_vals,
            color="#111827",
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            label=f"parent {g_parent}x{g_parent}(0,0) = exact sum",
        )
        ax_bar.set_title(
            f"Exact Additivity: sum({k}x{k} fine cells) == {g_parent}x{g_parent} parent cell (0,0)",
            fontsize=9.5,
        )
    else:
        ax_bar.bar(bin_idx, fine_grid[0, 0], width=0.72, color="#2563eb", label="cell (0,0)")
        ax_bar.set_title("Finest Cell (0,0) Histogram", fontsize=9.5)
    ax_bar.set_xlabel("Quantized intensity bin b")
    ax_bar.set_ylabel("Pixel count")
    ax_bar.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax_bar.legend(loc="upper right", fontsize=8)

    # Row 1: Normalized entropy heatmaps across each configured grid scale
    for col_idx, g in enumerate(grid_levels):
        ax_e = fig.add_subplot(gs[1, col_idx])
        grid_h = entropy_grids[g]
        im_e = ax_e.imshow(grid_h, cmap="magma", vmin=0.0, vmax=1.0)
        side = size // g
        ax_e.set_title(
            f"Scale {g}x{g} ({g * g} cells, {side}x{side} px)\n"
            f"min={grid_h.min():.3f}, max={grid_h.max():.3f}",
            fontsize=9,
        )
        ax_e.set_xlabel("col c")
        if col_idx == 0:
            ax_e.set_ylabel("row r")
        fig.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04)

    saved: list[Path] = []
    for fmt in formats:
        out_path = output_dir / f"explain_single_pass.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        saved.append(out_path)
    plt.close(fig)
    return saved


def run_explanation(
    cfg: EntropyOptimizationConfig,
    image_path: Path | None = None,
    force_synthetic: bool = False,
) -> list[Path]:
    """Execute pedagogical step-by-step verification and save visualization."""
    img_u8, source_label = _load_sample_image(cfg, image_path, force_synthetic)
    size = int(img_u8.shape[0])
    total_pixels = size * size
    n_levels = len(cfg.grid_levels)
    g_min = min(cfg.grid_levels)
    nlogn_lut = build_nlogn_lut((size // g_min) ** 2)

    logger.info("=== Stage 2: Single-Pass Entropy Pyramid Pedagogical Trace ===")
    logger.info(
        "Image shape=%s, dtype=%s, source=%s, levels=%s, bins=%d",
        img_u8.shape,
        img_u8.dtype,
        source_label,
        cfg.grid_levels,
        cfg.bins,
    )
    logger.info(
        "Pixel reads per image: baseline=%d passes (%d pixel reads + quantize), "
        "single-pass=1 pass (%d pixel reads total)",
        n_levels,
        n_levels * total_pixels,
        total_pixels,
    )

    quantized, hist_by_level, entropy_grids, vec_single, merge_edges = _compute_single_pass_trace(
        img_u8=img_u8,
        grid_levels=cfg.grid_levels,
        bins=cfg.bins,
        nlogn_lut=nlogn_lut,
    )

    logger.info(
        "Step 1 Quantize: min_bin=%d, max_bin=%d (sample corners TL=%d->%d, BR=%d->%d)",
        int(quantized.min()),
        int(quantized.max()),
        int(img_u8[0, 0]),
        int(quantized[0, 0]),
        int(img_u8[-1, -1]),
        int(quantized[-1, -1]),
    )
    g_max = max(cfg.grid_levels)
    logger.info(
        "Step 2 Finest %dx%d histograms: shape=%s, total_count_sum=%d (expected %d)",
        g_max,
        g_max,
        hist_by_level[g_max].shape,
        int(hist_by_level[g_max].sum()),
        total_pixels,
    )
    for g_curr, g_child, k in merge_edges:
        logger.info(
            "Step 3 Pyramid sum: %dx%d -> %dx%d (merge %dx%d children), asserted exact == direct region_histograms",
            g_child,
            g_child,
            g_curr,
            g_curr,
            k,
            k,
        )
    if merge_edges:
        g_parent, g_child, k = merge_edges[0]
        child_sample = hist_by_level[g_child][0:k, 0:k].sum(axis=(0, 1))
        parent_sample = hist_by_level[g_parent][0, 0]
        logger.info(
            "Sample cell (0,0) %dx%d sum check: first 6 bins child_sum=%s == parent=%s",
            g_parent,
            g_parent,
            child_sample[: min(6, cfg.bins)].tolist(),
            parent_sample[: min(6, cfg.bins)].tolist(),
        )

    plan = build_entropy_plan(size, cfg.grid_levels, cfg.bins)
    vec_oracle = compute_entropy_features(
        img_u8[None, ...],
        grid_levels=cfg.grid_levels,
        bins=cfg.bins,
        plan=plan,
    )[0]
    max_abs_err = float(np.max(np.abs(vec_single - vec_oracle)))
    logger.info(
        "Step 4 Entropy parity check vs src oracle: features=%d, max|diff|=%.3e (tolerance=%.1e)",
        vec_single.size,
        max_abs_err,
        cfg.protocol.tolerance_abs,
    )
    if max_abs_err > cfg.protocol.tolerance_abs:
        raise AssertionError(
            f"Single-pass pedagogical trace diverged from src oracle: "
            f"max|diff|={max_abs_err:.3e} > {cfg.protocol.tolerance_abs:.1e}"
        )

    saved_paths = _save_pedagogical_figure(
        img_u8=img_u8,
        quantized=quantized,
        hist_by_level=hist_by_level,
        entropy_grids=entropy_grids,
        grid_levels=cfg.grid_levels,
        bins=cfg.bins,
        source_label=source_label,
        max_abs_err=max_abs_err,
        output_dir=cfg.plots.output_dir,
        dpi=cfg.plots.dpi,
        formats=cfg.plots.formats,
    )
    for out_p in saved_paths:
        logger.info("Saved pedagogical visualization: %s", out_p)
    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for pedagogical single-pass entropy explanation lab."""
    parser = argparse.ArgumentParser(
        prog="explain_single_pass.py",
        description="Trace single-pass finest-grid histogramming, child summation, and oracle parity.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/optimization/entropy_maps.toml"),
        help="Path to optimization TOML config.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional explicit grayscale/RGB image path (resized if needed to TOML image_size).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force deterministic synthetic test pattern instead of scanning preprocessed images.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for Stage 2 pedagogical explanation script."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg = load_optimization_config(args.config, project_root=PROJECT_ROOT)
        _ = run_explanation(cfg, image_path=args.image, force_synthetic=bool(args.synthetic))
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        logger.error("Pedagogical lab failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
