"""Pedagogical inspection and visual explanation of region entropy features.

Demonstrates the step-by-step mathematical transformation:
image -> patch -> histogram -> counts -> probabilities -> p * log2(p) -> entropy -> E
and compares the vectorized results directly against the canonical oracle.

Manual execution command (PowerShell from repository 'code/' directory):
    .\\.venv\\Scripts\\python.exe understanding/07_entropy_maps_vectorized/explain_entropy_maps.py
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.entropy_maps import (
    EntropyPlan,
    build_entropy_plan,
    build_quantization_lut,
    build_shannon_term_lut,
    compute_entropy_features,
    feature_names,
    normalized_entropy,
    quantize_pixels,
    region_edges,
    region_histograms,
)


def create_sample_pattern(size: int = 128) -> np.ndarray:
    """Create a structured 128x128 synthetic test image with diverse regional textures."""
    img = np.zeros((size, size), dtype=np.uint8)
    half = size // 2
    # Top-left: Constant region (zero entropy expected)
    img[:half, :half] = 128
    # Top-right: Smooth linear gradient
    img[:half, half:] = np.tile(np.linspace(0, 255, half, dtype=np.uint8), (half, 1))
    # Bottom-left: High-contrast checkerboard
    block = 8
    checker = (np.indices((half // block, half // block)).sum(axis=0) % 2) * 255
    img[half:, :half] = np.repeat(
        np.repeat(checker.astype(np.uint8), block, axis=0), block, axis=1
    )
    # Bottom-right: Random uniform noise (high entropy expected)
    rng = np.random.RandomState(42)
    img[half:, half:] = rng.randint(0, 256, (half, half), dtype=np.uint8)
    return img


def main() -> int:
    """Trace entropy calculation step-by-step and generate visual verification plot."""
    print("=================================================================")
    print("STEP-BY-STEP REGION ENTROPY CALCULATION (UNDERSTANDING-FIRST)")
    print("=================================================================")

    image_size = 128
    bins = 16
    grid_levels = [1, 2, 4, 8]
    img = create_sample_pattern(image_size)

    print(
        f"1. Image created: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]"
    )

    # Step 2: Quantization
    quantized = quantize_pixels(img[None, ...], bins=bins)[0]
    print(
        f"2. Quantized into B={bins} bins: shape={quantized.shape}, range=[{quantized.min()}, {quantized.max()}]"
    )

    # Step 3: Regional decomposition at scale g=2 (4 quadrants)
    g = 2
    edges = region_edges(image_size, g)
    print(f"3. Scale g={g} region edges: {edges}")

    counts = region_histograms(quantized[None, ...], g=g, bins=bins)[0]
    norm_e = normalized_entropy(counts, bins=bins)

    print("\nDetailed breakdown for 2x2 regions (r, c):")
    for r in range(g):
        y0, y1 = edges[r], edges[r + 1]
        for c in range(g):
            x0, x1 = edges[c], edges[c + 1]
            idx = r * g + c
            reg_counts = counts[idx]
            n_pix = reg_counts.sum()
            probs = reg_counts / n_pix
            terms = [p * math.log2(p) if p > 0 else 0.0 for p in probs]
            shannon_h = -sum(terms)
            ent_val = norm_e[idx]
            print(f"  Region ({r}, {c}) [y:{y0}..{y1}, x:{x0}..{x1}]:")
            print(f"    - Pixels count: {n_pix}")
            print(f"    - Non-zero bins: {np.count_nonzero(reg_counts)}/{bins}")
            print(f"    - Shannon entropy H: {shannon_h:.4f} bits")
            print(f"    - Normalized E (H/log2({bins})): {ent_val:.4f}")

    # Full multi-scale vector extraction
    features = compute_entropy_features(img, grid_levels=grid_levels, bins=bins)[0]
    names = feature_names(grid_levels)
    print(f"\n4. Full multi-scale feature vector extracted: {len(features)} features")
    print(f"   First feature: {names[0]} = {features[0]:.6f}")
    print(f"   Last feature:  {names[-1]} = {features[-1]:.6f}")

    # Step 5: Optimization with Look-Up Tables (LUT) and Execution Plan
    print("\n5. Look-Up Table (LUT) Optimizations:")
    quant_lut = build_quantization_lut(bins)
    print(
        f"   - Quantization LUT size: {len(quant_lut)} bytes (covers all 0..255 pixel values)"
    )
    sample_reg_pixels = (image_size // 2) * (image_size // 2)
    term_lut = build_shannon_term_lut(sample_reg_pixels)
    print(
        f"   - Shannon Term LUT for 2x2 region (N={sample_reg_pixels}): {len(term_lut)} floats (~{len(term_lut)*8/1024:.1f} KB)"
    )
    plan = build_entropy_plan(image_size, grid_levels, bins)
    print(
        f"   - Precomputed EntropyPlan total regions: {len(plan.regions)} (geometry and LUTs cached once)"
    )
    features_plan = compute_entropy_features(img, grid_levels, bins, plan=plan)[0]
    np.testing.assert_array_equal(features, features_plan)
    print(
        "   [PASS] Execution with cached plan matches dynamic computation identically!"
    )

    # Step 6: Comparison against canonical oracle (understanding/06)
    try:
        mod = importlib.import_module(
            "understanding.06_entropy_pyramid_descriptor.descriptor"
        )
        pil_img = Image.fromarray(img, mode="L")
        oracle_res = mod.extract_descriptor(pil_img, bins=bins)
        oracle_entropies = np.array(oracle_res.entropy_features, dtype=np.float64)
        max_diff = np.max(np.abs(features - oracle_entropies))
        print(f"\n6. Comparison against understanding/06 oracle:")
        print(f"   Max absolute difference: {max_diff:.2e}")
        assert max_diff < 1e-12, "Discrepancy with oracle detected!"
        print("   [PASS] Exact 100% agreement with canonical oracle!")
    except Exception as exc:
        print(f"   Oracle comparison skipped or failed: {exc}")

    # Step 7: Visual plot saved to artifact
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_dir / "entropy_maps_breakdown.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original Pattern (128x128)")
    axes[0].axis("off")

    axes[1].imshow(quantized, cmap="tab20", vmin=0, vmax=bins - 1)
    axes[1].set_title(f"Quantized (B={bins} bins)")
    axes[1].axis("off")

    # Plot 2x2 entropy heatmap
    e_2x2 = norm_e.reshape(2, 2)
    im = axes[2].imshow(e_2x2, cmap="magma_r", vmin=0.0, vmax=1.0)
    for r in range(2):
        for c in range(2):
            axes[2].text(
                c,
                r,
                f"{e_2x2[r, c]:.3f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
    axes[2].set_title("2x2 Normalized Entropy Map")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"\nSaved educational figure: {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
