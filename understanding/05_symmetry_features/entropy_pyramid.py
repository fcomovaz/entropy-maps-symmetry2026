"""Educational spatial entropy pyramid; scalar Python and Pillow only."""
import math
from pathlib import Path
import sys

from PIL import Image

from symmetry_features import Grid

# Reuse the stable scalar Shannon reference, without running its benchmark.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "03_entropy_benchmark"))
from entropy_benchmark import e1

LEVELS = (1, 2, 4, 8)


def entropy_details(image: Image.Image, bins: int = 16) -> dict:
    """Return counts, p, p*log2(p), H and H/log2(B) for a nonempty grayscale region.

    Quantization is bin = intensity * B // 256; B is an integer in [2,256].
    Normalized entropy is mathematically in [0,1], without clipping.
    """
    if type(bins) is not int or not 2 <= bins <= 256:
        raise ValueError("bins must be an integer between 2 and 256")
    pixels = image.convert("L").tobytes()
    if not pixels:
        raise ValueError("Entropy requires a nonempty image region")
    counts = [0] * bins
    for pixel in pixels:
        counts[pixel * bins // 256] += 1
    probabilities = [count / len(pixels) for count in counts]
    terms = [p * math.log2(p) if p else 0.0 for p in probabilities]
    entropy = e1(counts, len(pixels))
    return {"counts": counts, "probabilities": probabilities, "p_log2_p": terms,
            "entropy_bits": entropy, "normalized_entropy": entropy / math.log2(bins)}


def compute_entropy_pyramid(image: Image.Image, bins: int = 16) -> dict[int, Grid]:
    """Partition image into [g,g] entropy grids, g=1,2,4,8; each value is H/log2(B).

    Require dimensions divisible by 8: equal-area cells, no resizing or padding.
    Rectangular images are accepted. Each level recomputes pooled histograms;
    coarse entropy is NOT the average of fine entropies.
    """
    gray = image.convert("L")
    width, height = gray.size
    if width < 8 or height < 8 or width % 8 or height % 8:
        raise ValueError("Image width and height must be positive multiples of 8")
    grids = {}
    for g in LEVELS:
        cell_w, cell_h = width // g, height // g
        grids[g] = [
            [entropy_details(gray.crop((j*cell_w, i*cell_h,
                                       (j+1)*cell_w, (i+1)*cell_h)), bins)
             ["normalized_entropy"] for j in range(g)] for i in range(g)
        ]
    return grids


def flatten_entropy_features(grids: dict[int, Grid]) -> list[float]:
    """Return 85 entropies: levels 1,2,4,8, each flattened in row-major order."""
    for g in LEVELS:
        if len(grids[g]) != g or any(len(row) != g for row in grids[g]):
            raise ValueError(f"Expected entropy grid {g}x{g}")
    return [value for g in LEVELS for row in grids[g] for value in row]
