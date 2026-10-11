"""Readable quantized histograms and the scalar Shannon reference from 03."""
from importlib import import_module
import math

from PIL import Image

e1 = import_module("understanding.03_entropy_benchmark.entropy_benchmark").e1
DEFAULT_BINS = 16  # Convention already established by experiment 05.


def validate_bins(bins: int) -> None:
    """Accept exactly integer B in [2,256], as in experiment 05."""
    if type(bins) is not int or not 2 <= bins <= 256:
        raise ValueError("bins must be an integer between 2 and 256")


def entropy_details(image: Image.Image, bins: int = DEFAULT_BINS) -> dict:
    """Trace a nonempty region: counts -> p -> p*log2(p) -> H -> H/log2(B).

    Convert to uint8 grayscale L before quantization. Python integers avoid
    overflow in pixel*B; for pixel in [0,255], pixel*B//256 is always in [0,B-1].
    No clipping. Exactly equiprobable counts use their analytical H=log2(B)
    to avoid a floating summation overshoot rejected by 05's strict grid checks.
    """
    validate_bins(bins)
    pixels = image.convert("L").tobytes()
    if not pixels:
        raise ValueError("Entropy requires a nonempty region")
    counts = [0] * bins
    for pixel in pixels:
        counts[pixel * bins // 256] += 1
    probabilities = [count / len(pixels) for count in counts]
    terms = [p * math.log2(p) if p else 0.0 for p in probabilities]
    entropy_bits = math.log2(bins) if len(set(counts)) == 1 else e1(counts, len(pixels))
    return {"counts": counts, "probabilities": probabilities,
            "p_log2_p": terms, "entropy_bits": entropy_bits,
            "normalized_entropy": entropy_bits / math.log2(bins)}
