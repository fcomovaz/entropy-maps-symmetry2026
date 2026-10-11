"""Canonical pipeline: image -> four entropy grids -> 85 entropies + 9 A."""
from dataclasses import dataclass
from importlib import import_module
from os import PathLike

from PIL import Image

from .entropy import DEFAULT_BINS, entropy_details, validate_bins
from .feature_names import ASYMMETRY_KEYS, ENTROPY_COORDINATES, LEVELS, get_feature_names
from .partition import partition_boxes

compute_nine_asymmetry_features = import_module(
    "understanding.05_symmetry_features.symmetry_features"
).compute_nine_asymmetry_features


@dataclass
class DescriptorResult:
    """Inspectable x94 result; grids[g] has shape [g,g], vector has length 94."""
    vector: list[float]
    feature_names: list[str]
    entropy_features: list[float]
    asymmetry_features: list[float]
    entropy_grids: dict[int, list[list[float]]]
    image_width: int
    image_height: int
    bins: int


def load_grayscale(image: Image.Image | str | PathLike[str]) -> Image.Image:
    """Return an independent L image; close files opened here, leave caller intact.

    Pillow convert('L') defines color conversion. No EXIF transpose, alpha
    compositing, resizing, intensity rescaling or normalization is applied.
    """
    if isinstance(image, Image.Image):
        return image.convert("L")
    with Image.open(image) as source:
        return source.convert("L")


def extract_descriptor(
    image: Image.Image | str | PathLike[str], bins: int = DEFAULT_BINS,
) -> DescriptorResult:
    """Extract x94 from an image or path; width and height must each be >= 8.

    Each level recomputes region histograms, never averages child entropies.
    Ordering is selected by the explicit schema, never dictionary iteration.
    """
    validate_bins(bins)
    gray = load_grayscale(image)
    width, height = gray.size
    if width < 8 or height < 8:
        raise ValueError("Image width and height must each be at least 8")

    grids = {}
    for g in LEVELS:
        values = [entropy_details(gray.crop(box), bins)["normalized_entropy"]
                  for box in partition_boxes(width, height, g)]
        grids[g] = [values[r * g:(r + 1) * g] for r in range(g)]

    entropies = [grids[g][r][c] for g, r, c in ENTROPY_COORDINATES]
    asymmetry_by_key = compute_nine_asymmetry_features(grids[2], grids[4], grids[8])
    asymmetries = [asymmetry_by_key[key] for key in ASYMMETRY_KEYS]
    return DescriptorResult(entropies + asymmetries, get_feature_names(),
                            entropies, asymmetries, grids, width, height, bins)


def extract_descriptor_vector(
    image: Image.Image | str | PathLike[str], bins: int = DEFAULT_BINS,
) -> list[float]:
    """Return only the 94 floats in the frozen feature order."""
    return extract_descriptor(image, bins).vector
