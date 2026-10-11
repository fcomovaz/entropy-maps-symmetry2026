"""Small reproducible inputs; reuse the established entropy-layout cases of 05."""
from importlib import import_module
import random

from PIL import Image

SEED = 20261001


def worked_image() -> Image.Image:
    """16x16: binary 1-pixel checkerboard in upper-left 8x8, zero elsewhere."""
    return Image.frombytes("L", (16, 16), bytes(
        255 * ((x + y) % 2) if x < 8 and y < 8 else 0
        for y in range(16) for x in range(16)))


def generate_examples(seed: int = SEED) -> dict[str, Image.Image]:
    """Return constant, IID noise, localized texture, symmetry, asymmetry, manual."""
    rng = random.Random(seed)
    noise = Image.frombytes("L", (256, 256), bytes(rng.randrange(256) for _ in range(256**2)))
    localized = Image.new("L", (256, 256))
    localized.paste(noise.crop((0, 0, 32, 32)), (64, 96))  # g=8: row 3, col 2
    previous = import_module("understanding.05_symmetry_features.synthetic_images").generate_images(seed)
    return {"uniform": Image.new("L", (256, 256), 91), "iid_noise": noise,
            "localized_texture": localized,
            "symmetric_layout": previous["image_01_full_symmetry"],
            "asymmetric_layout": previous["image_05_asymmetric"],
            "worked_16x16": worked_image()}
