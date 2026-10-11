"""Single-variable perturbations with explicit uint8 rounding and boundaries."""
import random
from PIL import Image, ImageFilter


def uint8(value: float) -> int:
    """Round with Python round (ties to even), then clamp to [0,255]."""
    return max(0, min(255, round(value)))


def intensity(image: Image.Image, offset: float = 0, factor: float = 1) -> Image.Image:
    """Contrast about fixed 127.5; additive offset applied afterward."""
    return image.point([uint8(127.5 + factor*(v-127.5) + offset) for v in range(256)])


def noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    """Same unit Gaussian realization per seed, scaled by sigma before rounding."""
    rng = random.Random(seed)
    return Image.frombytes("L", image.size, bytes(
        uint8(v + sigma*rng.gauss(0, 1)) for v in image.tobytes()))


def blur(image: Image.Image, radius: float) -> Image.Image:
    """Pillow GaussianBlur (its native boundary treatment)."""
    return image.filter(ImageFilter.GaussianBlur(radius))


def insert_defect(image: Image.Image, texture: Image.Image, x: int, y: int,
                  strength: float = 1) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Blend one fixed target into a half-open box, no implicit cropping at edges."""
    box = (x, y, x+texture.width, y+texture.height)
    if not (0 <= strength <= 1 and x >= 0 and y >= 0 and box[2] <= image.width and box[3] <= image.height):
        raise ValueError("Defect outside image or invalid strength")
    old = image.crop(box).tobytes()
    pixels = bytes(uint8(a + strength*(b-a)) for a, b in zip(old, texture.tobytes(), strict=True))
    result = image.copy()
    result.paste(Image.frombytes("L", texture.size, pixels), (x, y))
    return result, box


def translate(image: Image.Image, dx: int, dy: int) -> Image.Image:
    """Translate right/down for positive offsets; zero padding, discard overflow."""
    result = Image.new("L", image.size, 0)
    result.paste(image, (dx, dy))
    return result
