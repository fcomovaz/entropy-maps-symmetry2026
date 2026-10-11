"""Grayscale fixtures from stdlib/Pillow; geometry is evaluated natively."""
import math
import random
from PIL import Image


def textured_scene(size: int) -> Image.Image:
    """Native normalized-coordinate scene, finite sampling of one fixed function.

    Smooth illumination plus periodic texture; asymmetric textured rectangle.
    Frequencies and rectangle proportions stay constant across resolutions.
    """
    values = []
    for y in range(size):
        v = (y + .5) / size
        for x in range(size):
            u = (x + .5) / size
            value = 118 + 48*math.sin(2*math.pi*12*u)*math.cos(2*math.pi*8*v) + 48*u
            if .18 <= u < .43 and .25 <= v < .62:
                value += 24*math.sin(2*math.pi*32*v)
            values.append(max(0, min(255, round(value))))
    return Image.frombytes("L", (size, size), bytes(values))


def checkerboard(size: int, cell: int = 2) -> Image.Image:
    """Balanced 80/176 texture when size is a multiple of 2*cell."""
    return Image.frombytes("L", (size, size), bytes(
        80 + 96*((x//cell + y//cell) % 2) for y in range(size) for x in range(size)))


def noise_texture(size: int, seed: int) -> Image.Image:
    """One reproducible IID uint8 target; crops use the same realization."""
    rng = random.Random(seed)
    return Image.frombytes("L", (size, size), bytes(rng.randrange(256) for _ in range(size*size)))


def equal_histogram_pair(width: int, height: int) -> tuple[Image.Image, Image.Image]:
    """Same exact multiset: alternating columns versus sorted row-major pixels.

    Sorting preserves counts even for odd dimensions; the second has a horizontal
    division (plus at most one partially filled row).
    """
    pixels = bytes(255*(x % 2) for y in range(height) for x in range(width))
    return (Image.frombytes("L", (width, height), pixels),
            Image.frombytes("L", (width, height), bytes(sorted(pixels))))


def cellwise_collision(size: int) -> tuple[Image.Image, Image.Image]:
    """Different pixel structures, exact same histogram in EVERY finest cell."""
    from compare_descriptors import partition_boxes
    a, b = Image.new("L", (size, size)), Image.new("L", (size, size))
    for x0, y0, x1, y1 in partition_boxes(size, size, 8):
        first, second = equal_histogram_pair(x1-x0, y1-y0)
        a.paste(first, (x0, y0))
        b.paste(second, (x0, y0))
    return a, b
