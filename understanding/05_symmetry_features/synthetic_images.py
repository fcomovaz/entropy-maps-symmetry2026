"""Reproducible grayscale texture layouts using Pillow and scalar Python."""
import random
from PIL import Image

SEED = 20261001
SIZE = 256


def draw_uniform_region(size: int, value: int = 0) -> Image.Image:
    """Return a constant L image [size,size], entropy zero."""
    return Image.new("L", (size, size), value)


def draw_checkerboard_region(size: int) -> Image.Image:
    """Return balanced black/white checks (2-pixel squares for even multiples of 4)."""
    return Image.frombytes("L", (size, size), bytes(
        255 * ((x//2 + y//2) % 2) for y in range(size) for x in range(size)))


def draw_vertical_stripes(size: int, levels: int = 4) -> Image.Image:
    """Return vertical stripes with an equally spaced intensity alphabet."""
    return Image.frombytes("L", (size, size), bytes(
        round(255 * (x % levels)/(levels-1))
        for y in range(size) for x in range(size)))


def draw_horizontal_stripes(size: int, levels: int = 8) -> Image.Image:
    """Return horizontal stripes with an equally spaced intensity alphabet."""
    return Image.frombytes("L", (size, size), bytes(
        round(255 * (y % levels)/(levels-1))
        for y in range(size) for x in range(size)))


def draw_random_noise_region(size: int, seed: int = SEED) -> Image.Image:
    """Return shuffled balanced 16-level noise, repeated in 32x32 tiles.

    Not IID noise: exact counts remove random histogram imbalance from this demo.
    """
    tile_values = [17 * (i % 16) for i in range(32*32)]
    random.Random(seed).shuffle(tile_values)
    tile = Image.frombytes("L", (32, 32), bytes(tile_values))
    region = Image.new("L", (size, size))
    for y in range(0, size, 32):
        for x in range(0, size, 32):
            region.paste(tile, (x, y))
    return region


def generate_images(seed: int = SEED) -> dict[str, Image.Image]:
    """Return six texture-layout cases, a scale pair, and an equal-entropy collision.

    Each image is grayscale 256x256. Labels concern entropy layouts, not pixels.
    Five 64x64 texture types have H/4 = 0,.25,.5,.75,1 at B=16.
    """
    textures = [draw_uniform_region(64), draw_checkerboard_region(64),
                draw_vertical_stripes(64), draw_horizontal_stripes(64),
                draw_random_noise_region(64, seed)]
    lr = [[0, 1, 1, 0], [1, 2, 2, 1], [2, 3, 3, 2], [4, 3, 3, 4]]
    layouts = {
        "image_01_full_symmetry": [[0, 1, 1, 0], [2, 4, 4, 2],
                                   [2, 4, 4, 2], [0, 1, 1, 0]],
        "image_02_lr_symmetry": lr,
        "image_03_tb_symmetry": [list(row) for row in zip(*lr)],
        "image_04_rot180_symmetry": [[0, 1, 2, 3], [1, 2, 3, 4],
                                     [4, 3, 2, 1], [3, 2, 1, 0]],
        "image_05_asymmetric": [[0, 0, 1, 4], [0, 1, 3, 2],
                                [2, 3, 4, 4], [1, 0, 2, 3]],
    }
    images = {}
    for name, layout in layouts.items():
        image = Image.new("L", (SIZE, SIZE))
        for i, row in enumerate(layout):
            for j, texture_id in enumerate(row):
                image.paste(textures[texture_id], (j*64, i*64))
        images[name] = image
    defect = images["image_01_full_symmetry"].copy()
    defect.paste(draw_random_noise_region(32, seed), (0, 0))
    images["image_06_symmetric_plus_defect"] = defect
    baseline = draw_checkerboard_region(SIZE)
    images["image_07_scale_baseline"] = baseline
    scale_defect = baseline.copy()
    scale_defect.paste(draw_uniform_region(32), (32, 32))
    images["image_08_scale_defect"] = scale_defect
    collision = baseline.copy()
    collision.paste(draw_vertical_stripes(32, levels=2), (32, 32))
    images["image_09_equal_entropy_different_structure"] = collision
    return images
