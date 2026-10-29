"""Pure image transformation routines: grayscale conversion and letterboxing to square canvas.

Contains no I/O, no filesystem dependencies, and no configuration loading.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def to_grayscale(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a 2D uint8 NumPy grayscale array [H, W].

    Uses ITU-R 601 luma conversion when converting RGB/RGBA.
    Images already in mode 'L' pass through unmodified.
    """
    if image.mode != "L":
        converted = image.convert("L")
    else:
        converted = image
    return np.asarray(converted, dtype=np.uint8)


def letterbox_to_square(
    image: np.ndarray,
    size: int,
    pad_value: int = 0,
    interpolation: int | Image.Resampling = Image.Resampling.LANCZOS,
    anchor: str = "center",
) -> np.ndarray:
    """Resize and pad a 2D grayscale image/mask to a square canvas [size, size].

    Preserves aspect ratio by scaling the longer side to `size` and centering
    the content along the shorter side, filling the remainder with `pad_value`.
    Upscales images smaller than `size` and downscales images larger than `size`.

    Args:
        image: 2D uint8 NumPy array of shape [H, W].
        size: Target width and height in pixels (must be >= 1).
        pad_value: Pixel value for padding (0 to 255, default 0).
        interpolation: PIL interpolation method or Image.Resampling enum.
        anchor: Content placement anchor. Currently only 'center' is supported.

    Returns:
        Square 2D uint8 NumPy array of shape [size, size].

    Raises:
        ValueError: If size < 1, image is not 2D, pad_value is outside [0, 255],
            or anchor is unsupported.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    if image.ndim != 2:
        raise ValueError(f"image must be 2D [H, W], got shape {image.shape}")
    if not (0 <= pad_value <= 255):
        raise ValueError(f"pad_value must be in range [0, 255], got {pad_value}")
    if anchor != "center":
        raise ValueError(f"Unsupported anchor '{anchor}'; only 'center' is supported")

    height, width = image.shape
    if height == 0 or width == 0:
        raise ValueError(f"Image dimensions must be non-zero, got {(height, width)}")

    scale = size / max(height, width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))

    pil_img = Image.fromarray(image)
    resized_pil = pil_img.resize((new_width, new_height), resample=interpolation)
    resized_arr = np.asarray(resized_pil, dtype=np.uint8)

    canvas = np.full((size, size), fill_value=pad_value, dtype=np.uint8)
    offset_y = (size - new_height) // 2
    offset_x = (size - new_width) // 2

    canvas[offset_y : offset_y + new_height, offset_x : offset_x + new_width] = (
        resized_arr
    )
    return canvas


def invert_intensity(image: np.ndarray) -> np.ndarray:
    """Invert grayscale pixel values: 255 - x.

    Args:
        image: 2D uint8 NumPy array of shape [H, W].

    Returns:
        Inverted 2D uint8 NumPy array of shape [H, W].

    Raises:
        ValueError: If image is not 2D [H, W] or does not have uint8 dtype.
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2D [H, W], got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got dtype {image.dtype}")
    return np.subtract(255, image, dtype=np.uint8)

