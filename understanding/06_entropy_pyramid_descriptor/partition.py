"""Proportional, half-open image regions; no DROP, padding or resizing."""


def partition_boxes(width: int, height: int, g: int) -> list[tuple[int, int, int, int]]:
    """Return g*g (x0,y0,x1,y1) boxes in row-major order, covering every pixel.

    Boundaries are floor(k*dimension/g), calculated using exact integer division.
    Dimensions must be >= g so every region is nonempty.
    """
    if any(type(value) is not int for value in (width, height, g)):
        raise ValueError("width, height and g must be integers")
    if g < 1 or width < g or height < g:
        raise ValueError("Require g >= 1 and width, height >= g")
    return [(c * width // g, r * height // g,
             (c + 1) * width // g, (r + 1) * height // g)
            for r in range(g) for c in range(g)]
