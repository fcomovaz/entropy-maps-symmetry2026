"""Asymmetry of normalized entropy layouts, never of image pixels.

All inputs are nonempty square grids with finite values in [0, 1].
The entrywise L1 norm counts both members of each transformed pair.
"""
from math import fsum, isfinite

Grid = list[list[float]]


def _validate(grid: Grid, size: int | None = None) -> int:
    g = len(grid)
    if g == 0 or any(len(row) != g for row in grid):
        raise ValueError("Expected a nonempty square entropy grid [g, g]")
    if size is not None and g != size:
        raise ValueError(f"Expected {size}x{size} entropy grid; got {g}x{g}")
    if any(not isfinite(x) or not 0 <= x <= 1 for row in grid for x in row):
        raise ValueError("Entropy values must be finite and in [0, 1]; no clipping")
    return g


def mirror_left_right(grid: Grid) -> Grid:
    """Reflect columns: output[i][j] = grid[i][g-1-j].

    Args: grid, normalized square entropy matrix [g, g].
    Returns: new [g, g] matrix in [0, 1], reflected about the vertical axis.
    """
    _validate(grid)
    return [list(reversed(row)) for row in grid]


def mirror_top_bottom(grid: Grid) -> Grid:
    """Reflect rows: output[i][j] = grid[g-1-i][j].

    Args: grid, normalized square entropy matrix [g, g].
    Returns: new [g, g] matrix in [0, 1], reflected about the horizontal axis.
    """
    _validate(grid)
    return [row.copy() for row in reversed(grid)]


def rotate_180(grid: Grid) -> Grid:
    """Rotate both indices: output[i][j] = grid[g-1-i][g-1-j].

    Args: grid, normalized square entropy matrix [g, g].
    Returns: new [g, g] matrix in [0, 1], rotated 180 degrees.
    """
    _validate(grid)
    return [list(reversed(row)) for row in reversed(grid)]


def absolute_difference(grid: Grid, transformed: Grid) -> Grid:
    """For two equal-size normalized grids return |grid-transformed| in [0,1]."""
    g = _validate(grid)
    _validate(transformed, g)
    return [[abs(grid[i][j] - transformed[i][j]) for j in range(g)]
            for i in range(g)]


def _mean_difference(grid: Grid, transformed: Grid) -> float:
    differences = absolute_difference(grid, transformed)
    return fsum(x for row in differences for x in row) / len(grid)**2


def reflection_asymmetry_lr(grid: Grid) -> float:
    """Measure entropy-layout left-right asymmetry: sum|E[i,j]-E[i,g-1-j]|/g².

    Args: grid, normalized square entropy matrix [g, g].
    Returns: mean absolute residual in [0,1]; zero means exact LR symmetry.
    """
    return _mean_difference(grid, mirror_left_right(grid))


def reflection_asymmetry_tb(grid: Grid) -> float:
    """Measure entropy-layout top-bottom asymmetry: sum|E[i,j]-E[g-1-i,j]|/g².

    Args: grid, normalized square entropy matrix [g, g].
    Returns: mean absolute residual in [0,1]; zero means exact TB symmetry.
    """
    return _mean_difference(grid, mirror_top_bottom(grid))


def rot180_asymmetry(grid: Grid) -> float:
    """Measure entropy-layout rotational asymmetry: sum|E[i,j]-E[g-1-i,g-1-j]|/g².

    Args: grid, normalized square entropy matrix [g, g].
    Returns: mean absolute residual in [0,1]; zero means exact 180° symmetry.
    """
    return _mean_difference(grid, rotate_180(grid))


def asymmetry_lr_2x2(grid: Grid) -> float:
    """LR residual sum|E-LR(E)|/4. Args: grid [2,2] in [0,1]. Returns: A in [0,1]."""
    _validate(grid, 2)
    return reflection_asymmetry_lr(grid)


def asymmetry_tb_2x2(grid: Grid) -> float:
    """TB residual sum|E-TB(E)|/4. Args: grid [2,2] in [0,1]. Returns: A in [0,1]."""
    _validate(grid, 2)
    return reflection_asymmetry_tb(grid)


def asymmetry_rot180_2x2(grid: Grid) -> float:
    """R180 residual sum|E-R180(E)|/4. Args: grid [2,2] in [0,1]. Returns: A in [0,1]."""
    _validate(grid, 2)
    return rot180_asymmetry(grid)


def asymmetry_lr_4x4(grid: Grid) -> float:
    """LR residual sum|E-LR(E)|/16. Args: grid [4,4] in [0,1]. Returns: A in [0,1]."""
    _validate(grid, 4)
    return reflection_asymmetry_lr(grid)


def asymmetry_tb_4x4(grid: Grid) -> float:
    """TB residual sum|E-TB(E)|/16. Args: grid [4,4] in [0,1]. Returns: A in [0,1]."""
    _validate(grid, 4)
    return reflection_asymmetry_tb(grid)


def asymmetry_rot180_4x4(grid: Grid) -> float:
    """R180 residual sum|E-R180(E)|/16. Args: grid [4,4] in [0,1]. Returns: A in [0,1]."""
    _validate(grid, 4)
    return rot180_asymmetry(grid)


def asymmetry_lr_8x8(grid: Grid) -> float:
    """LR residual sum|E-LR(E)|/64. Args: grid [8,8] in [0,1]. Returns: A in [0,1]."""
    _validate(grid, 8)
    return reflection_asymmetry_lr(grid)


def asymmetry_tb_8x8(grid: Grid) -> float:
    """TB residual sum|E-TB(E)|/64. Args: grid [8,8] in [0,1]. Returns: A in [0,1]."""
    _validate(grid, 8)
    return reflection_asymmetry_tb(grid)


def asymmetry_rot180_8x8(grid: Grid) -> float:
    """R180 residual sum|E-R180(E)|/64. Args: grid [8,8] in [0,1]. Returns: A in [0,1]."""
    _validate(grid, 8)
    return rot180_asymmetry(grid)


def compute_nine_asymmetry_features(
    grid_2x2: Grid, grid_4x4: Grid, grid_8x8: Grid,
) -> dict[str, float]:
    """Compute sum|E-T(E)|/g² for T=LR,TB,R180 and g=2,4,8.

    Args: three normalized entropy grids of exactly the named shapes.
    Returns: nine A values in [0,1], ordered by scale then LR, TB, R180.
    Python 3.11 dictionaries preserve this insertion order. No 1x1 extras.
    """
    return {
        "lr_2x2": asymmetry_lr_2x2(grid_2x2),
        "tb_2x2": asymmetry_tb_2x2(grid_2x2),
        "rot180_2x2": asymmetry_rot180_2x2(grid_2x2),
        "lr_4x4": asymmetry_lr_4x4(grid_4x4),
        "tb_4x4": asymmetry_tb_4x4(grid_4x4),
        "rot180_4x4": asymmetry_rot180_4x4(grid_4x4),
        "lr_8x8": asymmetry_lr_8x8(grid_8x8),
        "tb_8x8": asymmetry_tb_8x8(grid_8x8),
        "rot180_8x8": asymmetry_rot180_8x8(grid_8x8),
    }
