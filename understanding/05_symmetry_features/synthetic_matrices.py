"""Hand-specified entropy grids, not entropies estimated from images."""
from symmetry_features import Grid


def controlled_matrices() -> dict[str, Grid]:
    """Return cases A-E and the manual 2x2 example, with values in [0,1]."""
    lr = [[.1, .8, .8, .1], [.2, .7, .7, .2],
          [.4, .3, .3, .4], [.9, .5, .5, .9]]
    return {
        "A_uniform": [[.5, .5], [.5, .5]],
        "B_lr_only": lr,
        "C_tb_only": [list(row) for row in zip(*lr)],
        "D_rot180_only": [[.1, .2, .3, .4], [.5, .6, .7, .8],
                          [.8, .7, .6, .5], [.4, .3, .2, .1]],
        "E_asymmetric": [[.1, .7], [.2, .4]],
    }


def progressive_defects() -> dict[float, Grid]:
    """Case F: change only E[0][0] from .5 to .5+delta in a uniform 4x4 grid."""
    cases = {}
    for delta in (0.0, .05, .10, .20, .40):
        grid = [[.5] * 4 for _ in range(4)]
        grid[0][0] += delta
        cases[delta] = grid
    return cases
