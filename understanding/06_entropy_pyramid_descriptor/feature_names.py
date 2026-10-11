"""Frozen x94 schema: explicit scale, row, column and transform order."""
LEVELS = (1, 2, 4, 8)
ASYMMETRY_LEVELS = (2, 4, 8)
ASYMMETRY_MEASURES = ("lr", "tb", "rot180")
ENTROPY_COORDINATES = tuple((g, r, c) for g in LEVELS
                            for r in range(g) for c in range(g))
ASYMMETRY_KEYS = tuple(f"{measure}_{g}x{g}" for g in ASYMMETRY_LEVELS
                       for measure in ASYMMETRY_MEASURES)
ENTROPY_NAMES = tuple(f"entropy_{g}x{g}_r{r}_c{c}" for g, r, c in ENTROPY_COORDINATES)
ASYMMETRY_NAMES = tuple(f"asym_{key}" for key in ASYMMETRY_KEYS)


def get_entropy_feature_names() -> list[str]:
    """Return a fresh list of 85 names, scales 1,2,4,8 in row-major order."""
    return list(ENTROPY_NAMES)


def get_asymmetry_feature_names() -> list[str]:
    """Return nine names, LR/TB/R180 at each scale 2,4,8."""
    return list(ASYMMETRY_NAMES)


def get_feature_names() -> list[str]:
    """Return the complete frozen 94-name schema, entropy then asymmetry."""
    return get_entropy_feature_names() + get_asymmetry_feature_names()
