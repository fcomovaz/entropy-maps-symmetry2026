"""Pure mathematical and vectorized operations for spatial entropy asymmetry features.

Computes multiscale spatial entropy asymmetry measurements (reflection LR,
reflection TB, and 180-degree rotational asymmetry) directly from normalized
entropy grids, matching the canonical reference oracle in understanding/05 and
understanding/06 while providing high-throughput batch vectorization.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

ALLOWED_ASYMMETRY_KINDS: tuple[str, ...] = ("lr", "tb", "rot180")
DEFAULT_ASYMMETRY_TEMPLATES: dict[str, str] = {
    "lr": "asym_lr_{g}x{g}",
    "tb": "asym_tb_{g}x{g}",
    "rot180": "asym_rot180_{g}x{g}",
}


def validate_scales(scales: Sequence[int], global_scale: int = 1) -> tuple[int, ...]:
    """Validate spatial asymmetry grid scales against positivity, uniqueness, and global scale.

    Args:
        scales: Sequence of integer scale factors g representing g x g region partitions.
        global_scale: Integer scale of the global entropy partition (e.g. 1) to exclude.

    Returns:
        Validated tuple of integer grid scales in strictly ascending order.

    Raises:
        ValueError: If scales is empty, contains non-integers, booleans, values < 2,
            duplicates, unsorted order, or contains global_scale.
    """
    if not scales:
        raise ValueError("asymmetry scales cannot be empty")

    for g in scales:
        if type(g) is not int or isinstance(g, bool):
            raise ValueError(f"scale '{g}' must be an integer, got {type(g).__name__}")
        if g == global_scale:
            raise ValueError(
                f"Global scale {global_scale} must not be included in asymmetry scales "
                "(asymmetry residuals are identically 0)"
            )
        if g < 2:
            raise ValueError(f"asymmetry scale must be >= 2, got {g}")

    if len(set(scales)) != len(scales):
        raise ValueError(f"scales contains duplicate levels: {scales}")

    if list(scales) != sorted(scales):
        raise ValueError(f"scales must be in strictly ascending order: {scales}")

    return tuple(int(g) for g in scales)


def validate_kinds(kinds: Sequence[str]) -> tuple[str, ...]:
    """Validate asymmetry measurement kinds against the allowed canonical set.

    Args:
        kinds: Sequence of asymmetry measurement names (e.g. ["lr", "tb", "rot180"]).

    Returns:
        Validated tuple of asymmetry kind strings.

    Raises:
        ValueError: If kinds is empty, contains unknown kinds, or contains duplicates.
    """
    if not kinds:
        raise ValueError("asymmetry kinds cannot be empty")

    for k in kinds:
        if not isinstance(k, str):
            raise ValueError(f"asymmetry kind must be a string, got {type(k).__name__}")
        if k not in ALLOWED_ASYMMETRY_KINDS:
            raise ValueError(
                f"Unknown asymmetry kind '{k}'; must be one of {ALLOWED_ASYMMETRY_KINDS}"
            )

    if len(set(kinds)) != len(kinds):
        raise ValueError(f"asymmetry kinds contains duplicates: {kinds}")

    return tuple(kinds)


def asymmetry_column_names(
    scales: Sequence[int],
    kinds: Sequence[str] = ALLOWED_ASYMMETRY_KINDS,
    templates: Mapping[str, str] | None = None,
) -> list[str]:
    """Generate ordered canonical asymmetry feature column names across configured scales and kinds.

    Args:
        scales: Sequence of spatial scales g (e.g. [2, 4, 8]).
        kinds: Sequence of asymmetry measures in order (e.g. ["lr", "tb", "rot180"]).
        templates: Optional mapping from kind to format template string supporting '{g}'.

    Returns:
        List of formatted column names in canonical order (e.g. asym_lr_2x2, asym_tb_2x2, ...).
    """
    active_templates = dict(DEFAULT_ASYMMETRY_TEMPLATES)
    if templates is not None:
        active_templates.update(templates)

    names: list[str] = []
    for g in scales:
        for k in kinds:
            template = active_templates[k]
            names.append(template.format(g=g))
    return names


def entropy_column_names(
    scale: int,
    template: str = "E_{g}x{g}_r{r}_c{c}",
) -> list[str]:
    """Generate ordered row-major entropy column names for a single spatial scale.

    Args:
        scale: Integer grid scale factor g.
        template: Format string template supporting '{g}', '{r}', and '{c}'.

    Returns:
        List of formatted column names in row-major order: r0c0, r0c1, ..., r{g-1}c{g-1}.
    """
    names: list[str] = []
    for r in range(scale):
        for c in range(scale):
            names.append(template.format(g=scale, r=r, c=c))
    return names


def reconstruct_grids(flat_values: np.ndarray, g: int) -> np.ndarray:
    """Reconstruct 2D row-major entropy matrices from flattened tabular features.

    Args:
        flat_values: 1D array of shape [g * g] or 2D array of shape [N, g * g] containing
            normalized entropy values in row-major order.
        g: Integer spatial scale factor.

    Returns:
        3D float64 array of shape [N, g, g] representing reconstructed entropy grids.

    Raises:
        ValueError: If array dimensions do not match g * g.
    """
    arr = np.asarray(flat_values, dtype=np.float64)
    expected_dim = g * g

    if arr.ndim == 1:
        if arr.shape[0] != expected_dim:
            raise ValueError(
                f"Expected 1D array of length {expected_dim} for scale {g}x{g}, got {arr.shape[0]}"
            )
        return arr.reshape(1, g, g)
    elif arr.ndim == 2:
        if arr.shape[1] != expected_dim:
            raise ValueError(
                f"Expected 2D array of shape [N, {expected_dim}] for scale {g}x{g}, got shape {arr.shape}"
            )
        return arr.reshape(-1, g, g)
    else:
        raise ValueError(f"Expected 1D or 2D array, got ndim={arr.ndim}")


def validate_entropy_values(values: np.ndarray, tolerance: float = 1e-6) -> None:
    """Defensively validate that entropy features are finite and within [0, 1 + tolerance].

    No clipping is applied; any value outside the valid range or containing NaN/Inf
    raises ValueError.

    Args:
        values: NumPy array of entropy values.
        tolerance: Allowed floating-point tolerance above 1.0 due to decimal rounding.

    Raises:
        ValueError: If any value is non-finite or outside [0.0, 1.0 + tolerance].
    """
    if values.size == 0:
        return

    if not np.all(np.isfinite(values)):
        raise ValueError("Entropy values must be finite numbers; NaN or Inf detected")

    min_val = float(np.min(values))
    max_val = float(np.max(values))

    if min_val < 0.0:
        raise ValueError(
            f"Entropy values must be >= 0.0; found minimum value {min_val:.8f}. "
            "No clipping is applied."
        )

    upper_bound = 1.0 + tolerance
    if max_val > upper_bound:
        raise ValueError(
            f"Entropy values must be <= {upper_bound:.8f}; found maximum value {max_val:.8f}. "
            "No clipping is applied."
        )


def compute_asymmetry_for_grid(
    grids: np.ndarray,
    kinds: Sequence[str] = ALLOWED_ASYMMETRY_KINDS,
    tolerance: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Compute multiscale spatial entropy asymmetry measurements from 3D grids.

    Computes:
        A_LR:   mean absolute difference across the vertical reflection axis (columns flipped).
        A_TB:   mean absolute difference across the horizontal reflection axis (rows flipped).
        A_R180: mean absolute difference across 180-degree rotation (both axes flipped).

    Args:
        grids: 3D float64 array of shape [N, g, g] containing normalized entropy grids.
        kinds: Sequence of asymmetry measurement names to compute.
        tolerance: Numerical tolerance for validating entropy values.

    Returns:
        Dictionary mapping kind string ('lr', 'tb', 'rot180') to 1D float64 array of shape [N].

    Raises:
        ValueError: If grids is not 3D, is empty, is non-square, or values violate range.
    """
    if grids.ndim != 3:
        raise ValueError(f"Expected 3D array of shape [N, g, g], got ndim={grids.ndim}")
    _, rows, cols = grids.shape
    if rows != cols:
        raise ValueError(f"Expected square entropy grids [N, g, g], got shape {grids.shape}")
    if rows < 2:
        raise ValueError(f"Entropy grid size must be >= 2, got {rows}")

    validate_entropy_values(grids, tolerance=tolerance)

    results: dict[str, np.ndarray] = {}
    for k in kinds:
        if k == "lr":
            # Reflection across columns: grid[..., r, c] vs grid[..., r, g-1-c]
            mirrored_lr = grids[:, :, ::-1]
            results["lr"] = np.mean(np.abs(grids - mirrored_lr), axis=(1, 2))
        elif k == "tb":
            # Reflection across rows: grid[..., r, c] vs grid[..., g-1-r, c]
            mirrored_tb = grids[:, ::-1, :]
            results["tb"] = np.mean(np.abs(grids - mirrored_tb), axis=(1, 2))
        elif k == "rot180":
            # 180-degree rotation: grid[..., r, c] vs grid[..., g-1-r, g-1-c]
            rotated_180 = grids[:, ::-1, ::-1]
            results["rot180"] = np.mean(np.abs(grids - rotated_180), axis=(1, 2))
        else:
            raise ValueError(f"Unsupported asymmetry kind '{k}'")

    return results


def compute_asymmetry_features(
    entropy_by_scale: Mapping[int, np.ndarray],
    scales: Sequence[int],
    kinds: Sequence[str] = ALLOWED_ASYMMETRY_KINDS,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Compute concatenated asymmetry feature array across spatial scales in canonical order.

    Canonical order:
        For each scale g in scales (e.g. 2, 4, 8):
            For each kind k in kinds (e.g. 'lr', 'tb', 'rot180'):
                column

    Args:
        entropy_by_scale: Mapping from spatial scale g to 2D array of shape [N, g * g]
            containing flattened entropy values in row-major order.
        scales: Sequence of spatial scales g (e.g. [2, 4, 8]).
        kinds: Sequence of asymmetry measurement names (e.g. ['lr', 'tb', 'rot180']).
        tolerance: Numerical tolerance for validating entropy values.

    Returns:
        2D float64 array of shape [N, len(scales) * len(kinds)] containing asymmetry features.

    Raises:
        ValueError: If scales or kinds are invalid, or if batch sizes across scales mismatch.
    """
    valid_scales = validate_scales(scales)
    valid_kinds = validate_kinds(kinds)

    n_samples: int | None = None
    scale_grids: dict[int, np.ndarray] = {}

    for g in valid_scales:
        if g not in entropy_by_scale:
            raise ValueError(f"Missing required scale {g} in entropy_by_scale mapping")
        flat_arr = np.asarray(entropy_by_scale[g], dtype=np.float64)
        grids = reconstruct_grids(flat_arr, g)
        if n_samples is None:
            n_samples = grids.shape[0]
        elif grids.shape[0] != n_samples:
            raise ValueError(
                f"Batch size mismatch across scales: expected {n_samples}, "
                f"got {grids.shape[0]} at scale {g}"
            )
        scale_grids[g] = grids

    if n_samples is None:
        raise ValueError("No scales provided to compute asymmetry features")

    total_features = len(valid_scales) * len(valid_kinds)
    out = np.zeros((n_samples, total_features), dtype=np.float64)

    col_idx = 0
    for g in valid_scales:
        asym_dict = compute_asymmetry_for_grid(
            scale_grids[g], kinds=valid_kinds, tolerance=tolerance
        )
        for k in valid_kinds:
            out[:, col_idx] = asym_dict[k]
            col_idx += 1

    return out
