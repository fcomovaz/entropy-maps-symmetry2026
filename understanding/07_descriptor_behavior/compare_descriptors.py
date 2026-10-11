"""Canonical imports and inspectable comparisons; no descriptor mathematics here."""
from importlib import import_module
import math
from pathlib import Path
import sys
from collections.abc import Sequence

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[2]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
PACKAGE = "understanding.06_entropy_pyramid_descriptor"
api = import_module(PACKAGE)
schema = import_module(PACKAGE + ".feature_names")
partition_boxes = import_module(PACKAGE + ".partition").partition_boxes
entropy_details = import_module(PACKAGE + ".entropy").entropy_details
NAMES = api.get_feature_names()
BLOCKS = {f"H_{g}x{g}": [i for i, n in enumerate(NAMES) if n.startswith(f"entropy_{g}x{g}_")]
          for g in schema.LEVELS}
BLOCKS.update({f"A_{g}x{g}": [i for i, n in enumerate(NAMES) if n.startswith("asym_") and n.endswith(f"_{g}x{g}")]
               for g in schema.ASYMMETRY_LEVELS})
GROUPS = {"all": list(range(94)), "entropy": list(range(85)), "asymmetry": list(range(85, 94)), **BLOCKS}


def extract(image, bins: int = 16) -> list[float]:
    """Delegate to 06 and fail loudly if its frozen x94 contract changes."""
    result = api.extract_descriptor(image, bins=bins)
    if len(result.vector) != 94 or result.feature_names != NAMES:
        raise ValueError("06 no longer exposes the expected x94 schema")
    if not all(math.isfinite(v) for v in result.vector):
        raise ValueError("Nonfinite descriptor")
    return result.vector


def metrics(x: Sequence[float], y: Sequence[float], tolerance: float = 1e-6) -> dict:
    """Compare equally sized nonempty vectors; changed means strictly > tolerance."""
    if not x or len(x) != len(y):
        raise ValueError("Expected equally sized nonempty vectors")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    if not all(math.isfinite(v) for v in (*x, *y)):
        raise ValueError("Nonfinite input")
    absolute = [abs(b - a) for a, b in zip(x, y, strict=True)]
    l1 = math.fsum(absolute)
    return {"l1": l1, "mad": l1 / len(x), "l2": math.sqrt(math.fsum(d*d for d in absolute)),
            "max_abs": max(absolute), "changed": sum(d > tolerance for d in absolute)}


def compare(x: list[float], y: list[float], tolerance: float = 1e-6) -> dict:
    """All five statistics for x94, entropy-only, asymmetry-only and seven blocks."""
    if len(x) != 94 or len(y) != 94:
        raise ValueError("Expected x94")
    return {f"{group}_{metric}": value for group, indices in GROUPS.items()
            for metric, value in metrics([x[i] for i in indices], [y[i] for i in indices], tolerance).items()}


def top_features(x: list[float], y: list[float], tolerance: float = 1e-6) -> list[dict]:
    """Up to ten genuinely changed dimensions; deterministic ties use schema order."""
    indices = sorted(range(94), key=lambda i: -abs(y[i] - x[i]))
    return [{"index": i, "name": NAMES[i], "base": x[i], "modified": y[i], "delta": y[i]-x[i]}
            for i in indices if abs(y[i]-x[i]) > tolerance][:10]


def locality(size: tuple[int, int], boxes: list[tuple], x: list[float], y: list[float], tolerance: float) -> dict:
    """Compare changed 8x8 cells against all cells intersecting physical defects."""
    expected = []
    for i, (a, b, c, d) in enumerate(partition_boxes(*size, 8)):
        if any(a < q[2] and c > q[0] and b < q[3] and d > q[1] for q in boxes):
            expected.append(NAMES[21+i])
    changed = [NAMES[i] for i in range(21, 85) if abs(y[i]-x[i]) > tolerance]
    most = max(range(21, 85), key=lambda i: abs(y[i]-x[i]))
    detected = abs(y[most]-x[most]) > tolerance
    return {"expected_affected_cells": expected, "changed_entropy_cells": changed,
            "most_affected_entropy_cell": NAMES[most] if detected else None,
            "locality_match": (NAMES[most] in expected and set(changed) <= set(expected)) if detected else None}


def curve_shape(values: list[float], tolerance: float = 1e-6) -> str:
    """Descriptive sampled trend only, not a continuity or significance test."""
    if max(values)-min(values) <= tolerance:
        return "approximately flat"
    steps = [b-a for a, b in zip(values, values[1:])]
    if all(d >= -tolerance for d in steps):
        return "monotonic nondecreasing"
    if all(d <= tolerance for d in steps):
        return "monotonic nonincreasing"
    return "non-monotonic"
