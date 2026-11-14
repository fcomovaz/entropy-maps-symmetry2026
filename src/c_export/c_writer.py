"""Deterministic C99 header and source code emitter.

Generates reproducible headers and C sources with provenance metadata, include guards,
compact hex formatting for large byte/weight arrays, and atomic temp-file swaps.
No wall-clock timestamps are written into generated C content so repeated exports produce
byte-identical outputs and identical SHA-256 checksums.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    """Return lowercase hexadecimal SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Return lowercase hexadecimal SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def format_c_float(value: float) -> str:
    """Format a finite float as a valid C99 float literal with an 'f' suffix."""
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"Cannot emit non-finite float literal in C99: {value}")
    s = f"{v:.9g}"
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return f"{s}f"


def format_c_array(
    name: str,
    values: Sequence[int | float] | np.ndarray,
    c_type: str,
    per_line: int = 12,
    hex_int8: bool = False,
) -> str:
    """Format a 1D or flattened sequence as a deterministic `static const` C99 array."""
    arr = np.asarray(values).ravel()
    lines = [f"static const {c_type} {name}[{arr.size}] = {{"]
    if arr.size == 0:
        lines.append("    0")
    else:
        for i in range(0, arr.size, per_line):
            chunk = arr[i : i + per_line]
            if c_type == "float":
                formatted = ", ".join(format_c_float(float(v)) for v in chunk)
            elif c_type in ("uint8_t",) or (hex_int8 and c_type in ("uint16_t", "uint32_t")):
                formatted = ", ".join(f"0x{int(v):02x}U" if c_type == "uint8_t" else f"0x{int(v):x}U" for v in chunk)
            elif c_type.startswith("uint"):
                formatted = ", ".join(f"{int(v)}U" for v in chunk)
            elif c_type in ("int64_t", "q63_t"):
                formatted = ", ".join(f"{int(v)}LL" for v in chunk)
            else:
                formatted = ", ".join(str(int(v)) for v in chunk)
            comma = "," if (i + per_line) < arr.size else ""
            lines.append(f"    {formatted}{comma}")
    lines.append("};")
    return "\n".join(lines)


def write_header(
    path: Path,
    guard: str,
    includes: Iterable[str],
    body: str,
    provenance_lines: Sequence[str] = (),
) -> str:
    """Write a C99 header atomically with deterministic formatting and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header_comments = [
        "/*",
        " * AUTO-GENERATED FILE - DO NOT EDIT BY HAND.",
    ]
    for line in provenance_lines:
        header_comments.append(f" * {line}")
    header_comments.append(" */")

    inc_lines = [f"#include {inc}" for inc in includes]
    parts = [
        "\n".join(header_comments),
        f"#ifndef {guard}",
        f"#define {guard}",
    ]
    if inc_lines:
        parts.append("\n".join(inc_lines))
    parts.append(body.rstrip())
    parts.append(f"#endif /* {guard} */")

    content = "\n\n".join(parts) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8", newline="\n")
    tmp_path.replace(path)
    return sha256_bytes(content.encode("utf-8"))


def write_c_source(
    path: Path,
    includes: Iterable[str],
    body: str,
    provenance_lines: Sequence[str] = (),
) -> str:
    """Write a C99 source file atomically with deterministic formatting and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header_comments = [
        "/*",
        " * AUTO-GENERATED FILE - DO NOT EDIT BY HAND.",
    ]
    for line in provenance_lines:
        header_comments.append(f" * {line}")
    header_comments.append(" */")
    inc_lines = [f"#include {inc}" for inc in includes]
    parts = ["\n".join(header_comments)]
    if inc_lines:
        parts.append("\n".join(inc_lines))
    parts.append(body.rstrip())
    content = "\n\n".join(parts) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8", newline="\n")
    tmp_path.replace(path)
    return sha256_bytes(content.encode("utf-8"))
