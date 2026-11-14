"""Shared MSVC C99 toolchain discovery and compilation helpers.

Reuses the verified vswhere / vcvars64 discovery helpers in `src.features.build_native`
without duplicating compiler lookup logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from typing import Sequence

from configs.logging import logger
from src.features.build_native import _query_cl_version, _resolve_toolchain


@dataclass(frozen=True, slots=True)
class CExecutableBuildResult:
    """Metadata returned after compiling a standalone C99 desktop executable."""

    built_at_utc: str
    exe_path: Path
    source_files: tuple[str, ...]
    include_dirs: tuple[str, ...]
    compiler_path: str
    compiler_version: str
    compiler_flags: tuple[str, ...]
    vcvars64_path: str | None


def compile_c_executable(
    sources: Sequence[Path],
    include_dirs: Sequence[Path],
    exe_path: Path,
    build_dir: Path,
    c_standard_flag: str = "/std:c11",
    compiler_flags: Sequence[str] = ("/O2", "/W4", "/D_CRT_SECURE_NO_WARNINGS"),
    vswhere_hint: str = "",
    timeout_s: int = 180,
) -> CExecutableBuildResult:
    """Compile a list of C sources into `exe_path` using MSVC `cl.exe`."""
    resolved_sources: list[Path] = []
    for src in sources:
        p = src.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"C source file not found: {p}")
        resolved_sources.append(p)

    resolved_includes = [inc.resolve() for inc in include_dirs]
    if os.name == "nt" and exe_path.suffix.lower() != ".exe":
        exe_path = Path(f"{exe_path}.exe")
    build_dir.mkdir(parents=True, exist_ok=True)
    exe_path.parent.mkdir(parents=True, exist_ok=True)

    cl_exe, build_env, vcvars64_used = _resolve_toolchain(vswhere_hint, scratch_dir=build_dir)
    cl_version = _query_cl_version(cl_exe, build_env)

    obj_prefix = str(build_dir.resolve()) + os.sep
    cmd = [
        cl_exe,
        "/nologo",
        c_standard_flag,
        *compiler_flags,
        *(f"/I{inc}" for inc in resolved_includes),
        *(str(p) for p in resolved_sources),
        f"/Fo{obj_prefix}",
        f"/Fe{exe_path.resolve()}",
    ]

    logger.info("Compiling C99 executable %s (%d sources)...", exe_path.name, len(resolved_sources))
    completed = subprocess.run(
        cmd,
        cwd=str(build_dir),
        env=build_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )
    if completed.returncode != 0 or not exe_path.is_file():
        output_msg = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        raise RuntimeError(f"MSVC compilation failed (exit={completed.returncode}):\n{output_msg}")

    return CExecutableBuildResult(
        built_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        exe_path=exe_path.resolve(),
        source_files=tuple(str(p) for p in resolved_sources),
        include_dirs=tuple(str(p) for p in resolved_includes),
        compiler_path=cl_exe,
        compiler_version=cl_version,
        compiler_flags=tuple(compiler_flags),
        vcvars64_path=vcvars64_used,
    )
