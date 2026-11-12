"""Locate MSVC x64 toolchain via vswhere/vcvars64 and compile the C entropy pyramid DLL."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, LAB_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from configs.logging import configure_logging, logger
from config_loader import EntropyOptimizationConfig, load_optimization_config

ENV_MARKER = "__ENTROPY_PYRAMID_VCVARS_START__"


@dataclass(frozen=True, slots=True)
class NativeBuildResult:
    """Metadata produced by compiling the native C entropy kernel DLL."""

    built_at_utc: str
    dll_path: Path
    implib_path: Path
    build_info_path: Path
    source_files: tuple[str, ...]
    compiler_path: str
    compiler_version: str
    compiler_flags: tuple[str, ...]
    vcvars64_path: str | None


def _expand_path(raw: str) -> Path:
    """Expand Windows environment variables cleanly into a resolved Path."""
    return Path(os.path.expandvars(raw)).expanduser()


def _find_vswhere(vswhere_hint: str) -> Path | None:
    """Locate vswhere.exe using TOML hint, PATH, or standard Visual Studio Installer directories."""
    candidates: list[Path] = []
    if vswhere_hint.strip():
        candidates.append(_expand_path(vswhere_hint))

    which_vswhere = shutil.which("vswhere.exe") or shutil.which("vswhere")
    if which_vswhere:
        candidates.append(Path(which_vswhere))

    for env_key in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(env_key)
        if base:
            candidates.append(Path(base) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def _find_vcvars64(vswhere_hint: str) -> Path | None:
    """Query vswhere for installed MSVC Build Tools / VS instances containing vcvars64.bat."""
    vswhere = _find_vswhere(vswhere_hint)
    if vswhere is None:
        return None

    cmd = [
        str(vswhere),
        "-products",
        "*",
        "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-sort",
        "-property",
        "installationPath",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=15)
    if completed.returncode != 0:
        return None

    for line in completed.stdout.splitlines():
        install_dir = line.strip()
        if not install_dir:
            continue
        vcvars = Path(install_dir) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if vcvars.is_file():
            return vcvars.resolve()
    return None


def _capture_vcvars64_env(vcvars64_path: Path, scratch_dir: Path) -> dict[str, str]:
    """Invoke vcvars64.bat via a temporary batch wrapper to avoid cmd.exe quote escaping bugs."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    helper_bat = scratch_dir / "_capture_vcvars64_env.cmd"
    helper_content = (
        "@echo off\r\n"
        f'call "{vcvars64_path}" >nul\r\n'
        "if errorlevel 1 exit /b %errorlevel%\r\n"
        f"echo {ENV_MARKER}\r\n"
        "set\r\n"
    )
    helper_bat.write_text(helper_content, encoding="utf-8")
    try:
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", str(helper_bat)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    finally:
        if helper_bat.exists():
            helper_bat.unlink()

    if completed.returncode != 0:
        stderr_msg = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"vcvars64.bat failed (exit={completed.returncode}): {stderr_msg}")

    lines = completed.stdout.splitlines()
    try:
        marker_idx = next(i for i, ln in enumerate(lines) if ln.strip() == ENV_MARKER)
    except StopIteration as exc:
        raise RuntimeError(
            f"Could not parse environment from '{vcvars64_path}' output."
        ) from exc

    captured: dict[str, str] = {}
    for ln in lines[marker_idx + 1 :]:
        if "=" not in ln:
            continue
        key, value = ln.split("=", 1)
        if key:
            captured[key] = value
    if not captured:
        raise RuntimeError(f"Empty environment captured from '{vcvars64_path}'.")
    return captured


def _query_cl_version(cl_exe: str, env: dict[str, str]) -> str:
    """Read MSVC cl.exe compiler version banner reliably."""
    completed = subprocess.run(
        [cl_exe],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=15,
    )
    combined = "\n".join(part for part in (completed.stderr, completed.stdout) if part)
    for line in combined.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "MSVC cl.exe (version string unavailable)"


def _resolve_toolchain(
    vswhere_hint: str, scratch_dir: Path
) -> tuple[str, dict[str, str], str | None]:
    """Locate cl.exe either on active PATH or via vswhere + vcvars64.bat."""
    cl_on_path = shutil.which("cl.exe") or shutil.which("cl")
    if cl_on_path:
        env = dict(os.environ)
        return str(Path(cl_on_path).resolve()), env, None

    vcvars64 = _find_vcvars64(vswhere_hint)
    if vcvars64 is None:
        raise RuntimeError(
            "MSVC C/C++ compiler (cl.exe) not found on PATH, and no Visual Studio / Build Tools "
            "installation with VC x86/x64 tools was located via vswhere.exe. "
            "Install 'Desktop development with C++' (MSVC v143 x64/x86 build tools) or run from "
            "an x64 Native Tools Command Prompt."
        )

    vc_env = _capture_vcvars64_env(vcvars64, scratch_dir=scratch_dir)
    path_val = next((v for k, v in vc_env.items() if k.upper() == "PATH"), "")
    cl_resolved = shutil.which("cl.exe", path=path_val)
    if not cl_resolved:
        raise RuntimeError(
            f"Initialized '{vcvars64}', but cl.exe is still missing from the resulting PATH."
        )
    return str(Path(cl_resolved).resolve()), vc_env, str(vcvars64)


def read_build_info(build_dir: Path) -> dict[str, object] | None:
    """Load persisted native build metadata JSON if present."""
    info_path = build_dir / "build_info.json"
    if not info_path.is_file():
        return None
    with info_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data) if isinstance(data, dict) else None


def build_native_library(
    cfg: EntropyOptimizationConfig,
    clean: bool = False,
) -> NativeBuildResult:
    """Compile the C kernel sources defined in TOML into the target shared library."""
    if os.name != "nt":
        raise RuntimeError(
            "build_native.py targets the MSVC Windows toolchain configured in "
            "[entropy_optimization.native]."
        )

    native_cfg = cfg.native
    source_paths: list[Path] = []
    for rel_src in native_cfg.c_sources:
        src_path = (native_cfg.source_dir / rel_src).resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"C source file not found: {src_path}")
        source_paths.append(src_path)

    build_dir = native_cfg.build_dir
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    cl_exe, build_env, vcvars64_used = _resolve_toolchain(
        native_cfg.vswhere_hint, scratch_dir=build_dir
    )
    cl_version = _query_cl_version(cl_exe, build_env)

    dll_path = native_cfg.dll_path
    implib_path = build_dir / f"{native_cfg.library_name}.lib"
    obj_prefix = str(build_dir) + os.sep

    cmd = [
        cl_exe,
        "/nologo",
        "/W4",
        *native_cfg.compiler_flags,
        f"/I{native_cfg.source_dir}",
        *(str(p) for p in source_paths),
        f"/Fo{obj_prefix}",
        f"/Fe{dll_path}",
        "/link",
        f"/IMPLIB:{implib_path}",
    ]

    logger.info("Compiling native kernel: %s", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=str(build_dir),
        env=build_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if completed.returncode != 0 or not dll_path.is_file():
        output_msg = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        raise RuntimeError(
            f"MSVC compilation failed (exit={completed.returncode}):\n{output_msg}"
        )

    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    build_info_path = build_dir / "build_info.json"
    result = NativeBuildResult(
        built_at_utc=built_at,
        dll_path=dll_path.resolve(),
        implib_path=implib_path.resolve(),
        build_info_path=build_info_path.resolve(),
        source_files=tuple(str(p) for p in source_paths),
        compiler_path=cl_exe,
        compiler_version=cl_version,
        compiler_flags=native_cfg.compiler_flags,
        vcvars64_path=vcvars64_used,
    )

    payload = asdict(result)
    payload["dll_path"] = str(result.dll_path)
    payload["implib_path"] = str(result.implib_path)
    payload["build_info_path"] = str(result.build_info_path)
    payload["compiler_stdout"] = completed.stdout.strip()
    payload["compiler_stderr"] = completed.stderr.strip()

    with build_info_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Built DLL: %s (%s)", result.dll_path, result.compiler_version)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for native DLL compilation."""
    parser = argparse.ArgumentParser(
        prog="build_native.py",
        description="Compile the Task 09 C single-pass entropy pyramid kernel into a Windows DLL.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/optimization/entropy_maps.toml"),
        help="Path to optimization TOML config relative to code/ or absolute.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing build directory before compiling.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for MSVC native C kernel build."""
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        cfg = load_optimization_config(args.config, project_root=PROJECT_ROOT)
        result = build_native_library(cfg, clean=bool(args.clean))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        logger.error("Native build failed: %s", exc)
        raise SystemExit(f"error: {exc}") from exc

    logger.info("Native build metadata written to %s", result.build_info_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
