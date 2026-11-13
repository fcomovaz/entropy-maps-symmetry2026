"""Optional C99 reference DLL/SO compilation and typed ctypes FFI bridge for Lab 14."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging import logger
from config_loader import NativeBuildConfig
from src.features.build_native import _query_cl_version, _resolve_toolchain

c_float_p = ctypes.POINTER(ctypes.c_float)
c_int32_p = ctypes.POINTER(ctypes.c_int32)
c_int16_p = ctypes.POINTER(ctypes.c_int16)


@dataclass(frozen=True, slots=True)
class NativeBuildMetadata:
    """Build provenance stored in build_info.json after compiling C99 kernels."""

    built_at_utc: str
    binary_path: str
    source_file: str
    compiler_path: str
    compiler_version: str
    compiler_flags: tuple[str, ...]
    vcvars64_path: str | None


class NativeClassifierLibrary:
    """Typed ctypes wrapper around mcu_classifiers shared library."""

    def __init__(self, binary_path: Path, build_metadata: NativeBuildMetadata | None = None) -> None:
        self.binary_path = binary_path
        self.build_metadata = build_metadata
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            self._dll_dir_handle = os.add_dll_directory(str(binary_path.parent))
        else:
            self._dll_dir_handle = None
        self._lib = ctypes.CDLL(str(binary_path))
        self._bind_signatures()

    def _bind_signatures(self) -> None:
        lib = self._lib

        lib.mcu_dense_dot_batch.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32, c_float_p, ctypes.c_float, c_float_p
        ]
        lib.mcu_dense_dot_batch.restype = None

        lib.mcu_sparse_dot_batch.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32,
            c_float_p, c_int32_p, ctypes.c_float, ctypes.c_int32, c_float_p,
        ]
        lib.mcu_sparse_dot_batch.restype = None

        lib.mcu_diag_quad_batch.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32,
            c_float_p, c_float_p, ctypes.c_float, c_float_p,
        ]
        lib.mcu_diag_quad_batch.restype = None

        lib.mcu_mlp_1hidden_batch.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            c_float_p, c_float_p, c_float_p, ctypes.c_float, c_float_p,
        ]
        lib.mcu_mlp_1hidden_batch.restype = None

        lib.mcu_mlp_1hidden_skip_batch.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            c_float_p, ctypes.c_float,
            c_float_p, c_float_p, c_float_p, c_float_p,
        ]
        lib.mcu_mlp_1hidden_skip_batch.restype = None

        lib.mcu_mlp_2hidden_batch.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, ctypes.c_float, c_float_p,
        ]
        lib.mcu_mlp_2hidden_batch.restype = None

        lib.mcu_tree_ensemble_batch.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            c_int32_p, c_int16_p, c_float_p, c_int16_p, c_int16_p, c_float_p,
            ctypes.c_float, ctypes.c_float, c_float_p,
        ]
        lib.mcu_tree_ensemble_batch.restype = None

        lib.mcu_bench_dense_dot.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32, c_float_p, ctypes.c_float, ctypes.c_int32
        ]
        lib.mcu_bench_dense_dot.restype = ctypes.c_float

        lib.mcu_bench_sparse_dot.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32,
            c_float_p, c_int32_p, ctypes.c_float, ctypes.c_int32, ctypes.c_int32,
        ]
        lib.mcu_bench_sparse_dot.restype = ctypes.c_float

        lib.mcu_bench_diag_quad.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32,
            c_float_p, c_float_p, ctypes.c_float, ctypes.c_int32,
        ]
        lib.mcu_bench_diag_quad.restype = ctypes.c_float

        lib.mcu_bench_mlp_1hidden.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            c_float_p, c_float_p, c_float_p, ctypes.c_float, ctypes.c_int32,
        ]
        lib.mcu_bench_mlp_1hidden.restype = ctypes.c_float

        lib.mcu_bench_mlp_1hidden_skip.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            c_float_p, ctypes.c_float,
            c_float_p, c_float_p, c_float_p, ctypes.c_int32,
        ]
        lib.mcu_bench_mlp_1hidden_skip.restype = ctypes.c_float

        lib.mcu_bench_mlp_2hidden.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            c_float_p, c_float_p, c_float_p, c_float_p, c_float_p, ctypes.c_float, ctypes.c_int32,
        ]
        lib.mcu_bench_mlp_2hidden.restype = ctypes.c_float

        lib.mcu_bench_tree_ensemble.argtypes = [
            c_float_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            c_int32_p, c_int16_p, c_float_p, c_int16_p, c_int16_p, c_float_p,
            ctypes.c_float, ctypes.c_float, ctypes.c_int32,
        ]
        lib.mcu_bench_tree_ensemble.restype = ctypes.c_float

    @property
    def raw(self) -> ctypes.CDLL:
        """Expose underlying CDLL handle cleanly for model strategy bindings."""
        return self._lib

    @staticmethod
    def f32_ptr(arr: np.ndarray) -> c_float_p:
        """Return float32 pointer without temporary copy; caller retains backing array lifetime."""
        if arr.dtype != np.float32 or not arr.flags["C_CONTIGUOUS"]:
            raise TypeError(
                f"Expected C-contiguous float32 array for FFI lifetime safety, "
                f"got dtype={arr.dtype}, c_contiguous={arr.flags['C_CONTIGUOUS']}"
            )
        return arr.ctypes.data_as(c_float_p)

    @staticmethod
    def i32_ptr(arr: np.ndarray) -> c_int32_p:
        """Return int32 pointer without temporary copy; caller retains backing array lifetime."""
        if arr.dtype != np.int32 or not arr.flags["C_CONTIGUOUS"]:
            raise TypeError(
                f"Expected C-contiguous int32 array for FFI lifetime safety, "
                f"got dtype={arr.dtype}, c_contiguous={arr.flags['C_CONTIGUOUS']}"
            )
        return arr.ctypes.data_as(c_int32_p)

    @staticmethod
    def i16_ptr(arr: np.ndarray) -> c_int16_p:
        """Return int16 pointer without temporary copy; caller retains backing array lifetime."""
        if arr.dtype != np.int16 or not arr.flags["C_CONTIGUOUS"]:
            raise TypeError(
                f"Expected C-contiguous int16 array for FFI lifetime safety, "
                f"got dtype={arr.dtype}, c_contiguous={arr.flags['C_CONTIGUOUS']}"
            )
        return arr.ctypes.data_as(c_int16_p)


def build_native_library(cfg: NativeBuildConfig, force_rebuild: bool = False) -> NativeBuildMetadata:
    """Compile mcu_classifiers.c using MSVC x64 on Windows or gcc/clang on POSIX."""
    if not cfg.source_file.is_file():
        raise FileNotFoundError(f"C99 kernel source missing: {cfg.source_file}")

    cfg.build_dir.mkdir(parents=True, exist_ok=True)
    binary_path = cfg.binary_path
    info_path = cfg.build_info_path

    if (
        not force_rebuild
        and binary_path.is_file()
        and info_path.is_file()
        and binary_path.stat().st_mtime >= cfg.source_file.stat().st_mtime
        and (not cfg.header_file.is_file() or binary_path.stat().st_mtime >= cfg.header_file.stat().st_mtime)
    ):
        raw_info: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
        return NativeBuildMetadata(
            built_at_utc=str(raw_info["built_at_utc"]),
            binary_path=str(raw_info["binary_path"]),
            source_file=str(raw_info["source_file"]),
            compiler_path=str(raw_info["compiler_path"]),
            compiler_version=str(raw_info["compiler_version"]),
            compiler_flags=tuple(str(f) for f in raw_info["compiler_flags"]),
            vcvars64_path=raw_info.get("vcvars64_path"),
        )

    if sys.platform == "win32":
        cl_exe, env, vcvars_path = _resolve_toolchain(cfg.vswhere_hint, scratch_dir=cfg.build_dir)
        compiler_ver = _query_cl_version(cl_exe, env)
        flags = ("/nologo", "/O2", "/fp:precise", "/W4", "/LD")
        obj_path = cfg.build_dir / f"{cfg.source_file.stem}.obj"
        implib_path = cfg.build_dir / f"{binary_path.stem}.lib"
        exp_path = cfg.build_dir / f"{binary_path.stem}.exp"
        cmd = [
            cl_exe,
            *flags,
            str(cfg.source_file),
            f"/Fo:{obj_path}",
            f"/Fe:{binary_path}",
            f"/link",
            f"/IMPLIB:{implib_path}",
        ]
        completed = subprocess.run(
            cmd,
            env=env,
            cwd=str(cfg.build_dir),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if completed.returncode != 0 or not binary_path.is_file():
            detail = (completed.stdout or "") + "\n" + (completed.stderr or "")
            raise RuntimeError(f"MSVC build failed (rc={completed.returncode}):\n{detail.strip()}")
        for tmp_file in (obj_path, exp_path):
            if tmp_file.exists():
                tmp_file.unlink()
    else:
        compiler = shutil.which("gcc") or shutil.which("clang")
        if compiler is None:
            raise RuntimeError("No C compiler (gcc/clang) found on PATH")
        ver_proc = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=False)
        compiler_ver = (ver_proc.stdout or ver_proc.stderr or compiler).splitlines()[0].strip()
        flags = ("-O3", "-std=c99", "-Wall", "-Wextra", "-shared", "-fPIC")
        vcvars_path = None
        cmd = [compiler, *flags, str(cfg.source_file), "-o", str(binary_path)]
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=60)
        if completed.returncode != 0 or not binary_path.is_file():
            detail = (completed.stdout or "") + "\n" + (completed.stderr or "")
            raise RuntimeError(f"C99 compiler build failed (rc={completed.returncode}):\n{detail.strip()}")

    meta = NativeBuildMetadata(
        built_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        binary_path=str(binary_path),
        source_file=str(cfg.source_file),
        compiler_path=cl_exe if sys.platform == "win32" else compiler,
        compiler_version=compiler_ver,
        compiler_flags=flags,
        vcvars64_path=vcvars_path,
    )
    info_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    return meta


def try_load_native_library(
    cfg: NativeBuildConfig,
    force_rebuild: bool = False,
) -> NativeClassifierLibrary | None:
    """Compile and load native library cleanly, returning None if disabled or compiler unavailable."""
    if not cfg.enabled:
        logger.info("Native C99 library disabled by configuration.")
        return None
    try:
        meta = build_native_library(cfg, force_rebuild=force_rebuild)
        lib = NativeClassifierLibrary(cfg.binary_path, build_metadata=meta)
        logger.info("Loaded C99 native library: %s (%s)", cfg.binary_path.name, meta.compiler_version)
        return lib
    except Exception as exc:
        logger.warning("C99 native build/load skipped (%s); Python/NumPy cores stay active.", exc)
        return None


def measure_c_inner_us_per_sample(
    call_once_fn: Any,
    inner_iters: int,
    warmup_calls: int = 10,
    timed_calls: int = 50,
) -> float:
    """Measure median per-sample C99 latency in microseconds over batched C loops."""
    for _ in range(warmup_calls):
        _ = call_once_fn()
    samples_us: list[float] = []
    for _ in range(timed_calls):
        t0 = time.perf_counter_ns()
        _ = call_once_fn()
        dt_ns = time.perf_counter_ns() - t0
        samples_us.append((dt_ns / 1e3) / float(inner_iters))
    return float(np.median(samples_us))
