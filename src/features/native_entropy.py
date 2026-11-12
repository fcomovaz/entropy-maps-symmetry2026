"""Ctypes bindings, compatibility validation, and preallocated execution for the C99 entropy kernel."""

from __future__ import annotations

from collections.abc import Sequence
import ctypes
from dataclasses import dataclass
import functools
import os
from pathlib import Path
import tomllib
from typing import Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NATIVE_CONFIG_REL = Path("configs/features/native_entropy.toml")

# Matches structural macro EP_MAX_COARSE_CELLS in src/features/native/entropy_pyramid.h
NATIVE_MAX_COARSE_CELLS = 128

_C_UINT8_P = ctypes.POINTER(ctypes.c_uint8)
_C_INT_P = ctypes.POINTER(ctypes.c_int)
_C_UINT32_P = ctypes.POINTER(ctypes.c_uint32)
_C_DOUBLE_P = ctypes.POINTER(ctypes.c_double)


@dataclass(frozen=True, slots=True)
class NativeKernelLimits:
    """Structural capacity bounds supported by the compiled C99 entropy pyramid DLL."""

    max_finest_grid: int
    max_coarse_cells: int = NATIVE_MAX_COARSE_CELLS
    max_bins: int = 256
    max_levels: int = 16


@dataclass(frozen=True, slots=True)
class NativeEntropyConfig:
    """MSVC compilation and shared-library path configuration from TOML."""

    source_dir: Path
    build_dir: Path
    library_name: str
    c_sources: tuple[str, ...]
    compiler_flags: tuple[str, ...]
    vswhere_hint: str

    @property
    def dll_path(self) -> Path:
        """Resolved path to compiled shared library."""
        suffix = ".dll" if os.name == "nt" else ".so"
        return self.build_dir / f"{self.library_name}{suffix}"


@dataclass(frozen=True, slots=True)
class NativePlanBuffers:
    """Immutable owner of aligned C buffers and bound ctypes pointers for one EntropyPlan."""

    dll_path: Path
    lib: ctypes.CDLL
    limits: NativeKernelLimits
    image_size: int
    grid_levels: tuple[int, ...]
    bins: int
    g_max: int
    total_features: int
    levels_i32: np.ndarray
    levels_ptr: object
    qlut_u8: np.ndarray
    qlut_ptr: object
    nlogn_lut: np.ndarray
    nlogn_ptr: object
    out_b1: np.ndarray
    out_b1_ptr: object
    dummy_img: np.ndarray
    dummy_img_ptr: object


def _require(table: Mapping[str, object], key: str, where: str) -> object:
    """Extract required key from TOML mapping or raise KeyError with context."""
    if key not in table:
        raise KeyError(f"Missing required key '{key}' in {where}")
    return table[key]


def _resolve_project_path(raw_path: Path | str, project_root: Path) -> Path:
    """Resolve raw_path relative to project_root when not already absolute."""
    p = Path(raw_path)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def load_native_config(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> NativeEntropyConfig:
    """Load, anchor paths, and fail-fast validate [entropy_native] TOML configuration."""
    root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    rel_or_abs = DEFAULT_NATIVE_CONFIG_REL if config_path is None else Path(config_path)
    resolved_cfg = _resolve_project_path(rel_or_abs, root)
    if not resolved_cfg.is_file():
        raise FileNotFoundError(f"Native entropy config file not found: {resolved_cfg}")

    with resolved_cfg.open("rb") as f:
        raw = tomllib.load(f)

    nat_obj = _require(raw, "entropy_native", str(resolved_cfg))
    if not isinstance(nat_obj, dict):
        raise ValueError(f"[entropy_native] section in {resolved_cfg} must be a TOML table")
    nat: Mapping[str, object] = nat_obj

    source_raw = str(_require(nat, "source_dir", "[entropy_native]")).strip()
    build_raw = str(_require(nat, "build_dir", "[entropy_native]")).strip()
    library_name = str(_require(nat, "library_name", "[entropy_native]")).strip()
    raw_sources = _require(nat, "c_sources", "[entropy_native]")
    raw_flags = _require(nat, "compiler_flags", "[entropy_native]")
    vswhere_hint = str(_require(nat, "vswhere_hint", "[entropy_native]"))

    if not source_raw or not build_raw:
        raise ValueError("'source_dir' and 'build_dir' in [entropy_native] cannot be empty")
    if not library_name:
        raise ValueError("'library_name' in [entropy_native] cannot be empty")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("'c_sources' in [entropy_native] must be a non-empty list")
    if not isinstance(raw_flags, list):
        raise ValueError("'compiler_flags' in [entropy_native] must be a list")

    return NativeEntropyConfig(
        source_dir=_resolve_project_path(source_raw, root),
        build_dir=_resolve_project_path(build_raw, root),
        library_name=library_name,
        c_sources=tuple(str(s) for s in raw_sources),
        compiler_flags=tuple(str(fl) for fl in raw_flags),
        vswhere_hint=vswhere_hint,
    )


def resolve_default_native_dll_path(project_root: Path | None = None) -> Path | None:
    """Return configured DLL path if configs/features/native_entropy.toml exists, else None."""
    root = PROJECT_ROOT if project_root is None else Path(project_root).resolve()
    cfg_path = root / DEFAULT_NATIVE_CONFIG_REL
    if not cfg_path.is_file():
        return None
    return load_native_config(cfg_path, project_root=root).dll_path


def _translate_c_error(rc: int, context: str) -> str:
    """Map native C return codes to actionable diagnostics."""
    if rc == -1:
        return f"{context}: invalid pointer or out-of-range scalar argument (rc=-1)"
    if rc == -2:
        return f"{context}: invalid, duplicate, oversized, or non-nested grid_levels (rc=-2)"
    if rc == -3:
        return f"{context}: image_size is not divisible by finest grid level (rc=-3)"
    return f"{context}: unexpected native return code {rc}"


@functools.lru_cache(maxsize=8)
def _load_cdll_cached(resolved_str: str, mtime_ns: int, size_bytes: int) -> ctypes.CDLL:
    """Load DLL and bind exact ctypes function prototypes once per DLL build version."""
    del mtime_ns, size_bytes
    lib = ctypes.CDLL(resolved_str)

    sig_main = [
        _C_UINT8_P,   # img
        ctypes.c_int, # n_images
        ctypes.c_int, # size
        _C_INT_P,     # levels
        ctypes.c_int, # n_levels
        ctypes.c_int, # bins
        _C_UINT8_P,   # qlut
        _C_DOUBLE_P,  # nlogn_lut
        _C_DOUBLE_P,  # out_features
    ]
    lib.entropy_pyramid_u8.argtypes = sig_main
    lib.entropy_pyramid_u8.restype = ctypes.c_int

    lib.entropy_pyramid_noop.argtypes = sig_main
    lib.entropy_pyramid_noop.restype = ctypes.c_int

    lib.entropy_pyramid_stage_fine_hist_u8.argtypes = [
        _C_UINT8_P,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _C_UINT8_P,
        _C_UINT32_P,
    ]
    lib.entropy_pyramid_stage_fine_hist_u8.restype = ctypes.c_int

    lib.entropy_pyramid_stage_sum.argtypes = [
        _C_UINT32_P,
        ctypes.c_int,
        ctypes.c_int,
        _C_INT_P,
        ctypes.c_int,
        ctypes.c_int,
        _C_UINT32_P,
    ]
    lib.entropy_pyramid_stage_sum.restype = ctypes.c_int

    lib.entropy_pyramid_stage_entropy.argtypes = [
        _C_UINT32_P,
        ctypes.c_int,
        ctypes.c_int,
        _C_INT_P,
        ctypes.c_int,
        ctypes.c_int,
        _C_DOUBLE_P,
        _C_DOUBLE_P,
    ]
    lib.entropy_pyramid_stage_entropy.restype = ctypes.c_int

    lib.entropy_pyramid_limits.argtypes = [_C_INT_P, _C_INT_P, _C_INT_P]
    lib.entropy_pyramid_limits.restype = None

    return lib


def load_native_library(dll_path: Path) -> ctypes.CDLL:
    """Load the compiled C99 entropy pyramid DLL with validated signatures."""
    resolved = Path(dll_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Native C entropy library not found at '{resolved}'. "
            "Build it first via '.\\.venv\\Scripts\\python.exe -m src.features.build_native'."
        )
    stat = resolved.stat()
    return _load_cdll_cached(str(resolved), int(stat.st_mtime_ns), int(stat.st_size))


def query_native_limits(dll_path: Path) -> NativeKernelLimits:
    """Query compiled structural bounds exported by the C99 shared library."""
    lib = load_native_library(dll_path)
    c_g = ctypes.c_int(0)
    c_b = ctypes.c_int(0)
    c_l = ctypes.c_int(0)
    lib.entropy_pyramid_limits(ctypes.byref(c_g), ctypes.byref(c_b), ctypes.byref(c_l))
    return NativeKernelLimits(
        max_finest_grid=int(c_g.value),
        max_coarse_cells=NATIVE_MAX_COARSE_CELLS,
        max_bins=int(c_b.value),
        max_levels=int(c_l.value),
    )


def check_native_compatibility(
    image_size: int,
    grid_levels: Sequence[int],
    bins: int,
    limits: NativeKernelLimits | None = None,
    dll_path: Path | None = None,
) -> tuple[bool, str | None]:
    """Validate nesting and C kernel capacity bounds without throwing on expected incompatibility."""
    if type(image_size) is not int or image_size < 1:
        return False, f"image_size must be a positive integer, got {image_size!r}"
    if type(bins) is not int or not (2 <= bins <= 256):
        return False, f"bins must be an integer in [2, 256], got {bins!r}"
    if not grid_levels:
        return False, "grid_levels cannot be empty"
    for g in grid_levels:
        if type(g) is not int or g < 1 or g > image_size:
            return False, f"grid level {g!r} out of valid range [1, {image_size}]"
    if len(set(grid_levels)) != len(grid_levels):
        return False, f"grid_levels contains duplicate levels: {list(grid_levels)}"

    g_max = max(grid_levels)
    if image_size % g_max != 0:
        return (
            False,
            f"image_size={image_size} is not divisible by finest grid g_max={g_max}",
        )
    for g in grid_levels:
        if g_max % g != 0:
            return (
                False,
                f"finest grid g_max={g_max} is not divisible by grid level g={g}",
            )

    active_limits = limits
    if active_limits is None:
        if dll_path is not None:
            resolved_dll = Path(dll_path).resolve()
            if not resolved_dll.is_file():
                return False, f"native DLL not found at '{resolved_dll}'"
            active_limits = query_native_limits(resolved_dll)
        else:
            active_limits = NativeKernelLimits(max_finest_grid=16)

    if g_max > active_limits.max_finest_grid:
        return (
            False,
            f"g_max={g_max} exceeds native limit {active_limits.max_finest_grid}",
        )
    if bins > active_limits.max_bins:
        return False, f"bins={bins} exceeds native limit {active_limits.max_bins}"
    if len(grid_levels) > active_limits.max_levels:
        return (
            False,
            f"len(grid_levels)={len(grid_levels)} exceeds native limit {active_limits.max_levels}",
        )
    coarse_cells = sum(g * g for g in grid_levels if g != g_max)
    if coarse_cells > active_limits.max_coarse_cells:
        return (
            False,
            f"coarse_cells={coarse_cells} exceeds native limit {active_limits.max_coarse_cells}",
        )
    return True, None


def _default_qlut(bins: int) -> np.ndarray:
    p = np.arange(256, dtype=np.uint16)
    return np.ascontiguousarray(((p * bins) // 256).astype(np.uint8))


def _default_nlogn_lut(max_pixels: int) -> np.ndarray:
    lut = np.zeros(max_pixels + 1, dtype=np.float64)
    if max_pixels >= 1:
        c = np.arange(1, max_pixels + 1, dtype=np.float64)
        lut[1:] = c * np.log2(c)
    return np.ascontiguousarray(lut)


def prepare_native_buffers(
    image_size: int,
    grid_levels: Sequence[int],
    bins: int,
    dll_path: Path,
    qlut: np.ndarray | None = None,
    nlogn_lut: np.ndarray | None = None,
) -> NativePlanBuffers:
    """Load DLL, verify native structural compatibility, and preallocate fixed C buffers."""
    resolved_dll = Path(dll_path).resolve()
    lib = load_native_library(resolved_dll)
    limits = query_native_limits(resolved_dll)

    ok, reason = check_native_compatibility(image_size, grid_levels, bins, limits=limits)
    if not ok:
        raise ValueError(f"c_single_pass backend incompatible: {reason}")

    validated_levels = tuple(int(g) for g in grid_levels)
    g_max = max(validated_levels)
    total_features = sum(g * g for g in validated_levels)

    levels_i32 = np.ascontiguousarray(validated_levels, dtype=np.int32)
    qlut_u8 = (
        np.ascontiguousarray(qlut, dtype=np.uint8)
        if qlut is not None
        else _default_qlut(bins)
    )
    if qlut_u8.shape != (256,):
        raise ValueError(f"qlut must have shape (256,), got {qlut_u8.shape}")

    min_cell_pixels_max = (image_size // min(validated_levels)) ** 2
    nlogn_arr = (
        np.ascontiguousarray(nlogn_lut, dtype=np.float64)
        if nlogn_lut is not None
        else _default_nlogn_lut(min_cell_pixels_max)
    )
    if nlogn_arr.ndim != 1 or nlogn_arr.shape[0] < (min_cell_pixels_max + 1):
        raise ValueError(
            f"nlogn_lut length {nlogn_arr.shape} insufficient for max cell pixels {min_cell_pixels_max}"
        )

    out_b1 = np.empty((1, total_features), dtype=np.float64)
    dummy_img = np.zeros((1, image_size, image_size), dtype=np.uint8)

    return NativePlanBuffers(
        dll_path=resolved_dll,
        lib=lib,
        limits=limits,
        image_size=int(image_size),
        grid_levels=validated_levels,
        bins=int(bins),
        g_max=int(g_max),
        total_features=int(total_features),
        levels_i32=levels_i32,
        levels_ptr=levels_i32.ctypes.data_as(_C_INT_P),
        qlut_u8=qlut_u8,
        qlut_ptr=qlut_u8.ctypes.data_as(_C_UINT8_P),
        nlogn_lut=nlogn_arr,
        nlogn_ptr=nlogn_arr.ctypes.data_as(_C_DOUBLE_P),
        out_b1=out_b1,
        out_b1_ptr=out_b1.ctypes.data_as(_C_DOUBLE_P),
        dummy_img=dummy_img,
        dummy_img_ptr=dummy_img.ctypes.data_as(_C_UINT8_P),
    )


def _normalize_u8_batch(images_u8: np.ndarray, expected_size: int) -> np.ndarray:
    """Validate uint8 [N, H, W] or [H, W] square images and enforce C-contiguity."""
    arr = np.asarray(images_u8)
    if arr.dtype != np.uint8:
        raise ValueError(f"c_single_pass requires uint8 images, got dtype {arr.dtype}")
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(
            f"images must have shape [N, H, W] or [H, W], got shape {arr.shape}"
        )
    _, height, width = arr.shape
    if height != width:
        raise ValueError(f"Images must be square, got {height}x{width}")
    if height != expected_size:
        raise ValueError(
            f"Native plan prepared for image_size={expected_size}, got {height}x{width}"
        )
    return arr if arr.flags.c_contiguous else np.ascontiguousarray(arr, dtype=np.uint8)


def run_native_entropy(buffers: NativePlanBuffers, images_u8: np.ndarray) -> np.ndarray:
    """Execute compiled C99 single-pass entropy pyramid on [N, H, W] or [H, W] uint8 images."""
    c_img = _normalize_u8_batch(images_u8, buffers.image_size)
    n_images = int(c_img.shape[0])
    if n_images == 0:
        return np.zeros((0, buffers.total_features), dtype=np.float64)

    if n_images == 1:
        rc = buffers.lib.entropy_pyramid_u8(
            c_img.ctypes.data_as(_C_UINT8_P),
            1,
            buffers.image_size,
            buffers.levels_ptr,
            len(buffers.grid_levels),
            buffers.bins,
            buffers.qlut_ptr,
            buffers.nlogn_ptr,
            buffers.out_b1_ptr,
        )
        if rc != 0:
            raise ValueError(_translate_c_error(rc, "entropy_pyramid_u8"))
        # Copy avoids aliasing across consecutive single-image calls.
        return buffers.out_b1.copy()

    out = np.empty((n_images, buffers.total_features), dtype=np.float64)
    rc = buffers.lib.entropy_pyramid_u8(
        c_img.ctypes.data_as(_C_UINT8_P),
        n_images,
        buffers.image_size,
        buffers.levels_ptr,
        len(buffers.grid_levels),
        buffers.bins,
        buffers.qlut_ptr,
        buffers.nlogn_ptr,
        out.ctypes.data_as(_C_DOUBLE_P),
    )
    if rc != 0:
        raise ValueError(_translate_c_error(rc, "entropy_pyramid_u8"))
    return out


def run_native_noop(buffers: NativePlanBuffers) -> int:
    """Invoke empty 9-argument DLL symbol for ctypes FFI call-overhead isolation."""
    return int(
        buffers.lib.entropy_pyramid_noop(
            buffers.dummy_img_ptr,
            1,
            buffers.image_size,
            buffers.levels_ptr,
            len(buffers.grid_levels),
            buffers.bins,
            buffers.qlut_ptr,
            buffers.nlogn_ptr,
            buffers.out_b1_ptr,
        )
    )


def run_native_stage_fine_hist(
    buffers: NativePlanBuffers,
    images_u8: np.ndarray,
    out_fine_hist: np.ndarray | None = None,
) -> np.ndarray:
    """Diagnostic C stage 1: fused pixel quantization + finest-grid histogram [N, g_max^2, bins]."""
    c_img = _normalize_u8_batch(images_u8, buffers.image_size)
    n_images = int(c_img.shape[0])
    fine_shape = (n_images, buffers.g_max * buffers.g_max, buffers.bins)
    if out_fine_hist is None:
        out_arr = np.empty(fine_shape, dtype=np.uint32)
    else:
        if (
            out_fine_hist.shape != fine_shape
            or out_fine_hist.dtype != np.uint32
            or not out_fine_hist.flags.c_contiguous
        ):
            raise ValueError(
                f"out_fine_hist must be C-contiguous uint32 with shape {fine_shape}, "
                f"got {out_fine_hist.shape} {out_fine_hist.dtype}"
            )
        out_arr = out_fine_hist
    if n_images == 0:
        return out_arr

    rc = buffers.lib.entropy_pyramid_stage_fine_hist_u8(
        c_img.ctypes.data_as(_C_UINT8_P),
        n_images,
        buffers.image_size,
        buffers.g_max,
        buffers.bins,
        buffers.qlut_ptr,
        out_arr.ctypes.data_as(_C_UINT32_P),
    )
    if rc != 0:
        raise ValueError(_translate_c_error(rc, "entropy_pyramid_stage_fine_hist_u8"))
    return out_arr


def run_native_stage_sum(
    buffers: NativePlanBuffers,
    fine_hist: np.ndarray,
    out_all_hist: np.ndarray | None = None,
) -> np.ndarray:
    """Diagnostic C stage 2: hierarchical pyramid child summation [N, total_features, bins]."""
    c_fine = np.ascontiguousarray(fine_hist, dtype=np.uint32)
    if c_fine.ndim != 3 or c_fine.shape[1:] != (buffers.g_max * buffers.g_max, buffers.bins):
        raise ValueError(
            f"fine_hist must have shape [N, {buffers.g_max * buffers.g_max}, {buffers.bins}], got {c_fine.shape}"
        )
    n_images = int(c_fine.shape[0])
    all_shape = (n_images, buffers.total_features, buffers.bins)
    if out_all_hist is None:
        out_arr = np.empty(all_shape, dtype=np.uint32)
    else:
        if (
            out_all_hist.shape != all_shape
            or out_all_hist.dtype != np.uint32
            or not out_all_hist.flags.c_contiguous
        ):
            raise ValueError(
                f"out_all_hist must be C-contiguous uint32 with shape {all_shape}, "
                f"got {out_all_hist.shape} {out_all_hist.dtype}"
            )
        out_arr = out_all_hist
    if n_images == 0:
        return out_arr

    rc = buffers.lib.entropy_pyramid_stage_sum(
        c_fine.ctypes.data_as(_C_UINT32_P),
        n_images,
        buffers.image_size,
        buffers.levels_ptr,
        len(buffers.grid_levels),
        buffers.bins,
        out_arr.ctypes.data_as(_C_UINT32_P),
    )
    if rc != 0:
        raise ValueError(_translate_c_error(rc, "entropy_pyramid_stage_sum"))
    return out_arr


def run_native_stage_entropy(
    buffers: NativePlanBuffers,
    all_hist: np.ndarray,
    out_features: np.ndarray | None = None,
) -> np.ndarray:
    """Diagnostic C stage 3: LUT Shannon entropy and analytical bounds [N, total_features]."""
    c_all = np.ascontiguousarray(all_hist, dtype=np.uint32)
    if c_all.ndim != 3 or c_all.shape[1:] != (buffers.total_features, buffers.bins):
        raise ValueError(
            f"all_hist must have shape [N, {buffers.total_features}, {buffers.bins}], got {c_all.shape}"
        )
    n_images = int(c_all.shape[0])
    feat_shape = (n_images, buffers.total_features)
    if out_features is None:
        out_arr = np.empty(feat_shape, dtype=np.float64)
    else:
        if (
            out_features.shape != feat_shape
            or out_features.dtype != np.float64
            or not out_features.flags.c_contiguous
        ):
            raise ValueError(
                f"out_features must be C-contiguous float64 with shape {feat_shape}, "
                f"got {out_features.shape} {out_features.dtype}"
            )
        out_arr = out_features
    if n_images == 0:
        return out_arr

    rc = buffers.lib.entropy_pyramid_stage_entropy(
        c_all.ctypes.data_as(_C_UINT32_P),
        n_images,
        buffers.image_size,
        buffers.levels_ptr,
        len(buffers.grid_levels),
        buffers.bins,
        buffers.nlogn_ptr,
        out_arr.ctypes.data_as(_C_DOUBLE_P),
    )
    if rc != 0:
        raise ValueError(_translate_c_error(rc, "entropy_pyramid_stage_entropy"))
    return out_arr
