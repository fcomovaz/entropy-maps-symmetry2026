"""Ctypes binding and preallocated wrapper (V3) for the C single-pass entropy pyramid kernel."""

from __future__ import annotations

from collections.abc import Sequence
import ctypes
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np

LAB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (PROJECT_ROOT, LAB_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config_loader import build_nlogn_lut, validate_nested_grid_levels
from src.features.entropy_maps import build_quantization_lut, validate_bins
from variants import normalize_and_validate_images_u8

_C_UINT8_P = ctypes.POINTER(ctypes.c_uint8)
_C_INT_P = ctypes.POINTER(ctypes.c_int)
_C_UINT32_P = ctypes.POINTER(ctypes.c_uint32)
_C_DOUBLE_P = ctypes.POINTER(ctypes.c_double)


@dataclass(frozen=True, slots=True)
class NativeKernelLimits:
    """Structural capacity bounds queried directly from the compiled C DLL."""

    max_finest_grid: int
    max_bins: int
    max_levels: int


def _translate_c_error(rc: int, context: str) -> str:
    """Map C kernel error return codes into descriptive diagnostics."""
    if rc == -1:
        return f"{context}: invalid pointer or out-of-range scalar argument (rc=-1)"
    if rc == -2:
        return f"{context}: invalid, duplicate, oversized, or non-nested grid_levels (rc=-2)"
    if rc == -3:
        return f"{context}: image_size is not divisible by finest grid level (rc=-3)"
    return f"{context}: unexpected native return code {rc}"


def load_native_library(dll_path: Path) -> ctypes.CDLL:
    """Load the compiled native DLL once and bind exact ctypes signatures."""
    resolved = dll_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Native C entropy library not found at '{resolved}'. "
            "Run '.\\.venv\\Scripts\\python.exe understanding/13_entropy_maps_optimization/build_native.py' first."
        )

    lib = ctypes.CDLL(str(resolved))

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
        _C_UINT8_P,   # img
        ctypes.c_int, # n_images
        ctypes.c_int, # size
        ctypes.c_int, # g_max
        ctypes.c_int, # bins
        _C_UINT8_P,   # qlut
        _C_UINT32_P,  # out_fine_hist
    ]
    lib.entropy_pyramid_stage_fine_hist_u8.restype = ctypes.c_int

    lib.entropy_pyramid_stage_sum.argtypes = [
        _C_UINT32_P,  # fine_hist
        ctypes.c_int, # n_images
        ctypes.c_int, # size
        _C_INT_P,     # levels
        ctypes.c_int, # n_levels
        ctypes.c_int, # bins
        _C_UINT32_P,  # out_all_hist
    ]
    lib.entropy_pyramid_stage_sum.restype = ctypes.c_int

    lib.entropy_pyramid_stage_entropy.argtypes = [
        _C_UINT32_P,  # all_hist
        ctypes.c_int, # n_images
        ctypes.c_int, # size
        _C_INT_P,     # levels
        ctypes.c_int, # n_levels
        ctypes.c_int, # bins
        _C_DOUBLE_P,  # nlogn_lut
        _C_DOUBLE_P,  # out_features
    ]
    lib.entropy_pyramid_stage_entropy.restype = ctypes.c_int

    lib.entropy_pyramid_limits.argtypes = [_C_INT_P, _C_INT_P, _C_INT_P]
    lib.entropy_pyramid_limits.restype = None

    return lib


class V3CSinglePassVariant:
    """V3: native C single-pass finest-grid histogram + pyramid sum + LUT entropy via ctypes."""

    name: str = "V3_c_single_pass"

    def __init__(self, dll_path: Path) -> None:
        self._dll_path = Path(dll_path).resolve()
        self._lib = load_native_library(self._dll_path)
        self._limits = self._query_limits()

        # Prepared state (set by prepare)
        self._image_size: int | None = None
        self._grid_levels: tuple[int, ...] = ()
        self._bins: int = 0
        self._g_max: int = 0
        self._total_features: int = 0

        # Owning NumPy arrays + cached ctypes pointers to prevent premature GC or re-marshalling
        self._levels_i32: np.ndarray | None = None
        self._levels_ptr: object = None
        self._qlut: np.ndarray | None = None
        self._qlut_ptr: object = None
        self._nlogn_lut: np.ndarray | None = None
        self._nlogn_ptr: object = None
        self._out_b1: np.ndarray | None = None
        self._out_b1_ptr: object = None
        self._dummy_img: np.ndarray | None = None
        self._dummy_img_ptr: object = None

    @property
    def limits(self) -> NativeKernelLimits:
        """Return compiled structural limits reported by the DLL."""
        return self._limits

    def _query_limits(self) -> NativeKernelLimits:
        """Query structural bounds compiled into the shared library."""
        c_g = ctypes.c_int(0)
        c_b = ctypes.c_int(0)
        c_l = ctypes.c_int(0)
        self._lib.entropy_pyramid_limits(ctypes.byref(c_g), ctypes.byref(c_b), ctypes.byref(c_l))
        return NativeKernelLimits(
            max_finest_grid=int(c_g.value),
            max_bins=int(c_b.value),
            max_levels=int(c_l.value),
        )

    def prepare(self, image_size: int, grid_levels: Sequence[int], bins: int) -> None:
        """Validate nesting and DLL structural limits, and preallocate fixed LUTs and batch=1 output."""
        if image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {image_size}")
        validated_levels = validate_nested_grid_levels(image_size, grid_levels)
        validated_bins = validate_bins(bins)

        g_max = max(validated_levels)
        if g_max > self._limits.max_finest_grid:
            raise ValueError(
                f"Finest grid level g_max={g_max} exceeds native limit {self._limits.max_finest_grid}"
            )
        if validated_bins > self._limits.max_bins:
            raise ValueError(
                f"bins={validated_bins} exceeds native limit {self._limits.max_bins}"
            )
        if len(validated_levels) > self._limits.max_levels:
            raise ValueError(
                f"len(grid_levels)={len(validated_levels)} exceeds native limit {self._limits.max_levels}"
            )

        self._image_size = int(image_size)
        self._grid_levels = validated_levels
        self._bins = validated_bins
        self._g_max = g_max
        self._total_features = sum(g * g for g in validated_levels)

        levels_arr = np.ascontiguousarray(validated_levels, dtype=np.int32)
        qlut_arr = np.ascontiguousarray(build_quantization_lut(validated_bins), dtype=np.uint8)
        g_min = min(validated_levels)
        max_cell_pixels = (self._image_size // g_min) ** 2
        nlogn_arr = np.ascontiguousarray(build_nlogn_lut(max_cell_pixels), dtype=np.float64)
        out_b1 = np.empty((1, self._total_features), dtype=np.float64)
        dummy_img = np.zeros((1, self._image_size, self._image_size), dtype=np.uint8)

        self._levels_i32 = levels_arr
        self._levels_ptr = levels_arr.ctypes.data_as(_C_INT_P)
        self._qlut = qlut_arr
        self._qlut_ptr = qlut_arr.ctypes.data_as(_C_UINT8_P)
        self._nlogn_lut = nlogn_arr
        self._nlogn_ptr = nlogn_arr.ctypes.data_as(_C_DOUBLE_P)
        self._out_b1 = out_b1
        self._out_b1_ptr = out_b1.ctypes.data_as(_C_DOUBLE_P)
        self._dummy_img = dummy_img
        self._dummy_img_ptr = dummy_img.ctypes.data_as(_C_UINT8_P)

    def _ensure_prepared(self) -> None:
        if self._image_size is None or self._out_b1 is None:
            raise RuntimeError(f"{self.name} called before prepare()")

    def run_into(self, images_u8: np.ndarray, out_features: np.ndarray) -> np.ndarray:
        """Compute entropy features directly into a preallocated contiguous [N, total_features] float64 array."""
        self._ensure_prepared()
        assert self._image_size is not None
        arr = normalize_and_validate_images_u8(images_u8, self._image_size, self.name)
        n_images = int(arr.shape[0])

        if out_features.dtype != np.float64:
            raise ValueError(f"out_features must have dtype float64, got {out_features.dtype}")
        if out_features.shape != (n_images, self._total_features):
            raise ValueError(
                f"out_features must have shape {(n_images, self._total_features)}, got {out_features.shape}"
            )
        if not out_features.flags.c_contiguous:
            raise ValueError("out_features must be C-contiguous")

        if n_images == 0:
            return out_features

        c_img = arr if arr.flags.c_contiguous else np.ascontiguousarray(arr, dtype=np.uint8)
        rc = self._lib.entropy_pyramid_u8(
            c_img.ctypes.data_as(_C_UINT8_P),
            n_images,
            self._image_size,
            self._levels_ptr,
            len(self._grid_levels),
            self._bins,
            self._qlut_ptr,
            self._nlogn_ptr,
            out_features.ctypes.data_as(_C_DOUBLE_P),
        )
        if rc != 0:
            raise ValueError(_translate_c_error(rc, "entropy_pyramid_u8"))
        return out_features

    def run(self, images_u8: np.ndarray) -> np.ndarray:
        """Compute normalized entropy features via C kernel, copying batch=1 buffer to prevent aliasing."""
        self._ensure_prepared()
        assert self._image_size is not None and self._out_b1 is not None
        arr = normalize_and_validate_images_u8(images_u8, self._image_size, self.name)
        n_images = int(arr.shape[0])
        if n_images == 0:
            return np.zeros((0, self._total_features), dtype=np.float64)

        c_img = arr if arr.flags.c_contiguous else np.ascontiguousarray(arr, dtype=np.uint8)
        if n_images == 1:
            rc = self._lib.entropy_pyramid_u8(
                c_img.ctypes.data_as(_C_UINT8_P),
                1,
                self._image_size,
                self._levels_ptr,
                len(self._grid_levels),
                self._bins,
                self._qlut_ptr,
                self._nlogn_ptr,
                self._out_b1_ptr,
            )
            if rc != 0:
                raise ValueError(_translate_c_error(rc, "entropy_pyramid_u8"))
            # Copy protects callers storing consecutive batch=1 outputs without aliasing
            return self._out_b1.copy()

        out = np.empty((n_images, self._total_features), dtype=np.float64)
        rc = self._lib.entropy_pyramid_u8(
            c_img.ctypes.data_as(_C_UINT8_P),
            n_images,
            self._image_size,
            self._levels_ptr,
            len(self._grid_levels),
            self._bins,
            self._qlut_ptr,
            self._nlogn_ptr,
            out.ctypes.data_as(_C_DOUBLE_P),
        )
        if rc != 0:
            raise ValueError(_translate_c_error(rc, "entropy_pyramid_u8"))
        return out

    def call_noop(self) -> int:
        """Invoke empty 9-argument C DLL symbol to isolate pure ctypes FFI overhead."""
        self._ensure_prepared()
        assert self._image_size is not None
        return int(
            self._lib.entropy_pyramid_noop(
                self._dummy_img_ptr,
                1,
                self._image_size,
                self._levels_ptr,
                len(self._grid_levels),
                self._bins,
                self._qlut_ptr,
                self._nlogn_ptr,
                self._out_b1_ptr,
            )
        )

    def measure_stages_once(self, images_u8: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Diagnostic single run splitting fine_hist (fused quant+hist), pyramid_sum, and entropy in C."""
        self._ensure_prepared()
        assert self._image_size is not None
        arr = normalize_and_validate_images_u8(images_u8, self._image_size, self.name)
        n_images = int(arr.shape[0])
        if n_images == 0:
            return np.zeros((0, self._total_features), dtype=np.float64), {
                "quantize_ms": 0.0,
                "histogram_ms": 0.0,
                "pyramid_sum_ms": 0.0,
                "entropy_ms": 0.0,
            }

        c_img = np.ascontiguousarray(arr, dtype=np.uint8)
        fine_hist = np.empty((n_images, self._g_max * self._g_max, self._bins), dtype=np.uint32)
        all_hist = np.empty((n_images, self._total_features, self._bins), dtype=np.uint32)
        out = np.empty((n_images, self._total_features), dtype=np.float64)

        t0 = time.perf_counter_ns()
        rc1 = self._lib.entropy_pyramid_stage_fine_hist_u8(
            c_img.ctypes.data_as(_C_UINT8_P),
            n_images,
            self._image_size,
            self._g_max,
            self._bins,
            self._qlut_ptr,
            fine_hist.ctypes.data_as(_C_UINT32_P),
        )
        t1 = time.perf_counter_ns()
        if rc1 != 0:
            raise ValueError(_translate_c_error(rc1, "entropy_pyramid_stage_fine_hist_u8"))

        rc2 = self._lib.entropy_pyramid_stage_sum(
            fine_hist.ctypes.data_as(_C_UINT32_P),
            n_images,
            self._image_size,
            self._levels_ptr,
            len(self._grid_levels),
            self._bins,
            all_hist.ctypes.data_as(_C_UINT32_P),
        )
        t2 = time.perf_counter_ns()
        if rc2 != 0:
            raise ValueError(_translate_c_error(rc2, "entropy_pyramid_stage_sum"))

        rc3 = self._lib.entropy_pyramid_stage_entropy(
            all_hist.ctypes.data_as(_C_UINT32_P),
            n_images,
            self._image_size,
            self._levels_ptr,
            len(self._grid_levels),
            self._bins,
            self._nlogn_ptr,
            out.ctypes.data_as(_C_DOUBLE_P),
        )
        t3 = time.perf_counter_ns()
        if rc3 != 0:
            raise ValueError(_translate_c_error(rc3, "entropy_pyramid_stage_entropy"))

        # Quantization in C is fused directly inside the single pixel pass (zero standalone pass).
        return out, {
            "quantize_ms": 0.0,
            "histogram_ms": (t1 - t0) / 1e6,
            "pyramid_sum_ms": (t2 - t1) / 1e6,
            "entropy_ms": (t3 - t2) / 1e6,
        }
