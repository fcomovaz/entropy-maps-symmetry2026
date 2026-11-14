"""Bit-exact Python mirror of C99 Qm.n fixed-point operations and quantization helpers.

Matches `c_imp/runtime/cimp_fixed_point.h` bit for bit:
    - half-up rounding: `floor((x + 2^(shift - 1)) / 2^shift)` == `(x + (1 << (shift - 1))) >> shift`
    - signed saturation: int8 [-128, 127], int16 [-32768, 32767], int32 [-2^31, 2^31-1]
    - arithmetic right shift semantics on negative integers (MSVC/GCC/Xtensa/Python agree)
"""

from __future__ import annotations

import math

import numpy as np


def sat8(x: int) -> int:
    """Saturate integer to signed 8-bit [-128, 127]."""
    if x > 127:
        return 127
    if x < -128:
        return -128
    return int(x)


def sat16(x: int) -> int:
    """Saturate integer to signed 16-bit [-32768, 32767]."""
    if x > 32767:
        return 32767
    if x < -32768:
        return -32768
    return int(x)


def sat32(x: int) -> int:
    """Saturate integer to signed 32-bit [-2147483648, 2147483647]."""
    if x > 2147483647:
        return 2147483647
    if x < -2147483648:
        return -2147483648
    return int(x)


def sat_array(x: np.ndarray, bits: int = 16) -> np.ndarray:
    """Saturate int64 NumPy array to signed `bits` range."""
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    return np.clip(np.asarray(x, dtype=np.int64), qmin, qmax)


def round_shift_right_i64(x: int, shift: int) -> int:
    """Bit-exact Python equivalent of `cimp_round_shift_right_i64(x, shift)`."""
    xi = int(x)
    if shift == 0:
        return xi
    if shift < 0:
        lshift = -shift
        if lshift >= 63:
            if xi > 0:
                return 0x7FFFFFFFFFFFFFFF
            if xi < 0:
                return -0x8000000000000000
            return 0
        # Wrap to signed 64-bit behavior then keep int
        return xi << lshift
    if shift >= 63:
        return 0
    return (xi + (1 << (shift - 1))) >> shift


def round_shift_right_array(x: np.ndarray, shift: int) -> np.ndarray:
    """Vectorized bit-exact equivalent of `cimp_round_shift_right_i64` for int64 arrays."""
    arr = np.asarray(x, dtype=np.int64)
    if shift == 0:
        return arr
    if shift < 0:
        return arr << (-shift)
    if shift >= 63:
        return np.zeros_like(arr, dtype=np.int64)
    return (arr + np.int64(1 << (shift - 1))) >> np.int64(shift)


def q15_mul_round(a: int, b: int) -> int:
    """Bit-exact Python equivalent of `cimp_q15_mul_round(q15_t a, q15_t b)`."""
    prod = int(a) * int(b) + (1 << 14)
    if prod >= 0x40000000:
        return 32767
    return int(prod >> 15)


def q15_add_sat(a: int, b: int) -> int:
    """Bit-exact Python equivalent of `cimp_q15_add_sat`."""
    return sat16(int(a) + int(b))


def q15_sub_sat(a: int, b: int) -> int:
    """Bit-exact Python equivalent of `cimp_q15_sub_sat`."""
    return sat16(int(a) - int(b))


def clz32(x: int) -> int:
    """Bit-exact Python equivalent of `cimp_clz32(uint32_t x)`."""
    v = int(x) & 0xFFFFFFFF
    if v == 0:
        return 32
    return 32 - v.bit_length()


def choose_frac_bits(
    max_abs: float,
    storage_bits: int = 16,
    headroom_bits: int = 0,
    min_frac: int = 0,
) -> int:
    """Select power-of-two fractional bit count Qm.n that avoids overflow of `max_abs`.

    Ensures max representable magnitude ((2^(storage_bits-1) - 1) / 2^n) >= max_abs * 2^headroom_bits.
    """
    if storage_bits not in (8, 16, 32, 64):
        raise ValueError(f"Unsupported storage_bits={storage_bits}")
    qmax = (1 << (storage_bits - 1)) - 1
    target = float(max_abs) * float(1 << max(0, headroom_bits))
    if not math.isfinite(target):
        raise ValueError(f"Non-finite max_abs={max_abs}")
    if target <= 1e-12:
        return max(min_frac, storage_bits - 1)
    # Find largest n <= (storage_bits - 1) such that target * 2^n <= qmax
    # Wait: negative max-abs values allow up to 2^(storage_bits-1), while +max clamps to 2^(storage_bits-1)-1.
    # For values like 1.0 in Q15, n = 15 leaves +1.0 saturating by 1 LSB (32767/32768).
    # For general tensors we require floor(target * 2^n + 0.5) <= qmax -> 2^n <= (qmax + 0.49) / target.
    ratio = float(qmax) / target
    if ratio < 1.0:
        # Needs negative or zero frac bits -> clamp to min_frac
        n = int(math.floor(math.log2(ratio))) if ratio > 0.0 else min_frac
    else:
        n = int(math.floor(math.log2(ratio)))
    return max(min_frac, min(storage_bits - 1, n))


def quantize(
    x: np.ndarray | float,
    frac_bits: int,
    bits: int = 16,
) -> tuple[np.ndarray, int]:
    """Quantize floating-point values into signed Qm.`frac_bits` integers with half-up rounding.

    Returns:
        Tuple `(quantized_int64_array, num_saturated_elements)`.
    """
    if bits not in (8, 16, 32, 64):
        raise ValueError(f"Unsupported integer bit-width {bits}")
    arr = np.asarray(x, dtype=np.float64)
    scale = math.ldexp(1.0, frac_bits)
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    raw = np.floor(arr * scale + 0.5).astype(np.int64)
    n_sat = int(np.sum((raw < qmin) | (raw > qmax)))
    clipped = np.clip(raw, qmin, qmax).astype(np.int64)
    return clipped, n_sat


def dequantize(q: np.ndarray | int, frac_bits: int) -> np.ndarray:
    """Convert fixed-point integer array back to float64."""
    return np.asarray(q, dtype=np.float64) * math.ldexp(1.0, -frac_bits)


def rescale_to_q15(val: int, frac_in: int, frac_out: int) -> int:
    """Bit-exact equivalent of `cimp_rescale_to_q15`."""
    return sat16(round_shift_right_i64(val, frac_in - frac_out))
