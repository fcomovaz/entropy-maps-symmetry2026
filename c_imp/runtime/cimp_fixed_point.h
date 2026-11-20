/**
 * @file cimp_fixed_point.h
 * @brief Portable C99 Qm.n fixed-point types, rounding shifts, saturation, and bit utilities.
 *
 * Strictly uses integer-only arithmetic (no <math.h>, no heap allocation, no compiler VLAs).
 * Half-up rounding matches Python `src/c_export/fixed_point.py` bit for bit.
 */
#ifndef CIMP_FIXED_POINT_H
#define CIMP_FIXED_POINT_H

#include <limits.h>
#include <stdint.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#if defined(__AVR__)
#include <avr/pgmspace.h>
#elif defined(ESP8266)
#include <pgmspace.h>
#endif

#ifndef CIMP_ENABLE_F32
#define CIMP_ENABLE_F32 1
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__AVR__) || defined(ESP8266)
#define CIMP_ROM PROGMEM
#define CIMP_ROM_READ_U8(p)  ((uint8_t)pgm_read_byte(p))
#define CIMP_ROM_READ_I8(p)  ((int8_t)pgm_read_byte(p))
#define CIMP_ROM_READ_U16(p) ((uint16_t)pgm_read_word(p))
#define CIMP_ROM_READ_I16(p) ((int16_t)pgm_read_word(p))
#define CIMP_ROM_READ_U32(p) ((uint32_t)pgm_read_dword(p))
#define CIMP_ROM_READ_I32(p) ((int32_t)pgm_read_dword(p))
#define CIMP_ROM_READ_F32(p) ((float)pgm_read_float(p))
static inline int64_t cimp_rom_read_i64(const int64_t *p) {
    const uint32_t *w = (const uint32_t *)(const void *)p;
    uint64_t lo = (uint64_t)pgm_read_dword(w);
    uint64_t hi = (uint64_t)pgm_read_dword(w + 1);
    return (int64_t)(lo | (hi << 32));
}
#define CIMP_ROM_READ_I64(p) cimp_rom_read_i64(p)
#else
#define CIMP_ROM
#define CIMP_ROM_READ_U8(p)  (*(const uint8_t *)(p))
#define CIMP_ROM_READ_I8(p)  (*(const int8_t *)(p))
#define CIMP_ROM_READ_U16(p) (*(const uint16_t *)(p))
#define CIMP_ROM_READ_I16(p) (*(const int16_t *)(p))
#define CIMP_ROM_READ_U32(p) (*(const uint32_t *)(p))
#define CIMP_ROM_READ_I32(p) (*(const int32_t *)(p))
#define CIMP_ROM_READ_F32(p) (*(const float *)(p))
#define CIMP_ROM_READ_I64(p) (*(const int64_t *)(p))
#endif

typedef int8_t q7_t;
typedef int16_t q15_t;
typedef int32_t q31_t;
typedef int64_t q63_t;

/** Saturate a signed 64-bit integer to int8_t [-128, 127]. */
static inline int8_t cimp_sat8(int64_t x) {
    if (x > 127LL) {
        return 127;
    }
    if (x < -128LL) {
        return -128;
    }
    return (int8_t)x;
}

/** Saturate a signed 64-bit integer to int16_t [-32768, 32767]. */
static inline int16_t cimp_sat16(int64_t x) {
    if (x > 32767LL) {
        return 32767;
    }
    if (x < -32768LL) {
        return -32768;
    }
    return (int16_t)x;
}

/** Saturate a signed 64-bit integer to int32_t [-2147483648, 2147483647]. */
static inline int32_t cimp_sat32(int64_t x) {
    if (x > (int64_t)INT32_MAX) {
        return INT32_MAX;
    }
    if (x < (int64_t)INT32_MIN) {
        return INT32_MIN;
    }
    return (int32_t)x;
}

/**
 * Arithmetic right shift on int64_t avoiding C99 implementation-defined negative shift.
 * Produces exact floor(x / 2^shift) identical to Python `x >> shift` for shift >= 0.
 */
static inline int64_t cimp_asr_i64(int64_t x, int32_t shift) {
    if (shift <= 0) {
        return x;
    }
    if (shift >= 63) {
        return (x < 0LL) ? -1LL : 0LL;
    }
    if (x >= 0LL) {
        return (int64_t)((uint64_t)x >> shift);
    }
    return -(int64_t)(((uint64_t)(-(x + 1LL)) >> shift) + 1ULL);
}

/**
 * Round-half-up shift for signed 64-bit accumulators.
 * Positive `shift` computes floor((x + 2^(shift-1)) / 2^shift).
 * Negative `shift` left-shifts safely by (-shift).
 */
static inline int64_t cimp_round_shift_right_i64(int64_t x, int32_t shift) {
    if (shift == 0) {
        return x;
    }
    if (shift < 0) {
        int32_t lshift = -shift;
        if (lshift >= 63) {
            if (x > 0LL) {
                return INT64_MAX;
            }
            if (x < 0LL) {
                return INT64_MIN;
            }
            return 0LL;
        }
        return (int64_t)((uint64_t)x << lshift);
    }
    if (shift >= 63) {
        return 0LL;
    }
    return cimp_asr_i64(x + ((int64_t)1 << (shift - 1)), shift);
}

/**
 * Q0.15 multiply with half-up rounding and saturation to +32767 on (-32768 * -32768).
 *
 * Line-by-line semantics match the project issue specification:
 *   1. Promote operands to int32_t so product fits in 31 bits without overflow.
 *   2. Add half-LSB (1 << 14) for round-half-up.
 *   3. Clamp >= 0x40000000 (the unique -1.0 * -1.0 = +1.0 overflow case) to 32767.
 *   4. Arithmetic right-shift by 15 via UB-free helper matching Python `prod >> 15`.
 */
static inline q15_t cimp_q15_mul_round(q15_t a, q15_t b) {
    int32_t prod = ((int32_t)a * (int32_t)b) + (INT32_C(1) << 14);
    if (prod >= INT32_C(0x40000000)) {
        return 32767;
    }
    return (q15_t)cimp_asr_i64((int64_t)prod, 15);
}

/** Saturating Q15 addition. */
static inline q15_t cimp_q15_add_sat(q15_t a, q15_t b) {
    return cimp_sat16((int64_t)a + (int64_t)b);
}

/** Saturating Q15 subtraction. */
static inline q15_t cimp_q15_sub_sat(q15_t a, q15_t b) {
    return cimp_sat16((int64_t)a - (int64_t)b);
}

/** Portable 32-bit count leading zeros (returns 32 when x == 0). */
static inline int32_t cimp_clz32(uint32_t x) {
    if (x == UINT32_C(0)) {
        return 32;
    }
#if defined(_MSC_VER)
    unsigned long idx = 0;
    _BitScanReverse(&idx, (unsigned long)x);
    return 31 - (int32_t)idx;
#elif (defined(__GNUC__) || defined(__clang__)) && defined(UINT_MAX) && (UINT_MAX == 0xFFFFFFFFU)
    return (int32_t)__builtin_clz((unsigned int)x);
#elif (defined(__GNUC__) || defined(__clang__)) && defined(ULONG_MAX) && (ULONG_MAX == 0xFFFFFFFFUL)
    return (int32_t)__builtin_clzl((unsigned long)x);
#else
    int32_t n = 0;
    if ((x & UINT32_C(0xFFFF0000)) == UINT32_C(0)) { n += 16; x <<= 16; }
    if ((x & UINT32_C(0xFF000000)) == UINT32_C(0)) { n += 8;  x <<= 8;  }
    if ((x & UINT32_C(0xF0000000)) == UINT32_C(0)) { n += 4;  x <<= 4;  }
    if ((x & UINT32_C(0xC0000000)) == UINT32_C(0)) { n += 2;  x <<= 2;  }
    if ((x & UINT32_C(0x80000000)) == UINT32_C(0)) { n += 1; }
    return n;
#endif
}

/** Rescale signed Qm.frac_in integer to Qm.frac_out with saturation to int16_t. */
static inline q15_t cimp_rescale_to_q15(int64_t val, int32_t frac_in, int32_t frac_out) {
    return cimp_sat16(cimp_round_shift_right_i64(val, frac_in - frac_out));
}

#ifdef __cplusplus
}
#endif

#endif /* CIMP_FIXED_POINT_H */
