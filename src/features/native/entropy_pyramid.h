#ifndef ENTROPY_PYRAMID_H
#define ENTROPY_PYRAMID_H

#include <stdint.h>

#ifdef _WIN32
#  ifdef ENTROPY_PYRAMID_EXPORTS
#    define EP_API __declspec(dllexport)
#  else
#    define EP_API __declspec(dllimport)
#  endif
#else
#  define EP_API
#endif

/* Structural capacity bounds for zero-allocation stack execution in embedded/desktop C. */
#define EP_MAX_FINEST_GRID 16
#define EP_MAX_FINEST_CELLS (EP_MAX_FINEST_GRID * EP_MAX_FINEST_GRID)
#define EP_MAX_COARSE_CELLS 128
#define EP_MAX_BINS 256
#define EP_MAX_LEVELS 16

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Single-pass multi-scale Shannon entropy pyramid over square uint8 grayscale images.
 *
 * Preconditions:
 *   - img: contiguous [n_images, size, size] uint8 buffer (non-NULL when n_images > 0)
 *   - levels: n_levels distinct integers in [1, EP_MAX_FINEST_GRID]
 *   - size divisible by max(levels), and max(levels) divisible by every levels[i]
 *   - bins in [2, EP_MAX_BINS]
 *   - qlut: 256-entry uint8 quantization LUT: qlut[p] = (p * bins) / 256
 *   - nlogn_lut: float64 LUT of length >= (size / min(levels))^2 + 1 with entry c = c * log2(c)
 *   - out_features: contiguous [n_images, sum(levels[i]^2)] float64 output buffer
 *
 * Returns:
 *    0 on success
 *   -1 invalid pointer or out-of-range scalar argument
 *   -2 invalid, duplicate, oversized, or non-nested grid levels
 *   -3 image size not divisible by finest grid level
 */
EP_API int entropy_pyramid_u8(
    const uint8_t* img,
    int n_images,
    int size,
    const int* levels,
    int n_levels,
    int bins,
    const uint8_t* qlut,
    const double* nlogn_lut,
    double* out_features
);

/**
 * Empty call sharing the exact ABI of entropy_pyramid_u8 to measure ctypes dispatch overhead.
 * Always returns 0.
 */
EP_API int entropy_pyramid_noop(
    const uint8_t* img,
    int n_images,
    int size,
    const int* levels,
    int n_levels,
    int bins,
    const uint8_t* qlut,
    const double* nlogn_lut,
    double* out_features
);

/**
 * Diagnostic stage 1 (single-pass quantized histogram of the finest grid).
 * out_fine_hist must hold n_images * g_max * g_max * bins uint32_t entries.
 */
EP_API int entropy_pyramid_stage_fine_hist_u8(
    const uint8_t* img,
    int n_images,
    int size,
    int g_max,
    int bins,
    const uint8_t* qlut,
    uint32_t* out_fine_hist
);

/**
 * Diagnostic stage 2 (pyramid child summation from finest grid to all levels in order).
 * out_all_hist must hold n_images * sum(levels[i]^2) * bins uint32_t entries.
 */
EP_API int entropy_pyramid_stage_sum(
    const uint32_t* fine_hist,
    int n_images,
    int size,
    const int* levels,
    int n_levels,
    int bins,
    uint32_t* out_all_hist
);

/**
 * Diagnostic stage 3 (LUT Shannon entropy and equiprobable rule across all level cells).
 * all_hist has shape [n_images, sum(levels[i]^2), bins] in caller levels order.
 */
EP_API int entropy_pyramid_stage_entropy(
    const uint32_t* all_hist,
    int n_images,
    int size,
    const int* levels,
    int n_levels,
    int bins,
    const double* nlogn_lut,
    double* out_features
);

/**
 * Query compiled structural capacity limits.
 */
EP_API void entropy_pyramid_limits(
    int* out_max_finest_grid,
    int* out_max_bins,
    int* out_max_levels
);

#ifdef __cplusplus
}
#endif

#endif /* ENTROPY_PYRAMID_H */
