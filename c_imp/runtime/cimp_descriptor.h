/**
 * @file cimp_descriptor.h
 * @brief Portable C99 multiscale entropy pyramid + asymmetry feature extractor (x94-v1).
 *
 * Implements both float32 (`cimp_descriptor_x94_f32`) and fixed-point Q15
 * (`cimp_descriptor_x94_q15`) extraction without heap allocation or external OS calls.
 */
#ifndef CIMP_DESCRIPTOR_H
#define CIMP_DESCRIPTOR_H

#include <stdint.h>

#include "cimp_config.h"
#include "cimp_fixed_point.h"
#include "cimp_tensor.h"

/**
 * Caller-owned static/stack scratch workspace for x94 extraction.
 * Avoids all dynamic allocation (malloc/free) on desktop and MCU targets.
 */
typedef struct {
    cimp_hist_count_t all_hist[CIMP_N_X85 * CIMP_BINS];
} CIMP_DescriptorWorkspace;

/**
 * @brief Stage 1: Single-pass pixel quantization and finest-grid histogram accumulation.
 * @param img       Input grayscale image [size * size], row-major uint8_t.
 * @param size      Square image side length (pixels).
 * @param g_max     Finest grid factor (patches = g_max x g_max).
 * @param bins      Number of histogram bins.
 * @param qlut      256-entry pixel -> bin quantization LUT.
 * @param fine_hist Output [g_max * g_max * bins] histogram counts (zero-initialized internally).
 */
int cimp_quantize_and_fine_hist(
    const uint8_t *img,
    int32_t size,
    int32_t g_max,
    int32_t bins,
    const uint8_t *qlut,
    cimp_hist_count_t *fine_hist
);

/**
 * @brief Stage 2: Accumulate coarse-level histograms upward from finest-grid child histograms.
 * @param ws Workspace whose finest-grid slice (offset of CIMP_FINEST_GRID) is already populated.
 */
int cimp_pyramid_sum(CIMP_DescriptorWorkspace *ws);

/**
 * @brief Stage 3 (float32): Compute normalized cell entropies E in [0, 1] for all pyramid levels.
 */
int cimp_entropy_cells_f32(const CIMP_DescriptorWorkspace *ws, float *out_x85);

/**
 * @brief Stage 3 (fixed-point Q15): Compute normalized cell entropies using compiled LUT mode.
 */
int cimp_entropy_cells_q15(const CIMP_DescriptorWorkspace *ws, q15_t *out_x85);

/**
 * @brief Stage 4 (float32): Compute spatial asymmetry features x9 from x85 entropy grids.
 */
int cimp_asymmetry_f32(const float *x85, float *out_x9);

/**
 * @brief Stage 4 (fixed-point Q15): Compute spatial asymmetry features x9 from Q15 x85 grids.
 */
int cimp_asymmetry_q15(const q15_t *x85, q15_t *out_x9);

/**
 * @brief Full float32 x94 pipeline: uint8 image [H*W] -> float out_x94[94].
 */
int cimp_descriptor_x94_f32(
    const uint8_t *img,
    CIMP_DescriptorWorkspace *ws,
    float *out_x94
);

/**
 * @brief Full fixed-point Q15 x94 pipeline: uint8 image [H*W] -> q15_t out_x94[94].
 */
int cimp_descriptor_x94_q15(
    const uint8_t *img,
    CIMP_DescriptorWorkspace *ws,
    q15_t *out_x94
);

#endif /* CIMP_DESCRIPTOR_H */
