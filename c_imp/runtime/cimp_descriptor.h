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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Caller-owned static/stack scratch workspace for x94 extraction.
 * Avoids all dynamic allocation (malloc/free) on desktop and MCU targets.
 */
typedef struct {
    cimp_hist_count_t all_hist[CIMP_N_X85 * CIMP_BINS];
} CIMP_DescriptorWorkspace;

/**
 * Caller-owned streaming workspace for row-by-row Q15 x94 extraction.
 * Keeps only one finest-grid band plus coarse accumulators in SRAM (~1 KiB).
 */
typedef struct {
    cimp_hist_count_t band_hist[CIMP_FINEST_GRID * CIMP_BINS];
    cimp_hist_count_t coarse_hist[CIMP_COARSE_CELLS * CIMP_BINS];
    q15_t fine_entropies[CIMP_FINEST_CELLS];
    uint16_t rows_pushed;
    uint8_t band_row_idx;
    uint8_t band_idx;
} CIMP_DescStream;

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
 * @brief Stage 3 (fixed-point Q15): Evaluate normalized Shannon entropy for a single histogram cell.
 */
q15_t cimp_eval_single_cell_q15(
    const cimp_hist_count_t *cnt,
    int32_t lvl_idx
);

#if CIMP_ENABLE_F32
/**
 * @brief Stage 3 (float32): Compute normalized cell entropies E in [0, 1] for all pyramid levels.
 */
int cimp_entropy_cells_f32(const CIMP_DescriptorWorkspace *ws, float *out_x85);
#endif

/**
 * @brief Stage 3 (fixed-point Q15): Compute normalized cell entropies using compiled LUT mode.
 */
int cimp_entropy_cells_q15(const CIMP_DescriptorWorkspace *ws, q15_t *out_x85);

#if CIMP_ENABLE_F32
/**
 * @brief Stage 4 (float32): Compute spatial asymmetry features x9 from x85 entropy grids.
 */
int cimp_asymmetry_f32(const float *x85, float *out_x9);
#endif

/**
 * @brief Stage 4 (fixed-point Q15): Compute spatial asymmetry features x9 from Q15 x85 grids.
 */
int cimp_asymmetry_q15(const q15_t *x85, q15_t *out_x9);

#if CIMP_ENABLE_F32
/**
 * @brief Full float32 x94 pipeline: uint8 image [H*W] -> float out_x94[94].
 */
int cimp_descriptor_x94_f32(
    const uint8_t *img,
    CIMP_DescriptorWorkspace *ws,
    float *out_x94
);
#endif

/**
 * @brief Full fixed-point Q15 x94 pipeline: uint8 image [H*W] -> q15_t out_x94[94].
 */
int cimp_descriptor_x94_q15(
    const uint8_t *img,
    CIMP_DescriptorWorkspace *ws,
    q15_t *out_x94
);

/**
 * @brief Initialize or reset a streaming Q15 x94 descriptor state.
 */
int cimp_desc_stream_init(CIMP_DescStream *s);

/**
 * @brief Push `n_rows` contiguous `CIMP_IMAGE_SIZE`-byte rows into the streaming descriptor.
 */
int cimp_desc_stream_push_rows(
    CIMP_DescStream *s,
    const uint8_t *rows,
    int32_t n_rows
);

/**
 * @brief Propagate coarse histograms, evaluate coarse/asymmetry features, and write Q15 `out_x94`.
 */
int cimp_desc_stream_finish_q15(
    CIMP_DescStream *s,
    q15_t out_x94[CIMP_N_X94]
);

#ifdef __cplusplus
}
#endif

#endif /* CIMP_DESCRIPTOR_H */
