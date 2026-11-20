/**
 * @file cimp_descriptor.c
 * @brief Portable C99 implementation of multiscale x94 (x85 entropy + x9 asymmetry).
 *
 * Counterpart mapping to Python `src/features/`:
 *   - cimp_quantize_and_fine_hist : src/features/native/entropy_pyramid.c stage_fine_hist_u8
 *   - cimp_pyramid_sum            : src/features/native/entropy_pyramid.c stage_sum
 *   - cimp_entropy_cells_*        : src/features/entropy_maps.py normalized_entropy
 *   - cimp_asymmetry_*            : src/features/asymmetry_maps.py compute_asymmetry_for_grid
 */
#include "cimp_descriptor.h"

#include <stddef.h>
#include <string.h>

#if CIMP_ENABLE_F32
#include <math.h>
#endif

#include "lut_bin_quant.h"
#include "lut_shannon.h"

static int32_t cimp_level_feature_offset(int32_t target_lvl_idx) {
    int32_t off = 0;
    for (int32_t i = 0; i < target_lvl_idx; ++i) {
        const int32_t g = CIMP_ROM_READ_I32(&CIMP_LEVELS[i]);
        off += g * g;
    }
    return off;
}

int cimp_quantize_and_fine_hist(
    const uint8_t *img,
    int32_t size,
    int32_t g_max,
    int32_t bins,
    const uint8_t *qlut,
    cimp_hist_count_t *fine_hist
) {
    if (img == NULL || qlut == NULL || fine_hist == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
    if (size <= 0 || g_max <= 0 || bins < 2 || (size % g_max) != 0) {
        return CIMP_ERR_SHAPE_MISMATCH;
    }
    const int32_t cell_side = size / g_max;
    const size_t total_slots = (size_t)g_max * (size_t)g_max * (size_t)bins;
    memset(fine_hist, 0, total_slots * sizeof(cimp_hist_count_t));

    for (int32_t r_cell = 0; r_cell < g_max; ++r_cell) {
        const int32_t y_base = r_cell * cell_side;
        for (int32_t dy = 0; dy < cell_side; ++dy) {
            const uint8_t *row_ptr = img + (size_t)(y_base + dy) * (size_t)size;
            for (int32_t c_cell = 0; c_cell < g_max; ++c_cell) {
                const int32_t x_base = c_cell * cell_side;
                cimp_hist_count_t *cell_hist =
                    fine_hist + ((size_t)(r_cell * g_max + c_cell) * (size_t)bins);
                for (int32_t dx = 0; dx < cell_side; ++dx) {
                    const uint8_t bin = CIMP_ROM_READ_U8(&qlut[row_ptr[x_base + dx]]);
                    cell_hist[bin] += 1;
                }
            }
        }
    }
    return CIMP_OK;
}

static int cimp_sum_coarse_level(cimp_hist_count_t *hist_base, int32_t lvl) {
    const int32_t g_curr = CIMP_ROM_READ_I32(&CIMP_LEVELS[lvl]);
    const int32_t g_child = CIMP_ROM_READ_I32(&CIMP_LEVELS[lvl + 1]);
    if ((g_child % g_curr) != 0) {
        return CIMP_ERR_SHAPE_MISMATCH;
    }
    const int32_t k = g_child / g_curr;
    const int32_t curr_off = cimp_level_feature_offset(lvl);
    const int32_t child_off = cimp_level_feature_offset(lvl + 1);
    cimp_hist_count_t *dst_base = hist_base + (size_t)curr_off * (size_t)CIMP_BINS;
    const cimp_hist_count_t *src_base = hist_base + (size_t)child_off * (size_t)CIMP_BINS;

    for (int32_t r = 0; r < g_curr; ++r) {
        for (int32_t c = 0; c < g_curr; ++c) {
            cimp_hist_count_t *dst = dst_base + (size_t)(r * g_curr + c) * (size_t)CIMP_BINS;
            memset(dst, 0, (size_t)CIMP_BINS * sizeof(cimp_hist_count_t));
            for (int32_t dy = 0; dy < k; ++dy) {
                for (int32_t dx = 0; dx < k; ++dx) {
                    const int32_t cr = r * k + dy;
                    const int32_t cc = c * k + dx;
                    const cimp_hist_count_t *src =
                        src_base + (size_t)(cr * g_child + cc) * (size_t)CIMP_BINS;
                    for (int32_t b = 0; b < CIMP_BINS; ++b) {
                        dst[b] = (cimp_hist_count_t)(dst[b] + src[b]);
                    }
                }
            }
        }
    }
    return CIMP_OK;
}

int cimp_pyramid_sum(CIMP_DescriptorWorkspace *ws) {
    if (ws == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
    /* CIMP_LEVELS is strictly ascending; merge downward from finest level idx = CIMP_N_LEVELS - 1. */
    for (int32_t lvl = CIMP_N_LEVELS - 2; lvl >= 0; --lvl) {
        const int rc = cimp_sum_coarse_level(ws->all_hist, lvl);
        if (rc != CIMP_OK) {
            return rc;
        }
    }
    return CIMP_OK;
}

#if CIMP_ENABLE_F32
int cimp_entropy_cells_f32(const CIMP_DescriptorWorkspace *ws, float *out_x85) {
    if (ws == NULL || out_x85 == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
    int32_t cell_cursor = 0;
    for (int32_t lvl = 0; lvl < CIMP_N_LEVELS; ++lvl) {
        const int32_t g = CIMP_ROM_READ_I32(&CIMP_LEVELS[lvl]);
        const int32_t n_cells = g * g;
        const int32_t n_pix = CIMP_ROM_READ_I32(&CIMP_LEVEL_NPIX[lvl]);
        const double inv_n = 1.0 / (double)n_pix;
        const double inv_log2_b = 1.0 / log2((double)CIMP_BINS);

        for (int32_t idx = 0; idx < n_cells; ++idx) {
            const cimp_hist_count_t *cnt =
                ws->all_hist + (size_t)(cell_cursor + idx) * (size_t)CIMP_BINS;
            const cimp_hist_count_t first = cnt[0];
            int equiprob = 1;
            double h_bits = 0.0;
            for (int32_t b = 0; b < CIMP_BINS; ++b) {
                if (cnt[b] != first) {
                    equiprob = 0;
                }
                if (cnt[b] > 0) {
                    const double p = (double)cnt[b] * inv_n;
                    h_bits -= p * log2(p);
                }
            }
            out_x85[cell_cursor + idx] = equiprob ? 1.0f : (float)(h_bits * inv_log2_b);
        }
        cell_cursor += n_cells;
    }
    return CIMP_OK;
}
#endif

q15_t cimp_eval_single_cell_q15(
    const cimp_hist_count_t *cnt,
    int32_t lvl_idx
) {
    if (cnt == NULL || lvl_idx < 0 || lvl_idx >= CIMP_N_LEVELS) {
        return 0;
    }
    const int32_t n_pix = CIMP_ROM_READ_I32(&CIMP_LEVEL_NPIX[lvl_idx]);
    if (n_pix <= 0) {
        return 0;
    }
    const cimp_hist_count_t first = cnt[0];
    int equiprob = 1;
    for (int32_t b = 1; b < CIMP_BINS; ++b) {
        if (cnt[b] != first) {
            equiprob = 0;
            break;
        }
    }
    if (equiprob) {
        return (q15_t)CIMP_FEATURE_Q_MAX;
    }

#if CIMP_SHANNON_LUT_MODE == CIMP_SHANNON_MODE_NLOGN_FULL
    uint64_t sum_nlogn = 0ULL;
    for (int32_t b = 0; b < CIMP_BINS; ++b) {
        const uint32_t c = (uint32_t)cnt[b];
        if (c > 0U) {
            sum_nlogn += (uint64_t)CIMP_ROM_READ_U32(&CIMP_LUT_NLOGN[c]);
        }
    }
    const uint64_t sum_q15 = sum_nlogn << (15 - CIMP_NLOGN_FRAC_BITS);
    const uint64_t level_log2_n_q15 = (uint64_t)CIMP_ROM_READ_U32(&CIMP_LEVEL_LOG2_N_Q15[lvl_idx]);
    const uint64_t total_q15 = (uint64_t)n_pix * level_log2_n_q15;
    const uint64_t numer_q15 = (total_q15 > sum_q15) ? (total_q15 - sum_q15) : 0ULL;
    const uint64_t num = numer_q15 << CIMP_FEATURE_FRAC_BITS;
    const uint64_t den = (uint64_t)n_pix * (uint64_t)CIMP_LOG2_BINS_Q15;
    int64_t eq = (den > 0ULL) ? (int64_t)((num + (den >> 1)) / den) : 0LL;
    if (eq < 0LL) eq = 0LL;
    if (eq > (int64_t)CIMP_FEATURE_Q_MAX) eq = (int64_t)CIMP_FEATURE_Q_MAX;
    return (q15_t)eq;

#elif CIMP_SHANNON_LUT_MODE == CIMP_SHANNON_MODE_TERM_PER_LEVEL
    const int32_t off = CIMP_ROM_READ_I32(&CIMP_LUT_SHANNON_TERM_OFFSETS[lvl_idx]);
    int32_t acc = 0;
    for (int32_t b = 0; b < CIMP_BINS; ++b) {
        acc += (int32_t)CIMP_ROM_READ_I16(&CIMP_LUT_SHANNON_TERM[off + (int32_t)cnt[b]]);
    }
    if (acc < 0) acc = 0;
    if (acc > CIMP_FEATURE_Q_MAX) acc = CIMP_FEATURE_Q_MAX;
    return (q15_t)acc;

#else /* CIMP_SHANNON_MODE_LOG2_MANTISSA */
    const int32_t frac_shift = 31 - CIMP_LOG2_MANTISSA_BITS;
    const uint32_t frac_mask = (UINT32_C(1) << frac_shift) - UINT32_C(1);
    uint64_t sum_clog2c_q15 = 0ULL;
    for (int32_t b = 0; b < CIMP_BINS; ++b) {
        const uint32_t c = (uint32_t)cnt[b];
        if (c > 1U) {
            const int32_t e = 31 - cimp_clz32(c);
            const uint32_t rem_shifted = (c << (31 - e)) & UINT32_C(0x7FFFFFFF);
            const uint32_t idx = rem_shifted >> frac_shift;
            const uint32_t frac_rem = rem_shifted & frac_mask;
            const int32_t y0 = (int32_t)CIMP_ROM_READ_U16(&CIMP_LUT_LOG2_MANTISSA[idx]);
            const int32_t y1 = (int32_t)CIMP_ROM_READ_U16(&CIMP_LUT_LOG2_MANTISSA[idx + 1U]);
            const int64_t interp_delta =
                (((int64_t)(y1 - y0) * (int64_t)frac_rem) + ((int64_t)1 << (frac_shift - 1))) >> frac_shift;
            const uint64_t log2_c_q15 = (uint64_t)(((int64_t)e << 15) + (int64_t)y0 + interp_delta);
            sum_clog2c_q15 += (uint64_t)c * log2_c_q15;
        }
    }
    const uint64_t level_log2_n_q15 = (uint64_t)CIMP_ROM_READ_U32(&CIMP_LEVEL_LOG2_N_Q15[lvl_idx]);
    const uint64_t total_q15 = (uint64_t)n_pix * level_log2_n_q15;
    const uint64_t numer_q15 = (total_q15 > sum_clog2c_q15) ? (total_q15 - sum_clog2c_q15) : 0ULL;
    const uint64_t num = numer_q15 << CIMP_FEATURE_FRAC_BITS;
    const uint64_t den = (uint64_t)n_pix * (uint64_t)CIMP_LOG2_BINS_Q15;
    int64_t eq = (den > 0ULL) ? (int64_t)((num + (den >> 1)) / den) : 0LL;
    if (eq < 0LL) eq = 0LL;
    if (eq > (int64_t)CIMP_FEATURE_Q_MAX) eq = (int64_t)CIMP_FEATURE_Q_MAX;
    return (q15_t)eq;
#endif
}

int cimp_entropy_cells_q15(const CIMP_DescriptorWorkspace *ws, q15_t *out_x85) {
    if (ws == NULL || out_x85 == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
    int32_t cell_cursor = 0;
    for (int32_t lvl = 0; lvl < CIMP_N_LEVELS; ++lvl) {
        const int32_t g = CIMP_ROM_READ_I32(&CIMP_LEVELS[lvl]);
        const int32_t n_cells = g * g;
        for (int32_t idx = 0; idx < n_cells; ++idx) {
            const cimp_hist_count_t *cnt =
                ws->all_hist + (size_t)(cell_cursor + idx) * (size_t)CIMP_BINS;
            out_x85[cell_cursor + idx] = cimp_eval_single_cell_q15(cnt, lvl);
        }
        cell_cursor += n_cells;
    }
    return CIMP_OK;
}

static int32_t cimp_find_level_offset_for_scale(int32_t scale) {
    int32_t off = 0;
    for (int32_t i = 0; i < CIMP_N_LEVELS; ++i) {
        const int32_t g = CIMP_ROM_READ_I32(&CIMP_LEVELS[i]);
        if (g == scale) {
            return off;
        }
        off += g * g;
    }
    return -1;
}

#if CIMP_ENABLE_F32
int cimp_asymmetry_f32(const float *x85, float *out_x9) {
    if (x85 == NULL || out_x9 == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
    int32_t out_idx = 0;
    for (int32_t si = 0; si < CIMP_N_ASYM_SCALES; ++si) {
        const int32_t g = CIMP_ROM_READ_I32(&CIMP_ASYM_SCALES[si]);
        const int32_t off = cimp_find_level_offset_for_scale(g);
        if (off < 0) {
            return CIMP_ERR_SHAPE_MISMATCH;
        }
        const float *grid = x85 + off;
        const float inv_g2 = 1.0f / (float)(g * g);

        for (int32_t ki = 0; ki < CIMP_N_ASYM_KINDS; ++ki) {
            const int32_t kind = CIMP_ROM_READ_I32(&CIMP_ASYM_KINDS[ki]);
            double acc = 0.0;
            for (int32_t r = 0; r < g; ++r) {
                for (int32_t c = 0; c < g; ++c) {
                    const float v = grid[r * g + c];
                    float m = v;
                    if (kind == 0) {
                        m = grid[r * g + (g - 1 - c)];
                    } else if (kind == 1) {
                        m = grid[(g - 1 - r) * g + c];
                    } else {
                        m = grid[(g - 1 - r) * g + (g - 1 - c)];
                    }
                    const double d = (double)v - (double)m;
                    acc += (d < 0.0) ? -d : d;
                }
            }
            out_x9[out_idx++] = (float)(acc * (double)inv_g2);
        }
    }
    return CIMP_OK;
}
#endif

int cimp_asymmetry_q15(const q15_t *x85, q15_t *out_x9) {
    if (x85 == NULL || out_x9 == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
    int32_t out_idx = 0;
    for (int32_t si = 0; si < CIMP_N_ASYM_SCALES; ++si) {
        const int32_t g = CIMP_ROM_READ_I32(&CIMP_ASYM_SCALES[si]);
        const int32_t off = cimp_find_level_offset_for_scale(g);
        if (off < 0) {
            return CIMP_ERR_SHAPE_MISMATCH;
        }
        const q15_t *grid = x85 + off;
        const int32_t div_shift = CIMP_ROM_READ_I32(&CIMP_ASYM_DIV_SHIFT[si]);
        const int32_t inv_q31 = CIMP_ROM_READ_I32(&CIMP_ASYM_INV_Q31[si]);

        for (int32_t ki = 0; ki < CIMP_N_ASYM_KINDS; ++ki) {
            const int32_t kind = CIMP_ROM_READ_I32(&CIMP_ASYM_KINDS[ki]);
            int64_t acc = 0LL;
            for (int32_t r = 0; r < g; ++r) {
                for (int32_t c = 0; c < g; ++c) {
                    const int32_t v = (int32_t)grid[r * g + c];
                    int32_t m = v;
                    if (kind == 0) {
                        m = (int32_t)grid[r * g + (g - 1 - c)];
                    } else if (kind == 1) {
                        m = (int32_t)grid[(g - 1 - r) * g + c];
                    } else {
                        m = (int32_t)grid[(g - 1 - r) * g + (g - 1 - c)];
                    }
                    const int32_t d = v - m;
                    acc += (int64_t)((d < 0) ? -d : d);
                }
            }
            int64_t qval = 0LL;
            if (div_shift >= 0) {
                qval = cimp_round_shift_right_i64(acc, div_shift);
            } else {
                qval = cimp_round_shift_right_i64(acc * (int64_t)inv_q31, 31);
            }
            if (qval < 0LL) qval = 0LL;
            if (qval > (int64_t)CIMP_FEATURE_Q_MAX) qval = (int64_t)CIMP_FEATURE_Q_MAX;
            out_x9[out_idx++] = (q15_t)qval;
        }
    }
    return CIMP_OK;
}

#if CIMP_ENABLE_F32
int cimp_descriptor_x94_f32(
    const uint8_t *img,
    CIMP_DescriptorWorkspace *ws,
    float *out_x94
) {
    if (img == NULL || ws == NULL || out_x94 == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
    const int32_t finest_off = cimp_level_feature_offset(CIMP_N_LEVELS - 1);
    cimp_hist_count_t *fine_hist = ws->all_hist + (size_t)finest_off * (size_t)CIMP_BINS;
    int rc = cimp_quantize_and_fine_hist(
        img, CIMP_IMAGE_SIZE, CIMP_FINEST_GRID, CIMP_BINS, CIMP_LUT_BIN_QUANT, fine_hist
    );
    if (rc != CIMP_OK) {
        return rc;
    }
    rc = cimp_pyramid_sum(ws);
    if (rc != CIMP_OK) {
        return rc;
    }
    rc = cimp_entropy_cells_f32(ws, out_x94);
    if (rc != CIMP_OK) {
        return rc;
    }
    return cimp_asymmetry_f32(out_x94, out_x94 + CIMP_N_X85);
}
#endif

int cimp_descriptor_x94_q15(
    const uint8_t *img,
    CIMP_DescriptorWorkspace *ws,
    q15_t *out_x94
) {
    if (img == NULL || ws == NULL || out_x94 == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
    const int32_t finest_off = cimp_level_feature_offset(CIMP_N_LEVELS - 1);
    cimp_hist_count_t *fine_hist = ws->all_hist + (size_t)finest_off * (size_t)CIMP_BINS;
    int rc = cimp_quantize_and_fine_hist(
        img, CIMP_IMAGE_SIZE, CIMP_FINEST_GRID, CIMP_BINS, CIMP_LUT_BIN_QUANT, fine_hist
    );
    if (rc != CIMP_OK) {
        return rc;
    }
    rc = cimp_pyramid_sum(ws);
    if (rc != CIMP_OK) {
        return rc;
    }
    rc = cimp_entropy_cells_q15(ws, out_x94);
    if (rc != CIMP_OK) {
        return rc;
    }
    return cimp_asymmetry_q15(out_x94, out_x94 + CIMP_N_X85);
}

int cimp_desc_stream_init(CIMP_DescStream *s) {
    if (s == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
    memset(s, 0, sizeof(*s));
    return CIMP_OK;
}

int cimp_desc_stream_push_rows(
    CIMP_DescStream *s,
    const uint8_t *rows,
    int32_t n_rows
) {
    if (s == NULL || rows == NULL || n_rows <= 0) {
        return CIMP_ERR_INVALID_ARG;
    }
    if ((CIMP_IMAGE_SIZE % CIMP_FINEST_GRID) != 0 || CIMP_BINS < 2) {
        return CIMP_ERR_SHAPE_MISMATCH;
    }
    if ((int32_t)s->rows_pushed > CIMP_IMAGE_SIZE ||
        n_rows > (CIMP_IMAGE_SIZE - (int32_t)s->rows_pushed)) {
        return CIMP_ERR_SHAPE_MISMATCH;
    }

    for (int32_t r = 0; r < n_rows; ++r) {
        const uint8_t *row_ptr = rows + (size_t)r * (size_t)CIMP_IMAGE_SIZE;
        for (int32_t c_cell = 0; c_cell < CIMP_FINEST_GRID; ++c_cell) {
            const int32_t x_base = c_cell * CIMP_CELL_SIDE;
            cimp_hist_count_t *cell_hist =
                s->band_hist + (size_t)c_cell * (size_t)CIMP_BINS;
            for (int32_t dx = 0; dx < CIMP_CELL_SIDE; ++dx) {
                const uint8_t bin = CIMP_ROM_READ_U8(&CIMP_LUT_BIN_QUANT[row_ptr[x_base + dx]]);
                cell_hist[bin] += 1;
            }
        }

        s->rows_pushed = (uint16_t)(s->rows_pushed + 1U);
        s->band_row_idx = (uint8_t)(s->band_row_idx + 1U);

        if ((int32_t)s->band_row_idx == CIMP_CELL_SIDE) {
            const int32_t fine_row = (int32_t)s->band_idx;
            q15_t *fine_out_row =
                s->fine_entropies + (size_t)fine_row * (size_t)CIMP_FINEST_GRID;
            for (int32_t c_cell = 0; c_cell < CIMP_FINEST_GRID; ++c_cell) {
                const cimp_hist_count_t *cnt =
                    s->band_hist + (size_t)c_cell * (size_t)CIMP_BINS;
                fine_out_row[c_cell] = cimp_eval_single_cell_q15(cnt, CIMP_N_LEVELS - 1);
            }

#if CIMP_N_LEVELS >= 2
            {
                const int32_t parent_lvl = CIMP_N_LEVELS - 2;
                const int32_t g_parent = CIMP_ROM_READ_I32(&CIMP_LEVELS[parent_lvl]);
                if ((CIMP_FINEST_GRID % g_parent) != 0) {
                    return CIMP_ERR_SHAPE_MISMATCH;
                }
                const int32_t k = CIMP_FINEST_GRID / g_parent;
                const int32_t parent_off = cimp_level_feature_offset(parent_lvl);
                const int32_t pr = fine_row / k;
                cimp_hist_count_t *parent_base =
                    s->coarse_hist + (size_t)parent_off * (size_t)CIMP_BINS;
                for (int32_t c_cell = 0; c_cell < CIMP_FINEST_GRID; ++c_cell) {
                    const int32_t pc = c_cell / k;
                    cimp_hist_count_t *dst =
                        parent_base + (size_t)(pr * g_parent + pc) * (size_t)CIMP_BINS;
                    const cimp_hist_count_t *src =
                        s->band_hist + (size_t)c_cell * (size_t)CIMP_BINS;
                    for (int32_t b = 0; b < CIMP_BINS; ++b) {
                        dst[b] = (cimp_hist_count_t)(dst[b] + src[b]);
                    }
                }
            }
#endif

            memset(s->band_hist, 0, sizeof(s->band_hist));
            s->band_row_idx = 0U;
            s->band_idx = (uint8_t)(s->band_idx + 1U);
        }
    }

    return CIMP_OK;
}

int cimp_desc_stream_finish_q15(
    CIMP_DescStream *s,
    q15_t out_x94[CIMP_N_X94]
) {
    if (s == NULL || out_x94 == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
    if ((int32_t)s->rows_pushed != CIMP_IMAGE_SIZE || s->band_row_idx != 0U) {
        return CIMP_ERR_SHAPE_MISMATCH;
    }

#if CIMP_N_LEVELS >= 3
    for (int32_t lvl = CIMP_N_LEVELS - 3; lvl >= 0; --lvl) {
        const int rc = cimp_sum_coarse_level(s->coarse_hist, lvl);
        if (rc != CIMP_OK) {
            return rc;
        }
    }
#endif

    int32_t cell_cursor = 0;
    for (int32_t lvl = 0; lvl < CIMP_N_LEVELS - 1; ++lvl) {
        const int32_t g = CIMP_ROM_READ_I32(&CIMP_LEVELS[lvl]);
        const int32_t n_cells = g * g;
        for (int32_t idx = 0; idx < n_cells; ++idx) {
            const cimp_hist_count_t *cnt =
                s->coarse_hist + (size_t)(cell_cursor + idx) * (size_t)CIMP_BINS;
            out_x94[cell_cursor + idx] = cimp_eval_single_cell_q15(cnt, lvl);
        }
        cell_cursor += n_cells;
    }

    memcpy(
        out_x94 + cell_cursor,
        s->fine_entropies,
        (size_t)CIMP_FINEST_CELLS * sizeof(q15_t)
    );

    return cimp_asymmetry_q15(out_x94, out_x94 + CIMP_N_X85);
}
