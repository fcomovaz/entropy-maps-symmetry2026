#define ENTROPY_PYRAMID_EXPORTS
#include "entropy_pyramid.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#if defined(_MSC_VER)
#  define EP_RESTRICT __restrict
#  define EP_NOINLINE __declspec(noinline)
#  if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 199901L
#    define EP_INLINE __inline
#  else
#    define EP_INLINE inline
#  endif
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#  define EP_RESTRICT restrict
#  define EP_INLINE inline
#  define EP_NOINLINE
#else
#  define EP_RESTRICT
#  define EP_INLINE
#  define EP_NOINLINE
#endif

/* Tiered fixed stack pools: small (<4 KB, zero __chkstk), medium (~14 KB), general (256 KB). */
#define EP_SMALL_FINE_SLOTS 1024u
#define EP_SMALL_COARSE_SLOTS 384u
#define EP_MED_FINE_SLOTS 4096u
#define EP_MED_COARSE_SLOTS 1536u
#define EP_U16_MAX_COUNT 65535u

typedef struct {
    int n_levels;
    int bins;
    int size;
    int g_max;
    int cell_side;
    int shift;
    int total_features;
    int n_fine_cells;
    int total_coarse_cells;
    int desc_order[EP_MAX_LEVELS];
    int out_offset[EP_MAX_LEVELS];
    int child_sorted_idx[EP_MAX_LEVELS];
    int factor_k[EP_MAX_LEVELS];
    int coarse_cell_offset[EP_MAX_LEVELS];
    uint32_t npix_level[EP_MAX_LEVELS];
    double log2_n_level[EP_MAX_LEVELS];
    double log2_bins;
} PyramidPlan;

static EP_INLINE int pow2_shift_for_bins(int bins) {
    int k = 0;
    int v = bins;
    if (bins <= 0 || (bins & (bins - 1)) != 0) {
        return -1;
    }
    while (v > 1) {
        v >>= 1;
        ++k;
    }
    return 8 - k;
}

static int build_plan(
    int size,
    const int* levels,
    int n_levels,
    int bins,
    PyramidPlan* plan
) {
    int i;
    int s;
    int j;
    int running_feat_off = 0;
    int running_coarse_cells = 0;

    if (size < 1 || levels == NULL || n_levels < 1 || n_levels > EP_MAX_LEVELS) {
        return -1;
    }
    if (bins < 2 || bins > EP_MAX_BINS) {
        return -1;
    }

    plan->n_levels = n_levels;
    plan->bins = bins;
    plan->size = size;
    plan->shift = pow2_shift_for_bins(bins);
    plan->log2_bins = log2((double)bins);

    for (i = 0; i < n_levels; ++i) {
        int g = levels[i];
        if (g < 1 || g > size || g > EP_MAX_FINEST_GRID) {
            return -2;
        }
        plan->out_offset[i] = running_feat_off;
        running_feat_off += g * g;
        plan->desc_order[i] = i;
    }
    plan->total_features = running_feat_off;

    /* Insertion sort indices in strictly descending order of grid level. */
    for (i = 1; i < n_levels; ++i) {
        int key_idx = plan->desc_order[i];
        int key_g = levels[key_idx];
        j = i - 1;
        while (j >= 0 && levels[plan->desc_order[j]] < key_g) {
            plan->desc_order[j + 1] = plan->desc_order[j];
            --j;
        }
        plan->desc_order[j + 1] = key_idx;
    }

    for (s = 1; s < n_levels; ++s) {
        if (levels[plan->desc_order[s - 1]] == levels[plan->desc_order[s]]) {
            return -2;
        }
    }

    plan->g_max = levels[plan->desc_order[0]];
    if (size % plan->g_max != 0) {
        return -3;
    }
    plan->cell_side = size / plan->g_max;
    plan->n_fine_cells = plan->g_max * plan->g_max;

    for (s = 0; s < n_levels; ++s) {
        int orig_idx = plan->desc_order[s];
        int g_s = levels[orig_idx];
        int side_s;
        if (plan->g_max % g_s != 0) {
            return -2;
        }
        side_s = size / g_s;
        plan->npix_level[s] = (uint32_t)side_s * (uint32_t)side_s;
        plan->log2_n_level[s] = log2((double)plan->npix_level[s]);
    }

    plan->child_sorted_idx[0] = -1;
    plan->factor_k[0] = 1;
    plan->coarse_cell_offset[0] = 0;

    for (s = 1; s < n_levels; ++s) {
        int g_curr = levels[plan->desc_order[s]];
        int chosen = -1;
        for (j = s - 1; j >= 0; --j) {
            int g_cand = levels[plan->desc_order[j]];
            if (g_cand % g_curr == 0) {
                chosen = j;
                break;
            }
        }
        if (chosen < 0) {
            return -2;
        }
        plan->child_sorted_idx[s] = chosen;
        plan->factor_k[s] = levels[plan->desc_order[chosen]] / g_curr;
        plan->coarse_cell_offset[s] = running_coarse_cells;
        running_coarse_cells += g_curr * g_curr;
    }

    if (running_coarse_cells > EP_MAX_COARSE_CELLS) {
        return -2;
    }
    plan->total_coarse_cells = running_coarse_cells;
    return 0;
}

static EP_INLINE double entropy_from_u16(
    const uint16_t* EP_RESTRICT counts,
    int bins,
    uint32_t n_pixels,
    double log2_n,
    double log2_bins,
    const double* EP_RESTRICT nlogn_lut
) {
    uint16_t c0;
    uint16_t max_c;
    int equiprobable;
    double s;
    double h;
    int b;

    if (n_pixels == 0u) {
        return 0.0;
    }
    /* nlogn_lut[0] == 0.0 allows branchless accumulation over all bins. */
    c0 = counts[0];
    max_c = 0u;
    equiprobable = 1;
    s = 0.0;
    for (b = 0; b < bins; ++b) {
        uint16_t c = counts[b];
        equiprobable &= (c == c0);
        if (c > max_c) {
            max_c = c;
        }
        s += nlogn_lut[c];
    }
    if ((uint32_t)max_c == n_pixels) {
        return 0.0;
    }
    if (equiprobable) {
        return 1.0;
    }
    h = (log2_n - (s / (double)n_pixels)) / log2_bins;
    return (h < 0.0) ? 0.0 : h;
}

static EP_INLINE double entropy_from_u32(
    const uint32_t* EP_RESTRICT counts,
    int bins,
    uint32_t n_pixels,
    double log2_n,
    double log2_bins,
    const double* EP_RESTRICT nlogn_lut
) {
    uint32_t c0;
    uint32_t max_c;
    int equiprobable;
    double s;
    double h;
    int b;

    if (n_pixels == 0u) {
        return 0.0;
    }
    c0 = counts[0];
    max_c = 0u;
    equiprobable = 1;
    s = 0.0;
    for (b = 0; b < bins; ++b) {
        uint32_t c = counts[b];
        equiprobable &= (c == c0);
        if (c > max_c) {
            max_c = c;
        }
        s += nlogn_lut[c];
    }
    if (max_c == n_pixels) {
        return 0.0;
    }
    if (equiprobable) {
        return 1.0;
    }
    h = (log2_n - (s / (double)n_pixels)) / log2_bins;
    return (h < 0.0) ? 0.0 : h;
}

static EP_INLINE void sum_u16_to_u32(
    const uint16_t* EP_RESTRICT src,
    int g,
    int k,
    int bins,
    uint32_t* EP_RESTRICT dst
) {
    int r;
    int c;
    int b;
    int g_child = g * k;

    if (k == 2) {
        for (r = 0; r < g; ++r) {
            int row0 = (2 * r) * g_child;
            int row1 = (2 * r + 1) * g_child;
            for (c = 0; c < g; ++c) {
                const uint16_t* EP_RESTRICT c00 = src + ((size_t)(row0 + 2 * c) * (size_t)bins);
                const uint16_t* EP_RESTRICT c01 = src + ((size_t)(row0 + 2 * c + 1) * (size_t)bins);
                const uint16_t* EP_RESTRICT c10 = src + ((size_t)(row1 + 2 * c) * (size_t)bins);
                const uint16_t* EP_RESTRICT c11 = src + ((size_t)(row1 + 2 * c + 1) * (size_t)bins);
                uint32_t* EP_RESTRICT dst_cell = dst + ((size_t)(r * g + c) * (size_t)bins);
                for (b = 0; b < bins; ++b) {
                    dst_cell[b] = (uint32_t)c00[b] + (uint32_t)c01[b]
                                + (uint32_t)c10[b] + (uint32_t)c11[b];
                }
            }
        }
    } else {
        int dr;
        int dc;
        for (r = 0; r < g; ++r) {
            for (c = 0; c < g; ++c) {
                uint32_t* EP_RESTRICT dst_cell = dst + ((size_t)(r * g + c) * (size_t)bins);
                const uint16_t* EP_RESTRICT first = src + ((size_t)((r * k) * g_child + (c * k)) * (size_t)bins);
                for (b = 0; b < bins; ++b) {
                    dst_cell[b] = (uint32_t)first[b];
                }
                for (dr = 0; dr < k; ++dr) {
                    for (dc = 0; dc < k; ++dc) {
                        if (dr == 0 && dc == 0) {
                            continue;
                        }
                        {
                            const uint16_t* EP_RESTRICT child =
                                src + ((size_t)((r * k + dr) * g_child + (c * k + dc)) * (size_t)bins);
                            for (b = 0; b < bins; ++b) {
                                dst_cell[b] += (uint32_t)child[b];
                            }
                        }
                    }
                }
            }
        }
    }
}

static EP_INLINE void sum_u32_to_u32(
    const uint32_t* EP_RESTRICT src,
    int g,
    int k,
    int bins,
    uint32_t* EP_RESTRICT dst
) {
    int r;
    int c;
    int b;
    int g_child = g * k;

    if (k == 2) {
        for (r = 0; r < g; ++r) {
            int row0 = (2 * r) * g_child;
            int row1 = (2 * r + 1) * g_child;
            for (c = 0; c < g; ++c) {
                const uint32_t* EP_RESTRICT c00 = src + ((size_t)(row0 + 2 * c) * (size_t)bins);
                const uint32_t* EP_RESTRICT c01 = src + ((size_t)(row0 + 2 * c + 1) * (size_t)bins);
                const uint32_t* EP_RESTRICT c10 = src + ((size_t)(row1 + 2 * c) * (size_t)bins);
                const uint32_t* EP_RESTRICT c11 = src + ((size_t)(row1 + 2 * c + 1) * (size_t)bins);
                uint32_t* EP_RESTRICT dst_cell = dst + ((size_t)(r * g + c) * (size_t)bins);
                for (b = 0; b < bins; ++b) {
                    dst_cell[b] = c00[b] + c01[b] + c10[b] + c11[b];
                }
            }
        }
    } else {
        int dr;
        int dc;
        for (r = 0; r < g; ++r) {
            for (c = 0; c < g; ++c) {
                uint32_t* EP_RESTRICT dst_cell = dst + ((size_t)(r * g + c) * (size_t)bins);
                const uint32_t* EP_RESTRICT first = src + ((size_t)((r * k) * g_child + (c * k)) * (size_t)bins);
                for (b = 0; b < bins; ++b) {
                    dst_cell[b] = first[b];
                }
                for (dr = 0; dr < k; ++dr) {
                    for (dc = 0; dc < k; ++dc) {
                        if (dr == 0 && dc == 0) {
                            continue;
                        }
                        {
                            const uint32_t* EP_RESTRICT child =
                                src + ((size_t)((r * k + dr) * g_child + (c * k + dc)) * (size_t)bins);
                            for (b = 0; b < bins; ++b) {
                                dst_cell[b] += child[b];
                            }
                        }
                    }
                }
            }
        }
    }
}

static EP_INLINE void fine_hist_pass_u16(
    const uint8_t* EP_RESTRICT img_n,
    int size,
    int g_max,
    int cell_side,
    int bins,
    int shift,
    const uint8_t* EP_RESTRICT qlut,
    uint16_t* EP_RESTRICT hist_fine
) {
    int br;
    int dy;
    int bc;
    int dx;

    memset(hist_fine, 0, (size_t)g_max * (size_t)g_max * (size_t)bins * sizeof(uint16_t));

    if (shift >= 0) {
        for (br = 0; br < g_max; ++br) {
            uint16_t* row_hists = hist_fine + (size_t)br * (size_t)g_max * (size_t)bins;
            for (dy = 0; dy < cell_side; ++dy) {
                const uint8_t* row_px =
                    img_n + ((size_t)br * (size_t)cell_side + (size_t)dy) * (size_t)size;
                for (bc = 0; bc < g_max; ++bc) {
                    uint16_t* cell_hist = row_hists + (size_t)bc * (size_t)bins;
                    const uint8_t* seg = row_px + (size_t)bc * (size_t)cell_side;
                    for (dx = 0; dx < cell_side; ++dx) {
                        cell_hist[(uint8_t)(seg[dx] >> shift)]++;
                    }
                }
            }
        }
    } else {
        for (br = 0; br < g_max; ++br) {
            uint16_t* row_hists = hist_fine + (size_t)br * (size_t)g_max * (size_t)bins;
            for (dy = 0; dy < cell_side; ++dy) {
                const uint8_t* row_px =
                    img_n + ((size_t)br * (size_t)cell_side + (size_t)dy) * (size_t)size;
                for (bc = 0; bc < g_max; ++bc) {
                    uint16_t* cell_hist = row_hists + (size_t)bc * (size_t)bins;
                    const uint8_t* seg = row_px + (size_t)bc * (size_t)cell_side;
                    for (dx = 0; dx < cell_side; ++dx) {
                        cell_hist[qlut[seg[dx]]]++;
                    }
                }
            }
        }
    }
}

static EP_INLINE void fine_hist_pass_u32(
    const uint8_t* EP_RESTRICT img_n,
    int size,
    int g_max,
    int cell_side,
    int bins,
    int shift,
    const uint8_t* EP_RESTRICT qlut,
    uint32_t* EP_RESTRICT hist_fine
) {
    int br;
    int dy;
    int bc;
    int dx;

    memset(hist_fine, 0, (size_t)g_max * (size_t)g_max * (size_t)bins * sizeof(uint32_t));

    if (shift >= 0) {
        for (br = 0; br < g_max; ++br) {
            uint32_t* row_hists = hist_fine + (size_t)br * (size_t)g_max * (size_t)bins;
            for (dy = 0; dy < cell_side; ++dy) {
                const uint8_t* row_px =
                    img_n + ((size_t)br * (size_t)cell_side + (size_t)dy) * (size_t)size;
                for (bc = 0; bc < g_max; ++bc) {
                    uint32_t* cell_hist = row_hists + (size_t)bc * (size_t)bins;
                    const uint8_t* seg = row_px + (size_t)bc * (size_t)cell_side;
                    for (dx = 0; dx < cell_side; ++dx) {
                        cell_hist[(uint8_t)(seg[dx] >> shift)]++;
                    }
                }
            }
        }
    } else {
        for (br = 0; br < g_max; ++br) {
            uint32_t* row_hists = hist_fine + (size_t)br * (size_t)g_max * (size_t)bins;
            for (dy = 0; dy < cell_side; ++dy) {
                const uint8_t* row_px =
                    img_n + ((size_t)br * (size_t)cell_side + (size_t)dy) * (size_t)size;
                for (bc = 0; bc < g_max; ++bc) {
                    uint32_t* cell_hist = row_hists + (size_t)bc * (size_t)bins;
                    const uint8_t* seg = row_px + (size_t)bc * (size_t)cell_side;
                    for (dx = 0; dx < cell_side; ++dx) {
                        cell_hist[qlut[seg[dx]]]++;
                    }
                }
            }
        }
    }
}

static EP_INLINE void execute_batch_u16(
    const uint8_t* EP_RESTRICT img,
    int n_images,
    const int* EP_RESTRICT levels,
    const PyramidPlan* EP_RESTRICT plan,
    const uint8_t* EP_RESTRICT qlut,
    const double* EP_RESTRICT nlogn_lut,
    uint16_t* EP_RESTRICT hist_fine,
    uint32_t* EP_RESTRICT coarse_pool,
    double* EP_RESTRICT out_features
) {
    int n;
    size_t img_stride = (size_t)plan->size * (size_t)plan->size;

    for (n = 0; n < n_images; ++n) {
        const uint8_t* img_n = img + (size_t)n * img_stride;
        double* out_n = out_features + (size_t)n * (size_t)plan->total_features;
        int idx;
        int s;
        int feat_off_0 = plan->out_offset[plan->desc_order[0]];
        uint32_t npix_0 = plan->npix_level[0];
        double log2_n_0 = plan->log2_n_level[0];

        fine_hist_pass_u16(
            img_n,
            plan->size,
            plan->g_max,
            plan->cell_side,
            plan->bins,
            plan->shift,
            qlut,
            hist_fine
        );

        for (idx = 0; idx < plan->n_fine_cells; ++idx) {
            out_n[feat_off_0 + idx] = entropy_from_u16(
                hist_fine + (size_t)idx * (size_t)plan->bins,
                plan->bins,
                npix_0,
                log2_n_0,
                plan->log2_bins,
                nlogn_lut
            );
        }

        for (s = 1; s < plan->n_levels; ++s) {
            int g = levels[plan->desc_order[s]];
            int n_cells_s = g * g;
            int src_s = plan->child_sorted_idx[s];
            int k = plan->factor_k[s];
            int feat_off_s = plan->out_offset[plan->desc_order[s]];
            uint32_t npix_s = plan->npix_level[s];
            double log2_n_s = plan->log2_n_level[s];
            uint32_t* dst_level =
                coarse_pool + (size_t)plan->coarse_cell_offset[s] * (size_t)plan->bins;

            if (src_s == 0) {
                sum_u16_to_u32(hist_fine, g, k, plan->bins, dst_level);
            } else {
                const uint32_t* src_level =
                    coarse_pool + (size_t)plan->coarse_cell_offset[src_s] * (size_t)plan->bins;
                sum_u32_to_u32(src_level, g, k, plan->bins, dst_level);
            }

            for (idx = 0; idx < n_cells_s; ++idx) {
                out_n[feat_off_s + idx] = entropy_from_u32(
                    dst_level + (size_t)idx * (size_t)plan->bins,
                    plan->bins,
                    npix_s,
                    log2_n_s,
                    plan->log2_bins,
                    nlogn_lut
                );
            }
        }
    }
}

static EP_NOINLINE void run_batch_u16_small(
    const uint8_t* EP_RESTRICT img,
    int n_images,
    const int* EP_RESTRICT levels,
    const PyramidPlan* EP_RESTRICT plan,
    const uint8_t* EP_RESTRICT qlut,
    const double* EP_RESTRICT nlogn_lut,
    double* EP_RESTRICT out_features
) {
    uint16_t hist_fine[EP_SMALL_FINE_SLOTS];
    uint32_t coarse_pool[EP_SMALL_COARSE_SLOTS];
    execute_batch_u16(
        img, n_images, levels, plan, qlut, nlogn_lut, hist_fine, coarse_pool, out_features
    );
}

static EP_NOINLINE void run_batch_u16_medium(
    const uint8_t* EP_RESTRICT img,
    int n_images,
    const int* EP_RESTRICT levels,
    const PyramidPlan* EP_RESTRICT plan,
    const uint8_t* EP_RESTRICT qlut,
    const double* EP_RESTRICT nlogn_lut,
    double* EP_RESTRICT out_features
) {
    uint16_t hist_fine[EP_MED_FINE_SLOTS];
    uint32_t coarse_pool[EP_MED_COARSE_SLOTS];
    execute_batch_u16(
        img, n_images, levels, plan, qlut, nlogn_lut, hist_fine, coarse_pool, out_features
    );
}

static EP_NOINLINE void run_batch_u16_general(
    const uint8_t* EP_RESTRICT img,
    int n_images,
    const int* EP_RESTRICT levels,
    const PyramidPlan* EP_RESTRICT plan,
    const uint8_t* EP_RESTRICT qlut,
    const double* EP_RESTRICT nlogn_lut,
    double* EP_RESTRICT out_features
) {
    uint16_t hist_fine[EP_MAX_FINEST_CELLS * EP_MAX_BINS];
    uint32_t coarse_pool[EP_MAX_COARSE_CELLS * EP_MAX_BINS];
    execute_batch_u16(
        img, n_images, levels, plan, qlut, nlogn_lut, hist_fine, coarse_pool, out_features
    );
}

static EP_NOINLINE void run_batch_u32_general(
    const uint8_t* EP_RESTRICT img,
    int n_images,
    const int* EP_RESTRICT levels,
    const PyramidPlan* EP_RESTRICT plan,
    const uint8_t* EP_RESTRICT qlut,
    const double* EP_RESTRICT nlogn_lut,
    double* EP_RESTRICT out_features
) {
    uint32_t hist_fine[EP_MAX_FINEST_CELLS * EP_MAX_BINS];
    uint32_t coarse_pool[EP_MAX_COARSE_CELLS * EP_MAX_BINS];
    int n;
    size_t img_stride = (size_t)plan->size * (size_t)plan->size;

    for (n = 0; n < n_images; ++n) {
        const uint8_t* img_n = img + (size_t)n * img_stride;
        double* out_n = out_features + (size_t)n * (size_t)plan->total_features;
        int idx;
        int s;
        int feat_off_0 = plan->out_offset[plan->desc_order[0]];
        uint32_t npix_0 = plan->npix_level[0];
        double log2_n_0 = plan->log2_n_level[0];

        fine_hist_pass_u32(
            img_n,
            plan->size,
            plan->g_max,
            plan->cell_side,
            plan->bins,
            plan->shift,
            qlut,
            hist_fine
        );

        for (idx = 0; idx < plan->n_fine_cells; ++idx) {
            out_n[feat_off_0 + idx] = entropy_from_u32(
                hist_fine + (size_t)idx * (size_t)plan->bins,
                plan->bins,
                npix_0,
                log2_n_0,
                plan->log2_bins,
                nlogn_lut
            );
        }

        for (s = 1; s < plan->n_levels; ++s) {
            int g = levels[plan->desc_order[s]];
            int n_cells_s = g * g;
            int src_s = plan->child_sorted_idx[s];
            int k = plan->factor_k[s];
            int feat_off_s = plan->out_offset[plan->desc_order[s]];
            uint32_t npix_s = plan->npix_level[s];
            double log2_n_s = plan->log2_n_level[s];
            uint32_t* dst_level =
                coarse_pool + (size_t)plan->coarse_cell_offset[s] * (size_t)plan->bins;

            if (src_s == 0) {
                sum_u32_to_u32(hist_fine, g, k, plan->bins, dst_level);
            } else {
                const uint32_t* src_level =
                    coarse_pool + (size_t)plan->coarse_cell_offset[src_s] * (size_t)plan->bins;
                sum_u32_to_u32(src_level, g, k, plan->bins, dst_level);
            }

            for (idx = 0; idx < n_cells_s; ++idx) {
                out_n[feat_off_s + idx] = entropy_from_u32(
                    dst_level + (size_t)idx * (size_t)plan->bins,
                    plan->bins,
                    npix_s,
                    log2_n_s,
                    plan->log2_bins,
                    nlogn_lut
                );
            }
        }
    }
}

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
) {
    PyramidPlan plan;
    int rc;

    if (n_images < 0 || qlut == NULL || nlogn_lut == NULL) {
        return -1;
    }
    rc = build_plan(size, levels, n_levels, bins, &plan);
    if (rc != 0) {
        return rc;
    }
    if (n_images == 0) {
        return 0;
    }
    if (img == NULL || out_features == NULL) {
        return -1;
    }

    if (plan.npix_level[0] <= EP_U16_MAX_COUNT) {
        size_t fine_slots = (size_t)plan.n_fine_cells * (size_t)plan.bins;
        size_t coarse_slots = (size_t)plan.total_coarse_cells * (size_t)plan.bins;
        if (fine_slots <= EP_SMALL_FINE_SLOTS && coarse_slots <= EP_SMALL_COARSE_SLOTS) {
            run_batch_u16_small(img, n_images, levels, &plan, qlut, nlogn_lut, out_features);
        } else if (fine_slots <= EP_MED_FINE_SLOTS && coarse_slots <= EP_MED_COARSE_SLOTS) {
            run_batch_u16_medium(img, n_images, levels, &plan, qlut, nlogn_lut, out_features);
        } else {
            run_batch_u16_general(img, n_images, levels, &plan, qlut, nlogn_lut, out_features);
        }
    } else {
        run_batch_u32_general(img, n_images, levels, &plan, qlut, nlogn_lut, out_features);
    }

    return 0;
}

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
) {
    (void)img;
    (void)n_images;
    (void)size;
    (void)levels;
    (void)n_levels;
    (void)bins;
    (void)qlut;
    (void)nlogn_lut;
    (void)out_features;
    return 0;
}

EP_API int entropy_pyramid_stage_fine_hist_u8(
    const uint8_t* img,
    int n_images,
    int size,
    int g_max,
    int bins,
    const uint8_t* qlut,
    uint32_t* out_fine_hist
) {
    int n;
    int cell_side;
    int shift;
    size_t img_stride;
    size_t fine_slots;

    if (n_images < 0 || size < 1 || g_max < 1 || g_max > size || g_max > EP_MAX_FINEST_GRID) {
        return -1;
    }
    if (bins < 2 || bins > EP_MAX_BINS || qlut == NULL) {
        return -1;
    }
    if (size % g_max != 0) {
        return -3;
    }
    if (n_images == 0) {
        return 0;
    }
    if (img == NULL || out_fine_hist == NULL) {
        return -1;
    }

    cell_side = size / g_max;
    shift = pow2_shift_for_bins(bins);
    img_stride = (size_t)size * (size_t)size;
    fine_slots = (size_t)g_max * (size_t)g_max * (size_t)bins;

    for (n = 0; n < n_images; ++n) {
        fine_hist_pass_u32(
            img + (size_t)n * img_stride,
            size,
            g_max,
            cell_side,
            bins,
            shift,
            qlut,
            out_fine_hist + (size_t)n * fine_slots
        );
    }
    return 0;
}

EP_API int entropy_pyramid_stage_sum(
    const uint32_t* fine_hist,
    int n_images,
    int size,
    const int* levels,
    int n_levels,
    int bins,
    uint32_t* out_all_hist
) {
    PyramidPlan plan;
    int rc;
    int n;
    size_t fine_slots;
    size_t total_slots;

    if (n_images < 0) {
        return -1;
    }
    rc = build_plan(size, levels, n_levels, bins, &plan);
    if (rc != 0) {
        return rc;
    }
    if (n_images == 0) {
        return 0;
    }
    if (fine_hist == NULL || out_all_hist == NULL) {
        return -1;
    }

    fine_slots = (size_t)plan.n_fine_cells * (size_t)plan.bins;
    total_slots = (size_t)plan.total_features * (size_t)plan.bins;

    for (n = 0; n < n_images; ++n) {
        const uint32_t* src_fine = fine_hist + (size_t)n * fine_slots;
        uint32_t* dst_all = out_all_hist + (size_t)n * total_slots;
        uint32_t* dst_0 = dst_all + (size_t)plan.out_offset[plan.desc_order[0]] * (size_t)plan.bins;
        int s;

        memcpy(dst_0, src_fine, fine_slots * sizeof(uint32_t));
        for (s = 1; s < plan.n_levels; ++s) {
            int orig_s = plan.desc_order[s];
            int g = levels[orig_s];
            int src_s = plan.child_sorted_idx[s];
            int orig_src = plan.desc_order[src_s];
            int k = plan.factor_k[s];
            const uint32_t* src_level =
                dst_all + (size_t)plan.out_offset[orig_src] * (size_t)plan.bins;
            uint32_t* dst_level =
                dst_all + (size_t)plan.out_offset[orig_s] * (size_t)plan.bins;
            sum_u32_to_u32(src_level, g, k, plan.bins, dst_level);
        }
    }
    return 0;
}

EP_API int entropy_pyramid_stage_entropy(
    const uint32_t* all_hist,
    int n_images,
    int size,
    const int* levels,
    int n_levels,
    int bins,
    const double* nlogn_lut,
    double* out_features
) {
    PyramidPlan plan;
    int rc;
    int n;
    size_t total_slots;

    if (n_images < 0 || nlogn_lut == NULL) {
        return -1;
    }
    rc = build_plan(size, levels, n_levels, bins, &plan);
    if (rc != 0) {
        return rc;
    }
    if (n_images == 0) {
        return 0;
    }
    if (all_hist == NULL || out_features == NULL) {
        return -1;
    }

    total_slots = (size_t)plan.total_features * (size_t)plan.bins;
    for (n = 0; n < n_images; ++n) {
        const uint32_t* hist_n = all_hist + (size_t)n * total_slots;
        double* out_n = out_features + (size_t)n * (size_t)plan.total_features;
        int s;
        for (s = 0; s < plan.n_levels; ++s) {
            int orig_s = plan.desc_order[s];
            int g = levels[orig_s];
            int n_cells_s = g * g;
            int feat_off_s = plan.out_offset[orig_s];
            uint32_t npix_s = plan.npix_level[s];
            double log2_n_s = plan.log2_n_level[s];
            int idx;
            for (idx = 0; idx < n_cells_s; ++idx) {
                out_n[feat_off_s + idx] = entropy_from_u32(
                    hist_n + (size_t)(feat_off_s + idx) * (size_t)plan.bins,
                    plan.bins,
                    npix_s,
                    log2_n_s,
                    plan.log2_bins,
                    nlogn_lut
                );
            }
        }
    }
    return 0;
}

EP_API void entropy_pyramid_limits(
    int* out_max_finest_grid,
    int* out_max_bins,
    int* out_max_levels
) {
    if (out_max_finest_grid != NULL) {
        *out_max_finest_grid = EP_MAX_FINEST_GRID;
    }
    if (out_max_bins != NULL) {
        *out_max_bins = EP_MAX_BINS;
    }
    if (out_max_levels != NULL) {
        *out_max_levels = EP_MAX_LEVELS;
    }
}
