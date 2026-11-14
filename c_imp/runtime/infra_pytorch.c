/**
 * @file infra_pytorch.c
 * @brief Portable C99 CNN inference kernels matching PyTorch layers.
 */
#include "infra_pytorch.h"

#define CIMP_CONV_TILE 256

void cimp_input_norm_u8_f32(
    const uint8_t *img,
    int32_t n_pixels,
    float mean,
    float std,
    float *out
) {
    const float inv_255 = 1.0f / 255.0f;
    const float inv_std = 1.0f / std;
    for (int32_t i = 0; i < n_pixels; ++i) {
        out[i] = ((float)img[i] * inv_255 - mean) * inv_std;
    }
}

void cimp_input_norm_u8_q15(
    const uint8_t *img,
    int32_t n_pixels,
    const q15_t *norm_lut_q,
    q15_t *out_q
) {
    for (int32_t i = 0; i < n_pixels; ++i) {
        out_q[i] = norm_lut_q[img[i]];
    }
}

void cimp_conv2d_f32(
    const float *in,
    int32_t cin,
    int32_t hin,
    int32_t win,
    const float *w,
    const float *bias,
    int32_t cout,
    int32_t kh,
    int32_t kw,
    int32_t stride,
    int32_t pad,
    int relu_fused,
    float *out,
    int32_t hout,
    int32_t wout
) {
    const int32_t in_plane = hin * win;
    const int32_t out_plane = hout * wout;
    const int32_t k_plane = kh * kw;
    double acc_tile[CIMP_CONV_TILE];
    if (kh == 1 && kw == 1 && stride == 1 && pad == 0 && hout == hin && wout == win) {
        for (int32_t oc = 0; oc < cout; ++oc) {
            const double b_val = (bias != NULL) ? (double)bias[oc] : 0.0;
            const float *w_oc = w + (size_t)oc * (size_t)cin;
            float *out_oc = out + (size_t)oc * (size_t)in_plane;
            for (int32_t idx_base = 0; idx_base < in_plane; idx_base += CIMP_CONV_TILE) {
                int32_t tile = in_plane - idx_base;
                if (tile > CIMP_CONV_TILE) {
                    tile = CIMP_CONV_TILE;
                }
                for (int32_t t = 0; t < tile; ++t) {
                    acc_tile[t] = b_val;
                }
                for (int32_t ic = 0; ic < cin; ++ic) {
                    const double w_val = (double)w_oc[ic];
                    const float *in_tile = in + (size_t)ic * (size_t)in_plane + (size_t)idx_base;
                    for (int32_t t = 0; t < tile; ++t) {
                        acc_tile[t] += (double)in_tile[t] * w_val;
                    }
                }
                float *out_tile = out_oc + (size_t)idx_base;
                for (int32_t t = 0; t < tile; ++t) {
                    float res = (float)acc_tile[t];
                    if (relu_fused && res < 0.0f) {
                        res = 0.0f;
                    }
                    out_tile[t] = res;
                }
            }
        }
        return;
    }
    for (int32_t oc = 0; oc < cout; ++oc) {
        const double b_val = (bias != NULL) ? (double)bias[oc] : 0.0;
        const float *w_oc = w + (size_t)oc * (size_t)(cin * k_plane);
        float *out_oc = out + (size_t)oc * (size_t)out_plane;
        for (int32_t oy = 0; oy < hout; ++oy) {
            const int32_t iy_base = oy * stride - pad;
            float *out_row = out_oc + (size_t)oy * (size_t)wout;
            for (int32_t ox_base = 0; ox_base < wout; ox_base += CIMP_CONV_TILE) {
                int32_t tile = wout - ox_base;
                if (tile > CIMP_CONV_TILE) {
                    tile = CIMP_CONV_TILE;
                }
                for (int32_t t = 0; t < tile; ++t) {
                    acc_tile[t] = b_val;
                }
                for (int32_t ic = 0; ic < cin; ++ic) {
                    const float *in_ic = in + (size_t)ic * (size_t)in_plane;
                    const float *w_ic = w_oc + (size_t)ic * (size_t)k_plane;
                    for (int32_t ky = 0; ky < kh; ++ky) {
                        const int32_t iy = iy_base + ky;
                        if ((uint32_t)iy >= (uint32_t)hin) {
                            continue;
                        }
                        const float *in_row = in_ic + (size_t)iy * (size_t)win;
                        const float *w_row = w_ic + (size_t)ky * (size_t)kw;
                        for (int32_t kx = 0; kx < kw; ++kx) {
                            const double w_val = (double)w_row[kx];
                            if (stride == 1) {
                                const int32_t ix0 = ox_base - pad + kx;
                                const int32_t t_min = (ix0 < 0) ? -ix0 : 0;
                                const int32_t t_max = (ix0 + tile > win) ? (win - ix0) : tile;
                                if (t_min < t_max) {
                                    const int32_t len = t_max - t_min;
                                    const float *src = in_row + (size_t)(ix0 + t_min);
                                    double *dst = acc_tile + t_min;
                                    for (int32_t t = 0; t < len; ++t) {
                                        dst[t] += (double)src[t] * w_val;
                                    }
                                }
                            } else {
                                for (int32_t t = 0; t < tile; ++t) {
                                    const int32_t ix = (ox_base + t) * stride - pad + kx;
                                    if ((uint32_t)ix < (uint32_t)win) {
                                        acc_tile[t] += (double)in_row[ix] * w_val;
                                    }
                                }
                            }
                        }
                    }
                }
                float *out_tile = out_row + (size_t)ox_base;
                for (int32_t t = 0; t < tile; ++t) {
                    float res = (float)acc_tile[t];
                    if (relu_fused && res < 0.0f) {
                        res = 0.0f;
                    }
                    out_tile[t] = res;
                }
            }
        }
    }
}

void cimp_conv2d_q15(
    const q15_t *in_q,
    int32_t cin,
    int32_t hin,
    int32_t win,
    const q15_t *w_q,
    const int64_t *bias_acc,
    int32_t cout,
    int32_t kh,
    int32_t kw,
    int32_t stride,
    int32_t pad,
    int32_t out_shift,
    int relu_fused,
    q15_t *out_q,
    int32_t hout,
    int32_t wout
) {
    const int32_t in_plane = hin * win;
    const int32_t out_plane = hout * wout;
    const int32_t k_plane = kh * kw;
    int64_t acc_tile[CIMP_CONV_TILE];
    if (kh == 1 && kw == 1 && stride == 1 && pad == 0 && hout == hin && wout == win) {
        for (int32_t oc = 0; oc < cout; ++oc) {
            const int64_t b_val = (bias_acc != NULL) ? bias_acc[oc] : 0LL;
            const q15_t *w_oc = w_q + (size_t)oc * (size_t)cin;
            q15_t *out_oc = out_q + (size_t)oc * (size_t)in_plane;
            for (int32_t idx_base = 0; idx_base < in_plane; idx_base += CIMP_CONV_TILE) {
                int32_t tile = in_plane - idx_base;
                if (tile > CIMP_CONV_TILE) {
                    tile = CIMP_CONV_TILE;
                }
                for (int32_t t = 0; t < tile; ++t) {
                    acc_tile[t] = b_val;
                }
                for (int32_t ic = 0; ic < cin; ++ic) {
                    const int64_t w_val = (int64_t)w_oc[ic];
                    const q15_t *in_tile = in_q + (size_t)ic * (size_t)in_plane + (size_t)idx_base;
                    for (int32_t t = 0; t < tile; ++t) {
                        acc_tile[t] += (int64_t)in_tile[t] * w_val;
                    }
                }
                q15_t *out_tile = out_oc + (size_t)idx_base;
                for (int32_t t = 0; t < tile; ++t) {
                    int64_t shifted = cimp_round_shift_right_i64(acc_tile[t], out_shift);
                    if (relu_fused && shifted < 0LL) {
                        shifted = 0LL;
                    }
                    out_tile[t] = cimp_sat16(shifted);
                }
            }
        }
        return;
    }
    for (int32_t oc = 0; oc < cout; ++oc) {
        const int64_t b_val = (bias_acc != NULL) ? bias_acc[oc] : 0LL;
        const q15_t *w_oc = w_q + (size_t)oc * (size_t)(cin * k_plane);
        q15_t *out_oc = out_q + (size_t)oc * (size_t)out_plane;
        for (int32_t oy = 0; oy < hout; ++oy) {
            const int32_t iy_base = oy * stride - pad;
            q15_t *out_row = out_oc + (size_t)oy * (size_t)wout;
            for (int32_t ox_base = 0; ox_base < wout; ox_base += CIMP_CONV_TILE) {
                int32_t tile = wout - ox_base;
                if (tile > CIMP_CONV_TILE) {
                    tile = CIMP_CONV_TILE;
                }
                for (int32_t t = 0; t < tile; ++t) {
                    acc_tile[t] = b_val;
                }
                for (int32_t ic = 0; ic < cin; ++ic) {
                    const q15_t *in_ic = in_q + (size_t)ic * (size_t)in_plane;
                    const q15_t *w_ic = w_oc + (size_t)ic * (size_t)k_plane;
                    for (int32_t ky = 0; ky < kh; ++ky) {
                        const int32_t iy = iy_base + ky;
                        if ((uint32_t)iy >= (uint32_t)hin) {
                            continue;
                        }
                        const q15_t *in_row = in_ic + (size_t)iy * (size_t)win;
                        const q15_t *w_row = w_ic + (size_t)ky * (size_t)kw;
                        for (int32_t kx = 0; kx < kw; ++kx) {
                            const int64_t w_val = (int64_t)w_row[kx];
                            if (stride == 1) {
                                const int32_t ix0 = ox_base - pad + kx;
                                const int32_t t_min = (ix0 < 0) ? -ix0 : 0;
                                const int32_t t_max = (ix0 + tile > win) ? (win - ix0) : tile;
                                if (t_min < t_max) {
                                    const int32_t len = t_max - t_min;
                                    const q15_t *src = in_row + (size_t)(ix0 + t_min);
                                    int64_t *dst = acc_tile + t_min;
                                    for (int32_t t = 0; t < len; ++t) {
                                        dst[t] += (int64_t)src[t] * w_val;
                                    }
                                }
                            } else {
                                for (int32_t t = 0; t < tile; ++t) {
                                    const int32_t ix = (ox_base + t) * stride - pad + kx;
                                    if ((uint32_t)ix < (uint32_t)win) {
                                        acc_tile[t] += (int64_t)in_row[ix] * w_val;
                                    }
                                }
                            }
                        }
                    }
                }
                q15_t *out_tile = out_row + (size_t)ox_base;
                for (int32_t t = 0; t < tile; ++t) {
                    int64_t shifted = cimp_round_shift_right_i64(acc_tile[t], out_shift);
                    if (relu_fused && shifted < 0LL) {
                        shifted = 0LL;
                    }
                    out_tile[t] = cimp_sat16(shifted);
                }
            }
        }
    }
}

void cimp_conv2d_q15_w8(
    const q15_t *in_q,
    int32_t cin,
    int32_t hin,
    int32_t win,
    const int8_t *w_q8,
    const int64_t *bias_acc,
    const int32_t *mul_per_ch,
    const int8_t *shift_per_ch,
    int32_t cout,
    int32_t kh,
    int32_t kw,
    int32_t stride,
    int32_t pad,
    int relu_fused,
    q15_t *out_q,
    int32_t hout,
    int32_t wout
) {
    const int32_t in_plane = hin * win;
    const int32_t out_plane = hout * wout;
    const int32_t k_plane = kh * kw;
    int64_t dot_tile[CIMP_CONV_TILE];
    if (kh == 1 && kw == 1 && stride == 1 && pad == 0 && hout == hin && wout == win) {
        for (int32_t oc = 0; oc < cout; ++oc) {
            const int64_t b_val = (bias_acc != NULL) ? bias_acc[oc] : 0LL;
            const int64_t mul = (int64_t)mul_per_ch[oc];
            const int32_t sh = (int32_t)shift_per_ch[oc];
            const int8_t *w_oc = w_q8 + (size_t)oc * (size_t)cin;
            q15_t *out_oc = out_q + (size_t)oc * (size_t)in_plane;
            for (int32_t idx_base = 0; idx_base < in_plane; idx_base += CIMP_CONV_TILE) {
                int32_t tile = in_plane - idx_base;
                if (tile > CIMP_CONV_TILE) {
                    tile = CIMP_CONV_TILE;
                }
                for (int32_t t = 0; t < tile; ++t) {
                    dot_tile[t] = 0LL;
                }
                for (int32_t ic = 0; ic < cin; ++ic) {
                    const int64_t w_val = (int64_t)w_oc[ic];
                    const q15_t *in_tile = in_q + (size_t)ic * (size_t)in_plane + (size_t)idx_base;
                    for (int32_t t = 0; t < tile; ++t) {
                        dot_tile[t] += (int64_t)in_tile[t] * w_val;
                    }
                }
                q15_t *out_tile = out_oc + (size_t)idx_base;
                for (int32_t t = 0; t < tile; ++t) {
                    int64_t scaled = dot_tile[t] * mul + b_val;
                    int64_t shifted = cimp_round_shift_right_i64(scaled, sh);
                    if (relu_fused && shifted < 0LL) {
                        shifted = 0LL;
                    }
                    out_tile[t] = cimp_sat16(shifted);
                }
            }
        }
        return;
    }
    for (int32_t oc = 0; oc < cout; ++oc) {
        const int64_t b_val = (bias_acc != NULL) ? bias_acc[oc] : 0LL;
        const int64_t mul = (int64_t)mul_per_ch[oc];
        const int32_t sh = (int32_t)shift_per_ch[oc];
        const int8_t *w_oc = w_q8 + (size_t)oc * (size_t)(cin * k_plane);
        q15_t *out_oc = out_q + (size_t)oc * (size_t)out_plane;
        for (int32_t oy = 0; oy < hout; ++oy) {
            const int32_t iy_base = oy * stride - pad;
            q15_t *out_row = out_oc + (size_t)oy * (size_t)wout;
            for (int32_t ox_base = 0; ox_base < wout; ox_base += CIMP_CONV_TILE) {
                int32_t tile = wout - ox_base;
                if (tile > CIMP_CONV_TILE) {
                    tile = CIMP_CONV_TILE;
                }
                for (int32_t t = 0; t < tile; ++t) {
                    dot_tile[t] = 0LL;
                }
                for (int32_t ic = 0; ic < cin; ++ic) {
                    const q15_t *in_ic = in_q + (size_t)ic * (size_t)in_plane;
                    const int8_t *w_ic = w_oc + (size_t)ic * (size_t)k_plane;
                    for (int32_t ky = 0; ky < kh; ++ky) {
                        const int32_t iy = iy_base + ky;
                        if ((uint32_t)iy >= (uint32_t)hin) {
                            continue;
                        }
                        const q15_t *in_row = in_ic + (size_t)iy * (size_t)win;
                        const int8_t *w_row = w_ic + (size_t)ky * (size_t)kw;
                        for (int32_t kx = 0; kx < kw; ++kx) {
                            const int64_t w_val = (int64_t)w_row[kx];
                            if (stride == 1) {
                                const int32_t ix0 = ox_base - pad + kx;
                                const int32_t t_min = (ix0 < 0) ? -ix0 : 0;
                                const int32_t t_max = (ix0 + tile > win) ? (win - ix0) : tile;
                                if (t_min < t_max) {
                                    const int32_t len = t_max - t_min;
                                    const q15_t *src = in_row + (size_t)(ix0 + t_min);
                                    int64_t *dst = dot_tile + t_min;
                                    for (int32_t t = 0; t < len; ++t) {
                                        dst[t] += (int64_t)src[t] * w_val;
                                    }
                                }
                            } else {
                                for (int32_t t = 0; t < tile; ++t) {
                                    const int32_t ix = (ox_base + t) * stride - pad + kx;
                                    if ((uint32_t)ix < (uint32_t)win) {
                                        dot_tile[t] += (int64_t)in_row[ix] * w_val;
                                    }
                                }
                            }
                        }
                    }
                }
                q15_t *out_tile = out_row + (size_t)ox_base;
                for (int32_t t = 0; t < tile; ++t) {
                    int64_t scaled = dot_tile[t] * mul + b_val;
                    int64_t shifted = cimp_round_shift_right_i64(scaled, sh);
                    if (relu_fused && shifted < 0LL) {
                        shifted = 0LL;
                    }
                    out_tile[t] = cimp_sat16(shifted);
                }
            }
        }
    }
}

void cimp_dwconv2d_f32(
    const float *in,
    int32_t c,
    int32_t hin,
    int32_t win,
    const float *w,
    const float *bias,
    int32_t kh,
    int32_t kw,
    int32_t stride,
    int32_t pad,
    int relu_fused,
    float *out,
    int32_t hout,
    int32_t wout
) {
    const int32_t in_plane = hin * win;
    const int32_t out_plane = hout * wout;
    const int32_t k_plane = kh * kw;
    double acc_tile[CIMP_CONV_TILE];
    for (int32_t ch = 0; ch < c; ++ch) {
        const double b_val = (bias != NULL) ? (double)bias[ch] : 0.0;
        const float *in_ch = in + (size_t)ch * (size_t)in_plane;
        const float *w_ch = w + (size_t)ch * (size_t)k_plane;
        float *out_ch = out + (size_t)ch * (size_t)out_plane;
        for (int32_t oy = 0; oy < hout; ++oy) {
            const int32_t iy_base = oy * stride - pad;
            float *out_row = out_ch + (size_t)oy * (size_t)wout;
            for (int32_t ox_base = 0; ox_base < wout; ox_base += CIMP_CONV_TILE) {
                int32_t tile = wout - ox_base;
                if (tile > CIMP_CONV_TILE) {
                    tile = CIMP_CONV_TILE;
                }
                for (int32_t t = 0; t < tile; ++t) {
                    acc_tile[t] = b_val;
                }
                for (int32_t ky = 0; ky < kh; ++ky) {
                    const int32_t iy = iy_base + ky;
                    if ((uint32_t)iy >= (uint32_t)hin) {
                        continue;
                    }
                    const float *in_row = in_ch + (size_t)iy * (size_t)win;
                    const float *w_row = w_ch + (size_t)ky * (size_t)kw;
                    for (int32_t kx = 0; kx < kw; ++kx) {
                        const double w_val = (double)w_row[kx];
                        if (stride == 1) {
                            const int32_t ix0 = ox_base - pad + kx;
                            const int32_t t_min = (ix0 < 0) ? -ix0 : 0;
                            const int32_t t_max = (ix0 + tile > win) ? (win - ix0) : tile;
                            if (t_min < t_max) {
                                const int32_t len = t_max - t_min;
                                const float *src = in_row + (size_t)(ix0 + t_min);
                                double *dst = acc_tile + t_min;
                                for (int32_t t = 0; t < len; ++t) {
                                    dst[t] += (double)src[t] * w_val;
                                }
                            }
                        } else {
                            for (int32_t t = 0; t < tile; ++t) {
                                const int32_t ix = (ox_base + t) * stride - pad + kx;
                                if ((uint32_t)ix < (uint32_t)win) {
                                    acc_tile[t] += (double)in_row[ix] * w_val;
                                }
                            }
                        }
                    }
                }
                float *out_tile = out_row + (size_t)ox_base;
                for (int32_t t = 0; t < tile; ++t) {
                    float res = (float)acc_tile[t];
                    if (relu_fused && res < 0.0f) {
                        res = 0.0f;
                    }
                    out_tile[t] = res;
                }
            }
        }
    }
}

void cimp_dwconv2d_q15(
    const q15_t *in_q,
    int32_t c,
    int32_t hin,
    int32_t win,
    const q15_t *w_q,
    const int64_t *bias_acc,
    int32_t kh,
    int32_t kw,
    int32_t stride,
    int32_t pad,
    int32_t out_shift,
    int relu_fused,
    q15_t *out_q,
    int32_t hout,
    int32_t wout
) {
    const int32_t in_plane = hin * win;
    const int32_t out_plane = hout * wout;
    const int32_t k_plane = kh * kw;
    int64_t acc_tile[CIMP_CONV_TILE];
    for (int32_t ch = 0; ch < c; ++ch) {
        const int64_t b_val = (bias_acc != NULL) ? bias_acc[ch] : 0LL;
        const q15_t *in_ch = in_q + (size_t)ch * (size_t)in_plane;
        const q15_t *w_ch = w_q + (size_t)ch * (size_t)k_plane;
        q15_t *out_ch = out_q + (size_t)ch * (size_t)out_plane;
        for (int32_t oy = 0; oy < hout; ++oy) {
            const int32_t iy_base = oy * stride - pad;
            q15_t *out_row = out_ch + (size_t)oy * (size_t)wout;
            for (int32_t ox_base = 0; ox_base < wout; ox_base += CIMP_CONV_TILE) {
                int32_t tile = wout - ox_base;
                if (tile > CIMP_CONV_TILE) {
                    tile = CIMP_CONV_TILE;
                }
                for (int32_t t = 0; t < tile; ++t) {
                    acc_tile[t] = b_val;
                }
                for (int32_t ky = 0; ky < kh; ++ky) {
                    const int32_t iy = iy_base + ky;
                    if ((uint32_t)iy >= (uint32_t)hin) {
                        continue;
                    }
                    const q15_t *in_row = in_ch + (size_t)iy * (size_t)win;
                    const q15_t *w_row = w_ch + (size_t)ky * (size_t)kw;
                    for (int32_t kx = 0; kx < kw; ++kx) {
                        const int64_t w_val = (int64_t)w_row[kx];
                        if (stride == 1) {
                            const int32_t ix0 = ox_base - pad + kx;
                            const int32_t t_min = (ix0 < 0) ? -ix0 : 0;
                            const int32_t t_max = (ix0 + tile > win) ? (win - ix0) : tile;
                            if (t_min < t_max) {
                                const int32_t len = t_max - t_min;
                                const q15_t *src = in_row + (size_t)(ix0 + t_min);
                                int64_t *dst = acc_tile + t_min;
                                for (int32_t t = 0; t < len; ++t) {
                                    dst[t] += (int64_t)src[t] * w_val;
                                }
                            }
                        } else {
                            for (int32_t t = 0; t < tile; ++t) {
                                const int32_t ix = (ox_base + t) * stride - pad + kx;
                                if ((uint32_t)ix < (uint32_t)win) {
                                    acc_tile[t] += (int64_t)in_row[ix] * w_val;
                                }
                            }
                        }
                    }
                }
                q15_t *out_tile = out_row + (size_t)ox_base;
                for (int32_t t = 0; t < tile; ++t) {
                    int64_t shifted = cimp_round_shift_right_i64(acc_tile[t], out_shift);
                    if (relu_fused && shifted < 0LL) {
                        shifted = 0LL;
                    }
                    out_tile[t] = cimp_sat16(shifted);
                }
            }
        }
    }
}

void cimp_dwconv2d_q15_w8(
    const q15_t *in_q,
    int32_t c,
    int32_t hin,
    int32_t win,
    const int8_t *w_q8,
    const int64_t *bias_acc,
    const int32_t *mul_per_ch,
    const int8_t *shift_per_ch,
    int32_t kh,
    int32_t kw,
    int32_t stride,
    int32_t pad,
    int relu_fused,
    q15_t *out_q,
    int32_t hout,
    int32_t wout
) {
    const int32_t in_plane = hin * win;
    const int32_t out_plane = hout * wout;
    const int32_t k_plane = kh * kw;
    int64_t dot_tile[CIMP_CONV_TILE];
    for (int32_t ch = 0; ch < c; ++ch) {
        const int64_t b_val = (bias_acc != NULL) ? bias_acc[ch] : 0LL;
        const int64_t mul = (int64_t)mul_per_ch[ch];
        const int32_t sh = (int32_t)shift_per_ch[ch];
        const q15_t *in_ch = in_q + (size_t)ch * (size_t)in_plane;
        const int8_t *w_ch = w_q8 + (size_t)ch * (size_t)k_plane;
        q15_t *out_ch = out_q + (size_t)ch * (size_t)out_plane;
        for (int32_t oy = 0; oy < hout; ++oy) {
            const int32_t iy_base = oy * stride - pad;
            q15_t *out_row = out_ch + (size_t)oy * (size_t)wout;
            for (int32_t ox_base = 0; ox_base < wout; ox_base += CIMP_CONV_TILE) {
                int32_t tile = wout - ox_base;
                if (tile > CIMP_CONV_TILE) {
                    tile = CIMP_CONV_TILE;
                }
                for (int32_t t = 0; t < tile; ++t) {
                    dot_tile[t] = 0LL;
                }
                for (int32_t ky = 0; ky < kh; ++ky) {
                    const int32_t iy = iy_base + ky;
                    if ((uint32_t)iy >= (uint32_t)hin) {
                        continue;
                    }
                    const q15_t *in_row = in_ch + (size_t)iy * (size_t)win;
                    const int8_t *w_row = w_ch + (size_t)ky * (size_t)kw;
                    for (int32_t kx = 0; kx < kw; ++kx) {
                        const int64_t w_val = (int64_t)w_row[kx];
                        if (stride == 1) {
                            const int32_t ix0 = ox_base - pad + kx;
                            const int32_t t_min = (ix0 < 0) ? -ix0 : 0;
                            const int32_t t_max = (ix0 + tile > win) ? (win - ix0) : tile;
                            if (t_min < t_max) {
                                const int32_t len = t_max - t_min;
                                const q15_t *src = in_row + (size_t)(ix0 + t_min);
                                int64_t *dst = dot_tile + t_min;
                                for (int32_t t = 0; t < len; ++t) {
                                    dst[t] += (int64_t)src[t] * w_val;
                                }
                            }
                        } else {
                            for (int32_t t = 0; t < tile; ++t) {
                                const int32_t ix = (ox_base + t) * stride - pad + kx;
                                if ((uint32_t)ix < (uint32_t)win) {
                                    dot_tile[t] += (int64_t)in_row[ix] * w_val;
                                }
                            }
                        }
                    }
                }
                q15_t *out_tile = out_row + (size_t)ox_base;
                for (int32_t t = 0; t < tile; ++t) {
                    int64_t scaled = dot_tile[t] * mul + b_val;
                    int64_t shifted = cimp_round_shift_right_i64(scaled, sh);
                    if (relu_fused && shifted < 0LL) {
                        shifted = 0LL;
                    }
                    out_tile[t] = cimp_sat16(shifted);
                }
            }
        }
    }
}

void cimp_relu_f32(const float *in, float *out, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        float v = in[i];
        out[i] = (v > 0.0f) ? v : 0.0f;
    }
}

void cimp_relu_q15(const q15_t *in_q, q15_t *out_q, int32_t n, int32_t out_shift) {
    for (int32_t i = 0; i < n; ++i) {
        int64_t v = (int64_t)in_q[i];
        if (v <= 0LL) {
            out_q[i] = 0;
        } else {
            out_q[i] = cimp_sat16(cimp_round_shift_right_i64(v, out_shift));
        }
    }
}

void cimp_hardswish_f32(const float *in, float *out, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        float x = in[i];
        if (x <= -3.0f) {
            out[i] = 0.0f;
        } else if (x >= 3.0f) {
            out[i] = x;
        } else {
            out[i] = x * (x + 3.0f) / 6.0f;
        }
    }
}

void cimp_hardswish_q15(
    const q15_t *in_q,
    q15_t *out_q,
    int32_t n,
    int frac_in,
    int frac_out
) {
    /* 1/6 in Q24: floor(2^24 / 6 + 0.5) = 2796203 */
    const int64_t inv6_q24 = 2796203LL;
    const int64_t three_q = (frac_in >= 0) ? ((int64_t)3 << frac_in) : 0LL;
    const int64_t neg_three_q = -three_q;
    const int32_t identity_shift = (int32_t)(frac_in - frac_out);
    const int32_t prod_shift = (int32_t)(2 * frac_in + 24 - frac_out);

    for (int32_t i = 0; i < n; ++i) {
        int64_t x = (int64_t)in_q[i];
        if (x <= neg_three_q) {
            out_q[i] = 0;
        } else if (x >= three_q) {
            out_q[i] = cimp_sat16(cimp_round_shift_right_i64(x, identity_shift));
        } else {
            int64_t num = x * (x + three_q);
            int64_t scaled = num * inv6_q24;
            out_q[i] = cimp_sat16(cimp_round_shift_right_i64(scaled, prod_shift));
        }
    }
}

void cimp_hardsigmoid_f32(const float *in, float *out, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        float x = in[i];
        if (x <= -3.0f) {
            out[i] = 0.0f;
        } else if (x >= 3.0f) {
            out[i] = 1.0f;
        } else {
            out[i] = x / 6.0f + 0.5f;
        }
    }
}

void cimp_hardsigmoid_q15(
    const q15_t *in_q,
    q15_t *out_q,
    int32_t n,
    int frac_in,
    int frac_out
) {
    const int64_t inv6_q24 = 2796203LL;
    const int64_t three_q = (frac_in >= 0) ? ((int64_t)3 << frac_in) : 0LL;
    const int64_t neg_three_q = -three_q;
    const int64_t one_q = (frac_out >= 15) ? 32767LL : ((int64_t)1 << frac_out);
    const int32_t prod_shift = (int32_t)(frac_in + 24 - frac_out);

    for (int32_t i = 0; i < n; ++i) {
        int64_t x = (int64_t)in_q[i];
        if (x <= neg_three_q) {
            out_q[i] = 0;
        } else if (x >= three_q) {
            out_q[i] = cimp_sat16(one_q);
        } else {
            int64_t scaled = (x + three_q) * inv6_q24;
            int64_t res = cimp_round_shift_right_i64(scaled, prod_shift);
            if (res < 0LL) {
                res = 0LL;
            } else if (res > one_q) {
                res = one_q;
            }
            out_q[i] = cimp_sat16(res);
        }
    }
}

void cimp_maxpool2d_2x2_f32(
    const float *in,
    int32_t c,
    int32_t hin,
    int32_t win,
    float *out,
    int32_t hout,
    int32_t wout
) {
    const int32_t in_plane = hin * win;
    const int32_t out_plane = hout * wout;
    for (int32_t ch = 0; ch < c; ++ch) {
        const float *in_ch = in + (size_t)ch * (size_t)in_plane;
        float *out_ch = out + (size_t)ch * (size_t)out_plane;
        for (int32_t oy = 0; oy < hout; ++oy) {
            const float *r0 = in_ch + (size_t)(oy * 2) * (size_t)win;
            const float *r1 = r0 + win;
            for (int32_t ox = 0; ox < wout; ++ox) {
                int32_t ix = ox * 2;
                float m0 = (r0[ix] > r0[ix + 1]) ? r0[ix] : r0[ix + 1];
                float m1 = (r1[ix] > r1[ix + 1]) ? r1[ix] : r1[ix + 1];
                out_ch[oy * wout + ox] = (m0 > m1) ? m0 : m1;
            }
        }
    }
}

void cimp_maxpool2d_2x2_q15(
    const q15_t *in_q,
    int32_t c,
    int32_t hin,
    int32_t win,
    q15_t *out_q,
    int32_t hout,
    int32_t wout
) {
    const int32_t in_plane = hin * win;
    const int32_t out_plane = hout * wout;
    for (int32_t ch = 0; ch < c; ++ch) {
        const q15_t *in_ch = in_q + (size_t)ch * (size_t)in_plane;
        q15_t *out_ch = out_q + (size_t)ch * (size_t)out_plane;
        for (int32_t oy = 0; oy < hout; ++oy) {
            const q15_t *r0 = in_ch + (size_t)(oy * 2) * (size_t)win;
            const q15_t *r1 = r0 + win;
            for (int32_t ox = 0; ox < wout; ++ox) {
                int32_t ix = ox * 2;
                q15_t m0 = (r0[ix] > r0[ix + 1]) ? r0[ix] : r0[ix + 1];
                q15_t m1 = (r1[ix] > r1[ix + 1]) ? r1[ix] : r1[ix + 1];
                out_ch[oy * wout + ox] = (m0 > m1) ? m0 : m1;
            }
        }
    }
}

void cimp_global_avgpool2d_f32(
    const float *in,
    int32_t c,
    int32_t hin,
    int32_t win,
    float *out
) {
    const int32_t hw = hin * win;
    const double inv_hw = 1.0 / (double)hw;
    for (int32_t ch = 0; ch < c; ++ch) {
        const float *in_ch = in + (size_t)ch * (size_t)hw;
        double acc = 0.0;
        for (int32_t i = 0; i < hw; ++i) {
            acc += (double)in_ch[i];
        }
        out[ch] = (float)(acc * inv_hw);
    }
}

void cimp_global_avgpool2d_q15(
    const q15_t *in_q,
    int32_t c,
    int32_t hin,
    int32_t win,
    int32_t out_shift,
    q15_t *out_q
) {
    const int64_t hw = (int64_t)(hin * win);
    const int64_t half_hw = hw / 2;
    for (int32_t ch = 0; ch < c; ++ch) {
        const q15_t *in_ch = in_q + (size_t)ch * (size_t)hw;
        int64_t sum = 0LL;
        for (int64_t i = 0; i < hw; ++i) {
            sum += (int64_t)in_ch[i];
        }
        int64_t mean_q = (sum >= 0LL) ? ((sum + half_hw) / hw) : -((-sum + half_hw) / hw);
        out_q[ch] = cimp_sat16(cimp_round_shift_right_i64(mean_q, out_shift));
    }
}

void cimp_linear_f32(
    const float *in,
    const float *w,
    const float *bias,
    int32_t cin,
    int32_t cout,
    int relu_fused,
    float *out
) {
    for (int32_t oc = 0; oc < cout; ++oc) {
        double acc = (bias != NULL) ? (double)bias[oc] : 0.0;
        const float *w_row = w + (size_t)oc * (size_t)cin;
        for (int32_t ic = 0; ic < cin; ++ic) {
            acc += (double)in[ic] * (double)w_row[ic];
        }
        float res = (float)acc;
        if (relu_fused && res < 0.0f) {
            res = 0.0f;
        }
        out[oc] = res;
    }
}

void cimp_linear_q15(
    const q15_t *in_q,
    const q15_t *w_q,
    const int64_t *bias_acc,
    int32_t cin,
    int32_t cout,
    int32_t out_shift,
    int relu_fused,
    q15_t *out_q
) {
    for (int32_t oc = 0; oc < cout; ++oc) {
        int64_t acc = (bias_acc != NULL) ? bias_acc[oc] : 0LL;
        const q15_t *w_row = w_q + (size_t)oc * (size_t)cin;
        for (int32_t ic = 0; ic < cin; ++ic) {
            acc += (int64_t)in_q[ic] * (int64_t)w_row[ic];
        }
        int64_t shifted = cimp_round_shift_right_i64(acc, out_shift);
        if (relu_fused && shifted < 0LL) {
            shifted = 0LL;
        }
        out_q[oc] = cimp_sat16(shifted);
    }
}

void cimp_linear_q15_w8(
    const q15_t *in_q,
    const int8_t *w_q8,
    const int64_t *bias_acc,
    const int32_t *mul_per_ch,
    const int8_t *shift_per_ch,
    int32_t cin,
    int32_t cout,
    int relu_fused,
    q15_t *out_q
) {
    for (int32_t oc = 0; oc < cout; ++oc) {
        int64_t dot = 0LL;
        const int8_t *w_row = w_q8 + (size_t)oc * (size_t)cin;
        for (int32_t ic = 0; ic < cin; ++ic) {
            dot += (int64_t)in_q[ic] * (int64_t)w_row[ic];
        }
        int64_t b_val = (bias_acc != NULL) ? bias_acc[oc] : 0LL;
        int64_t scaled = dot * (int64_t)mul_per_ch[oc] + b_val;
        int64_t shifted = cimp_round_shift_right_i64(scaled, (int32_t)shift_per_ch[oc]);
        if (relu_fused && shifted < 0LL) {
            shifted = 0LL;
        }
        out_q[oc] = cimp_sat16(shifted);
    }
}

void cimp_linear_logits_q15(
    const q15_t *in_q,
    const q15_t *w_q,
    const int64_t *bias_acc,
    int32_t cin,
    int32_t cout,
    int32_t out_shift,
    int32_t *out_logits_q
) {
    for (int32_t oc = 0; oc < cout; ++oc) {
        int64_t acc = (bias_acc != NULL) ? bias_acc[oc] : 0LL;
        const q15_t *w_row = w_q + (size_t)oc * (size_t)cin;
        for (int32_t ic = 0; ic < cin; ++ic) {
            acc += (int64_t)in_q[ic] * (int64_t)w_row[ic];
        }
        out_logits_q[oc] = cimp_sat32(cimp_round_shift_right_i64(acc, out_shift));
    }
}

void cimp_linear_logits_q15_w8(
    const q15_t *in_q,
    const int8_t *w_q8,
    const int64_t *bias_acc,
    const int32_t *mul_per_ch,
    const int8_t *shift_per_ch,
    int32_t cin,
    int32_t cout,
    int32_t *out_logits_q
) {
    for (int32_t oc = 0; oc < cout; ++oc) {
        int64_t dot = 0LL;
        const int8_t *w_row = w_q8 + (size_t)oc * (size_t)cin;
        for (int32_t ic = 0; ic < cin; ++ic) {
            dot += (int64_t)in_q[ic] * (int64_t)w_row[ic];
        }
        int64_t b_val = (bias_acc != NULL) ? bias_acc[oc] : 0LL;
        int64_t scaled = dot * (int64_t)mul_per_ch[oc] + b_val;
        out_logits_q[oc] = cimp_sat32(
            cimp_round_shift_right_i64(scaled, (int32_t)shift_per_ch[oc])
        );
    }
}

void cimp_add_f32(
    const float *a,
    const float *b,
    float *out,
    int32_t n,
    int relu_fused
) {
    for (int32_t i = 0; i < n; ++i) {
        float res = a[i] + b[i];
        if (relu_fused && res < 0.0f) {
            res = 0.0f;
        }
        out[i] = res;
    }
}

void cimp_add_q15(
    const q15_t *a_q,
    const q15_t *b_q,
    q15_t *out_q,
    int32_t n,
    int32_t shift_a,
    int32_t shift_b,
    int relu_fused
) {
    for (int32_t i = 0; i < n; ++i) {
        int64_t va = cimp_round_shift_right_i64((int64_t)a_q[i], shift_a);
        int64_t vb = cimp_round_shift_right_i64((int64_t)b_q[i], shift_b);
        int64_t sum = va + vb;
        if (relu_fused && sum < 0LL) {
            sum = 0LL;
        }
        out_q[i] = cimp_sat16(sum);
    }
}

void cimp_scale_channels_f32(
    const float *in,
    const float *scale,
    int32_t c,
    int32_t hw,
    float *out
) {
    for (int32_t ch = 0; ch < c; ++ch) {
        const float s = scale[ch];
        const float *in_ch = in + (size_t)ch * (size_t)hw;
        float *out_ch = out + (size_t)ch * (size_t)hw;
        for (int32_t i = 0; i < hw; ++i) {
            out_ch[i] = in_ch[i] * s;
        }
    }
}

void cimp_scale_channels_q15(
    const q15_t *in_q,
    const q15_t *scale_q,
    int32_t c,
    int32_t hw,
    int32_t out_shift,
    q15_t *out_q
) {
    for (int32_t ch = 0; ch < c; ++ch) {
        const int64_t s = (int64_t)scale_q[ch];
        const q15_t *in_ch = in_q + (size_t)ch * (size_t)hw;
        q15_t *out_ch = out_q + (size_t)ch * (size_t)hw;
        for (int32_t i = 0; i < hw; ++i) {
            int64_t prod = (int64_t)in_ch[i] * s;
            out_ch[i] = cimp_sat16(cimp_round_shift_right_i64(prod, out_shift));
        }
    }
}
