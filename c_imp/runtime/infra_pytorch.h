/**
 * @file infra_pytorch.h
 * @brief Lightweight C99 CNN inference kernels mirroring PyTorch layers.
 *
 * All tensor buffers use row-major NCHW layout (with N=1, i.e., [C, H, W]).
 * Fixed-point `_q15` kernels operate strictly in integer arithmetic (no <math.h>,
 * no heap allocation) and support both 16-bit uniform Q-weights and 8-bit
 * per-channel quantized weights (`_w8`).
 */
#ifndef CIMP_INFRA_PYTORCH_H
#define CIMP_INFRA_PYTORCH_H

#include <stddef.h>
#include <stdint.h>

#include "cimp_fixed_point.h"

/** Input uint8 image [H*W] -> normalized float32 [1, H, W]: (px/255 - mean) / std. */
void cimp_input_norm_u8_f32(
    const uint8_t *img,
    int32_t n_pixels,
    float mean,
    float std,
    float *out
);

/** Input uint8 image [H*W] -> Q15/Qm.n [1, H, W] via 256-entry precomputed LUT. */
void cimp_input_norm_u8_q15(
    const uint8_t *img,
    int32_t n_pixels,
    const q15_t *norm_lut_q,
    q15_t *out_q
);

/** Standard Conv2d (groups=1) in float32 with optional fused ReLU. */
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
);

/** Standard Conv2d (groups=1) in fixed-point Q15 activations and 16-bit weights. */
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
);

/** Standard Conv2d (groups=1) with Q15 activations and 8-bit per-channel weights. */
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
);

/** Depthwise Conv2d (groups=C) in float32 with optional fused ReLU. */
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
);

/** Depthwise Conv2d (groups=C) in fixed-point Q15 activations and 16-bit weights. */
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
);

/** Depthwise Conv2d (groups=C) with Q15 activations and 8-bit per-channel weights. */
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
);

/** Elementwise ReLU in float32 (may run in-place when out == in). */
void cimp_relu_f32(const float *in, float *out, int32_t n);

/** Elementwise ReLU in Q15 (with optional shift to adjust frac_bits). */
void cimp_relu_q15(const q15_t *in_q, q15_t *out_q, int32_t n, int32_t out_shift);

/** Elementwise Hardswish in float32. */
void cimp_hardswish_f32(const float *in, float *out, int32_t n);

/** Elementwise Hardswish in pure integer Qm.n arithmetic (no <math.h>). */
void cimp_hardswish_q15(
    const q15_t *in_q,
    q15_t *out_q,
    int32_t n,
    int frac_in,
    int frac_out
);

/** Elementwise Hardsigmoid in float32. */
void cimp_hardsigmoid_f32(const float *in, float *out, int32_t n);

/** Elementwise Hardsigmoid in pure integer Qm.n arithmetic (no <math.h>). */
void cimp_hardsigmoid_q15(
    const q15_t *in_q,
    q15_t *out_q,
    int32_t n,
    int frac_in,
    int frac_out
);

/** 2x2 stride-2 MaxPool2d in float32. */
void cimp_maxpool2d_2x2_f32(
    const float *in,
    int32_t c,
    int32_t hin,
    int32_t win,
    float *out,
    int32_t hout,
    int32_t wout
);

/** 2x2 stride-2 MaxPool2d in Q15. */
void cimp_maxpool2d_2x2_q15(
    const q15_t *in_q,
    int32_t c,
    int32_t hin,
    int32_t win,
    q15_t *out_q,
    int32_t hout,
    int32_t wout
);

/** Global AdaptiveAvgPool2d(1) in float32: [C, H, W] -> [C]. */
void cimp_global_avgpool2d_f32(
    const float *in,
    int32_t c,
    int32_t hin,
    int32_t win,
    float *out
);

/** Global AdaptiveAvgPool2d(1) in Q15 with integer rounded spatial mean and out_shift. */
void cimp_global_avgpool2d_q15(
    const q15_t *in_q,
    int32_t c,
    int32_t hin,
    int32_t win,
    int32_t out_shift,
    q15_t *out_q
);

/** Fully connected Linear layer in float32 with optional fused ReLU. */
void cimp_linear_f32(
    const float *in,
    const float *w,
    const float *bias,
    int32_t cin,
    int32_t cout,
    int relu_fused,
    float *out
);

/** Fully connected Linear layer producing Q15 activations (16-bit weights). */
void cimp_linear_q15(
    const q15_t *in_q,
    const q15_t *w_q,
    const int64_t *bias_acc,
    int32_t cin,
    int32_t cout,
    int32_t out_shift,
    int relu_fused,
    q15_t *out_q
);

/** Fully connected Linear layer producing Q15 activations (8-bit per-channel weights). */
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
);

/** Final Linear classifier head producing int32 logits (16-bit weights). */
void cimp_linear_logits_q15(
    const q15_t *in_q,
    const q15_t *w_q,
    const int64_t *bias_acc,
    int32_t cin,
    int32_t cout,
    int32_t out_shift,
    int32_t *out_logits_q
);

/** Final Linear classifier head producing int32 logits (8-bit per-channel weights). */
void cimp_linear_logits_q15_w8(
    const q15_t *in_q,
    const int8_t *w_q8,
    const int64_t *bias_acc,
    const int32_t *mul_per_ch,
    const int8_t *shift_per_ch,
    int32_t cin,
    int32_t cout,
    int32_t *out_logits_q
);

/** Elementwise residual addition in float32 with optional fused ReLU. */
void cimp_add_f32(
    const float *a,
    const float *b,
    float *out,
    int32_t n,
    int relu_fused
);

/** Elementwise residual addition in Q15 aligning two fractional precisions before sum. */
void cimp_add_q15(
    const q15_t *a_q,
    const q15_t *b_q,
    q15_t *out_q,
    int32_t n,
    int32_t shift_a,
    int32_t shift_b,
    int relu_fused
);

/** Channelwise Squeeze-and-Excite multiplication in float32: out[c,y,x] = in[c,y,x] * scale[c]. */
void cimp_scale_channels_f32(
    const float *in,
    const float *scale,
    int32_t c,
    int32_t hw,
    float *out
);

/** Channelwise Squeeze-and-Excite multiplication in Q15. */
void cimp_scale_channels_q15(
    const q15_t *in_q,
    const q15_t *scale_q,
    int32_t c,
    int32_t hw,
    int32_t out_shift,
    q15_t *out_q
);

#endif /* CIMP_INFRA_PYTORCH_H */
