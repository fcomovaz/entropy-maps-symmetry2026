/**
 * @file infra_sklearn.h
 * @brief Inference kernels for classical tabular linear and 1-hidden-layer MLP models.
 *
 * Supports float32 (`_f32`) and pure integer Qm.n fixed-point (`_q15`, with q15 or q31
 * folded weights and int64 accumulators) with zero dynamic memory allocation and no <math.h>
 * calls in fixed-point paths.
 */
#ifndef CIMP_INFRA_SKLEARN_H
#define CIMP_INFRA_SKLEARN_H

#include <stddef.h>
#include <stdint.h>

#include "cimp_fixed_point.h"

/** Compute dense dot product logit = b + sum_i x[i] * w[i] in float32 (double accumulator). */
float cimp_dense_dot_f32(const float *x, const float *w, float b, size_t n);

/** Classify binary logit in float32: returns 1 if (b + x . w) > threshold else 0. */
int cimp_dense_classify_f32(
    const float *x,
    const float *w,
    float b,
    float threshold,
    size_t n,
    float *out_logit
);

/**
 * Compute fixed-point dense dot product with 16-bit Q-weights and 64-bit bias accumulator.
 * Returns saturated int32 logit at target fractional precision (`acc = b_acc + sum x_q[i]*w_q[i]`,
 * then round-half-up right shift by `out_shift`).
 */
int32_t cimp_dense_dot_q15(
    const q15_t *x_q,
    const q15_t *w_q,
    int64_t b_acc,
    size_t n,
    int out_shift
);

/** Compute fixed-point dense dot product with 32-bit Q-weights and 64-bit bias accumulator. */
int32_t cimp_dense_dot_q15_w32(
    const q15_t *x_q,
    const int32_t *w_q,
    int64_t b_acc,
    size_t n,
    int out_shift
);

/** Classify fixed-point linear model (`w_bits` is 16 or 32), writing `out_logit_q` if non-NULL. */
int cimp_dense_classify_q15(
    const q15_t *x_q,
    const void *w_q,
    int w_bits,
    int64_t b_acc,
    int32_t threshold_q,
    size_t n,
    int out_shift,
    int32_t *out_logit_q
);

/** Compute 1-hidden-layer ReLU MLP logit in float32 using caller-provided `hidden_buf[d_hidden]`. */
float cimp_mlp_1hidden_f32(
    const float *x,
    const float *w1,
    const float *b1,
    const float *w2,
    float b2,
    size_t d_in,
    size_t d_hidden,
    float *hidden_buf
);

/** Classify 1-hidden-layer ReLU MLP in float32: returns 1 if logit > threshold else 0. */
int cimp_mlp_1hidden_classify_f32(
    const float *x,
    const float *w1,
    const float *b1,
    const float *w2,
    float b2,
    float threshold,
    size_t d_in,
    size_t d_hidden,
    float *hidden_buf,
    float *out_logit
);

/**
 * Compute 1-hidden-layer ReLU MLP logit in integer fixed-point Qm.n.
 * Supports 16-bit (`int16_t`) or 32-bit (`int32_t`) weights per layer via `w1_bits`, `w2_bits`.
 * Caller provides scratch buffer `hidden_q[d_hidden]`. Pure integer arithmetic; no <math.h>.
 */
int32_t cimp_mlp_1hidden_q15(
    const q15_t *x_q,
    const void *w1_q,
    int w1_bits,
    const int64_t *b1_acc,
    int shift1,
    const void *w2_q,
    int w2_bits,
    int64_t b2_acc,
    int shift2,
    size_t d_in,
    size_t d_hidden,
    q15_t *hidden_q
);

/** Classify fixed-point 1-hidden-layer ReLU MLP: returns 1 if logit_q > threshold_q else 0. */
int cimp_mlp_1hidden_classify_q15(
    const q15_t *x_q,
    const void *w1_q,
    int w1_bits,
    const int64_t *b1_acc,
    int shift1,
    const void *w2_q,
    int w2_bits,
    int64_t b2_acc,
    int shift2,
    int32_t threshold_q,
    size_t d_in,
    size_t d_hidden,
    q15_t *hidden_q,
    int32_t *out_logit_q
);

#endif /* CIMP_INFRA_SKLEARN_H */
