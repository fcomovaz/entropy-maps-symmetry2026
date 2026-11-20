/**
 * @file infra_sklearn.c
 * @brief C99 implementation of linear dot-product and 1-hidden-layer ReLU MLP inference kernels.
 */
#include "infra_sklearn.h"

#include "cimp_tensor.h"

#if CIMP_ENABLE_F32
float cimp_dense_dot_f32(const float *x, const float *w, float b, size_t n) {
    double acc = (double)b;
    for (size_t i = 0; i < n; ++i) {
        acc += (double)x[i] * (double)CIMP_ROM_READ_F32(&w[i]);
    }
    return (float)acc;
}

int cimp_dense_classify_f32(
    const float *x,
    const float *w,
    float b,
    float threshold,
    size_t n,
    float *out_logit
) {
    float logit = cimp_dense_dot_f32(x, w, b, n);
    if (out_logit != NULL) {
        *out_logit = logit;
    }
    return (logit > threshold) ? 1 : 0;
}
#endif

int32_t cimp_dense_dot_q15(
    const q15_t *x_q,
    const q15_t *w_q,
    int64_t b_acc,
    size_t n,
    int out_shift
) {
    int64_t acc = b_acc;
    for (size_t i = 0; i < n; ++i) {
        acc += (int64_t)x_q[i] * (int64_t)CIMP_ROM_READ_I16(&w_q[i]);
    }
    return cimp_sat32(cimp_round_shift_right_i64(acc, (int32_t)out_shift));
}

int32_t cimp_dense_dot_q15_w32(
    const q15_t *x_q,
    const int32_t *w_q,
    int64_t b_acc,
    size_t n,
    int out_shift
) {
    int64_t acc = b_acc;
    for (size_t i = 0; i < n; ++i) {
        acc += (int64_t)x_q[i] * (int64_t)CIMP_ROM_READ_I32(&w_q[i]);
    }
    return cimp_sat32(cimp_round_shift_right_i64(acc, (int32_t)out_shift));
}

int cimp_dense_classify_q15(
    const q15_t *x_q,
    const void *w_q,
    int w_bits,
    int64_t b_acc,
    int32_t threshold_q,
    size_t n,
    int out_shift,
    int32_t *out_logit_q
) {
    int32_t logit_q = 0;
    if (w_bits == 32) {
        logit_q = cimp_dense_dot_q15_w32(x_q, (const int32_t *)w_q, b_acc, n, out_shift);
    } else {
        logit_q = cimp_dense_dot_q15(x_q, (const q15_t *)w_q, b_acc, n, out_shift);
    }
    if (out_logit_q != NULL) {
        *out_logit_q = logit_q;
    }
    return (logit_q > threshold_q) ? 1 : 0;
}

#if CIMP_ENABLE_F32
float cimp_mlp_1hidden_f32(
    const float *x,
    const float *w1,
    const float *b1,
    const float *w2,
    float b2,
    size_t d_in,
    size_t d_hidden,
    float *hidden_buf
) {
    for (size_t h = 0; h < d_hidden; ++h) {
        const float *w_row = w1 + h * d_in;
        double acc = (double)CIMP_ROM_READ_F32(&b1[h]);
        for (size_t i = 0; i < d_in; ++i) {
            acc += (double)x[i] * (double)CIMP_ROM_READ_F32(&w_row[i]);
        }
        float z = (float)acc;
        hidden_buf[h] = (z > 0.0f) ? z : 0.0f;
    }
    double out_acc = (double)b2;
    for (size_t h = 0; h < d_hidden; ++h) {
        out_acc += (double)hidden_buf[h] * (double)CIMP_ROM_READ_F32(&w2[h]);
    }
    return (float)out_acc;
}

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
) {
    float logit = cimp_mlp_1hidden_f32(x, w1, b1, w2, b2, d_in, d_hidden, hidden_buf);
    if (out_logit != NULL) {
        *out_logit = logit;
    }
    return (logit > threshold) ? 1 : 0;
}
#endif

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
) {
    if (w1_bits == 32) {
        const int32_t *w1_32 = (const int32_t *)w1_q;
        for (size_t h = 0; h < d_hidden; ++h) {
            const int32_t *w_row = w1_32 + h * d_in;
            int64_t acc = CIMP_ROM_READ_I64(&b1_acc[h]);
            for (size_t i = 0; i < d_in; ++i) {
                acc += (int64_t)x_q[i] * (int64_t)CIMP_ROM_READ_I32(&w_row[i]);
            }
            int64_t shifted = cimp_round_shift_right_i64(acc, (int32_t)shift1);
            hidden_q[h] = (shifted <= 0LL) ? (q15_t)0 : cimp_sat16(shifted);
        }
    } else {
        const q15_t *w1_16 = (const q15_t *)w1_q;
        for (size_t h = 0; h < d_hidden; ++h) {
            const q15_t *w_row = w1_16 + h * d_in;
            int64_t acc = CIMP_ROM_READ_I64(&b1_acc[h]);
            for (size_t i = 0; i < d_in; ++i) {
                acc += (int64_t)x_q[i] * (int64_t)CIMP_ROM_READ_I16(&w_row[i]);
            }
            int64_t shifted = cimp_round_shift_right_i64(acc, (int32_t)shift1);
            hidden_q[h] = (shifted <= 0LL) ? (q15_t)0 : cimp_sat16(shifted);
        }
    }

    int64_t out_acc = b2_acc;
    if (w2_bits == 32) {
        const int32_t *w2_32 = (const int32_t *)w2_q;
        for (size_t h = 0; h < d_hidden; ++h) {
            out_acc += (int64_t)hidden_q[h] * (int64_t)CIMP_ROM_READ_I32(&w2_32[h]);
        }
    } else {
        const q15_t *w2_16 = (const q15_t *)w2_q;
        for (size_t h = 0; h < d_hidden; ++h) {
            out_acc += (int64_t)hidden_q[h] * (int64_t)CIMP_ROM_READ_I16(&w2_16[h]);
        }
    }
    return cimp_sat32(cimp_round_shift_right_i64(out_acc, (int32_t)shift2));
}

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
) {
    int32_t logit_q = cimp_mlp_1hidden_q15(
        x_q,
        w1_q,
        w1_bits,
        b1_acc,
        shift1,
        w2_q,
        w2_bits,
        b2_acc,
        shift2,
        d_in,
        d_hidden,
        hidden_q
    );
    if (out_logit_q != NULL) {
        *out_logit_q = logit_q;
    }
    return (logit_q > threshold_q) ? 1 : 0;
}
