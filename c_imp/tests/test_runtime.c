/**
 * @file test_runtime.c
 * @brief Standalone C99 unit tests for CIMP fixed-point, linear/MLP, and CNN kernels.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cimp_fixed_point.h"
#include "cimp_tensor.h"
#include "infra_pytorch.h"
#include "infra_sklearn.h"

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg)); \
            return 1;                                                        \
        }                                                                    \
    } while (0)

static int test_fixed_point_ops(void) {
    CHECK(cimp_sat16(40000LL) == 32767, "cimp_sat16 upper clamp");
    CHECK(cimp_sat16(-40000LL) == -32768, "cimp_sat16 lower clamp");
    CHECK(cimp_sat8(128LL) == 127, "cimp_sat8 upper clamp");
    CHECK(cimp_sat8(-129LL) == -128, "cimp_sat8 lower clamp");

    CHECK(cimp_q15_mul_round(-32768, -32768) == 32767, "-1.0 * -1.0 must saturate to 32767");
    CHECK(cimp_q15_mul_round(16384, 16384) == 8192, "0.5 * 0.5 == 0.25");
    CHECK(cimp_q15_mul_round(-16384, 16384) == -8192, "-0.5 * 0.5 == -0.25");
    CHECK(cimp_q15_mul_round(-1, 16384) == 0, "half-up tie rounds -0.5 -> 0");
    CHECK(cimp_q15_mul_round(-1, 16385) == -1, "just below -0.5 rounds to -1");

    CHECK(cimp_round_shift_right_i64(-1LL, 1) == 0LL, "rshift(-1, 1) == 0");
    CHECK(cimp_round_shift_right_i64(-2LL, 1) == -1LL, "rshift(-2, 1) == -1");
    CHECK(cimp_round_shift_right_i64(5LL, -2) == 20LL, "rshift(5, -2) == 20");

    CHECK(cimp_clz32(0U) == 32, "clz32(0)");
    CHECK(cimp_clz32(1U) == 31, "clz32(1)");
    CHECK(cimp_clz32(256U) == 23, "clz32(256)");
    CHECK(cimp_clz32(0x80000000U) == 0, "clz32(MSB)");
    return 0;
}

static int test_sklearn_kernels(void) {
    const float x[4] = {0.5f, 0.25f, 0.75f, 0.5f};
    const float w[4] = {1.0f, -2.0f, 0.5f, 0.5f};
    float logit = 0.0f;
    int lbl = cimp_dense_classify_f32(x, w, -0.25f, 0.0f, 4, &logit);
    CHECK(fabsf(logit - 0.375f) < 1e-6f, "dense dot f32 value");
    CHECK(lbl == 1, "dense dot f32 label");

    /* Q15: x_q in Q15, w_q in Q12, out in Q12 -> out_shift = 15 */
    const q15_t x_q[4] = {16384, 8192, 24576, 16384};
    const q15_t w_q[4] = {4096, -8192, 2048, 2048};
    int64_t b_acc = (int64_t)(-0.25 * (double)(1LL << 27));
    int32_t logit_q = 0;
    int lbl_q = cimp_dense_classify_q15(x_q, w_q, 16, b_acc, 0, 4, 15, &logit_q);
    CHECK(logit_q == (int32_t)(0.375 * 4096.0), "dense dot q15 exact");
    CHECK(lbl_q == 1, "dense dot q15 label");

    /* 1-hidden MLP check */
    const float w1[4] = {1.0f, -1.0f, 0.0f, 1.0f};
    const float b1[2] = {0.0f, -0.5f};
    const float w2[2] = {2.0f, -1.0f};
    float hbuf[2] = {0.0f, 0.0f};
    const float x2[2] = {0.5f, 0.25f};
    /* h = [0.25, max(0, -0.25)=0]; out = 0.5 - 0.1 = 0.4 */
    int ml = cimp_mlp_1hidden_classify_f32(x2, w1, b1, w2, -0.1f, 0.0f, 2, 2, hbuf, &logit);
    CHECK(fabsf(logit - 0.4f) < 1e-6f && ml == 1, "mlp f32 exact");
    return 0;
}

static int test_pytorch_kernels(void) {
    /* 1x2x2 Conv2d 1x1x2x2 */
    const float in_f32[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float w_f32[4] = {1.0f, 0.0f, 0.0f, -1.0f};
    const float b_f32[1] = {0.5f};
    float out_f32[1] = {0.0f};
    cimp_conv2d_f32(in_f32, 1, 2, 2, w_f32, b_f32, 1, 2, 2, 1, 0, 1, out_f32, 1, 1);
    /* 1*1 - 4 + 0.5 = -2.5 -> relu -> 0 */
    CHECK(fabsf(out_f32[0] - 0.0f) < 1e-6f, "conv2d f32 fused relu");

    /* Hardswish and Hardsigmoid exact integer Q12 check on x=0 -> hw=0, hs=0.5 */
    const q15_t x_q[3] = {-16384, 0, 16384}; /* [-4.0, 0.0, +4.0] in Q12 */
    q15_t hs_q[3] = {0};
    q15_t hsig_q[3] = {0};
    cimp_hardswish_q15(x_q, hs_q, 3, 12, 12);
    cimp_hardsigmoid_q15(x_q, hsig_q, 3, 12, 14);
    CHECK(hs_q[0] == 0 && hs_q[1] == 0 && hs_q[2] == 16384, "hardswish q15 piecewise");
    CHECK(hsig_q[0] == 0 && hsig_q[1] == 8192 && hsig_q[2] == 16384, "hardsigmoid q15 piecewise");

    /* Residual add with shift alignment + relu */
    const q15_t a_q[2] = {4096, -8192}; /* [1.0, -2.0] in Q12 */
    const q15_t b_q[2] = {1024, 1024};  /* [1.0, 1.0] in Q10 */
    q15_t sum_q[2] = {0};
    cimp_add_q15(a_q, b_q, sum_q, 2, 1, -1, 1); /* target Q11: a>>1 + b<<1 -> [4096, -2048 -> 0] */
    CHECK(sum_q[0] == 4096 && sum_q[1] == 0, "add q15 alignment + relu");

    /* Scale channels */
    const q15_t feat_q[4] = {4096, 8192, -4096, 2048};
    const q15_t sc_q[2] = {8192, 16384}; /* [0.5, 1.0] in Q14 */
    q15_t scaled_q[4] = {0};
    cimp_scale_channels_q15(feat_q, sc_q, 2, 2, 14, scaled_q);
    CHECK(scaled_q[0] == 2048 && scaled_q[1] == 4096 && scaled_q[2] == -4096 && scaled_q[3] == 2048,
          "scale_channels q15");
    return 0;
}

int main(void) {
    if (test_fixed_point_ops() != 0) {
        return 1;
    }
    if (test_sklearn_kernels() != 0) {
        return 1;
    }
    if (test_pytorch_kernels() != 0) {
        return 1;
    }
    printf("cimp test_runtime: ALL PASSED\n");
    return 0;
}
