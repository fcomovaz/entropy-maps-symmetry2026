/**
 * @file cimp_tensor.h
 * @brief Non-owning [C, H, W] tensor views and status codes for CIMP runtime kernels.
 */
#ifndef CIMP_TENSOR_H
#define CIMP_TENSOR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Row-major CHW flat index helper for tensor [C, H, W]. */
#define CIMP_IDX(c, y, x, h, w) ((((int32_t)(c) * (int32_t)(h)) + (int32_t)(y)) * (int32_t)(w) + (int32_t)(x))

/** Runtime status codes returned by all public cimp_* entry points (0 = OK, < 0 = error). */
enum {
    CIMP_OK = 0,
    CIMP_ERR_INVALID_ARG = -1,
    CIMP_ERR_SHAPE_MISMATCH = -2,
    CIMP_ERR_ARENA_TOO_SMALL = -3,
    CIMP_ERR_IO = -4,
    CIMP_ERR_UNSUPPORTED = -5
};

/** Non-owning view over a signed Qm.frac_bits [C, H, W] int16 tensor. */
typedef struct {
    int16_t *data;
    int32_t c;
    int32_t h;
    int32_t w;
    int8_t frac_bits;
} cimp_tensor_q15;

/** Non-owning view over a float32 [C, H, W] tensor. */
typedef struct {
    float *data;
    int32_t c;
    int32_t h;
    int32_t w;
} cimp_tensor_f32;

#ifdef __cplusplus
}
#endif

#endif /* CIMP_TENSOR_H */
