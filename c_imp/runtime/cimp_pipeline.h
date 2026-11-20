/**
 * @file cimp_pipeline.h
 * @brief Unified C99 runtime dispatcher for descriptor x94 classifiers and exported CNNs.
 */
#ifndef CIMP_PIPELINE_H
#define CIMP_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#include "cimp_config.h"
#include "cimp_descriptor.h"
#include "cimp_fixed_point.h"
#include "cimp_models_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Return 1 if `model_name` is an exported descriptor classifier (`lr_dot` or `mlp_16_dot`).
 */
int cimp_is_tabular_model_supported(const char *model_name);

/**
 * Return 1 if `model_name` is an exported CNN architecture in this build.
 */
int cimp_is_cnn_model_supported(const char *model_name);

/** Return fractional Q-bits used for fixed-point logits of a tabular classifier, or -1 if invalid. */
int cimp_get_tabular_out_frac(const char *model_name);

/** Return fractional Q-bits used for fixed-point logits of an exported CNN, or -1 if invalid. */
int cimp_get_cnn_out_frac(const char *model_name);

#if CIMP_ENABLE_F32
/** Run exported tabular classifier in float32 over extracted `x94[CIMP_N_X94]`. Returns label or <0. */
int cimp_predict_tabular_f32(
    const char *model_name,
    const float *x94,
    float *out_logit
);
#endif

/** Run exported tabular classifier in fixed-point Q15 over extracted `x94_q[CIMP_N_X94]`. Returns label or <0. */
int cimp_predict_tabular_q15(
    const char *model_name,
    const q15_t *x94_q,
    int32_t *out_logit_q
);

#if CIMP_ENABLE_F32
/** Run exported CNN forward pass in float32 over raw uint8 image. Returns CIMP_OK or <0. */
int cimp_predict_cnn_f32(
    const char *model_name,
    const uint8_t *img_u8,
    uint8_t *arena,
    size_t arena_bytes,
    float out_logits[2],
    int *out_label
);
#endif

/** Run exported CNN forward pass in fixed-point Q15 over raw uint8 image. Returns CIMP_OK or <0. */
int cimp_predict_cnn_q15(
    const char *model_name,
    const uint8_t *img_u8,
    uint8_t *arena,
    size_t arena_bytes,
    int32_t out_logits_q[2],
    int *out_label
);

#ifdef __cplusplus
}
#endif

#endif /* CIMP_PIPELINE_H */
