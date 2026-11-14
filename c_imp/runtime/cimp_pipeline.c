/**
 * @file cimp_pipeline.c
 * @brief Unified dispatcher routing inference to compiled tabular classifiers or CNNs.
 */
#include "cimp_pipeline.h"

#include <string.h>

#include "cimp_tensor.h"
#include "infra_sklearn.h"

static int cimp_streq(const char *a, const char *b) {
    return (a != NULL && b != NULL && strcmp(a, b) == 0);
}

int cimp_is_tabular_model_supported(const char *model_name) {
    (void)model_name;
#if CIMP_HAS_MODEL_LR_DOT
    if (cimp_streq(model_name, "lr_dot")) {
        return 1;
    }
#endif
#if CIMP_HAS_MODEL_MLP_16_DOT
    if (cimp_streq(model_name, "mlp_16_dot")) {
        return 1;
    }
#endif
    return 0;
}

int cimp_is_cnn_model_supported(const char *model_name) {
#if CIMP_HAS_CNN_ULTRA_TINY
    if (cimp_streq(model_name, "ultra_tiny")) {
        return 1;
    }
#endif
#if CIMP_HAS_CNN_TINY_CNN
    if (cimp_streq(model_name, "tiny_cnn")) {
        return 1;
    }
#endif
#if CIMP_HAS_CNN_DS_CNN
    if (cimp_streq(model_name, "ds_cnn")) {
        return 1;
    }
#endif
#if CIMP_HAS_CNN_SMALL_CNN
    if (cimp_streq(model_name, "small_cnn")) {
        return 1;
    }
#endif
#if CIMP_HAS_CNN_MOBILENET_LIKE
    if (cimp_streq(model_name, "mobilenet_like")) {
        return 1;
    }
#endif
#if CIMP_HAS_CNN_TINY_RESNET
    if (cimp_streq(model_name, "tiny_resnet")) {
        return 1;
    }
#endif
#if CIMP_HAS_CNN_TORCHVISION_MOBILENET
    if (cimp_streq(model_name, "torchvision_mobilenet")) {
        return 1;
    }
#endif
    return 0;
}

int cimp_get_tabular_out_frac(const char *model_name) {
    (void)model_name;
#if CIMP_HAS_MODEL_LR_DOT
    if (cimp_streq(model_name, "lr_dot")) {
        return CIMP_LR_DOT_OUT_FRAC;
    }
#endif
#if CIMP_HAS_MODEL_MLP_16_DOT
    if (cimp_streq(model_name, "mlp_16_dot")) {
        return CIMP_MLP_16_DOT_OUT_FRAC;
    }
#endif
    return -1;
}

int cimp_get_cnn_out_frac(const char *model_name) {
    (void)model_name;
#if CIMP_HAS_CNN_ULTRA_TINY
    if (cimp_streq(model_name, "ultra_tiny")) {
        return CIMP_CNN_ULTRA_TINY_OUT_FRAC;
    }
#endif
#if CIMP_HAS_CNN_TINY_CNN
    if (cimp_streq(model_name, "tiny_cnn")) {
        return CIMP_CNN_TINY_CNN_OUT_FRAC;
    }
#endif
#if CIMP_HAS_CNN_DS_CNN
    if (cimp_streq(model_name, "ds_cnn")) {
        return CIMP_CNN_DS_CNN_OUT_FRAC;
    }
#endif
#if CIMP_HAS_CNN_SMALL_CNN
    if (cimp_streq(model_name, "small_cnn")) {
        return CIMP_CNN_SMALL_CNN_OUT_FRAC;
    }
#endif
#if CIMP_HAS_CNN_MOBILENET_LIKE
    if (cimp_streq(model_name, "mobilenet_like")) {
        return CIMP_CNN_MOBILENET_LIKE_OUT_FRAC;
    }
#endif
#if CIMP_HAS_CNN_TINY_RESNET
    if (cimp_streq(model_name, "tiny_resnet")) {
        return CIMP_CNN_TINY_RESNET_OUT_FRAC;
    }
#endif
#if CIMP_HAS_CNN_TORCHVISION_MOBILENET
    if (cimp_streq(model_name, "torchvision_mobilenet")) {
        return CIMP_CNN_TORCHVISION_MOBILENET_OUT_FRAC;
    }
#endif
    return -1;
}

int cimp_predict_tabular_f32(
    const char *model_name,
    const float *x94,
    float *out_logit
) {
    (void)model_name;
    (void)out_logit;
    if (x94 == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
#if CIMP_HAS_MODEL_LR_DOT
    if (cimp_streq(model_name, "lr_dot")) {
        return cimp_dense_classify_f32(
            x94,
            CIMP_LR_DOT_W_F32,
            CIMP_LR_DOT_B_EFF_F32,
            CIMP_LR_DOT_THRESHOLD_F32,
            (size_t)CIMP_LR_DOT_D_IN,
            out_logit
        );
    }
#endif
#if CIMP_HAS_MODEL_MLP_16_DOT
    if (cimp_streq(model_name, "mlp_16_dot")) {
        float hidden_buf[CIMP_MLP_16_DOT_D_HIDDEN];
        return cimp_mlp_1hidden_classify_f32(
            x94,
            CIMP_MLP_16_DOT_W1_F32,
            CIMP_MLP_16_DOT_B1_F32,
            CIMP_MLP_16_DOT_W2_F32,
            CIMP_MLP_16_DOT_B2_EFF_F32,
            CIMP_MLP_16_DOT_THRESHOLD_F32,
            (size_t)CIMP_MLP_16_DOT_D_IN,
            (size_t)CIMP_MLP_16_DOT_D_HIDDEN,
            hidden_buf,
            out_logit
        );
    }
#endif
    return CIMP_ERR_UNSUPPORTED;
}

int cimp_predict_tabular_q15(
    const char *model_name,
    const q15_t *x94_q,
    int32_t *out_logit_q
) {
    (void)model_name;
    (void)out_logit_q;
    if (x94_q == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
#if CIMP_HAS_MODEL_LR_DOT
    if (cimp_streq(model_name, "lr_dot")) {
        return cimp_dense_classify_q15(
            x94_q,
            CIMP_LR_DOT_W_Q,
            CIMP_LR_DOT_W_BITS,
            CIMP_LR_DOT_B_ACC,
            CIMP_LR_DOT_THRESHOLD_Q,
            (size_t)CIMP_LR_DOT_D_IN,
            CIMP_LR_DOT_OUT_SHIFT,
            out_logit_q
        );
    }
#endif
#if CIMP_HAS_MODEL_MLP_16_DOT
    if (cimp_streq(model_name, "mlp_16_dot")) {
        q15_t hidden_q[CIMP_MLP_16_DOT_D_HIDDEN];
        return cimp_mlp_1hidden_classify_q15(
            x94_q,
            CIMP_MLP_16_DOT_W1_Q,
            CIMP_MLP_16_DOT_W1_BITS,
            CIMP_MLP_16_DOT_B1_ACC,
            CIMP_MLP_16_DOT_SHIFT1,
            CIMP_MLP_16_DOT_W2_Q,
            CIMP_MLP_16_DOT_W2_BITS,
            CIMP_MLP_16_DOT_B2_ACC,
            CIMP_MLP_16_DOT_SHIFT2,
            CIMP_MLP_16_DOT_THRESHOLD_Q,
            (size_t)CIMP_MLP_16_DOT_D_IN,
            (size_t)CIMP_MLP_16_DOT_D_HIDDEN,
            hidden_q,
            out_logit_q
        );
    }
#endif
    return CIMP_ERR_UNSUPPORTED;
}

int cimp_predict_cnn_f32(
    const char *model_name,
    const uint8_t *img_u8,
    uint8_t *arena,
    size_t arena_bytes,
    float out_logits[2],
    int *out_label
) {
    if (img_u8 == NULL || arena == NULL || out_logits == NULL || out_label == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
#if CIMP_HAS_CNN_ULTRA_TINY
    if (cimp_streq(model_name, "ultra_tiny")) {
        int rc = cimp_ultra_tiny_forward_f32(img_u8, arena, arena_bytes, out_logits);
        if (rc == CIMP_OK) {
            *out_label = ((out_logits[1] - out_logits[0]) > CIMP_CNN_ULTRA_TINY_THRESHOLD_LOGIT_F32) ? 1 : 0;
        }
        return rc;
    }
#endif
#if CIMP_HAS_CNN_TINY_CNN
    if (cimp_streq(model_name, "tiny_cnn")) {
        int rc = cimp_tiny_cnn_forward_f32(img_u8, arena, arena_bytes, out_logits);
        if (rc == CIMP_OK) {
            *out_label = ((out_logits[1] - out_logits[0]) > CIMP_CNN_TINY_CNN_THRESHOLD_LOGIT_F32) ? 1 : 0;
        }
        return rc;
    }
#endif
#if CIMP_HAS_CNN_DS_CNN
    if (cimp_streq(model_name, "ds_cnn")) {
        int rc = cimp_ds_cnn_forward_f32(img_u8, arena, arena_bytes, out_logits);
        if (rc == CIMP_OK) {
            *out_label = ((out_logits[1] - out_logits[0]) > CIMP_CNN_DS_CNN_THRESHOLD_LOGIT_F32) ? 1 : 0;
        }
        return rc;
    }
#endif
#if CIMP_HAS_CNN_SMALL_CNN
    if (cimp_streq(model_name, "small_cnn")) {
        int rc = cimp_small_cnn_forward_f32(img_u8, arena, arena_bytes, out_logits);
        if (rc == CIMP_OK) {
            *out_label = ((out_logits[1] - out_logits[0]) > CIMP_CNN_SMALL_CNN_THRESHOLD_LOGIT_F32) ? 1 : 0;
        }
        return rc;
    }
#endif
#if CIMP_HAS_CNN_MOBILENET_LIKE
    if (cimp_streq(model_name, "mobilenet_like")) {
        int rc = cimp_mobilenet_like_forward_f32(img_u8, arena, arena_bytes, out_logits);
        if (rc == CIMP_OK) {
            *out_label = ((out_logits[1] - out_logits[0]) > CIMP_CNN_MOBILENET_LIKE_THRESHOLD_LOGIT_F32) ? 1 : 0;
        }
        return rc;
    }
#endif
#if CIMP_HAS_CNN_TINY_RESNET
    if (cimp_streq(model_name, "tiny_resnet")) {
        int rc = cimp_tiny_resnet_forward_f32(img_u8, arena, arena_bytes, out_logits);
        if (rc == CIMP_OK) {
            *out_label = ((out_logits[1] - out_logits[0]) > CIMP_CNN_TINY_RESNET_THRESHOLD_LOGIT_F32) ? 1 : 0;
        }
        return rc;
    }
#endif
#if CIMP_HAS_CNN_TORCHVISION_MOBILENET
    if (cimp_streq(model_name, "torchvision_mobilenet")) {
        int rc = cimp_torchvision_mobilenet_forward_f32(img_u8, arena, arena_bytes, out_logits);
        if (rc == CIMP_OK) {
            *out_label = ((out_logits[1] - out_logits[0]) > CIMP_CNN_TORCHVISION_MOBILENET_THRESHOLD_LOGIT_F32) ? 1 : 0;
        }
        return rc;
    }
#endif
    return CIMP_ERR_UNSUPPORTED;
}

int cimp_predict_cnn_q15(
    const char *model_name,
    const uint8_t *img_u8,
    uint8_t *arena,
    size_t arena_bytes,
    int32_t out_logits_q[2],
    int *out_label
) {
    if (img_u8 == NULL || arena == NULL || out_logits_q == NULL || out_label == NULL) {
        return CIMP_ERR_INVALID_ARG;
    }
#if CIMP_HAS_CNN_ULTRA_TINY
    if (cimp_streq(model_name, "ultra_tiny")) {
        int rc = cimp_ultra_tiny_forward_q15(img_u8, arena, arena_bytes, out_logits_q);
        if (rc == CIMP_OK) {
            int64_t margin = (int64_t)out_logits_q[1] - (int64_t)out_logits_q[0];
            *out_label = (margin > (int64_t)CIMP_CNN_ULTRA_TINY_THRESHOLD_LOGIT_Q) ? 1 : 0;
        }
        return rc;
    }
#endif
#if CIMP_HAS_CNN_TINY_CNN
    if (cimp_streq(model_name, "tiny_cnn")) {
        int rc = cimp_tiny_cnn_forward_q15(img_u8, arena, arena_bytes, out_logits_q);
        if (rc == CIMP_OK) {
            int64_t margin = (int64_t)out_logits_q[1] - (int64_t)out_logits_q[0];
            *out_label = (margin > (int64_t)CIMP_CNN_TINY_CNN_THRESHOLD_LOGIT_Q) ? 1 : 0;
        }
        return rc;
    }
#endif
#if CIMP_HAS_CNN_DS_CNN
    if (cimp_streq(model_name, "ds_cnn")) {
        int rc = cimp_ds_cnn_forward_q15(img_u8, arena, arena_bytes, out_logits_q);
        if (rc == CIMP_OK) {
            int64_t margin = (int64_t)out_logits_q[1] - (int64_t)out_logits_q[0];
            *out_label = (margin > (int64_t)CIMP_CNN_DS_CNN_THRESHOLD_LOGIT_Q) ? 1 : 0;
        }
        return rc;
    }
#endif
#if CIMP_HAS_CNN_SMALL_CNN
    if (cimp_streq(model_name, "small_cnn")) {
        int rc = cimp_small_cnn_forward_q15(img_u8, arena, arena_bytes, out_logits_q);
        if (rc == CIMP_OK) {
            int64_t margin = (int64_t)out_logits_q[1] - (int64_t)out_logits_q[0];
            *out_label = (margin > (int64_t)CIMP_CNN_SMALL_CNN_THRESHOLD_LOGIT_Q) ? 1 : 0;
        }
        return rc;
    }
#endif
#if CIMP_HAS_CNN_MOBILENET_LIKE
    if (cimp_streq(model_name, "mobilenet_like")) {
        int rc = cimp_mobilenet_like_forward_q15(img_u8, arena, arena_bytes, out_logits_q);
        if (rc == CIMP_OK) {
            int64_t margin = (int64_t)out_logits_q[1] - (int64_t)out_logits_q[0];
            *out_label = (margin > (int64_t)CIMP_CNN_MOBILENET_LIKE_THRESHOLD_LOGIT_Q) ? 1 : 0;
        }
        return rc;
    }
#endif
#if CIMP_HAS_CNN_TINY_RESNET
    if (cimp_streq(model_name, "tiny_resnet")) {
        int rc = cimp_tiny_resnet_forward_q15(img_u8, arena, arena_bytes, out_logits_q);
        if (rc == CIMP_OK) {
            int64_t margin = (int64_t)out_logits_q[1] - (int64_t)out_logits_q[0];
            *out_label = (margin > (int64_t)CIMP_CNN_TINY_RESNET_THRESHOLD_LOGIT_Q) ? 1 : 0;
        }
        return rc;
    }
#endif
#if CIMP_HAS_CNN_TORCHVISION_MOBILENET
    if (cimp_streq(model_name, "torchvision_mobilenet")) {
        int rc = cimp_torchvision_mobilenet_forward_q15(img_u8, arena, arena_bytes, out_logits_q);
        if (rc == CIMP_OK) {
            int64_t margin = (int64_t)out_logits_q[1] - (int64_t)out_logits_q[0];
            *out_label = (margin > (int64_t)CIMP_CNN_TORCHVISION_MOBILENET_THRESHOLD_LOGIT_Q) ? 1 : 0;
        }
        return rc;
    }
#endif
    return CIMP_ERR_UNSUPPORTED;
}
