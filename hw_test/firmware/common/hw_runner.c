/**
 * @file hw_runner.c
 * @brief MCU binary protocol handler, streaming/buffered image receiver, and timed inference runner.
 */
#include "hw_build_config.h"

#include "hw_runner.h"
#include "hw_board.h"
#include "hw_crc.h"
#include "hw_protocol.h"

#include <string.h>

#include "cimp_fixed_point.h"
#include "cimp_pipeline.h"
#include "cimp_tensor.h"

#if (HW_MODE_Q15 == 0) && !CIMP_ENABLE_F32
#error "HW_MODE_Q15 == 0 requires CIMP_ENABLE_F32 == 1"
#endif

#if HW_STREAMING && (HW_MODEL_KIND != 0)
#error "Streaming mode (HW_STREAMING) supports tabular models (HW_MODEL_KIND == 0) only"
#endif

#if HW_STREAMING && (HW_MODE_Q15 == 0)
#error "Streaming mode (HW_STREAMING) supports q15 mode (HW_MODE_Q15 == 1) only"
#endif

#define HW_IMAGE_BYTES ((uint32_t)HW_IMAGE_WIDTH * (uint32_t)HW_IMAGE_HEIGHT)

#if HW_STREAMING
#define HW_TX_MAX_BUF_BYTES ((size_t)HW_RESULT_PAYLOAD_BYTES + (size_t)HW_PROTO_OVERHEAD_BYTES)
#define HW_FRAME_BUF_BYTES \
    (((size_t)HW_MAX_PAYLOAD_BYTES > HW_TX_MAX_BUF_BYTES) ? (size_t)HW_MAX_PAYLOAD_BYTES : HW_TX_MAX_BUF_BYTES)
#else
#define HW_X94_BUF_REQ ((size_t)HW_X94_F32_PAYLOAD_BYTES + (size_t)HW_PROTO_OVERHEAD_BYTES)
#define HW_RX_BUF_REQ  ((size_t)HW_MAX_PAYLOAD_BYTES + (size_t)HW_PROTO_OVERHEAD_BYTES)
#define HW_FRAME_BUF_BYTES ((HW_RX_BUF_REQ > HW_X94_BUF_REQ) ? HW_RX_BUF_REQ : HW_X94_BUF_REQ)
#endif

typedef enum {
    HW_RX_IDLE = 0,
    HW_RX_RECEIVING,
    HW_RX_DONE,
    HW_RX_ERROR
} hw_rx_state_t;

typedef struct {
    uint32_t img_id;
    uint32_t img_total;
    uint32_t expected_crc32;
    uint32_t expected_offset;
    uint32_t crc32_state;
    uint16_t repeats;
    uint16_t warmup;
    uint32_t avr_sel_us_acc;
    hw_rx_state_t state;
    uint8_t x94_valid;
} hw_rx_context_t;

static hw_runner_config_t g_cfg;
static hw_proto_parser_t g_parser;
static uint8_t g_frame_buf[HW_FRAME_BUF_BYTES];
static hw_rx_context_t g_rx;

#if HW_STREAMING
static CIMP_DescStream g_stream;
static q15_t g_x94_q15[CIMP_N_X94];
#else
static uint8_t g_img_buf[(size_t)HW_IMAGE_WIDTH * (size_t)HW_IMAGE_HEIGHT];
static uint32_t g_sel_us[HW_MAX_TIMED_ITERS];
static uint32_t g_model_us[HW_MAX_TIMED_ITERS];
#if HW_MODEL_KIND == 0
static CIMP_DescriptorWorkspace g_desc_ws;
#if HW_MODE_Q15
static q15_t g_x94_q15[CIMP_N_X94];
#else
static float g_x94_f32[CIMP_N_X94];
#endif
#endif /* HW_MODEL_KIND == 0 */
#endif /* HW_STREAMING */

static void hw_tx_flush(void) {
    if (g_cfg.flush != NULL) {
        g_cfg.flush(g_cfg.ctx);
    }
}

static void hw_send_sealed_frame(uint8_t type, uint16_t seq, uint16_t payload_len) {
    if (g_cfg.write == NULL) {
        return;
    }
    uint16_t total = hw_proto_seal_in_place(
        g_frame_buf,
        sizeof(g_frame_buf),
        (uint8_t)HW_PROTOCOL_VERSION,
        type,
        seq,
        payload_len
    );
    if (total > 0U) {
        g_cfg.write(g_frame_buf, (size_t)total, g_cfg.ctx);
    }
}

static void hw_send_error(uint16_t seq, uint16_t code, const char *text) {
    uint16_t plen = hw_proto_encode_error(
        &g_frame_buf[HW_PROTO_HEADER_BYTES],
        (uint16_t)(sizeof(g_frame_buf) - HW_PROTO_OVERHEAD_BYTES),
        code,
        text
    );
    hw_send_sealed_frame(HW_MSG_ERROR, seq, plen);
    hw_tx_flush();
}

static void hw_send_chunk_ack(uint16_t seq, uint32_t offset, int8_t status) {
    hw_proto_encode_chunk_ack(&g_frame_buf[HW_PROTO_HEADER_BYTES], offset, status);
    hw_send_sealed_frame(HW_MSG_CHUNK_ACK, seq, HW_CHUNK_ACK_PAYLOAD_BYTES);
    hw_tx_flush();
}

static void hw_send_ack_rx(uint16_t seq, uint32_t image_id, uint8_t crc_ok) {
    hw_proto_encode_ack_rx(&g_frame_buf[HW_PROTO_HEADER_BYTES], image_id, crc_ok);
    hw_send_sealed_frame(HW_MSG_ACK_RX, seq, HW_ACK_RX_PAYLOAD_BYTES);
    hw_tx_flush();
}

static void hw_send_info(uint16_t seq) {
    hw_info_payload_t info;
    info.board_id = HW_STR_ROM(HW_BOARD_ID);
    info.model = HW_STR_ROM(HW_MODEL_NAME);
    info.mode = (HW_MODE_Q15 != 0) ? HW_WIRE_MODE_Q15 : HW_WIRE_MODE_F32;
    info.protocol = (uint8_t)HW_PROTOCOL_VERSION;
    info.width = (uint16_t)HW_IMAGE_WIDTH;
    info.height = (uint16_t)HW_IMAGE_HEIGHT;
    info.streaming = (HW_STREAMING != 0) ? 1U : 0U;
    info.max_payload = (uint16_t)HW_MAX_PAYLOAD_BYTES;
    info.free_ram_boot = hw_board_free_ram_boot();
    info.psram_size = hw_board_psram_size();
    info.git_hash = HW_STR_ROM(HW_GIT_HASH);

    hw_proto_encode_info(&g_frame_buf[HW_PROTO_HEADER_BYTES], &info);
    hw_send_sealed_frame(HW_MSG_INFO, seq, HW_INFO_PAYLOAD_BYTES);
    hw_tx_flush();
}

static void hw_reset_transfer_state(void) {
    g_rx.img_id = 0UL;
    g_rx.img_total = 0UL;
    g_rx.expected_crc32 = 0UL;
    g_rx.expected_offset = 0UL;
    g_rx.crc32_state = hw_crc32_init();
    g_rx.repeats = 0U;
    g_rx.warmup = 0U;
    g_rx.avr_sel_us_acc = 0UL;
    g_rx.state = HW_RX_IDLE;
    g_rx.x94_valid = 0U;
}

#if HW_MODEL_KIND == 0
#if HW_MODE_Q15
static uint32_t hw_compute_x94_q15_crc32(const q15_t *x94) {
    uint32_t state = hw_crc32_init();
    uint8_t le[2];
    for (size_t i = 0U; i < (size_t)HW_X94_COUNT; ++i) {
        hw_write_i16_le(le, x94[i]);
        state = hw_crc32_update(state, le, sizeof(le));
    }
    return hw_crc32_final(state);
}
#else
static uint32_t hw_compute_x94_f32_crc32(const float *x94) {
    uint32_t state = hw_crc32_init();
    uint8_t le[4];
    for (size_t i = 0U; i < (size_t)HW_X94_COUNT; ++i) {
        hw_write_f32_le(le, x94[i]);
        state = hw_crc32_update(state, le, sizeof(le));
    }
    return hw_crc32_final(state);
}
#endif
#endif /* HW_MODEL_KIND == 0 */

#if !HW_STREAMING
static void hw_sort_u32_asc(uint32_t *arr, uint16_t n) {
    for (uint16_t i = 1U; i < n; ++i) {
        uint32_t key = arr[i];
        uint16_t j = i;
        while (j > 0U && arr[j - 1U] > key) {
            arr[j] = arr[j - 1U];
            --j;
        }
        arr[j] = key;
    }
}

static void hw_compute_stats_esp(uint16_t n_iters, uint32_t stats[5][3]) {
    stats[0][0] = g_sel_us[0];
    stats[0][1] = g_model_us[0];
    stats[0][2] = g_sel_us[0] + g_model_us[0];

    if (n_iters <= 1U) {
        for (size_t r = 1U; r < 5U; ++r) {
            stats[r][0] = stats[0][0];
            stats[r][1] = stats[0][1];
            stats[r][2] = stats[0][2];
        }
        return;
    }

    const uint16_t count = (uint16_t)(n_iters - 1U);
    uint32_t tmp[HW_MAX_TIMED_ITERS];

    for (size_t col = 0U; col < 3U; ++col) {
        uint64_t sum = 0ULL;
        for (uint16_t k = 0U; k < count; ++k) {
            uint16_t idx = (uint16_t)(k + 1U);
            uint32_t v;
            if (col == 0U) {
                v = g_sel_us[idx];
            } else if (col == 1U) {
                v = g_model_us[idx];
            } else {
                v = g_sel_us[idx] + g_model_us[idx];
            }
            tmp[k] = v;
            sum += (uint64_t)v;
        }
        hw_sort_u32_asc(tmp, count);

        stats[1][col] = tmp[0];
        stats[2][col] = ((count & 1U) != 0U)
            ? tmp[count / 2U]
            : (uint32_t)(((uint64_t)tmp[(count / 2U) - 1U] + (uint64_t)tmp[count / 2U]) / 2ULL);
        stats[3][col] = (uint32_t)((sum + (uint64_t)(count >> 1)) / (uint64_t)count);
        stats[4][col] = tmp[count - 1U];
    }
}
#endif /* !HW_STREAMING */

static void hw_run_inference_and_send_result(uint16_t seq) {
#if HW_STREAMING
    /* AVR streaming Q15 tabular path: minimal stack, zero stats arrays. */
    uint32_t t0 = hw_board_micros();
    int rc = cimp_desc_stream_finish_q15(&g_stream, g_x94_q15);
    uint32_t t1 = hw_board_micros();
    g_rx.avr_sel_us_acc += (uint32_t)(t1 - t0);

    int32_t logit_q = 0;
    uint32_t model_us = 0UL;
    int16_t status = (int16_t)rc;
    int8_t label = -1;
    uint32_t x94_crc = 0UL;

    if (rc == CIMP_OK) {
        x94_crc = hw_compute_x94_q15_crc32(g_x94_q15);
        g_rx.x94_valid = 1U;

        uint32_t m0 = hw_board_micros();
        int pred = cimp_predict_tabular_q15(HW_MODEL_NAME, g_x94_q15, &logit_q);
        uint32_t m1 = hw_board_micros();
        model_us = (uint32_t)(m1 - m0);

        if (pred < 0) {
            status = (int16_t)pred;
            label = -1;
        } else {
            status = 0;
            label = (int8_t)pred;
        }
    }

    int frac = cimp_get_tabular_out_frac(HW_MODEL_NAME);
    uint32_t total_us = g_rx.avr_sel_us_acc + model_us;

    hw_result_payload_t res;
    res.image_id = g_rx.img_id;
    res.status = status;
    res.label = label;
    res.n_logits = 1U;
    res.logit_raw[0] = logit_q;
    res.logit_raw[1] = 0;
    res.frac_bits = (frac >= 0) ? (uint8_t)frac : 0U;
    res.x94_crc32 = x94_crc;
    res.timer_resolution_us = (uint8_t)HW_TIMER_RES_US;
    res.n_iters = 1U;
    for (size_t r = 0U; r < 5U; ++r) {
        res.stats[r][0] = g_rx.avr_sel_us_acc;
        res.stats[r][1] = model_us;
        res.stats[r][2] = total_us;
    }
    res.free_heap_min = hw_board_free_heap_min();
    res.stack_hwm = hw_board_stack_hwm();
    res.free_psram_min = hw_board_free_psram_min();
    res.arena_bytes = 0UL;
    res.arena_location = HW_ARENA_LOC_NONE;

    hw_proto_encode_result(&g_frame_buf[HW_PROTO_HEADER_BYTES], &res);
    hw_send_sealed_frame(HW_MSG_RESULT, seq, HW_RESULT_PAYLOAD_BYTES);
    hw_tx_flush();

#else
    /* ESP8266 / ESP32 buffered path: warmup + timed repeats. */
    uint16_t n_timed = g_rx.repeats;
    if (n_timed == 0U) {
        n_timed = 1U;
    } else if (n_timed > HW_MAX_TIMED_ITERS) {
        n_timed = HW_MAX_TIMED_ITERS;
    }

    int16_t status = 0;
    int8_t label = -1;
    int32_t logit_raw[2] = {0, 0};
    uint8_t n_logits = (HW_MODEL_KIND == 0) ? 1U : 2U;
    uint8_t frac_bits = 0U;
    uint32_t x94_crc = 0UL;
    uint16_t completed_iters = 0U;

#if HW_MODEL_KIND == 1
    if (g_cfg.arena == NULL || g_cfg.arena_bytes < (size_t)HW_ARENA_BYTES) {
        status = (int16_t)CIMP_ERR_ARENA_TOO_SMALL;
        g_sel_us[0] = 0UL;
        g_model_us[0] = 0UL;
        completed_iters = 1U;
    } else
#endif
    {
        for (uint16_t w = 0U; w < g_rx.warmup; ++w) {
            hw_board_yield();
#if HW_MODEL_KIND == 0
#if HW_MODE_Q15
            int rc = cimp_descriptor_x94_q15(g_img_buf, &g_desc_ws, g_x94_q15);
            if (rc == CIMP_OK) {
                int32_t tmp_q = 0;
                int pred = cimp_predict_tabular_q15(HW_MODEL_NAME, g_x94_q15, &tmp_q);
                if (pred < 0) {
                    status = (int16_t)pred;
                    break;
                }
            } else {
                status = (int16_t)rc;
                break;
            }
#else
            int rc = cimp_descriptor_x94_f32(g_img_buf, &g_desc_ws, g_x94_f32);
            if (rc == CIMP_OK) {
                float tmp_f = 0.0f;
                int pred = cimp_predict_tabular_f32(HW_MODEL_NAME, g_x94_f32, &tmp_f);
                if (pred < 0) {
                    status = (int16_t)pred;
                    break;
                }
            } else {
                status = (int16_t)rc;
                break;
            }
#endif
#else /* HW_MODEL_KIND == 1 (CNN) */
            int pred_label = -1;
#if HW_MODE_Q15
            int32_t tmp_q[2] = {0, 0};
            int rc = cimp_predict_cnn_q15(
                HW_MODEL_NAME, g_img_buf, g_cfg.arena, g_cfg.arena_bytes, tmp_q, &pred_label
            );
#else
            float tmp_f[2] = {0.0f, 0.0f};
            int rc = cimp_predict_cnn_f32(
                HW_MODEL_NAME, g_img_buf, g_cfg.arena, g_cfg.arena_bytes, tmp_f, &pred_label
            );
#endif
            if (rc != CIMP_OK) {
                status = (int16_t)rc;
                break;
            }
#endif
            hw_board_yield();
        }

        if (status == 0) {
            for (uint16_t k = 0U; k < n_timed; ++k) {
                if (k > 0U) {
                    hw_board_yield();
                }
#if HW_MODEL_KIND == 0
                uint32_t t0 = hw_board_micros();
#if HW_MODE_Q15
                int rc = cimp_descriptor_x94_q15(g_img_buf, &g_desc_ws, g_x94_q15);
#else
                int rc = cimp_descriptor_x94_f32(g_img_buf, &g_desc_ws, g_x94_f32);
#endif
                uint32_t t1 = hw_board_micros();
                g_sel_us[k] = (uint32_t)(t1 - t0);

                if (rc != CIMP_OK) {
                    g_model_us[k] = 0UL;
                    status = (int16_t)rc;
                    label = -1;
                    completed_iters = (uint16_t)(k + 1U);
                    break;
                }

#if HW_MODE_Q15
                int32_t out_q = 0;
                uint32_t m0 = hw_board_micros();
                int pred = cimp_predict_tabular_q15(HW_MODEL_NAME, g_x94_q15, &out_q);
                uint32_t m1 = hw_board_micros();
                g_model_us[k] = (uint32_t)(m1 - m0);
                if (pred < 0) {
                    status = (int16_t)pred;
                    label = -1;
                    completed_iters = (uint16_t)(k + 1U);
                    break;
                }
                logit_raw[0] = out_q;
                logit_raw[1] = 0;
                label = (int8_t)pred;
#else
                float out_f = 0.0f;
                uint32_t m0 = hw_board_micros();
                int pred = cimp_predict_tabular_f32(HW_MODEL_NAME, g_x94_f32, &out_f);
                uint32_t m1 = hw_board_micros();
                g_model_us[k] = (uint32_t)(m1 - m0);
                if (pred < 0) {
                    status = (int16_t)pred;
                    label = -1;
                    completed_iters = (uint16_t)(k + 1U);
                    break;
                }
                memcpy(&logit_raw[0], &out_f, sizeof(out_f));
                logit_raw[1] = 0;
                label = (int8_t)pred;
#endif
#else /* HW_MODEL_KIND == 1 (CNN) */
                g_sel_us[k] = 0UL;
                int pred_label = -1;
#if HW_MODE_Q15
                int32_t out_q[2] = {0, 0};
                uint32_t m0 = hw_board_micros();
                int rc = cimp_predict_cnn_q15(
                    HW_MODEL_NAME, g_img_buf, g_cfg.arena, g_cfg.arena_bytes, out_q, &pred_label
                );
                uint32_t m1 = hw_board_micros();
                g_model_us[k] = (uint32_t)(m1 - m0);
                if (rc != CIMP_OK) {
                    status = (int16_t)rc;
                    label = -1;
                    completed_iters = (uint16_t)(k + 1U);
                    break;
                }
                logit_raw[0] = out_q[0];
                logit_raw[1] = out_q[1];
                label = (int8_t)pred_label;
#else
                float out_f[2] = {0.0f, 0.0f};
                uint32_t m0 = hw_board_micros();
                int rc = cimp_predict_cnn_f32(
                    HW_MODEL_NAME, g_img_buf, g_cfg.arena, g_cfg.arena_bytes, out_f, &pred_label
                );
                uint32_t m1 = hw_board_micros();
                g_model_us[k] = (uint32_t)(m1 - m0);
                if (rc != CIMP_OK) {
                    status = (int16_t)rc;
                    label = -1;
                    completed_iters = (uint16_t)(k + 1U);
                    break;
                }
                memcpy(&logit_raw[0], &out_f[0], sizeof(out_f[0]));
                memcpy(&logit_raw[1], &out_f[1], sizeof(out_f[1]));
                label = (int8_t)pred_label;
#endif
#endif
                completed_iters = (uint16_t)(k + 1U);
                hw_board_yield();
            }
        } else {
            g_sel_us[0] = 0UL;
            g_model_us[0] = 0UL;
            completed_iters = 1U;
        }
    }

#if HW_MODEL_KIND == 0
    if (status == 0) {
#if HW_MODE_Q15
        x94_crc = hw_compute_x94_q15_crc32(g_x94_q15);
        int frac = cimp_get_tabular_out_frac(HW_MODEL_NAME);
        frac_bits = (frac >= 0) ? (uint8_t)frac : 0U;
#else
        x94_crc = hw_compute_x94_f32_crc32(g_x94_f32);
        frac_bits = 0U;
#endif
        g_rx.x94_valid = 1U;
    }
#else
#if HW_MODE_Q15
    if (status == 0) {
        int frac = cimp_get_cnn_out_frac(HW_MODEL_NAME);
        frac_bits = (frac >= 0) ? (uint8_t)frac : 0U;
    }
#else
    frac_bits = 0U;
#endif
#endif

    hw_result_payload_t res;
    res.image_id = g_rx.img_id;
    res.status = status;
    res.label = label;
    res.n_logits = n_logits;
    res.logit_raw[0] = logit_raw[0];
    res.logit_raw[1] = logit_raw[1];
    res.frac_bits = frac_bits;
    res.x94_crc32 = x94_crc;
    res.timer_resolution_us = (uint8_t)HW_TIMER_RES_US;
    res.n_iters = (completed_iters > 0U) ? completed_iters : 1U;
    hw_compute_stats_esp(res.n_iters, res.stats);
    res.free_heap_min = hw_board_free_heap_min();
    res.stack_hwm = hw_board_stack_hwm();
    res.free_psram_min = hw_board_free_psram_min();
    res.arena_bytes = (uint32_t)g_cfg.arena_bytes;
    res.arena_location = g_cfg.arena_location;

    hw_proto_encode_result(&g_frame_buf[HW_PROTO_HEADER_BYTES], &res);
    hw_send_sealed_frame(HW_MSG_RESULT, seq, HW_RESULT_PAYLOAD_BYTES);
    hw_tx_flush();
#endif /* HW_STREAMING */
}

static void hw_handle_img_begin(const hw_proto_frame_t *frame) {
    hw_img_begin_payload_t req;
    if (hw_proto_decode_img_begin(frame->payload, frame->length, &req) != 0) {
        hw_send_error(frame->seq, HW_ERR_BAD_LENGTH, HW_STR_ROM("bad img_begin len"));
        return;
    }
    if (req.total != HW_IMAGE_BYTES) {
        hw_reset_transfer_state();
        hw_send_error(frame->seq, HW_ERR_BAD_SIZE, HW_STR_ROM("bad image total"));
        return;
    }

    hw_reset_transfer_state();
    g_rx.img_id = req.image_id;
    g_rx.img_total = req.total;
    g_rx.expected_crc32 = req.crc32;
    g_rx.repeats = req.repeats;
    g_rx.warmup = req.warmup;

#if HW_STREAMING
    if (cimp_desc_stream_init(&g_stream) != CIMP_OK) {
        g_rx.state = HW_RX_ERROR;
        hw_send_error(frame->seq, HW_ERR_STREAM_FAIL, HW_STR_ROM("stream init failed"));
        return;
    }
#endif

    g_rx.state = HW_RX_RECEIVING;
    hw_send_chunk_ack(frame->seq, HW_UNAVAILABLE_U32, HW_CHUNK_OK);
}

static void hw_handle_img_chunk(const hw_proto_frame_t *frame) {
    if (frame->length < 4U) {
        hw_send_error(frame->seq, HW_ERR_BAD_LENGTH, HW_STR_ROM("chunk len < 4"));
        return;
    }
    const uint32_t offset = hw_read_u32_le(&frame->payload[0]);
    const uint16_t data_len = (uint16_t)(frame->length - 4U);
    const uint8_t *data = &frame->payload[4];

    if (data_len == 0U) {
        hw_send_error(frame->seq, HW_ERR_BAD_LENGTH, HW_STR_ROM("empty chunk"));
        return;
    }
    if (g_rx.state == HW_RX_IDLE || g_rx.state == HW_RX_ERROR) {
        hw_send_error(frame->seq, HW_ERR_BAD_STATE, HW_STR_ROM("no active transfer"));
        return;
    }

    /* Duplicate check: never touch CRC32 or stream/image buffer on retries. */
    if (offset < g_rx.expected_offset) {
        const uint32_t covered = g_rx.expected_offset - offset;
        if ((uint32_t)data_len > covered) {
            hw_send_chunk_ack(frame->seq, offset, HW_CHUNK_BAD_OFFSET);
            return;
        }
        hw_send_chunk_ack(frame->seq, offset, HW_CHUNK_DUP);
        return;
    }

    if (g_rx.state != HW_RX_RECEIVING) {
        hw_send_error(frame->seq, HW_ERR_BAD_STATE, HW_STR_ROM("transfer already done"));
        return;
    }
    if (offset > g_rx.expected_offset) {
        hw_send_chunk_ack(frame->seq, offset, HW_CHUNK_BAD_OFFSET);
        return;
    }
    if ((uint32_t)data_len > (g_rx.img_total - g_rx.expected_offset)) {
        hw_send_chunk_ack(frame->seq, offset, HW_CHUNK_OVERFLOW);
        return;
    }

#if HW_STREAMING
    if ((data_len % (uint16_t)HW_IMAGE_WIDTH) != 0U) {
        hw_send_chunk_ack(frame->seq, offset, HW_CHUNK_OVERFLOW);
        return;
    }
    g_rx.crc32_state = hw_crc32_update(g_rx.crc32_state, data, (size_t)data_len);
    const int32_t n_rows = (int32_t)(data_len / (uint16_t)HW_IMAGE_WIDTH);
    uint32_t t0 = hw_board_micros();
    int rc = cimp_desc_stream_push_rows(&g_stream, data, n_rows);
    uint32_t t1 = hw_board_micros();
    g_rx.avr_sel_us_acc += (uint32_t)(t1 - t0);
    if (rc != CIMP_OK) {
        g_rx.state = HW_RX_ERROR;
        hw_send_chunk_ack(frame->seq, offset, HW_CHUNK_OVERFLOW);
        return;
    }
#else
    g_rx.crc32_state = hw_crc32_update(g_rx.crc32_state, data, (size_t)data_len);
    memcpy(&g_img_buf[offset], data, (size_t)data_len);
#endif

    g_rx.expected_offset += (uint32_t)data_len;
    hw_send_chunk_ack(frame->seq, offset, HW_CHUNK_OK);

    if (g_rx.expected_offset == g_rx.img_total) {
        const uint32_t actual_crc = hw_crc32_final(g_rx.crc32_state);
        const uint8_t crc_ok = (actual_crc == g_rx.expected_crc32) ? 1U : 0U;
        hw_send_ack_rx(frame->seq, g_rx.img_id, crc_ok);
        if (!crc_ok) {
            g_rx.state = HW_RX_ERROR;
            return;
        }
        g_rx.state = HW_RX_DONE;
        hw_run_inference_and_send_result(frame->seq);
    }
}

static void hw_handle_get_x94(const hw_proto_frame_t *frame) {
    if (frame->length != 0U) {
        hw_send_error(frame->seq, HW_ERR_BAD_LENGTH, HW_STR_ROM("get_x94 not empty"));
        return;
    }
#if HW_MODEL_KIND == 0
    if (!g_rx.x94_valid) {
        hw_send_error(frame->seq, HW_ERR_NO_X94, HW_STR_ROM("x94 not ready"));
        return;
    }
#if HW_STREAMING
    /* Stream 188-byte RESULT_X94 in small chunks to avoid enlarging AVR g_frame_buf. */
    if (g_cfg.write == NULL) {
        return;
    }
    uint8_t hdr[HW_PROTO_HEADER_BYTES];
    hw_write_u16_le(&hdr[0], HW_PROTO_MAGIC);
    hdr[2] = (uint8_t)HW_PROTOCOL_VERSION;
    hdr[3] = HW_MSG_RESULT_X94;
    hw_write_u16_le(&hdr[4], frame->seq);
    hw_write_u16_le(&hdr[6], HW_X94_Q15_PAYLOAD_BYTES);

    uint16_t crc = hw_crc16_ccitt_block(hw_crc16_ccitt_init(), &hdr[2], 6U);
    g_cfg.write(hdr, sizeof(hdr), g_cfg.ctx);

    size_t idx = 0U;
    while (idx < (size_t)HW_X94_COUNT) {
        size_t batch = (size_t)HW_X94_COUNT - idx;
        if (batch > 8U) {
            batch = 8U;
        }
        for (size_t k = 0U; k < batch; ++k) {
            hw_write_i16_le(&g_frame_buf[k * 2U], g_x94_q15[idx + k]);
        }
        size_t nbytes = batch * 2U;
        crc = hw_crc16_ccitt_block(crc, g_frame_buf, nbytes);
        g_cfg.write(g_frame_buf, nbytes, g_cfg.ctx);
        idx += batch;
    }

    uint8_t crc_buf[HW_PROTO_CRC_BYTES];
    hw_write_u16_le(crc_buf, crc);
    g_cfg.write(crc_buf, sizeof(crc_buf), g_cfg.ctx);
    hw_tx_flush();
#else /* !HW_STREAMING */
    uint8_t *dst = &g_frame_buf[HW_PROTO_HEADER_BYTES];
#if HW_MODE_Q15
    for (size_t i = 0U; i < (size_t)HW_X94_COUNT; ++i) {
        hw_write_i16_le(&dst[i * 2U], g_x94_q15[i]);
    }
    hw_send_sealed_frame(HW_MSG_RESULT_X94, frame->seq, HW_X94_Q15_PAYLOAD_BYTES);
#else
    for (size_t i = 0U; i < (size_t)HW_X94_COUNT; ++i) {
        hw_write_f32_le(&dst[i * 4U], g_x94_f32[i]);
    }
    hw_send_sealed_frame(HW_MSG_RESULT_X94, frame->seq, HW_X94_F32_PAYLOAD_BYTES);
#endif
    hw_tx_flush();
#endif /* HW_STREAMING */
#else /* HW_MODEL_KIND == 1 */
    hw_send_error(frame->seq, HW_ERR_NO_X94, HW_STR_ROM("cnn has no x94"));
#endif
}

static void hw_dispatch_frame(const hw_proto_frame_t *frame) {
    switch (frame->type) {
    case HW_MSG_HELLO:
        if (frame->length != 0U) {
            hw_send_error(frame->seq, HW_ERR_BAD_LENGTH, HW_STR_ROM("hello not empty"));
            return;
        }
        hw_reset_transfer_state();
        hw_send_info(frame->seq);
        return;

    case HW_MSG_IMG_BEGIN:
        hw_handle_img_begin(frame);
        return;

    case HW_MSG_IMG_CHUNK:
        hw_handle_img_chunk(frame);
        return;

    case HW_MSG_GET_X94:
        hw_handle_get_x94(frame);
        return;

    default:
        hw_send_error(frame->seq, HW_ERR_BAD_CMD, HW_STR_ROM("unknown msg type"));
        return;
    }
}

void hw_runner_init(const hw_runner_config_t *cfg) {
    if (cfg != NULL) {
        g_cfg = *cfg;
    } else {
        memset(&g_cfg, 0, sizeof(g_cfg));
    }
    hw_reset_transfer_state();
    hw_proto_parser_init(
        &g_parser,
        (uint8_t)HW_PROTOCOL_VERSION,
        g_frame_buf,
        (uint16_t)HW_MAX_PAYLOAD_BYTES
    );
}

void hw_runner_feed_byte(uint8_t byte) {
    hw_proto_frame_t frame;
    if (hw_proto_parser_feed(&g_parser, byte, &frame) != 0) {
        hw_dispatch_frame(&frame);
    }
}

void hw_runner_feed_bytes(const uint8_t *data, size_t len) {
    if (data == NULL) {
        return;
    }
    for (size_t i = 0U; i < len; ++i) {
        hw_runner_feed_byte(data[i]);
    }
}

void hw_runner_on_disconnect(void) {
    hw_reset_transfer_state();
    hw_proto_parser_reset(&g_parser);
}
