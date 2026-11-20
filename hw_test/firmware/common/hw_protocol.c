/**
 * @file hw_protocol.c
 * @brief Incremental wire protocol parser and little-endian payload codecs.
 */
#include "hw_protocol.h"

static uint8_t hw_read_rom_u8(const char *p) {
#if defined(__AVR__)
    return (uint8_t)pgm_read_byte((const void *)p);
#else
    return (uint8_t)(*p);
#endif
}

static void hw_copy_fixed_str(uint8_t *dst, size_t cap, const char *src) {
    memset(dst, 0, cap);
    if (src == NULL) {
        return;
    }
    for (size_t i = 0U; i < cap; ++i) {
        uint8_t ch = hw_read_rom_u8(&src[i]);
        if (ch == 0U) {
            break;
        }
        dst[i] = ch;
    }
}

void hw_proto_parser_init(
    hw_proto_parser_t *parser,
    uint8_t expected_version,
    uint8_t *payload_buf,
    uint16_t payload_cap
) {
    if (parser == NULL) {
        return;
    }
    parser->expected_version = expected_version;
    parser->payload_buf = payload_buf;
    parser->payload_cap = payload_cap;
    hw_proto_parser_reset(parser);
}

void hw_proto_parser_reset(hw_proto_parser_t *parser) {
    if (parser == NULL) {
        return;
    }
    parser->state = HW_PARSE_MAGIC_0;
    parser->hdr_pos = 0U;
    parser->payload_len = 0U;
    parser->payload_pos = 0U;
    parser->crc_calc = hw_crc16_ccitt_init();
    parser->crc_rx = 0U;
}

int hw_proto_parser_feed(
    hw_proto_parser_t *parser,
    uint8_t byte,
    hw_proto_frame_t *out_frame
) {
    if (parser == NULL || out_frame == NULL) {
        return 0;
    }

    const uint8_t magic_lo = (uint8_t)(HW_PROTO_MAGIC & 0xFFU);
    const uint8_t magic_hi = (uint8_t)((HW_PROTO_MAGIC >> 8) & 0xFFU);

    switch (parser->state) {
    case HW_PARSE_MAGIC_0:
        if (byte == magic_lo) {
            parser->state = HW_PARSE_MAGIC_1;
        }
        return 0;

    case HW_PARSE_MAGIC_1:
        if (byte == magic_hi) {
            parser->hdr_pos = 0U;
            parser->crc_calc = hw_crc16_ccitt_init();
            parser->state = HW_PARSE_HEADER;
        } else if (byte != magic_lo) {
            parser->state = HW_PARSE_MAGIC_0;
        }
        return 0;

    case HW_PARSE_HEADER:
        parser->hdr[parser->hdr_pos++] = byte;
        parser->crc_calc = hw_crc16_ccitt_update(parser->crc_calc, byte);
        if (parser->hdr_pos == 6U) {
            uint8_t version = parser->hdr[0];
            uint16_t len = hw_read_u16_le(&parser->hdr[4]);
            if (version != parser->expected_version || len > parser->payload_cap) {
                hw_proto_parser_reset(parser);
                return 0;
            }
            if (len > 0U && parser->payload_buf == NULL) {
                hw_proto_parser_reset(parser);
                return 0;
            }
            parser->payload_len = len;
            parser->payload_pos = 0U;
            parser->state = (len == 0U) ? HW_PARSE_CRC_0 : HW_PARSE_PAYLOAD;
        }
        return 0;

    case HW_PARSE_PAYLOAD:
        parser->payload_buf[parser->payload_pos++] = byte;
        parser->crc_calc = hw_crc16_ccitt_update(parser->crc_calc, byte);
        if (parser->payload_pos == parser->payload_len) {
            parser->state = HW_PARSE_CRC_0;
        }
        return 0;

    case HW_PARSE_CRC_0:
        parser->crc_rx = (uint16_t)byte;
        parser->state = HW_PARSE_CRC_1;
        return 0;

    case HW_PARSE_CRC_1:
        parser->crc_rx |= (uint16_t)((uint16_t)byte << 8);
        if (parser->crc_rx == parser->crc_calc) {
            out_frame->version = parser->hdr[0];
            out_frame->type = parser->hdr[1];
            out_frame->seq = hw_read_u16_le(&parser->hdr[2]);
            out_frame->length = parser->payload_len;
            out_frame->payload = (parser->payload_len > 0U) ? parser->payload_buf : NULL;
            hw_proto_parser_reset(parser);
            return 1;
        }
        hw_proto_parser_reset(parser);
        return 0;

    default:
        hw_proto_parser_reset(parser);
        return 0;
    }
}

int hw_proto_decode_img_begin(
    const uint8_t *payload,
    uint16_t len,
    hw_img_begin_payload_t *out
) {
    if (payload == NULL || out == NULL || len != HW_IMG_BEGIN_PAYLOAD_BYTES) {
        return -1;
    }
    out->image_id = hw_read_u32_le(&payload[0]);
    out->total = hw_read_u32_le(&payload[4]);
    out->crc32 = hw_read_u32_le(&payload[8]);
    out->repeats = hw_read_u16_le(&payload[12]);
    out->warmup = hw_read_u16_le(&payload[14]);
    return 0;
}

void hw_proto_encode_info(uint8_t out[HW_INFO_PAYLOAD_BYTES], const hw_info_payload_t *info) {
    if (out == NULL || info == NULL) {
        return;
    }
    hw_copy_fixed_str(&out[0], 16U, info->board_id);
    hw_copy_fixed_str(&out[16], 24U, info->model);
    out[40] = info->mode;
    out[41] = info->protocol;
    hw_write_u16_le(&out[42], info->width);
    hw_write_u16_le(&out[44], info->height);
    out[46] = info->streaming;
    hw_write_u16_le(&out[47], info->max_payload);
    hw_write_u32_le(&out[49], info->free_ram_boot);
    hw_write_u32_le(&out[53], info->psram_size);
    hw_copy_fixed_str(&out[57], 8U, info->git_hash);
}

void hw_proto_encode_chunk_ack(uint8_t out[HW_CHUNK_ACK_PAYLOAD_BYTES], uint32_t offset, int8_t status) {
    if (out == NULL) {
        return;
    }
    hw_write_u32_le(&out[0], offset);
    out[4] = (uint8_t)status;
}

void hw_proto_encode_ack_rx(uint8_t out[HW_ACK_RX_PAYLOAD_BYTES], uint32_t image_id, uint8_t crc_ok) {
    if (out == NULL) {
        return;
    }
    hw_write_u32_le(&out[0], image_id);
    out[4] = crc_ok;
}

void hw_proto_encode_result(uint8_t out[HW_RESULT_PAYLOAD_BYTES], const hw_result_payload_t *res) {
    if (out == NULL || res == NULL) {
        return;
    }
    hw_write_u32_le(&out[0], res->image_id);
    hw_write_i16_le(&out[4], res->status);
    out[6] = (uint8_t)res->label;
    out[7] = res->n_logits;
    hw_write_i32_le(&out[8], res->logit_raw[0]);
    hw_write_i32_le(&out[12], res->logit_raw[1]);
    out[16] = res->frac_bits;
    hw_write_u32_le(&out[17], res->x94_crc32);
    out[21] = res->timer_resolution_us;
    hw_write_u16_le(&out[22], res->n_iters);

    size_t pos = 24U;
    for (size_t r = 0U; r < 5U; ++r) {
        for (size_t c = 0U; c < 3U; ++c) {
            hw_write_u32_le(&out[pos], res->stats[r][c]);
            pos += 4U;
        }
    }
    hw_write_u32_le(&out[84], res->free_heap_min);
    hw_write_u32_le(&out[88], res->stack_hwm);
    hw_write_u32_le(&out[92], res->free_psram_min);
    hw_write_u32_le(&out[96], res->arena_bytes);
    out[100] = res->arena_location;
}

uint16_t hw_proto_encode_error(uint8_t *out, uint16_t cap, uint16_t code, const char *text) {
    if (out == NULL || cap < 3U) {
        return 0U;
    }
    hw_write_u16_le(&out[0], code);
    size_t max_text = (size_t)(cap - 3U);
    if (max_text > HW_ERROR_MAX_TEXT_BYTES) {
        max_text = HW_ERROR_MAX_TEXT_BYTES;
    }
    size_t n = 0U;
    if (text != NULL) {
        while (n < max_text) {
            uint8_t ch = hw_read_rom_u8(&text[n]);
            if (ch == 0U) {
                break;
            }
            out[3U + n] = ch;
            ++n;
        }
    }
    out[2] = (uint8_t)n;
    return (uint16_t)(3U + n);
}

uint16_t hw_proto_seal_in_place(
    uint8_t *frame_buf,
    size_t frame_cap,
    uint8_t version,
    uint8_t type,
    uint16_t seq,
    uint16_t payload_len
) {
    const size_t total = (size_t)HW_PROTO_OVERHEAD_BYTES + (size_t)payload_len;
    if (frame_buf == NULL || frame_cap < total) {
        return 0U;
    }
    hw_write_u16_le(&frame_buf[0], HW_PROTO_MAGIC);
    frame_buf[2] = version;
    frame_buf[3] = type;
    hw_write_u16_le(&frame_buf[4], seq);
    hw_write_u16_le(&frame_buf[6], payload_len);

    uint16_t crc = hw_crc16_ccitt_block(
        hw_crc16_ccitt_init(),
        &frame_buf[2],
        (size_t)6U + (size_t)payload_len
    );
    hw_write_u16_le(&frame_buf[HW_PROTO_HEADER_BYTES + payload_len], crc);
    return (uint16_t)total;
}

void hw_proto_stream_send(
    hw_proto_write_fn write_fn,
    void *ctx,
    uint8_t version,
    uint8_t type,
    uint16_t seq,
    const uint8_t *payload,
    uint16_t payload_len
) {
    if (write_fn == NULL) {
        return;
    }
    uint8_t hdr[HW_PROTO_HEADER_BYTES];
    hw_write_u16_le(&hdr[0], HW_PROTO_MAGIC);
    hdr[2] = version;
    hdr[3] = type;
    hw_write_u16_le(&hdr[4], seq);
    hw_write_u16_le(&hdr[6], payload_len);

    uint16_t crc = hw_crc16_ccitt_block(hw_crc16_ccitt_init(), &hdr[2], 6U);
    if (payload_len > 0U && payload != NULL) {
        crc = hw_crc16_ccitt_block(crc, payload, (size_t)payload_len);
    }

    uint8_t crc_buf[HW_PROTO_CRC_BYTES];
    hw_write_u16_le(crc_buf, crc);

    write_fn(hdr, sizeof(hdr), ctx);
    if (payload_len > 0U && payload != NULL) {
        write_fn(payload, (size_t)payload_len, ctx);
    }
    write_fn(crc_buf, sizeof(crc_buf), ctx);
}
