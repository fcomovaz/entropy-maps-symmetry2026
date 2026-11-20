/**
 * @file hw_protocol.h
 * @brief Frozen binary wire protocol framing, parser, and little-endian payload codecs.
 *
 * Frame layout (little-endian):
 *   magic u16 (0x5345) | version u8 (1) | type u8 | seq u16 | length u16 | payload[length] | crc16 u16
 * CRC16-CCITT-FALSE covers bytes [version .. payload_end].
 */
#ifndef HW_PROTOCOL_H
#define HW_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVR__)
#include <avr/pgmspace.h>
#define HW_STR_ROM(s) PSTR(s)
#else
#define HW_STR_ROM(s) (s)
#endif

#include "hw_crc.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HW_PROTO_MAGIC          0x5345U
#define HW_PROTO_VERSION_V1     1U
#define HW_PROTO_HEADER_BYTES   8U
#define HW_PROTO_CRC_BYTES      2U
#define HW_PROTO_OVERHEAD_BYTES (HW_PROTO_HEADER_BYTES + HW_PROTO_CRC_BYTES)

/* Wire message types. */
#define HW_MSG_HELLO      0x01U
#define HW_MSG_INFO       0x02U
#define HW_MSG_IMG_BEGIN  0x10U
#define HW_MSG_IMG_CHUNK  0x11U
#define HW_MSG_CHUNK_ACK  0x12U
#define HW_MSG_ACK_RX     0x13U
#define HW_MSG_RESULT     0x20U
#define HW_MSG_RESULT_X94 0x21U
#define HW_MSG_GET_X94    0x22U
#define HW_MSG_ERROR      0x7FU

/* Fixed payload byte lengths. */
#define HW_INFO_PAYLOAD_BYTES      65U
#define HW_IMG_BEGIN_PAYLOAD_BYTES 16U
#define HW_CHUNK_ACK_PAYLOAD_BYTES 5U
#define HW_ACK_RX_PAYLOAD_BYTES    5U
#define HW_RESULT_PAYLOAD_BYTES    101U
#define HW_X94_COUNT               94U
#define HW_X94_Q15_PAYLOAD_BYTES   (HW_X94_COUNT * 2U)
#define HW_X94_F32_PAYLOAD_BYTES   (HW_X94_COUNT * 4U)
#define HW_ERROR_MAX_TEXT_BYTES    48U

#define HW_UNAVAILABLE_U32 0xFFFFFFFFUL
#define HW_MAX_TIMED_ITERS 64U

/* CHUNK_ACK status values (int8_t). */
#define HW_CHUNK_OK         ((int8_t)0)
#define HW_CHUNK_DUP        ((int8_t)1)
#define HW_CHUNK_BAD_OFFSET ((int8_t)-1)
#define HW_CHUNK_OVERFLOW   ((int8_t)-2)

/* Numeric mode values on wire (INFO.mode). */
#define HW_WIRE_MODE_Q15 0U
#define HW_WIRE_MODE_F32 1U

/* Arena location values on wire (RESULT.arena_location). */
#define HW_ARENA_LOC_NONE  0U
#define HW_ARENA_LOC_SRAM  1U
#define HW_ARENA_LOC_PSRAM 2U

/* Wire ERROR codes (uint16_t). */
#define HW_ERR_BAD_LENGTH  1U
#define HW_ERR_BAD_STATE   2U
#define HW_ERR_BAD_SIZE    3U
#define HW_ERR_BAD_CMD     4U
#define HW_ERR_NO_X94      5U
#define HW_ERR_STREAM_FAIL 6U
#define HW_ERR_ARENA_FAIL  7U
#define HW_ERR_UNSUPPORTED 8U

/* Explicit unaligned-safe little-endian helpers. */
static inline uint16_t hw_read_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | (uint16_t)((uint16_t)p[1] << 8);
}

static inline uint32_t hw_read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static inline void hw_write_u16_le(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFFU);
    p[1] = (uint8_t)((v >> 8) & 0xFFU);
}

static inline void hw_write_i16_le(uint8_t *p, int16_t v) {
    hw_write_u16_le(p, (uint16_t)v);
}

static inline void hw_write_u32_le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFFUL);
    p[1] = (uint8_t)((v >> 8) & 0xFFUL);
    p[2] = (uint8_t)((v >> 16) & 0xFFUL);
    p[3] = (uint8_t)((v >> 24) & 0xFFUL);
}

static inline void hw_write_i32_le(uint8_t *p, int32_t v) {
    hw_write_u32_le(p, (uint32_t)v);
}

static inline void hw_write_f32_le(uint8_t *p, float v) {
    uint32_t bits = 0UL;
    memcpy(&bits, &v, sizeof(bits));
    hw_write_u32_le(p, bits);
}

typedef struct {
    uint8_t version;
    uint8_t type;
    uint16_t seq;
    uint16_t length;
    const uint8_t *payload;
} hw_proto_frame_t;

typedef enum {
    HW_PARSE_MAGIC_0 = 0,
    HW_PARSE_MAGIC_1,
    HW_PARSE_HEADER,
    HW_PARSE_PAYLOAD,
    HW_PARSE_CRC_0,
    HW_PARSE_CRC_1
} hw_parse_state_t;

typedef struct {
    hw_parse_state_t state;
    uint8_t expected_version;
    uint8_t hdr[6]; /* version, type, seq(2), length(2) */
    uint8_t hdr_pos;
    uint16_t payload_len;
    uint16_t payload_pos;
    uint16_t crc_calc;
    uint16_t crc_rx;
    uint8_t *payload_buf;
    uint16_t payload_cap;
} hw_proto_parser_t;

typedef struct {
    const char *board_id;
    const char *model;
    uint8_t mode;
    uint8_t protocol;
    uint16_t width;
    uint16_t height;
    uint8_t streaming;
    uint16_t max_payload;
    uint32_t free_ram_boot;
    uint32_t psram_size;
    const char *git_hash;
} hw_info_payload_t;

typedef struct {
    uint32_t image_id;
    uint32_t total;
    uint32_t crc32;
    uint16_t repeats;
    uint16_t warmup;
} hw_img_begin_payload_t;

typedef struct {
    uint32_t image_id;
    int16_t status;
    int8_t label;
    uint8_t n_logits;
    int32_t logit_raw[2];
    uint8_t frac_bits;
    uint32_t x94_crc32;
    uint8_t timer_resolution_us;
    uint16_t n_iters;
    uint32_t stats[5][3]; /* [first, min, median, mean, max] x [selector, model, total] */
    uint32_t free_heap_min;
    uint32_t stack_hwm;
    uint32_t free_psram_min;
    uint32_t arena_bytes;
    uint8_t arena_location;
} hw_result_payload_t;

typedef void (*hw_proto_write_fn)(const uint8_t *data, size_t len, void *ctx);

/** Initialize incremental frame parser with caller-owned payload buffer. */
void hw_proto_parser_init(
    hw_proto_parser_t *parser,
    uint8_t expected_version,
    uint8_t *payload_buf,
    uint16_t payload_cap
);

/** Reset frame parser state to wait for next MAGIC sync. */
void hw_proto_parser_reset(hw_proto_parser_t *parser);

/**
 * Feed one received byte into the parser.
 * Returns 1 if a complete CRC-valid frame was decoded into `out_frame`, else 0.
 */
int hw_proto_parser_feed(
    hw_proto_parser_t *parser,
    uint8_t byte,
    hw_proto_frame_t *out_frame
);

/** Decode 16-byte IMG_BEGIN payload. Returns 0 on success, -1 on length mismatch. */
int hw_proto_decode_img_begin(
    const uint8_t *payload,
    uint16_t len,
    hw_img_begin_payload_t *out
);

/** Encode 65-byte INFO payload into `out`. */
void hw_proto_encode_info(uint8_t out[HW_INFO_PAYLOAD_BYTES], const hw_info_payload_t *info);

/** Encode 5-byte CHUNK_ACK payload into `out`. */
void hw_proto_encode_chunk_ack(uint8_t out[HW_CHUNK_ACK_PAYLOAD_BYTES], uint32_t offset, int8_t status);

/** Encode 5-byte ACK_RX payload into `out`. */
void hw_proto_encode_ack_rx(uint8_t out[HW_ACK_RX_PAYLOAD_BYTES], uint32_t image_id, uint8_t crc_ok);

/** Encode 101-byte RESULT payload into `out`. */
void hw_proto_encode_result(uint8_t out[HW_RESULT_PAYLOAD_BYTES], const hw_result_payload_t *res);

/** Encode ERROR payload into `out`. Returns encoded payload byte length (3..51). */
uint16_t hw_proto_encode_error(uint8_t *out, uint16_t cap, uint16_t code, const char *text);

/**
 * Seal header + CRC16 around a payload already written at `frame_buf + HW_PROTO_HEADER_BYTES`.
 * Returns total frame byte length (HW_PROTO_OVERHEAD_BYTES + payload_len), or 0 if buffer too small.
 */
uint16_t hw_proto_seal_in_place(
    uint8_t *frame_buf,
    size_t frame_cap,
    uint8_t version,
    uint8_t type,
    uint16_t seq,
    uint16_t payload_len
);

/**
 * Stream a frame header + payload + CRC16 without needing the full frame to fit in one buffer.
 * Uses `scratch` (at least 10 bytes) for header/CRC writes.
 */
void hw_proto_stream_send(
    hw_proto_write_fn write_fn,
    void *ctx,
    uint8_t version,
    uint8_t type,
    uint16_t seq,
    const uint8_t *payload,
    uint16_t payload_len
);

#ifdef __cplusplus
}
#endif

#endif /* HW_PROTOCOL_H */
