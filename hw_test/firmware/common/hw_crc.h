/**
 * @file hw_crc.h
 * @brief Table-free bitwise CRC16-CCITT-FALSE and IEEE 802.3 CRC32 helpers.
 *
 * Uses zero static lookup-table RAM so it fits comfortably on AVR (2 KiB SRAM).
 */
#ifndef HW_CRC_H
#define HW_CRC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Initial state for CRC16-CCITT-FALSE (poly 0x1021, init 0xFFFF, no xorout). */
static inline uint16_t hw_crc16_ccitt_init(void) {
    return (uint16_t)0xFFFFU;
}

/** Feed a single byte into CRC16-CCITT-FALSE. */
static inline uint16_t hw_crc16_ccitt_update(uint16_t crc, uint8_t byte) {
    crc ^= (uint16_t)((uint16_t)byte << 8);
    for (uint8_t i = 0U; i < 8U; ++i) {
        if ((crc & 0x8000U) != 0U) {
            crc = (uint16_t)(((uint16_t)(crc << 1)) ^ 0x1021U);
        } else {
            crc = (uint16_t)(crc << 1);
        }
    }
    return crc;
}

/** Feed a contiguous byte slice into CRC16-CCITT-FALSE. */
static inline uint16_t hw_crc16_ccitt_block(
    uint16_t crc,
    const uint8_t *data,
    size_t len
) {
    if (data == NULL) {
        return crc;
    }
    for (size_t i = 0U; i < len; ++i) {
        crc = hw_crc16_ccitt_update(crc, data[i]);
    }
    return crc;
}

/** Initial state for IEEE 802.3 CRC32 (poly 0xEDB88320, init 0xFFFFFFFF). */
static inline uint32_t hw_crc32_init(void) {
    return 0xFFFFFFFFUL;
}

/** Feed a contiguous byte slice into reflected IEEE 802.3 CRC32. */
static inline uint32_t hw_crc32_update(
    uint32_t state,
    const uint8_t *data,
    size_t len
) {
    if (data == NULL) {
        return state;
    }
    for (size_t i = 0U; i < len; ++i) {
        state ^= (uint32_t)data[i];
        for (uint8_t b = 0U; b < 8U; ++b) {
            if ((state & 1UL) != 0UL) {
                state = (state >> 1) ^ 0xEDB88320UL;
            } else {
                state >>= 1;
            }
        }
    }
    return state;
}

/** Finalize IEEE 802.3 CRC32 state. */
static inline uint32_t hw_crc32_final(uint32_t state) {
    return state ^ 0xFFFFFFFFUL;
}

#ifdef __cplusplus
}
#endif

#endif /* HW_CRC_H */
