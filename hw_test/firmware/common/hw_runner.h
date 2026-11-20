/**
 * @file hw_runner.h
 * @brief MCU protocol state machine and timed inference runner interface.
 */
#ifndef HW_RUNNER_H
#define HW_RUNNER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*hw_tx_write_fn)(const uint8_t *data, size_t len, void *ctx);
typedef void (*hw_tx_flush_fn)(void *ctx);

typedef struct {
    hw_tx_write_fn write;
    hw_tx_flush_fn flush;
    void *ctx;
    uint8_t *arena;
    size_t arena_bytes;
    uint8_t arena_location; /* 0=none, 1=sram, 2=psram */
} hw_runner_config_t;

/** Initialize the protocol parser, transfer state, and transport callbacks. */
void hw_runner_init(const hw_runner_config_t *cfg);

/** Feed one incoming transport byte to the protocol state machine. */
void hw_runner_feed_byte(uint8_t byte);

/** Feed a slice of incoming transport bytes to the protocol state machine. */
void hw_runner_feed_bytes(const uint8_t *data, size_t len);

/** Reset parser and transfer state on TCP client disconnect. */
void hw_runner_on_disconnect(void);

#ifdef __cplusplus
}
#endif

#endif /* HW_RUNNER_H */
