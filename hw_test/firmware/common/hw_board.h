/**
 * @file hw_board.h
 * @brief Platform-specific memory, timing, and yield probes implemented by each board sketch.
 */
#ifndef HW_BOARD_H
#define HW_BOARD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Free RAM/heap (bytes) measured once at boot after setup initialization. */
uint32_t hw_board_free_ram_boot(void);

/** Minimum free heap (bytes) observed since boot, or 0xFFFFFFFF if unavailable (AVR). */
uint32_t hw_board_free_heap_min(void);

/** Stack high-water mark (minimum remaining free stack bytes), or 0xFFFFFFFF if unavailable (ESP8266). */
uint32_t hw_board_stack_hwm(void);

/** Total PSRAM size (bytes), or 0 if the board has no PSRAM. */
uint32_t hw_board_psram_size(void);

/** Minimum free PSRAM (bytes) observed since boot, or 0xFFFFFFFF when no PSRAM is present. */
uint32_t hw_board_free_psram_min(void);

/** Cooperative yield between benchmark repeats (outside timed regions); no-op on AVR. */
void hw_board_yield(void);

/** Monotonic 32-bit microsecond timer tick (4 us resolution on AVR 16 MHz, 1 us on ESP). */
uint32_t hw_board_micros(void);

#ifdef __cplusplus
}
#endif

#endif /* HW_BOARD_H */
