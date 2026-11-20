/**
 * @file hw_avr.ino
 * @brief Arduino AVR (ATmega328P Nano/UNO) streaming UART firmware for hw_test.
 */
#include <Arduino.h>
#include <avr/interrupt.h>
#include <avr/io.h>
#include <stdint.h>

#include "hw_build_config.h"
#include "hw_board.h"
#include "hw_protocol.h"
#include "hw_runner.h"
#include "cimp_platform.h"

extern int __heap_start;
extern void *__brkval;

static uint32_t g_free_ram_boot = 0UL;

static uint32_t avr_free_ram_now(void) {
    const uintptr_t heap_end = (__brkval == 0) ? (uintptr_t)&__heap_start : (uintptr_t)__brkval;
    const uintptr_t sp = (uintptr_t)SP;
    return (sp > heap_end) ? (uint32_t)(sp - heap_end) : 0UL;
}

static void avr_paint_stack_canary(void) {
    if (__brkval != 0) {
        return;
    }
    const uint8_t saved_sreg = SREG;
    cli();
    const uintptr_t sp = (uintptr_t)SP;
    volatile uint8_t *p = (volatile uint8_t *)&__heap_start;
    if (sp > (uintptr_t)p + 16U) {
        const uintptr_t limit = sp - 16U;
        while ((uintptr_t)p < limit) {
            *p++ = 0xA5U;
        }
    }
    SREG = saved_sreg;
}

extern "C" uint64_t cimp_mcu_time_ns(void) {
    return (uint64_t)micros() * 1000ULL;
}

extern "C" uint32_t hw_board_free_ram_boot(void) {
    return g_free_ram_boot;
}

extern "C" uint32_t hw_board_free_heap_min(void) {
    return HW_UNAVAILABLE_U32;
}

extern "C" uint32_t hw_board_stack_hwm(void) {
    if (__brkval != 0) {
        return avr_free_ram_now();
    }
    const volatile uint8_t *p = (const volatile uint8_t *)&__heap_start;
    const uintptr_t sp = (uintptr_t)SP;
    uint32_t unused = 0UL;
    while ((uintptr_t)p < sp && *p == 0xA5U) {
        ++unused;
        ++p;
    }
    return unused;
}

extern "C" uint32_t hw_board_psram_size(void) {
    return 0UL;
}

extern "C" uint32_t hw_board_free_psram_min(void) {
    return HW_UNAVAILABLE_U32;
}

extern "C" void hw_board_yield(void) {
    /* Single-threaded bare-metal AVR: nothing to yield. */
}

extern "C" uint32_t hw_board_micros(void) {
    return (uint32_t)micros();
}

static void avr_tx_write(const uint8_t *data, size_t len, void *ctx) {
    (void)ctx;
    if (data != NULL && len > 0U) {
        Serial.write(data, len);
    }
}

static void avr_tx_flush(void *ctx) {
    (void)ctx;
    Serial.flush();
}

void setup() {
    Serial.begin((unsigned long)HW_UART_BAUD);
    g_free_ram_boot = avr_free_ram_now();
    avr_paint_stack_canary();

    hw_runner_config_t cfg;
    cfg.write = avr_tx_write;
    cfg.flush = avr_tx_flush;
    cfg.ctx = NULL;
    cfg.arena = NULL;
    cfg.arena_bytes = 0U;
    cfg.arena_location = HW_ARENA_LOC_NONE;
    hw_runner_init(&cfg);
}

void loop() {
    while (Serial.available() > 0) {
        int ch = Serial.read();
        if (ch >= 0) {
            hw_runner_feed_byte((uint8_t)ch);
        }
    }
}
