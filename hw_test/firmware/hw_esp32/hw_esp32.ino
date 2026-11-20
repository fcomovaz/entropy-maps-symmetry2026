/**
 * @file hw_esp32.ino
 * @brief ESP32-CAM firmware supporting UART or WiFi/TCP transport and SRAM/PSRAM arena allocation.
 */
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_task_wdt.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <stdint.h>

#include "hw_build_config.h"
#include "hw_board.h"
#include "hw_protocol.h"
#include "hw_runner.h"
#include "cimp_platform.h"

#if HW_USE_WIFI
#include <WiFi.h>
#endif

static uint32_t g_free_ram_boot = 0UL;
static uint32_t g_min_free_psram = HW_UNAVAILABLE_U32;
static uint8_t *g_arena_ptr = NULL;

static void esp32_update_min_psram(void) {
    if (!psramFound()) {
        return;
    }
    const uint32_t cur = (uint32_t)heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    if (cur < g_min_free_psram) {
        g_min_free_psram = cur;
    }
}

extern "C" uint64_t cimp_mcu_time_ns(void) {
    return (uint64_t)micros() * 1000ULL;
}

extern "C" uint32_t hw_board_free_ram_boot(void) {
    return g_free_ram_boot;
}

extern "C" uint32_t hw_board_free_heap_min(void) {
    return (uint32_t)heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
}

extern "C" uint32_t hw_board_stack_hwm(void) {
    return (uint32_t)uxTaskGetStackHighWaterMark(NULL) * (uint32_t)sizeof(StackType_t);
}

extern "C" uint32_t hw_board_psram_size(void) {
    return psramFound() ? (uint32_t)ESP.getPsramSize() : 0UL;
}

extern "C" uint32_t hw_board_free_psram_min(void) {
    if (!psramFound()) {
        return HW_UNAVAILABLE_U32;
    }
    esp32_update_min_psram();
    return g_min_free_psram;
}

extern "C" void hw_board_yield(void) {
    esp32_update_min_psram();
    yield();
}

extern "C" uint32_t hw_board_micros(void) {
    return (uint32_t)micros();
}

static void esp32_alloc_arena(
    uint8_t **out_ptr,
    size_t *out_bytes,
    uint8_t *out_loc
) {
    *out_ptr = NULL;
    *out_bytes = 0U;
    *out_loc = HW_ARENA_LOC_NONE;

#if (HW_MODEL_KIND == 1) && (HW_ARENA_BYTES > 0)
    const size_t req = (size_t)HW_ARENA_BYTES;
    uint8_t *buf = NULL;

#if HW_ARENA_PREF == 1
    buf = (uint8_t *)heap_caps_malloc(req, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    if (buf != NULL) {
        *out_ptr = buf;
        *out_bytes = req;
        *out_loc = HW_ARENA_LOC_SRAM;
    }
#elif HW_ARENA_PREF == 2
    if (psramFound()) {
        buf = (uint8_t *)heap_caps_malloc(req, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
        if (buf != NULL) {
            *out_ptr = buf;
            *out_bytes = req;
            *out_loc = HW_ARENA_LOC_PSRAM;
        }
    }
#elif HW_ARENA_PREF == 3
    if ((uint32_t)req <= (uint32_t)HW_USABLE_SRAM_BUDGET) {
        buf = (uint8_t *)heap_caps_malloc(req, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
        if (buf != NULL) {
            *out_ptr = buf;
            *out_bytes = req;
            *out_loc = HW_ARENA_LOC_SRAM;
            return;
        }
    }
    if (psramFound()) {
        buf = (uint8_t *)heap_caps_malloc(req, MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
        if (buf != NULL) {
            *out_ptr = buf;
            *out_bytes = req;
            *out_loc = HW_ARENA_LOC_PSRAM;
        }
    }
#endif
#endif /* (HW_MODEL_KIND == 1) && (HW_ARENA_BYTES > 0) */
}

#if HW_USE_WIFI

static WiFiServer g_server((uint16_t)HW_TCP_PORT);
static WiFiClient g_client;
static bool g_client_active = false;

static void esp32_tx_write(const uint8_t *data, size_t len, void *ctx) {
    (void)ctx;
    if (data == NULL || len == 0U || !g_client_active) {
        return;
    }
    size_t off = 0U;
    while (off < len && g_client && g_client.connected()) {
        size_t n = g_client.write(data + off, len - off);
        if (n == 0U) {
            yield();
        } else {
            off += n;
        }
    }
}

static void esp32_tx_flush(void *ctx) {
    (void)ctx;
    if (g_client) {
        g_client.flush();
    }
}

#else /* !HW_USE_WIFI */

static void esp32_tx_write(const uint8_t *data, size_t len, void *ctx) {
    (void)ctx;
    if (data != NULL && len > 0U) {
        Serial.write(data, len);
    }
}

static void esp32_tx_flush(void *ctx) {
    (void)ctx;
    Serial.flush();
}

#endif /* HW_USE_WIFI */

void setup() {
    if (esp_task_wdt_status(NULL) == ESP_OK) {
        esp_task_wdt_delete(NULL);
    }

#if HW_USE_WIFI
    Serial.begin(115200UL);
    WiFi.persistent(false);
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(HW_WIFI_SSID, HW_WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) {
        delay(250);
    }
    Serial.print("HW_TEST_WIFI_IP=");
    Serial.print(WiFi.localIP());
    Serial.print(" PORT=");
    Serial.println((unsigned)HW_TCP_PORT);
    Serial.flush();

    g_server.begin();
    g_server.setNoDelay(true);
#else
    Serial.setRxBufferSize(2048U);
    Serial.begin((unsigned long)HW_UART_BAUD);
#endif

    size_t arena_bytes = 0U;
    uint8_t arena_loc = HW_ARENA_LOC_NONE;
    esp32_alloc_arena(&g_arena_ptr, &arena_bytes, &arena_loc);

    g_free_ram_boot = (uint32_t)heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    if (psramFound()) {
        g_min_free_psram = (uint32_t)heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    } else {
        g_min_free_psram = HW_UNAVAILABLE_U32;
    }

    hw_runner_config_t cfg;
    cfg.write = esp32_tx_write;
    cfg.flush = esp32_tx_flush;
    cfg.ctx = NULL;
    cfg.arena = g_arena_ptr;
    cfg.arena_bytes = arena_bytes;
    cfg.arena_location = arena_loc;
    hw_runner_init(&cfg);
}

void loop() {
#if HW_USE_WIFI
    if (g_server.hasClient()) {
        WiFiClient next_client = g_server.accept();
        if (next_client) {
            if (g_client_active) {
                g_client.stop();
                hw_runner_on_disconnect();
            }
            g_client = next_client;
            g_client.setNoDelay(true);
            g_client_active = true;
        }
    }

    if (g_client_active) {
        while (g_client.available() > 0) {
            uint8_t buf[128];
            int avail = g_client.available();
            size_t to_read = ((size_t)avail > sizeof(buf)) ? sizeof(buf) : (size_t)avail;
            int n = g_client.read(buf, to_read);
            if (n > 0) {
                hw_runner_feed_bytes(buf, (size_t)n);
            } else {
                break;
            }
        }
        if (!g_client.connected() && g_client.available() == 0) {
            g_client.stop();
            g_client_active = false;
            hw_runner_on_disconnect();
        }
    }
#else
    while (Serial.available() > 0) {
        int ch = Serial.read();
        if (ch >= 0) {
            hw_runner_feed_byte((uint8_t)ch);
        }
    }
#endif
}
