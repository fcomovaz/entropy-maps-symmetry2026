/**
 * @file hw_esp8266.ino
 * @brief NodeMCU ESP8266 (ESP-12E) firmware supporting UART or WiFi/TCP transport.
 */
#include <Arduino.h>
#include <stdint.h>

#include "hw_build_config.h"
#include "hw_board.h"
#include "hw_protocol.h"
#include "hw_runner.h"
#include "cimp_platform.h"

#if HW_USE_WIFI
#include <ESP8266WiFi.h>
#endif

static uint32_t g_free_ram_boot = 0UL;
static uint32_t g_min_free_heap = 0xFFFFFFFFUL;

static void esp8266_update_min_heap(void) {
    const uint32_t cur = (uint32_t)ESP.getFreeHeap();
    if (cur < g_min_free_heap) {
        g_min_free_heap = cur;
    }
}

extern "C" uint64_t cimp_mcu_time_ns(void) {
    return (uint64_t)micros() * 1000ULL;
}

extern "C" uint32_t hw_board_free_ram_boot(void) {
    return g_free_ram_boot;
}

extern "C" uint32_t hw_board_free_heap_min(void) {
    esp8266_update_min_heap();
    return g_min_free_heap;
}

extern "C" uint32_t hw_board_stack_hwm(void) {
    return HW_UNAVAILABLE_U32;
}

extern "C" uint32_t hw_board_psram_size(void) {
    return 0UL;
}

extern "C" uint32_t hw_board_free_psram_min(void) {
    return HW_UNAVAILABLE_U32;
}

extern "C" void hw_board_yield(void) {
    esp8266_update_min_heap();
    yield();
}

extern "C" uint32_t hw_board_micros(void) {
    return (uint32_t)micros();
}

#if HW_USE_WIFI

static WiFiServer g_server((uint16_t)HW_TCP_PORT);
static WiFiClient g_client;
static bool g_client_active = false;

static void esp8266_tx_write(const uint8_t *data, size_t len, void *ctx) {
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

static void esp8266_tx_flush(void *ctx) {
    (void)ctx;
    if (g_client) {
        g_client.flush();
    }
}

#else /* !HW_USE_WIFI */

static void esp8266_tx_write(const uint8_t *data, size_t len, void *ctx) {
    (void)ctx;
    if (data != NULL && len > 0U) {
        Serial.write(data, len);
    }
}

static void esp8266_tx_flush(void *ctx) {
    (void)ctx;
    Serial.flush();
}

#endif /* HW_USE_WIFI */

void setup() {
#if HW_USE_WIFI
    Serial.begin(115200UL);
    WiFi.persistent(false);
    WiFi.mode(WIFI_STA);
    WiFi.setSleepMode(WIFI_NONE_SLEEP);
    WiFi.begin(HW_WIFI_SSID, HW_WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) {
        delay(250);
    }
    Serial.print(F("HW_TEST_WIFI_IP="));
    Serial.print(WiFi.localIP());
    Serial.print(F(" PORT="));
    Serial.println((unsigned)HW_TCP_PORT);
    Serial.flush();

    g_server.begin();
    g_server.setNoDelay(true);
#else
    Serial.setRxBufferSize(2048U);
    Serial.begin((unsigned long)HW_UART_BAUD);
#endif

    g_free_ram_boot = (uint32_t)ESP.getFreeHeap();
    g_min_free_heap = g_free_ram_boot;

    hw_runner_config_t cfg;
    cfg.write = esp8266_tx_write;
    cfg.flush = esp8266_tx_flush;
    cfg.ctx = NULL;
    cfg.arena = NULL;
    cfg.arena_bytes = 0U;
    cfg.arena_location = HW_ARENA_LOC_NONE;
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
