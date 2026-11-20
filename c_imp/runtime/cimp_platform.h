/**
 * @file cimp_platform.h
 * @brief High-resolution monotonic nanosecond timer abstraction (Windows/POSIX/MCU hook).
 */
#ifndef CIMP_PLATFORM_H
#define CIMP_PLATFORM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Return monotonic elapsed clock reading in nanoseconds.
 * Uses QueryPerformanceCounter on Windows, clock_gettime on POSIX, or an external
 * cimp_mcu_time_ns() hook on bare-metal / ESP32 targets.
 */
uint64_t cimp_time_ns(void);

#ifdef __cplusplus
}
#endif

#endif /* CIMP_PLATFORM_H */
