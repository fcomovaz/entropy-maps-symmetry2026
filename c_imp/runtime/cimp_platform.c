/**
 * @file cimp_platform.c
 * @brief Platform timer implementation (desktop Windows/POSIX or weak MCU hook).
 */
#include "cimp_platform.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

uint64_t cimp_time_ns(void) {
    static LARGE_INTEGER freq;
    static int initialized = 0;
    LARGE_INTEGER counter;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    QueryPerformanceCounter(&counter);
    uint64_t c = (uint64_t)counter.QuadPart;
    uint64_t f = (uint64_t)freq.QuadPart;
    return (c / f) * 1000000000ULL + ((c % f) * 1000000000ULL) / f;
}
#elif defined(__unix__) || defined(__APPLE__)
#include <time.h>

uint64_t cimp_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
#else
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak)) uint64_t cimp_mcu_time_ns(void) {
    return 0ULL;
}
#else
extern uint64_t cimp_mcu_time_ns(void);
#endif

uint64_t cimp_time_ns(void) {
    return cimp_mcu_time_ns();
}
#endif
