/**
 * @file test_streaming.c
 * @brief C99 parity and error-handling tests for the streaming Q15 x94 descriptor API.
 *
 * Compiled with CIMP_ENABLE_F32=0 (without <math.h>) to verify zero float/math
 * dependency and bit-identical output against `cimp_descriptor_x94_q15`.
 */
#include <stdio.h>
#include <string.h>

#include "cimp_config.h"
#include "cimp_descriptor.h"
#include "test_image_0.h"
#include "test_image_1.h"
#include "test_image_2.h"
#include "test_image_3.h"

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, (msg)); \
            return 1;                                                        \
        }                                                                    \
    } while (0)

static CIMP_DescriptorWorkspace g_ws;
static CIMP_DescStream g_stream;

static int test_image_streaming(int img_idx, const uint8_t *img) {
    q15_t ref_x94[CIMP_N_X94];
    q15_t stream_x94[CIMP_N_X94];
    const int32_t band_sizes[5] = {1, 2, 4, 8, 16};

    CHECK(cimp_descriptor_x94_q15(img, &g_ws, ref_x94) == CIMP_OK, "reference x94 q15");

    for (int bi = 0; bi < 5; ++bi) {
        const int32_t band_rows = band_sizes[bi];
        CHECK(cimp_desc_stream_init(&g_stream) == CIMP_OK, "stream init");
        for (int32_t row = 0; row < CIMP_IMAGE_SIZE; row += band_rows) {
            const uint8_t *chunk = img + (size_t)row * (size_t)CIMP_IMAGE_SIZE;
            CHECK(
                cimp_desc_stream_push_rows(&g_stream, chunk, band_rows) == CIMP_OK,
                "stream push rows"
            );
        }
        memset(stream_x94, 0x5A, sizeof(stream_x94));
        CHECK(cimp_desc_stream_finish_q15(&g_stream, stream_x94) == CIMP_OK, "stream finish");

        for (int f = 0; f < CIMP_N_X94; ++f) {
            if (stream_x94[f] != ref_x94[f]) {
                fprintf(
                    stderr,
                    "MISMATCH image=%d band_rows=%d feat=%d: stream=%d ref=%d\n",
                    img_idx,
                    (int)band_rows,
                    f,
                    (int)stream_x94[f],
                    (int)ref_x94[f]
                );
                return 1;
            }
        }
    }

    /* Mixed cross-band chunking check (3 + 15 + remaining rows) */
    CHECK(cimp_desc_stream_init(&g_stream) == CIMP_OK, "mixed chunk init");
    CHECK(cimp_desc_stream_push_rows(&g_stream, img, 3) == CIMP_OK, "mixed chunk 1");
    CHECK(
        cimp_desc_stream_push_rows(&g_stream, img + (size_t)3 * (size_t)CIMP_IMAGE_SIZE, 15) == CIMP_OK,
        "mixed chunk 2"
    );
    CHECK(
        cimp_desc_stream_push_rows(
            &g_stream,
            img + (size_t)18 * (size_t)CIMP_IMAGE_SIZE,
            CIMP_IMAGE_SIZE - 18
        ) == CIMP_OK,
        "mixed chunk 3"
    );
    CHECK(cimp_desc_stream_finish_q15(&g_stream, stream_x94) == CIMP_OK, "mixed chunk finish");
    CHECK(
        memcmp(stream_x94, ref_x94, sizeof(ref_x94)) == 0,
        "mixed cross-band chunk bit-identical"
    );

    return 0;
}

static int test_invalid_args(void) {
    q15_t out_x94[CIMP_N_X94];

    CHECK(cimp_desc_stream_init(NULL) == CIMP_ERR_INVALID_ARG, "init NULL");
    CHECK(
        cimp_desc_stream_push_rows(NULL, CIMP_TEST_IMAGE_0_U8, 1) == CIMP_ERR_INVALID_ARG,
        "push NULL stream"
    );
    CHECK(cimp_desc_stream_init(&g_stream) == CIMP_OK, "re-init");
    CHECK(
        cimp_desc_stream_push_rows(&g_stream, NULL, 1) == CIMP_ERR_INVALID_ARG,
        "push NULL rows"
    );
    CHECK(
        cimp_desc_stream_push_rows(&g_stream, CIMP_TEST_IMAGE_0_U8, 0) == CIMP_ERR_INVALID_ARG,
        "push 0 rows"
    );
    CHECK(
        cimp_desc_stream_push_rows(&g_stream, CIMP_TEST_IMAGE_0_U8, -4) == CIMP_ERR_INVALID_ARG,
        "push negative rows"
    );
    CHECK(
        cimp_desc_stream_finish_q15(NULL, out_x94) == CIMP_ERR_INVALID_ARG,
        "finish NULL stream"
    );
    CHECK(
        cimp_desc_stream_finish_q15(&g_stream, NULL) == CIMP_ERR_INVALID_ARG,
        "finish NULL out"
    );
    CHECK(
        cimp_desc_stream_finish_q15(&g_stream, out_x94) == CIMP_ERR_SHAPE_MISMATCH,
        "finish before any rows pushed"
    );

    CHECK(
        cimp_desc_stream_push_rows(&g_stream, CIMP_TEST_IMAGE_0_U8, 16) == CIMP_OK,
        "partial push 16 rows"
    );
    CHECK(
        cimp_desc_stream_finish_q15(&g_stream, out_x94) == CIMP_ERR_SHAPE_MISMATCH,
        "finish with partial rows"
    );
    CHECK(
        cimp_desc_stream_push_rows(&g_stream, CIMP_TEST_IMAGE_0_U8, CIMP_IMAGE_SIZE) ==
            CIMP_ERR_SHAPE_MISMATCH,
        "push overflowing past image height"
    );
    CHECK(
        cimp_desc_stream_push_rows(
            &g_stream,
            CIMP_TEST_IMAGE_0_U8 + (size_t)16 * (size_t)CIMP_IMAGE_SIZE,
            CIMP_IMAGE_SIZE - 16
        ) == CIMP_OK,
        "complete remaining rows"
    );
    CHECK(
        cimp_desc_stream_push_rows(&g_stream, CIMP_TEST_IMAGE_0_U8, 1) == CIMP_ERR_SHAPE_MISMATCH,
        "push extra row after completion"
    );
    CHECK(cimp_desc_stream_finish_q15(&g_stream, out_x94) == CIMP_OK, "finish after overflow guard");

    return 0;
}

int main(void) {
    const uint8_t *images[4] = {
        CIMP_TEST_IMAGE_0_U8,
        CIMP_TEST_IMAGE_1_U8,
        CIMP_TEST_IMAGE_2_U8,
        CIMP_TEST_IMAGE_3_U8
    };

    printf(
        "sizeof(CIMP_DescStream) = %zu bytes (sizeof(CIMP_DescriptorWorkspace) = %zu bytes)\n",
        sizeof(CIMP_DescStream),
        sizeof(CIMP_DescriptorWorkspace)
    );
    CHECK(
        sizeof(CIMP_DescStream) < sizeof(CIMP_DescriptorWorkspace),
        "stream struct must be smaller than full workspace"
    );

    for (int i = 0; i < 4; ++i) {
        if (test_image_streaming(i, images[i]) != 0) {
            return 1;
        }
    }
    if (test_invalid_args() != 0) {
        return 1;
    }

    printf("cimp test_streaming: ALL PASSED\n");
    return 0;
}
