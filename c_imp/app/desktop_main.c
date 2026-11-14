/**
 * @file desktop_main.c
 * @brief Desktop verification CLI executing CIMP descriptor, classical, and CNN pipelines.
 */
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cimp_config.h"
#include "cimp_descriptor.h"
#include "cimp_models_registry.h"
#include "cimp_pipeline.h"
#include "cimp_platform.h"
#include "cimp_tensor.h"

#define CIMP_PIXELS (CIMP_IMAGE_SIZE * CIMP_IMAGE_SIZE)
#define CIMP_MAX_MANIFEST_ROWS 4096

typedef struct {
    int index;
    int label;
    char file_name[256];
    char bin_relpath[256];
} ManifestRow;

/* Static aligned runtime buffers matching MCU embedded footprint (zero heap in inference). */
static CIMP_DescriptorWorkspace g_desc_ws;
static uint8_t g_img_buf[CIMP_PIXELS];
static uint64_t g_arena_words[(CIMP_MAX_CNN_ARENA_BYTES + 7) / 8];
static ManifestRow g_rows[CIMP_MAX_MANIFEST_ROWS];

static double ldexp_neg(int32_t val, int frac_bits) {
    return ldexp((double)val, -frac_bits);
}

static int parse_manifest_csv(const char *path, ManifestRow *out_rows, int max_rows) {
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "error: cannot open manifest: %s\n", path);
        return -1;
    }
    char line[1024];
    int count = 0;
    int line_no = 0;
    while (fgets(line, (int)sizeof(line), fp) != NULL) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len == 0) {
            continue;
        }
        line_no++;
        if (line_no == 1) {
            continue; /* header */
        }
        if (count >= max_rows) {
            fprintf(stderr, "error: manifest exceeds %d rows\n", max_rows);
            fclose(fp);
            return -1;
        }
        /* index,file_name,split,label,bin_relpath,sha256 */
        char *p1 = strchr(line, ',');
        char *p2 = (p1 != NULL) ? strchr(p1 + 1, ',') : NULL;
        char *p3 = (p2 != NULL) ? strchr(p2 + 1, ',') : NULL;
        char *p4 = (p3 != NULL) ? strchr(p3 + 1, ',') : NULL;
        char *p5 = (p4 != NULL) ? strchr(p4 + 1, ',') : NULL;
        if (p1 == NULL || p2 == NULL || p3 == NULL || p4 == NULL || p5 == NULL) {
            fprintf(stderr, "error: malformed manifest line %d\n", line_no);
            fclose(fp);
            return -1;
        }
        *p1 = '\0';
        *p2 = '\0';
        *p3 = '\0';
        *p4 = '\0';
        *p5 = '\0';
        ManifestRow *r = &out_rows[count];
        r->index = atoi(line);
        snprintf(r->file_name, sizeof(r->file_name), "%s", p1 + 1);
        r->label = atoi(p3 + 1);
        snprintf(r->bin_relpath, sizeof(r->bin_relpath), "%s", p4 + 1);
        count++;
    }
    fclose(fp);
    return count;
}

static int load_image_bin(const char *images_dir, const char *relpath, uint8_t *dst, size_t expected) {
    char full_path[1024];
    snprintf(full_path, sizeof(full_path), "%s/%s", images_dir, relpath);
    FILE *fp = fopen(full_path, "rb");
    if (fp == NULL) {
        fprintf(stderr, "error: cannot open image bin: %s\n", full_path);
        return -1;
    }
    size_t got = fread(dst, 1, expected, fp);
    fclose(fp);
    if (got != expected) {
        fprintf(stderr, "error: expected %zu bytes in %s, got %zu\n", expected, full_path, got);
        return -1;
    }
    return 0;
}

static void print_usage(void) {
    fprintf(
        stderr,
        "Usage: cimp_desktop --manifest <csv> --images-dir <dir> --mode <descriptor|cnn> "
        "--model <name> --arith <f32|q15> --out-csv <csv> [--x94-csv <csv>] "
        "[--bench-warmup N] [--bench-reps N] [--bench-csv <csv>]\n"
    );
}

int main(int argc, char **argv) {
    const char *manifest_path = NULL;
    const char *images_dir = NULL;
    const char *mode = NULL;
    const char *model_name = NULL;
    const char *arith = NULL;
    const char *out_csv = NULL;
    const char *x94_csv = NULL;
    const char *bench_csv = NULL;
    int bench_warmup = 0;
    int bench_reps = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (strcmp(argv[i], "--images-dir") == 0 && i + 1 < argc) {
            images_dir = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_name = argv[++i];
        } else if (strcmp(argv[i], "--arith") == 0 && i + 1 < argc) {
            arith = argv[++i];
        } else if (strcmp(argv[i], "--out-csv") == 0 && i + 1 < argc) {
            out_csv = argv[++i];
        } else if (strcmp(argv[i], "--x94-csv") == 0 && i + 1 < argc) {
            x94_csv = argv[++i];
        } else if (strcmp(argv[i], "--bench-warmup") == 0 && i + 1 < argc) {
            bench_warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench-reps") == 0 && i + 1 < argc) {
            bench_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bench-csv") == 0 && i + 1 < argc) {
            bench_csv = argv[++i];
        } else {
            print_usage();
            return 2;
        }
    }

    if (manifest_path == NULL || images_dir == NULL || mode == NULL ||
        model_name == NULL || arith == NULL || out_csv == NULL) {
        print_usage();
        return 2;
    }

    int is_q15 = (strcmp(arith, "q15") == 0) ? 1 : 0;
    if (!is_q15 && strcmp(arith, "f32") != 0) {
        fprintf(stderr, "error: invalid --arith '%s'\n", arith);
        return 2;
    }
    int is_descriptor = (strcmp(mode, "descriptor") == 0) ? 1 : 0;
    int is_cnn = (strcmp(mode, "cnn") == 0) ? 1 : 0;
    if (!is_descriptor && !is_cnn) {
        fprintf(stderr, "error: invalid --mode '%s'\n", mode);
        return 2;
    }
    if (is_descriptor && strcmp(model_name, "x94_only") != 0 &&
        !cimp_is_tabular_model_supported(model_name)) {
        fprintf(stderr, "error: unsupported descriptor model '%s'\n", model_name);
        return 3;
    }
    if (is_cnn && !cimp_is_cnn_model_supported(model_name)) {
        fprintf(stderr, "error: unsupported CNN model '%s'\n", model_name);
        return 3;
    }

    int n_rows = parse_manifest_csv(manifest_path, g_rows, CIMP_MAX_MANIFEST_ROWS);
    if (n_rows <= 0) {
        fprintf(stderr, "error: no rows loaded from manifest\n");
        return 1;
    }

    FILE *fp_out = fopen(out_csv, "w");
    if (fp_out == NULL) {
        fprintf(stderr, "error: cannot create %s\n", out_csv);
        return 1;
    }
    FILE *fp_x94 = NULL;
    if (x94_csv != NULL && is_descriptor) {
        fp_x94 = fopen(x94_csv, "w");
        if (fp_x94 == NULL) {
            fclose(fp_out);
            fprintf(stderr, "error: cannot create %s\n", x94_csv);
            return 1;
        }
        fprintf(fp_x94, "index,label");
        for (int f = 0; f < CIMP_N_X94; ++f) {
            fprintf(fp_x94, ",f%d", f);
        }
        fprintf(fp_x94, "\n");
    }

    if (is_descriptor) {
        fprintf(
            fp_out,
            "index,file_name,true_label,c_label,c_logit,c_logit_raw_q,out_frac,selector_ms,model_ms,total_ms\n"
        );
    } else {
        fprintf(
            fp_out,
            "index,file_name,true_label,c_label,c_logit0,c_logit1,c_margin,"
            "c_logit0_raw_q,c_logit1_raw_q,c_margin_raw_q,out_frac,selector_ms,model_ms,total_ms\n"
        );
    }

    int tab_frac = is_descriptor ? cimp_get_tabular_out_frac(model_name) : -1;
    int cnn_frac = is_cnn ? cimp_get_cnn_out_frac(model_name) : -1;
    uint8_t *arena_bytes_ptr = (uint8_t *)(void *)g_arena_words;
    size_t arena_cap = sizeof(g_arena_words);

    for (int i = 0; i < n_rows; ++i) {
        if (load_image_bin(images_dir, g_rows[i].bin_relpath, g_img_buf, CIMP_PIXELS) != 0) {
            fclose(fp_out);
            if (fp_x94 != NULL) {
                fclose(fp_x94);
            }
            return 1;
        }

        if (is_descriptor) {
            float x94_f32[CIMP_N_X94];
            q15_t x94_q15[CIMP_N_X94];
            uint64_t t0 = cimp_time_ns();
            int rc = is_q15 ? cimp_descriptor_x94_q15(g_img_buf, &g_desc_ws, x94_q15)
                            : cimp_descriptor_x94_f32(g_img_buf, &g_desc_ws, x94_f32);
            uint64_t t1 = cimp_time_ns();
            if (rc != CIMP_OK) {
                fprintf(stderr, "error: descriptor failed rc=%d\n", rc);
                fclose(fp_out);
                if (fp_x94 != NULL) {
                    fclose(fp_x94);
                }
                return 4;
            }
            if (fp_x94 != NULL) {
                fprintf(fp_x94, "%d,%d", g_rows[i].index, g_rows[i].label);
                for (int f = 0; f < CIMP_N_X94; ++f) {
                    if (is_q15) {
                        fprintf(fp_x94, ",%d", (int)x94_q15[f]);
                    } else {
                        fprintf(fp_x94, ",%.9g", (double)x94_f32[f]);
                    }
                }
                fprintf(fp_x94, "\n");
            }

            int pred_label = 0;
            float logit_f32 = 0.0f;
            int32_t logit_q = 0;
            uint64_t t2 = t1;
            uint64_t t3 = t1;
            if (strcmp(model_name, "x94_only") != 0) {
                t2 = cimp_time_ns();
                pred_label = is_q15 ? cimp_predict_tabular_q15(model_name, x94_q15, &logit_q)
                                    : cimp_predict_tabular_f32(model_name, x94_f32, &logit_f32);
                t3 = cimp_time_ns();
                if (pred_label < 0) {
                    fprintf(stderr, "error: tabular predict failed rc=%d\n", pred_label);
                    fclose(fp_out);
                    if (fp_x94 != NULL) {
                        fclose(fp_x94);
                    }
                    return 4;
                }
            }
            double sel_ms = (double)(t1 - t0) * 1e-6;
            double mod_ms = (double)(t3 - t2) * 1e-6;
            double logit_val = is_q15 ? ldexp_neg(logit_q, tab_frac) : (double)logit_f32;
            fprintf(
                fp_out,
                "%d,%s,%d,%d,%.9g,%d,%d,%.6f,%.6f,%.6f\n",
                g_rows[i].index,
                g_rows[i].file_name,
                g_rows[i].label,
                pred_label,
                logit_val,
                (int)logit_q,
                tab_frac,
                sel_ms,
                mod_ms,
                sel_ms + mod_ms
            );
        } else {
            int pred_label = 0;
            float logits_f32[2] = {0.0f, 0.0f};
            int32_t logits_q[2] = {0, 0};
            uint64_t t0 = cimp_time_ns();
            int rc = is_q15
                ? cimp_predict_cnn_q15(model_name, g_img_buf, arena_bytes_ptr, arena_cap, logits_q, &pred_label)
                : cimp_predict_cnn_f32(model_name, g_img_buf, arena_bytes_ptr, arena_cap, logits_f32, &pred_label);
            uint64_t t1 = cimp_time_ns();
            if (rc != CIMP_OK) {
                fprintf(stderr, "error: cnn predict failed rc=%d\n", rc);
                fclose(fp_out);
                return 4;
            }
            double mod_ms = (double)(t1 - t0) * 1e-6;
            double l0 = is_q15 ? ldexp_neg(logits_q[0], cnn_frac) : (double)logits_f32[0];
            double l1 = is_q15 ? ldexp_neg(logits_q[1], cnn_frac) : (double)logits_f32[1];
            int64_t m_raw = (int64_t)logits_q[1] - (int64_t)logits_q[0];
            double margin = l1 - l0;
            fprintf(
                fp_out,
                "%d,%s,%d,%d,%.9g,%.9g,%.9g,%d,%d,%lld,%d,0.000000,%.6f,%.6f\n",
                g_rows[i].index,
                g_rows[i].file_name,
                g_rows[i].label,
                pred_label,
                l0,
                l1,
                margin,
                (int)logits_q[0],
                (int)logits_q[1],
                (long long)m_raw,
                cnn_frac,
                mod_ms,
                mod_ms
            );
        }
    }

    fclose(fp_out);
    if (fp_x94 != NULL) {
        fclose(fp_x94);
    }

    if (bench_reps > 0 && bench_csv != NULL) {
        uint8_t *all_imgs = (uint8_t *)malloc((size_t)n_rows * (size_t)CIMP_PIXELS);
        if (all_imgs == NULL) {
            fprintf(stderr, "error: out of memory preloading benchmark images\n");
            return 5;
        }
        for (int i = 0; i < n_rows; ++i) {
            if (load_image_bin(images_dir, g_rows[i].bin_relpath, all_imgs + (size_t)i * CIMP_PIXELS, CIMP_PIXELS) != 0) {
                free(all_imgs);
                return 1;
            }
        }
        volatile int sink = 0;
        double acc_sel_ns = 0.0;
        double acc_mod_ns = 0.0;
        int total_passes = bench_warmup + bench_reps;
        for (int rep = 0; rep < total_passes; ++rep) {
            int record = (rep >= bench_warmup);
            const uint8_t *im = all_imgs + (size_t)(rep % n_rows) * CIMP_PIXELS;
            if (is_descriptor) {
                float x94_f32[CIMP_N_X94];
                q15_t x94_q15[CIMP_N_X94];
                uint64_t t0 = cimp_time_ns();
                if (is_q15) {
                    cimp_descriptor_x94_q15(im, &g_desc_ws, x94_q15);
                } else {
                    cimp_descriptor_x94_f32(im, &g_desc_ws, x94_f32);
                }
                uint64_t t1 = cimp_time_ns();
                int lbl = 0;
                uint64_t t2 = t1;
                uint64_t t3 = t1;
                if (strcmp(model_name, "x94_only") != 0) {
                    if (is_q15) {
                        int32_t lq = 0;
                        t2 = cimp_time_ns();
                        lbl = cimp_predict_tabular_q15(model_name, x94_q15, &lq);
                        t3 = cimp_time_ns();
                    } else {
                        float lf = 0.0f;
                        t2 = cimp_time_ns();
                        lbl = cimp_predict_tabular_f32(model_name, x94_f32, &lf);
                        t3 = cimp_time_ns();
                    }
                }
                sink ^= lbl;
                if (record) {
                    acc_sel_ns += (double)(t1 - t0);
                    acc_mod_ns += (double)(t3 - t2);
                }
            } else {
                int lbl = 0;
                uint64_t t0 = cimp_time_ns();
                if (is_q15) {
                    int32_t lq[2];
                    cimp_predict_cnn_q15(model_name, im, arena_bytes_ptr, arena_cap, lq, &lbl);
                } else {
                    float lf[2];
                    cimp_predict_cnn_f32(model_name, im, arena_bytes_ptr, arena_cap, lf, &lbl);
                }
                uint64_t t1 = cimp_time_ns();
                sink ^= lbl;
                if (record) {
                    acc_mod_ns += (double)(t1 - t0);
                }
            }
        }
        free(all_imgs);

        FILE *fp_b = fopen(bench_csv, "w");
        if (fp_b == NULL) {
            fprintf(stderr, "error: cannot create %s\n", bench_csv);
            return 1;
        }
        double denom = (double)bench_reps;
        double sel_ms = (acc_sel_ns / denom) * 1e-6;
        double mod_ms = (acc_mod_ns / denom) * 1e-6;
        fprintf(
            fp_b,
            "mode,model,arith,n_images,reps,selector_ms_per_img,model_ms_per_img,total_ms_per_img,sink\n"
        );
        fprintf(
            fp_b,
            "%s,%s,%s,%d,%d,%.6f,%.6f,%.6f,%d\n",
            mode,
            model_name,
            arith,
            n_rows,
            bench_reps,
            sel_ms,
            mod_ms,
            sel_ms + mod_ms,
            sink
        );
        fclose(fp_b);
    }

    return 0;
}
