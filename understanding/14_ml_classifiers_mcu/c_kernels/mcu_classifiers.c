#define MCU_CLASSIFIERS_EXPORTS
#include "mcu_classifiers.h"

#define MCU_MAX_HIDDEN 128

float mcu_dense_dot_score(
    const float *x,
    const float *w,
    float b,
    int32_t n
) {
    float acc = b;
    for (int32_t i = 0; i < n; ++i) {
        acc += w[i] * x[i];
    }
    return acc;
}

float mcu_sparse_dot_score(
    const float *x,
    const float *w_nz,
    const int32_t *idx_nz,
    float b,
    int32_t nnz
) {
    float acc = b;
    for (int32_t k = 0; k < nnz; ++k) {
        acc += w_nz[k] * x[idx_nz[k]];
    }
    return acc;
}

float mcu_diag_quad_score(
    const float *x,
    const float *a,
    const float *b_lin,
    float c,
    int32_t n
) {
    /* Accumulate per-feature terms before adding large GaussianNB log-ratio offset c. */
    float acc = 0.0f;
    for (int32_t i = 0; i < n; ++i) {
        const float xi = x[i];
        acc += xi * (a[i] * xi + b_lin[i]);
    }
    return acc + c;
}

float mcu_mlp_1hidden_score(
    const float *x,
    int32_t n_in,
    int32_t n_h,
    const float *w1,
    const float *b1,
    const float *w2,
    float b2,
    float *scratch_h
) {
    for (int32_t j = 0; j < n_h; ++j) {
        const float *row = w1 + j * n_in;
        float z = b1[j];
        for (int32_t i = 0; i < n_in; ++i) {
            z += row[i] * x[i];
        }
        scratch_h[j] = (z > 0.0f) ? z : 0.0f;
    }

    float out = b2;
    for (int32_t j = 0; j < n_h; ++j) {
        out += w2[j] * scratch_h[j];
    }
    return out;
}

float mcu_mlp_1hidden_skip_score(
    const float *x,
    int32_t n_in,
    int32_t n_h,
    const float *w_skip,
    float b_skip,
    const float *w1,
    const float *b1,
    const float *w2,
    float *scratch_h
) {
    float out = b_skip;
    for (int32_t i = 0; i < n_in; ++i) {
        out += w_skip[i] * x[i];
    }

    for (int32_t j = 0; j < n_h; ++j) {
        const float *row = w1 + j * n_in;
        float z = b1[j];
        for (int32_t i = 0; i < n_in; ++i) {
            z += row[i] * x[i];
        }
        scratch_h[j] = (z > 0.0f) ? z : 0.0f;
    }

    for (int32_t j = 0; j < n_h; ++j) {
        out += w2[j] * scratch_h[j];
    }
    return out;
}

float mcu_mlp_2hidden_score(
    const float *x,
    int32_t n_in,
    int32_t n_h1,
    int32_t n_h2,
    const float *w1,
    const float *b1,
    const float *w2,
    const float *b2,
    const float *w3,
    float b3,
    float *scratch_h1,
    float *scratch_h2
) {
    for (int32_t j = 0; j < n_h1; ++j) {
        const float *row = w1 + j * n_in;
        float z = b1[j];
        for (int32_t i = 0; i < n_in; ++i) {
            z += row[i] * x[i];
        }
        scratch_h1[j] = (z > 0.0f) ? z : 0.0f;
    }

    for (int32_t k = 0; k < n_h2; ++k) {
        const float *row = w2 + k * n_h1;
        float z = b2[k];
        for (int32_t j = 0; j < n_h1; ++j) {
            z += row[j] * scratch_h1[j];
        }
        scratch_h2[k] = (z > 0.0f) ? z : 0.0f;
    }

    float out = b3;
    for (int32_t k = 0; k < n_h2; ++k) {
        out += w3[k] * scratch_h2[k];
    }
    return out;
}

float mcu_tree_ensemble_score(
    const float *x,
    int32_t n_trees,
    const int32_t *tree_offsets,
    const int16_t *features,
    const float *thresholds,
    const int16_t *left_children,
    const int16_t *right_children,
    const float *leaf_values,
    float base_score,
    float scale
) {
    float tree_sum = 0.0f;
    for (int32_t t = 0; t < n_trees; ++t) {
        int32_t node = tree_offsets[t];
        while (left_children[node] >= 0) {
            const int16_t f_idx = features[node];
            if (x[f_idx] <= thresholds[node]) {
                node = (int32_t)left_children[node];
            } else {
                node = (int32_t)right_children[node];
            }
        }
        tree_sum += leaf_values[node];
    }
    return base_score + scale * tree_sum;
}

void mcu_dense_dot_batch(
    const float *X, int32_t batch_size, int32_t n_in,
    const float *w, float b, float *out_scores
) {
    for (int32_t row = 0; row < batch_size; ++row) {
        out_scores[row] = mcu_dense_dot_score(X + row * n_in, w, b, n_in);
    }
}

void mcu_sparse_dot_batch(
    const float *X, int32_t batch_size, int32_t n_in,
    const float *w_nz, const int32_t *idx_nz, float b, int32_t nnz, float *out_scores
) {
    for (int32_t row = 0; row < batch_size; ++row) {
        out_scores[row] = mcu_sparse_dot_score(X + row * n_in, w_nz, idx_nz, b, nnz);
    }
}

void mcu_diag_quad_batch(
    const float *X, int32_t batch_size, int32_t n_in,
    const float *a, const float *b_lin, float c, float *out_scores
) {
    for (int32_t row = 0; row < batch_size; ++row) {
        out_scores[row] = mcu_diag_quad_score(X + row * n_in, a, b_lin, c, n_in);
    }
}

void mcu_mlp_1hidden_batch(
    const float *X, int32_t batch_size, int32_t n_in, int32_t n_h,
    const float *w1, const float *b1, const float *w2, float b2, float *out_scores
) {
    float scratch_h[MCU_MAX_HIDDEN];
    if (n_h > MCU_MAX_HIDDEN) {
        for (int32_t row = 0; row < batch_size; ++row) {
            out_scores[row] = 0.0f;
        }
        return;
    }
    for (int32_t row = 0; row < batch_size; ++row) {
        out_scores[row] = mcu_mlp_1hidden_score(
            X + row * n_in, n_in, n_h, w1, b1, w2, b2, scratch_h
        );
    }
}

void mcu_mlp_1hidden_skip_batch(
    const float *X, int32_t batch_size, int32_t n_in, int32_t n_h,
    const float *w_skip, float b_skip,
    const float *w1, const float *b1, const float *w2, float *out_scores
) {
    float scratch_h[MCU_MAX_HIDDEN];
    if (n_h > MCU_MAX_HIDDEN) {
        for (int32_t row = 0; row < batch_size; ++row) {
            out_scores[row] = 0.0f;
        }
        return;
    }
    for (int32_t row = 0; row < batch_size; ++row) {
        out_scores[row] = mcu_mlp_1hidden_skip_score(
            X + row * n_in, n_in, n_h, w_skip, b_skip, w1, b1, w2, scratch_h
        );
    }
}

void mcu_mlp_2hidden_batch(
    const float *X, int32_t batch_size, int32_t n_in, int32_t n_h1, int32_t n_h2,
    const float *w1, const float *b1, const float *w2, const float *b2,
    const float *w3, float b3, float *out_scores
) {
    float scratch_h1[MCU_MAX_HIDDEN];
    float scratch_h2[MCU_MAX_HIDDEN];
    if (n_h1 > MCU_MAX_HIDDEN || n_h2 > MCU_MAX_HIDDEN) {
        for (int32_t row = 0; row < batch_size; ++row) {
            out_scores[row] = 0.0f;
        }
        return;
    }
    for (int32_t row = 0; row < batch_size; ++row) {
        out_scores[row] = mcu_mlp_2hidden_score(
            X + row * n_in, n_in, n_h1, n_h2, w1, b1, w2, b2, w3, b3, scratch_h1, scratch_h2
        );
    }
}

void mcu_tree_ensemble_batch(
    const float *X, int32_t batch_size, int32_t n_in, int32_t n_trees,
    const int32_t *tree_offsets, const int16_t *features, const float *thresholds,
    const int16_t *left_children, const int16_t *right_children, const float *leaf_values,
    float base_score, float scale, float *out_scores
) {
    for (int32_t row = 0; row < batch_size; ++row) {
        out_scores[row] = mcu_tree_ensemble_score(
            X + row * n_in, n_trees, tree_offsets, features, thresholds,
            left_children, right_children, leaf_values, base_score, scale
        );
    }
}

float mcu_bench_dense_dot(
    const float *X, int32_t n_samples, int32_t n_in,
    const float *w, float b, int32_t iters
) {
    volatile float sink = 0.0f;
    float acc = 0.0f;
    for (int32_t k = 0; k < iters; ++k) {
        const float *x = X + (k % n_samples) * n_in;
        acc += mcu_dense_dot_score(x, w, b, n_in);
    }
    sink = acc;
    return sink;
}

float mcu_bench_sparse_dot(
    const float *X, int32_t n_samples, int32_t n_in,
    const float *w_nz, const int32_t *idx_nz, float b, int32_t nnz, int32_t iters
) {
    volatile float sink = 0.0f;
    float acc = 0.0f;
    for (int32_t k = 0; k < iters; ++k) {
        const float *x = X + (k % n_samples) * n_in;
        acc += mcu_sparse_dot_score(x, w_nz, idx_nz, b, nnz);
    }
    sink = acc;
    return sink;
}

float mcu_bench_diag_quad(
    const float *X, int32_t n_samples, int32_t n_in,
    const float *a, const float *b_lin, float c, int32_t iters
) {
    volatile float sink = 0.0f;
    float acc = 0.0f;
    for (int32_t k = 0; k < iters; ++k) {
        const float *x = X + (k % n_samples) * n_in;
        acc += mcu_diag_quad_score(x, a, b_lin, c, n_in);
    }
    sink = acc;
    return sink;
}

float mcu_bench_mlp_1hidden(
    const float *X, int32_t n_samples, int32_t n_in, int32_t n_h,
    const float *w1, const float *b1, const float *w2, float b2, int32_t iters
) {
    volatile float sink = 0.0f;
    float scratch_h[MCU_MAX_HIDDEN];
    if (n_h > MCU_MAX_HIDDEN) {
        return 0.0f;
    }
    float acc = 0.0f;
    for (int32_t k = 0; k < iters; ++k) {
        const float *x = X + (k % n_samples) * n_in;
        acc += mcu_mlp_1hidden_score(x, n_in, n_h, w1, b1, w2, b2, scratch_h);
    }
    sink = acc;
    return sink;
}

float mcu_bench_mlp_1hidden_skip(
    const float *X, int32_t n_samples, int32_t n_in, int32_t n_h,
    const float *w_skip, float b_skip,
    const float *w1, const float *b1, const float *w2, int32_t iters
) {
    volatile float sink = 0.0f;
    float scratch_h[MCU_MAX_HIDDEN];
    if (n_h > MCU_MAX_HIDDEN) {
        return 0.0f;
    }
    float acc = 0.0f;
    for (int32_t k = 0; k < iters; ++k) {
        const float *x = X + (k % n_samples) * n_in;
        acc += mcu_mlp_1hidden_skip_score(
            x, n_in, n_h, w_skip, b_skip, w1, b1, w2, scratch_h
        );
    }
    sink = acc;
    return sink;
}

float mcu_bench_mlp_2hidden(
    const float *X, int32_t n_samples, int32_t n_in, int32_t n_h1, int32_t n_h2,
    const float *w1, const float *b1, const float *w2, const float *b2,
    const float *w3, float b3, int32_t iters
) {
    volatile float sink = 0.0f;
    float scratch_h1[MCU_MAX_HIDDEN];
    float scratch_h2[MCU_MAX_HIDDEN];
    if (n_h1 > MCU_MAX_HIDDEN || n_h2 > MCU_MAX_HIDDEN) {
        return 0.0f;
    }
    float acc = 0.0f;
    for (int32_t k = 0; k < iters; ++k) {
        const float *x = X + (k % n_samples) * n_in;
        acc += mcu_mlp_2hidden_score(
            x, n_in, n_h1, n_h2, w1, b1, w2, b2, w3, b3, scratch_h1, scratch_h2
        );
    }
    sink = acc;
    return sink;
}

float mcu_bench_tree_ensemble(
    const float *X, int32_t n_samples, int32_t n_in, int32_t n_trees,
    const int32_t *tree_offsets, const int16_t *features, const float *thresholds,
    const int16_t *left_children, const int16_t *right_children, const float *leaf_values,
    float base_score, float scale, int32_t iters
) {
    volatile float sink = 0.0f;
    float acc = 0.0f;
    for (int32_t k = 0; k < iters; ++k) {
        const float *x = X + (k % n_samples) * n_in;
        acc += mcu_tree_ensemble_score(
            x, n_trees, tree_offsets, features, thresholds,
            left_children, right_children, leaf_values, base_score, scale
        );
    }
    sink = acc;
    return sink;
}
