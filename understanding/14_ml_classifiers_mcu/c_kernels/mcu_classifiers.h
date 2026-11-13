#ifndef MCU_CLASSIFIERS_H
#define MCU_CLASSIFIERS_H

#include <stdint.h>

#ifdef _WIN32
  #ifdef MCU_CLASSIFIERS_EXPORTS
    #define MCU_API __declspec(dllexport)
  #else
    #define MCU_API __declspec(dllimport)
  #endif
#else
  #define MCU_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Dense linear dot product: b + sum_{i=0}^{n-1} w[i] * x[i]. */
MCU_API float mcu_dense_dot_score(
    const float *x,
    const float *w,
    float b,
    int32_t n
);

/* Sparse indexed dot product (L1 feature selection): b + sum_{k=0}^{nnz-1} w_nz[k] * x[idx_nz[k]]. */
MCU_API float mcu_sparse_dot_score(
    const float *x,
    const float *w_nz,
    const int32_t *idx_nz,
    float b,
    int32_t nnz
);

/* Diagonal quadratic log-likelihood ratio (GaussianNB): c + sum_{i=0}^{n-1} x[i] * (a[i]*x[i] + b_lin[i]). */
MCU_API float mcu_diag_quad_score(
    const float *x,
    const float *a,
    const float *b_lin,
    float c,
    int32_t n
);

/* Scaler-folded 1-hidden-layer ReLU MLP: caller passes small stack scratch_h[n_h]. */
MCU_API float mcu_mlp_1hidden_score(
    const float *x,
    int32_t n_in,
    int32_t n_h,
    const float *w1,
    const float *b1,
    const float *w2,
    float b2,
    float *scratch_h
);

/* Scaler-folded 1-hidden-layer ReLU MLP with direct linear skip connection. */
MCU_API float mcu_mlp_1hidden_skip_score(
    const float *x,
    int32_t n_in,
    int32_t n_h,
    const float *w_skip,
    float b_skip,
    const float *w1,
    const float *b1,
    const float *w2,
    float *scratch_h
);

/* Scaler-folded 2-hidden-layer ReLU MLP: caller passes stack scratch_h1[n_h1], scratch_h2[n_h2]. */
MCU_API float mcu_mlp_2hidden_score(
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
);

/* Flat-table binary tree / boosted tree ensemble evaluator (0 multiplications inside traversal). */
MCU_API float mcu_tree_ensemble_score(
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
);

/* Batch evaluation helpers (stride n_in) for verifying exact vector parity vs NumPy/PyTorch. */
MCU_API void mcu_dense_dot_batch(
    const float *X, int32_t batch_size, int32_t n_in,
    const float *w, float b, float *out_scores
);

MCU_API void mcu_sparse_dot_batch(
    const float *X, int32_t batch_size, int32_t n_in,
    const float *w_nz, const int32_t *idx_nz, float b, int32_t nnz, float *out_scores
);

MCU_API void mcu_diag_quad_batch(
    const float *X, int32_t batch_size, int32_t n_in,
    const float *a, const float *b_lin, float c, float *out_scores
);

MCU_API void mcu_mlp_1hidden_batch(
    const float *X, int32_t batch_size, int32_t n_in, int32_t n_h,
    const float *w1, const float *b1, const float *w2, float b2, float *out_scores
);

MCU_API void mcu_mlp_1hidden_skip_batch(
    const float *X, int32_t batch_size, int32_t n_in, int32_t n_h,
    const float *w_skip, float b_skip,
    const float *w1, const float *b1, const float *w2, float *out_scores
);

MCU_API void mcu_mlp_2hidden_batch(
    const float *X, int32_t batch_size, int32_t n_in, int32_t n_h1, int32_t n_h2,
    const float *w1, const float *b1, const float *w2, const float *b2,
    const float *w3, float b3, float *out_scores
);

MCU_API void mcu_tree_ensemble_batch(
    const float *X, int32_t batch_size, int32_t n_in, int32_t n_trees,
    const int32_t *tree_offsets, const int16_t *features, const float *thresholds,
    const int16_t *left_children, const int16_t *right_children, const float *leaf_values,
    float base_score, float scale, float *out_scores
);

/* Inner-loop C99 benchmark helpers returning volatile checksum to defeat dead-code elimination. */
MCU_API float mcu_bench_dense_dot(
    const float *X, int32_t n_samples, int32_t n_in,
    const float *w, float b, int32_t iters
);

MCU_API float mcu_bench_sparse_dot(
    const float *X, int32_t n_samples, int32_t n_in,
    const float *w_nz, const int32_t *idx_nz, float b, int32_t nnz, int32_t iters
);

MCU_API float mcu_bench_diag_quad(
    const float *X, int32_t n_samples, int32_t n_in,
    const float *a, const float *b_lin, float c, int32_t iters
);

MCU_API float mcu_bench_mlp_1hidden(
    const float *X, int32_t n_samples, int32_t n_in, int32_t n_h,
    const float *w1, const float *b1, const float *w2, float b2, int32_t iters
);

MCU_API float mcu_bench_mlp_1hidden_skip(
    const float *X, int32_t n_samples, int32_t n_in, int32_t n_h,
    const float *w_skip, float b_skip,
    const float *w1, const float *b1, const float *w2, int32_t iters
);

MCU_API float mcu_bench_mlp_2hidden(
    const float *X, int32_t n_samples, int32_t n_in, int32_t n_h1, int32_t n_h2,
    const float *w1, const float *b1, const float *w2, const float *b2,
    const float *w3, float b3, int32_t iters
);

MCU_API float mcu_bench_tree_ensemble(
    const float *X, int32_t n_samples, int32_t n_in, int32_t n_trees,
    const int32_t *tree_offsets, const int16_t *features, const float *thresholds,
    const int16_t *left_children, const int16_t *right_children, const float *leaf_values,
    float base_score, float scale, int32_t iters
);

#ifdef __cplusplus
}
#endif

#endif
