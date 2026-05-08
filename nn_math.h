#ifndef NN_MATH_H
#define NN_MATH_H

#include <stddef.h>
#include <stdint.h>

/* =========================================================================
 * Data structures
 * =========================================================================
 *
 * Tensor3D / Matrix are plain row-major float buffers.
 *
 * Tensor3D layout: [batch_size, seq_len, d_model]
 *   element (b, s, d) = data[b * seq_len * d_model + s * d_model + d]
 *
 * Matrix layout: [rows, cols]
 *   element (r, c) = data[r * cols + c]
 *
 * Throughout this file we write linear layers as row-vector products:
 *     y[..., out] = sum_in W[out, in] * x[..., in]
 * so a Matrix `W` of shape (out_dim, in_dim) maps an `in_dim` row-vector
 * to an `out_dim` row-vector.
 * ========================================================================= */

typedef struct {
    int batch_size;
    int seq_len;
    int d_model;
    float *data;
} Tensor3D;

typedef struct {
    int rows;
    int cols;
    float *data;
} Matrix;

typedef enum {
    ACT_IDENTITY = 0,
    ACT_RELU = 1,
    ACT_GELU = 2,
    ACT_SILU = 3,
} ActivationKind;

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float eps;
    int step; /* 1-indexed Adam time-step counter */
    /* Optional linear-warmup + cosine-decay schedule.  When warmup_steps == 0
     * and decay_steps == 0 the schedule is disabled (factor == 1 always),
     * preserving legacy behaviour.  When enabled the effective learning rate
     * is `learning_rate * factor(step)`, where:
     *   step <= warmup_steps :   factor = step / warmup_steps
     *   step <= warmup+decay :   factor = 0.5 * (1 + cos(pi * t)),
     *                            t = (step - warmup) / decay   in [0, 1]
     *   step >  warmup+decay :   factor = min_factor
     */
    int warmup_steps;
    int decay_steps;
    float min_factor;
} AdamOptimizer;

typedef struct {
    float loss;
    Tensor3D grad; /* dL/dlogits, same shape as the logits */
} LossResult;

/* =========================================================================
 * Tensor3D / Matrix lifetime + helpers
 * ========================================================================= */

Tensor3D tensor_create(int batch_size, int seq_len, int d_model);
void tensor_free(Tensor3D *t);
void tensor_zero(Tensor3D *t);
Tensor3D tensor_clone(const Tensor3D *src);
size_t tensor_numel(const Tensor3D *t);
void tensor_print(const Tensor3D *t, const char *name);

Matrix matrix_create(int rows, int cols);
void matrix_free(Matrix *m);
void matrix_zero(Matrix *m);
size_t matrix_numel(const Matrix *m);

/* =========================================================================
 * Initialization
 * ========================================================================= */

/* Uniform U(-0.05, 0.05); used for biases. */
void matrix_init_uniform_small(Matrix *m);

/* Xavier / Glorot uniform: U(-scale, scale), scale = sqrt(2 / (rows + cols)). */
void matrix_init_xavier(Matrix *m);

/* Fill every element with `value`. */
void matrix_init_constant(Matrix *m, float value);

/* =========================================================================
 * Mask helpers
 *
 * Causal mask shape: (1, seq_len, seq_len)  — broadcast over batch.
 *     mask[q_pos][k_pos] = 1 if k_pos <= q_pos else 0
 *
 * Padding mask shape: (batch_size, 1, seq_len)
 *     mask[b][k_pos] = 1 if k_pos < valid_lens[b] else 0
 * ========================================================================= */

Tensor3D mask_causal_create(int seq_len);
Tensor3D mask_padding_create(int batch_size, int seq_len, const int *valid_lens);

/* =========================================================================
 * Activation functions (element-wise)
 * ========================================================================= */

float activation_forward(ActivationKind kind, float x);
float activation_grad_at_pre(ActivationKind kind, float pre_activation_x);

/* =========================================================================
 * Linear layer y = x @ W^T  (W has shape [out_dim, in_dim])
 *
 * forward:    y[b, s, out] = sum_in W[out, in] * x[b, s, in]
 * backward:   given dy, accumulate dW (optional) and write/accumulate dx.
 *
 * Note: linear_backward writes dx (overwrites). Use linear_backward_accum
 *       if you need to accumulate into an existing buffer.
 * ========================================================================= */

void linear_forward(const Matrix *W, const float *x, float *y,
                    int batch_size, int seq_len, int in_dim, int out_dim);

void linear_backward(const Matrix *W, const float *x, const float *dy,
                     int batch_size, int seq_len, int in_dim, int out_dim,
                     Matrix *dW, float *dx);

void linear_backward_accum(const Matrix *W, const float *x, const float *dy,
                           int batch_size, int seq_len, int in_dim, int out_dim,
                           Matrix *dW, float *dx);

/* Convenience: y = x @ W^T using Tensor3D shapes (in_dim = x->d_model,
 * out_dim = W->cols, output shape = [x->batch_size, x->seq_len, W->cols]). */
void tensor_linear_forward(const Tensor3D *x, const Matrix *W, Tensor3D *y);

/* =========================================================================
 * Softmax (along the last axis)
 * ========================================================================= */

void tensor_softmax_inplace(Tensor3D *t);

/* Backprop through softmax given y = softmax(x):
 *     dx[i] = y[i] * (dy[i] - sum_j dy[j] * y[j])
 */
void tensor_softmax_backward(const Tensor3D *softmax_output,
                             const Tensor3D *grad_output,
                             Tensor3D *grad_input);

/* =========================================================================
 * Layer Normalisation (per-token, across d_model)
 * ========================================================================= */

void tensor_layer_norm_inplace(Tensor3D *x, const Matrix *gamma, const Matrix *beta, float eps);

/* Backward pass.  Optionally accumulates into `grad_gamma` / `grad_beta`. */
void tensor_layer_norm_backward(const Tensor3D *x_pre,
                                const Tensor3D *grad_output,
                                const Matrix *gamma,
                                Tensor3D *grad_input,
                                float eps,
                                Matrix *grad_gamma,
                                Matrix *grad_beta);

/* =========================================================================
 * Dropout (training-only; inference is a no-op).  Inverted-scale variant.
 * ========================================================================= */

void tensor_dropout_inplace(Tensor3D *t, float p, int training);
void matrix_dropout_inplace(Matrix *m, float p, int training);

/* =========================================================================
 * Adam optimiser
 *
 *   m_state, v_state must have the same shape as `param`.
 *
 *   The caller is responsible for advancing `opt->step` once per logical
 *   optimisation step.  `adam_apply_matrix` reads (but does not modify)
 *   `opt->step` and uses it to compute the bias-corrected learning rate.
 * ========================================================================= */

AdamOptimizer adam_create(float learning_rate, float beta1, float beta2, float eps);

/* Configure linear-warmup + cosine-decay schedule. Pass warmup_steps=0 and
 * decay_steps=0 to disable (default after adam_create). `min_factor` is the
 * floor reached after warmup+decay steps (e.g. 0.1 for a 10x decay). */
void adam_set_schedule(AdamOptimizer *opt, int warmup_steps, int decay_steps,
                       float min_factor);

/* Returns the schedule multiplier in [min_factor, 1] for the current step. */
float adam_schedule_factor(const AdamOptimizer *opt);

/* lr_t = lr * sqrt(1 - beta2^step) / (1 - beta1^step) * schedule_factor */
float adam_corrected_lr(const AdamOptimizer *opt);

void adam_apply_matrix(Matrix *param, const Matrix *grad,
                       Matrix *m_state, Matrix *v_state,
                       const AdamOptimizer *opt, float lr_t);

/* =========================================================================
 * Loss functions
 *
 * cross_entropy_loss applies a softmax along the last axis using only the
 * first `vocab_size` columns of `logits` as class scores.  Useful when
 * logits are kept in d_model space but vocab_size <= d_model.
 * ========================================================================= */

LossResult cross_entropy_loss(const Tensor3D *logits, const int *targets, int vocab_size);
LossResult mse_loss(const Tensor3D *pred, const Tensor3D *target);
void loss_print(const LossResult *loss, const char *name);

#endif /* NN_MATH_H */
