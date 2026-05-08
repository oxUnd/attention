#include "nn_math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =========================================================================
 * Tensor3D / Matrix lifetime
 * ========================================================================= */

Tensor3D tensor_create(int batch_size, int seq_len, int d_model) {
    Tensor3D t;
    t.batch_size = batch_size;
    t.seq_len = seq_len;
    t.d_model = d_model;
    size_t numel = (size_t)batch_size * (size_t)seq_len * (size_t)d_model;
    t.data = numel > 0 ? (float *)calloc(numel, sizeof(float)) : NULL;
    return t;
}

void tensor_free(Tensor3D *t) {
    if (t && t->data) {
        free(t->data);
        t->data = NULL;
    }
}

void tensor_zero(Tensor3D *t) {
    if (t && t->data) {
        memset(t->data, 0, tensor_numel(t) * sizeof(float));
    }
}

Tensor3D tensor_clone(const Tensor3D *src) {
    Tensor3D out = tensor_create(src->batch_size, src->seq_len, src->d_model);
    if (out.data && src->data) {
        memcpy(out.data, src->data, tensor_numel(src) * sizeof(float));
    }
    return out;
}

size_t tensor_numel(const Tensor3D *t) {
    return (size_t)t->batch_size * (size_t)t->seq_len * (size_t)t->d_model;
}

void tensor_print(const Tensor3D *t, const char *name) {
    printf("%s [batch=%d, seq=%d, d_model=%d]:\n", name, t->batch_size, t->seq_len, t->d_model);
    for (int b = 0; b < t->batch_size && b < 2; b++) {
        printf("  batch %d:\n", b);
        for (int s = 0; s < t->seq_len && s < 5; s++) {
            printf("    seq %d: [", s);
            for (int d = 0; d < t->d_model && d < 8; d++) {
                printf("%.3f ", t->data[b * t->seq_len * t->d_model + s * t->d_model + d]);
            }
            printf("]\n");
        }
    }
}

Matrix matrix_create(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    size_t numel = (size_t)rows * (size_t)cols;
    m.data = numel > 0 ? (float *)calloc(numel, sizeof(float)) : NULL;
    return m;
}

void matrix_free(Matrix *m) {
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
    }
}

void matrix_zero(Matrix *m) {
    if (m && m->data) {
        memset(m->data, 0, matrix_numel(m) * sizeof(float));
    }
}

size_t matrix_numel(const Matrix *m) {
    return (size_t)m->rows * (size_t)m->cols;
}

/* =========================================================================
 * Initialization
 * ========================================================================= */

static float urand_unit(void) {
    return (float)rand() / (float)RAND_MAX;
}

void matrix_init_uniform_small(Matrix *m) {
    size_t n = matrix_numel(m);
    for (size_t i = 0; i < n; i++) {
        m->data[i] = (urand_unit() - 0.5f) * 0.1f;
    }
}

void matrix_init_xavier(Matrix *m) {
    float scale = sqrtf(2.0f / (float)(m->rows + m->cols));
    size_t n = matrix_numel(m);
    for (size_t i = 0; i < n; i++) {
        m->data[i] = (urand_unit() * 2.0f - 1.0f) * scale;
    }
}

void matrix_init_constant(Matrix *m, float value) {
    size_t n = matrix_numel(m);
    for (size_t i = 0; i < n; i++) {
        m->data[i] = value;
    }
}

/* =========================================================================
 * Mask helpers
 * ========================================================================= */

Tensor3D mask_causal_create(int seq_len) {
    Tensor3D mask = tensor_create(1, seq_len, seq_len);
    for (int q_pos = 0; q_pos < seq_len; q_pos++) {
        for (int k_pos = 0; k_pos < seq_len; k_pos++) {
            mask.data[q_pos * seq_len + k_pos] = (k_pos <= q_pos) ? 1.0f : 0.0f;
        }
    }
    return mask;
}

Tensor3D mask_padding_create(int batch_size, int seq_len, const int *valid_lens) {
    Tensor3D mask = tensor_create(batch_size, 1, seq_len);
    for (int b = 0; b < batch_size; b++) {
        int valid = valid_lens ? valid_lens[b] : seq_len;
        for (int k_pos = 0; k_pos < seq_len; k_pos++) {
            mask.data[b * seq_len + k_pos] = (k_pos < valid) ? 1.0f : 0.0f;
        }
    }
    return mask;
}

/* =========================================================================
 * Activation functions
 * ========================================================================= */

static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

static float gelu_f(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / (float)M_PI) * (x + 0.044715f * x * x * x)));
}

float activation_forward(ActivationKind kind, float x) {
    switch (kind) {
        case ACT_RELU: return x > 0 ? x : 0.0f;
        case ACT_GELU: return gelu_f(x);
        case ACT_SILU: return x * sigmoid_f(x);
        case ACT_IDENTITY:
        default:       return x;
    }
}

float activation_grad_at_pre(ActivationKind kind, float x) {
    switch (kind) {
        case ACT_RELU: return x > 0 ? 1.0f : 0.0f;
        case ACT_GELU: {
            float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / (float)M_PI) * (x + 0.044715f * x * x * x)));
            float pdf = expf(-0.5f * x * x) / sqrtf(2.0f * (float)M_PI);
            return x * pdf + cdf;
        }
        case ACT_SILU: {
            float s = sigmoid_f(x);
            return s * (1.0f + x * (1.0f - s));
        }
        case ACT_IDENTITY:
        default:       return 1.0f;
    }
}

/* =========================================================================
 * Linear layer y = x @ W^T
 * ========================================================================= */

void linear_forward(const Matrix *W, const float *x, float *y,
                    int batch_size, int seq_len, int in_dim, int out_dim) {
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            const float *xrow = x + (b * seq_len + s) * in_dim;
            float *yrow = y + (b * seq_len + s) * out_dim;
            for (int out_idx = 0; out_idx < out_dim; out_idx++) {
                const float *wrow = W->data + out_idx * in_dim;
                float acc = 0.0f;
                for (int in_idx = 0; in_idx < in_dim; in_idx++) {
                    acc += wrow[in_idx] * xrow[in_idx];
                }
                yrow[out_idx] = acc;
            }
        }
    }
}

static void linear_backward_impl(const Matrix *W, const float *x, const float *dy,
                                 int batch_size, int seq_len, int in_dim, int out_dim,
                                 Matrix *dW, float *dx, int dx_accumulate) {
    if (dW) {
        /* dW[out, in] += sum_b sum_s dy[b, s, out] * x[b, s, in] */
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < seq_len; s++) {
                const float *xrow = x + (b * seq_len + s) * in_dim;
                const float *dyrow = dy + (b * seq_len + s) * out_dim;
                for (int out_idx = 0; out_idx < out_dim; out_idx++) {
                    float g = dyrow[out_idx];
                    float *dwrow = dW->data + out_idx * in_dim;
                    for (int in_idx = 0; in_idx < in_dim; in_idx++) {
                        dwrow[in_idx] += g * xrow[in_idx];
                    }
                }
            }
        }
    }
    if (dx) {
        /* dx[b, s, in] = sum_out W[out, in] * dy[b, s, out]  (or +=) */
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < seq_len; s++) {
                const float *dyrow = dy + (b * seq_len + s) * out_dim;
                float *dxrow = dx + (b * seq_len + s) * in_dim;
                for (int in_idx = 0; in_idx < in_dim; in_idx++) {
                    float acc = 0.0f;
                    for (int out_idx = 0; out_idx < out_dim; out_idx++) {
                        acc += W->data[out_idx * in_dim + in_idx] * dyrow[out_idx];
                    }
                    if (dx_accumulate) {
                        dxrow[in_idx] += acc;
                    } else {
                        dxrow[in_idx] = acc;
                    }
                }
            }
        }
    }
}

void linear_backward(const Matrix *W, const float *x, const float *dy,
                     int batch_size, int seq_len, int in_dim, int out_dim,
                     Matrix *dW, float *dx) {
    linear_backward_impl(W, x, dy, batch_size, seq_len, in_dim, out_dim, dW, dx, /*accum=*/0);
}

void linear_backward_accum(const Matrix *W, const float *x, const float *dy,
                           int batch_size, int seq_len, int in_dim, int out_dim,
                           Matrix *dW, float *dx) {
    linear_backward_impl(W, x, dy, batch_size, seq_len, in_dim, out_dim, dW, dx, /*accum=*/1);
}

void tensor_linear_forward(const Tensor3D *x, const Matrix *W, Tensor3D *y) {
    linear_forward(W, x->data, y->data, x->batch_size, x->seq_len, x->d_model, W->cols);
}

/* =========================================================================
 * Softmax
 * ========================================================================= */

void tensor_softmax_inplace(Tensor3D *t) {
    int batch_size = t->batch_size;
    int seq_len = t->seq_len;
    int d_model = t->d_model;
    int rows = batch_size * seq_len;

    for (int row = 0; row < rows; row++) {
        float *vec = t->data + row * d_model;
        float max_val = vec[0];
        for (int d = 1; d < d_model; d++) {
            if (vec[d] > max_val) max_val = vec[d];
        }
        float sum = 0.0f;
        for (int d = 0; d < d_model; d++) {
            vec[d] = expf(vec[d] - max_val);
            sum += vec[d];
        }
        if (sum <= 0.0f) sum = 1e-12f;
        float inv = 1.0f / sum;
        for (int d = 0; d < d_model; d++) vec[d] *= inv;
    }
}

void tensor_softmax_backward(const Tensor3D *softmax_output,
                             const Tensor3D *grad_output,
                             Tensor3D *grad_input) {
    int batch_size = softmax_output->batch_size;
    int seq_len = softmax_output->seq_len;
    int d_model = softmax_output->d_model;
    tensor_zero(grad_input);

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            const float *y = softmax_output->data + (b * seq_len + s) * d_model;
            const float *gy = grad_output->data + (b * seq_len + s) * d_model;
            float *gx = grad_input->data + (b * seq_len + s) * d_model;
            float dot = 0.0f;
            for (int d = 0; d < d_model; d++) dot += gy[d] * y[d];
            for (int d = 0; d < d_model; d++) gx[d] = y[d] * (gy[d] - dot);
        }
    }
}

/* =========================================================================
 * Layer Norm
 * ========================================================================= */

void tensor_layer_norm_inplace(Tensor3D *x, const Matrix *gamma, const Matrix *beta, float eps) {
    int batch_size = x->batch_size;
    int seq_len = x->seq_len;
    int d_model = x->d_model;

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            float *vec = x->data + (b * seq_len + s) * d_model;
            float mean = 0.0f;
            for (int d = 0; d < d_model; d++) mean += vec[d];
            mean /= (float)d_model;

            float var = 0.0f;
            for (int d = 0; d < d_model; d++) {
                float diff = vec[d] - mean;
                var += diff * diff;
            }
            var /= (float)d_model;
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int d = 0; d < d_model; d++) {
                float xhat = (vec[d] - mean) * inv_std;
                vec[d] = xhat * gamma->data[d] + beta->data[d];
            }
        }
    }
}

void tensor_layer_norm_backward(const Tensor3D *x_pre,
                                const Tensor3D *grad_output,
                                const Matrix *gamma,
                                Tensor3D *grad_input,
                                float eps,
                                Matrix *grad_gamma,
                                Matrix *grad_beta) {
    int batch_size = x_pre->batch_size;
    int seq_len = x_pre->seq_len;
    int d_model = x_pre->d_model;
    tensor_zero(grad_input);

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            const float *xrow = x_pre->data + (b * seq_len + s) * d_model;
            const float *gyrow = grad_output->data + (b * seq_len + s) * d_model;
            float *gxrow = grad_input->data + (b * seq_len + s) * d_model;

            float mean = 0.0f, var = 0.0f;
            for (int d = 0; d < d_model; d++) mean += xrow[d];
            mean /= (float)d_model;
            for (int d = 0; d < d_model; d++) {
                float diff = xrow[d] - mean;
                var += diff * diff;
            }
            var /= (float)d_model;

            float inv_std = 1.0f / sqrtf(var + eps);

            float dxhat_sum = 0.0f;
            float dxhat_xhat_sum = 0.0f;
            for (int d = 0; d < d_model; d++) {
                float xhat = (xrow[d] - mean) * inv_std;
                float dxhat = gyrow[d] * gamma->data[d];
                dxhat_sum += dxhat;
                dxhat_xhat_sum += dxhat * xhat;
                if (grad_gamma) grad_gamma->data[d] += gyrow[d] * xhat;
                if (grad_beta)  grad_beta->data[d]  += gyrow[d];
            }

            for (int d = 0; d < d_model; d++) {
                float dxhat = gyrow[d] * gamma->data[d];
                gxrow[d] = inv_std * (
                    dxhat
                    - dxhat_sum / (float)d_model
                    - (xrow[d] - mean) * dxhat_xhat_sum / ((float)d_model * (var + eps))
                );
            }
        }
    }
}

/* =========================================================================
 * Dropout
 * ========================================================================= */

void tensor_dropout_inplace(Tensor3D *t, float p, int training) {
    if (!training || p <= 0.0f) return;
    float scale = 1.0f / (1.0f - p);
    size_t n = tensor_numel(t);
    for (size_t i = 0; i < n; i++) {
        if (urand_unit() < p) {
            t->data[i] = 0.0f;
        } else {
            t->data[i] *= scale;
        }
    }
}

void matrix_dropout_inplace(Matrix *m, float p, int training) {
    if (!training || p <= 0.0f) return;
    float scale = 1.0f / (1.0f - p);
    size_t n = matrix_numel(m);
    for (size_t i = 0; i < n; i++) {
        if (urand_unit() < p) {
            m->data[i] = 0.0f;
        } else {
            m->data[i] *= scale;
        }
    }
}

/* =========================================================================
 * Adam
 * ========================================================================= */

AdamOptimizer adam_create(float learning_rate, float beta1, float beta2, float eps) {
    AdamOptimizer opt;
    opt.learning_rate = learning_rate;
    opt.beta1 = beta1;
    opt.beta2 = beta2;
    opt.eps = eps;
    opt.step = 0;
    return opt;
}

float adam_corrected_lr(const AdamOptimizer *opt) {
    if (opt->step <= 0) return opt->learning_rate;
    return opt->learning_rate
         * sqrtf(1.0f - powf(opt->beta2, (float)opt->step))
         / (1.0f - powf(opt->beta1, (float)opt->step));
}

void adam_apply_matrix(Matrix *param, const Matrix *grad,
                       Matrix *m_state, Matrix *v_state,
                       const AdamOptimizer *opt, float lr_t) {
    size_t n = matrix_numel(param);
    for (size_t i = 0; i < n; i++) {
        float g = grad->data[i];
        m_state->data[i] = opt->beta1 * m_state->data[i] + (1.0f - opt->beta1) * g;
        v_state->data[i] = opt->beta2 * v_state->data[i] + (1.0f - opt->beta2) * g * g;
        param->data[i] -= lr_t * m_state->data[i] / (sqrtf(v_state->data[i]) + opt->eps);
    }
}

/* =========================================================================
 * Loss functions
 * ========================================================================= */

LossResult cross_entropy_loss(const Tensor3D *logits, const int *targets, int vocab_size) {
    LossResult result;
    int batch_size = logits->batch_size;
    int seq_len = logits->seq_len;
    int d_model = logits->d_model;
    int vsize = (vocab_size > 0 && vocab_size < d_model) ? vocab_size : d_model;
    float scale = 1.0f / (float)(batch_size * seq_len);

    result.loss = 0.0f;
    result.grad = tensor_create(batch_size, seq_len, d_model);
    tensor_zero(&result.grad);

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            const float *logit_row = logits->data + (b * seq_len + s) * d_model;
            float *grad_row = result.grad.data + (b * seq_len + s) * d_model;

            float max_val = logit_row[0];
            for (int v = 1; v < vsize; v++) {
                if (logit_row[v] > max_val) max_val = logit_row[v];
            }
            float sum_exp = 0.0f;
            for (int v = 0; v < vsize; v++) {
                sum_exp += expf(logit_row[v] - max_val);
            }
            int target = targets[b * seq_len + s];
            for (int v = 0; v < vsize; v++) {
                float prob = expf(logit_row[v] - max_val) / sum_exp;
                grad_row[v] = prob * scale;
                if (v == target) {
                    grad_row[v] -= scale;
                    result.loss -= logf(prob + 1e-8f);
                }
            }
        }
    }
    result.loss *= scale;
    return result;
}

LossResult mse_loss(const Tensor3D *pred, const Tensor3D *target) {
    LossResult result;
    size_t n = tensor_numel(pred);
    result.loss = 0.0f;
    result.grad = tensor_create(pred->batch_size, pred->seq_len, pred->d_model);
    float inv = 1.0f / (float)n;
    for (size_t i = 0; i < n; i++) {
        float diff = pred->data[i] - target->data[i];
        result.loss += diff * diff;
        result.grad.data[i] = 2.0f * diff * inv;
    }
    result.loss *= inv;
    return result;
}

void loss_print(const LossResult *loss, const char *name) {
    printf("Loss %s: %.6f\n", name, loss->loss);
}
