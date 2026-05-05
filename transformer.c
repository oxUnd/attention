#include "transformer.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define PI 3.14159265358979323846

static float relu(float x) {
    return x > 0 ? x : 0;
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / PI) * (x + 0.044715f * x * x * x)));
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float silu(float x) {
    return x * sigmoid(x);
}

static float act_forward(Activation act, float x) {
    switch (act) {
        case RELU: return relu(x);
        case GELU: return gelu(x);
        case SILU: return silu(x);
        case DIM_LAYER:
        case GEOF:
        default:
            return x;
    }
}

static float act_backward(Activation act, float x) {
    switch (act) {
        case RELU: return x > 0 ? 1.0f : 0.0f;
        case GELU: {
            float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / PI) * (x + 0.044715f * x * x * x)));
            float pdf = expf(-0.5f * x * x) / sqrtf(2.0f * PI);
            return x * pdf + cdf;
        }
        case SILU: {
            float s = sigmoid(x);
            return s * (1.0f + x * (1.0f - s));
        }
        case DIM_LAYER:
        case GEOF:
        default:
            return 1.0f;
    }
}

void tensor_zero(Tensor3D *t) {
    memset(t->data, 0, t->batch_size * t->seq_len * t->d_model * sizeof(float));
}

Tensor3D tensor_create(int batch_size, int seq_len, int d_model) {
    Tensor3D t;
    t.batch_size = batch_size;
    t.seq_len = seq_len;
    t.d_model = d_model;
    t.data = (float *)calloc(batch_size * seq_len * d_model, sizeof(float));
    return t;
}

void tensor_free(Tensor3D *t) {
    if (t->data) free(t->data);
    t->data = NULL;
}

Matrix matrix_create(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float *)calloc(rows * cols, sizeof(float));
    return m;
}

void matrix_free(Matrix *m) {
    if (m->data) free(m->data);
    m->data = NULL;
}

void matrix_zero(Matrix *m) {
    memset(m->data, 0, m->rows * m->cols * sizeof(float));
}

void matrix_rand(Matrix *m) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
}

void matrix_xavier(Matrix *m) {
    float scale = sqrtf(2.0f / (m->rows + m->cols));
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
    }
}

void tensor_softmax(Tensor3D *t, int axis) {
    int b = t->batch_size;
    int s_orig = t->seq_len;
    int d = t->d_model;

    if (axis == 2) {
        for (int i = 0; i < b * s_orig; i++) {
            float max_val = -1e9f;
            for (int j = 0; j < d; j++) {
                int idx = i * d + j;
                if (t->data[idx] > max_val) max_val = t->data[idx];
            }
            float sum = 0;
            for (int j = 0; j < d; j++) {
                int idx = i * d + j;
                sum += expf(t->data[idx] - max_val);
            }
            for (int j = 0; j < d; j++) {
                int idx = i * d + j;
                t->data[idx] = expf(t->data[idx] - max_val) / sum;
            }
        }
    }
}

void tensor_layer_norm(Tensor3D *x, Matrix *gamma, Matrix *beta, float eps) {
    int b = x->batch_size;
    int s = x->seq_len;
    int d = x->d_model;

    for (int i = 0; i < b; i++) {
        for (int j = 0; j < s; j++) {
            float mean = 0;
            for (int k = 0; k < d; k++) {
                mean += x->data[i * s * d + j * d + k];
            }
            mean /= d;

            float var = 0;
            for (int k = 0; k < d; k++) {
                float diff = x->data[i * s * d + j * d + k] - mean;
                var += diff * diff;
            }
            var /= d;

            for (int k = 0; k < d; k++) {
                float xhat = (x->data[i * s * d + j * d + k] - mean) / sqrtf(var + eps);
                x->data[i * s * d + j * d + k] = xhat * gamma->data[k] + beta->data[k];
            }
        }
    }
}

void tensor_dropout(Tensor3D *t, float dropout_prob, int training) {
    if (!training || dropout_prob <= 0.0f) return;
    float scale = 1.0f / (1.0f - dropout_prob);
    int size = t->batch_size * t->seq_len * t->d_model;
    for (int i = 0; i < size; i++) {
        if ((float)rand() / RAND_MAX < dropout_prob) {
            t->data[i] = 0;
        } else {
            t->data[i] *= scale;
        }
    }
}

void matrix_dropout(Matrix *m, int rows, int cols, float dropout_prob, int training) {
    if (!training || dropout_prob <= 0.0f) return;
    float scale = 1.0f / (1.0f - dropout_prob);
    int size = rows * cols;
    for (int i = 0; i < size; i++) {
        if ((float)rand() / RAND_MAX < dropout_prob) {
            m->data[i] = 0;
        } else {
            m->data[i] *= scale;
        }
    }
}

MultiHeadAttention *multi_head_attn_create(int nhead, int d_model, float dropout) {
    MultiHeadAttention *attn = (MultiHeadAttention *)malloc(sizeof(MultiHeadAttention));
    attn->nhead = nhead;
    attn->d_k = d_model / nhead;
    attn->dropout = dropout;

    attn->wq = (Matrix *)malloc(sizeof(Matrix));
    *attn->wq = matrix_create(d_model, d_model);
    matrix_xavier(attn->wq);

    attn->wk = (Matrix *)malloc(sizeof(Matrix));
    *attn->wk = matrix_create(d_model, d_model);
    matrix_xavier(attn->wk);

    attn->wv = (Matrix *)malloc(sizeof(Matrix));
    *attn->wv = matrix_create(d_model, d_model);
    matrix_xavier(attn->wv);

    attn->wq_out = (Matrix *)malloc(sizeof(Matrix));
    *attn->wq_out = matrix_create(d_model, d_model);
    matrix_xavier(attn->wq_out);

    return attn;
}

void multi_head_attn_free(MultiHeadAttention *attn) {
    matrix_free(attn->wq);
    matrix_free(attn->wk);
    matrix_free(attn->wv);
    matrix_free(attn->wq_out);
    free(attn->wq);
    free(attn->wk);
    free(attn->wv);
    free(attn->wq_out);
    free(attn);
}

static void mat_vec_mul(Matrix *w, float *in, float *out, int batch, int seq, int d_in, int d_out) {
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < seq; i++) {
            for (int o = 0; o < d_out; o++) {
                float sum = 0;
                for (int j = 0; j < d_in; j++) {
                    sum += w->data[o * d_in + j] * in[b * seq * d_in + i * d_in + j];
                }
                out[b * seq * d_out + i * d_out + o] = sum;
            }
        }
    }
}

void mat_vec_mul_backward(Matrix *w, float *in, float *grad_out, int batch, int seq, int d_in, int d_out, Matrix *grad_w, float *grad_in) {
    if (grad_w) {
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                for (int o = 0; o < d_out; o++) {
                    float g = grad_out[b * seq * d_out + s * d_out + o];
                    for (int j = 0; j < d_in; j++) {
                        grad_w->data[o * d_in + j] += g * in[b * seq * d_in + s * d_in + j];
                    }
                }
            }
        }
    }
    if (grad_in) {
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                for (int j = 0; j < d_in; j++) {
                    float sum = 0;
                    for (int o = 0; o < d_out; o++) {
                        sum += w->data[o * d_in + j] * grad_out[b * seq * d_out + s * d_out + o];
                    }
                    grad_in[b * seq * d_in + s * d_in + j] += sum;
                }
            }
        }
    }
}

static void tensor_mat_mul(Tensor3D *x, Matrix *w, Tensor3D *out) {
    mat_vec_mul(w, x->data, out->data, x->batch_size, x->seq_len, x->d_model, w->cols);
}

Tensor3D multi_head_attn_forward(
    MultiHeadAttention *attn,
    Tensor3D *query,
    Tensor3D *key,
    Tensor3D *value,
    Tensor3D *attn_mask,
    AttnCache *cache
) {
    int b = query->batch_size;
    int s = query->seq_len;
    int d_k = attn->d_k;
    int nhead = attn->nhead;
    int d_model = query->d_model;

    if (cache) {
        cache->input = clone_tensor(query);
        cache->key_input = clone_tensor(key);
        cache->value_input = clone_tensor(value);
    }

    Tensor3D q = tensor_create(b, s, d_model);
    Tensor3D k = tensor_create(b, s, d_model);
    Tensor3D v = tensor_create(b, s, d_model);

    tensor_mat_mul(query, attn->wq, &q);
    tensor_mat_mul(key, attn->wk, &k);
    tensor_mat_mul(value, attn->wv, &v);

    Tensor3D q_reshaped = tensor_create(b, nhead, s * d_k);
    Tensor3D k_reshaped = tensor_create(b, nhead, s * d_k);
    Tensor3D v_reshaped = tensor_create(b, nhead, s * d_k);

    for (int i = 0; i < b; i++) {
        for (int h = 0; h < nhead; h++) {
            for (int j = 0; j < s; j++) {
                for (int dk = 0; dk < d_k; dk++) {
                    q_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + dk] =
                        q.data[i * s * d_model + j * d_model + h * d_k + dk];
                    k_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + dk] =
                        k.data[i * s * d_model + j * d_model + h * d_k + dk];
                    v_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + dk] =
                        v.data[i * s * d_model + j * d_model + h * d_k + dk];
                }
            }
        }
    }

    Tensor3D scores = tensor_create(b, nhead, s * s);
    float scale = sqrtf((float)d_k);

    for (int i = 0; i < b; i++) {
        for (int h = 0; h < nhead; h++) {
            for (int j = 0; j < s; j++) {
                for (int t = 0; t < s; t++) {
                    float dot = 0;
                    for (int kk = 0; kk < d_k; kk++) {
                        dot += q_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + kk] *
                               k_reshaped.data[i * nhead * s * d_k + h * s * d_k + t * d_k + kk];
                    }
                    scores.data[i * nhead * s * s + h * s * s + j * s + t] = dot / scale;
                }
            }
        }
    }

    if (attn_mask && attn_mask->data) {
        for (int i = 0; i < b; i++) {
            for (int h = 0; h < nhead; h++) {
                for (int j = 0; j < s; j++) {
                    for (int t = 0; t < s; t++) {
                        if (attn_mask->data[i * s + t] == 0) {
                            scores.data[i * nhead * s * s + h * s * s + j * s + t] = -1e9f;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < b; i++) {
        for (int h = 0; h < nhead; h++) {
            for (int q = 0; q < s; q++) {
                float max_val = -1e9f;
                for (int kk = 0; kk < s; kk++) {
                    int idx = i * nhead * s * s + h * s * s + q * s + kk;
                    if (scores.data[idx] > max_val) max_val = scores.data[idx];
                }
                float sum = 0;
                for (int kk = 0; kk < s; kk++) {
                    int idx = i * nhead * s * s + h * s * s + q * s + kk;
                    sum += expf(scores.data[idx] - max_val);
                }
                for (int kk = 0; kk < s; kk++) {
                    int idx = i * nhead * s * s + h * s * s + q * s + kk;
                    scores.data[idx] = expf(scores.data[idx] - max_val) / sum;
                }
            }
        }
    }

    if (cache) {
        cache->attn_weights = clone_tensor(&scores);
    }

    Tensor3D attend = tensor_create(b, nhead, s * d_k);

    for (int i = 0; i < b; i++) {
        for (int h = 0; h < nhead; h++) {
            for (int j = 0; j < s; j++) {
                for (int kk = 0; kk < d_k; kk++) {
                    float sum = 0;
                    for (int t = 0; t < s; t++) {
                        sum += scores.data[i * nhead * s * s + h * s * s + j * s + t] *
                               v_reshaped.data[i * nhead * s * d_k + h * s * d_k + t * d_k + kk];
                    }
                    attend.data[i * nhead * s * d_k + h * s * d_k + j * d_k + kk] = sum;
                }
            }
        }
    }

    Tensor3D output = tensor_create(b, s, d_model);

    for (int i = 0; i < b; i++) {
        for (int j = 0; j < s; j++) {
            for (int h = 0; h < nhead; h++) {
                for (int kk = 0; kk < d_k; kk++) {
                    output.data[i * s * d_model + j * d_model + h * d_k + kk] =
                        attend.data[i * nhead * s * d_k + h * s * d_k + j * d_k + kk];
                }
            }
        }
    }

    Tensor3D out = tensor_create(b, s, d_model);
    tensor_mat_mul(&output, attn->wq_out, &out);

    tensor_free(&q);
    tensor_free(&k);
    tensor_free(&v);
    tensor_free(&q_reshaped);
    tensor_free(&k_reshaped);
    tensor_free(&v_reshaped);
    tensor_free(&scores);
    tensor_free(&attend);
    tensor_free(&output);

    return out;
}

FeedForward *feed_forward_create(int d_model, int d_ff, Activation activation) {
    FeedForward *ffn = (FeedForward *)malloc(sizeof(FeedForward));
    ffn->d_model = d_model;
    ffn->d_ff = d_ff;
    ffn->activation = activation;

    ffn->w1 = (Matrix *)malloc(sizeof(Matrix));
    *ffn->w1 = matrix_create(d_model, d_ff);
    matrix_xavier(ffn->w1);

    ffn->b1 = (Matrix *)malloc(sizeof(Matrix));
    *ffn->b1 = matrix_create(1, d_ff);
    matrix_rand(ffn->b1);

    ffn->w2 = (Matrix *)malloc(sizeof(Matrix));
    *ffn->w2 = matrix_create(d_ff, d_model);
    matrix_xavier(ffn->w2);

    ffn->b2 = (Matrix *)malloc(sizeof(Matrix));
    *ffn->b2 = matrix_create(1, d_model);
    matrix_rand(ffn->b2);

    return ffn;
}

void feed_forward_free(FeedForward *ffn) {
    matrix_free(ffn->w1);
    matrix_free(ffn->b1);
    matrix_free(ffn->w2);
    matrix_free(ffn->b2);
    free(ffn->w1);
    free(ffn->b1);
    free(ffn->w2);
    free(ffn->b2);
    free(ffn);
}

Tensor3D feed_forward_forward(FeedForward *ffn, Tensor3D *x, Tensor3D *hidden_cache) {
    int b = x->batch_size;
    int s = x->seq_len;
    int d_model = x->d_model;
    int d_ff = ffn->d_ff;

    Tensor3D hidden = tensor_create(b, s, d_ff);

    for (int bs = 0; bs < b; bs++) {
        for (int seq = 0; seq < s; seq++) {
            for (int j = 0; j < d_ff; j++) {
                float sum = ffn->b1->data[j];
                for (int k = 0; k < d_model; k++) {
                    sum += x->data[bs * s * d_model + seq * d_model + k] * ffn->w1->data[k * d_ff + j];
                }
                if (hidden_cache) {
                    hidden_cache->data[bs * s * d_ff + seq * d_ff + j] = sum;
                }
                hidden.data[bs * s * d_ff + seq * d_ff + j] = act_forward(ffn->activation, sum);
            }
        }
    }

    Tensor3D out = tensor_create(b, s, d_model);

    for (int bs = 0; bs < b; bs++) {
        for (int seq = 0; seq < s; seq++) {
            for (int j = 0; j < d_model; j++) {
                float sum = ffn->b2->data[j];
                for (int k = 0; k < d_ff; k++) {
                    sum += hidden.data[bs * s * d_ff + seq * d_ff + k] * ffn->w2->data[k * d_model + j];
                }
                out.data[bs * s * d_model + seq * d_model + j] = sum;
            }
        }
    }

    tensor_free(&hidden);
    return out;
}

PositionalEncoding *positional_encoding_create(int d_model, int max_len, float dropout_unused) {
    (void)dropout_unused;
    PositionalEncoding *pe = (PositionalEncoding *)malloc(sizeof(PositionalEncoding));
    pe->d_model = d_model;
    pe->max_len = max_len;
    pe->pe = (Matrix *)malloc(sizeof(Matrix));
    *pe->pe = matrix_create(max_len, d_model);

    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i += 2) {
            float angle = pos / powf(10000.0f, (2.0f * i) / d_model);
            pe->pe->data[pos * d_model + i] = sinf(angle);
            if (i + 1 < d_model) {
                pe->pe->data[pos * d_model + i + 1] = cosf(angle);
            }
        }
    }

    return pe;
}

void positional_encoding_free(PositionalEncoding *pe) {
    matrix_free(pe->pe);
    free(pe->pe);
    free(pe);
}

Tensor3D positional_encoding_forward(PositionalEncoding *pe, int seq_len) {
    Tensor3D out = tensor_create(1, seq_len, pe->d_model);

    for (int i = 0; i < seq_len && i < pe->max_len; i++) {
        for (int j = 0; j < pe->d_model; j++) {
            out.data[i * pe->d_model + j] = pe->pe->data[i * pe->d_model + j];
        }
    }

    return out;
}

EncoderLayer encoder_layer_create(int nhead, int d_model, int d_ff, float dropout) {
    EncoderLayer layer;
    layer.self_attn = multi_head_attn_create(nhead, d_model, dropout);
    layer.ffn = feed_forward_create(d_model, d_ff, GELU);
    layer.d_model = d_model;
    layer.ln1_gamma = (Matrix *)malloc(sizeof(Matrix));
    *layer.ln1_gamma = matrix_create(1, d_model);
    layer.ln1_beta = (Matrix *)malloc(sizeof(Matrix));
    *layer.ln1_beta = matrix_create(1, d_model);
    layer.ln2_gamma = (Matrix *)malloc(sizeof(Matrix));
    *layer.ln2_gamma = matrix_create(1, d_model);
    layer.ln2_beta = (Matrix *)malloc(sizeof(Matrix));
    *layer.ln2_beta = matrix_create(1, d_model);
    for (int i = 0; i < d_model; i++) {
        layer.ln1_gamma->data[i] = 1.0f;
        layer.ln1_beta->data[i] = 0.0f;
        layer.ln2_gamma->data[i] = 1.0f;
        layer.ln2_beta->data[i] = 0.0f;
    }
    return layer;
}

void encoder_layer_free(EncoderLayer *layer) {
    multi_head_attn_free(layer->self_attn);
    feed_forward_free(layer->ffn);
    matrix_free(layer->ln1_gamma);
    matrix_free(layer->ln1_beta);
    matrix_free(layer->ln2_gamma);
    matrix_free(layer->ln2_beta);
    free(layer->ln1_gamma);
    free(layer->ln1_beta);
    free(layer->ln2_gamma);
    free(layer->ln2_beta);
}

Tensor3D clone_tensor(Tensor3D *t) {
    Tensor3D c = tensor_create(t->batch_size, t->seq_len, t->d_model);
    memcpy(c.data, t->data, t->batch_size * t->seq_len * t->d_model * sizeof(float));
    return c;
}

Tensor3D encoder_layer_forward(
    EncoderLayer *layer,
    Tensor3D *x,
    Tensor3D *mask,
    EncoderLayerCache *cache
) {
    if (cache) {
        cache->x = clone_tensor(x);
    }

    AttnCache *attn_cache = cache ? &cache->self_attn_cache : NULL;
    Tensor3D attn_out = multi_head_attn_forward(layer->self_attn, x, x, x, mask, attn_cache);

    Tensor3D residual = clone_tensor(x);
    for (int i = 0; i < x->batch_size * x->seq_len * x->d_model; i++) {
        residual.data[i] += attn_out.data[i];
    }
    tensor_free(&attn_out);

    if (cache) {
        cache->residual = clone_tensor(&residual);
    }

    tensor_layer_norm(&residual, layer->ln1_gamma, layer->ln1_beta, 1e-5f);

    if (cache) {
        cache->ln1_out = clone_tensor(&residual);
    }

    Tensor3D *hidden_ptr = cache ? &cache->ffn_hidden : NULL;
    if (cache) {
        cache->ffn_hidden = tensor_create(x->batch_size, x->seq_len, layer->ffn->d_ff);
    }
    Tensor3D ff_out = feed_forward_forward(layer->ffn, &residual, hidden_ptr);

    for (int i = 0; i < residual.batch_size * residual.seq_len * residual.d_model; i++) {
        ff_out.data[i] += residual.data[i];
    }

    if (cache) {
        cache->ff_out = clone_tensor(&ff_out);
    }

    tensor_free(&residual);
    tensor_layer_norm(&ff_out, layer->ln2_gamma, layer->ln2_beta, 1e-5f);

    if (cache) {
        cache->ln2_out = clone_tensor(&ff_out);
    }

    return ff_out;
}

DecoderLayer decoder_layer_create(int nhead, int d_model, int d_ff, float dropout) {
    DecoderLayer layer;
    layer.self_attn = multi_head_attn_create(nhead, d_model, dropout);
    layer.cross_attn = multi_head_attn_create(nhead, d_model, dropout);
    layer.ffn = feed_forward_create(d_model, d_ff, GELU);
    layer.d_model = d_model;
    layer.d_ff = d_ff;
    layer.ln1_gamma = (Matrix *)malloc(sizeof(Matrix));
    *layer.ln1_gamma = matrix_create(1, d_model);
    layer.ln1_beta = (Matrix *)malloc(sizeof(Matrix));
    *layer.ln1_beta = matrix_create(1, d_model);
    layer.ln2_gamma = (Matrix *)malloc(sizeof(Matrix));
    *layer.ln2_gamma = matrix_create(1, d_model);
    layer.ln2_beta = (Matrix *)malloc(sizeof(Matrix));
    *layer.ln2_beta = matrix_create(1, d_model);
    layer.ln3_gamma = (Matrix *)malloc(sizeof(Matrix));
    *layer.ln3_gamma = matrix_create(1, d_model);
    layer.ln3_beta = (Matrix *)malloc(sizeof(Matrix));
    *layer.ln3_beta = matrix_create(1, d_model);
    for (int i = 0; i < d_model; i++) {
        layer.ln1_gamma->data[i] = 1.0f;
        layer.ln1_beta->data[i] = 0.0f;
        layer.ln2_gamma->data[i] = 1.0f;
        layer.ln2_beta->data[i] = 0.0f;
        layer.ln3_gamma->data[i] = 1.0f;
        layer.ln3_beta->data[i] = 0.0f;
    }
    return layer;
}

void decoder_layer_free(DecoderLayer *layer) {
    multi_head_attn_free(layer->self_attn);
    multi_head_attn_free(layer->cross_attn);
    feed_forward_free(layer->ffn);
    matrix_free(layer->ln1_gamma);
    matrix_free(layer->ln1_beta);
    matrix_free(layer->ln2_gamma);
    matrix_free(layer->ln2_beta);
    matrix_free(layer->ln3_gamma);
    matrix_free(layer->ln3_beta);
    free(layer->ln1_gamma);
    free(layer->ln1_beta);
    free(layer->ln2_gamma);
    free(layer->ln2_beta);
    free(layer->ln3_gamma);
    free(layer->ln3_beta);
}

Tensor3D decoder_layer_forward(
    DecoderLayer *layer,
    Tensor3D *x,
    Tensor3D *encoder_out,
    Tensor3D *src_mask,
    Tensor3D *tgt_mask,
    DecoderLayerCache *cache
) {
    if (cache) {
        cache->x = clone_tensor(x);
        cache->encoder_out = clone_tensor(encoder_out);
    }

    AttnCache *self_cache = cache ? &cache->self_attn_cache : NULL;
    Tensor3D attn_out = multi_head_attn_forward(layer->self_attn, x, x, x, tgt_mask, self_cache);

    Tensor3D residual1 = clone_tensor(x);
    for (int i = 0; i < x->batch_size * x->seq_len * x->d_model; i++) {
        residual1.data[i] += attn_out.data[i];
    }
    tensor_free(&attn_out);

    if (cache) {
        cache->residual1 = clone_tensor(&residual1);
    }
    tensor_layer_norm(&residual1, layer->ln1_gamma, layer->ln1_beta, 1e-5f);

    if (cache) {
        cache->ln1_out = clone_tensor(&residual1);
    }

    AttnCache *cross_cache = cache ? &cache->cross_attn_cache : NULL;
    Tensor3D cross_out = multi_head_attn_forward(layer->cross_attn, &residual1, encoder_out, encoder_out, src_mask, cross_cache);

    Tensor3D residual2 = clone_tensor(&residual1);
    for (int i = 0; i < residual1.batch_size * residual1.seq_len * residual1.d_model; i++) {
        residual2.data[i] += cross_out.data[i];
    }
    tensor_free(&cross_out);

    if (cache) {
        cache->residual2 = clone_tensor(&residual2);
    }
    tensor_layer_norm(&residual2, layer->ln2_gamma, layer->ln2_beta, 1e-5f);

    if (cache) {
        cache->ln2_out = clone_tensor(&residual2);
    }

    Tensor3D *hidden_ptr = cache ? &cache->ffn_hidden : NULL;
    if (cache) {
        cache->ffn_hidden = tensor_create(x->batch_size, x->seq_len, layer->ffn->d_ff);
    }
    Tensor3D ff_out = feed_forward_forward(layer->ffn, &residual2, hidden_ptr);

    Tensor3D ff_residual = clone_tensor(&residual2);
    for (int i = 0; i < residual2.batch_size * residual2.seq_len * residual2.d_model; i++) {
        ff_out.data[i] += ff_residual.data[i];
    }
    tensor_free(&ff_residual);

    if (cache) {
        cache->ff_out = clone_tensor(&ff_out);
    }
    tensor_layer_norm(&ff_out, layer->ln3_gamma, layer->ln3_beta, 1e-5f);

    if (cache) {
        cache->ln3_out = clone_tensor(&ff_out);
    }

    tensor_free(&residual1);
    tensor_free(&residual2);
    return ff_out;
}

Encoder *encoder_create(int num_layers, int nhead, int d_model, int d_ff, int max_len, float dropout) {
    Encoder *enc = (Encoder *)malloc(sizeof(Encoder));
    enc->num_layers = num_layers;
    enc->layers = (EncoderLayer *)malloc(sizeof(EncoderLayer) * num_layers);
    enc->pe = positional_encoding_create(d_model, max_len, dropout);

    for (int i = 0; i < num_layers; i++) {
        enc->layers[i] = encoder_layer_create(nhead, d_model, d_ff, dropout);
    }

    return enc;
}

void encoder_free(Encoder *enc) {
    for (int i = 0; i < enc->num_layers; i++) {
        encoder_layer_free(&enc->layers[i]);
    }
    positional_encoding_free(enc->pe);
    free(enc->layers);
    free(enc);
}

Tensor3D encoder_forward(Encoder *enc, Tensor3D *src, Tensor3D *src_mask, EncoderCache *cache) {
    Tensor3D x = clone_tensor(src);
    Tensor3D pe = positional_encoding_forward(enc->pe, src->seq_len);

    for (int i = 0; i < src->batch_size * src->seq_len * src->d_model; i++) {
        x.data[i] += pe.data[i % (pe.seq_len * pe.d_model)];
    }
    tensor_free(&pe);

    if (cache) {
        cache->layer_caches = (EncoderLayerCache *)malloc(sizeof(EncoderLayerCache) * enc->num_layers);
        cache->num_layers = enc->num_layers;
    }

    for (int i = 0; i < enc->num_layers; i++) {
        EncoderLayerCache *lc = cache ? &cache->layer_caches[i] : NULL;
        Tensor3D new_x = encoder_layer_forward(&enc->layers[i], &x, src_mask, lc);
        tensor_free(&x);
        x = new_x;
    }

    return x;
}

Decoder *decoder_create(int num_layers, int nhead, int d_model, int d_ff, int max_len, float dropout) {
    Decoder *dec = (Decoder *)malloc(sizeof(Decoder));
    dec->num_layers = num_layers;
    dec->layers = (DecoderLayer *)malloc(sizeof(DecoderLayer) * num_layers);
    dec->pe = positional_encoding_create(d_model, max_len, dropout);

    for (int i = 0; i < num_layers; i++) {
        dec->layers[i] = decoder_layer_create(nhead, d_model, d_ff, dropout);
    }

    return dec;
}

void decoder_free(Decoder *dec) {
    for (int i = 0; i < dec->num_layers; i++) {
        decoder_layer_free(&dec->layers[i]);
    }
    positional_encoding_free(dec->pe);
    free(dec->layers);
    free(dec);
}

Tensor3D decoder_forward(
    Decoder *dec,
    Tensor3D *tgt,
    Tensor3D *encoder_out,
    Tensor3D *src_mask,
    Tensor3D *tgt_mask,
    DecoderCache *cache
) {
    Tensor3D x = clone_tensor(tgt);
    Tensor3D pe = positional_encoding_forward(dec->pe, tgt->seq_len);

    for (int i = 0; i < tgt->batch_size * tgt->seq_len * tgt->d_model; i++) {
        x.data[i] += pe.data[i % (pe.seq_len * pe.d_model)];
    }
    tensor_free(&pe);

    if (cache) {
        cache->tgt_proj = clone_tensor(&x);
        cache->encoder_out = clone_tensor(encoder_out);
        cache->layer_caches = (DecoderLayerCache *)malloc(sizeof(DecoderLayerCache) * dec->num_layers);
        cache->num_layers = dec->num_layers;
    }

    for (int i = 0; i < dec->num_layers; i++) {
        DecoderLayerCache *dc = cache ? &cache->layer_caches[i] : NULL;
        Tensor3D new_x = decoder_layer_forward(&dec->layers[i], &x, encoder_out, src_mask, tgt_mask, dc);
        tensor_free(&x);
        x = new_x;
    }

    return x;
}

Transformer *transformer_create(TransformerConfig *config) {
    Transformer *t = (Transformer *)malloc(sizeof(Transformer));
    t->config = *config;

    t->encoder = encoder_create(
        config->encoder_layers,
        config->nhead,
        config->d_model,
        config->d_ff,
        config->max_len,
        config->dropout
    );

    t->decoder = decoder_create(
        config->decoder_layers,
        config->nhead,
        config->d_model,
        config->d_ff,
        config->max_len,
        config->dropout
    );

    t->input_projection = (Matrix *)malloc(sizeof(Matrix));
    *t->input_projection = matrix_create(config->d_model, config->d_model);
    matrix_xavier(t->input_projection);

    t->output_projection = (Matrix *)malloc(sizeof(Matrix));
    *t->output_projection = matrix_create(config->d_model, config->d_model);
    matrix_xavier(t->output_projection);

    return t;
}

void transformer_free(Transformer *t) {
    encoder_free(t->encoder);
    decoder_free(t->decoder);
    matrix_free(t->input_projection);
    free(t->input_projection);
    matrix_free(t->output_projection);
    free(t->output_projection);
    free(t);
}

Tensor3D transformer_forward(
    Transformer *t,
    Tensor3D *src,
    Tensor3D *tgt,
    Tensor3D *src_mask,
    Tensor3D *tgt_mask,
    TransformerCache *cache
) {
    Tensor3D src_proj = tensor_create(src->batch_size, src->seq_len, src->d_model);
    tensor_mat_mul(src, t->input_projection, &src_proj);
    if (cache) {
        cache->src_proj = clone_tensor(&src_proj);
    }

    Tensor3D encoder_out;
    if (cache) {
        encoder_out = encoder_forward(t->encoder, &src_proj, src_mask, &cache->encoder_cache);
    } else {
        encoder_out = encoder_forward(t->encoder, &src_proj, src_mask, NULL);
        tensor_free(&src_proj);
    }

    Tensor3D tgt_proj = tensor_create(tgt->batch_size, tgt->seq_len, tgt->d_model);
    tensor_mat_mul(tgt, t->input_projection, &tgt_proj);
    if (cache) {
        cache->tgt_proj = clone_tensor(&tgt_proj);
    }

    Tensor3D decoder_out;
    if (cache) {
        decoder_out = decoder_forward(t->decoder, &tgt_proj, &encoder_out, src_mask, tgt_mask, &cache->decoder_cache);
        cache->decoder_out = clone_tensor(&decoder_out);
    } else {
        decoder_out = decoder_forward(t->decoder, &tgt_proj, &encoder_out, src_mask, tgt_mask, NULL);
        tensor_free(&tgt_proj);
        tensor_free(&encoder_out);
    }

    Tensor3D out = tensor_create(decoder_out.batch_size, decoder_out.seq_len, t->config.d_model);
    tensor_mat_mul(&decoder_out, t->output_projection, &out);
    if (!cache) {
        tensor_free(&decoder_out);
    }

    return out;
}

void tensor_layer_norm_backward(Tensor3D *x, Tensor3D *grad_output, Matrix *gamma, Matrix *beta,
                                 Tensor3D *grad_input, float eps, Matrix *grad_gamma, Matrix *grad_beta) {
    (void)beta;
    int b = x->batch_size;
    int s = x->seq_len;
    int d = x->d_model;
    tensor_zero(grad_input);

    for (int i = 0; i < b; i++) {
        for (int j = 0; j < s; j++) {
            float mean = 0, var = 0;
            for (int k = 0; k < d; k++) {
                mean += x->data[i * s * d + j * d + k];
            }
            mean /= d;
            for (int k = 0; k < d; k++) {
                float diff = x->data[i * s * d + j * d + k] - mean;
                var += diff * diff;
            }
            var /= d;
            float std_inv = 1.0f / sqrtf(var + eps);

            float dx_hat_sum = 0;
            float dx_hat_xhat_sum = 0;

            for (int k = 0; k < d; k++) {
                float xhat = (x->data[i * s * d + j * d + k] - mean) * std_inv;
                float dx_hat = grad_output->data[i * s * d + j * d + k] * gamma->data[k];
                dx_hat_sum += dx_hat;
                dx_hat_xhat_sum += dx_hat * xhat;

                if (grad_gamma) grad_gamma->data[k] += grad_output->data[i * s * d + j * d + k] * xhat;
                if (grad_beta) grad_beta->data[k] += grad_output->data[i * s * d + j * d + k];
            }

            for (int k = 0; k < d; k++) {
                float dx_hat = grad_output->data[i * s * d + j * d + k] * gamma->data[k];
                grad_input->data[i * s * d + j * d + k] = std_inv * (
                    dx_hat - dx_hat_sum / d -
                    (x->data[i * s * d + j * d + k] - mean) * dx_hat_xhat_sum / (d * (var + eps))
                );
            }
        }
    }
}

void multi_head_attn_backward(MultiHeadAttention *attn, Tensor3D *query, Tensor3D *key, Tensor3D *value,
    Tensor3D *attn_mask_unused, Tensor3D *grad_output, Tensor3D *grad_query, Tensor3D *grad_key,
    Tensor3D *grad_value, AttnCache *attn_cache, TrainingState *ts) {
    (void)attn_mask_unused;
    (void)grad_key;
    (void)grad_value;
    int b = query->batch_size;
    int s = query->seq_len;
    int d_k = attn->d_k;
    int nhead = attn->nhead;
    int d_model = query->d_model;

    tensor_zero(grad_query);

    Tensor3D grad_out_proj = tensor_create(b, s, d_model);
    tensor_zero(&grad_out_proj);

    Matrix *grad_wq_out = find_grad(ts, attn->wq_out);
    mat_vec_mul_backward(attn->wq_out, attn_cache->input.data, grad_output->data, b, s, d_model, d_model,
                         grad_wq_out ? grad_wq_out : NULL, grad_out_proj.data);

    Tensor3D grad_attend = tensor_create(b, nhead, s * d_k);
    tensor_zero(&grad_attend);

    for (int i = 0; i < b; i++) {
        for (int j = 0; j < s; j++) {
            for (int h = 0; h < nhead; h++) {
                for (int kk = 0; kk < d_k; kk++) {
                    grad_attend.data[i * nhead * s * d_k + h * s * d_k + j * d_k + kk] =
                        grad_out_proj.data[i * s * d_model + j * d_model + h * d_k + kk];
                }
            }
        }
    }

    Tensor3D *attn_weights = &attn_cache->attn_weights;

    Tensor3D v_in = tensor_create(b, s, d_model);
    tensor_mat_mul(value, attn->wv, &v_in);
    Tensor3D v_reshaped = tensor_create(b, nhead, s * d_k);
    for (int i = 0; i < b; i++) {
        for (int h = 0; h < nhead; h++) {
            for (int j = 0; j < s; j++) {
                for (int kk = 0; kk < d_k; kk++) {
                    v_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + kk] =
                        v_in.data[i * s * d_model + j * d_model + h * d_k + kk];
                }
            }
        }
    }

    Tensor3D q_in = tensor_create(b, s, d_model);
    tensor_mat_mul(query, attn->wq, &q_in);
    Tensor3D k_in = tensor_create(b, s, d_model);
    tensor_mat_mul(key, attn->wk, &k_in);

    Tensor3D q_reshaped = tensor_create(b, nhead, s * d_k);
    Tensor3D k_reshaped = tensor_create(b, nhead, s * d_k);
    for (int i = 0; i < b; i++) {
        for (int h = 0; h < nhead; h++) {
            for (int j = 0; j < s; j++) {
                for (int kk = 0; kk < d_k; kk++) {
                    q_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + kk] =
                        q_in.data[i * s * d_model + j * d_model + h * d_k + kk];
                    k_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + kk] =
                        k_in.data[i * s * d_model + j * d_model + h * d_k + kk];
                }
            }
        }
    }

    Tensor3D grad_scores = tensor_create(b, nhead, s * s);
    tensor_zero(&grad_scores);

    Tensor3D grad_attn_probs = tensor_create(b, nhead, s * s);
    Tensor3D grad_v_reshaped = tensor_create(b, nhead, s * d_k);
    tensor_zero(&grad_v_reshaped);

    for (int i = 0; i < b; i++) {
        for (int h = 0; h < nhead; h++) {
            for (int q = 0; q < s; q++) {
                for (int kk = 0; kk < d_k; kk++) {
                    float gattend = grad_attend.data[i * nhead * s * d_k + h * s * d_k + q * d_k + kk];

                    float gattn_sum = 0;
                    for (int t = 0; t < s; t++) {
                        float aw = attn_weights->data[i * nhead * s * s + h * s * s + q * s + t];
                        gattn_sum += gattend * v_reshaped.data[i * nhead * s * d_k + h * s * d_k + t * d_k + kk] * aw;
                        grad_v_reshaped.data[i * nhead * s * d_k + h * s * d_k + t * d_k + kk] +=
                            aw * gattend;
                    }
                    grad_attn_probs.data[i * nhead * s * s + h * s * s + q * s] = 0;
                }
            }
        }
    }

    for (int i = 0; i < b; i++) {
        for (int h = 0; h < nhead; h++) {
            for (int q = 0; q < s; q++) {
                float dot_ga = 0;
                for (int kk = 0; kk < d_k; kk++) {
                    dot_ga += grad_attend.data[i * nhead * s * d_k + h * s * d_k + q * d_k + kk] *
                              v_reshaped.data[i * nhead * s * d_k + h * s * d_k + q * d_k + kk];
                }

                for (int t = 0; t < s; t++) {
                    float aw = attn_weights->data[i * nhead * s * s + h * s * s + q * s + t];
                    float sum_ga_v = 0;
                    for (int kk = 0; kk < d_k; kk++) {
                        sum_ga_v += grad_attend.data[i * nhead * s * d_k + h * s * d_k + q * d_k + kk] *
                                    v_reshaped.data[i * nhead * s * d_k + h * s * d_k + t * d_k + kk];
                    }

                    float dot_ga_t = 0;
                    for (int kk2 = 0; kk2 < d_k; kk2++) {
                        dot_ga_t += grad_attend.data[i * nhead * s * d_k + h * s * d_k + q * d_k + kk2] *
                                    v_reshaped.data[i * nhead * s * d_k + h * s * d_k + t * d_k + kk2];
                    }

                    grad_scores.data[i * nhead * s * s + h * s * s + q * s + t] = aw * (dot_ga_t - dot_ga);
                }
            }
        }
    }

    float scale = 1.0f / sqrtf((float)d_k);

    Tensor3D grad_q_reshaped = tensor_create(b, nhead, s * d_k);
    Tensor3D grad_k_reshaped = tensor_create(b, nhead, s * d_k);
    tensor_zero(&grad_q_reshaped);
    tensor_zero(&grad_k_reshaped);

    for (int i = 0; i < b; i++) {
        for (int h = 0; h < nhead; h++) {
            for (int q = 0; q < s; q++) {
                for (int kk = 0; kk < d_k; kk++) {
                    float gq = 0, gk = 0;
                    for (int t = 0; t < s; t++) {
                        float gs = grad_scores.data[i * nhead * s * s + h * s * s + q * s + t] * scale;
                        gq += gs * k_reshaped.data[i * nhead * s * d_k + h * s * d_k + t * d_k + kk];
                        gk += grad_scores.data[i * nhead * s * s + h * s * s + t * s + q] * scale *
                              q_reshaped.data[i * nhead * s * d_k + h * s * d_k + t * d_k + kk];
                    }
                    grad_q_reshaped.data[i * nhead * s * d_k + h * s * d_k + q * d_k + kk] += gq;
                    grad_k_reshaped.data[i * nhead * s * d_k + h * s * d_k + q * d_k + kk] += gk;
                }
            }
        }
    }

    Tensor3D grad_q = tensor_create(b, s, d_model);
    Tensor3D grad_k = tensor_create(b, s, d_model);
    Tensor3D grad_v = tensor_create(b, s, d_model);
    tensor_zero(&grad_q);
    tensor_zero(&grad_k);
    tensor_zero(&grad_v);

    for (int i = 0; i < b; i++) {
        for (int j = 0; j < s; j++) {
            for (int h = 0; h < nhead; h++) {
                for (int kk = 0; kk < d_k; kk++) {
                    grad_q.data[i * s * d_model + j * d_model + h * d_k + kk] =
                        grad_q_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + kk];
                    grad_k.data[i * s * d_model + j * d_model + h * d_k + kk] =
                        grad_k_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + kk];
                    grad_v.data[i * s * d_model + j * d_model + h * d_k + kk] =
                        grad_v_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + kk];
                }
            }
        }
    }

    Matrix *grad_wq = find_grad(ts, attn->wq);
    Matrix *grad_wk = find_grad(ts, attn->wk);
    Matrix *grad_wv = find_grad(ts, attn->wv);

    mat_vec_mul_backward(attn->wq, attn_cache->input.data, grad_q.data, b, s, d_model, d_model,
                         grad_wq, grad_query->data);
    if (grad_key && grad_key->data) {
        mat_vec_mul_backward(attn->wk, attn_cache->key_input.data, grad_k.data, b, s, d_model, d_model,
                             grad_wk, grad_key->data);
    } else {
        mat_vec_mul_backward(attn->wk, attn_cache->key_input.data, grad_k.data, b, s, d_model, d_model,
                             grad_wk, grad_query->data);
    }
    if (grad_value && grad_value->data) {
        mat_vec_mul_backward(attn->wv, attn_cache->value_input.data, grad_v.data, b, s, d_model, d_model,
                             grad_wv, grad_value->data);
    } else {
        mat_vec_mul_backward(attn->wv, attn_cache->value_input.data, grad_v.data, b, s, d_model, d_model,
                             grad_wv, grad_query->data);
    }
    tensor_free(&q_in);
    tensor_free(&k_in);
    tensor_free(&v_in);
    tensor_free(&q_reshaped);
    tensor_free(&k_reshaped);
    tensor_free(&v_reshaped);
    tensor_free(&grad_out_proj);
    tensor_free(&grad_attend);
    tensor_free(&grad_scores);
    tensor_free(&grad_attn_probs);
    tensor_free(&grad_v_reshaped);
    tensor_free(&grad_q_reshaped);
    tensor_free(&grad_k_reshaped);
    tensor_free(&grad_q);
    tensor_free(&grad_k);
    tensor_free(&grad_v);
}

void feed_forward_backward(FeedForward *ffn, Tensor3D *x, Tensor3D *grad_output, Tensor3D *grad_input, Tensor3D *hidden_cache, TrainingState *ts) {
    int b = x->batch_size;
    int s = x->seq_len;
    int d_model = x->d_model;
    int d_ff = ffn->d_ff;
    tensor_zero(grad_input);

    Matrix *grad_w2 = find_grad(ts, ffn->w2);
    Matrix *grad_b2 = find_grad(ts, ffn->b2);
    Matrix *grad_w1 = find_grad(ts, ffn->w1);
    Matrix *grad_b1 = find_grad(ts, ffn->b1);

    Tensor3D grad_hidden_after_act = tensor_create(b, s, d_ff);
    tensor_zero(&grad_hidden_after_act);

    for (int bs = 0; bs < b; bs++) {
        for (int seq = 0; seq < s; seq++) {
            for (int j = 0; j < d_model; j++) {
                float go = grad_output->data[bs * s * d_model + seq * d_model + j];
                if (grad_b2) grad_b2->data[j] += go;
                for (int k = 0; k < d_ff; k++) {
                    float hidden_post = hidden_cache ? act_forward(ffn->activation, hidden_cache->data[bs * s * d_ff + seq * d_ff + k]) : 0;
                    if (grad_w2) grad_w2->data[k * d_model + j] += go * hidden_post;
                    grad_hidden_after_act.data[bs * s * d_ff + seq * d_ff + k] +=
                        go * ffn->w2->data[k * d_model + j];
                }
            }
        }
    }

    Tensor3D grad_hidden = tensor_create(b, s, d_ff);
    for (int i = 0; i < b * s * d_ff; i++) {
        if (hidden_cache) {
            float pre_act = hidden_cache->data[i];
            float deriv = act_backward(ffn->activation, pre_act);
            grad_hidden.data[i] = grad_hidden_after_act.data[i] * deriv;
        } else {
            grad_hidden.data[i] = grad_hidden_after_act.data[i];
        }
    }

    for (int bs = 0; bs < b; bs++) {
        for (int seq = 0; seq < s; seq++) {
            for (int j = 0; j < d_ff; j++) {
                if (grad_b1) grad_b1->data[j] += grad_hidden.data[bs * s * d_ff + seq * d_ff + j];
                for (int k = 0; k < d_model; k++) {
                    if (grad_w1) grad_w1->data[k * d_ff + j] +=
                        grad_hidden.data[bs * s * d_ff + seq * d_ff + j] * x->data[bs * s * d_model + seq * d_model + k];
                    grad_input->data[bs * s * d_model + seq * d_model + k] +=
                        grad_hidden.data[bs * s * d_ff + seq * d_ff + j] * ffn->w1->data[k * d_ff + j];
                }
            }
        }
    }

    tensor_free(&grad_hidden_after_act);
    tensor_free(&grad_hidden);
}

void encoder_layer_backward(
    EncoderLayer *layer,
    EncoderLayerCache *cache,
    Tensor3D *grad_output,
    Tensor3D *grad_input,
    TrainingState *ts
) {
    int b = grad_output->batch_size;
    int s = grad_output->seq_len;
    int d = grad_output->d_model;

    Matrix *grad_ln2_gamma = find_grad(ts, layer->ln2_gamma);
    Matrix *grad_ln2_beta = find_grad(ts, layer->ln2_beta);
    Matrix *grad_ln1_gamma = find_grad(ts, layer->ln1_gamma);
    Matrix *grad_ln1_beta = find_grad(ts, layer->ln1_beta);

    Tensor3D grad_ln2 = tensor_create(b, s, d);
    tensor_layer_norm_backward(&cache->ff_out, grad_output, layer->ln2_gamma, layer->ln2_beta, &grad_ln2, 1e-5f, grad_ln2_gamma, grad_ln2_beta);

    Tensor3D grad_residual2 = clone_tensor(&grad_ln2);
    Tensor3D grad_ff = clone_tensor(&grad_ln2);

    Tensor3D grad_ff_input = tensor_create(b, s, layer->ffn->d_ff);
    tensor_zero(&grad_ff_input);
    feed_forward_backward(layer->ffn, &cache->ln1_out, &grad_ff, &grad_ff_input, &cache->ffn_hidden, ts);
    tensor_free(&grad_ff);

    Tensor3D grad_ln1_in = tensor_create(b, s, d);
    tensor_layer_norm_backward(&cache->residual, &grad_ff_input, layer->ln1_gamma, layer->ln1_beta, &grad_ln1_in, 1e-5f, grad_ln1_gamma, grad_ln1_beta);
    tensor_free(&grad_ff_input);

    for (int i = 0; i < b * s * d; i++) {
        grad_ln1_in.data[i] += grad_residual2.data[i];
    }
    tensor_free(&grad_residual2);

    grad_input->batch_size = b;
    grad_input->seq_len = s;
    grad_input->d_model = d;
    grad_input->data = (float *)calloc(b * s * d, sizeof(float));

    Tensor3D grad_attn_out = clone_tensor(&grad_ln1_in);
    tensor_free(&grad_ln1_in);

    multi_head_attn_backward(layer->self_attn, &cache->x, &cache->x, &cache->x, NULL,
                              &grad_attn_out, grad_input, grad_input, grad_input, &cache->self_attn_cache, ts);

    for (int i = 0; i < b * s * d; i++) {
        grad_input->data[i] += grad_attn_out.data[i];
    }
    tensor_free(&grad_attn_out);
    tensor_free(&grad_ln2);
}

void decoder_layer_backward(
    DecoderLayer *layer,
    DecoderLayerCache *cache,
    Tensor3D *grad_output,
    Tensor3D *grad_input,
    Tensor3D *grad_enc_out,
    TrainingState *ts
) {
    int b = grad_output->batch_size;
    int s = grad_output->seq_len;
    int d = grad_output->d_model;

    Matrix *grad_ln3_gamma = find_grad(ts, layer->ln3_gamma);
    Matrix *grad_ln3_beta = find_grad(ts, layer->ln3_beta);
    Matrix *grad_ln2_gamma = find_grad(ts, layer->ln2_gamma);
    Matrix *grad_ln2_beta = find_grad(ts, layer->ln2_beta);
    Matrix *grad_ln1_gamma = find_grad(ts, layer->ln1_gamma);
    Matrix *grad_ln1_beta = find_grad(ts, layer->ln1_beta);

    Tensor3D grad_ln3 = tensor_create(b, s, d);
    tensor_layer_norm_backward(&cache->ff_out, grad_output, layer->ln3_gamma, layer->ln3_beta, &grad_ln3, 1e-5f, grad_ln3_gamma, grad_ln3_beta);

    Tensor3D grad_residual3 = clone_tensor(&grad_ln3);
    Tensor3D grad_ff = clone_tensor(&grad_ln3);

    Tensor3D grad_ff_input = tensor_create(b, s, layer->ffn->d_ff);
    tensor_zero(&grad_ff_input);
    feed_forward_backward(layer->ffn, &cache->ln2_out, &grad_ff, &grad_ff_input, &cache->ffn_hidden, ts);
    tensor_free(&grad_ff);

    Tensor3D grad_ln2_in = tensor_create(b, s, d);
    tensor_layer_norm_backward(&cache->residual2, &grad_ff_input, layer->ln2_gamma, layer->ln2_beta, &grad_ln2_in, 1e-5f, grad_ln2_gamma, grad_ln2_beta);
    tensor_free(&grad_ff_input);

    for (int i = 0; i < b * s * d; i++) {
        grad_ln2_in.data[i] += grad_residual3.data[i];
    }
    tensor_free(&grad_residual3);

    Tensor3D grad_cross_input = clone_tensor(&grad_ln2_in);
    tensor_free(&grad_ln2_in);

    Tensor3D grad_ln1_in_from_cross = tensor_create(b, s, d);
    tensor_zero(&grad_ln1_in_from_cross);
    Tensor3D local_grad_enc = tensor_create(b, s, d);
    tensor_zero(&local_grad_enc);

    multi_head_attn_backward(layer->cross_attn, &cache->ln1_out, &cache->encoder_out, &cache->encoder_out, NULL,
                              &grad_cross_input, &grad_ln1_in_from_cross, &local_grad_enc, &local_grad_enc, &cache->cross_attn_cache, ts);

    for (int i = 0; i < b * s * d; i++) {
        grad_enc_out->data[i] += local_grad_enc.data[i];
    }
    tensor_free(&local_grad_enc);

    Tensor3D grad_ln1 = tensor_create(b, s, d);
    tensor_layer_norm_backward(&cache->x, &grad_ln1_in_from_cross, layer->ln1_gamma, layer->ln1_beta, &grad_ln1, 1e-5f, grad_ln1_gamma, grad_ln1_beta);
    tensor_free(&grad_ln1_in_from_cross);
    tensor_free(&grad_cross_input);

    Tensor3D grad_self_input = clone_tensor(&grad_ln1);
    tensor_free(&grad_ln1);

    grad_input->batch_size = b;
    grad_input->seq_len = s;
    grad_input->d_model = d;
    grad_input->data = (float *)calloc(b * s * d, sizeof(float));

    multi_head_attn_backward(layer->self_attn, &cache->x, &cache->x, &cache->x, NULL,
                              &grad_self_input, grad_input, grad_input, grad_input, &cache->self_attn_cache, ts);

    for (int i = 0; i < b * s * d; i++) {
        grad_input->data[i] += grad_self_input.data[i];
    }
    tensor_free(&grad_self_input);
    tensor_free(&grad_ln3);
}

void transformer_backward(
    Transformer *t,
    TransformerCache *cache,
    Tensor3D *grad,
    Tensor3D *tgt,
    TrainingState *ts
) {
    (void)tgt;
    int batch = grad->batch_size;
    int seq = grad->seq_len;
    int d_model = grad->d_model;

    Tensor3D decoder_out = cache->decoder_out;

    Matrix *grad_output_proj = find_grad(ts, t->output_projection);
    if (grad_output_proj) {
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                for (int i = 0; i < d_model; i++) {
                    float g = grad->data[b * seq * d_model + s * d_model + i];
                    for (int j = 0; j < d_model; j++) {
                        grad_output_proj->data[i * d_model + j] +=
                            g * decoder_out.data[b * seq * d_model + s * d_model + j];
                    }
                }
            }
        }
    }

    Tensor3D grad_decoder_out = tensor_create(batch, seq, d_model);
    tensor_zero(&grad_decoder_out);
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            for (int i = 0; i < d_model; i++) {
                float g = grad->data[b * seq * d_model + s * d_model + i];
                for (int j = 0; j < d_model; j++) {
                    grad_decoder_out.data[b * seq * d_model + s * d_model + j] += g * t->output_projection->data[i * d_model + j];
                }
            }
        }
    }

    Tensor3D grad_encoder_out = tensor_create(batch, seq, d_model);
    tensor_zero(&grad_encoder_out);

    for (int i = t->decoder->num_layers - 1; i >= 0; i--) {
        DecoderLayerCache *dlc = &cache->decoder_cache.layer_caches[i];
        Tensor3D grad_input_dec = {0};
        decoder_layer_backward(&t->decoder->layers[i], dlc, &grad_decoder_out, &grad_input_dec, &grad_encoder_out, ts);
        tensor_free(&grad_decoder_out);
        grad_decoder_out = grad_input_dec;
    }

    Matrix *grad_input_proj = find_grad(ts, t->input_projection);

    for (int i = t->encoder->num_layers - 1; i >= 0; i--) {
        EncoderLayerCache *elc = &cache->encoder_cache.layer_caches[i];
        Tensor3D grad_input_enc = {0};
        encoder_layer_backward(&t->encoder->layers[i], elc, &grad_encoder_out, &grad_input_enc, ts);
        tensor_free(&grad_encoder_out);
        grad_encoder_out = grad_input_enc;
    }

    if (grad_input_proj) {
        Tensor3D src = cache->src_proj;
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seq; s++) {
                for (int i = 0; i < d_model; i++) {
                    float g = grad_encoder_out.data[b * seq * d_model + s * d_model + i];
                    for (int j = 0; j < d_model; j++) {
                        grad_input_proj->data[i * d_model + j] +=
                            g * src.data[b * seq * d_model + s * d_model + j];
                    }
                }
            }
        }
    }

    tensor_free(&grad_decoder_out);
    tensor_free(&grad_encoder_out);
}

void transformer_cache_free(TransformerCache *cache, int encoder_layers, int decoder_layers) {
    tensor_free(&cache->src_proj);
    tensor_free(&cache->tgt_proj);
    tensor_free(&cache->decoder_out);

    for (int i = 0; i < encoder_layers; i++) {
        EncoderLayerCache *ec = &cache->encoder_cache.layer_caches[i];
        tensor_free(&ec->x);
        tensor_free(&ec->residual);
        tensor_free(&ec->ln1_out);
        tensor_free(&ec->ffn_hidden);
        tensor_free(&ec->ff_out);
        tensor_free(&ec->ln2_out);
        tensor_free(&ec->self_attn_cache.attn_weights);
        tensor_free(&ec->self_attn_cache.input);
        tensor_free(&ec->self_attn_cache.key_input);
        tensor_free(&ec->self_attn_cache.value_input);
    }
    free(cache->encoder_cache.layer_caches);

    for (int i = 0; i < decoder_layers; i++) {
        DecoderLayerCache *dc = &cache->decoder_cache.layer_caches[i];
        tensor_free(&dc->x);
        tensor_free(&dc->residual1);
        tensor_free(&dc->ln1_out);
        tensor_free(&dc->residual2);
        tensor_free(&dc->ln2_out);
        tensor_free(&dc->ffn_hidden);
        tensor_free(&dc->ff_out);
        tensor_free(&dc->ln3_out);
        tensor_free(&dc->encoder_out);
        tensor_free(&dc->self_attn_cache.attn_weights);
        tensor_free(&dc->self_attn_cache.input);
        tensor_free(&dc->self_attn_cache.key_input);
        tensor_free(&dc->self_attn_cache.value_input);
        tensor_free(&dc->cross_attn_cache.attn_weights);
        tensor_free(&dc->cross_attn_cache.input);
        tensor_free(&dc->cross_attn_cache.key_input);
        tensor_free(&dc->cross_attn_cache.value_input);
    }
    free(cache->decoder_cache.layer_caches);
    tensor_free(&cache->decoder_cache.encoder_out);
}

void tensor_print(Tensor3D *t, const char *name) {
    printf("%s [batch=%d, seq=%d, d_model=%d]:\n", name, t->batch_size, t->seq_len, t->d_model);
    for (int i = 0; i < t->batch_size && i < 2; i++) {
        printf("  batch %d:\n", i);
        for (int j = 0; j < t->seq_len && j < 5; j++) {
            printf("    seq %d: [", j);
            for (int k = 0; k < t->d_model && k < 8; k++) {
                printf("%.3f ", t->data[i * t->seq_len * t->d_model + j * t->d_model + k]);
            }
            printf("]\n");
        }
    }
}

AdamOptimizer adam_create(float lr, float beta1, float beta2, float eps) {
    AdamOptimizer opt;
    opt.learning_rate = lr;
    opt.beta1 = beta1;
    opt.beta2 = beta2;
    opt.eps = eps;
    opt.t = 0;
    return opt;
}

void adam_update_matrix(Matrix *m, Matrix *grad, Matrix *m_moment, Matrix *v_moment, AdamOptimizer *opt) {
    opt->t++;
    float lr_t = opt->learning_rate * sqrtf(1.0f - powf(opt->beta2, opt->t)) / (1.0f - powf(opt->beta1, opt->t));
    int size = m->rows * m->cols;
    for (int i = 0; i < size; i++) {
        m_moment->data[i] = opt->beta1 * m_moment->data[i] + (1.0f - opt->beta1) * grad->data[i];
        v_moment->data[i] = opt->beta2 * v_moment->data[i] + (1.0f - opt->beta2) * grad->data[i] * grad->data[i];
        m->data[i] -= lr_t * m_moment->data[i] / (sqrtf(v_moment->data[i]) + opt->eps);
    }
}

void adam_update_tensor(Tensor3D *t, Tensor3D *grad, Tensor3D *m_moment, Tensor3D *v_moment, AdamOptimizer *opt) {
    opt->t++;
    float lr_t = opt->learning_rate * sqrtf(1.0f - powf(opt->beta2, opt->t)) / (1.0f - powf(opt->beta1, opt->t));
    int size = t->batch_size * t->seq_len * t->d_model;
    for (int i = 0; i < size; i++) {
        m_moment->data[i] = opt->beta1 * m_moment->data[i] + (1.0f - opt->beta1) * grad->data[i];
        v_moment->data[i] = opt->beta2 * v_moment->data[i] + (1.0f - opt->beta2) * grad->data[i] * grad->data[i];
        t->data[i] -= lr_t * m_moment->data[i] / (sqrtf(v_moment->data[i]) + opt->eps);
    }
}

void tensor_softmax_backward(Tensor3D *output, Tensor3D *grad_output, Tensor3D *grad_input, int axis) {
    (void)axis;
    int b = output->batch_size;
    int s = output->seq_len;
    int d = output->d_model;
    tensor_zero(grad_input);

    for (int i = 0; i < b; i++) {
        for (int j = 0; j < s; j++) {
            float sum = 0;
            for (int k = 0; k < d; k++) {
                int idx = i * s * d + j * d + k;
                sum += grad_output->data[idx] * output->data[idx];
            }
            for (int k = 0; k < d; k++) {
                int idx = i * s * d + j * d + k;
                grad_input->data[idx] = output->data[idx] * (grad_output->data[idx] - sum);
            }
        }
    }
}

Matrix *find_grad(TrainingState *ts, Matrix *param) {
    for (int i = 0; i < ts->count; i++) {
        if (ts->entries[i].param == param) {
            return &ts->entries[i].grad;
        }
    }
    return NULL;
}

static void register_param(TrainingState *ts, Matrix *param) {
    int i = ts->count++;
    ts->entries[i].param = param;
    ts->entries[i].grad = matrix_create(param->rows, param->cols);
    ts->entries[i].adam_m = matrix_create(param->rows, param->cols);
    ts->entries[i].adam_v = matrix_create(param->rows, param->cols);
    matrix_zero(&ts->entries[i].grad);
    matrix_zero(&ts->entries[i].adam_m);
    matrix_zero(&ts->entries[i].adam_v);
}

TrainingState *training_state_create(Transformer *model, float lr, float beta1, float beta2, float eps) {
    TrainingState *ts = (TrainingState *)calloc(1, sizeof(TrainingState));
    ts->optimizer = adam_create(lr, beta1, beta2, eps);

    for (int i = 0; i < model->encoder->num_layers; i++) {
        EncoderLayer *l = &model->encoder->layers[i];
        register_param(ts, l->self_attn->wq);
        register_param(ts, l->self_attn->wk);
        register_param(ts, l->self_attn->wv);
        register_param(ts, l->self_attn->wq_out);
        register_param(ts, l->ffn->w1);
        register_param(ts, l->ffn->b1);
        register_param(ts, l->ffn->w2);
        register_param(ts, l->ffn->b2);
        register_param(ts, l->ln1_gamma);
        register_param(ts, l->ln1_beta);
        register_param(ts, l->ln2_gamma);
        register_param(ts, l->ln2_beta);
    }

    for (int i = 0; i < model->decoder->num_layers; i++) {
        DecoderLayer *l = &model->decoder->layers[i];
        register_param(ts, l->self_attn->wq);
        register_param(ts, l->self_attn->wk);
        register_param(ts, l->self_attn->wv);
        register_param(ts, l->self_attn->wq_out);
        register_param(ts, l->cross_attn->wq);
        register_param(ts, l->cross_attn->wk);
        register_param(ts, l->cross_attn->wv);
        register_param(ts, l->cross_attn->wq_out);
        register_param(ts, l->ffn->w1);
        register_param(ts, l->ffn->b1);
        register_param(ts, l->ffn->w2);
        register_param(ts, l->ffn->b2);
        register_param(ts, l->ln1_gamma);
        register_param(ts, l->ln1_beta);
        register_param(ts, l->ln2_gamma);
        register_param(ts, l->ln2_beta);
        register_param(ts, l->ln3_gamma);
        register_param(ts, l->ln3_beta);
    }

    register_param(ts, model->input_projection);
    register_param(ts, model->output_projection);

    return ts;
}

void training_state_free(TrainingState *ts) {
    for (int i = 0; i < ts->count; i++) {
        matrix_free(&ts->entries[i].grad);
        matrix_free(&ts->entries[i].adam_m);
        matrix_free(&ts->entries[i].adam_v);
    }
    free(ts);
}

void training_state_zero_grads(TrainingState *ts) {
    for (int i = 0; i < ts->count; i++) {
        matrix_zero(&ts->entries[i].grad);
    }
}

void training_state_update(TrainingState *ts) {
    for (int i = 0; i < ts->count; i++) {
        adam_update_matrix(ts->entries[i].param, &ts->entries[i].grad,
                          &ts->entries[i].adam_m, &ts->entries[i].adam_v,
                          &ts->optimizer);
        matrix_zero(&ts->entries[i].grad);
    }
    ts->optimizer.t = 0;
}

void training_state_clip_grads(TrainingState *ts, float max_norm) {
    float total_norm = 0;
    for (int i = 0; i < ts->count; i++) {
        int size = ts->entries[i].grad.rows * ts->entries[i].grad.cols;
        for (int j = 0; j < size; j++) {
            total_norm += ts->entries[i].grad.data[j] * ts->entries[i].grad.data[j];
        }
    }
    total_norm = sqrtf(total_norm);
    if (total_norm > max_norm) {
        float scale = max_norm / total_norm;
        for (int i = 0; i < ts->count; i++) {
            int size = ts->entries[i].grad.rows * ts->entries[i].grad.cols;
            for (int j = 0; j < size; j++) {
                ts->entries[i].grad.data[j] *= scale;
            }
        }
    }
}

LossResult cross_entropy_loss(Tensor3D *pred, int *targets, int vocab_size_unused) {
    (void)vocab_size_unused;
    LossResult result;
    int batch = pred->batch_size;
    int seq = pred->seq_len;
    int d_model = pred->d_model;

    result.loss = 0;
    result.grad = tensor_create(batch, seq, d_model);
    tensor_zero(&result.grad);

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            float max_val = -1e9f;
            for (int v = 0; v < d_model; v++) {
                int i = b * seq * d_model + s * d_model + v;
                if (pred->data[i] > max_val) max_val = pred->data[i];
            }

            float sum = 0;
            for (int v = 0; v < d_model; v++) {
                int i = b * seq * d_model + s * d_model + v;
                sum += expf(pred->data[i] - max_val);
            }

            int target = targets[b * seq + s];
            for (int v = 0; v < d_model; v++) {
                int i = b * seq * d_model + s * d_model + v;
                float prob = expf(pred->data[i] - max_val) / sum;
                result.grad.data[i] = prob;
                if (v == target) {
                    result.grad.data[i] -= 1.0f;
                    result.loss -= logf(prob + 1e-8f);
                }
            }
        }
    }
    result.loss /= (batch * seq);
    return result;
}

LossResult mse_loss(Tensor3D *pred, Tensor3D *target) {
    LossResult result;
    int size = pred->batch_size * pred->seq_len * pred->d_model;
    result.loss = 0;
    result.grad = tensor_create(pred->batch_size, pred->seq_len, pred->d_model);

    for (int i = 0; i < size; i++) {
        float diff = pred->data[i] - target->data[i];
        result.loss += diff * diff;
        result.grad.data[i] = 2.0f * diff / size;
    }
    result.loss /= size;
    return result;
}

void loss_print(LossResult *loss, const char *name) {
    printf("Loss %s: %.6f\n", name, loss->loss);
}

Trainer trainer_create(Transformer *model, float lr, float beta1, float beta2, float eps) {
    Trainer trainer;
    trainer.model = model;
    trainer.ts = training_state_create(model, lr, beta1, beta2, eps);
    trainer.learning_rate = lr;
    return trainer;
}

void trainer_free(Trainer *trainer) {
    if (trainer->ts) {
        training_state_free(trainer->ts);
    }
}

void trainer_train_step(Trainer *trainer, Tensor3D *src, Tensor3D *tgt, int *targets, int vocab_size) {
    training_state_zero_grads(trainer->ts);

    TransformerCache cache = {0};
    Tensor3D output = transformer_forward(trainer->model, src, tgt, NULL, NULL, &cache);

    LossResult loss = cross_entropy_loss(&output, targets, vocab_size);

    transformer_backward(trainer->model, &cache, &loss.grad, tgt, trainer->ts);

    training_state_update(trainer->ts);

    transformer_cache_free(&cache, trainer->model->config.encoder_layers, trainer->model->config.decoder_layers);
    tensor_free(&output);
    tensor_free(&loss.grad);
}

void trainer_train_epoch(Trainer *trainer, Tensor3D *src, Tensor3D *tgt, int *targets, int vocab_size, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("Epoch %d\n", epoch + 1);
        trainer_train_step(trainer, src, tgt, targets, vocab_size);
    }
}

void trainer_save(Trainer *trainer, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return;

    Transformer *t = trainer->model;
    fwrite(&t->config, sizeof(TransformerConfig), 1, f);

    for (int i = 0; i < t->encoder->num_layers; i++) {
        EncoderLayer *l = &t->encoder->layers[i];
        fwrite(l->self_attn->wq->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->self_attn->wk->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->self_attn->wv->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->self_attn->wq_out->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->ffn->w1->data, sizeof(float), t->config.d_model * l->ffn->d_ff, f);
        fwrite(l->ffn->b1->data, sizeof(float), l->ffn->d_ff, f);
        fwrite(l->ffn->w2->data, sizeof(float), l->ffn->d_ff * t->config.d_model, f);
        fwrite(l->ffn->b2->data, sizeof(float), t->config.d_model, f);
        fwrite(l->ln1_gamma->data, sizeof(float), t->config.d_model, f);
        fwrite(l->ln1_beta->data, sizeof(float), t->config.d_model, f);
        fwrite(l->ln2_gamma->data, sizeof(float), t->config.d_model, f);
        fwrite(l->ln2_beta->data, sizeof(float), t->config.d_model, f);
    }

    for (int i = 0; i < t->decoder->num_layers; i++) {
        DecoderLayer *l = &t->decoder->layers[i];
        fwrite(l->self_attn->wq->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->self_attn->wk->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->self_attn->wv->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->self_attn->wq_out->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->cross_attn->wq->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->cross_attn->wk->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->cross_attn->wv->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->cross_attn->wq_out->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->ffn->w1->data, sizeof(float), t->config.d_model * l->ffn->d_ff, f);
        fwrite(l->ffn->b1->data, sizeof(float), l->ffn->d_ff, f);
        fwrite(l->ffn->w2->data, sizeof(float), l->ffn->d_ff * t->config.d_model, f);
        fwrite(l->ffn->b2->data, sizeof(float), t->config.d_model, f);
        fwrite(l->ln1_gamma->data, sizeof(float), t->config.d_model, f);
        fwrite(l->ln1_beta->data, sizeof(float), t->config.d_model, f);
        fwrite(l->ln2_gamma->data, sizeof(float), t->config.d_model, f);
        fwrite(l->ln2_beta->data, sizeof(float), t->config.d_model, f);
        fwrite(l->ln3_gamma->data, sizeof(float), t->config.d_model, f);
        fwrite(l->ln3_beta->data, sizeof(float), t->config.d_model, f);
    }

    fwrite(t->input_projection->data, sizeof(float), t->config.d_model * t->config.d_model, f);
    fwrite(t->output_projection->data, sizeof(float), t->config.d_model * t->config.d_model, f);

    fclose(f);
}

Trainer trainer_load(const char *filename) {
    (void)filename;
    Trainer trainer = {0};
    return trainer;
}