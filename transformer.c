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

void matrix_rand(Matrix *m) {
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
}

void matrix_xavier(Matrix *m) {
    float scale = sqrtf(2.0f / (m->rows + m->cols));
    for (int i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    }
}

void tensor_softmax(Tensor3D *t, int axis) {
    // General softmax for 3D tensor
    // For attention scores: t is (batch, nhead, s*s) where last dim is s*s
    // axis=2 means softmax over the last dimension
    
    int b = t->batch_size;
    int s_orig = t->seq_len;
    int d = t->d_model;
    
    if (axis == 2) {
        // For each (batch, seq), softmax over d_model
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
                x->data[i * s * d + j * d + k] =
                    (x->data[i * s * d + j * d + k] - mean) / sqrtf(var + eps) * gamma->data[k] + beta->data[k];
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

static void tensor_mat_mul(Tensor3D *x, Matrix *w, Tensor3D *out) {
    mat_vec_mul(w, x->data, out->data, x->batch_size, x->seq_len, x->d_model, w->cols);
}

Tensor3D multi_head_attn_forward(
    MultiHeadAttention *attn,
    Tensor3D *query,
    Tensor3D *key,
    Tensor3D *value,
    Tensor3D *attn_mask
) {
    int b = query->batch_size;
    int s = query->seq_len;
    int d_k = attn->d_k;
    int nhead = attn->nhead;
    int d_model = query->d_model;

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
                    for (int k = 0; k < d_k; k++) {
                        dot += q_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + k] *
                               k_reshaped.data[i * nhead * s * d_k + h * s * d_k + t * d_k + k];
                    }
                    scores.data[i * nhead * s * s + h * s * s + j * s + t] = dot / scale;
                }
            }
        }
    }

    if (attn_mask && attn_mask->data) {
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < s; j++) {
                for (int t = 0; t < s; t++) {
                    if (attn_mask->data[i * s + j * s + t] == 0) {
                        scores.data[i * nhead * s * s + (j / s) * s * s + j % s + t] = -1e9f;
                    }
                }
            }
        }
    }

    // Softmax over key dimension for each (batch, head, query)
    for (int i = 0; i < b; i++) {
        for (int h = 0; h < nhead; h++) {
            for (int q = 0; q < s; q++) {
                float max_val = -1e9f;
                for (int k = 0; k < s; k++) {
                    int idx = i * nhead * s * s + h * s * s + q * s + k;
                    if (scores.data[idx] > max_val) max_val = scores.data[idx];
                }
                float sum = 0;
                for (int k = 0; k < s; k++) {
                    int idx = i * nhead * s * s + h * s * s + q * s + k;
                    sum += expf(scores.data[idx] - max_val);
                }
                for (int k = 0; k < s; k++) {
                    int idx = i * nhead * s * s + h * s * s + q * s + k;
                    scores.data[idx] = expf(scores.data[idx] - max_val) / sum;
                }
            }
        }
    }

    Tensor3D attend = tensor_create(b, nhead, s * d_k);

    for (int i = 0; i < b; i++) {
        for (int h = 0; h < nhead; h++) {
            for (int j = 0; j < s; j++) {
                for (int k = 0; k < d_k; k++) {
                    float sum = 0;
                    for (int t = 0; t < s; t++) {
                        sum += scores.data[i * nhead * s * s + h * s * s + j * s + t] *
                               v_reshaped.data[i * nhead * s * d_k + h * s * d_k + t * d_k + k];
                    }
                    attend.data[i * nhead * s * d_k + h * s * d_k + j * d_k + k] = sum;
                }
            }
        }
    }

    Tensor3D output = tensor_create(b, s, d_model);

    for (int i = 0; i < b; i++) {
        for (int j = 0; j < s; j++) {
            for (int h = 0; h < nhead; h++) {
                for (int k = 0; k < d_k; k++) {
                    output.data[i * s * d_model + j * d_model + h * d_k + k] =
                        attend.data[i * nhead * s * d_k + h * s * d_k + j * d_k + k];
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

Tensor3D feed_forward_forward(FeedForward *ffn, Tensor3D *x) {
    int b = x->batch_size;
    int s = x->seq_len;
    int d_model = x->d_model;
    int d_ff = ffn->d_model;

    Tensor3D hidden = tensor_create(b, s, d_ff);

    for (int i = 0; i < b * s * d_ff; i++) {
        hidden.data[i] = 0;
    }

    for (int bs = 0; bs < b; bs++) {
        for (int seq = 0; seq < s; seq++) {
            for (int j = 0; j < d_ff; j++) {
                float sum = ffn->b1->data[j];
                for (int k = 0; k < d_model; k++) {
                    sum += x->data[bs * s * d_model + seq * d_model + k] * ffn->w1->data[k * d_ff + j];
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
    
    Tensor3D attn_out = multi_head_attn_forward(layer->self_attn, x, x, x, mask);
    
    // Residual + LayerNorm
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
    
    Tensor3D ff_out = feed_forward_forward(layer->ffn, &residual);
    
    // Residual + LayerNorm
    for (int i = 0; i < residual.batch_size * residual.seq_len * residual.d_model; i++) {
        ff_out.data[i] += residual.data[i];
    }
    tensor_free(&residual);
    
    if (cache) {
        cache->ff_out = clone_tensor(&ff_out);
    }
    
    tensor_layer_norm(&ff_out, layer->ln2_gamma, layer->ln2_beta, 1e-5f);
    
    if (cache) {
        cache->ln2_out = clone_tensor(&ff_out);
    }

    return ff_out;
}

void encoder_layer_backward(
    EncoderLayer *layer,
    EncoderLayerCache *cache,
    Tensor3D *grad_output,
    Tensor3D *grad_input
) {
    // Backward through second layer norm
    Tensor3D grad_ln2 = tensor_create(grad_output->batch_size, grad_output->seq_len, grad_output->d_model);
    tensor_layer_norm_backward(&cache->ff_out, grad_output, layer->ln2_gamma, layer->ln2_beta, &grad_ln2, 1e-5f);
    
    // Backward through residual
    Tensor3D grad_ff_out = clone_tensor(&grad_ln2);
    for (int i = 0; i < grad_ln2.batch_size * grad_ln2.seq_len * grad_ln2.d_model; i++) {
        grad_ln2.data[i] += cache->ff_out.data[i];
    }
    tensor_free(&grad_ln2);
    
    // Backward through feed forward
    Tensor3D grad_ff_input = tensor_create(grad_ff_out.batch_size, grad_ff_out.seq_len, grad_ff_out.d_model);
    feed_forward_backward(layer->ffn, &cache->ln1_out, &grad_ff_out, &grad_ff_input);
    tensor_free(&grad_ff_out);
    
    // Backward through first layer norm
    Tensor3D grad_ln1 = tensor_create(grad_ff_input.batch_size, grad_ff_input.seq_len, grad_ff_input.d_model);
    tensor_layer_norm_backward(&cache->residual, &grad_ff_input, layer->ln1_gamma, layer->ln1_beta, &grad_ln1, 1e-5f);
    tensor_free(&grad_ff_input);
    
    // Backward through residual and attention
    Tensor3D grad_attn_out = clone_tensor(&grad_ln1);
    for (int i = 0; i < grad_ln1.batch_size * grad_ln1.seq_len * grad_ln1.d_model; i++) {
        grad_ln1.data[i] += cache->x.data[i];
    }
    tensor_free(&grad_ln1);
    
    // Backward through multi-head attention
    Tensor3D grad_query = tensor_create(grad_attn_out.batch_size, grad_attn_out.seq_len, grad_attn_out.d_model);
    Tensor3D grad_key = tensor_create(grad_attn_out.batch_size, grad_attn_out.seq_len, grad_attn_out.d_model);
    Tensor3D grad_value = tensor_create(grad_attn_out.batch_size, grad_attn_out.seq_len, grad_attn_out.d_model);
    multi_head_attn_backward(layer->self_attn, &cache->x, &cache->x, &cache->x, NULL, &grad_attn_out, &grad_query, &grad_key, &grad_value);
    tensor_free(&grad_attn_out);
    
    // Accumulate gradients to grad_input
    for (int i = 0; i < grad_input->batch_size * grad_input->seq_len * grad_input->d_model; i++) {
        grad_input->data[i] = grad_query.data[i]; // Simplified: only use grad_query
    }
    
    tensor_free(&grad_query);
    tensor_free(&grad_key);
    tensor_free(&grad_value);
}

void decoder_layer_backward(
    DecoderLayer *layer,
    DecoderLayerCache *cache,
    Tensor3D *grad_output,
    Tensor3D *grad_input
) {
    // Backward through third layer norm
    Tensor3D grad_ln3 = tensor_create(grad_output->batch_size, grad_output->seq_len, grad_output->d_model);
    tensor_layer_norm_backward(&cache->ff_out, grad_output, layer->ln3_gamma, layer->ln3_beta, &grad_ln3, 1e-5f);
    
    // Backward through residual and feed forward
    Tensor3D grad_ff_out = clone_tensor(&grad_ln3);
    for (int i = 0; i < grad_ln3.batch_size * grad_ln3.seq_len * grad_ln3.d_model; i++) {
        grad_ln3.data[i] += cache->ff_out.data[i];
    }
    tensor_free(&grad_ln3);
    
    Tensor3D grad_ff_input = tensor_create(grad_ff_out.batch_size, grad_ff_out.seq_len, grad_ff_out.d_model);
    feed_forward_backward(layer->ffn, &cache->ln2_out, &grad_ff_out, &grad_ff_input);
    tensor_free(&grad_ff_out);
    
    // Backward through second layer norm
    Tensor3D grad_ln2 = tensor_create(grad_ff_input.batch_size, grad_ff_input.seq_len, grad_ff_input.d_model);
    tensor_layer_norm_backward(&cache->residual2, &grad_ff_input, layer->ln2_gamma, layer->ln2_beta, &grad_ln2, 1e-5f);
    tensor_free(&grad_ff_input);
    
    // Backward through residual and cross-attention
    Tensor3D grad_cross_out = clone_tensor(&grad_ln2);
    for (int i = 0; i < grad_ln2.batch_size * grad_ln2.seq_len * grad_ln2.d_model; i++) {
        grad_ln2.data[i] += cache->residual2.data[i];
    }
    tensor_free(&grad_ln2);
    
    // Backward through cross-attention
    Tensor3D grad_cross_query = tensor_create(grad_cross_out.batch_size, grad_cross_out.seq_len, grad_cross_out.d_model);
    Tensor3D grad_cross_key = tensor_create(grad_cross_out.batch_size, grad_cross_out.seq_len, grad_cross_out.d_model);
    Tensor3D grad_cross_value = tensor_create(grad_cross_out.batch_size, grad_cross_out.seq_len, grad_cross_out.d_model);
    multi_head_attn_backward(layer->cross_attn, &cache->residual1, &cache->encoder_out, &cache->encoder_out, NULL, &grad_cross_out, &grad_cross_query, &grad_cross_key, &grad_cross_value);
    tensor_free(&grad_cross_out);
    
    // Backward through first layer norm
    Tensor3D grad_ln1 = tensor_create(grad_cross_query.batch_size, grad_cross_query.seq_len, grad_cross_query.d_model);
    tensor_layer_norm_backward(&cache->x, &grad_cross_query, layer->ln1_gamma, layer->ln1_beta, &grad_ln1, 1e-5f);
    tensor_free(&grad_cross_query);
    tensor_free(&grad_cross_key);
    tensor_free(&grad_cross_value);
    
    // Backward through residual and self-attention
    Tensor3D grad_self_out = clone_tensor(&grad_ln1);
    for (int i = 0; i < grad_ln1.batch_size * grad_ln1.seq_len * grad_ln1.d_model; i++) {
        grad_ln1.data[i] += cache->x.data[i];
    }
    tensor_free(&grad_ln1);
    
    // Backward through self-attention
    Tensor3D grad_query = tensor_create(grad_self_out.batch_size, grad_self_out.seq_len, grad_self_out.d_model);
    Tensor3D grad_key = tensor_create(grad_self_out.batch_size, grad_self_out.seq_len, grad_self_out.d_model);
    Tensor3D grad_value = tensor_create(grad_self_out.batch_size, grad_self_out.seq_len, grad_self_out.d_model);
    multi_head_attn_backward(layer->self_attn, &cache->x, &cache->x, &cache->x, NULL, &grad_self_out, &grad_query, &grad_key, &grad_value);
    tensor_free(&grad_self_out);
    
    // Accumulate to grad_input
    for (int i = 0; i < grad_input->batch_size * grad_input->seq_len * grad_input->d_model; i++) {
        grad_input->data[i] = grad_query.data[i]; // Simplified: only use self-attn grad
    }
    
    tensor_free(&grad_query);
    tensor_free(&grad_key);
    tensor_free(&grad_value);
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

    Tensor3D attn_out = multi_head_attn_forward(layer->self_attn, x, x, x, tgt_mask);

    // Residual + LayerNorm
    Tensor3D residual1 = clone_tensor(x);
    for (int i = 0; i < x->batch_size * x->seq_len * x->d_model; i++) {
        residual1.data[i] += attn_out.data[i];
    }
    tensor_free(&attn_out);
    
    if (cache) {
        cache->residual1 = clone_tensor(&residual1);
    }
    tensor_layer_norm(&residual1, layer->ln1_gamma, layer->ln1_beta, 1e-5f);

    Tensor3D cross_out = multi_head_attn_forward(layer->cross_attn, &residual1, encoder_out, encoder_out, src_mask);

    // Residual + LayerNorm
    Tensor3D residual2 = clone_tensor(&residual1);
    for (int i = 0; i < residual1.batch_size * residual1.seq_len * residual1.d_model; i++) {
        residual2.data[i] += cross_out.data[i];
    }
    tensor_free(&cross_out);
    
    if (cache) {
        cache->residual2 = clone_tensor(&residual2);
    }
    tensor_layer_norm(&residual2, layer->ln2_gamma, layer->ln2_beta, 1e-5f);

    Tensor3D ff_out = feed_forward_forward(layer->ffn, &residual2);

    // Residual + LayerNorm
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
    if (cache) {
        cache->src_proj = tensor_create(src->batch_size, src->seq_len, src->d_model);
        tensor_mat_mul(src, t->input_projection, &cache->src_proj);
    }
    Tensor3D src_proj = cache ? cache->src_proj : tensor_create(src->batch_size, src->seq_len, src->d_model);
    if (!cache) tensor_mat_mul(src, t->input_projection, &src_proj);

    Tensor3D encoder_out;
    if (cache) {
        encoder_out = encoder_forward(t->encoder, &src_proj, src_mask, &cache->encoder_cache);
    } else {
        encoder_out = encoder_forward(t->encoder, &src_proj, src_mask, NULL);
        tensor_free(&src_proj);
    }

    if (cache) {
        cache->tgt_proj = tensor_create(tgt->batch_size, tgt->seq_len, tgt->d_model);
        tensor_mat_mul(tgt, t->input_projection, &cache->tgt_proj);
    }
    Tensor3D tgt_proj = cache ? cache->tgt_proj : tensor_create(tgt->batch_size, tgt->seq_len, tgt->d_model);
    if (!cache) tensor_mat_mul(tgt, t->input_projection, &tgt_proj);

    Tensor3D decoder_out;
    if (cache) {
        decoder_out = decoder_forward(t->decoder, &tgt_proj, &encoder_out, src_mask, tgt_mask, &cache->decoder_cache);
    } else {
        decoder_out = decoder_forward(t->decoder, &tgt_proj, &encoder_out, src_mask, tgt_mask, NULL);
        tensor_free(&tgt_proj);
        tensor_free(&encoder_out);
    }

    Tensor3D out = tensor_create(decoder_out.batch_size, decoder_out.seq_len, t->config.d_model);
    tensor_mat_mul(&decoder_out, t->output_projection, &out);
    if (!cache) tensor_free(&decoder_out);

    return out;
}

void transformer_backward(
    Transformer *t,
    TransformerCache *cache,
    Tensor3D *grad,
    Tensor3D *tgt
) {
    // Backward through output projection
    Tensor3D grad_decoder_out = tensor_create(grad->batch_size, grad->seq_len, grad->d_model);
    tensor_zero(&grad_decoder_out);
    for (int i = 0; i < grad->batch_size * grad->seq_len * grad->d_model; i++) {
        grad_decoder_out.data[i] = grad->data[i];
    }

    // Backward through decoder
    Tensor3D grad_tgt_proj = tensor_create(tgt->batch_size, tgt->seq_len, tgt->d_model);
    tensor_zero(&grad_tgt_proj);
    
    for (int i = t->decoder->num_layers - 1; i >= 0; i--) {
        DecoderLayerCache *dlc = &cache->decoder_cache.layer_caches[i];
        Tensor3D grad_input = tensor_create(tgt->batch_size, tgt->seq_len, tgt->d_model);
        tensor_zero(&grad_input);
        decoder_layer_backward(&t->decoder->layers[i], dlc, &grad_decoder_out, &grad_input);
        tensor_free(&grad_decoder_out);
        grad_decoder_out = grad_input;
    }
    
    // Backward through encoder
    Tensor3D grad_encoder_out = clone_tensor(&cache->decoder_cache.encoder_out);
    
    for (int i = t->encoder->num_layers - 1; i >= 0; i--) {
        EncoderLayerCache *elc = &cache->encoder_cache.layer_caches[i];
        Tensor3D grad_input = tensor_create(tgt->batch_size, tgt->seq_len, tgt->d_model);
        tensor_zero(&grad_input);
        encoder_layer_backward(&t->encoder->layers[i], elc, &grad_encoder_out, &grad_input);
        tensor_free(&grad_encoder_out);
        grad_encoder_out = grad_input;
    }
    
    tensor_free(&grad_decoder_out);
    tensor_free(&grad_tgt_proj);
    tensor_free(&grad_encoder_out);
    tensor_free(&cache->src_proj);
    tensor_free(&cache->tgt_proj);
    tensor_free(&cache->decoder_cache.encoder_out);
    free(cache->encoder_cache.layer_caches);
    free(cache->decoder_cache.layer_caches);
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

// Training functions implementation
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
    int n = (axis == 0) ? output->d_model : output->seq_len;
    int m = (axis == 0) ? output->seq_len * output->d_model : output->batch_size * output->d_model;
    tensor_zero(grad_input);

    for (int i = 0; i < m; i++) {
        int offset = (axis == 0) ? i / output->seq_len * output->seq_len * output->d_model + i % output->seq_len * output->d_model
                                : i / output->d_model * output->seq_len * output->d_model + i % output->d_model;
        for (int j = 0; j < n; j++) {
            int idx_j = offset + ((axis == 0) ? j * output->d_model : j * output->d_model * output->d_model);
            for (int k = 0; k < n; k++) {
                int idx_k = offset + ((axis == 0) ? k * output->d_model : k * output->d_model * output->d_model);
                float s = (j == k) ? output->data[idx_j] * (1.0f - output->data[idx_j]) : -output->data[idx_j] * output->data[idx_k];
                grad_input->data[idx_k] += grad_output->data[idx_j] * s;
            }
        }
    }
}

void tensor_layer_norm_backward(Tensor3D *x, Tensor3D *grad_output, Matrix *gamma, Matrix *beta, Tensor3D *grad_input, float eps) {
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

            for (int k = 0; k < d; k++) {
                float dx = 0;
                for (int l = 0; l < d; l++) {
                    float dl_dxk = (k == l) ? 1.0f : 0.0f;
                    float dvar_dxk = 2.0f * (x->data[i * s * d + j * d + k] - mean) / d;
                    float dstd_dxk = -0.5f * powf(var + eps, -1.5f) * dvar_dxk;
                    float dnorm_dxk = gamma->data[k] * (dl_dxk - 1.0f/d) * std_inv + gamma->data[k] * (x->data[i * s * d + j * d + k] - mean) * dstd_dxk;
                    dx += grad_output->data[i * s * d + j * d + l] * dnorm_dxk;
                }
                grad_input->data[i * s * d + j * d + k] = dx;
            }
        }
    }
}

void mat_vec_mul_backward(Matrix *w, float *in, float *grad_out, int batch, int seq, int d_in, int d_out, Matrix *grad_w, float *grad_in) {
    for (int i = 0; i < batch; i++) {
        for (int s = 0; s < seq; s++) {
            for (int o = 0; o < d_out; o++) {
                for (int j = 0; j < d_in; j++) {
                    grad_w->data[o * d_in + j] += grad_out[i * seq * d_out + s * d_out + o] * in[i * seq * d_in + s * d_in + j];
                    grad_in[i * seq * d_in + s * d_in + j] += w->data[o * d_in + j] * grad_out[i * seq * d_out + s * d_out + o];
                }
            }
        }
    }
}

void multi_head_attn_backward(MultiHeadAttention *attn, Tensor3D *query, Tensor3D *key, Tensor3D *value, Tensor3D *attn_mask_unused, Tensor3D *grad_output, Tensor3D *grad_query, Tensor3D *grad_key, Tensor3D *grad_value) {
    (void)attn_mask_unused;
    int b = query->batch_size;
    int s = query->seq_len;
    int d_k = attn->d_k;
    int nhead = attn->nhead;
    int d_model = query->d_model;

    Tensor3D grad_q = tensor_create(b, s, d_model);
    Tensor3D grad_k = tensor_create(b, s, d_model);
    Tensor3D grad_v = tensor_create(b, s, d_model);
    Tensor3D grad_out_proj = tensor_create(b, s, d_model);
    tensor_zero(&grad_q);
    tensor_zero(&grad_k);
    tensor_zero(&grad_v);
    tensor_zero(&grad_out_proj);

    // Backward through output projection
    mat_vec_mul_backward(attn->wq_out, grad_output->data, grad_out_proj.data, b, s, d_model, d_model, attn->wq_out, grad_out_proj.data);

    // Reshape gradients
    Tensor3D grad_attend = tensor_create(b, nhead, s * d_k);
    tensor_zero(&grad_attend);
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < s; j++) {
            for (int h = 0; h < nhead; h++) {
                for (int k = 0; k < d_k; k++) {
                    grad_attend.data[i * nhead * s * d_k + h * s * d_k + j * d_k + k] =
                        grad_out_proj.data[i * s * d_model + j * d_model + h * d_k + k];
                }
            }
        }
    }

    // Backward through attention and reshape to q,k,v
    // (Simplified - full implementation would include softmax backward)
    Tensor3D grad_q_reshaped = tensor_create(b, nhead, s * d_k);
    Tensor3D grad_k_reshaped = tensor_create(b, nhead, s * d_k);
    Tensor3D grad_v_reshaped = tensor_create(b, nhead, s * d_k);
    tensor_zero(&grad_q_reshaped);
    tensor_zero(&grad_k_reshaped);
    tensor_zero(&grad_v_reshaped);

    // Reshape back to d_model
    for (int i = 0; i < b; i++) {
        for (int h = 0; h < nhead; h++) {
            for (int j = 0; j < s; j++) {
                for (int k = 0; k < d_k; k++) {
                    grad_q.data[i * s * d_model + j * d_model + h * d_k + k] =
                        grad_q_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + k];
                    grad_k.data[i * s * d_model + j * d_model + h * d_k + k] =
                        grad_k_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + k];
                    grad_v.data[i * s * d_model + j * d_model + h * d_k + k] =
                        grad_v_reshaped.data[i * nhead * s * d_k + h * s * d_k + j * d_k + k];
                }
            }
        }
    }

    // Backward through Q,K,V projections
    mat_vec_mul_backward(attn->wq, query->data, grad_q.data, b, s, d_model, d_model, attn->wq, grad_query->data);
    mat_vec_mul_backward(attn->wk, key->data, grad_k.data, b, s, d_model, d_model, attn->wk, grad_key->data);
    mat_vec_mul_backward(attn->wv, value->data, grad_v.data, b, s, d_model, d_model, attn->wv, grad_value->data);

    tensor_free(&grad_q);
    tensor_free(&grad_k);
    tensor_free(&grad_v);
    tensor_free(&grad_out_proj);
    tensor_free(&grad_attend);
    tensor_free(&grad_q_reshaped);
    tensor_free(&grad_k_reshaped);
    tensor_free(&grad_v_reshaped);
}

void feed_forward_backward(FeedForward *ffn, Tensor3D *x, Tensor3D *grad_output, Tensor3D *grad_input) {
    int b = x->batch_size;
    int s = x->seq_len;
    int d_model = x->d_model;
    int d_ff = ffn->d_ff;
    tensor_zero(grad_input);

    // Backward through second linear layer
    Matrix grad_w2 = matrix_create(d_ff, d_model);
    matrix_rand(&grad_w2);
    Tensor3D grad_hidden = tensor_create(b, s, d_ff);
    tensor_zero(&grad_hidden);

    for (int bs = 0; bs < b; bs++) {
        for (int seq = 0; seq < s; seq++) {
            for (int j = 0; j < d_model; j++) {
                for (int k = 0; k < d_ff; k++) {
                    grad_w2.data[k * d_model + j] += grad_output->data[bs * s * d_model + seq * d_model + j] * 
                        ffn->w2->data[k * d_model + j];
                    grad_hidden.data[bs * s * d_ff + seq * d_ff + k] += ffn->w2->data[k * d_model + j] * 
                        grad_output->data[bs * s * d_model + seq * d_model + j];
                }
            }
        }
    }

    // Backward through activation
    for (int i = 0; i < b * s * d_ff; i++) {
        int bs = i / (s * d_ff);
        int seq = (i % (s * d_ff)) / d_ff;
        int j = i % d_ff;
        float val = 0;
        for (int k = 0; k < d_model; k++) {
            val += x->data[bs * s * d_model + seq * d_model + k] * ffn->w1->data[k * d_ff + j];
        }
        val += ffn->b1->data[j];
        float act_grad = (ffn->activation == RELU) ? (val > 0 ? 1.0f : 0.0f) : 1.0f;
        grad_hidden.data[i] *= act_grad;
    }

    // Backward through first linear layer
    Matrix grad_w1 = matrix_create(d_model, d_ff);
    matrix_rand(&grad_w1);
    for (int bs = 0; bs < b; bs++) {
        for (int seq = 0; seq < s; seq++) {
            for (int j = 0; j < d_ff; j++) {
                for (int k = 0; k < d_model; k++) {
                    grad_w1.data[k * d_ff + j] += grad_hidden.data[bs * s * d_ff + seq * d_ff + j] * 
                        x->data[bs * s * d_model + seq * d_model + k];
                    grad_input->data[bs * s * d_model + seq * d_model + k] += ffn->w1->data[k * d_ff + j] * 
                        grad_hidden.data[bs * s * d_ff + seq * d_ff + j];
                }
            }
        }
    }

    matrix_free(&grad_w2);
    matrix_free(&grad_w1);
    tensor_free(&grad_hidden);
}

// Loss functions implementation
LossResult cross_entropy_loss(Tensor3D *pred, int *targets, int vocab_size_unused) {
    (void)vocab_size_unused;
    LossResult result;
    int batch = pred->batch_size;
    int seq = pred->seq_len;
    int d_model = pred->d_model;
    
    // Assume pred is (batch, seq, vocab_size) after projection
    // For simplicity, assume d_model == vocab_size
    result.loss = 0;
    result.grad = tensor_create(batch, seq, d_model);
    tensor_zero(&result.grad);
    
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            // Find max for numerical stability
            float max_val = -1e9f;
            for (int v = 0; v < d_model; v++) {
                int i = b * seq * d_model + s * d_model + v;
                if (pred->data[i] > max_val) max_val = pred->data[i];
            }
            
            // Compute softmax and loss
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
    result.loss /= (batch * seq); // Average loss
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

// Training implementation
Trainer trainer_create(Transformer *model, float lr, float beta1, float beta2, float eps) {
    Trainer trainer;
    trainer.model = model;
    trainer.optimizer = adam_create(lr, beta1, beta2, eps);
    trainer.learning_rate = lr;
    return trainer;
}

void trainer_free(Trainer *trainer) {
    (void)trainer;
}

void trainer_train_step(Trainer *trainer, Tensor3D *src, Tensor3D *tgt, int *targets, int vocab_size) {
    // Forward pass with cache
    TransformerCache cache = {0};
    Tensor3D output = transformer_forward(trainer->model, src, tgt, NULL, NULL, &cache);

    // Compute loss
    LossResult loss = cross_entropy_loss(&output, targets, vocab_size);
    loss_print(&loss, "train_step");

    // Backward pass
    Tensor3D grad = tensor_create(output.batch_size, output.seq_len, output.d_model);
    for (int i = 0; i < output.batch_size * output.seq_len * output.d_model; i++) {
        grad.data[i] = loss.grad.data[i];
    }
    
    Tensor3D grad_input = tensor_create(tgt->batch_size, tgt->seq_len, tgt->d_model);
    tensor_zero(&grad_input);
    transformer_backward(trainer->model, &cache, &grad, tgt);
    
    // Update model parameters using Adam
    // (Simplified: full implementation would update all parameters)
    
    // Cleanup
    tensor_free(&grad);
    tensor_free(&grad_input);
    tensor_free(&output);
    tensor_free(&loss.grad);
    tensor_free(&cache.src_proj);
    tensor_free(&cache.tgt_proj);
    for (int i = 0; i < cache.encoder_cache.num_layers; i++) {
        tensor_free(&cache.encoder_cache.layer_caches[i].x);
        tensor_free(&cache.encoder_cache.layer_caches[i].residual);
        tensor_free(&cache.encoder_cache.layer_caches[i].ln1_out);
        tensor_free(&cache.encoder_cache.layer_caches[i].ff_out);
        tensor_free(&cache.encoder_cache.layer_caches[i].ln2_out);
    }
    free(cache.encoder_cache.layer_caches);
    for (int i = 0; i < cache.decoder_cache.num_layers; i++) {
        tensor_free(&cache.decoder_cache.layer_caches[i].x);
        tensor_free(&cache.decoder_cache.layer_caches[i].residual1);
        tensor_free(&cache.decoder_cache.layer_caches[i].ln1_out);
        tensor_free(&cache.decoder_cache.layer_caches[i].residual2);
        tensor_free(&cache.decoder_cache.layer_caches[i].ln2_out);
        tensor_free(&cache.decoder_cache.layer_caches[i].ff_out);
        tensor_free(&cache.decoder_cache.layer_caches[i].ln3_out);
        tensor_free(&cache.decoder_cache.layer_caches[i].encoder_out);
    }
    free(cache.decoder_cache.layer_caches);
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
    
    // Save encoder layers
    for (int i = 0; i < t->encoder->num_layers; i++) {
        EncoderLayer *l = &t->encoder->layers[i];
        fwrite(l->self_attn->wq->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->self_attn->wk->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->self_attn->wv->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->self_attn->wq_out->data, sizeof(float), t->config.d_model * t->config.d_model, f);
        fwrite(l->ffn->w1->data, sizeof(float), t->config.d_model * l->ffn->d_ff, f);
        fwrite(l->ffn->w2->data, sizeof(float), l->ffn->d_ff * t->config.d_model, f);
    }
    
    // Save decoder layers
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
        fwrite(l->ffn->w2->data, sizeof(float), l->ffn->d_ff * t->config.d_model, f);
    }
    
    fclose(f);
}

Trainer trainer_load(const char *filename) {
    (void)filename;
    Trainer trainer = {0};
    return trainer;
}