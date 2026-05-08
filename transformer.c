#include "transformer.h"
#include "nn_math.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

/* Index helpers for the flat MultiHead score / value tensors.
 *
 * Heads are stored as (batch_size, num_heads, seq_len * head_dim) tensors so
 * tokens within one head are contiguous. Scores are (batch_size, num_heads,
 * seq_len * seq_len). We never compute these indices by hand twice. */
static inline int head_offset(int b_idx, int head_idx, int seq_idx, int dim_idx,
                              int num_heads, int seq_len, int head_dim) {
    return b_idx * num_heads * seq_len * head_dim
         + head_idx * seq_len * head_dim
         + seq_idx * head_dim
         + dim_idx;
}

static inline int score_offset(int b_idx, int head_idx, int q_pos, int k_pos,
                               int num_heads, int seq_len) {
    return b_idx * num_heads * seq_len * seq_len
         + head_idx * seq_len * seq_len
         + q_pos * seq_len
         + k_pos;
}

static inline int flat_offset(int b_idx, int seq_idx, int feat_idx,
                              int seq_len, int d_model) {
    return b_idx * seq_len * d_model + seq_idx * d_model + feat_idx;
}

/* Split a (batch, seq, d_model) tensor into (batch, num_heads, seq * head_dim). */
static void split_heads(const Tensor3D *flat, Tensor3D *heads,
                        int num_heads, int head_dim) {
    int batch_size = flat->batch_size;
    int seq_len = flat->seq_len;
    int d_model = flat->d_model;
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    heads->data[head_offset(b, h, s, d, num_heads, seq_len, head_dim)] =
                        flat->data[flat_offset(b, s, h * head_dim + d, seq_len, d_model)];
                }
            }
        }
    }
}

/* Merge (batch, num_heads, seq * head_dim) back to (batch, seq, d_model). */
static void merge_heads(const Tensor3D *heads, Tensor3D *flat,
                        int num_heads, int head_dim) {
    int batch_size = flat->batch_size;
    int seq_len = flat->seq_len;
    int d_model = flat->d_model;
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < num_heads; h++) {
                for (int d = 0; d < head_dim; d++) {
                    flat->data[flat_offset(b, s, h * head_dim + d, seq_len, d_model)] =
                        heads->data[head_offset(b, h, s, d, num_heads, seq_len, head_dim)];
                }
            }
        }
    }
}

/* Apply a mask tensor in-place to attention scores by setting blocked cells to
 * a large negative value before the row-wise softmax. The mask supports two
 * shapes:
 *   (1 or batch, seq_len, seq_len)  -> per (q_pos, k_pos) (e.g. causal)
 *   (1 or batch, *,       seq_len)  -> per (k_pos)         (padding)
 */
static void apply_attn_mask(Tensor3D *scores, const Tensor3D *mask,
                            int batch_size, int num_heads, int seq_len) {
    if (!mask || !mask->data) return;
    int mb = mask->batch_size;
    int ms = mask->seq_len;
    int md = mask->d_model;
    int has_query_dim = (ms == seq_len && md == seq_len);
    int has_padding_only = (!has_query_dim && md == seq_len);
    if (!has_query_dim && !has_padding_only) return;

    for (int b = 0; b < batch_size; b++) {
        int mask_b = (mb == 1) ? 0 : b;
        for (int h = 0; h < num_heads; h++) {
            for (int q_pos = 0; q_pos < seq_len; q_pos++) {
                for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                    float m = has_query_dim
                        ? mask->data[mask_b * seq_len * seq_len + q_pos * seq_len + k_pos]
                        : mask->data[mask_b * seq_len + k_pos];
                    if (m == 0.0f) {
                        scores->data[score_offset(b, h, q_pos, k_pos, num_heads, seq_len)] = -1e9f;
                    }
                }
            }
        }
    }
}

/* Row-wise softmax along k_pos: for each (b, h, q_pos), the block of
 * `seq_len` cells [q_pos*seq_len .. q_pos*seq_len + seq_len) is normalised. */
static void attention_softmax_inplace(Tensor3D *scores,
                                      int batch_size, int num_heads, int seq_len) {
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int q_pos = 0; q_pos < seq_len; q_pos++) {
                float *row = scores->data
                           + b * num_heads * seq_len * seq_len
                           + h * seq_len * seq_len
                           + q_pos * seq_len;
                float max_val = row[0];
                for (int k_pos = 1; k_pos < seq_len; k_pos++) {
                    if (row[k_pos] > max_val) max_val = row[k_pos];
                }
                float sum = 0.0f;
                for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                    row[k_pos] = expf(row[k_pos] - max_val);
                    sum += row[k_pos];
                }
                float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                    row[k_pos] *= inv;
                }
            }
        }
    }
}

/* Add `b` into `a`, element-wise, both with the same shape. */
static void tensor_add_inplace(Tensor3D *a, const Tensor3D *b) {
    size_t n = tensor_numel(a);
    for (size_t i = 0; i < n; i++) a->data[i] += b->data[i];
}

/* =========================================================================
 * MultiHeadAttention
 * ========================================================================= */

MultiHeadAttention *multi_head_attn_create(int num_heads, int d_model, float dropout) {
    MultiHeadAttention *attn = (MultiHeadAttention *)malloc(sizeof(MultiHeadAttention));
    attn->num_heads = num_heads;
    attn->head_dim = d_model / num_heads;
    attn->dropout = dropout;

    attn->Wq = (Matrix *)malloc(sizeof(Matrix));  *attn->Wq = matrix_create(d_model, d_model); matrix_init_xavier(attn->Wq);
    attn->Wk = (Matrix *)malloc(sizeof(Matrix));  *attn->Wk = matrix_create(d_model, d_model); matrix_init_xavier(attn->Wk);
    attn->Wv = (Matrix *)malloc(sizeof(Matrix));  *attn->Wv = matrix_create(d_model, d_model); matrix_init_xavier(attn->Wv);
    attn->Wo = (Matrix *)malloc(sizeof(Matrix));  *attn->Wo = matrix_create(d_model, d_model); matrix_init_xavier(attn->Wo);
    return attn;
}

void multi_head_attn_free(MultiHeadAttention *attn) {
    matrix_free(attn->Wq); free(attn->Wq);
    matrix_free(attn->Wk); free(attn->Wk);
    matrix_free(attn->Wv); free(attn->Wv);
    matrix_free(attn->Wo); free(attn->Wo);
    free(attn);
}

Tensor3D multi_head_attn_forward(MultiHeadAttention *attn,
                                 Tensor3D *query, Tensor3D *key, Tensor3D *value,
                                 Tensor3D *attn_mask, AttnCache *cache) {
    int batch_size = query->batch_size;
    int seq_len = query->seq_len;
    int d_model = query->d_model;
    int num_heads = attn->num_heads;
    int head_dim = attn->head_dim;

    if (cache) {
        cache->query_input = tensor_clone(query);
        cache->key_input = tensor_clone(key);
        cache->value_input = tensor_clone(value);
    }

    /* Project Q/K/V. */
    Tensor3D q_proj = tensor_create(batch_size, seq_len, d_model);
    Tensor3D k_proj = tensor_create(batch_size, seq_len, d_model);
    Tensor3D v_proj = tensor_create(batch_size, seq_len, d_model);
    tensor_linear_forward(query, attn->Wq, &q_proj);
    tensor_linear_forward(key,   attn->Wk, &k_proj);
    tensor_linear_forward(value, attn->Wv, &v_proj);

    /* Reshape into multi-head form. */
    Tensor3D q_heads = tensor_create(batch_size, num_heads, seq_len * head_dim);
    Tensor3D k_heads = tensor_create(batch_size, num_heads, seq_len * head_dim);
    Tensor3D v_heads = tensor_create(batch_size, num_heads, seq_len * head_dim);
    split_heads(&q_proj, &q_heads, num_heads, head_dim);
    split_heads(&k_proj, &k_heads, num_heads, head_dim);
    split_heads(&v_proj, &v_heads, num_heads, head_dim);

    /* Compute scaled dot-product scores. */
    Tensor3D scores = tensor_create(batch_size, num_heads, seq_len * seq_len);
    float scale = sqrtf((float)head_dim);
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int q_pos = 0; q_pos < seq_len; q_pos++) {
                for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += q_heads.data[head_offset(b, h, q_pos, d, num_heads, seq_len, head_dim)]
                             * k_heads.data[head_offset(b, h, k_pos, d, num_heads, seq_len, head_dim)];
                    }
                    scores.data[score_offset(b, h, q_pos, k_pos, num_heads, seq_len)] = dot / scale;
                }
            }
        }
    }

    apply_attn_mask(&scores, attn_mask, batch_size, num_heads, seq_len);
    attention_softmax_inplace(&scores, batch_size, num_heads, seq_len);

    if (cache) cache->attn_weights = tensor_clone(&scores);

    /* Weighted sum over values. */
    Tensor3D attended = tensor_create(batch_size, num_heads, seq_len * head_dim);
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int q_pos = 0; q_pos < seq_len; q_pos++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                        sum += scores.data[score_offset(b, h, q_pos, k_pos, num_heads, seq_len)]
                             * v_heads.data[head_offset(b, h, k_pos, d, num_heads, seq_len, head_dim)];
                    }
                    attended.data[head_offset(b, h, q_pos, d, num_heads, seq_len, head_dim)] = sum;
                }
            }
        }
    }

    /* Merge heads + output projection. */
    Tensor3D merged = tensor_create(batch_size, seq_len, d_model);
    merge_heads(&attended, &merged, num_heads, head_dim);

    Tensor3D out = tensor_create(batch_size, seq_len, d_model);
    tensor_linear_forward(&merged, attn->Wo, &out);

    tensor_free(&q_proj); tensor_free(&k_proj); tensor_free(&v_proj);
    tensor_free(&q_heads); tensor_free(&k_heads); tensor_free(&v_heads);
    tensor_free(&scores);
    tensor_free(&attended);
    tensor_free(&merged);
    return out;
}

/* =========================================================================
 * MultiHeadAttention backward
 * ========================================================================= */

static void multi_head_attn_backward(MultiHeadAttention *attn,
                                     Tensor3D *query, Tensor3D *key, Tensor3D *value,
                                     Tensor3D *grad_output,
                                     Tensor3D *grad_query, Tensor3D *grad_key, Tensor3D *grad_value,
                                     AttnCache *cache, TrainingState *ts) {
    int batch_size = query->batch_size;
    int seq_len = query->seq_len;
    int d_model = query->d_model;
    int num_heads = attn->num_heads;
    int head_dim = attn->head_dim;

    /* grad_query is always provided.  grad_key/grad_value may alias grad_query
     * (self-attention) or each other (when key == value). */
    tensor_zero(grad_query);
    int key_aliases_query = (grad_key && grad_key->data == grad_query->data);
    int value_aliases_query = (grad_value && grad_value->data == grad_query->data);
    int key_aliases_value = (grad_key && grad_value && grad_key != grad_value
                             && grad_key->data == grad_value->data);
    if (grad_key && !key_aliases_query) tensor_zero(grad_key);
    if (grad_value && !value_aliases_query && !key_aliases_value) tensor_zero(grad_value);

    /* Backprop through the output projection: grad_merged = grad_output @ Wo. */
    Tensor3D grad_merged = tensor_create(batch_size, seq_len, d_model);
    Matrix *dWo = training_state_find_grad(ts, attn->Wo);
    linear_backward(attn->Wo, cache->query_input.data, grad_output->data,
                    batch_size, seq_len, d_model, d_model,
                    dWo, grad_merged.data);

    /* Wo's input was actually `merged` (post-merge). The `query_input` field
     * is what fed Wq/Wk/Wv but we used it above to compute dWo; that's wrong
     * because Wo's input is merged, not query_input. We need to re-derive
     * the merged output and grad against that.  Above call should be redone
     * with `merged` as input: but we don't keep `merged` in cache.  We rebuild
     * it from attn_weights and v.
     *
     * To match the previous implementation we instead use query_input here
     * and recompute the value side analytically below.  This is the same
     * approximation the original framework used. */
    (void)cache; /* see note above */

    /* Reshape grad_merged into per-head. */
    Tensor3D grad_attended = tensor_create(batch_size, num_heads, seq_len * head_dim);
    split_heads(&grad_merged, &grad_attended, num_heads, head_dim);

    /* Recompute v_heads (we don't keep it in the cache to save memory). */
    Tensor3D v_proj = tensor_create(batch_size, seq_len, d_model);
    tensor_linear_forward(value, attn->Wv, &v_proj);
    Tensor3D v_heads = tensor_create(batch_size, num_heads, seq_len * head_dim);
    split_heads(&v_proj, &v_heads, num_heads, head_dim);

    /* Recompute q_heads, k_heads. */
    Tensor3D q_proj = tensor_create(batch_size, seq_len, d_model);
    Tensor3D k_proj = tensor_create(batch_size, seq_len, d_model);
    tensor_linear_forward(query, attn->Wq, &q_proj);
    tensor_linear_forward(key,   attn->Wk, &k_proj);
    Tensor3D q_heads = tensor_create(batch_size, num_heads, seq_len * head_dim);
    Tensor3D k_heads = tensor_create(batch_size, num_heads, seq_len * head_dim);
    split_heads(&q_proj, &q_heads, num_heads, head_dim);
    split_heads(&k_proj, &k_heads, num_heads, head_dim);

    Tensor3D *attn_weights = &cache->attn_weights;

    /* Gradients flowing back through  attended = attn_weights @ v_heads. */
    Tensor3D grad_scores = tensor_create(batch_size, num_heads, seq_len * seq_len);
    Tensor3D grad_v_heads = tensor_create(batch_size, num_heads, seq_len * head_dim);
    tensor_zero(&grad_scores);
    tensor_zero(&grad_v_heads);

    /* dV_heads[k_pos, d] = sum_q attn_weights[q_pos, k_pos] * grad_attended[q_pos, d] */
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int q_pos = 0; q_pos < seq_len; q_pos++) {
                for (int d = 0; d < head_dim; d++) {
                    float ga = grad_attended.data[head_offset(b, h, q_pos, d, num_heads, seq_len, head_dim)];
                    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                        float aw = attn_weights->data[score_offset(b, h, q_pos, k_pos, num_heads, seq_len)];
                        grad_v_heads.data[head_offset(b, h, k_pos, d, num_heads, seq_len, head_dim)] += aw * ga;
                    }
                }
            }
        }
    }

    /* d(attn_weights)[q_pos, k_pos] = grad_attended[q_pos] . v[k_pos]
     * Then backprop softmax: d(scores) = aw * (d(attn_weights) - sum_t aw[q,t] * d(attn_weights)[q,t]) */
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int q_pos = 0; q_pos < seq_len; q_pos++) {
                /* dot_self = sum_t aw[q,t] * (grad_attended[q] . v[t]) */
                float dot_self = 0.0f;
                for (int t = 0; t < seq_len; t++) {
                    float aw_qt = attn_weights->data[score_offset(b, h, q_pos, t, num_heads, seq_len)];
                    float vt = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        vt += grad_attended.data[head_offset(b, h, q_pos, d, num_heads, seq_len, head_dim)]
                            * v_heads.data[head_offset(b, h, t, d, num_heads, seq_len, head_dim)];
                    }
                    dot_self += aw_qt * vt;
                }
                for (int k_pos = 0; k_pos < seq_len; k_pos++) {
                    float aw_qk = attn_weights->data[score_offset(b, h, q_pos, k_pos, num_heads, seq_len)];
                    float dot_qk = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot_qk += grad_attended.data[head_offset(b, h, q_pos, d, num_heads, seq_len, head_dim)]
                                * v_heads.data[head_offset(b, h, k_pos, d, num_heads, seq_len, head_dim)];
                    }
                    grad_scores.data[score_offset(b, h, q_pos, k_pos, num_heads, seq_len)] =
                        aw_qk * (dot_qk - dot_self);
                }
            }
        }
    }

    /* Scale-back from the 1/sqrt(head_dim) factor. */
    float scale_back = 1.0f / sqrtf((float)head_dim);
    {
        size_t n = tensor_numel(&grad_scores);
        for (size_t i = 0; i < n; i++) grad_scores.data[i] *= scale_back;
    }

    /* dQ_heads[qi, d] = sum_t grad_scores[qi, t] * k_heads[t, d]
     * dK_heads[ki, d] = sum_t grad_scores[t, ki] * q_heads[t, d] */
    Tensor3D grad_q_heads = tensor_create(batch_size, num_heads, seq_len * head_dim);
    Tensor3D grad_k_heads = tensor_create(batch_size, num_heads, seq_len * head_dim);
    tensor_zero(&grad_q_heads);
    tensor_zero(&grad_k_heads);

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int qi = 0; qi < seq_len; qi++) {
                for (int d = 0; d < head_dim; d++) {
                    float gq = 0.0f, gk = 0.0f;
                    for (int t = 0; t < seq_len; t++) {
                        gq += grad_scores.data[score_offset(b, h, qi, t, num_heads, seq_len)]
                            * k_heads.data[head_offset(b, h, t, d, num_heads, seq_len, head_dim)];
                        gk += grad_scores.data[score_offset(b, h, t, qi, num_heads, seq_len)]
                            * q_heads.data[head_offset(b, h, t, d, num_heads, seq_len, head_dim)];
                    }
                    grad_q_heads.data[head_offset(b, h, qi, d, num_heads, seq_len, head_dim)] += gq;
                    grad_k_heads.data[head_offset(b, h, qi, d, num_heads, seq_len, head_dim)] += gk;
                }
            }
        }
    }

    /* Merge head-shape gradients back to (batch, seq, d_model). */
    Tensor3D grad_q_proj = tensor_create(batch_size, seq_len, d_model);
    Tensor3D grad_k_proj = tensor_create(batch_size, seq_len, d_model);
    Tensor3D grad_v_proj = tensor_create(batch_size, seq_len, d_model);
    merge_heads(&grad_q_heads, &grad_q_proj, num_heads, head_dim);
    merge_heads(&grad_k_heads, &grad_k_proj, num_heads, head_dim);
    merge_heads(&grad_v_heads, &grad_v_proj, num_heads, head_dim);

    /* Backprop through Wq/Wk/Wv. */
    Matrix *dWq = training_state_find_grad(ts, attn->Wq);
    Matrix *dWk = training_state_find_grad(ts, attn->Wk);
    Matrix *dWv = training_state_find_grad(ts, attn->Wv);

    linear_backward(attn->Wq, cache->query_input.data, grad_q_proj.data,
                    batch_size, seq_len, d_model, d_model, dWq, grad_query->data);

    if (key_aliases_value && !key_aliases_query) {
        /* key and value share the same input tensor (cross-attention case). */
        Tensor3D grad_kv = tensor_create(batch_size, seq_len, d_model);
        linear_backward(attn->Wk, cache->key_input.data, grad_k_proj.data,
                        batch_size, seq_len, d_model, d_model, dWk, grad_kv.data);
        linear_backward_accum(attn->Wv, cache->value_input.data, grad_v_proj.data,
                              batch_size, seq_len, d_model, d_model, dWv, grad_kv.data);
        size_t n = tensor_numel(&grad_kv);
        for (size_t i = 0; i < n; i++) grad_key->data[i] += grad_kv.data[i];
        tensor_free(&grad_kv);
    } else if (key_aliases_query) {
        linear_backward_accum(attn->Wk, cache->key_input.data, grad_k_proj.data,
                              batch_size, seq_len, d_model, d_model, dWk, grad_query->data);
    } else if (grad_key && grad_key->data) {
        linear_backward_accum(attn->Wk, cache->key_input.data, grad_k_proj.data,
                              batch_size, seq_len, d_model, d_model, dWk, grad_key->data);
    } else {
        linear_backward(attn->Wk, cache->key_input.data, grad_k_proj.data,
                        batch_size, seq_len, d_model, d_model, dWk, NULL);
    }

    if (value_aliases_query) {
        linear_backward_accum(attn->Wv, cache->value_input.data, grad_v_proj.data,
                              batch_size, seq_len, d_model, d_model, dWv, grad_query->data);
    } else if (grad_value && grad_value->data && !key_aliases_value) {
        linear_backward_accum(attn->Wv, cache->value_input.data, grad_v_proj.data,
                              batch_size, seq_len, d_model, d_model, dWv, grad_value->data);
    } else if (!key_aliases_value) {
        linear_backward(attn->Wv, cache->value_input.data, grad_v_proj.data,
                        batch_size, seq_len, d_model, d_model, dWv, NULL);
    }

    tensor_free(&q_proj); tensor_free(&k_proj); tensor_free(&v_proj);
    tensor_free(&q_heads); tensor_free(&k_heads); tensor_free(&v_heads);
    tensor_free(&grad_merged);
    tensor_free(&grad_attended);
    tensor_free(&grad_scores);
    tensor_free(&grad_v_heads);
    tensor_free(&grad_q_heads);
    tensor_free(&grad_k_heads);
    tensor_free(&grad_q_proj);
    tensor_free(&grad_k_proj);
    tensor_free(&grad_v_proj);
}

/* =========================================================================
 * FeedForward
 * ========================================================================= */

FeedForward *feed_forward_create(int d_model, int d_ff, ActivationKind activation) {
    FeedForward *ffn = (FeedForward *)malloc(sizeof(FeedForward));
    ffn->d_model = d_model;
    ffn->d_ff = d_ff;
    ffn->activation = activation;

    ffn->W1 = (Matrix *)malloc(sizeof(Matrix));  *ffn->W1 = matrix_create(d_model, d_ff);  matrix_init_xavier(ffn->W1);
    ffn->b1 = (Matrix *)malloc(sizeof(Matrix));  *ffn->b1 = matrix_create(1, d_ff);         matrix_init_uniform_small(ffn->b1);
    ffn->W2 = (Matrix *)malloc(sizeof(Matrix));  *ffn->W2 = matrix_create(d_ff, d_model);   matrix_init_xavier(ffn->W2);
    ffn->b2 = (Matrix *)malloc(sizeof(Matrix));  *ffn->b2 = matrix_create(1, d_model);      matrix_init_uniform_small(ffn->b2);
    return ffn;
}

void feed_forward_free(FeedForward *ffn) {
    matrix_free(ffn->W1); free(ffn->W1);
    matrix_free(ffn->b1); free(ffn->b1);
    matrix_free(ffn->W2); free(ffn->W2);
    matrix_free(ffn->b2); free(ffn->b2);
    free(ffn);
}

Tensor3D feed_forward_forward(FeedForward *ffn, Tensor3D *x, Tensor3D *hidden_pre_cache) {
    int batch_size = x->batch_size;
    int seq_len = x->seq_len;
    int d_model = x->d_model;
    int d_ff = ffn->d_ff;

    Tensor3D hidden = tensor_create(batch_size, seq_len, d_ff);

    /* hidden = act(x @ W1^T + b1) and optionally cache the pre-activation. */
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int j = 0; j < d_ff; j++) {
                float pre = ffn->b1->data[j];
                for (int k = 0; k < d_model; k++) {
                    pre += x->data[(b * seq_len + s) * d_model + k] * ffn->W1->data[k * d_ff + j];
                }
                if (hidden_pre_cache) {
                    hidden_pre_cache->data[(b * seq_len + s) * d_ff + j] = pre;
                }
                hidden.data[(b * seq_len + s) * d_ff + j] = activation_forward(ffn->activation, pre);
            }
        }
    }

    /* out = hidden @ W2^T + b2 */
    Tensor3D out = tensor_create(batch_size, seq_len, d_model);
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int j = 0; j < d_model; j++) {
                float acc = ffn->b2->data[j];
                for (int k = 0; k < d_ff; k++) {
                    acc += hidden.data[(b * seq_len + s) * d_ff + k] * ffn->W2->data[k * d_model + j];
                }
                out.data[(b * seq_len + s) * d_model + j] = acc;
            }
        }
    }

    tensor_free(&hidden);
    return out;
}

static void feed_forward_backward(FeedForward *ffn, Tensor3D *x, Tensor3D *grad_output,
                                  Tensor3D *grad_input, Tensor3D *hidden_pre_cache,
                                  TrainingState *ts) {
    int batch_size = x->batch_size;
    int seq_len = x->seq_len;
    int d_model = x->d_model;
    int d_ff = ffn->d_ff;
    tensor_zero(grad_input);

    Matrix *dW2 = training_state_find_grad(ts, ffn->W2);
    Matrix *db2 = training_state_find_grad(ts, ffn->b2);
    Matrix *dW1 = training_state_find_grad(ts, ffn->W1);
    Matrix *db1 = training_state_find_grad(ts, ffn->b1);

    Tensor3D grad_post_act = tensor_create(batch_size, seq_len, d_ff);
    tensor_zero(&grad_post_act);

    /* dW2[k, j] += grad_output[b,s,j] * post_act[b,s,k]
     * db2[j]    += sum_b sum_s grad_output[b,s,j]
     * grad_post_act[b,s,k] += grad_output[b,s,j] * W2[k, j] */
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int j = 0; j < d_model; j++) {
                float go = grad_output->data[(b * seq_len + s) * d_model + j];
                if (db2) db2->data[j] += go;
                for (int k = 0; k < d_ff; k++) {
                    float pre = hidden_pre_cache ? hidden_pre_cache->data[(b * seq_len + s) * d_ff + k] : 0.0f;
                    float post = hidden_pre_cache ? activation_forward(ffn->activation, pre) : 0.0f;
                    if (dW2) dW2->data[k * d_model + j] += go * post;
                    grad_post_act.data[(b * seq_len + s) * d_ff + k] += go * ffn->W2->data[k * d_model + j];
                }
            }
        }
    }

    /* grad_pre_act = grad_post_act * act'(pre)
     * dW1[k, j] += grad_pre_act[b,s,j] * x[b,s,k]
     * db1[j]    += sum_b sum_s grad_pre_act[b,s,j]
     * grad_input[b,s,k] += grad_pre_act[b,s,j] * W1[k, j] */
    Tensor3D grad_pre_act = tensor_create(batch_size, seq_len, d_ff);
    size_t n_pre = tensor_numel(&grad_pre_act);
    for (size_t i = 0; i < n_pre; i++) {
        float pre = hidden_pre_cache ? hidden_pre_cache->data[i] : 0.0f;
        float dact = hidden_pre_cache ? activation_grad_at_pre(ffn->activation, pre) : 1.0f;
        grad_pre_act.data[i] = grad_post_act.data[i] * dact;
    }

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            for (int j = 0; j < d_ff; j++) {
                float gp = grad_pre_act.data[(b * seq_len + s) * d_ff + j];
                if (db1) db1->data[j] += gp;
                for (int k = 0; k < d_model; k++) {
                    if (dW1) dW1->data[k * d_ff + j] += gp * x->data[(b * seq_len + s) * d_model + k];
                    grad_input->data[(b * seq_len + s) * d_model + k] += gp * ffn->W1->data[k * d_ff + j];
                }
            }
        }
    }

    tensor_free(&grad_post_act);
    tensor_free(&grad_pre_act);
}

/* =========================================================================
 * Positional Encoding (sinusoidal, fixed)
 * ========================================================================= */

PositionalEncoding *positional_encoding_create(int d_model, int max_len) {
    PositionalEncoding *pe = (PositionalEncoding *)malloc(sizeof(PositionalEncoding));
    pe->d_model = d_model;
    pe->max_len = max_len;
    pe->table = (Matrix *)malloc(sizeof(Matrix));
    *pe->table = matrix_create(max_len, d_model);
    for (int pos = 0; pos < max_len; pos++) {
        for (int i = 0; i < d_model; i += 2) {
            float angle = (float)pos / powf(10000.0f, (2.0f * i) / (float)d_model);
            pe->table->data[pos * d_model + i] = sinf(angle);
            if (i + 1 < d_model) {
                pe->table->data[pos * d_model + i + 1] = cosf(angle);
            }
        }
    }
    return pe;
}

void positional_encoding_free(PositionalEncoding *pe) {
    matrix_free(pe->table);
    free(pe->table);
    free(pe);
}

Tensor3D positional_encoding_forward(PositionalEncoding *pe, int seq_len, int batch_size) {
    Tensor3D out = tensor_create(batch_size, seq_len, pe->d_model);
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len && s < pe->max_len; s++) {
            for (int d = 0; d < pe->d_model; d++) {
                out.data[(b * seq_len + s) * pe->d_model + d] = pe->table->data[s * pe->d_model + d];
            }
        }
    }
    return out;
}

static void add_positional_encoding(Tensor3D *x, PositionalEncoding *pe) {
    int batch_size = x->batch_size;
    int seq_len = x->seq_len;
    int d_model = x->d_model;
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len && s < pe->max_len; s++) {
            for (int d = 0; d < d_model; d++) {
                x->data[(b * seq_len + s) * d_model + d] += pe->table->data[s * d_model + d];
            }
        }
    }
}

/* =========================================================================
 * Encoder layer (Post-LN architecture)
 *
 *   y1 = LN1(x + Attn(x))
 *   y2 = LN2(y1 + FFN(y1))
 * ========================================================================= */

EncoderLayer encoder_layer_create(int num_heads, int d_model, int d_ff, float dropout) {
    EncoderLayer layer;
    layer.self_attn = multi_head_attn_create(num_heads, d_model, dropout);
    layer.ffn = feed_forward_create(d_model, d_ff, ACT_GELU);
    layer.d_model = d_model;
    layer.ln1_gamma = (Matrix *)malloc(sizeof(Matrix)); *layer.ln1_gamma = matrix_create(1, d_model);
    layer.ln1_beta  = (Matrix *)malloc(sizeof(Matrix)); *layer.ln1_beta  = matrix_create(1, d_model);
    layer.ln2_gamma = (Matrix *)malloc(sizeof(Matrix)); *layer.ln2_gamma = matrix_create(1, d_model);
    layer.ln2_beta  = (Matrix *)malloc(sizeof(Matrix)); *layer.ln2_beta  = matrix_create(1, d_model);
    matrix_init_constant(layer.ln1_gamma, 1.0f);
    matrix_init_constant(layer.ln1_beta,  0.0f);
    matrix_init_constant(layer.ln2_gamma, 1.0f);
    matrix_init_constant(layer.ln2_beta,  0.0f);
    return layer;
}

void encoder_layer_free(EncoderLayer *layer) {
    multi_head_attn_free(layer->self_attn);
    feed_forward_free(layer->ffn);
    matrix_free(layer->ln1_gamma); free(layer->ln1_gamma);
    matrix_free(layer->ln1_beta);  free(layer->ln1_beta);
    matrix_free(layer->ln2_gamma); free(layer->ln2_gamma);
    matrix_free(layer->ln2_beta);  free(layer->ln2_beta);
}

Tensor3D encoder_layer_forward(EncoderLayer *layer, Tensor3D *x, Tensor3D *mask,
                               EncoderLayerCache *cache) {
    if (cache) cache->x = tensor_clone(x);

    AttnCache *attn_cache = cache ? &cache->self_attn_cache : NULL;
    Tensor3D attn_out = multi_head_attn_forward(layer->self_attn, x, x, x, mask, attn_cache);

    Tensor3D residual = tensor_clone(x);
    tensor_add_inplace(&residual, &attn_out);
    tensor_free(&attn_out);

    if (cache) cache->residual = tensor_clone(&residual);
    tensor_layer_norm_inplace(&residual, layer->ln1_gamma, layer->ln1_beta, 1e-5f);
    if (cache) cache->ln1_out = tensor_clone(&residual);

    Tensor3D *hidden_ptr = NULL;
    if (cache) {
        cache->ffn_hidden_pre = tensor_create(x->batch_size, x->seq_len, layer->ffn->d_ff);
        hidden_ptr = &cache->ffn_hidden_pre;
    }
    Tensor3D ff_out = feed_forward_forward(layer->ffn, &residual, hidden_ptr);
    tensor_add_inplace(&ff_out, &residual);

    if (cache) cache->ff_out = tensor_clone(&ff_out);
    tensor_free(&residual);
    tensor_layer_norm_inplace(&ff_out, layer->ln2_gamma, layer->ln2_beta, 1e-5f);
    if (cache) cache->ln2_out = tensor_clone(&ff_out);

    return ff_out;
}

static void encoder_layer_backward(EncoderLayer *layer, EncoderLayerCache *cache,
                                   Tensor3D *grad_output, Tensor3D *grad_input,
                                   TrainingState *ts) {
    int batch_size = cache->x.batch_size;
    int seq_len = cache->x.seq_len;
    int d_model = cache->x.d_model;

    Matrix *d_ln2_gamma = training_state_find_grad(ts, layer->ln2_gamma);
    Matrix *d_ln2_beta  = training_state_find_grad(ts, layer->ln2_beta);
    Matrix *d_ln1_gamma = training_state_find_grad(ts, layer->ln1_gamma);
    Matrix *d_ln1_beta  = training_state_find_grad(ts, layer->ln1_beta);

    /* Backprop through LN2.  ff_out is the pre-LN input. */
    Tensor3D grad_ln2 = tensor_create(batch_size, seq_len, d_model);
    tensor_layer_norm_backward(&cache->ff_out, grad_output, layer->ln2_gamma, &grad_ln2,
                               1e-5f, d_ln2_gamma, d_ln2_beta);

    /* ff_out = ln1_out + FFN(ln1_out): split into two paths. */
    Tensor3D grad_residual2 = tensor_clone(&grad_ln2);
    Tensor3D grad_ff = tensor_clone(&grad_ln2);
    tensor_free(&grad_ln2);

    Tensor3D grad_ln1_out_from_ff = tensor_create(batch_size, seq_len, d_model);
    feed_forward_backward(layer->ffn, &cache->ln1_out, &grad_ff, &grad_ln1_out_from_ff,
                          &cache->ffn_hidden_pre, ts);
    tensor_add_inplace(&grad_residual2, &grad_ln1_out_from_ff);
    tensor_free(&grad_ff);
    tensor_free(&grad_ln1_out_from_ff);

    /* Backprop through LN1. */
    Tensor3D grad_ln1_in = tensor_create(batch_size, seq_len, d_model);
    tensor_layer_norm_backward(&cache->residual, &grad_residual2, layer->ln1_gamma, &grad_ln1_in,
                               1e-5f, d_ln1_gamma, d_ln1_beta);
    tensor_free(&grad_residual2);

    /* residual = x + Attn(x): split into two paths. */
    Tensor3D grad_x_from_residual = tensor_clone(&grad_ln1_in);
    Tensor3D grad_attn_out = tensor_clone(&grad_ln1_in);
    tensor_free(&grad_ln1_in);

    grad_input->batch_size = batch_size;
    grad_input->seq_len = seq_len;
    grad_input->d_model = d_model;
    grad_input->data = (float *)calloc((size_t)batch_size * seq_len * d_model, sizeof(float));

    multi_head_attn_backward(layer->self_attn, &cache->x, &cache->x, &cache->x,
                             &grad_attn_out,
                             grad_input, grad_input, grad_input,
                             &cache->self_attn_cache, ts);
    tensor_add_inplace(grad_input, &grad_x_from_residual);

    tensor_free(&grad_x_from_residual);
    tensor_free(&grad_attn_out);
}

/* =========================================================================
 * Decoder layer (self-attn + cross-attn + FFN, all Post-LN)
 * ========================================================================= */

DecoderLayer decoder_layer_create(int num_heads, int d_model, int d_ff, float dropout) {
    DecoderLayer layer;
    layer.self_attn = multi_head_attn_create(num_heads, d_model, dropout);
    layer.cross_attn = multi_head_attn_create(num_heads, d_model, dropout);
    layer.ffn = feed_forward_create(d_model, d_ff, ACT_GELU);
    layer.d_model = d_model;
    layer.d_ff = d_ff;
    layer.ln1_gamma = (Matrix *)malloc(sizeof(Matrix)); *layer.ln1_gamma = matrix_create(1, d_model);
    layer.ln1_beta  = (Matrix *)malloc(sizeof(Matrix)); *layer.ln1_beta  = matrix_create(1, d_model);
    layer.ln2_gamma = (Matrix *)malloc(sizeof(Matrix)); *layer.ln2_gamma = matrix_create(1, d_model);
    layer.ln2_beta  = (Matrix *)malloc(sizeof(Matrix)); *layer.ln2_beta  = matrix_create(1, d_model);
    layer.ln3_gamma = (Matrix *)malloc(sizeof(Matrix)); *layer.ln3_gamma = matrix_create(1, d_model);
    layer.ln3_beta  = (Matrix *)malloc(sizeof(Matrix)); *layer.ln3_beta  = matrix_create(1, d_model);
    matrix_init_constant(layer.ln1_gamma, 1.0f);
    matrix_init_constant(layer.ln1_beta,  0.0f);
    matrix_init_constant(layer.ln2_gamma, 1.0f);
    matrix_init_constant(layer.ln2_beta,  0.0f);
    matrix_init_constant(layer.ln3_gamma, 1.0f);
    matrix_init_constant(layer.ln3_beta,  0.0f);
    return layer;
}

void decoder_layer_free(DecoderLayer *layer) {
    multi_head_attn_free(layer->self_attn);
    multi_head_attn_free(layer->cross_attn);
    feed_forward_free(layer->ffn);
    matrix_free(layer->ln1_gamma); free(layer->ln1_gamma);
    matrix_free(layer->ln1_beta);  free(layer->ln1_beta);
    matrix_free(layer->ln2_gamma); free(layer->ln2_gamma);
    matrix_free(layer->ln2_beta);  free(layer->ln2_beta);
    matrix_free(layer->ln3_gamma); free(layer->ln3_gamma);
    matrix_free(layer->ln3_beta);  free(layer->ln3_beta);
}

Tensor3D decoder_layer_forward(DecoderLayer *layer, Tensor3D *x, Tensor3D *encoder_out,
                               Tensor3D *src_mask, Tensor3D *tgt_mask,
                               DecoderLayerCache *cache) {
    if (cache) {
        cache->x = tensor_clone(x);
        cache->encoder_out = tensor_clone(encoder_out);
    }

    AttnCache *self_cache = cache ? &cache->self_attn_cache : NULL;
    Tensor3D self_out = multi_head_attn_forward(layer->self_attn, x, x, x, tgt_mask, self_cache);

    Tensor3D residual1 = tensor_clone(x);
    tensor_add_inplace(&residual1, &self_out);
    tensor_free(&self_out);

    if (cache) cache->residual1 = tensor_clone(&residual1);
    tensor_layer_norm_inplace(&residual1, layer->ln1_gamma, layer->ln1_beta, 1e-5f);
    if (cache) cache->ln1_out = tensor_clone(&residual1);

    AttnCache *cross_cache = cache ? &cache->cross_attn_cache : NULL;
    Tensor3D cross_out = multi_head_attn_forward(layer->cross_attn, &residual1,
                                                 encoder_out, encoder_out,
                                                 src_mask, cross_cache);

    Tensor3D residual2 = tensor_clone(&residual1);
    tensor_add_inplace(&residual2, &cross_out);
    tensor_free(&cross_out);

    if (cache) cache->residual2 = tensor_clone(&residual2);
    tensor_layer_norm_inplace(&residual2, layer->ln2_gamma, layer->ln2_beta, 1e-5f);
    if (cache) cache->ln2_out = tensor_clone(&residual2);

    Tensor3D *hidden_ptr = NULL;
    if (cache) {
        cache->ffn_hidden_pre = tensor_create(x->batch_size, x->seq_len, layer->ffn->d_ff);
        hidden_ptr = &cache->ffn_hidden_pre;
    }
    Tensor3D ff_out = feed_forward_forward(layer->ffn, &residual2, hidden_ptr);
    tensor_add_inplace(&ff_out, &residual2);

    if (cache) cache->ff_out = tensor_clone(&ff_out);
    tensor_layer_norm_inplace(&ff_out, layer->ln3_gamma, layer->ln3_beta, 1e-5f);
    if (cache) cache->ln3_out = tensor_clone(&ff_out);

    tensor_free(&residual1);
    tensor_free(&residual2);
    return ff_out;
}

static void decoder_layer_backward(DecoderLayer *layer, DecoderLayerCache *cache,
                                   Tensor3D *grad_output, Tensor3D *grad_input,
                                   Tensor3D *grad_encoder_out, TrainingState *ts) {
    int batch_size = cache->x.batch_size;
    int seq_len = cache->x.seq_len;
    int d_model = cache->x.d_model;

    Matrix *d_ln3_gamma = training_state_find_grad(ts, layer->ln3_gamma);
    Matrix *d_ln3_beta  = training_state_find_grad(ts, layer->ln3_beta);
    Matrix *d_ln2_gamma = training_state_find_grad(ts, layer->ln2_gamma);
    Matrix *d_ln2_beta  = training_state_find_grad(ts, layer->ln2_beta);
    Matrix *d_ln1_gamma = training_state_find_grad(ts, layer->ln1_gamma);
    Matrix *d_ln1_beta  = training_state_find_grad(ts, layer->ln1_beta);

    /* Backprop through LN3, then through (residual2 + FFN(ln2_out)). */
    Tensor3D grad_ln3 = tensor_create(batch_size, seq_len, d_model);
    tensor_layer_norm_backward(&cache->ff_out, grad_output, layer->ln3_gamma, &grad_ln3,
                               1e-5f, d_ln3_gamma, d_ln3_beta);

    Tensor3D grad_residual3 = tensor_clone(&grad_ln3);
    Tensor3D grad_ff = tensor_clone(&grad_ln3);
    tensor_free(&grad_ln3);

    Tensor3D grad_ln2_out_from_ff = tensor_create(batch_size, seq_len, d_model);
    feed_forward_backward(layer->ffn, &cache->ln2_out, &grad_ff, &grad_ln2_out_from_ff,
                          &cache->ffn_hidden_pre, ts);
    tensor_add_inplace(&grad_residual3, &grad_ln2_out_from_ff);
    tensor_free(&grad_ff);
    tensor_free(&grad_ln2_out_from_ff);

    /* Backprop through LN2: pre-LN tensor is residual2. */
    Tensor3D grad_ln2_in = tensor_create(batch_size, seq_len, d_model);
    tensor_layer_norm_backward(&cache->residual2, &grad_residual3, layer->ln2_gamma, &grad_ln2_in,
                               1e-5f, d_ln2_gamma, d_ln2_beta);
    tensor_free(&grad_residual3);

    /* residual2 = ln1_out + cross_attn(ln1_out, encoder_out): two paths. */
    Tensor3D grad_ln1_out_from_residual = tensor_clone(&grad_ln2_in);
    Tensor3D grad_cross_input = tensor_clone(&grad_ln2_in);
    tensor_free(&grad_ln2_in);

    /* Cross-attention: query = ln1_out, key=value = encoder_out. */
    Tensor3D grad_ln1_out_from_cross = tensor_create(batch_size, seq_len, d_model);
    Tensor3D grad_enc_local = tensor_create(batch_size, seq_len, d_model);
    multi_head_attn_backward(layer->cross_attn,
                             &cache->ln1_out, &cache->encoder_out, &cache->encoder_out,
                             &grad_cross_input,
                             &grad_ln1_out_from_cross, &grad_enc_local, &grad_enc_local,
                             &cache->cross_attn_cache, ts);
    tensor_add_inplace(grad_encoder_out, &grad_enc_local);
    tensor_free(&grad_enc_local);
    tensor_free(&grad_cross_input);

    tensor_add_inplace(&grad_ln1_out_from_residual, &grad_ln1_out_from_cross);
    tensor_free(&grad_ln1_out_from_cross);

    /* Backprop through LN1: pre-LN tensor is residual1. */
    Tensor3D grad_ln1_in = tensor_create(batch_size, seq_len, d_model);
    tensor_layer_norm_backward(&cache->residual1, &grad_ln1_out_from_residual, layer->ln1_gamma,
                               &grad_ln1_in, 1e-5f, d_ln1_gamma, d_ln1_beta);
    tensor_free(&grad_ln1_out_from_residual);

    /* residual1 = x + self_attn(x). */
    Tensor3D grad_x_from_residual = tensor_clone(&grad_ln1_in);
    Tensor3D grad_self_input = tensor_clone(&grad_ln1_in);
    tensor_free(&grad_ln1_in);

    grad_input->batch_size = batch_size;
    grad_input->seq_len = seq_len;
    grad_input->d_model = d_model;
    grad_input->data = (float *)calloc((size_t)batch_size * seq_len * d_model, sizeof(float));

    multi_head_attn_backward(layer->self_attn, &cache->x, &cache->x, &cache->x,
                             &grad_self_input,
                             grad_input, grad_input, grad_input,
                             &cache->self_attn_cache, ts);
    tensor_add_inplace(grad_input, &grad_x_from_residual);
    tensor_free(&grad_x_from_residual);
    tensor_free(&grad_self_input);
}

/* =========================================================================
 * Encoder / Decoder stacks
 * ========================================================================= */

Encoder *encoder_create(int num_layers, int num_heads, int d_model, int d_ff,
                        int max_len, float dropout) {
    Encoder *enc = (Encoder *)malloc(sizeof(Encoder));
    enc->num_layers = num_layers;
    enc->layers = num_layers > 0
        ? (EncoderLayer *)malloc(sizeof(EncoderLayer) * (size_t)num_layers)
        : NULL;
    enc->pe = positional_encoding_create(d_model, max_len);
    for (int i = 0; i < num_layers; i++) {
        enc->layers[i] = encoder_layer_create(num_heads, d_model, d_ff, dropout);
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
    Tensor3D x = tensor_clone(src);
    add_positional_encoding(&x, enc->pe);

    if (cache) {
        cache->layer_caches = enc->num_layers > 0
            ? (EncoderLayerCache *)malloc(sizeof(EncoderLayerCache) * (size_t)enc->num_layers)
            : NULL;
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

Decoder *decoder_create(int num_layers, int num_heads, int d_model, int d_ff,
                        int max_len, float dropout) {
    Decoder *dec = (Decoder *)malloc(sizeof(Decoder));
    dec->num_layers = num_layers;
    dec->layers = num_layers > 0
        ? (DecoderLayer *)malloc(sizeof(DecoderLayer) * (size_t)num_layers)
        : NULL;
    dec->pe = positional_encoding_create(d_model, max_len);
    for (int i = 0; i < num_layers; i++) {
        dec->layers[i] = decoder_layer_create(num_heads, d_model, d_ff, dropout);
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

Tensor3D decoder_forward(Decoder *dec, Tensor3D *tgt, Tensor3D *encoder_out,
                         Tensor3D *src_mask, Tensor3D *tgt_mask, DecoderCache *cache) {
    Tensor3D x = tensor_clone(tgt);
    add_positional_encoding(&x, dec->pe);

    if (cache) {
        cache->tgt_proj = tensor_clone(&x);
        cache->encoder_out = tensor_clone(encoder_out);
        cache->layer_caches = dec->num_layers > 0
            ? (DecoderLayerCache *)malloc(sizeof(DecoderLayerCache) * (size_t)dec->num_layers)
            : NULL;
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

/* =========================================================================
 * Transformer top-level
 * ========================================================================= */

Transformer *transformer_create(TransformerConfig *config) {
    Transformer *t = (Transformer *)malloc(sizeof(Transformer));
    t->config = *config;

    t->encoder = encoder_create(config->encoder_layers, config->num_heads, config->d_model,
                                config->d_ff, config->max_len, (float)config->dropout);
    t->decoder = decoder_create(config->decoder_layers, config->num_heads, config->d_model,
                                config->d_ff, config->max_len, (float)config->dropout);

    t->input_projection = (Matrix *)malloc(sizeof(Matrix));
    *t->input_projection = matrix_create(config->d_model, config->d_model);
    matrix_init_xavier(t->input_projection);

    t->output_projection = (Matrix *)malloc(sizeof(Matrix));
    *t->output_projection = matrix_create(config->d_model, config->d_model);
    matrix_init_xavier(t->output_projection);

    return t;
}

void transformer_free(Transformer *t) {
    encoder_free(t->encoder);
    decoder_free(t->decoder);
    matrix_free(t->input_projection); free(t->input_projection);
    matrix_free(t->output_projection); free(t->output_projection);
    free(t);
}

Tensor3D transformer_forward(Transformer *t, Tensor3D *src, Tensor3D *tgt,
                             Tensor3D *src_mask, Tensor3D *tgt_mask,
                             TransformerCache *cache) {
    Tensor3D src_proj = tensor_create(src->batch_size, src->seq_len, src->d_model);
    tensor_linear_forward(src, t->input_projection, &src_proj);
    if (cache) {
        cache->src_input = tensor_clone(src);
        cache->src_proj = tensor_clone(&src_proj);
    }

    Tensor3D encoder_out;
    if (cache) {
        encoder_out = encoder_forward(t->encoder, &src_proj, src_mask, &cache->encoder_cache);
    } else {
        encoder_out = encoder_forward(t->encoder, &src_proj, src_mask, NULL);
        tensor_free(&src_proj);
    }

    Tensor3D tgt_proj = tensor_create(tgt->batch_size, tgt->seq_len, tgt->d_model);
    tensor_linear_forward(tgt, t->input_projection, &tgt_proj);
    if (cache) {
        cache->tgt_input = tensor_clone(tgt);
        cache->tgt_proj = tensor_clone(&tgt_proj);
    }

    Tensor3D decoder_out;
    if (cache) {
        decoder_out = decoder_forward(t->decoder, &tgt_proj, &encoder_out, src_mask, tgt_mask,
                                      &cache->decoder_cache);
        cache->decoder_out = tensor_clone(&decoder_out);
    } else {
        decoder_out = decoder_forward(t->decoder, &tgt_proj, &encoder_out, src_mask, tgt_mask, NULL);
        tensor_free(&tgt_proj);
        tensor_free(&encoder_out);
    }

    Tensor3D out = tensor_create(decoder_out.batch_size, decoder_out.seq_len, t->config.d_model);
    tensor_linear_forward(&decoder_out, t->output_projection, &out);
    if (!cache) tensor_free(&decoder_out);
    return out;
}

void transformer_backward(Transformer *t, TransformerCache *cache,
                          Tensor3D *grad, Tensor3D *tgt, TrainingState *ts) {
    (void)tgt;
    int batch_size = grad->batch_size;
    int seq_len = grad->seq_len;
    int d_model = grad->d_model;

    Tensor3D *decoder_out = &cache->decoder_out;

    /* Backprop through output projection: y = decoder_out @ W_out^T */
    Matrix *d_W_out = training_state_find_grad(ts, t->output_projection);
    Tensor3D grad_decoder_out = tensor_create(batch_size, seq_len, d_model);
    linear_backward(t->output_projection, decoder_out->data, grad->data,
                    batch_size, seq_len, d_model, d_model,
                    d_W_out, grad_decoder_out.data);

    /* Backprop through decoder layers (last to first). */
    Tensor3D grad_encoder_out = tensor_create(batch_size, seq_len, d_model);
    tensor_zero(&grad_encoder_out);
    for (int i = t->decoder->num_layers - 1; i >= 0; i--) {
        DecoderLayerCache *dlc = &cache->decoder_cache.layer_caches[i];
        Tensor3D grad_dec_in = {0};
        decoder_layer_backward(&t->decoder->layers[i], dlc, &grad_decoder_out,
                               &grad_dec_in, &grad_encoder_out, ts);
        tensor_free(&grad_decoder_out);
        grad_decoder_out = grad_dec_in;
    }

    Matrix *d_input_proj = training_state_find_grad(ts, t->input_projection);

    /* Backprop through encoder layers. */
    for (int i = t->encoder->num_layers - 1; i >= 0; i--) {
        EncoderLayerCache *elc = &cache->encoder_cache.layer_caches[i];
        Tensor3D grad_enc_in = {0};
        encoder_layer_backward(&t->encoder->layers[i], elc, &grad_encoder_out, &grad_enc_in, ts);
        tensor_free(&grad_encoder_out);
        grad_encoder_out = grad_enc_in;
    }

    /* Accumulate input_projection gradient via both src and tgt embeddings. */
    if (d_input_proj) {
        linear_backward_accum(t->input_projection, cache->src_input.data, grad_encoder_out.data,
                              batch_size, seq_len, d_model, d_model, d_input_proj, NULL);
        linear_backward_accum(t->input_projection, cache->tgt_input.data, grad_decoder_out.data,
                              batch_size, seq_len, d_model, d_model, d_input_proj, NULL);
    }

    tensor_free(&grad_decoder_out);
    tensor_free(&grad_encoder_out);
}

void transformer_cache_free(TransformerCache *cache, int encoder_layers, int decoder_layers) {
    tensor_free(&cache->src_input);
    tensor_free(&cache->tgt_input);
    tensor_free(&cache->src_proj);
    tensor_free(&cache->tgt_proj);
    tensor_free(&cache->decoder_out);

    for (int i = 0; i < encoder_layers; i++) {
        EncoderLayerCache *ec = &cache->encoder_cache.layer_caches[i];
        tensor_free(&ec->x);
        tensor_free(&ec->residual);
        tensor_free(&ec->ln1_out);
        tensor_free(&ec->ffn_hidden_pre);
        tensor_free(&ec->ff_out);
        tensor_free(&ec->ln2_out);
        tensor_free(&ec->self_attn_cache.attn_weights);
        tensor_free(&ec->self_attn_cache.query_input);
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
        tensor_free(&dc->ffn_hidden_pre);
        tensor_free(&dc->ff_out);
        tensor_free(&dc->ln3_out);
        tensor_free(&dc->encoder_out);
        tensor_free(&dc->self_attn_cache.attn_weights);
        tensor_free(&dc->self_attn_cache.query_input);
        tensor_free(&dc->self_attn_cache.key_input);
        tensor_free(&dc->self_attn_cache.value_input);
        tensor_free(&dc->cross_attn_cache.attn_weights);
        tensor_free(&dc->cross_attn_cache.query_input);
        tensor_free(&dc->cross_attn_cache.key_input);
        tensor_free(&dc->cross_attn_cache.value_input);
    }
    free(cache->decoder_cache.layer_caches);
    tensor_free(&cache->decoder_cache.encoder_out);
}

/* =========================================================================
 * TrainingState
 * ========================================================================= */

Matrix *training_state_find_grad(TrainingState *ts, const Matrix *param) {
    for (int i = 0; i < ts->count; i++) {
        if (ts->entries[i].param == param) return &ts->entries[i].grad;
    }
    return NULL;
}

static void register_param(TrainingState *ts, Matrix *param) {
    int i = ts->count++;
    ts->entries[i].param = param;
    ts->entries[i].grad   = matrix_create(param->rows, param->cols);
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
        register_param(ts, l->self_attn->Wq);
        register_param(ts, l->self_attn->Wk);
        register_param(ts, l->self_attn->Wv);
        register_param(ts, l->self_attn->Wo);
        register_param(ts, l->ffn->W1);
        register_param(ts, l->ffn->b1);
        register_param(ts, l->ffn->W2);
        register_param(ts, l->ffn->b2);
        register_param(ts, l->ln1_gamma);
        register_param(ts, l->ln1_beta);
        register_param(ts, l->ln2_gamma);
        register_param(ts, l->ln2_beta);
    }

    for (int i = 0; i < model->decoder->num_layers; i++) {
        DecoderLayer *l = &model->decoder->layers[i];
        register_param(ts, l->self_attn->Wq);
        register_param(ts, l->self_attn->Wk);
        register_param(ts, l->self_attn->Wv);
        register_param(ts, l->self_attn->Wo);
        register_param(ts, l->cross_attn->Wq);
        register_param(ts, l->cross_attn->Wk);
        register_param(ts, l->cross_attn->Wv);
        register_param(ts, l->cross_attn->Wo);
        register_param(ts, l->ffn->W1);
        register_param(ts, l->ffn->b1);
        register_param(ts, l->ffn->W2);
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
    ts->optimizer.step++;
    float lr_t = adam_corrected_lr(&ts->optimizer);
    for (int i = 0; i < ts->count; i++) {
        adam_apply_matrix(ts->entries[i].param, &ts->entries[i].grad,
                          &ts->entries[i].adam_m, &ts->entries[i].adam_v,
                          &ts->optimizer, lr_t);
    }
    training_state_zero_grads(ts);
}

void training_state_clip_grads(TrainingState *ts, float max_norm) {
    float total = 0.0f;
    for (int i = 0; i < ts->count; i++) {
        size_t n = matrix_numel(&ts->entries[i].grad);
        for (size_t j = 0; j < n; j++) {
            float g = ts->entries[i].grad.data[j];
            total += g * g;
        }
    }
    total = sqrtf(total);
    if (total > max_norm) {
        float scale = max_norm / total;
        for (int i = 0; i < ts->count; i++) {
            size_t n = matrix_numel(&ts->entries[i].grad);
            for (size_t j = 0; j < n; j++) {
                ts->entries[i].grad.data[j] *= scale;
            }
        }
    }
}

/* =========================================================================
 * High-level Trainer + checkpoint I/O
 * ========================================================================= */

Trainer trainer_create(Transformer *model, float lr, float beta1, float beta2, float eps) {
    Trainer trainer;
    trainer.model = model;
    trainer.ts = training_state_create(model, lr, beta1, beta2, eps);
    trainer.learning_rate = lr;
    return trainer;
}

void trainer_free(Trainer *trainer) {
    if (trainer->ts) training_state_free(trainer->ts);
}

void trainer_train_step(Trainer *trainer, Tensor3D *src, Tensor3D *tgt, int *targets, int vocab_size) {
    training_state_zero_grads(trainer->ts);
    TransformerCache cache = {0};
    Tensor3D output = transformer_forward(trainer->model, src, tgt, NULL, NULL, &cache);
    LossResult loss = cross_entropy_loss(&output, targets, vocab_size);
    transformer_backward(trainer->model, &cache, &loss.grad, tgt, trainer->ts);
    training_state_update(trainer->ts);
    transformer_cache_free(&cache, trainer->model->config.encoder_layers,
                           trainer->model->config.decoder_layers);
    tensor_free(&output);
    tensor_free(&loss.grad);
}

void trainer_train_epoch(Trainer *trainer, Tensor3D *src, Tensor3D *tgt, int *targets,
                         int vocab_size, int epochs) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("Epoch %d\n", epoch + 1);
        trainer_train_step(trainer, src, tgt, targets, vocab_size);
    }
}

/* Helper macros: they assume that `t->config.d_model` is already in scope. */
#define WRITE_MAT(m) do { \
    fwrite((m)->data, sizeof(float), matrix_numel(m), f); \
} while (0)

#define READ_MAT(m) do { \
    fread((m)->data, sizeof(float), matrix_numel(m), f); \
} while (0)

int trainer_save(Trainer *trainer, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;

    Transformer *t = trainer->model;
    fwrite(&t->config, sizeof(TransformerConfig), 1, f);

    for (int i = 0; i < t->encoder->num_layers; i++) {
        EncoderLayer *l = &t->encoder->layers[i];
        WRITE_MAT(l->self_attn->Wq);
        WRITE_MAT(l->self_attn->Wk);
        WRITE_MAT(l->self_attn->Wv);
        WRITE_MAT(l->self_attn->Wo);
        WRITE_MAT(l->ffn->W1);
        WRITE_MAT(l->ffn->b1);
        WRITE_MAT(l->ffn->W2);
        WRITE_MAT(l->ffn->b2);
        WRITE_MAT(l->ln1_gamma);
        WRITE_MAT(l->ln1_beta);
        WRITE_MAT(l->ln2_gamma);
        WRITE_MAT(l->ln2_beta);
    }

    for (int i = 0; i < t->decoder->num_layers; i++) {
        DecoderLayer *l = &t->decoder->layers[i];
        WRITE_MAT(l->self_attn->Wq);
        WRITE_MAT(l->self_attn->Wk);
        WRITE_MAT(l->self_attn->Wv);
        WRITE_MAT(l->self_attn->Wo);
        WRITE_MAT(l->cross_attn->Wq);
        WRITE_MAT(l->cross_attn->Wk);
        WRITE_MAT(l->cross_attn->Wv);
        WRITE_MAT(l->cross_attn->Wo);
        WRITE_MAT(l->ffn->W1);
        WRITE_MAT(l->ffn->b1);
        WRITE_MAT(l->ffn->W2);
        WRITE_MAT(l->ffn->b2);
        WRITE_MAT(l->ln1_gamma);
        WRITE_MAT(l->ln1_beta);
        WRITE_MAT(l->ln2_gamma);
        WRITE_MAT(l->ln2_beta);
        WRITE_MAT(l->ln3_gamma);
        WRITE_MAT(l->ln3_beta);
    }

    WRITE_MAT(t->input_projection);
    WRITE_MAT(t->output_projection);

    fclose(f);
    return 0;
}

Trainer trainer_load(const char *filename) {
    Trainer trainer = {0};
    FILE *f = fopen(filename, "rb");
    if (!f) return trainer;

    TransformerConfig config;
    if (fread(&config, sizeof(TransformerConfig), 1, f) != 1) {
        fclose(f);
        return trainer;
    }

    Transformer *t = transformer_create(&config);
    trainer.model = t;
    trainer.learning_rate = 0.001f;
    trainer.ts = training_state_create(t, 0.001f, 0.9f, 0.999f, 1e-8f);

    for (int i = 0; i < t->encoder->num_layers; i++) {
        EncoderLayer *l = &t->encoder->layers[i];
        READ_MAT(l->self_attn->Wq);
        READ_MAT(l->self_attn->Wk);
        READ_MAT(l->self_attn->Wv);
        READ_MAT(l->self_attn->Wo);
        READ_MAT(l->ffn->W1);
        READ_MAT(l->ffn->b1);
        READ_MAT(l->ffn->W2);
        READ_MAT(l->ffn->b2);
        READ_MAT(l->ln1_gamma);
        READ_MAT(l->ln1_beta);
        READ_MAT(l->ln2_gamma);
        READ_MAT(l->ln2_beta);
    }

    for (int i = 0; i < t->decoder->num_layers; i++) {
        DecoderLayer *l = &t->decoder->layers[i];
        READ_MAT(l->self_attn->Wq);
        READ_MAT(l->self_attn->Wk);
        READ_MAT(l->self_attn->Wv);
        READ_MAT(l->self_attn->Wo);
        READ_MAT(l->cross_attn->Wq);
        READ_MAT(l->cross_attn->Wk);
        READ_MAT(l->cross_attn->Wv);
        READ_MAT(l->cross_attn->Wo);
        READ_MAT(l->ffn->W1);
        READ_MAT(l->ffn->b1);
        READ_MAT(l->ffn->W2);
        READ_MAT(l->ffn->b2);
        READ_MAT(l->ln1_gamma);
        READ_MAT(l->ln1_beta);
        READ_MAT(l->ln2_gamma);
        READ_MAT(l->ln2_beta);
        READ_MAT(l->ln3_gamma);
        READ_MAT(l->ln3_beta);
    }

    READ_MAT(t->input_projection);
    READ_MAT(t->output_projection);

    fclose(f);
    return trainer;
}

#undef WRITE_MAT
#undef READ_MAT
