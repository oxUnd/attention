#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "nn_math.h"

/* Maximum number of parameter tensors we ever register with a TrainingState. */
#define TRANSFORMER_MAX_PARAMS 128

/* =========================================================================
 * Model configuration & top-level types
 * ========================================================================= */

typedef struct {
    int d_model;
    int num_heads;
    int d_ff;
    int encoder_layers;
    int decoder_layers;
    int max_len;
    double dropout;
    ActivationKind activation;
    /* When > 0 the model is in language-model mode: an explicit token
     * embedding table (vocab_size, d_model) and a logit head
     * (vocab_size, d_model) are allocated, and the legacy d_model-space
     * input/output projections are NOT registered for training. Forward
     * pass goes through transformer_forward_lm / transformer_backward_lm.
     *
     * When == 0 (default) the legacy d_model in / d_model out path is used
     * via transformer_forward / transformer_backward, preserving binary
     * compatibility with existing demos (train_simple, train_full, ...). */
    int vocab_size;
} TransformerConfig;

/* Multi-head self/cross attention.
 *
 *   Wq, Wk, Wv: (d_model, d_model) projections for query/key/value.
 *   Wo:         (d_model, d_model) output projection after head merge.
 *
 *   head_dim = d_model / num_heads.
 */
typedef struct {
    int num_heads;
    int head_dim;
    Matrix *Wq;
    Matrix *Wk;
    Matrix *Wv;
    Matrix *Wo;
    float dropout;
} MultiHeadAttention;

/* Position-wise feed-forward block:
 *   y = act(x @ W1^T + b1) @ W2^T + b2
 */
typedef struct {
    int d_model;
    int d_ff;
    Matrix *W1;
    Matrix *b1;
    Matrix *W2;
    Matrix *b2;
    ActivationKind activation;
} FeedForward;

typedef struct {
    int d_model;
    int max_len;
    Matrix *table; /* (max_len, d_model) sinusoidal table */
} PositionalEncoding;

/* =========================================================================
 * Encoder / Decoder layers
 * ========================================================================= */

typedef struct {
    MultiHeadAttention *self_attn;
    FeedForward *ffn;
    Matrix *ln1_gamma;
    Matrix *ln1_beta;
    Matrix *ln2_gamma;
    Matrix *ln2_beta;
    int d_model;
} EncoderLayer;

typedef struct {
    MultiHeadAttention *self_attn;
    MultiHeadAttention *cross_attn;
    FeedForward *ffn;
    Matrix *ln1_gamma;
    Matrix *ln1_beta;
    Matrix *ln2_gamma;
    Matrix *ln2_beta;
    Matrix *ln3_gamma;
    Matrix *ln3_beta;
    int d_model;
    int d_ff;
} DecoderLayer;

typedef struct {
    EncoderLayer *layers;
    int num_layers;
    PositionalEncoding *pe;
} Encoder;

typedef struct {
    DecoderLayer *layers;
    int num_layers;
    PositionalEncoding *pe;
} Decoder;

typedef struct {
    TransformerConfig config;
    Encoder *encoder;
    Decoder *decoder;
    /* Legacy d_model-space projections (used when config.vocab_size == 0). */
    Matrix *input_projection;
    Matrix *output_projection;
    /* LM mode (config.vocab_size > 0): token embedding table and logit head.
     * Both have shape (vocab_size, d_model); embedding lookup picks a row,
     * the logit head is applied as logits = decoder_out @ logit_head^T. */
    Matrix *token_embedding;
    Matrix *logit_head;
} Transformer;

/* =========================================================================
 * Activation / forward caches (used so backward() can avoid recomputation)
 * ========================================================================= */

typedef struct {
    Tensor3D attn_weights;
    Tensor3D query_input;
    Tensor3D key_input;
    Tensor3D value_input;
} AttnCache;

typedef struct {
    Tensor3D x;
    Tensor3D residual;
    Tensor3D ln1_out;
    Tensor3D ffn_hidden_pre;
    Tensor3D ff_out;
    Tensor3D ln2_out;
    AttnCache self_attn_cache;
} EncoderLayerCache;

typedef struct {
    EncoderLayerCache *layer_caches;
    int num_layers;
} EncoderCache;

typedef struct {
    Tensor3D x;
    Tensor3D residual1;
    Tensor3D ln1_out;
    Tensor3D residual2;
    Tensor3D ln2_out;
    Tensor3D ffn_hidden_pre;
    Tensor3D ff_out;
    Tensor3D ln3_out;
    Tensor3D encoder_out;
    AttnCache self_attn_cache;
    AttnCache cross_attn_cache;
} DecoderLayerCache;

typedef struct {
    DecoderLayerCache *layer_caches;
    int num_layers;
    Tensor3D tgt_proj;
    Tensor3D encoder_out;
} DecoderCache;

typedef struct {
    EncoderCache encoder_cache;
    DecoderCache decoder_cache;
    Tensor3D src_input;
    Tensor3D tgt_input;
    Tensor3D src_proj;
    Tensor3D tgt_proj;
    Tensor3D decoder_out;
} TransformerCache;

/* =========================================================================
 * Training state (Adam moments + parameter registry)
 * ========================================================================= */

typedef struct {
    Matrix *param;
    Matrix grad;
    Matrix adam_m;
    Matrix adam_v;
} ParamEntry;

typedef struct {
    ParamEntry entries[TRANSFORMER_MAX_PARAMS];
    int count;
    AdamOptimizer optimizer;
} TrainingState;

/* =========================================================================
 * Component lifecycle + forward APIs
 * ========================================================================= */

MultiHeadAttention *multi_head_attn_create(int num_heads, int d_model, float dropout);
void multi_head_attn_free(MultiHeadAttention *attn);
Tensor3D multi_head_attn_forward(MultiHeadAttention *attn,
                                 Tensor3D *query, Tensor3D *key, Tensor3D *value,
                                 Tensor3D *attn_mask, AttnCache *cache);

FeedForward *feed_forward_create(int d_model, int d_ff, ActivationKind activation);
void feed_forward_free(FeedForward *ffn);
Tensor3D feed_forward_forward(FeedForward *ffn, Tensor3D *x, Tensor3D *hidden_pre_cache);

PositionalEncoding *positional_encoding_create(int d_model, int max_len);
void positional_encoding_free(PositionalEncoding *pe);
Tensor3D positional_encoding_forward(PositionalEncoding *pe, int seq_len, int batch_size);

EncoderLayer encoder_layer_create(int num_heads, int d_model, int d_ff, float dropout);
void encoder_layer_free(EncoderLayer *layer);
Tensor3D encoder_layer_forward(EncoderLayer *layer, Tensor3D *x, Tensor3D *mask,
                               EncoderLayerCache *cache);

DecoderLayer decoder_layer_create(int num_heads, int d_model, int d_ff, float dropout);
void decoder_layer_free(DecoderLayer *layer);
Tensor3D decoder_layer_forward(DecoderLayer *layer, Tensor3D *x, Tensor3D *encoder_out,
                               Tensor3D *src_mask, Tensor3D *tgt_mask,
                               DecoderLayerCache *cache);

Encoder *encoder_create(int num_layers, int num_heads, int d_model, int d_ff,
                        int max_len, float dropout);
void encoder_free(Encoder *enc);
Tensor3D encoder_forward(Encoder *enc, Tensor3D *src, Tensor3D *src_mask, EncoderCache *cache);

Decoder *decoder_create(int num_layers, int num_heads, int d_model, int d_ff,
                        int max_len, float dropout);
void decoder_free(Decoder *dec);
Tensor3D decoder_forward(Decoder *dec, Tensor3D *tgt, Tensor3D *encoder_out,
                         Tensor3D *src_mask, Tensor3D *tgt_mask, DecoderCache *cache);

Transformer *transformer_create(TransformerConfig *config);
void transformer_free(Transformer *t);
Tensor3D transformer_forward(Transformer *t, Tensor3D *src, Tensor3D *tgt,
                             Tensor3D *src_mask, Tensor3D *tgt_mask,
                             TransformerCache *cache);
void transformer_backward(Transformer *t, TransformerCache *cache,
                          Tensor3D *grad, Tensor3D *tgt, TrainingState *ts);
void transformer_cache_free(TransformerCache *cache, int encoder_layers, int decoder_layers);

/* =========================================================================
 * Language-model forward/backward (token-id input, vocab logits output).
 *
 * Required: config.vocab_size > 0 (i.e. token_embedding / logit_head allocated).
 *
 * src_ids, tgt_ids: row-major arrays of shape (batch_size * src_len) and
 * (batch_size * tgt_len) respectively. Token IDs must be in [0, vocab_size).
 *
 * Output logits have shape (batch_size, tgt_len, vocab_size).
 *
 * The cache holds enough state to backprop through both the embedding
 * lookup and the logit head; the caller is responsible for keeping the
 * src_ids/tgt_ids arrays alive until backward returns.
 * ========================================================================= */
Tensor3D transformer_forward_lm(Transformer *t,
                                const int *src_ids, int src_len,
                                const int *tgt_ids, int tgt_len,
                                int batch_size,
                                Tensor3D *src_mask, Tensor3D *tgt_mask,
                                TransformerCache *cache);

void transformer_backward_lm(Transformer *t, TransformerCache *cache,
                             Tensor3D *grad_logits,
                             const int *src_ids, const int *tgt_ids,
                             TrainingState *ts);

/* =========================================================================
 * Incremental decoding with a per-layer KV cache.
 *
 * Workflow:
 *   1) cache = transformer_kv_cache_create(model, max_decode_len);
 *   2) transformer_lm_init_cache(model, src_ids, src_len, cache);
 *      - runs the encoder once
 *      - pre-projects encoder_out through every decoder layer's cross-attn
 *        Wk/Wv so the cross-attention K/V are reusable for the rest of the
 *        generation
 *   3) repeatedly call:
 *      logits = transformer_lm_step(model, next_token_id, cache);
 *      The cache appends fresh self-attn K/V on every call. cache->cur_len
 *      tracks how many tgt tokens have been pushed so far (also drives
 *      positional encoding offsets).
 *   4) transformer_kv_cache_free(cache);
 *
 * Only batch_size == 1 is supported by the cached path (we don't currently
 * generate in batched mode). The cache is purely an inference optimisation;
 * gradients never flow through it.
 * ========================================================================= */
typedef struct {
    /* Self-attn history per decoder layer, projected through Wk / Wv but
     * still in flat (cache_len, d_model) layout (split_heads is done on the
     * fly each step). Buffers are sized for max_len entries. */
    float *self_k;
    float *self_v;
    /* Cross-attn K / V, projected once from encoder_out and reused. */
    float *cross_k;
    float *cross_v;
} DecoderLayerKV;

typedef struct {
    int num_layers;
    int d_model;
    int max_len;        /* allocated capacity for self-attn cache */
    int cur_len;        /* tokens already pushed via lm_step */
    int src_len;        /* fixed once init_cache runs */
    DecoderLayerKV *layers;
} TransformerKVCache;

TransformerKVCache *transformer_kv_cache_create(const Transformer *t, int max_len);
void transformer_kv_cache_free(TransformerKVCache *cache);
/* Resets cur_len=0; cross-attn K/V are recomputed for the new src.
 * `src_mask` is forwarded to the encoder verbatim (NULL = no masking). It
 * MUST match the mask used at training/inference of the equivalent full
 * forward path, otherwise the encoder outputs - and therefore the cross-
 * attention K/V - will diverge from the no-cache path. */
void transformer_lm_init_cache(Transformer *t,
                               const int *src_ids, int src_len,
                               Tensor3D *src_mask,
                               TransformerKVCache *cache);
/* Returns logits of shape (1, 1, vocab_size). Caller owns the buffer and
 * must tensor_free it. */
Tensor3D transformer_lm_step(Transformer *t, int next_token,
                             TransformerKVCache *cache);

/* =========================================================================
 * Training state API
 * ========================================================================= */

TrainingState *training_state_create(Transformer *model, float lr, float beta1, float beta2, float eps);
void training_state_free(TrainingState *ts);
void training_state_zero_grads(TrainingState *ts);
void training_state_update(TrainingState *ts);
void training_state_clip_grads(TrainingState *ts, float max_norm);
Matrix *training_state_find_grad(TrainingState *ts, const Matrix *param);

/* =========================================================================
 * High-level Trainer (one step / one epoch helpers, save/load)
 * ========================================================================= */

typedef struct {
    Transformer *model;
    TrainingState *ts;
    float learning_rate;
} Trainer;

Trainer trainer_create(Transformer *model, float lr, float beta1, float beta2, float eps);
void trainer_free(Trainer *trainer);
void trainer_train_step(Trainer *trainer, Tensor3D *src, Tensor3D *tgt,
                        int *targets, int vocab_size);
void trainer_train_epoch(Trainer *trainer, Tensor3D *src, Tensor3D *tgt,
                         int *targets, int vocab_size, int epochs);
int trainer_save(Trainer *trainer, const char *filename);
Trainer trainer_load(const char *filename);

#endif /* TRANSFORMER_H */
