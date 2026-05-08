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
    Matrix *input_projection;
    Matrix *output_projection;
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
