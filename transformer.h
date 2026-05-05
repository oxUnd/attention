#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stddef.h>
#include <stdint.h>

#define MAX_SEQ_LEN 512
#define MAX_BATCH_SIZE 64
#define MAX_PARAMS 128

typedef enum {
    DIM_LAYER,
    RELU,
    GELU,
    GEOF,
    SILU,
} Activation;

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float eps;
    int t;
} AdamOptimizer;

typedef struct {
    int d_model;
    int nhead;
    int d_ff;
    int encoder_layers;
    int decoder_layers;
    int max_len;
    double dropout;
    Activation activation;
} TransformerConfig;

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

typedef struct {
    int nhead;
    int d_k;
    Matrix *wq;
    Matrix *wk;
    Matrix *wv;
    Matrix *wq_out;
    float dropout;
} MultiHeadAttention;

typedef struct {
    int d_model;
    int d_ff;
    Matrix *w1;
    Matrix *b1;
    Matrix *w2;
    Matrix *b2;
    Activation activation;
    Matrix *gamma;
    Matrix *beta;
    float eps;
} FeedForward;

typedef struct {
    int d_model;
    int max_len;
    Matrix *pe;
} PositionalEncoding;

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

typedef struct {
    Tensor3D attn_weights;
    Tensor3D input;
    Tensor3D key_input;
    Tensor3D value_input;
} AttnCache;

typedef struct {
    Tensor3D x;
    Tensor3D residual;
    Tensor3D ln1_out;
    Tensor3D ffn_hidden;
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
    Tensor3D ffn_hidden;
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

typedef struct {
    Matrix *param;
    Matrix grad;
    Matrix adam_m;
    Matrix adam_v;
} ParamEntry;

typedef struct {
    ParamEntry entries[MAX_PARAMS];
    int count;
    AdamOptimizer optimizer;
} TrainingState;

void tensor_zero(Tensor3D *t);
Tensor3D tensor_create(int batch_size, int seq_len, int d_model);
void tensor_free(Tensor3D *t);
Tensor3D clone_tensor(Tensor3D *t);
void tensor_softmax(Tensor3D *t, int axis);
void tensor_layer_norm(Tensor3D *x, Matrix *gamma, Matrix *beta, float eps);
void tensor_dropout(Tensor3D *t, float dropout_prob, int training);
void matrix_dropout(Matrix *m, int rows, int cols, float dropout_prob, int training);
void matrix_zero(Matrix *m);
void matrix_rand(Matrix *m);
void matrix_xavier(Matrix *m);

MultiHeadAttention *multi_head_attn_create(int nhead, int d_model, float dropout);
void multi_head_attn_free(MultiHeadAttention *attn);
Tensor3D multi_head_attn_forward(
    MultiHeadAttention *attn,
    Tensor3D *query,
    Tensor3D *key,
    Tensor3D *value,
    Tensor3D *attn_mask,
    AttnCache *cache
);

FeedForward *feed_forward_create(int d_model, int d_ff, Activation activation);
void feed_forward_free(FeedForward *ffn);
Tensor3D feed_forward_forward(FeedForward *ffn, Tensor3D *x, Tensor3D *hidden_cache);

PositionalEncoding *positional_encoding_create(int d_model, int max_len, float dropout);
void positional_encoding_free(PositionalEncoding *pe);
Tensor3D positional_encoding_forward(PositionalEncoding *pe, int seq_len, int batch_size);

EncoderLayer encoder_layer_create(int nhead, int d_model, int d_ff, float dropout);
void encoder_layer_free(EncoderLayer *layer);
Tensor3D encoder_layer_forward(EncoderLayer *layer, Tensor3D *x, Tensor3D *mask, EncoderLayerCache *cache);

DecoderLayer decoder_layer_create(int nhead, int d_model, int d_ff, float dropout);
void decoder_layer_free(DecoderLayer *layer);
Tensor3D decoder_layer_forward(DecoderLayer *layer, Tensor3D *x, Tensor3D *encoder_out, Tensor3D *src_mask, Tensor3D *tgt_mask, DecoderLayerCache *cache);

Encoder *encoder_create(int num_layers, int nhead, int d_model, int d_ff, int max_len, float dropout);
void encoder_free(Encoder *enc);
Tensor3D encoder_forward(Encoder *enc, Tensor3D *src, Tensor3D *src_mask, EncoderCache *cache);

Decoder *decoder_create(int num_layers, int nhead, int d_model, int d_ff, int max_len, float dropout);
void decoder_free(Decoder *dec);
Tensor3D decoder_forward(Decoder *dec, Tensor3D *tgt, Tensor3D *encoder_out, Tensor3D *src_mask, Tensor3D *tgt_mask, DecoderCache *cache);

Transformer *transformer_create(TransformerConfig *config);
void transformer_free(Transformer *t);
Tensor3D transformer_forward(Transformer *t, Tensor3D *src, Tensor3D *tgt, Tensor3D *src_mask, Tensor3D *tgt_mask, TransformerCache *cache);
void transformer_backward(Transformer *t, TransformerCache *cache, Tensor3D *grad, Tensor3D *tgt, TrainingState *ts);
void transformer_cache_free(TransformerCache *cache, int encoder_layers, int decoder_layers);

void tensor_print(Tensor3D *t, const char *name);

AdamOptimizer adam_create(float lr, float beta1, float beta2, float eps);
void adam_update_matrix(Matrix *m, Matrix *grad, Matrix *m_moment, Matrix *v_moment, AdamOptimizer *opt);
void adam_update_tensor(Tensor3D *t, Tensor3D *grad, Tensor3D *m_moment, Tensor3D *v_moment, AdamOptimizer *opt);

void tensor_softmax_backward(Tensor3D *output, Tensor3D *grad_output, Tensor3D *grad_input, int axis);
void tensor_layer_norm_backward(Tensor3D *x, Tensor3D *grad_output, Matrix *gamma, Matrix *beta, Tensor3D *grad_input, float eps, Matrix *grad_gamma, Matrix *grad_beta);
void mat_vec_mul_backward(Matrix *w, float *in, float *grad_out, int batch, int seq, int d_in, int d_out, Matrix *grad_w, float *grad_in);
void multi_head_attn_backward(MultiHeadAttention *attn, Tensor3D *query, Tensor3D *key, Tensor3D *value, Tensor3D *attn_mask_unused, Tensor3D *grad_output, Tensor3D *grad_query, Tensor3D *grad_key, Tensor3D *grad_value, AttnCache *attn_cache, TrainingState *ts);
void feed_forward_backward(FeedForward *ffn, Tensor3D *x, Tensor3D *grad_output, Tensor3D *grad_input, Tensor3D *hidden_cache, TrainingState *ts);

typedef struct {
    float loss;
    Tensor3D grad;
} LossResult;

LossResult cross_entropy_loss(Tensor3D *pred, int *targets, int vocab_size);
LossResult mse_loss(Tensor3D *pred, Tensor3D *target);
void loss_print(LossResult *loss, const char *name);

TrainingState *training_state_create(Transformer *model, float lr, float beta1, float beta2, float eps);
void training_state_free(TrainingState *ts);
void training_state_zero_grads(TrainingState *ts);
void training_state_update(TrainingState *ts);
void training_state_clip_grads(TrainingState *ts, float max_norm);
Matrix *find_grad(TrainingState *ts, Matrix *param);

typedef struct {
    Transformer *model;
    TrainingState *ts;
    float learning_rate;
} Trainer;

Trainer trainer_create(Transformer *model, float lr, float beta1, float beta2, float eps);
void trainer_free(Trainer *trainer);
void trainer_train_step(Trainer *trainer, Tensor3D *src, Tensor3D *tgt, int *targets, int vocab_size);
void trainer_train_epoch(Trainer *trainer, Tensor3D *src, Tensor3D *tgt, int *targets, int vocab_size, int epochs);
int trainer_save(Trainer *trainer, const char *filename);
Trainer trainer_load(const char *filename);

#endif