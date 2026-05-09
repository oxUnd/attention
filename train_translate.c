#include "transformer.h"
#include "tokenizer.h"
#include "nn_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static const char *PAIRS[] = {
    "hello", "你好",
    "world", "世界",
    "good morning", "早上好",
    "good night", "晚安",
    "thank you", "谢谢",
    "you are welcome", "不客气",
    "how are you", "你好吗",
    "i am fine", "我很好",
    "what is your name", "你叫什么名字",
    "my name is tom", "我的名字是汤姆",
    "i love you", "我爱你",
    "happy birthday", "生日快乐",
    "see you later", "再见",
    "have a good day", "祝你有美好的一天",
    "nice to meet you", "很高兴见到你",
    NULL
};

typedef struct {
    int *en_ids;
    int en_len;
    int *zh_ids;
    int zh_len;
} TranslationPair;

typedef struct {
    TranslationPair *pairs;
    int num_pairs;
} TranslationCorpus;

static int load_translation_corpus(const Tokenizer *tok, TranslationCorpus *out) {
    int num_pairs = 0;
    for (int i = 0; PAIRS[i]; i += 2) num_pairs++;

    out->pairs = (TranslationPair *)calloc(num_pairs, sizeof(TranslationPair));
    if (!out->pairs) return -1;

    int idx = 0;
    for (int i = 0; PAIRS[i]; i += 2) {
        const char *en = PAIRS[i];
        const char *zh = PAIRS[i + 1];

        int en_cap = (int)strlen(en) + 8;
        int zh_cap = (int)strlen(zh) + 8;

        out->pairs[idx].en_ids = (int *)malloc(en_cap * sizeof(int));
        out->pairs[idx].zh_ids = (int *)malloc(zh_cap * sizeof(int));

        out->pairs[idx].en_len = tokenizer_encode(tok, en, out->pairs[idx].en_ids, en_cap);
        out->pairs[idx].zh_len = tokenizer_encode(tok, zh, out->pairs[idx].zh_ids, zh_cap);

        if (out->pairs[idx].en_len > 0 && out->pairs[idx].zh_len > 0) {
            idx++;
        } else {
            free(out->pairs[idx].en_ids);
            free(out->pairs[idx].zh_ids);
        }
    }

    out->num_pairs = idx;
    return (idx > 0) ? 0 : -1;
}

static void free_translation_corpus(TranslationCorpus *c) {
    if (!c) return;
    for (int i = 0; i < c->num_pairs; i++) {
        free(c->pairs[i].en_ids);
        free(c->pairs[i].zh_ids);
    }
    free(c->pairs);
    c->pairs = NULL;
    c->num_pairs = 0;
}

/* Concatenate every string in PAIRS (both English and Chinese) into a single
 * corpus buffer, used to build the utf8 tokenizer's vocabulary. The caller
 * owns the returned pointer and must free it. */
static char *build_pairs_corpus_string(void) {
    size_t total = 1;
    for (int i = 0; PAIRS[i]; i++) total += strlen(PAIRS[i]) + 1;
    char *buf = (char *)malloc(total);
    if (!buf) return NULL;
    size_t pos = 0;
    for (int i = 0; PAIRS[i]; i++) {
        size_t L = strlen(PAIRS[i]);
        memcpy(buf + pos, PAIRS[i], L);
        pos += L;
        buf[pos++] = ' ';
    }
    buf[pos] = 0;
    return buf;
}

typedef struct {
    int seq_len;
    int batch_size;
    int d_model;
    int d_ff;
    int num_heads;
    int encoder_layers;
    int decoder_layers;
    int max_epochs;
    float learning_rate;
    float dropout;
    int vocab_size;        /* 0 means: use tokenizer's vocab_size */
    int log_every;
    float convergence_loss;
} TranslateHyperparams;

static const TranslateHyperparams default_translate_hp = {
    .seq_len = 16,
    .batch_size = 4,
    .d_model = 64,
    .d_ff = 256,
    .num_heads = 4,
    .encoder_layers = 2,
    .decoder_layers = 2,
    .max_epochs = 500,
    .learning_rate = 0.005f,
    .dropout = 0.0f,
    .vocab_size = 0,
    .log_every = 20,
    .convergence_loss = 0.05f,
};

/* Fisher-Yates shuffle on a pre-allocated buffer. */
static void shuffle_indices(int *order, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }
}

static int translate_train(const TranslationCorpus *corpus, const TranslateHyperparams *hp,
                           Transformer **model_out, TrainingState **ts_out) {
    const int seq_len = hp->seq_len;
    const int batch_size = hp->batch_size;
    const int vocab_size = hp->vocab_size;

    TransformerConfig config = {
        .d_model = hp->d_model,
        .num_heads = hp->num_heads,
        .d_ff = hp->d_ff,
        .encoder_layers = hp->encoder_layers,
        .decoder_layers = hp->decoder_layers,
        .max_len = seq_len + 5,
        .dropout = hp->dropout,
        .activation = ACT_GELU,
        .vocab_size = vocab_size
    };

    Transformer *model = transformer_create(&config);
    TrainingState *ts = training_state_create(model, hp->learning_rate, 0.9f, 0.999f, 1e-8f);
    if (!model || !ts) {
        if (model) transformer_free(model);
        if (ts) training_state_free(ts);
        return -1;
    }

    /* Pre-allocate every per-batch buffer once. */
    Tensor3D causal = mask_causal_create(seq_len);
    int *src_ids = (int *)malloc((size_t)batch_size * seq_len * sizeof(int));
    int *tgt_ids = (int *)malloc((size_t)batch_size * seq_len * sizeof(int));
    int *targets = (int *)malloc((size_t)batch_size * seq_len * sizeof(int));
    int *src_valid_lens = (int *)malloc((size_t)batch_size * sizeof(int));
    int *order = (int *)malloc((size_t)corpus->num_pairs * sizeof(int));

    if (!src_ids || !tgt_ids || !targets || !src_valid_lens || !order) {
        free(src_ids); free(tgt_ids); free(targets);
        free(src_valid_lens); free(order);
        tensor_free(&causal);
        training_state_free(ts);
        transformer_free(model);
        return -1;
    }

    printf("Training: %d pairs, seq_len=%d, batch=%d, vocab=%d\n\n",
           corpus->num_pairs, seq_len, batch_size, vocab_size);

    clock_t total_t0 = clock();

    for (int epoch = 0; epoch < hp->max_epochs; epoch++) {
        clock_t epoch_t0 = clock();
        float sum_loss = 0.0f;
        int num_batches = 0;

        for (int i = 0; i < corpus->num_pairs; i++) order[i] = i;
        shuffle_indices(order, corpus->num_pairs);

        for (int b = 0; b < corpus->num_pairs; b += batch_size) {
            int cur_batch = (corpus->num_pairs - b < batch_size) ?
                            corpus->num_pairs - b : batch_size;

            for (int i = 0; i < cur_batch; i++) {
                const TranslationPair *pair = &corpus->pairs[order[b + i]];
                int en_clip = pair->en_len < seq_len ? pair->en_len : seq_len;
                src_valid_lens[i] = en_clip;

                int *src_row = src_ids + i * seq_len;
                int *tgt_row = tgt_ids + i * seq_len;
                int *out_row = targets + i * seq_len;

                for (int s = 0; s < seq_len; s++) {
                    src_row[s] = (s < en_clip) ? pair->en_ids[s] : TOK_PAD;
                }

                tgt_row[0] = TOK_BOS;
                out_row[0] = (pair->zh_len > 0) ? pair->zh_ids[0] : TOK_EOS;
                for (int s = 1; s < seq_len; s++) {
                    int prev = s - 1;
                    tgt_row[s] = (prev < pair->zh_len) ? pair->zh_ids[prev] : TOK_PAD;
                    out_row[s] = (s   < pair->zh_len) ? pair->zh_ids[s]   : TOK_EOS;
                }
            }

            /* Padding mask for the encoder so cross-attn ignores <pad>. */
            Tensor3D src_mask = mask_padding_create(cur_batch, seq_len, src_valid_lens);

            training_state_zero_grads(ts);

            TransformerCache cache = {0};
            Tensor3D output = transformer_forward_lm(model,
                                                     src_ids, seq_len,
                                                     tgt_ids, seq_len,
                                                     cur_batch,
                                                     &src_mask, &causal, &cache);

            LossResult loss = cross_entropy_loss(&output, targets, vocab_size);

            transformer_backward_lm(model, &cache, &loss.grad, src_ids, tgt_ids, ts);
            training_state_clip_grads(ts, 1.0f);
            training_state_update(ts);

            sum_loss += loss.loss;
            num_batches++;

            transformer_cache_free(&cache, config.encoder_layers, config.decoder_layers);
            tensor_free(&output);
            tensor_free(&loss.grad);
            tensor_free(&src_mask);
        }

        float avg_loss = sum_loss / (float)num_batches;
        int converged = avg_loss < hp->convergence_loss;
        if (epoch % hp->log_every == 0 || epoch == hp->max_epochs - 1 || converged) {
            double dt = (double)(clock() - epoch_t0) / CLOCKS_PER_SEC;
            printf("Epoch %4d: loss=%.4f  (%.2fs)\n", epoch, avg_loss, dt);
        }
        if (converged) {
            printf("Converged at epoch %d: loss=%.4f\n", epoch, avg_loss);
            break;
        }
    }

    double total_dt = (double)(clock() - total_t0) / CLOCKS_PER_SEC;
    printf("Total training time: %.2fs\n", total_dt);

    free(src_ids);
    free(tgt_ids);
    free(targets);
    free(src_valid_lens);
    free(order);
    tensor_free(&causal);

    *model_out = model;
    *ts_out = ts;
    return 0;
}

/* Sample with temperature. probs_buf must hold at least `size` floats. */
static int sample_logits(const float *logits, int size, float temperature, float *probs_buf) {
    if (temperature < 1e-4f) temperature = 1e-4f;
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        probs_buf[i] = expf((logits[i] - max_val) / temperature);
        sum += probs_buf[i];
    }
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        cumsum += probs_buf[i] * inv_sum;
        if (cumsum >= r) return i;
    }
    return size - 1;
}

/* Incremental, KV-cached generation. Each step is O(1) wrt past tokens. */
static void translate_generate(Transformer *model, const TranslateHyperparams *hp,
                               const Tokenizer *tok,
                               const char *en_text, int max_len, float temperature) {
    const int vocab_size = hp->vocab_size;
    const int seq_len    = hp->seq_len;

    int src_cap = (int)strlen(en_text) + 8;
    int *src_ids = (int *)malloc((size_t)src_cap * sizeof(int));
    int src_len = tokenizer_encode(tok, en_text, src_ids, src_cap);
    if (src_len <= 0) {
        free(src_ids);
        return;
    }
    if (src_len > seq_len) src_len = seq_len;

    int *tgt_buf = (int *)malloc((size_t)(max_len + 2) * sizeof(int));
    float *probs_buf = (float *)malloc((size_t)vocab_size * sizeof(float));
    int decode_cap = max_len + 4;
    TransformerKVCache *kv = transformer_kv_cache_create(model, decode_cap);
    if (!tgt_buf || !probs_buf || !kv) {
        free(src_ids); free(tgt_buf); free(probs_buf);
        if (kv) transformer_kv_cache_free(kv);
        return;
    }

    transformer_lm_init_cache(model, src_ids, src_len, kv);

    printf("EN: %s\nZH: ", en_text);

    int tgt_len = 0;
    int next = TOK_BOS;
    for (int step = 0; step < max_len; step++) {
        Tensor3D logits = transformer_lm_step(model, next, kv);
        int pred = sample_logits(logits.data, vocab_size, temperature, probs_buf);
        tensor_free(&logits);

        if (pred == TOK_EOS) break;
        tgt_buf[tgt_len++] = pred;
        next = pred;
    }

    char text_buf[4096];
    tokenizer_decode(tok, tgt_buf, tgt_len, text_buf, sizeof(text_buf));
    printf("%s\n\n", text_buf);

    free(src_ids);
    free(tgt_buf);
    free(probs_buf);
    transformer_kv_cache_free(kv);
}

static const char *kDemoPrompts[] = {
    "hello",
    "good morning",
    "thank you",
    "i love you",
    "what is your name",
    NULL
};

static void run_demo_pass(Transformer *model, const TranslateHyperparams *hp,
                          const Tokenizer *tok, const char *title, float temperature) {
    printf("=== %s ===\n\n", title);
    for (int i = 0; kDemoPrompts[i]; i++) {
        translate_generate(model, hp, tok, kDemoPrompts[i], 16, temperature);
    }
}

int main(void) {
    srand((unsigned)time(NULL));

    printf("=== English to Chinese Translation Demo ===\n\n");

    /* UTF-8 char-level tokenizer: each Chinese character / each ASCII char
     * occupies exactly one token, so the model generates one full character
     * per autoregressive step and can never emit a half multi-byte sequence. */
    char *corpus_str = build_pairs_corpus_string();
    if (!corpus_str) {
        fprintf(stderr, "Failed to build corpus string\n");
        return 1;
    }
    printf("Using UTF-8 char-level tokenizer (built from training corpus)...\n");
    Tokenizer *tok = tokenizer_create_utf8(corpus_str, 1024, 1);
    free(corpus_str);
    if (!tok) {
        fprintf(stderr, "Failed to create tokenizer\n");
        return 1;
    }
    printf("Vocab size: %d\n\n", tok->vocab_size);

    TranslationCorpus corpus = {0};
    if (load_translation_corpus(tok, &corpus) != 0) {
        fprintf(stderr, "Failed to load corpus\n");
        tokenizer_free(tok);
        return 1;
    }
    printf("Loaded %d translation pairs\n\n", corpus.num_pairs);

    TranslateHyperparams hp = default_translate_hp;
    hp.vocab_size = tok->vocab_size;

    Transformer *model = NULL;
    TrainingState *ts = NULL;
    if (translate_train(&corpus, &hp, &model, &ts) != 0) {
        fprintf(stderr, "Training failed\n");
        free_translation_corpus(&corpus);
        tokenizer_free(tok);
        return 1;
    }

    run_demo_pass(model, &hp, tok, "Translation Test (greedy)", 0.01f);
    run_demo_pass(model, &hp, tok, "Translation Test (sampling T=0.5)", 0.5f);

    free_translation_corpus(&corpus);
    if (ts) training_state_free(ts);
    if (model) transformer_free(model);
    tokenizer_free(tok);
    return 0;
}
