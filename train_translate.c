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

        int en_cap = strlen(en) + 8;
        int zh_cap = strlen(zh) + 8;

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
    int vocab_size;
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
    .vocab_size = 260
};

static int translate_train(const TranslationCorpus *corpus, const TranslateHyperparams *hp,
                           Transformer **model_out, TrainingState **ts_out) {
    int seq_len = hp->seq_len;
    int batch_size = hp->batch_size;
    int vocab_size = hp->vocab_size;

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

    Tensor3D causal = mask_causal_create(seq_len);
    int *src_ids = (int *)malloc(batch_size * seq_len * sizeof(int));
    int *tgt_ids = (int *)malloc(batch_size * seq_len * sizeof(int));
    int *targets = (int *)malloc(batch_size * seq_len * sizeof(int));

    if (!src_ids || !tgt_ids || !targets) {
        free(src_ids); free(tgt_ids); free(targets);
        tensor_free(&causal);
        training_state_free(ts);
        transformer_free(model);
        return -1;
    }

    printf("Training: %d pairs, seq_len=%d, batch=%d, vocab=%d\n\n",
           corpus->num_pairs, seq_len, batch_size, vocab_size);

    for (int epoch = 0; epoch < hp->max_epochs; epoch++) {
        float sum_loss = 0.0f;

        int *order = (int *)malloc(corpus->num_pairs * sizeof(int));
        for (int i = 0; i < corpus->num_pairs; i++) order[i] = i;
        for (int i = corpus->num_pairs - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int t = order[i]; order[i] = order[j]; order[j] = t;
        }

        int num_batches = 0;
        for (int b = 0; b < corpus->num_pairs; b += batch_size) {
            int cur_batch = (corpus->num_pairs - b < batch_size) ?
                           corpus->num_pairs - b : batch_size;

            for (int i = 0; i < cur_batch; i++) {
                const TranslationPair *pair = &corpus->pairs[order[b + i]];
                for (int s = 0; s < seq_len; s++) {
                    src_ids[i * seq_len + s] = (s < pair->en_len) ? pair->en_ids[s] : TOK_PAD;
                    if (s == 0) {
                        tgt_ids[i * seq_len + s] = TOK_BOS;
                        targets[i * seq_len + s] = (0 < pair->zh_len) ? pair->zh_ids[0] : TOK_EOS;
                    } else if (s - 1 < pair->zh_len) {
                        tgt_ids[i * seq_len + s] = pair->zh_ids[s - 1];
                        targets[i * seq_len + s] = (s < pair->zh_len) ? pair->zh_ids[s] : TOK_EOS;
                    } else {
                        tgt_ids[i * seq_len + s] = TOK_PAD;
                        targets[i * seq_len + s] = TOK_EOS;
                    }
                }
            }

            training_state_zero_grads(ts);

            TransformerCache cache = {0};
            Tensor3D output = transformer_forward_lm(model,
                                                     src_ids, seq_len,
                                                     tgt_ids, seq_len,
                                                     cur_batch,
                                                     NULL, &causal, &cache);

            LossResult loss = cross_entropy_loss(&output, targets, vocab_size);

            transformer_backward_lm(model, &cache, &loss.grad, src_ids, tgt_ids, ts);
            training_state_clip_grads(ts, 1.0f);
            training_state_update(ts);

            sum_loss += loss.loss;
            num_batches++;

            transformer_cache_free(&cache, config.encoder_layers, config.decoder_layers);
            tensor_free(&output);
            tensor_free(&loss.grad);
        }

        free(order);

        float avg_loss = sum_loss / (float)num_batches;
        if (epoch % 20 == 0 || epoch == hp->max_epochs - 1 || avg_loss < 0.5f) {
            printf("Epoch %4d: loss=%.4f\n", epoch, avg_loss);
        }

        if (avg_loss < 0.3f) {
            printf("Converged at epoch %d: loss=%.4f\n", epoch, avg_loss);
            break;
        }
    }

    free(src_ids);
    free(tgt_ids);
    free(targets);
    tensor_free(&causal);

    *model_out = model;
    *ts_out = ts;
    return 0;
}

static int sample_logits(const float *logits, int size, float temperature) {
    if (temperature < 1e-4f) temperature = 1e-4f;
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float *probs = (float *)malloc((size_t)size * sizeof(float));
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        probs[i] = expf((logits[i] - max_val) / temperature);
        sum += probs[i];
    }
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    int chosen = size - 1;
    for (int i = 0; i < size; i++) {
        cumsum += probs[i] / sum;
        if (cumsum >= r) { chosen = i; break; }
    }
    free(probs);
    return chosen;
}

static void translate_generate(Transformer *model, const TranslateHyperparams *hp,
                                const Tokenizer *tok,
                                const char *en_text, int max_len, float temperature) {
    int seq_len = hp->seq_len;
    int vocab_size = hp->vocab_size;

    int src_cap = strlen(en_text) + 8;
    int *src_raw = (int *)malloc(src_cap * sizeof(int));
    int src_len_raw = tokenizer_encode(tok, en_text, src_raw, src_cap);
    if (src_len_raw <= 0) {
        free(src_raw);
        return;
    }

    int *tgt_buf = (int *)malloc((size_t)(max_len + 2) * sizeof(int));
    tgt_buf[0] = TOK_BOS;
    int tgt_len = 1;

    int fixed_len = seq_len;

    int *src_pad = (int *)malloc((size_t)fixed_len * sizeof(int));

    for (int s = 0; s < fixed_len; s++) {
        src_pad[s] = (s < src_len_raw) ? src_raw[s] : TOK_PAD;
    }

    printf("EN: %s\nZH: ", en_text);

    for (int step = 0; step < max_len && tgt_len < fixed_len; step++) {
        int *tgt_pad = (int *)malloc((size_t)fixed_len * sizeof(int));
        for (int s = 0; s < fixed_len; s++) {
            tgt_pad[s] = (s < tgt_len) ? tgt_buf[s] : TOK_PAD;
        }

        Tensor3D causal = mask_causal_create(fixed_len);

        TransformerCache cache = {0};
        Tensor3D output = transformer_forward_lm(model,
                                                 src_pad, fixed_len,
                                                 tgt_pad, fixed_len,
                                                 1,
                                                 NULL, &causal, &cache);

        int logit_offset = (tgt_len - 1) * vocab_size;
        int pred = sample_logits(&output.data[logit_offset], vocab_size, temperature);

        transformer_cache_free(&cache, model->config.encoder_layers, model->config.decoder_layers);
        tensor_free(&output);
        tensor_free(&causal);
        free(tgt_pad);

        if (pred == TOK_EOS) break;

        tgt_buf[tgt_len++] = pred;
    }

    {
        char text_buf[4096];
        tokenizer_decode(tok, tgt_buf + 1, tgt_len - 1, text_buf, sizeof(text_buf));
        printf("%s\n\n", text_buf);
    }

    printf("\n\n");

    free(src_raw);
    free(src_pad);
    free(tgt_buf);
}

int main(void) {
    srand((unsigned)time(NULL));

    printf("=== English to Chinese Translation Demo ===\n\n");

    printf("Using byte-level tokenizer (shared for EN and ZH)...\n");
    Tokenizer *tok = tokenizer_create_byte();
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

    printf("\n=== Translation Test (greedy) ===\n\n");
    translate_generate(model, &hp, tok, "hello", 16, 0.01f);
    translate_generate(model, &hp, tok, "good morning", 16, 0.01f);
    translate_generate(model, &hp, tok, "thank you", 16, 0.01f);
    translate_generate(model, &hp, tok, "i love you", 16, 0.01f);
    translate_generate(model, &hp, tok, "what is your name", 16, 0.01f);

    printf("=== Translation Test (sampling T=0.5) ===\n\n");
    translate_generate(model, &hp, tok, "hello", 16, 0.5f);
    translate_generate(model, &hp, tok, "good morning", 16, 0.5f);
    translate_generate(model, &hp, tok, "thank you", 16, 0.5f);
    translate_generate(model, &hp, tok, "i love you", 16, 0.5f);
    translate_generate(model, &hp, tok, "what is your name", 16, 0.5f);

    free_translation_corpus(&corpus);
    if (ts) training_state_free(ts);
    if (model) transformer_free(model);
    tokenizer_free(tok);
    return 0;
}
