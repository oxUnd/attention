#include "text_lm.h"
#include "tokenizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char LM_CHARS[] = "abcdefghijklmnopqrstuvwxyz .";

const TextLmHyperparams text_lm_default_hyperparams = {
    .seq_len = 24,
    .batch_size = 8,
    .d_model = 64,
    .d_ff = 128,
    .nhead = 4,
    .encoder_layers = 1,
    .decoder_layers = 1,
    .max_epochs = 80,
    .early_stop_patience = 14,
    .learning_rate = 0.0008f,
    .dropout = 0.0f,
    .vocab_size = TEXT_LM_VOCAB_SIZE,
};

const char *text_lm_vocab_chars(void) { return LM_CHARS; }

char text_lm_idx_to_char(int idx) {
    if (idx >= 0 && idx < TEXT_LM_VOCAB_SIZE) {
        return LM_CHARS[idx];
    }
    return ' ';
}

int text_lm_char_to_idx(char c) {
    if (c >= 'A' && c <= 'Z') {
        c += 32;
    }
    if (c >= 'a' && c <= 'z') {
        return c - 'a';
    }
    if (c == ' ') {
        return 26;
    }
    if (c == '.') {
        return 27;
    }
    return 26;
}

int text_corpus_from_text(const char *text, TextCorpus *out) {
    out->token_ids = NULL;
    out->len = 0;
    if (!text || !out) {
        return -1;
    }
    size_t cap = strlen(text) + 1;
    int *buf = (int *)malloc(cap * sizeof(int));
    if (!buf) {
        return -1;
    }
    int n = 0;
    for (size_t i = 0; text[i]; i++) {
        int idx = text_lm_char_to_idx(text[i]);
        if (idx >= 0 && idx < TEXT_LM_VOCAB_SIZE) {
            buf[n++] = idx;
        }
    }
    if (n == 0) {
        free(buf);
        return -1;
    }
    out->token_ids = buf;
    out->len = n;
    return 0;
}

int text_corpus_from_text_with_tokenizer(const char *text, const Tokenizer *tok, TextCorpus *out) {
    if (!text || !tok || !out) return -1;
    out->token_ids = NULL;
    out->len = 0;
    size_t cap = strlen(text) + 16;
    int *buf = (int *)malloc(cap * sizeof(int));
    if (!buf) return -1;
    int n = tokenizer_encode(tok, text, buf, (int)cap);
    if (n <= 0) {
        free(buf);
        return -1;
    }
    out->token_ids = buf;
    out->len = n;
    return 0;
}

void text_corpus_free(TextCorpus *c) {
    if (c && c->token_ids) {
        free(c->token_ids);
        c->token_ids = NULL;
        c->len = 0;
    }
}

static void fill_one_hot_batch(const int *corpus_idx, Tensor3D *src, Tensor3D *tgt, int *targets,
                               int d_model, int seq_len, int batch_size, const int *starts) {
    tensor_zero(src);
    tensor_zero(tgt);
    for (int b = 0; b < batch_size; b++) {
        int start = starts[b];
        for (int s = 0; s < seq_len; s++) {
            int ci = corpus_idx[start + s];
            if (ci >= 0 && ci < d_model) {
                src->data[b * seq_len * d_model + s * d_model + ci] = 1.0f;
                tgt->data[b * seq_len * d_model + s * d_model + ci] = 1.0f;
            }
            targets[b * seq_len + s] = corpus_idx[start + s + 1];
        }
    }
}

static void shuffle_int_range(int *a, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

static float compute_accuracy(Tensor3D *output, int *targets, int seq_len, int batch_size, int vocab_size) {
    int correct = 0;
    int total = batch_size * seq_len;
    int vsize = (vocab_size > 0 && vocab_size <= output->d_model) ? vocab_size : output->d_model;
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int offset = b * seq_len * output->d_model + s * output->d_model;
            int pred = 0;
            float max_val = output->data[offset];
            for (int v = 1; v < vsize; v++) {
                if (output->data[offset + v] > max_val) {
                    max_val = output->data[offset + v];
                    pred = v;
                }
            }
            if (pred == targets[b * seq_len + s]) {
                correct++;
            }
        }
    }
    return (float)correct / total * 100.0f;
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
        if (cumsum >= r) {
            chosen = i;
            break;
        }
    }
    free(probs);
    return chosen;
}

static int run_one_step(Transformer *model, const int *context, int ctx_len, int d_model,
                        int vocab_size, float temperature) {
    Tensor3D src = tensor_create(1, ctx_len, d_model);
    Tensor3D tgt = tensor_create(1, ctx_len, d_model);
    for (int s = 0; s < ctx_len; s++) {
        int ci = context[s];
        if (ci >= 0 && ci < d_model) {
            src.data[s * d_model + ci] = 1.0f;
            tgt.data[s * d_model + ci] = 1.0f;
        }
    }
    Tensor3D mask = causal_mask_create(ctx_len);
    Tensor3D output = transformer_forward(model, &src, &tgt, &mask, &mask, NULL);
    int vsize = (vocab_size > 0 && vocab_size <= d_model) ? vocab_size : d_model;
    int pred = sample_logits(&output.data[(ctx_len - 1) * d_model], vsize, temperature);
    tensor_free(&src);
    tensor_free(&tgt);
    tensor_free(&mask);
    tensor_free(&output);
    return pred;
}

void text_lm_generate(Transformer *model, const TextLmHyperparams *hp, const char *seed, int length,
                      float temperature, FILE *out) {
    int d_model = hp->d_model;
    int seq_cap = hp->seq_len;
    int seed_len = (int)strlen(seed);
    int buf_cap = seed_len + length + 8;
    int *buf = (int *)malloc((size_t)buf_cap * sizeof(int));
    if (!buf) return;
    int buf_len = 0;
    for (int i = 0; seed[i]; i++) {
        buf[buf_len++] = text_lm_char_to_idx(seed[i]);
    }

    fprintf(out, "Seed: \"%s\"\nGenerated: %s", seed, seed);

    int vocab_size = hp->vocab_size > 0 ? hp->vocab_size : TEXT_LM_VOCAB_SIZE;
    for (int step = 0; step < length; step++) {
        int start = (buf_len > seq_cap) ? (buf_len - seq_cap) : 0;
        int cur_seq_len = buf_len - start;
        int pred = run_one_step(model, buf + start, cur_seq_len, d_model, vocab_size, temperature);
        buf[buf_len++] = pred;
        fputc(text_lm_idx_to_char(pred), out);
        fflush(out);
    }

    fprintf(out, "\n\n");
    free(buf);
}

void text_lm_generate_with_tokenizer(Transformer *model, const TextLmHyperparams *hp,
                                     const Tokenizer *tok, const char *seed, int length,
                                     float temperature, FILE *out) {
    if (!tok) {
        text_lm_generate(model, hp, seed, length, temperature, out);
        return;
    }
    int d_model = hp->d_model;
    int seq_cap = hp->seq_len;
    int buf_cap = length + 64;
    int *buf = (int *)malloc((size_t)buf_cap * sizeof(int));
    if (!buf) return;
    int buf_len = 0;

    if (seed && seed[0]) {
        buf_len = tokenizer_encode(tok, seed, buf, buf_cap);
    }
    if (buf_len == 0) {
        buf[buf_len++] = TOK_BOS;
    }

    char text_buf[2048];
    tokenizer_decode(tok, buf, buf_len, text_buf, sizeof(text_buf));
    fprintf(out, "Seed: \"%s\"\nGenerated: %s", seed ? seed : "", text_buf);
    fflush(out);

    int vocab_size = hp->vocab_size > 0 ? hp->vocab_size : tok->vocab_size;
    int last_emit_len = (int)strlen(text_buf);

    for (int step = 0; step < length; step++) {
        int start = (buf_len > seq_cap) ? (buf_len - seq_cap) : 0;
        int cur_seq_len = buf_len - start;
        int pred = run_one_step(model, buf + start, cur_seq_len, d_model, vocab_size, temperature);
        if (pred == TOK_EOS) break;
        if (buf_len >= buf_cap) break;
        buf[buf_len++] = pred;

        tokenizer_decode(tok, buf, buf_len, text_buf, sizeof(text_buf));
        int new_len = (int)strlen(text_buf);
        if (new_len > last_emit_len) {
            fputs(text_buf + last_emit_len, out);
            fflush(out);
            last_emit_len = new_len;
        }
    }
    fprintf(out, "\n\n");
    free(buf);
}

int text_lm_train(const TextCorpus *corpus, const TextLmHyperparams *hp, Transformer **model_out,
                  TrainingState **ts_out) {
    if (!corpus || !corpus->token_ids || corpus->len <= 0 || !hp || !model_out || !ts_out) {
        return -1;
    }
    const int *corpus_idx = corpus->token_ids;
    int corpus_len = corpus->len;

    int seq_len = hp->seq_len;
    int batch_size = hp->batch_size;
    int n_starts = corpus_len - seq_len - 1;
    if (n_starts < batch_size) {
        return -2;
    }

    int vocab_size = hp->vocab_size > 0 ? hp->vocab_size : TEXT_LM_VOCAB_SIZE;
    if (vocab_size > hp->d_model) {
        fprintf(stderr, "vocab_size (%d) must be <= d_model (%d)\n", vocab_size, hp->d_model);
        return -3;
    }

    int *window_order = (int *)malloc((size_t)n_starts * sizeof(int));
    if (!window_order) return -1;
    for (int i = 0; i < n_starts; i++) window_order[i] = i;

    int num_batches = n_starts / batch_size;

    TransformerConfig config = {.d_model = hp->d_model,
                                .nhead = hp->nhead,
                                .d_ff = hp->d_ff,
                                .encoder_layers = hp->encoder_layers,
                                .decoder_layers = hp->decoder_layers,
                                .max_len = seq_len + 5,
                                .dropout = hp->dropout,
                                .activation = GELU};

    Transformer *model = transformer_create(&config);
    TrainingState *ts = training_state_create(model, hp->learning_rate, 0.9f, 0.999f, 1e-8f);
    if (!model || !ts) {
        free(window_order);
        if (model) transformer_free(model);
        if (ts) training_state_free(ts);
        return -1;
    }

    Tensor3D src = tensor_create(batch_size, seq_len, hp->d_model);
    Tensor3D tgt = tensor_create(batch_size, seq_len, hp->d_model);
    Tensor3D causal = causal_mask_create(seq_len);
    int *targets = (int *)malloc((size_t)(batch_size * seq_len) * sizeof(int));
    if (!targets) {
        tensor_free(&src);
        tensor_free(&tgt);
        tensor_free(&causal);
        free(window_order);
        training_state_free(ts);
        transformer_free(model);
        return -1;
    }

    float best_avg_loss = 1e9f;
    int patience = 0;

    printf("Windows per epoch: %d (%d gradient steps), max epochs: %d, vocab: %d\n\n",
           n_starts, num_batches, hp->max_epochs, vocab_size);

    for (int epoch = 0; epoch < hp->max_epochs; epoch++) {
        shuffle_int_range(window_order, n_starts);

        float sum_loss = 0.0f;
        float sum_acc = 0.0f;

        for (int bi = 0; bi < num_batches; bi++) {
            int starts[batch_size];
            int base = bi * batch_size;
            for (int b = 0; b < batch_size; b++) {
                starts[b] = window_order[base + b];
            }

            fill_one_hot_batch(corpus_idx, &src, &tgt, targets, hp->d_model, seq_len, batch_size, starts);

            training_state_zero_grads(ts);

            TransformerCache cache = {0};
            Tensor3D output = transformer_forward(model, &src, &tgt, &causal, &causal, &cache);

            LossResult loss = cross_entropy_loss(&output, targets, vocab_size);
            float acc = compute_accuracy(&output, targets, seq_len, batch_size, vocab_size);

            sum_loss += loss.loss;
            sum_acc += acc;

            transformer_backward(model, &cache, &loss.grad, &tgt, ts);
            training_state_clip_grads(ts, 1.0f);
            training_state_update(ts);

            transformer_cache_free(&cache, config.encoder_layers, config.decoder_layers);
            tensor_free(&output);
            tensor_free(&loss.grad);
        }

        float avg_loss = sum_loss / (float)num_batches;
        float avg_acc = sum_acc / (float)num_batches;

        if (epoch % 2 == 0 || epoch == hp->max_epochs - 1 || avg_loss < 1.2f) {
            printf("Epoch %4d: avg_loss=%.4f avg_acc=%.1f%%\n", epoch, avg_loss, avg_acc);
        }

        if (avg_loss < best_avg_loss - 0.002f) {
            best_avg_loss = avg_loss;
            patience = 0;
        } else {
            patience++;
        }

        if (patience >= hp->early_stop_patience || avg_loss < 0.25f) {
            printf("Stopped at epoch %d: avg_loss=%.4f avg_acc=%.1f%% (%s)\n", epoch, avg_loss, avg_acc,
                   patience >= hp->early_stop_patience ? "patience" : "loss threshold");
            break;
        }
    }

    free(targets);
    tensor_free(&src);
    tensor_free(&tgt);
    tensor_free(&causal);
    free(window_order);

    *model_out = model;
    *ts_out = ts;
    return 0;
}

void text_lm_free_session(Transformer *model, TrainingState *ts) {
    if (ts) training_state_free(ts);
    if (model) transformer_free(model);
}

int text_lm_save(Transformer *model, TrainingState *ts, float lr, const char *path) {
    if (!model || !path) return -1;
    Trainer tr = {.model = model, .ts = ts, .learning_rate = lr};
    return trainer_save(&tr, path);
}

int text_lm_load(const char *path, Transformer **model_out, TrainingState **ts_out) {
    if (!path || !model_out || !ts_out) return -1;
    Trainer tr = trainer_load(path);
    if (!tr.model) {
        *model_out = NULL;
        *ts_out = NULL;
        return -1;
    }
    *model_out = tr.model;
    *ts_out = tr.ts;
    return 0;
}
