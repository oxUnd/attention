#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define VOCAB_SIZE 28
#define SEQ_LEN 16
#define D_MODEL 64
#define BATCH_SIZE 4
#define MAX_EPOCHS 120
#define EARLY_STOP_PATIENCE 18
#define D_FF 128
#define NHEAD 4

static const char CHARS[] = "abcdefghijklmnopqrstuvwxyz .";

static int char_to_idx(char c) {
    if (c >= 'A' && c <= 'Z') c += 32;
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c == ' ') return 26;
    if (c == '.') return 27;
    return 26;
}

static char idx_to_char(int idx) {
    if (idx >= 0 && idx < VOCAB_SIZE) return CHARS[idx];
    return ' ';
}

static const char *CORPUS =
    "the quick brown fox jumps over the lazy dog. "
    "she sells sea shells by the sea shore. "
    "the early bird catches the worm. "
    "a journey of a thousand miles begins with a single step. "
    "all that glitters is not gold. "
    "practice makes perfect. "
    "where there is a will there is a way. "
    "actions speak louder than words. "
    "knowledge is power. "
    "the pen is mightier than the sword. "
    "an apple a day keeps the doctor away. "
    "better late than never. "
    "curiosity killed the cat. "
    "every cloud has a silver lining. "
    "the only thing we have to fear is fear itself. "
    "to be or not to be that is the question. "
    "rome was not built in a day. "
    "when in rome do as the romans do. "
    "the grass is always greener on the other side. "
    "do not count your chickens before they hatch. ";

static int corpus_idx[2048];
static int corpus_len = 0;

static void preprocess_corpus(void) {
    corpus_len = 0;
    for (int i = 0; CORPUS[i]; i++) {
        int idx = char_to_idx(CORPUS[i]);
        if (idx >= 0 && idx < VOCAB_SIZE) {
            corpus_idx[corpus_len++] = idx;
        }
    }
}

/* Valid window starts: 0 .. corpus_len - SEQ_LEN - 1 (inclusive), count = corpus_len - SEQ_LEN. */
static void generate_text_batch_at(
    Tensor3D *src, Tensor3D *tgt, int *targets, int d_model, const int *starts
) {
    for (int b = 0; b < BATCH_SIZE; b++) {
        int start = starts[b];
        for (int s = 0; s < SEQ_LEN; s++) {
            int ci = corpus_idx[start + s];
            for (int d = 0; d < d_model; d++) {
                src->data[b * SEQ_LEN * d_model + s * d_model + d] = (d == ci) ? 1.0f : 0.0f;
                tgt->data[b * SEQ_LEN * d_model + s * d_model + d] = (d == ci) ? 1.0f : 0.0f;
            }
            targets[b * SEQ_LEN + s] = corpus_idx[start + s + 1];
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

static float compute_accuracy(Tensor3D *output, int *targets) {
    int correct = 0;
    int total = BATCH_SIZE * SEQ_LEN;
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            int offset = b * SEQ_LEN * output->d_model + s * output->d_model;
            int pred = 0;
            float max_val = output->data[offset];
            for (int v = 1; v < VOCAB_SIZE; v++) {
                if (output->data[offset + v] > max_val) {
                    max_val = output->data[offset + v];
                    pred = v;
                }
            }
            if (pred == targets[b * SEQ_LEN + s]) correct++;
        }
    }
    return (float)correct / total * 100.0f;
}

static int sample_output(float *logits, int size, float temperature) {
    float max_val = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    float probs[64];
    for (int i = 0; i < size; i++) {
        probs[i] = expf((logits[i] - max_val) / temperature);
        sum += probs[i];
    }
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < size; i++) {
        cumsum += probs[i] / sum;
        if (cumsum >= r) return i;
    }
    return size - 1;
}

static void generate_text(Transformer *model, const char *seed, int length, float temperature, int d_model) {
    int seed_len = strlen(seed);
    int buf_cap = seed_len + length;
    int *buf = malloc(buf_cap * sizeof(int));
    int buf_len = 0;

    for (int i = 0; seed[i]; i++) {
        buf[buf_len++] = char_to_idx(seed[i]);
    }

    printf("Seed: \"%s\"\nGenerated: %s", seed, seed);

    for (int step = 0; step < length; step++) {
        int start = (buf_len > SEQ_LEN) ? (buf_len - SEQ_LEN) : 0;
        int seq_len = buf_len - start;

        Tensor3D src = tensor_create(1, seq_len, d_model);
        Tensor3D tgt = tensor_create(1, seq_len, d_model);

        for (int s = 0; s < seq_len; s++) {
            int ci = buf[start + s];
            for (int d = 0; d < d_model; d++) {
                src.data[s * d_model + d] = (d == ci) ? 1.0f : 0.0f;
                tgt.data[s * d_model + d] = (d == ci) ? 1.0f : 0.0f;
            }
        }

        Tensor3D output = transformer_forward(model, &src, &tgt, NULL, NULL, NULL);

        int pred = sample_output(&output.data[(seq_len - 1) * d_model], VOCAB_SIZE, temperature);
        buf[buf_len++] = pred;
        printf("%c", idx_to_char(pred));
        fflush(stdout);

        tensor_free(&src);
        tensor_free(&tgt);
        tensor_free(&output);
    }

    printf("\n\n");
    free(buf);
}

int main(void) {
    srand(time(NULL));
    printf("=== Transformer Character-Level Text Training ===\n");
    printf("Vocab: %d chars (%s), Seq: %d, Batch: %d, d_model: %d\n\n",
           VOCAB_SIZE, CHARS, SEQ_LEN, BATCH_SIZE, D_MODEL);

    preprocess_corpus();
    printf("Corpus: %d characters\n", corpus_len);
    printf("Sample: \"%.50s...\"\n\n", CORPUS);

    TransformerConfig config = {
        .d_model = D_MODEL,
        .nhead = NHEAD,
        .d_ff = D_FF,
        .encoder_layers = 1,
        .decoder_layers = 1,
        .max_len = SEQ_LEN + 5,
        .dropout = 0.0f,
        .activation = GELU
    };

    int n_starts = corpus_len - SEQ_LEN;
    if (n_starts < BATCH_SIZE) {
        fprintf(stderr, "Corpus too short for SEQ_LEN=%d (need at least %d chars).\n", SEQ_LEN,
                SEQ_LEN + BATCH_SIZE);
        return 1;
    }

    int *window_order = malloc((size_t)n_starts * sizeof(int));
    if (!window_order) {
        perror("malloc");
        return 1;
    }
    for (int i = 0; i < n_starts; i++) {
        window_order[i] = i;
    }

    int num_batches = n_starts / BATCH_SIZE;

    Transformer *model = transformer_create(&config);
    TrainingState *ts = training_state_create(model, 0.002f, 0.9f, 0.999f, 1e-8f);

    Tensor3D src = tensor_create(BATCH_SIZE, SEQ_LEN, D_MODEL);
    Tensor3D tgt = tensor_create(BATCH_SIZE, SEQ_LEN, D_MODEL);
    int targets[BATCH_SIZE * SEQ_LEN];

    float best_avg_loss = 1e9f;
    int patience = 0;

    printf("Windows per epoch: %d (%d gradient steps), max epochs: %d\n\n", n_starts, num_batches,
           MAX_EPOCHS);

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        shuffle_int_range(window_order, n_starts);

        float sum_loss = 0.0f;
        float sum_acc = 0.0f;

        for (int bi = 0; bi < num_batches; bi++) {
            int starts[BATCH_SIZE];
            int base = bi * BATCH_SIZE;
            for (int b = 0; b < BATCH_SIZE; b++) {
                starts[b] = window_order[base + b];
            }

            generate_text_batch_at(&src, &tgt, targets, D_MODEL, starts);

            training_state_zero_grads(ts);

            TransformerCache cache = {0};
            Tensor3D output = transformer_forward(model, &src, &tgt, NULL, NULL, &cache);

            LossResult loss = cross_entropy_loss(&output, targets, VOCAB_SIZE);
            float acc = compute_accuracy(&output, targets);

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

        if (epoch % 5 == 0 || epoch == MAX_EPOCHS - 1 || avg_loss < 1.2f) {
            printf("Epoch %4d: avg_loss=%.4f avg_acc=%.1f%%\n", epoch, avg_loss, avg_acc);
        }

        if (avg_loss < best_avg_loss - 0.002f) {
            best_avg_loss = avg_loss;
            patience = 0;
        } else {
            patience++;
        }

        if (patience >= EARLY_STOP_PATIENCE || avg_loss < 0.35f) {
            printf("Stopped at epoch %d: avg_loss=%.4f avg_acc=%.1f%% (%s)\n", epoch, avg_loss,
                   avg_acc, patience >= EARLY_STOP_PATIENCE ? "patience" : "loss threshold");
            break;
        }
    }

    free(window_order);

    printf("\n=== Training Complete ===\n\n");

    printf("--- Temperature 0.5 (conservative) ---\n");
    generate_text(model, "the ", 60, 0.5f, D_MODEL);

    printf("--- Temperature 1.0 (balanced) ---\n");
    generate_text(model, "the ", 60, 1.0f, D_MODEL);

    printf("--- Temperature 1.5 (creative) ---\n");
    generate_text(model, "the ", 60, 1.5f, D_MODEL);

    printf("--- Different seed: \"she \" ---\n");
    generate_text(model, "she ", 60, 0.8f, D_MODEL);

    tensor_free(&src);
    tensor_free(&tgt);
    training_state_free(ts);
    transformer_free(model);

    return 0;
}