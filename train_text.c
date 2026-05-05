#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define VOCAB_SIZE 28
#define SEQ_LEN 32
#define D_MODEL VOCAB_SIZE
#define BATCH_SIZE 4
#define NUM_EPOCHS 5000
#define D_FF 256
#define NHEAD 2

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

static void encode_onehot(int idx, float *dst) {
    for (int d = 0; d < D_MODEL; d++) {
        dst[d] = (d == idx) ? 1.0f : 0.0f;
    }
}

static void generate_text_batch(Tensor3D *src, Tensor3D *tgt, int *targets) {
    for (int b = 0; b < BATCH_SIZE; b++) {
        int start = rand() % (corpus_len - SEQ_LEN - 1);
        for (int s = 0; s < SEQ_LEN; s++) {
            int ci = corpus_idx[start + s];
            encode_onehot(ci, &src->data[b * SEQ_LEN * D_MODEL + s * D_MODEL]);
            encode_onehot(ci, &tgt->data[b * SEQ_LEN * D_MODEL + s * D_MODEL]);
            targets[b * SEQ_LEN + s] = corpus_idx[start + s + 1];
        }
    }
}

static float compute_accuracy(Tensor3D *output, int *targets) {
    int correct = 0;
    int total = BATCH_SIZE * SEQ_LEN;
    for (int b = 0; b < BATCH_SIZE; b++) {
        for (int s = 0; s < SEQ_LEN; s++) {
            int offset = b * SEQ_LEN * D_MODEL + s * D_MODEL;
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

static void generate_text(Transformer *model, const char *seed, int length, float temperature) {
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

        Tensor3D src = tensor_create(1, seq_len, D_MODEL);
        Tensor3D tgt = tensor_create(1, seq_len, D_MODEL);

        for (int s = 0; s < seq_len; s++) {
            encode_onehot(buf[start + s], &src.data[s * D_MODEL]);
            encode_onehot(buf[start + s], &tgt.data[s * D_MODEL]);
        }

        Tensor3D output = transformer_forward(model, &src, &tgt, NULL, NULL, NULL);

        int pred = sample_output(&output.data[(seq_len - 1) * D_MODEL], D_MODEL, temperature);
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
        .encoder_layers = 2,
        .decoder_layers = 2,
        .max_len = SEQ_LEN + 5,
        .dropout = 0.0f,
        .activation = GELU
    };

    Transformer *model = transformer_create(&config);
    Trainer trainer = trainer_create(model, 0.001f, 0.9f, 0.999f, 1e-8f);

    Tensor3D src = tensor_create(BATCH_SIZE, SEQ_LEN, D_MODEL);
    Tensor3D tgt = tensor_create(BATCH_SIZE, SEQ_LEN, D_MODEL);
    int targets[BATCH_SIZE * SEQ_LEN];

    float prev_loss = 1e9f;
    int no_improve = 0;

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        generate_text_batch(&src, &tgt, targets);

        TransformerCache cache = {0};
        Tensor3D output = transformer_forward(model, &src, &tgt, NULL, NULL, &cache);

        LossResult loss = cross_entropy_loss(&output, targets, VOCAB_SIZE);
        float acc = compute_accuracy(&output, targets);

        if (epoch % 500 == 0 || loss.loss < 1.5f) {
            printf("Epoch %d: loss=%.4f, acc=%.1f%%\n", epoch, loss.loss, acc);
        }

        transformer_backward(model, &cache, &loss.grad, &tgt);

        tensor_free(&output);
        tensor_free(&loss.grad);

        if (loss.loss < prev_loss - 0.0001f) {
            no_improve = 0;
        } else {
            no_improve++;
        }
        prev_loss = loss.loss;

        if (no_improve > 800 || loss.loss < 0.3f) {
            printf("Early stop at epoch %d: loss=%.4f, acc=%.1f%%\n", epoch, loss.loss, acc);
            break;
        }
    }

    printf("\n=== Training Complete ===\n\n");

    printf("--- Temperature 0.5 (conservative) ---\n");
    generate_text(model, "the ", 80, 0.5f);

    printf("--- Temperature 1.0 (balanced) ---\n");
    generate_text(model, "the ", 80, 1.0f);

    printf("--- Temperature 1.5 (creative) ---\n");
    generate_text(model, "the ", 80, 1.5f);

    printf("--- Different seed: \"she \" ---\n");
    generate_text(model, "she ", 80, 0.8f);

    tensor_free(&src);
    tensor_free(&tgt);
    trainer_free(&trainer);
    transformer_free(model);

    return 0;
}
