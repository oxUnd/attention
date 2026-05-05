#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VOCAB_SIZE 16
#define SEQ_LEN 8
#define D_MODEL 16
#define BATCH_SIZE 4

void generate_batch(Tensor3D *src, int *targets, int batch_size, int seq_len, int vocab_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int token = rand() % vocab_size;
            for (int d = 0; d < D_MODEL; d++) {
                src->data[b * seq_len * D_MODEL + s * D_MODEL + d] =
                    (float)token / vocab_size;
            }
            targets[b * seq_len + s] = token;
        }
    }
}

int predict_token(Tensor3D *output, int batch, int seq, int vocab_size) {
    int offset = (batch * output->seq_len + seq) * output->d_model;
    int best = 0;
    float best_val = output->data[offset];
    for (int v = 1; v < vocab_size; v++) {
        if (output->data[offset + v] > best_val) {
            best_val = output->data[offset + v];
            best = v;
        }
    }
    return best;
}

float compute_accuracy(Tensor3D *output, int *targets, int batch_size, int seq_len, int vocab_size) {
    int correct = 0;
    int total = batch_size * seq_len;
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int pred = predict_token(output, b, s, vocab_size);
            if (pred == targets[b * seq_len + s]) correct++;
        }
    }
    return (float)correct / total * 100.0f;
}

int main() {
    srand(time(NULL));

    printf("=== Transformer Copy Task Training ===\n");
    printf("Vocab: %d, Seq: %d, Batch: %d, d_model: %d\n\n", VOCAB_SIZE, SEQ_LEN, BATCH_SIZE, D_MODEL);

    TransformerConfig config = {
        .d_model = D_MODEL,
        .nhead = 2,
        .d_ff = 32,
        .encoder_layers = 2,
        .decoder_layers = 2,
        .max_len = SEQ_LEN + 10,
        .dropout = 0.0,
        .activation = GELU
    };

    Transformer *model = transformer_create(&config);
    TrainingState *ts = training_state_create(model, 0.001f, 0.9f, 0.999f, 1e-8f);

    Tensor3D src = tensor_create(BATCH_SIZE, SEQ_LEN, D_MODEL);
    Tensor3D tgt = tensor_create(BATCH_SIZE, SEQ_LEN, D_MODEL);
    int targets[BATCH_SIZE * SEQ_LEN];

    generate_batch(&tgt, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);

    int num_epochs = 1000;
    float prev_loss = 1e9;
    int no_improve = 0;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        generate_batch(&src, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
        for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_MODEL; i++) {
            tgt.data[i] = src.data[i];
        }

        training_state_zero_grads(ts);

        TransformerCache cache = {0};
        Tensor3D output = transformer_forward(model, &src, &tgt, NULL, NULL, &cache);

        LossResult loss = cross_entropy_loss(&output, targets, VOCAB_SIZE);
        float acc = compute_accuracy(&output, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);

        if (epoch % 100 == 0 || loss.loss < 0.1f) {
            printf("Epoch %d: loss=%.4f, accuracy=%.1f%%\n", epoch, loss.loss, acc);
        }

        transformer_backward(model, &cache, &loss.grad, &tgt, ts);
        training_state_clip_grads(ts, 1.0f);
        training_state_update(ts);

        transformer_cache_free(&cache, config.encoder_layers, config.decoder_layers);
        tensor_free(&output);
        tensor_free(&loss.grad);

        if (loss.loss < prev_loss - 0.001f) {
            no_improve = 0;
        } else {
            no_improve++;
        }
        prev_loss = loss.loss;

        if (no_improve > 100 || loss.loss < 0.01f) {
            printf("Early stopping at epoch %d\n", epoch);
            break;
        }
    }

    printf("\nTraining complete!\n");

    printf("\n=== Test ===\n");
    generate_batch(&src, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
    for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_MODEL; i++) {
        tgt.data[i] = src.data[i];
    }

    Tensor3D output = transformer_forward(model, &src, &tgt, NULL, NULL, NULL);
    float acc = compute_accuracy(&output, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
    printf("Test accuracy: %.1f%%\n", acc);

    printf("\nPrediction examples (first 3 samples):\n");
    for (int b = 0; b < 3 && b < BATCH_SIZE; b++) {
        printf("Sample %d: ", b);
        for (int s = 0; s < SEQ_LEN; s++) {
            int target = targets[b * SEQ_LEN + s];
            int pred = predict_token(&output, b, s, VOCAB_SIZE);
            printf("%d->%d ", target, pred);
        }
        printf("\n");
    }

    tensor_free(&src);
    tensor_free(&tgt);
    tensor_free(&output);
    training_state_free(ts);
    transformer_free(model);

    return 0;
}