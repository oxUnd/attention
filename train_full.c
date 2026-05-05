#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VOCAB_SIZE 16
#define SEQ_LEN 4
#define D_MODEL 16
#define BATCH_SIZE 8
#define NUM_EPOCHS 500

void generate_copy_batch(Tensor3D *src, Tensor3D *tgt, int *targets, int batch_size, int seq_len, int vocab_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int token = rand() % vocab_size;
            targets[b * seq_len + s] = token;
            for (int d = 0; d < D_MODEL; d++) {
                float val = (d == token % D_MODEL) ? 1.0f : 0.0f;
                src->data[b * seq_len * D_MODEL + s * D_MODEL + d] = val;
                tgt->data[b * seq_len * D_MODEL + s * D_MODEL + d] = val;
            }
        }
    }
}

float compute_accuracy(Tensor3D *output, int *targets, int batch_size, int seq_len, int vocab_size) {
    int correct = 0;
    int total = batch_size * seq_len;

    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int offset = b * seq_len * output->d_model + s * output->d_model;
            int pred = 0;
            float max_val = output->data[offset];
            for (int v = 1; v < vocab_size && v < output->d_model; v++) {
                if (output->data[offset + v] > max_val) {
                    max_val = output->data[offset + v];
                    pred = v;
                }
            }
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
        .encoder_layers = 1,
        .decoder_layers = 1,
        .max_len = SEQ_LEN + 5,
        .dropout = 0.0,
        .activation = GELU
    };

    Transformer *model = transformer_create(&config);
    TrainingState *ts = training_state_create(model, 0.001f, 0.9f, 0.999f, 1e-8f);

    Tensor3D src = tensor_create(BATCH_SIZE, SEQ_LEN, D_MODEL);
    Tensor3D tgt = tensor_create(BATCH_SIZE, SEQ_LEN, D_MODEL);
    int targets[BATCH_SIZE * SEQ_LEN];

    float prev_loss = 1e9;
    int no_improve = 0;

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        generate_copy_batch(&src, &tgt, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);

        training_state_zero_grads(ts);

        TransformerCache cache = {0};
        Tensor3D output = transformer_forward(model, &src, &tgt, NULL, NULL, &cache);

        LossResult loss = cross_entropy_loss(&output, targets, VOCAB_SIZE);
        float acc = compute_accuracy(&output, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);

        if (epoch % 50 == 0 || loss.loss < 0.5f) {
            printf("Epoch %d: loss=%.4f, acc=%.1f%%\n", epoch, loss.loss, acc);
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

        if (no_improve > 100 || loss.loss < 0.1f) {
            printf("Early stop at epoch %d: loss=%.4f\n", epoch, loss.loss);
            break;
        }
    }

    printf("\n=== Training Complete ===\n\n");

    printf("Testing...\n");
    generate_copy_batch(&src, &tgt, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
    Tensor3D test_output = transformer_forward(model, &src, &tgt, NULL, NULL, NULL);
    float test_acc = compute_accuracy(&test_output, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
    printf("Test accuracy: %.1f%%\n", test_acc);

    printf("\nPrediction examples (first 3 samples):\n");
    for (int b = 0; b < 3 && b < BATCH_SIZE; b++) {
        printf("Sample %d: ", b);
        for (int s = 0; s < SEQ_LEN; s++) {
            int offset = b * SEQ_LEN * D_MODEL + s * D_MODEL;
            int pred = 0;
            float max_val = test_output.data[offset];
            for (int v = 1; v < VOCAB_SIZE && v < D_MODEL; v++) {
                if (test_output.data[offset + v] > max_val) {
                    max_val = test_output.data[offset + v];
                    pred = v;
                }
            }
            printf("%d->%d ", targets[b * SEQ_LEN + s], pred);
        }
        printf("\n");
    }

    tensor_free(&src);
    tensor_free(&tgt);
    tensor_free(&test_output);
    training_state_free(ts);
    transformer_free(model);

    return 0;
}