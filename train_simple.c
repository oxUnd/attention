#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VOCAB_SIZE 8
#define SEQ_LEN 4
#define D_MODEL 16

int main() {
    srand(time(NULL));
    printf("=== Simple Training Test ===\n\n");

    TransformerConfig config = {
        .d_model = D_MODEL,
        .num_heads = 2,
        .d_ff = 32,
        .encoder_layers = 1,
        .decoder_layers = 1,
        .max_len = 10,
        .dropout = 0.0,
        .activation = ACT_GELU
    };

    Transformer *model = transformer_create(&config);
    TrainingState *ts = training_state_create(model, 0.01f, 0.9f, 0.999f, 1e-8f);

    Tensor3D src = tensor_create(1, SEQ_LEN, D_MODEL);
    Tensor3D tgt = tensor_create(1, SEQ_LEN, D_MODEL);
    int targets[SEQ_LEN] = {1, 3, 5, 7};

    for (int s = 0; s < SEQ_LEN; s++) {
        for (int d = 0; d < D_MODEL; d++) {
            src.data[s * D_MODEL + d] = (float)targets[s] / VOCAB_SIZE;
            tgt.data[s * D_MODEL + d] = (float)targets[s] / VOCAB_SIZE;
        }
    }

    printf("Starting training (copy task)...\n");
    printf("Targets: ");
    for (int s = 0; s < SEQ_LEN; s++) printf("%d ", targets[s]);
    printf("\n\n");

    for (int epoch = 0; epoch < 10; epoch++) {
        training_state_zero_grads(ts);

        TransformerCache cache = {0};
        Tensor3D output = transformer_forward(model, &src, &tgt, NULL, NULL, &cache);

        LossResult loss = cross_entropy_loss(&output, targets, VOCAB_SIZE);

        transformer_backward(model, &cache, &loss.grad, &tgt, ts);
        training_state_update(ts);

        transformer_cache_free(&cache, config.encoder_layers, config.decoder_layers);
        tensor_free(&output);
        tensor_free(&loss.grad);

        printf("Epoch %d: loss = %.4f\n", epoch, loss.loss);

        if (loss.loss < 0.01f) {
            printf("Loss low enough, stopping early\n");
            break;
        }
    }

    printf("\nTraining complete!\n");

    Tensor3D test_output = transformer_forward(model, &src, &tgt, NULL, NULL, NULL);
    printf("Predictions (argmax): ");
    for (int s = 0; s < SEQ_LEN; s++) {
        int offset = s * D_MODEL;
        int best = 0;
        for (int v = 1; v < VOCAB_SIZE; v++) {
            if (test_output.data[offset + v] > test_output.data[offset + best]) {
                best = v;
            }
        }
        printf("%d ", best);
    }
    printf("\n");

    tensor_free(&test_output);
    tensor_free(&src);
    tensor_free(&tgt);
    training_state_free(ts);
    transformer_free(model);

    return 0;
}