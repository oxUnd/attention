#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    srand(123);
    int targets[2] = {1, 2};
    
    printf("Testing loss functions...\n");
    
    // Create a simple prediction tensor (batch=1, seq=1, vocab_size=16)
    Tensor3D pred = tensor_create(1, 1, 16);
    for (int i = 0; i < 16; i++) pred.data[i] = (float)i / 16.0f;
    
    LossResult loss = cross_entropy_loss(&pred, targets, 16);
    loss_print(&loss, "cross_entropy");
    
    tensor_free(&pred);
    tensor_free(&loss.grad);
    
    printf("Testing trainer...\n");
    TransformerConfig config = {
        .d_model = 16,
        .nhead = 2,
        .d_ff = 32,
        .encoder_layers = 1,
        .decoder_layers = 1,
        .max_len = 10,
        .dropout = 0.1,
        .activation = GELU
    };
    
    Transformer *model = transformer_create(&config);
    Trainer trainer = trainer_create(model, 0.001f, 0.9f, 0.999f, 1e-8f);
    
    Tensor3D src = tensor_create(1, 2, config.d_model);
    Tensor3D tgt = tensor_create(1, 2, config.d_model);
    
    for (int i = 0; i < 1*2*16; i++) {
        src.data[i] = (float)rand() / RAND_MAX - 0.5f;
        tgt.data[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    
    printf("Running one training step...\n");
    trainer_train_step(&trainer, &src, &tgt, targets, 16);
    
    tensor_free(&src);
    tensor_free(&tgt);
    trainer_free(&trainer);
    transformer_free(model);
    printf("Done!\n");
    return 0;
}
