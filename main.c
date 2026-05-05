#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    srand(123);
    
    TransformerConfig config = {
        .d_model = 16,
        .nhead = 2,
        .d_ff = 32,
        .encoder_layers = 1,
        .decoder_layers = 1,
        .max_len = 10,
        .dropout = 0.0,
        .activation = GELU
    };

    Transformer *t = transformer_create(&config);

    Tensor3D src = tensor_create(1, 2, config.d_model);
    Tensor3D tgt = tensor_create(1, 2, config.d_model);
    for (int i = 0; i < 1*2*config.d_model; i++) src.data[i] = (float)i / (1*2*config.d_model);
    for (int i = 0; i < 1*2*config.d_model; i++) tgt.data[i] = (float)i / (1*2*config.d_model);

    Tensor3D out = transformer_forward(t, &src, &tgt, NULL, NULL, NULL);
    printf("Output: %d x %d x %d\n", out.batch_size, out.seq_len, out.d_model);

    tensor_free(&out);
    tensor_free(&src);
    tensor_free(&tgt);
    transformer_free(t);
    return 0;
}