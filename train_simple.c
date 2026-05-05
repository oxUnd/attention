#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VOCAB_SIZE 8
#define SEQ_LEN 4
#define D_MODEL 16

int main() {
    srand(time(NULL));
    printf("=== 简单训练测试 ===\n\n");
    
    // 创建模型
    TransformerConfig config = {
        .d_model = D_MODEL,
        .nhead = 2,
        .d_ff = 32,
        .encoder_layers = 1,
        .decoder_layers = 1,
        .max_len = 10,
        .dropout = 0.0,  // 训练时关闭dropout
        .activation = GELU
    };
    
    Transformer *model = transformer_create(&config);
    Trainer trainer = trainer_create(model, 0.01f, 0.9f, 0.999f, 1e-8f);
    
    // 创建数据
    Tensor3D src = tensor_create(1, SEQ_LEN, D_MODEL);
    Tensor3D tgt = tensor_create(1, SEQ_LEN, D_MODEL);
    int targets[SEQ_LEN] = {1, 3, 5, 7};  // 固定目标
    
    // 初始化输入（用targets作为源）
    for (int s = 0; s < SEQ_LEN; s++) {
        for (int d = 0; d < D_MODEL; d++) {
            src.data[s * D_MODEL + d] = (float)targets[s] / VOCAB_SIZE;
            tgt.data[s * D_MODEL + d] = (float)targets[s] / VOCAB_SIZE;
        }
    }
    
    printf("开始训练（复制任务）...\n");
    printf("初始数据: ");
    for (int s = 0; s < SEQ_LEN; s++) printf("%d ", targets[s]);
    printf("\n\n");
    
    // 训练10轮
    for (int epoch = 0; epoch < 10; epoch++) {
        // Forward
        TransformerCache cache = {0};
        Tensor3D output = transformer_forward(model, &src, &tgt, NULL, NULL, &cache);
        
        // Loss
        LossResult loss = cross_entropy_loss(&output, targets, VOCAB_SIZE);
        
        // Backward（简化版，只更新输出层）
        transformer_backward(model, &cache, &loss.grad, &tgt);
        
        // 清理缓存 - 简化版
        tensor_free(&loss.grad);
        tensor_free(&output);
        
        printf("Epoch %d: loss = %.4f\n", epoch, loss.loss);
        
        if (loss.loss < 0.01f) {
            printf("损失足够低，提前停止\n");
            break;
        }
    }
    
    printf("\n训练完成！\n");
    
    // 测试
    Tensor3D test_output = transformer_forward(model, &src, &tgt, NULL, NULL, NULL);
    printf("预测结果（取argmax）: ");
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
    trainer_free(&trainer);
    transformer_free(model);
    
    return 0;
}
