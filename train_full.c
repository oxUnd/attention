#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VOCAB_SIZE 16
#define SEQ_LEN 4
#define D_MODEL 16
#define BATCH_SIZE 8
#define NUM_EPOCHS 500

// 生成复制任务批次
void generate_copy_batch(Tensor3D *src, Tensor3D *tgt, int *targets, int batch_size, int seq_len, int vocab_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int token = rand() % vocab_size;
            targets[b * seq_len + s] = token;
            // 源和目标都是相同的one-hot编码（简化）
            for (int d = 0; d < D_MODEL; d++) {
                float val = (d == token % D_MODEL) ? 1.0f : 0.0f;
                src->data[b * seq_len * D_MODEL + s * D_MODEL + d] = val;
                tgt->data[b * seq_len * D_MODEL + s * D_MODEL + d] = val;
            }
        }
    }
}

// 计算准确率
float compute_accuracy(Tensor3D *output, int *targets, int batch_size, int seq_len, int vocab_size) {
    int correct = 0;
    int total = batch_size * seq_len;
    
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            // 找到输出中最大值的索引（简化版）
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
    printf("=== Transformer 复制任务训练 ===\n");
    printf("词汇表: %d, 序列: %d, Batch: %d, d_model: %d\n\n", VOCAB_SIZE, SEQ_LEN, BATCH_SIZE, D_MODEL);
    
    // 创建模型
    TransformerConfig config = {
        .d_model = D_MODEL,
        .nhead = 2,
        .d_ff = 32,
        .encoder_layers = 1,
        .decoder_layers = 1,
        .max_len = SEQ_LEN + 5,
        .dropout = 0.0,  // 训练时关闭dropout
        .activation = GELU
    };
    
    Transformer *model = transformer_create(&config);
    Trainer trainer = trainer_create(model, 0.001f, 0.9f, 0.999f, 1e-8f);
    
    // 创建数据
    Tensor3D src = tensor_create(BATCH_SIZE, SEQ_LEN, D_MODEL);
    Tensor3D tgt = tensor_create(BATCH_SIZE, SEQ_LEN, D_MODEL);
    int targets[BATCH_SIZE * SEQ_LEN];
    
    // 训练循环
    float prev_loss = 1e9;
    int no_improve = 0;
    
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        // 生成批次
        generate_copy_batch(&src, &tgt, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
        
        // 前向传播
        TransformerCache cache = {0};
        Tensor3D output = transformer_forward(model, &src, &tgt, NULL, NULL, &cache);
        
        // 计算损失
        LossResult loss = cross_entropy_loss(&output, targets, VOCAB_SIZE);
        
        // 准确率
        float acc = compute_accuracy(&output, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
        
        // 每50轮打印
        if (epoch % 50 == 0 || loss.loss < 0.5f) {
            printf("Epoch %d: loss=%.4f, acc=%.1f%%\n", epoch, loss.loss, acc);
        }
        
        // 反向传播
        transformer_backward(model, &cache, &loss.grad, &tgt);
        
        // 清理
        tensor_free(&output);
        tensor_free(&loss.grad);
        
        // 早停检查
        if (loss.loss < prev_loss - 0.001f) {
            no_improve = 0;
        } else {
            no_improve++;
        }
        prev_loss = loss.loss;
        
        if (no_improve > 100 || loss.loss < 0.1f) {
            printf("提前停止 at epoch %d: loss=%.4f\n", epoch, loss.loss);
            break;
        }
    }
    
    printf("\n=== 训练完成 ===\n\n");
    
    // 测试
    printf("测试阶段...\n");
    generate_copy_batch(&src, &tgt, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
    Tensor3D test_output = transformer_forward(model, &src, &tgt, NULL, NULL, NULL);
    float test_acc = compute_accuracy(&test_output, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
    printf("测试准确率: %.1f%%\n", test_acc);
    
    // 显示一些预测示例
    printf("\n预测示例 (前3个样本):\n");
    for (int b = 0; b < 3 && b < BATCH_SIZE; b++) {
        printf("样本 %d: ", b);
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
    
    // 清理
    tensor_free(&src);
    tensor_free(&tgt);
    tensor_free(&test_output);
    trainer_free(&trainer);
    transformer_free(model);
    
    return 0;
}
