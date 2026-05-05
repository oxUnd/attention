#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define VOCAB_SIZE 16
#define SEQ_LEN 8
#define D_MODEL 16
#define BATCH_SIZE 4

// 生成随机序列作为训练数据
void generate_batch(Tensor3D *src, int *targets, int batch_size, int seq_len, int vocab_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            // 输入：随机token
            int token = rand() % vocab_size;
            for (int d = 0; d < D_MODEL; d++) {
                src->data[b * seq_len * D_MODEL + s * D_MODEL + d] = 
                    (float)token / vocab_size;
            }
            // 目标：复制输入（简单复制任务）
            targets[b * seq_len + s] = token;
        }
    }
}

// 将预测转换为token ID（取argmax）
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

// 计算准确率
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
    
    printf("=== Transformer 复制任务训练 ===\n");
    printf("词汇表大小: %d, 序列长度: %d, 批次大小: %d\n", VOCAB_SIZE, SEQ_LEN, BATCH_SIZE);
    printf("d_model: %d, 训练轮数: 1000\n\n", D_MODEL);
    
    // 创建模型
    TransformerConfig config = {
        .d_model = D_MODEL,
        .nhead = 2,
        .d_ff = 32,
        .encoder_layers = 2,
        .decoder_layers = 2,
        .max_len = SEQ_LEN + 10,
        .dropout = 0.1,
        .activation = GELU
    };
    
    Transformer *model = transformer_create(&config);
    Trainer trainer = trainer_create(model, 0.001f, 0.9f, 0.999f, 1e-8f);
    
    // 创建训练数据
    Tensor3D src = tensor_create(BATCH_SIZE, SEQ_LEN, D_MODEL);
    Tensor3D tgt = tensor_create(BATCH_SIZE, SEQ_LEN, D_MODEL);
    int targets[BATCH_SIZE * SEQ_LEN];
    
    // 初始生成目标
    generate_batch(&tgt, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
    
    // 训练循环
    int num_epochs = 1000;
    float prev_loss = 1e9;
    int no_improve = 0;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // 生成新批次
        generate_batch(&src, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
        // 复制任务：target = source
        for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_MODEL; i++) {
            tgt.data[i] = src.data[i];
        }
        
        // 前向传播
        TransformerCache cache = {0};
        Tensor3D output = transformer_forward(model, &src, &tgt, NULL, NULL, &cache);
        
        // 计算损失
        LossResult loss = cross_entropy_loss(&output, targets, VOCAB_SIZE);
        
        // 准确率
        float acc = compute_accuracy(&output, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
        
        if (epoch % 100 == 0 || loss.loss < 0.1f) {
            printf("Epoch %d: loss=%.4f, accuracy=%.1f%%\n", epoch, loss.loss, acc);
        }
        
        // 反向传播
        transformer_backward(model, &cache, &loss.grad, &tgt);
        
        // 清理
        tensor_free(&loss.grad);
        tensor_free(&output);
        tensor_free(&cache.src_proj);
        tensor_free(&cache.tgt_proj);
        tensor_free(&cache.encoder_cache.layer_caches[0].x);
        tensor_free(&cache.encoder_cache.layer_caches[0].residual);
        tensor_free(&cache.encoder_cache.layer_caches[0].ln1_out);
        tensor_free(&cache.encoder_cache.layer_caches[0].ff_out);
        tensor_free(&cache.encoder_cache.layer_caches[0].ln2_out);
        tensor_free(&cache.encoder_cache.layer_caches[1].x);
        tensor_free(&cache.encoder_cache.layer_caches[1].residual);
        tensor_free(&cache.encoder_cache.layer_caches[1].ln1_out);
        tensor_free(&cache.encoder_cache.layer_caches[1].ff_out);
        tensor_free(&cache.encoder_cache.layer_caches[1].ln2_out);
        tensor_free(&cache.decoder_cache.layer_caches[0].x);
        tensor_free(&cache.decoder_cache.layer_caches[0].residual1);
        tensor_free(&cache.decoder_cache.layer_caches[0].ln1_out);
        tensor_free(&cache.decoder_cache.layer_caches[0].residual2);
        tensor_free(&cache.decoder_cache.layer_caches[0].ln2_out);
        tensor_free(&cache.decoder_cache.layer_caches[0].ff_out);
        tensor_free(&cache.decoder_cache.layer_caches[0].ln3_out);
        tensor_free(&cache.decoder_cache.layer_caches[0].encoder_out);
        tensor_free(&cache.decoder_cache.layer_caches[1].x);
        tensor_free(&cache.decoder_cache.layer_caches[1].residual1);
        tensor_free(&cache.decoder_cache.layer_caches[1].ln1_out);
        tensor_free(&cache.decoder_cache.layer_caches[1].residual2);
        tensor_free(&cache.decoder_cache.layer_caches[1].ln2_out);
        tensor_free(&cache.decoder_cache.layer_caches[1].ff_out);
        tensor_free(&cache.decoder_cache.layer_caches[1].ln3_out);
        tensor_free(&cache.decoder_cache.layer_caches[1].encoder_out);
        tensor_free(&cache.decoder_cache.encoder_out);
        free(cache.encoder_cache.layer_caches);
        free(cache.decoder_cache.layer_caches);
        
        // 早停检查
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
    
    printf("\n训练完成！\n");
    
    // 测试
    printf("\n=== 测试 ===\n");
    generate_batch(&src, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
    for (int i = 0; i < BATCH_SIZE * SEQ_LEN * D_MODEL; i++) {
        tgt.data[i] = src.data[i];
    }
    
    Tensor3D output = transformer_forward(model, &src, &tgt, NULL, NULL, NULL);
    float acc = compute_accuracy(&output, targets, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE);
    printf("测试准确率: %.1f%%\n", acc);
    
    // 显示一些预测示例
    printf("\n预测示例（前3个样本）:\n");
    for (int b = 0; b < 3 && b < BATCH_SIZE; b++) {
        printf("样本 %d: ", b);
        for (int s = 0; s < SEQ_LEN; s++) {
            int target = targets[b * SEQ_LEN + s];
            int pred = predict_token(&output, b, s, VOCAB_SIZE);
            printf("%d->%d ", target, pred);
        }
        printf("\n");
    }
    
    // 清理
    tensor_free(&src);
    tensor_free(&tgt);
    tensor_free(&output);
    trainer_free(&trainer);
    transformer_free(model);
    
    return 0;
}
