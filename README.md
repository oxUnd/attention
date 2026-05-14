# attention

纯 C 实现的 Transformer（Encoder–Decoder、多头注意力、KV Cache 增量解码），附带可插拔分词器与交互式前向可视化。

## 快速开始

```bash
make            # 构建全部
make text       # 文本训练 demo
make train      # copy-task 演示
make translate  # 英→中翻译 demo
make clean      # 清理
```

依赖：C99 编译器（GCC/Clang）+ `-lm`

## 代码结构

| 文件 | 职责 |
|------|------|
| `nn_math.{h,c}` | Tensor/Matrix、Linear、Softmax、LayerNorm、Adam、Mask 等底层数学原语 |
| `transformer.{h,c}` | MultiHeadAttention / Encoder / Decoder / Transformer 前向反向、KV cache |
| `tokenizer.{h,c}` | char / byte / word / utf8 分词器 + 词表持久化 |
| `text_lm.{h,c}` | 文本语言模型训练循环与采样生成 |
| `main.c` | 主 CLI：训练 / 加载 / 续写 / bench |
| `visualize.c` | 加载模型，导出前向全过程 HTML 可视化 |
| `train_text.c` | 词级 + 字符级文本训练 demo |
| `train_translate.c` | 英→中翻译 demo（KV cache 加速） |
| `train_{full,copy_task,simple}.c` | 其它训练演示 |

## 使用示例

```bash
# 训练
./transformer --train corpus.txt --tokenizer word --vocab-size 64 \
    --d-model 64 --layers 1,1 --seq-len 16 --batch 16 --epochs 60 --lr 0.003 \
    --out model.bin

# 续写
./transformer --load model.bin --generate "the cat" --length 40 --temperature 0.7

# KV cache 加速推理
./transformer --load model.bin --generate "the cat" --kv-cache

# 可视化
./visualize --load model.bin --seed "the cat" --out viz.html
```

## 特性

- **Pre-LN Transformer** + warmup/cosine schedule
- **Causal Mask**：encoder / decoder self-attn / cross-attn 均支持
- **KV Cache**：增量解码，长序列加速 50×–490×
- **解耦 Embedding**：`vocab_size` 不再受 `d_model` 限制
- **四种分词器**：char / byte / word / utf8，自动保存/加载词表
- **前向可视化**：单文件离线 HTML，时间轴自动播放，热图交互

## 许可

MIT