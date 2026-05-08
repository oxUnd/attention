# attention

纯 C 实现的 **Transformer**（Encoder–Decoder、多头注意力、前馈与训练循环），附带可插拔的 **Tokenizer** 模块、**Causal Mask** 支持、字符/词级语言模型与若干小型训练演示。

## 目录

- [快速开始](#快速开始)
- [可执行文件](#可执行文件)
- [Tokenizer 能力](#tokenizer-能力)
- [字符 / 词级语言模型 CLI](#字符--词级语言模型-cli)
- [对原框架的评估](#对原框架的评估)
- [本次主要改动](#本次主要改动)

---

## 快速开始

```bash
make            # 构建全部可执行文件
make text       # 运行内置文本训练 demo（词级 + 字符级）
make train      # 运行 copy-task 演示
make test       # 打印 transformer 的命令行帮助
make clean      # 清理产物
```

依赖：C99 编译器（GCC/Clang），数学库 `-lm`。

## 可执行文件

| 目标 | 说明 |
|------|------|
| `transformer` | 主程序：可选 char/word/byte 任一 tokenizer，训练并保存模型 + 词表，加载续写 |
| `train_text` | 内置语料的文本训练 demo：同时演示词级和字符级两种 tokenizer |
| `train_full` | 较大规模的随机「复制」任务 |
| `train_copy_task` | 另一套复制任务设置 |
| `train_simple` | 单序列复制 sanity check |

## Tokenizer 能力

新增 `tokenizer.h` / `tokenizer.c` 模块。三种内置分词器，统一加上 4 个特殊符号：

| ID | Token |
|----|-------|
| 0 | `<pad>` |
| 1 | `<unk>` |
| 2 | `<bos>` |
| 3 | `<eos>` |

| Kind | 用途 | 词表大小 |
|------|------|----------|
| `char` | 仅保留指定 charset，自动小写折叠 | 4 + |charset| |
| `byte` | 256 字节级，UTF-8 / 二进制安全 | 4 + 256 = 260 |
| `word` | 从语料按词频构建（去重 + punctuation 拆分），可设上限 | ≤ `--vocab-size` |

API 摘要（详见 `tokenizer.h`）：

```c
Tokenizer *tokenizer_create_char(const char *charset, int lowercase);
Tokenizer *tokenizer_create_byte(void);
Tokenizer *tokenizer_create_word(const char *corpus, int max_vocab, int lowercase);

int tokenizer_encode(const Tokenizer *t, const char *text, int *out, int max_out);
int tokenizer_decode(const Tokenizer *t, const int *ids, int n, char *out, int max_out);

int tokenizer_save(const Tokenizer *t, const char *path);
Tokenizer *tokenizer_load(const char *path);
```

词表磁盘格式为可读纯文本：

```
tokenizer_v1
kind 2
vocab 13
lowercase 1
max_token_len 6
0	<pad>
1	<unk>
2	<bos>
3	<eos>
4	the
5	.
...
```

> 由于本框架的输入仍是 d_model 空间的 one-hot，词表大小受 `vocab_size ≤ d_model` 约束。需要更大词表时请相应放大 `--d-model`。

## 字符 / 词级语言模型 CLI

```bash
./transformer --train corpus.txt \
    --tokenizer word --vocab-size 64 --d-model 64 \
    --layers 1,1 --seq-len 16 --batch 16 --epochs 60 --lr 0.003 \
    --out model.bin
```

训练后会自动保存：

- `model.bin`     — 模型权重（`TransformerConfig` + 各层参数）
- `model.bin.tok` — 与之配套的词表

加载续写：

```bash
./transformer --load model.bin --generate "the cat" --length 40 --temperature 0.7
```

`--load` 时会自动尝试加载 `<model>.tok`；找不到时降级到字符级解码。

完整 CLI：

| 选项 | 说明 |
|------|------|
| `--train <path>` | 训练模式：从文本训练并（可选）保存。 |
| `--load <path>` | 加载模式：读取已保存模型 + 词表。 |
| `--out <path>` | 训练完成后保存到此路径（自动同时写 `<path>.tok`）。 |
| `--generate <seed>` | 生成模式：从 seed 文本采样续写。 |
| `--tokenizer <kind>` | `char` / `word` / `byte`，默认 `char`。 |
| `--vocab-size <n>` | `word` 模式下的词表上限（默认 60）。 |
| `--d-model <n>` | 模型宽度（默认 64，须 ≥ 词表大小）。 |
| `--layers <enc>,<dec>` | 编码/解码层数。 |
| `--seq-len <n>` / `--batch <n>` / `--epochs <n>` / `--lr <f>` | 训练超参。 |
| `--length <n>` | 生成长度（默认 60）。 |
| `--temperature <t>` | 采样温度（默认 0.8）。 |

`./train_text` 一口气演示词级 + 字符级两种 tokenizer 的训练 + 评估 + 采样，使用一段精心设计的小语料（约 800 字符 / 220 词 token），整体耗时 ~70s。

## 对原框架的评估

在加入这次的修改之前，原仓库（`text_lm.c`/`train_text.c`）有如下问题：

1. **编码方式：one-hot in d_model 空间。** 输入为 `(B, S, d_model)` 张量，要求 `vocab_size ≤ d_model`。`input_projection` 是 `(d_model, d_model)` 方阵，实际只有前 `vocab_size` 行/列被用作 embedding，其余是死参数。
2. **输出投影同样是 `(d_model, d_model)`**，cross-entropy 只会读取前 `vocab_size` 维作为 logits，剩余维度白计算。
3. **没有 causal mask。** Decoder 自注意力默认全可见，再加上 `src=tgt` 的训练设置，模型在训练时**理论上可以读取未来 token**。原始 demo 中损失却长期停留在 ~2.85（≈ log 17），说明它甚至没把这个泄漏学利索，反而陷入了次优解。
4. **Encoder–Decoder 架构对自回归语言模型并不合适**：encoder 看到的就是 decoder 的目标内容，cross-attention 不加 mask 就是直接泄漏。
5. **PostLN + Xavier 初始化 + 较小模型**导致训练动力学不稳定（在 `train_full` 上也能观察到 1000 epoch 仍在 2.6–3.0 区间震荡）。
6. **生成是 O(n²) 的**，每生成一个 token 都重新跑整个上下文的前向；没有 KV cache。
7. **大量 `clone_tensor` 复制**：每个 forward / backward 都要克隆若干张量做缓存，吞吐被严重拖累；而且没有 SIMD/BLAS。
8. **API 限制**：注意力 mask 原本只支持 `[batch, key_pos]` 形式（padding mask），无法表达 query 维度的 causal mask。
9. **词表硬编码为 28 字符**（a–z + 空格 + 句点），改动需要重写 `text_lm.c`。
10. **采样辅助数组容量写死为 64**，扩词表会越界。

整体可以把它定位成"教学/玩具实现"：组件齐全（multi-head attention、LayerNorm、PE、Adam、grad-clip、checkpoint），但缺很多让训练真正稳定/高质量收敛的工程细节（PreLN、warmup、KV cache、SIMD 等）。

## 本次主要改动

| 改动 | 文件 | 说明 |
|------|------|------|
| 新增 Tokenizer 模块 | `tokenizer.h/.c` | char/byte/word 三种分词器 + 特殊符号 + save/load |
| Causal mask 支持 | `transformer.h/.c` | `causal_mask_create`、`padding_mask_create`，注意力中按形状自动识别 `[*, S, S]` 的 query-key mask 与 `[*, 1, S]` 的 padding mask |
| 接入 tokenizer 的 LM 训练/生成 | `text_lm.h/.c` | `text_corpus_from_text_with_tokenizer`、`text_lm_generate_with_tokenizer`；训练时同时把 causal mask 传给 encoder self-attn / decoder self-attn / cross-attn，杜绝原本的未来信息泄漏 |
| 主 CLI 重构 | `main.c` | `--tokenizer char/word/byte`、`--vocab-size`、`--d-model`、`--layers`、`--seq-len`、`--batch`、`--epochs`、`--lr`，模型与词表配对保存/加载 |
| 重写文本 demo | `train_text.c` | 精简、可重复的词级/字符级双 demo + held-out 评估 |
| Makefile 与文档 | `Makefile`, `README.md` | 新增 `make text`，并整理用法 |

向后兼容：`train_full` / `train_copy_task` / `train_simple` 不依赖新接口（原本就用 `NULL` mask），编译与运行均不受影响。

### 评估示例输出

```text
Tokenizer{kind=word, vocab=17, max_token_len=6, lowercase=1}
Corpus tokens: 217
Windows per epoch: 208 (13 gradient steps), max epochs: 60, vocab: 17
Epoch    0: avg_loss=2.5389 avg_acc=16.2%
Epoch    2: avg_loss=2.3613 avg_acc=21.2%
...
Held-out eval (32 windows): loss=2.3691 acc=18.4%
```

模型规模（d_model=64、单层 enc/dec）和小语料的搭配下，词级 LM 大约能学到一阶/二阶词频统计；字符级 LM 在 ~30 epoch 内可以从 log(28)≈3.33 降到 ~2.35 损失（约 28% top-1）。这是当前框架的工程上限——若需要进一步提升，建议：

- 改成 **Pre-LayerNorm**（最大收益）；
- 引入 **KV cache** 加速生成；
- 用 **embedding 表 + 真正的 logit head**（解耦 vocab_size 与 d_model）；
- 用 **SIMD/BLAS** 替换三重循环的矩阵乘；
- 加入 **warmup + cosine schedule**。

## 其它

`train_test.c` 用于损失函数与单步训练的快速验证，**未**列入当前 `Makefile`；如需单独编译：

```bash
gcc -Wall -Wextra -O2 -std=c99 -c transformer.c -o transformer.o
gcc -Wall -Wextra -O2 -std=c99 train_test.c transformer.o -o train_test -lm
```
