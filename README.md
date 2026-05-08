# attention

纯 C 实现的 **Transformer**（Encoder–Decoder、多头注意力、前馈与训练循环），附带可插拔的 **Tokenizer** 模块、**Causal Mask** 支持、字符/词级语言模型与若干小型训练演示。底层数学算子已抽离到独立的 `nn_math.h/.c`，便于复用。

## 目录

- [快速开始](#快速开始)
- [代码结构](#代码结构)
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

## 代码结构

```
nn_math.{h,c}      Tensor3D / Matrix 数据结构、激活、Linear、Softmax、LayerNorm、
                   Dropout、Adam、Cross-Entropy / MSE Loss、causal/padding mask 等
                   底层数学原语（与具体模型解耦，方便复用）
transformer.{h,c}  MultiHeadAttention / FeedForward / PositionalEncoding /
                   Encoder / Decoder / Transformer 的前向反向、TrainingState /
                   Trainer 高层封装、checkpoint I/O
tokenizer.{h,c}    char / byte / word 三种分词器 + 特殊符号 + 词表持久化
text_lm.{h,c}      文本语言模型训练循环、采样生成（含 tokenizer 接入）
main.c             带参 CLI（训练 / 加载 / 续写）
train_*.c          各种独立训练演示
```

`transformer.{h,c}` 里的字段命名遵循当前主流框架习惯：`num_heads`、`head_dim`、
`Wq`/`Wk`/`Wv`/`Wo`（attention 投影矩阵）、`W1`/`b1`/`W2`/`b2`（FFN）、
`ln{1,2,3}_{gamma,beta}`（LayerNorm 参数）、`activation`（`ActivationKind` 枚举）。
原先的 `nhead`/`d_k`/`wq`/`GELU` 等旧名一律被替换。

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

> 自从加入了独立的 token embedding + logit head，**`vocab_size ≤ d_model` 的约束已经解除**。词表可以独立放大；输入端走 `(vocab, d_model)` 的 embedding lookup，输出端走 `(d_model, vocab)` 的 logit 头（见下文 [KV Cache 与 embedding 解耦](#kv-cache-与-embedding-解耦)）。

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
| `--kv-cache` / `--no-kv-cache` | 启用/禁用 KV-cache 加速生成（默认禁用，保持向后兼容）。 |
| `--bench-kv-cache` | 对加载的模型跑一次 with/without cache 时间对比并退出。 |

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
| 抽离底层数学算子 | `nn_math.h/.c` | Tensor/Matrix、激活、Linear、Softmax、LayerNorm、Dropout、Adam、Loss、Mask 工具，全部用更可读的命名（`tensor_softmax_inplace`、`linear_forward/backward`、`adam_apply_matrix` 等）。`transformer.{h,c}` 仅依赖 `nn_math.h`，专注于模型组件本身 |
| 重构 transformer | `transformer.{h,c}` | `nhead → num_heads`，`d_k → head_dim`，`wq/wk/wv/wq_out → Wq/Wk/Wv/Wo`，`w1/b1/w2/b2 → W1/b1/W2/b2`，`pe → table`，`Activation → ActivationKind`，去掉 `FeedForward` 中冗余的 `gamma/beta/eps` 字段，attention/encoder/decoder 内部拆出 `split_heads`/`merge_heads`/`apply_attn_mask`/`attention_softmax_inplace` 等小函数 |
| Causal mask 支持 | `nn_math.h/.c`、`transformer.c` | `mask_causal_create`、`mask_padding_create`，注意力中按形状自动识别 `[*, S, S]` 的 query-key mask 与 `[*, 1, S]` 的 padding mask |
| Adam 一步即一次 | `nn_math.h/.c`、`transformer.c` | 拆出 `adam_corrected_lr` + `adam_apply_matrix`，由 `training_state_update` 统一负责 `step` 自增，避免每个参数都重复推进时间步 |
| 接入 tokenizer 的 LM 训练/生成 | `text_lm.{h,c}` | `text_corpus_from_text_with_tokenizer`、`text_lm_generate_with_tokenizer`；训练时同时把 causal mask 传给 encoder self-attn / decoder self-attn / cross-attn，杜绝原本的未来信息泄漏 |
| 主 CLI 重构 | `main.c` | `--tokenizer char/word/byte`、`--vocab-size`、`--d-model`、`--layers`、`--seq-len`、`--batch`、`--epochs`、`--lr`，模型与词表配对保存/加载 |
| 重写文本 demo | `train_text.c` | 精简、可重复的词级/字符级双 demo + held-out 评估 |
| Makefile 与文档 | `Makefile`, `README.md` | 编译加入 `nn_math.o`，依赖关系更新；新增 `make text` 等目标 |

向后兼容：`train_full` / `train_copy_task` / `train_simple` 已同步迁移到新命名（`num_heads`、`ACT_GELU`），运行结果不变。

### 评估示例输出

```text
Epoch    0: avg_loss=2.0325 avg_acc=38.3%
Epoch    9: avg_loss=0.3620 avg_acc=83.5%
Epoch   29: avg_loss=0.3052 avg_acc=85.5%
Held-out eval (32 windows): loss=0.2952 acc=86.3%

--- Char-level samples (T=0.4) ---
Seed: "the cat "
Generated: the cat is happy. the bird is sad. the cat sees the dog. ...
```

模型规模（d_model=64、单层 enc/dec）和小语料的搭配下，词级 LM 大约能学到一阶/二阶词频统计；字符级 LM 在 ~30 epoch 内能把损失从 log(28)≈3.33 降到 ~0.30（≈86% top-1）。

下方"KV Cache 与 embedding 解耦"一节记录了在此基础上的进一步改造（**Pre-LN + warmup/cosine + 解耦 embedding + KV cache**）。剩余优化方向（仍未做）：

- 用 **SIMD/BLAS** 替换三重循环的矩阵乘；
- 学**真正的 BPE / sentencepiece** 分词器（当前 byte-level 已可用，BPE 未实现）。

## KV Cache 与 embedding 解耦

后续工程改造把上面"建议清单"里的前四条做了：

- ✅ **Pre-LayerNorm**：`encoder_layer_forward/backward` 与 `decoder_layer_forward/backward` 都改写成 Pre-LN（每个子模块前归一化，残差直接加在原始张量上）。`train_simple` / `train_full` 的复制任务和 `news.txt` LM 训练全部通过。
- ✅ **warmup + cosine schedule**：`AdamOptimizer` 加 `warmup_steps` / `decay_steps` / `min_factor`，`adam_set_schedule` 配置；`text_lm.c` 默认 2-epoch 线性 warmup 后 cosine 衰减到 10% base lr。
- ✅ **Embedding 表 + logit head**：`TransformerConfig.vocab_size > 0` 时分配 `token_embedding (vocab, d_model)` 与 `logit_head (d_model, vocab)`，旧的 `(d_model, d_model)` 输入/输出投影保留作为 legacy d_model-space 路径（让 `train_simple/full/copy_task` 仍能跑）。新 API：`transformer_forward_lm` / `transformer_backward_lm`，输入是 token-id 数组、输出是 `(B, S, vocab)` 的 logits。`vocab > d_model` 现在不再受限。
- ✅ **KV Cache（增量解码）**：见下文。

对真实新闻语料 `news.txt`（246 unique words / 500 tokens / 13 行 TechCrunch 新闻片段）做的对比：

| 配置 | vocab | best loss | best acc | normalized = loss/log(vocab) |
|------|------:|---------:|---------:|------:|
| 改造前：PostLN，无 schedule | 128 | 2.34 | 41.5% | 0.483 |
| Pre-LN + warmup/cosine | 128 | **1.78** | **48.5%** | **0.367** |
| Pre-LN + sched + 解耦 embedding | 253 | 2.78 | 47.7% | 0.503 |

第二行相对第一行直降 24%。第三行 raw loss 看起来更高是因为 vocab 涨到 253 让 baseline 也涨了；但 inference 输出已经不再有 `<unk>`，能稳定生成 `moonshot ai`、`open- weight ai lab`、`billion dollars` 等专有名词组合。

### KV Cache：API 与加速比

```c
TransformerKVCache *cache = transformer_kv_cache_create(model, max_decode_len);
transformer_lm_init_cache(model, src_ids, src_len, cache);  /* 跑一次 encoder + 预投影 cross K/V */
for (int step = 0; step < N; step++) {
    Tensor3D logits = transformer_lm_step(model, next_token_id, cache);
    int next = sample(logits);
    tensor_free(&logits);
    /* feed `next` next iteration */
}
transformer_kv_cache_free(cache);
```

KV cache 路径只在推理时启用（`--kv-cache` 或调用 `text_lm_generate_with_tokenizer_cached`）。每个 decoder 层维护两份缓存：

- self-attn：`(max_len, d_model)` 大小的 K/V 缓冲，每 step 增量 append 一行
- cross-attn：encoder_out 经各层 cross-attn 的 `Wk` / `Wv` 预投影后存住，`init_cache` 之后整个生成过程不再访问 encoder

正确性校验（greedy，T=10⁻⁴，看下一个采样 token 是否一致）：

```
[MATCH] seed="the company"          no-cache => raised   kv-cache => raised
[MATCH] seed="ai lab"               no-cache => ,        kv-cache => ,
[MATCH] seed="moonshot ai"          no-cache => and      kv-cache => and
[MATCH] seed="moonshot ai is one of" no-cache => open    kv-cache => open
... (7/7 multi-token seeds match exactly, 5/5 single-token seeds match)
```

时间对比（`news_phase2.bin`，d_model=128，1 enc + 1 dec layer，4 heads，seq_len=16）：

| length | no-cache | kv-cache | speedup |
|-------:|---------:|---------:|--------:|
|     50 |  0.118 s |  0.002 s |    54× |
|    100 |  0.250 s |  0.002 s |   106× |
|    200 |  0.520 s |  0.002 s |   228× |
|    400 |  1.077 s |  0.002 s |   490× |

no-cache 时间随 `length` 线性增长（因为每步还要重跑完整 encoder + decoder forward over `seq_cap` 个位置）；cache 路径每步只跑一次增量 decoder step，时间几乎随 length 不变。

### 行为差异（重要）

cache 路径的 encoder 上下文**冻结在 seed**——这是 KV cache 的本质前提（不冻结就得每步重跑 encoder，cache 失去意义）。这跟无 cache 路径的 "src 跟着新生成 token 一起滚动" 的行为略有差异，所以**多步生成下** with-cache 和 without-cache 的文字会发散，但每一步的 logits 在相同 src 下严格一致。

## 其它

`train_test.c` 用于损失函数与单步训练的快速验证，**未**列入当前 `Makefile`；如需单独编译：

```bash
gcc -Wall -Wextra -O2 -std=c99 -c transformer.c -o transformer.o
gcc -Wall -Wextra -O2 -std=c99 train_test.c transformer.o -o train_test -lm
```
