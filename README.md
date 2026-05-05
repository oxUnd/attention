# attention

纯 C 实现的 **Transformer**（编码器–解码器、多头注意力、前馈与训练循环），附带字符级语言模型与若干小型训练演示。

## 依赖与构建

- C99 编译器（如 GCC）、`-lm` 数学库  
- 在项目根目录执行：

```bash
make
```

生成以下可执行文件：

| 目标 | 说明 |
|------|------|
| `transformer` | 主程序：从文本训练字符 LM、保存/加载 checkpoint、文本续写 |
| `train_text` | 使用内置英文短语语料训练并采样（无需外部文件） |
| `train_full` | 较大规模的随机「复制」任务训练演示 |
| `train_copy_task` | 另一套复制任务设置（不同 batch/序列等） |
| `train_simple` | 最小单序列复制训练，便于快速 sanity check |

清理：

```bash
make clean
```

Makefile 还提供：

- `make test`：运行 `./transformer --help`  
- `make train`：运行 `./train_full`

## 字符语言模型：`transformer`

语料仅保留 **小写字母 a–z、空格、句点**（共 28 类）；其它字符在构建语料时会被过滤。

训练并保存模型：

```bash
./transformer --train corpus.txt --out model.bin
```

加载并续写（种子中含空格时请加引号）：

```bash
./transformer --load model.bin --generate "the quick " --length 80 --temperature 0.8
```

常用参数：`--out`、`--generate`、`--length`（默认 60）、`--temperature`（默认 0.8）。详见 `./transformer --help`。

默认训练超参在 `text_lm.c` 中的 `text_lm_default_hyperparams`（例如 `seq_len=16`、`batch_size=4`、`d_model=64` 等）。

## 源码结构（概要）

| 文件 | 作用 |
|------|------|
| `transformer.c` / `transformer.h` | 模型、张量、优化器与训练 API |
| `text_lm.c` / `text_lm.h` | 字符语料、LM 训练、生成与 checkpoint 读写 |
| `main.c` | 上述 CLI |
| `train_*.c` | 独立演示程序（见上表） |

## 其它

`train_test.c` 用于损失函数与单步训练的快速验证，**未**列入当前 `Makefile`；若需要可自行编译，例如：

```bash
gcc -Wall -Wextra -O2 -std=c99 -c transformer.c -o transformer.o
gcc -Wall -Wextra -O2 -std=c99 train_test.c transformer.o -o train_test -lm
```
