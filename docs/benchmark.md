# Benchmark 使用指南

## 概述

`benchmark/` 包用于对 DataMind 进行并发推理测评和答案评估，直接调用 Python API（不经过 HTTP），支持：

- 可配置的并发数
- 每个请求独立的 Session 隔离（无记忆交叉污染）
- 实时进度条
- 自动统计延迟分布（Avg / P50 / P90 / P95 / Max）和吞吐量（QPS）
- 通过环境变量灵活切换模型、检索策略等配置

---

## 数据准备

Benchmark 需要两部分数据：**知识库文档**和**问题集**。

### 1. 知识库文档

知识库的准备方式与正常使用 DataMind 完全一致，参见 [data.md](data.md)。

简单来说：
- **原始文档**放入 `data/profiles/{profile}/` 目录（方式 A，系统自动分块）
- **预分块数据**放入 `data/profiles/{profile}/chunks/` 目录（方式 B，JSONL 格式）

```
data/profiles/2wiki/chunks/corpus.jsonl    ← 每行一个 {"text": "...", "metadata": {...}}
```

通过 `DATA_PROFILE` 环境变量指定使用哪个 profile，索引会自动隔离，不需要手动删除 `storage/`。

格式与 data.md 中 RAG 方式 B 完全相同：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | 是 | chunk 的文本内容 |
| `metadata` | object | 否 | 任意键值对（来源、章节等） |

**关于 metadata**：metadata **不参与向量检索的相似度计算**（检索只看 text 的 embedding），但会随检索结果一起传递给 LLM 作为上下文。比如填了 `{"source": "技术文档.md", "chapter": "概述"}`，LLM 在生成回答时就能看到这些来源信息。对于 benchmark 场景，metadata 不影响测评结果，填一个 `source` 方便排查即可，也可以完全不填：

```jsonl
{"text": "chunk content here..."}
{"text": "another chunk...", "metadata": {"source": "2wikimultihop"}}
```

> **注意**：切换 profile 即可使用不同知识库，不需要手动清除索引。如需在同一 profile 下重建索引：`rm -rf storage/{profile}/`。

### 2. 问题集

问题集为 **JSONL 格式**，放在任意位置，运行时通过 `--questions` 参数指定路径。

```
data/bench/questions.jsonl    ← 每行一个 JSON
```

**最简格式**（仅测延迟和吞吐）：

```jsonl
{"question": "RAG的核心原理是什么？"}
{"question": "张三的工资是多少？"}
{"question": "代码审查前需要做哪些自查？"}
```

**带参考答案**（可用于后续准确率评估）：

```jsonl
{"question": "When was X born?", "reference_answer": "1982", "question_id": "q_001"}
{"question": "Who directed film Y?", "reference_answer": "John Doe", "question_id": "q_002"}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | 是 | 问题文本 |
| `reference_answer` | string | 否 | 标准答案（不影响运行，仅用于事后分析） |
| `question_id` | string | 否 | 问题 ID（不影响运行，仅用于事后分析） |

如果文件不是 JSON 格式，每行纯文本也会被当作一个问题。

---

## 运行

```bash
# 基础用法（默认 5 并发，使用 default profile）
python -m benchmark.run --questions data/bench/questions.jsonl

# 指定并发数
python -m benchmark.run --questions data/bench/questions.jsonl --concurrency 30

# 指定输出文件
python -m benchmark.run --questions data/bench/questions.jsonl --concurrency 50 --output results.json
```

### 通过环境变量切换配置

```bash
# 切换 data profile（使用不同的知识库）
DATA_PROFILE=2wiki python -m benchmark.run --questions data/bench/2wiki.jsonl

# 切换检索模式
RETRIEVER_MODE=multi_query python -m benchmark.run --questions data/bench/questions.jsonl

# 切换模型
LLM_MODEL=deepseek-chat python -m benchmark.run --questions data/bench/questions.jsonl

# 调整检索 top_k
SIMILARITY_TOP_K=5 python -m benchmark.run --questions data/bench/questions.jsonl

# 组合: 不同 profile + 不同检索策略
DATA_PROFILE=2wiki RETRIEVER_MODE=multi_query SIMILARITY_TOP_K=5 LLM_MODEL=gpt-4o \
  python -m benchmark.run --questions data/bench/2wiki.jsonl --concurrency 50
```

所有可用环境变量参见 `.env.example`。

---

## 输出

### 终端输出

```
[INFO] 加载了 1000 个问题，并发数: 50
[INFO] DataMind 初始化完成

  Running 1000 queries (concurrency=50) ...
  [████████████████████████████████████████] 1000/1000 (100.0%)

==================================================
  Benchmark Results
==================================================
  Total queries:  1000
  Concurrency:    50
  Errors:         0
  Wall time:      168.090s
--------------------------------------------------
  Avg latency:    8.095s
  Min latency:    1.231s
  P50 latency:    7.036s
  P90 latency:    12.273s
  P95 latency:    15.612s
  Max latency:    46.291s
  Throughput:     5.95 QPS
==================================================
```

### JSON 结果文件

每条记录包含：

```json
{
  "index": 0,
  "question": "Where does X's wife work at?",
  "answer": "According to the information...",
  "error": null,
  "latency_s": 5.632,
  "reference_answer": "Sunday Times",
  "question_id": "9d054e98..."
}
```

`reference_answer` 和 `question_id` 仅在问题集中包含这些字段时才会出现。

---

## 答案评估

当问题集包含 `reference_answer` 时，可使用评估脚本对比生成答案与标准答案：

```bash
python -m benchmark.evaluate benchmark_results.json
```

评估指标：

| 指标 | 说明 |
|------|------|
| **Exact Match (EM)** | 标准答案（normalize 后）是否完整出现在生成答案中 |
| **Token F1** | 基于 token 重叠计算的 F1 Score，适用于答案较长或表述不一致的场景 |

终端会输出汇总报告和未命中的问题列表，同时自动保存 JSON 评估报告：

```
=======================================================
  Evaluation Report
=======================================================
  Total evaluated:   1000
  Exact Match:       360/1000 (36.0%)
  Avg Token-F1:      0.4521
=======================================================
```

指定输出路径：

```bash
python -m benchmark.evaluate benchmark_results.json --output my_eval.json
```

---

## 使用公开 RAG 数据集

推荐使用 [A-RAG Benchmark](https://huggingface.co/datasets/Ayanami0730/rag_test)，它提供了现成的 chunks 和 questions，只需转为上述格式即可。

### 可用数据集

| 数据集 | Chunks | Questions | 特点 |
|--------|--------|-----------|------|
| `2wikimultihop` | 658 | 1,000 | 多跳推理，体积最小，推荐入门 |
| `hotpotqa` | 1,311 | 1,000 | 多跳推理 |
| `musique` | 1,354 | 1,000 | 2-4 跳推理 |
| `medical` | 225 | 2,062 | 医学领域 |
| `novel` | 1,117 | 2,010 | 长文本文学 |

### 下载

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
for f in ['chunks.json', 'questions.json']:
    hf_hub_download('Ayanami0730/rag_test', f'2wikimultihop/{f}',
                    repo_type='dataset', local_dir='data/bench_raw')
"
```

### 数据集原始格式

下载后的文件格式如下，需要转换后才能被 DataMind 使用：

**chunks.json** — JSON 数组，每项是 `"id:text"` 格式的字符串：

```json
["0:teutberga (died 11 november...", "1:##lus the little pfalzgraf..."]
```

**questions.json** — JSON 数组，每项包含 question 和 answer：

```json
[{"id": "xxx", "question": "When did X happen?", "answer": "1982", ...}]
```

### 转换为 DataMind 格式

将 chunks 转为 `data/profiles/2wiki/chunks/*.jsonl`（每行一个 `{"text": "...", "metadata": {...}}`），将 questions 转为 `data/bench/*.jsonl`（每行一个 `{"question": "..."}`）。

具体的转换脚本取决于你的数据集格式和需要的 metadata，这里不做硬性规定。转换后的目录结构应该是：

```
data/
├── profiles/2wiki/
│   └── chunks/
│       └── 2wiki_chunks.jsonl       ← 知识库 chunks (JSONL)
└── bench/
    └── 2wiki.jsonl                  ← 问题集 (JSONL)
```

### 运行

```bash
DATA_PROFILE=2wiki python -m benchmark.run --questions data/bench/2wiki.jsonl --concurrency 50 --output benchmark_2wiki.json
```

---

## 参考数据

以下为 2WikiMultiHop 数据集（658 chunks, gpt-4o）在不同并发下的实测结果：

| 并发数 | 问题数 | 错误 | Wall Time | Avg Latency | P50 | P95 | 吞吐量 |
|--------|--------|------|-----------|-------------|-----|-----|--------|
| 3 | 20 | 0 | 56.1s | 6.32s | 4.75s | 32.99s | 0.36 QPS |
| 30 | 20 | 0 | 16.1s | 7.33s | 7.28s | 16.12s | 1.24 QPS |
| 50 | 1000 | 0 | 168.1s | 8.10s | 7.04s | 15.61s | 5.95 QPS |

准确率（reference answer 包含在回复中）：**36.0%**

> 2WikiMultiHop 是多跳推理数据集，单次向量检索 top_k=3 天然较难。可通过 `RETRIEVER_MODE=multi_query` 或增大 `SIMILARITY_TOP_K` 来提升召回率。
