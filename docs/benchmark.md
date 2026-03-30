# DataMind Benchmark 使用指南

## 概述

`benchmark.py` 用于对 DataMind 进行并发推理测评，直接调用 Python API（不经过 HTTP），支持：

- 可配置的并发数
- 每个请求独立的 Session 隔离（无记忆交叉污染）
- 自动统计延迟分布（Avg / P50 / P90 / P95 / Max）和吞吐量（QPS）
- 通过环境变量灵活切换模型、检索策略等配置

## 快速开始

### 1. 准备环境

```bash
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入真实的 API Key
```

### 2. 准备问题集

问题集为 JSONL 格式，每行一个 JSON 对象：

```json
{"question": "RAG的核心原理是什么？"}
{"question": "张三的工资是多少？"}
{"question": "代码审查前需要做哪些自查？"}
```

可选字段（不影响运行，仅用于后续分析）：

```json
{"question": "When did X happen?", "reference_answer": "1982", "question_id": "q_001"}
```

### 3. 运行 Benchmark

```bash
# 基础用法（默认 5 并发）
python benchmark.py --questions data/bench_questions.jsonl

# 指定并发数
python benchmark.py --questions data/bench_questions.jsonl --concurrency 30

# 指定输出文件
python benchmark.py --questions data/bench_questions.jsonl --concurrency 50 --output results.json
```

### 4. 通过环境变量切换配置

```bash
# 切换检索模式为 multi-query
RETRIEVER_MODE=multi_query python benchmark.py --questions data/bench_questions.jsonl

# 切换 LLM 模型
LLM_MODEL=deepseek-chat python benchmark.py --questions data/bench_questions.jsonl

# 调整检索 top_k
SIMILARITY_TOP_K=5 python benchmark.py --questions data/bench_questions.jsonl

# 组合使用
RETRIEVER_MODE=multi_query SIMILARITY_TOP_K=5 LLM_MODEL=gpt-4o \
  python benchmark.py --questions data/bench_questions.jsonl --concurrency 50
```

所有可用环境变量参见 `.env.example`。

## 使用真实 RAG Benchmark 数据集

以 [A-RAG / 2WikiMultiHop](https://huggingface.co/datasets/Ayanami0730/rag_test) 为例：

### 下载数据

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
for f in ['chunks.json', 'questions.json']:
    hf_hub_download('Ayanami0730/rag_test', f'2wikimultihop/{f}',
                    repo_type='dataset', local_dir='data/bench_raw')
"
```

### 转换格式

```python
import json

# 1. chunks → DataMind JSONL 格式
with open('data/bench_raw/2wikimultihop/chunks.json') as f:
    raw = json.load(f)

with open('data/chunks/2wiki_chunks.jsonl', 'w') as out:
    for item in raw:
        idx = item.index(':')
        chunk_id, text = item[:idx], item[idx+1:].strip()
        out.write(json.dumps({
            'text': text,
            'metadata': {'source': '2wikimultihop', 'chunk_id': chunk_id}
        }, ensure_ascii=False) + '\n')

# 2. questions → bench JSONL
with open('data/bench_raw/2wikimultihop/questions.json') as f:
    qs = json.load(f)

with open('data/bench_2wiki.jsonl', 'w') as out:
    for q in qs:
        out.write(json.dumps({
            'question': q['question'],
            'reference_answer': q['answer'],
            'question_id': q['id'],
        }, ensure_ascii=False) + '\n')
```

### 清除旧索引并运行

```bash
# 清除旧索引（重要：换数据后必须清除）
rm -rf storage/

# 运行（首次会自动建索引）
python benchmark.py --questions data/bench_2wiki.jsonl --concurrency 50 --output benchmark_2wiki.json
```

## 输出格式

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
  "question": "RAG的核心原理是什么？",
  "answer": "RAG 的核心原理是...",
  "error": null,
  "latency_s": 5.632
}
```

## 参考数据

以下为 2WikiMultiHop 数据集（658 chunks, 1000 questions）在 gpt-4o 下的实测结果：

| 并发数 | 问题数 | 错误 | Wall Time | Avg Latency | P50 | P95 | 吞吐量 |
|--------|--------|------|-----------|-------------|-----|-----|--------|
| 3 | 20 | 0 | 56.1s | 6.32s | 4.75s | 32.99s | 0.36 QPS |
| 30 | 20 | 0 | 16.1s | 7.33s | 7.28s | 16.12s | 1.24 QPS |
| 50 | 1000 | 0 | 168.1s | 8.10s | 7.04s | 15.61s | 5.95 QPS |

准确率（reference answer 包含在回复中）：**36.0%**（单次向量检索 top_k=3，multi-hop 问题天然较难）。

## 可用数据集

| 数据集 | 来源 | Chunks | Questions | 难度 |
|--------|------|--------|-----------|------|
| 2wikimultihop | A-RAG | 658 | 1,000 | 多跳推理 |
| hotpotqa | A-RAG | 1,311 | 1,000 | 多跳推理 |
| musique | A-RAG | 1,354 | 1,000 | 2-4 跳推理 |
| medical | A-RAG | 225 | 2,062 | 医学领域 |
| novel | A-RAG | 1,117 | 2,010 | 长文本文学 |

均可从 `Ayanami0730/rag_test` 下载，格式相同。
