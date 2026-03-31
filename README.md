# DataMind

基于 LlamaIndex 的一体化智能助手，集成五大模块：

- **RAG** — 向量语义检索（Chroma），支持多模态（CLIP / VLM 文本化）
- **GraphRAG** — 知识图谱检索（NetworkX）
- **Database** — 自然语言查数据库（SQLite NL2SQL）
- **Skills** — 自定义工具/技能（FunctionTool）
- **Memory** — 对话记忆（短期 + 长期）

Agent 会根据用户问题**自动选择**使用哪个工具，无需手动指定。

支持两种使用方式：**Web 界面**（带可视化管理面板）和**终端命令行**。

> 完整文档请访问 **[DataMind-Doc](https://haolpku.github.io/DataMind-Doc/zh/)** — 包含各模块详细说明、数据格式规范、演示指南和 Benchmark 使用指南。

---

## 快速开始

```bash
# 1. 创建并激活环境
conda create -n datamind python=3.12
conda activate datamind

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 API Key 和 API Base

# 4. 将文档放入 data/profiles/default/ 目录
```

### Web 界面（推荐）

```bash
python server.py
```

打开浏览器访问 **http://localhost:8000**。左侧为流式对话，右侧面板可管理 RAG / GraphRAG / Database / Skills / Memory。

### 终端命令行

```bash
python main.py
```

首次运行会自动构建向量索引和知识图谱（需要调用 API），后续启动直接加载已有索引。

---

## 各模块简介

| 模块 | 功能 | 详细文档 |
|------|------|---------|
| **RAG** | 文档向量化 + 语义检索，支持多模态 (CLIP / VLM) | [RAG 文档](https://haolpku.github.io/DataMind-Doc/zh/) |
| **GraphRAG** | 知识图谱实体关系检索，多跳推理 | [GraphRAG 文档](https://haolpku.github.io/DataMind-Doc/zh/) |
| **Database** | 自然语言转 SQL，支持 SQL 文件自动导入 | [Database 文档](https://haolpku.github.io/DataMind-Doc/zh/) |
| **Skills** | 工具型（Python 函数）+ 知识型（Markdown 检索） | [Skills 文档](https://haolpku.github.io/DataMind-Doc/zh/) |
| **Memory** | FIFO 短期记忆 + LLM 摘要长期记忆 | [Memory 文档](https://haolpku.github.io/DataMind-Doc/zh/) |

---

## 数据管理

通过 `DATA_PROFILE` 环境变量管理多套知识库，数据和索引完全隔离：

```bash
DATA_PROFILE=default python main.py     # 默认 profile
DATA_PROFILE=mydata python main.py      # 切换 profile
```

数据放入对应目录即可：

```
data/profiles/{profile}/
├── *.txt / *.md / *.pdf    → RAG 原始文档（自动分块）
├── chunks/*.jsonl          → RAG 预分块（跳过分块）
├── triplets/*.jsonl        → GraphRAG 三元组
├── tables/*.sql            → Database SQL 文件
└── images/                 → 多模态图片
```

详细的数据格式规范请参阅 [数据格式文档](https://haolpku.github.io/DataMind-Doc/zh/)。

---

## 重建索引

**Web 界面**：RAG / GraphRAG 面板中点击"重建索引"按钮。

**命令行**：

```bash
rm -rf storage/default/
python main.py
```

---

## Benchmark

内置并发推理测评，支持准确率评估：

```bash
python -m benchmark.run --questions data/bench/2wiki.jsonl --concurrency 50
python -m benchmark.evaluate benchmark_results.json
```

详见 [Benchmark 文档](https://haolpku.github.io/DataMind-Doc/zh/)。

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| 框架 | LlamaIndex | 核心编排 |
| LLM | OpenAI 兼容 API | 不需要 GPU |
| 向量数据库 | Chroma | 本地, 纯 Python |
| 知识图谱 | NetworkX | 本地, 纯 Python |
| 关系数据库 | SQLite | 零配置 |
| Agent | FunctionAgent | 自动工具选择 |
| Web 后端 | FastAPI | 异步, SSE 流式输出 |
| Web 前端 | 纯 HTML/CSS/JS | 无需 npm |
