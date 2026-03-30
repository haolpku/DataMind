# 数据预处理规范

本文档面向**数据处理流水线的开发者**，说明如何将处理好的数据导入 DataMind 的四个数据模块。

数据预处理（清洗、转换、标注等）应在上游的数据处理仓库中完成，处理后的输出按照以下规范组织，即可一键导入 DataMind。

---

## Data Profile 机制

DataMind 支持多套知识库共存，每套称为一个 **profile**。通过环境变量 `DATA_PROFILE` 切换：

```bash
# 使用默认 profile
DATA_PROFILE=default python main.py

# 切换到 2wiki 数据集
DATA_PROFILE=2wiki python main.py

# 对比不同分块策略
DATA_PROFILE=2wiki_chunk512 python -m benchmark.run --questions data/bench/2wiki.jsonl
DATA_PROFILE=2wiki_chunk1024 python -m benchmark.run --questions data/bench/2wiki.jsonl
```

每个 profile 的**数据和索引完全隔离**，切换 profile 不需要手动删除索引。

### 目录结构

```
data/
├── profiles/                    ← 知识库（按 profile 隔离）
│   ├── default/                 ← DATA_PROFILE=default
│   │   ├── chunks/*.jsonl       ← RAG 预分块数据
│   │   ├── triplets/*.jsonl     ← GraphRAG 三元组
│   │   ├── tables/*.sql         ← Database SQL 文件
│   │   └── *.txt / *.md / ...   ← 原始文档
│   ├── 2wiki/
│   │   └── chunks/*.jsonl
│   └── {自定义profile}/
│       └── ...
├── bench/                       ← 问题集（跨 profile 共享）
│   ├── questions.jsonl
│   └── 2wiki.jsonl
├── skills/                      ← 技能文档（跨 profile 共享）
│   └── *.md
└── bench_raw/                   ← 原始下载缓存（.gitignore）
```

```
storage/
├── default/                     ← DATA_PROFILE=default 的索引
│   ├── chroma.sqlite3
│   ├── demo.db
│   └── graph/
├── 2wiki/                       ← DATA_PROFILE=2wiki 的索引
│   └── ...
```

- `data/profiles/{profile}/` — profile 级数据，每个 profile 独立
- `data/skills/` — 技能文档，跨 profile 共享
- `data/bench/` — benchmark 问题集，跨 profile 共享
- `storage/{profile}/` — 索引持久化，按 profile 自动隔离

---

## 整体数据流

```
上游数据处理流水线
    │
    ├── 非结构化文档 ──→ profiles/{profile}/ ───→ RAG 方式A (自动分块 + Embedding)
    │                                              GraphRAG 方式A (LLM 自动抽取)
    │
    ├── 预分块数据 ───→ profiles/{profile}/chunks/*.jsonl → RAG 方式B (跳过分块)
    │
    ├── 预构建三元组 ─→ profiles/{profile}/triplets/*.jsonl → GraphRAG 方式B (直接导入)
    │
    ├── SQL 建表脚本 ─→ profiles/{profile}/tables/*.sql → Database (自动建表+导入)
    │
    ├── 技能/SOP 文档 → data/skills/*.md ────→ Skills (知识型技能检索)
    │
    └── 结构化数据 ──→ SQLite 文件 ───────────→ Database (NL2SQL 查询)
```

四个模块的数据**相互独立**，可以只准备其中一个或多个。每个模块的两种方式也可以二选一。

---

## 1. RAG 知识库

### 数据要求

将非结构化文档直接放入 profile 目录（如 `data/profiles/default/`），支持子目录。

### 支持的格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| 纯文本 | `.txt` | 最简单，推荐 UTF-8 编码 |
| Markdown | `.md` | 保留标题结构，分块效果好 |
| PDF | `.pdf` | 自动提取文字，扫描件需先 OCR |
| Word | `.docx` | 自动提取文字和表格 |
| CSV | `.csv` | 每行作为一条记录 |
| HTML | `.html` | 自动提取正文 |
| JSON | `.json` | 需要是文本内容的 JSON |
| EPUB | `.epub` | 电子书格式 |

### 目录结构示例

```
data/profiles/default/
├── 产品文档/
│   ├── API参考手册.md
│   ├── 用户指南.pdf
│   └── 常见问题.txt
├── 技术报告/
│   ├── 2024年度技术报告.docx
│   └── 架构设计文档.md
└── 知识库/
    ├── 公司制度.txt
    └── 培训材料.pdf
```

### 最佳实践

**文档粒度**：
- 每个文件覆盖一个主题，避免单个文件包含大量不相关内容
- 大文件（>100页 PDF）建议按章节拆分为多个文件

**文档格式偏好（检索效果从优到劣）**：
1. **Markdown** — 标题结构清晰，分块效果最好
2. **TXT** — 简洁直接，适合结构化的知识条目
3. **PDF** — 适合已有的正式文档，但提取质量取决于 PDF 本身
4. **DOCX** — 适合已有的 Word 文档

**编码**：所有文本文件统一使用 **UTF-8** 编码

**元数据**：文件名本身会作为元数据，建议用有意义的文件名（如 `API参考手册.md` 而非 `doc1.md`）

### 方式 B：预分块数据（JSONL 格式）

如果上游已完成分块，可以直接提供预分块数据，系统只负责 Embedding + 存储，跳过 SentenceSplitter 分块步骤。

**输入格式：JSON Lines 文件**，放入 `data/profiles/{profile}/chunks/` 目录。

```jsonl
{"text": "LlamaIndex 是一个用于构建 RAG 应用的 Python 框架...", "metadata": {"source": "技术文档.md", "chapter": "概述"}}
{"text": "向量检索通过将文本转换为高维向量来实现语义搜索...", "metadata": {"source": "技术文档.md", "chapter": "检索原理"}}
{"text": "张三是工程部的高级工程师，负责 RAG 项目的开发...", "metadata": {"source": "人员信息.txt"}}
```

每行一个 JSON 对象（字段定义见 `schema/types.py` 中的 `RAGChunk`）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | 是 | chunk 的文本内容 |
| `metadata` | object | 否 | 任意键值对，用于检索时的上下文展示和过滤 |
| `image_path` | string | 否 | 图片文件路径（多模态预留，当前不消费） |
| `modality` | string | 否 | `"text"` / `"image"` / `"text_image"`（多模态预留，默认 `"text"`） |

**自动检测逻辑**：系统启动时按以下优先级加载：
1. 已有索引 → 直接加载（不重新构建）
2. `profiles/{profile}/chunks/*.jsonl` 存在 → 方式 B（预分块，跳过 SentenceSplitter）
3. `profiles/{profile}/` 下有文档 → 方式 A（自动分块）

**分块建议**：
- 每个 chunk 建议 200-1000 字（中文），过长会降低检索精度，过短会丢失上下文
- metadata 中建议包含 `source` 字段标明来源文件，方便排查

### 导入方式

将文件放入 profile 目录（方式 A）或 `profiles/{profile}/chunks/` 目录（方式 B）后：
- Web 界面：RAG 面板 → 点击"重建索引"
- 命令行：`rm -rf storage/{profile}/ && python main.py`

### 数据处理流水线输出示例

```python
# 方式 A: 输出原始文档（系统自动分块 + Embedding）
import os

output_dir = "/path/to/DataMind/data/profiles/default"

for doc in processed_documents:
    filepath = os.path.join(output_dir, f"{doc['title']}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {doc['title']}\n\n")
        f.write(doc["content"])
```

```python
# 方式 B: 输出预分块 JSONL（系统只做 Embedding，跳过分块）
import json, os

chunks_dir = "/path/to/DataMind/data/profiles/default/chunks"
os.makedirs(chunks_dir, exist_ok=True)

with open(os.path.join(chunks_dir, "my_chunks.jsonl"), "w", encoding="utf-8") as f:
    for chunk in pre_chunked_data:
        f.write(json.dumps({
            "text": chunk["text"],
            "metadata": {"source": chunk.get("source", ""), "page": chunk.get("page", "")}
        }, ensure_ascii=False) + "\n")
```

---

## 2. GraphRAG 知识图谱

### 数据要求

GraphRAG 有**两种数据输入方式**：

#### 方式 A：自动抽取（默认）

与 RAG 共享同一个 profile 目录。LLM 会自动从文档中抽取实体和关系，无需额外处理。

适合：文档中包含丰富的实体关系描述（人物、组织、产品、技术之间的关系）。

**优化建议**：如果希望提高图谱抽取质量，建议数据预处理时：
- 确保文档中实体名称一致（如统一用"张三"而非混用"张三"/"小张"/"张经理"）
- 每个段落围绕明确的实体和关系展开
- 减少噪声文本（如页眉页脚、目录、版权声明等）

#### 方式 B：预构建三元组（JSONL 格式）

如果上游已经抽取好了实体和关系，可以直接导入三元组数据，**跳过 LLM 抽取步骤，不消耗任何 API token**。

**输入格式：JSON Lines 文件**，放入 `data/profiles/{profile}/triplets/` 目录。

```jsonl
{"subject": "张三", "relation": "任职于", "object": "工程部"}
{"subject": "工程部", "relation": "负责", "object": "RAG智能助手项目"}
{"subject": "张三", "relation": "担任", "object": "高级工程师"}
{"subject": "RAG智能助手项目", "relation": "使用技术", "object": "LlamaIndex"}
{"subject": "LlamaIndex", "relation": "属于", "object": "Python框架"}
```

每行一个 JSON 对象（字段定义见 `schema/types.py` 中的 `GraphTriple`）：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `subject` | string | 是 | 主体实体 |
| `relation` | string | 是 | 关系类型 |
| `object` | string | 是 | 客体实体 |
| `subject_type` | string | 否 | 主体类型（如 "Person"），默认 "entity" |
| `object_type` | string | 否 | 客体类型（如 "Organization"），默认 "entity" |
| `subject_properties` | object | 否 | 主体附加属性（多模态预留，如 `{"image": "img/a.png"}`） |
| `object_properties` | object | 否 | 客体附加属性（多模态预留） |
| `confidence` | float | 否 | 置信度（默认 1.0） |
| `source` | string | 否 | 来源标识 |

**自动检测逻辑**：系统启动时按以下优先级加载：
1. 已有图索引 → 直接加载
2. `profiles/{profile}/triplets/*.jsonl` 存在 → 方式 B（直接导入，不经过 LLM）
3. `profiles/{profile}/` 下有文档 → 方式 A（LLM 自动抽取）

**导入方式**：将 JSONL 文件放入 profile 的 `triplets/` 目录后，系统会**自动检测并导入**，无需修改任何代码。

### 数据处理流水线输出示例

```python
import json

output_path = "/path/to/DataMind/data/profiles/default/triplets/knowledge_graph.jsonl"

# 上游抽取的三元组
triplets = [
    {"subject": "Python", "relation": "是", "object": "编程语言"},
    {"subject": "LlamaIndex", "relation": "基于", "object": "Python"},
    {"subject": "DataMind", "relation": "使用", "object": "LlamaIndex"},
]

with open(output_path, "w", encoding="utf-8") as f:
    for t in triplets:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")
```

---

## 3. Database 结构化数据

### 数据要求

Database 模块使用 SQLite，系统会**自动检测数据来源**，按以下优先级加载：

1. profile 目录下有 `tables/*.sql` → 执行 SQL 文件建表 + 插入数据
2. 无 SQL 文件 → fallback 到内置 demo 员工数据库

**不需要手动修改任何代码**，系统会自动识别表名并配置 Agent 工具描述。

#### 方式 A：SQL 文件导入（推荐）

将 `.sql` 文件放入 `data/profiles/{profile}/tables/` 目录：

```
data/profiles/mydata/tables/
├── 01_schema.sql     ← 建表语句（按文件名排序执行）
└── 02_data.sql       ← 插入数据
```

SQL 文件示例 (`01_schema.sql`):

```sql
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock INTEGER NOT NULL
);
```

SQL 文件示例 (`02_data.sql`):

```sql
INSERT OR REPLACE INTO products VALUES (1, '笔记本电脑', '电子产品', 6999.0, 50);
INSERT OR REPLACE INTO products VALUES (2, '机械键盘', '外设', 399.0, 200);
INSERT OR REPLACE INTO products VALUES (3, '显示器', '电子产品', 2499.0, 80);
```

SQL 文件按文件名排序依次执行，建议用数字前缀控制顺序。

#### 方式 B：直接提供 SQLite 数据库文件

也可以在上游预处理时直接生成 SQLite 文件，放到 `storage/{profile}/demo.db`。

```python
import sqlite3

db_path = "/path/to/DataMind/storage/default/demo.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS products (...)")
cursor.executemany("INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?)", data)
conn.commit()
conn.close()
```

### 数据类型建议

| Python/CSV 类型 | SQLite 类型 | 说明 |
|----------------|-------------|------|
| int | INTEGER | 整数、ID、数量 |
| float | REAL | 价格、分数、百分比 |
| str | TEXT | 名称、描述、类别 |
| date/datetime | TEXT | 建议存为 `YYYY-MM-DD` 格式字符串 |
| bool | INTEGER | 0 / 1 |

---

## 4. Skills 技能知识

### 数据要求

将操作指南、SOP、领域专业知识等 **Markdown 文件** 放入 `data/skills/` 目录。系统会自动索引，Agent 在遇到相关问题时会通过 `skill_search` 工具检索这些知识。

与 RAG 知识库的区别：
- **RAG 知识库** (`data/`): 通用文档，用于回答关于文档内容的问题
- **Skills 知识库** (`data/skills/`): 操作流程、最佳实践、SOP，用于指导"怎么做"类问题

### 文件格式

标准 Markdown 文件，建议结构：

```markdown
# 技能/SOP 标题

## 适用场景
描述什么时候应该使用这个技能。

## 操作步骤
1. 第一步...
2. 第二步...

## 注意事项
- 注意点 1
- 注意点 2
```

### 目录结构示例

```
data/skills/
├── 数据库运维SOP.md
├── 代码审查指南.md
├── 部署流程.md
└── 故障排查手册.md
```

### 最佳实践

- **一个文件 = 一个技能/流程**，不要把多个不相关的 SOP 放在一个文件中
- **标题清晰**：第一行 `# 标题` 会在前端展示为技能名称
- **包含适用场景**：帮助 Agent 判断何时检索这个技能
- 文件名使用有意义的中文或英文命名

### 导入方式

将 `.md` 文件放入 `data/skills/` 目录后：
- Web 界面：Skills 面板 → 点击"重建索引"
- 命令行：删除 `storage/` 后重启，系统会自动检测并构建索引

### 数据处理流水线输出示例

```python
import os

skills_dir = "/path/to/DataMind/data/skills"
os.makedirs(skills_dir, exist_ok=True)

skills = [
    {"title": "数据库运维SOP", "content": "# 数据库运维 SOP\n\n## 适用场景\n..."},
    {"title": "代码审查指南", "content": "# 代码审查指南\n\n## 审查流程\n..."},
]

for skill in skills:
    filepath = os.path.join(skills_dir, f"{skill['title']}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(skill["content"])
```

---

## 一键导入脚本模板

以下是一个完整的数据预处理输出脚本模板，上游 pipeline 可以直接调用：

```python
"""
数据预处理输出脚本
将处理好的数据写入 DataMind 的 data/ 和 storage/ 目录
"""

import os
import json
import sqlite3

DATAMIND_ROOT = "/path/to/DataMind"
PROFILE = "default"  # 目标 profile 名称
DATA_DIR = os.path.join(DATAMIND_ROOT, "data", "profiles", PROFILE)
STORAGE_DIR = os.path.join(DATAMIND_ROOT, "storage", PROFILE)


def export_rag_documents(documents: list[dict]):
    """
    方式 A: 导出 RAG 原始文档（系统自动分块 + Embedding）

    Args:
        documents: [{"title": "文档标题", "content": "文档内容", "category": "分类"}]
    """
    for doc in documents:
        category_dir = os.path.join(DATA_DIR, doc.get("category", ""))
        os.makedirs(category_dir, exist_ok=True)
        filepath = os.path.join(category_dir, f"{doc['title']}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc["content"])
    print(f"[Export] RAG: 已导出 {len(documents)} 个文档到 {DATA_DIR}")


def export_rag_chunks(chunks: list[dict], filename: str = "chunks.jsonl"):
    """
    方式 B: 导出 RAG 预分块数据（系统只做 Embedding，跳过分块）

    Args:
        chunks: [{"text": "chunk文本", "metadata": {"source": "来源", ...}}]
        filename: 输出文件名
    """
    chunks_dir = os.path.join(DATA_DIR, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    filepath = os.path.join(chunks_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"[Export] RAG Chunks: 已导出 {len(chunks)} 个预分块到 {filepath}")


def export_graph_triplets(triplets: list[dict]):
    """
    导出 GraphRAG 三元组

    Args:
        triplets: [{"subject": "实体A", "relation": "关系", "object": "实体B"}]
    """
    triplet_dir = os.path.join(DATA_DIR, "triplets")
    os.makedirs(triplet_dir, exist_ok=True)
    filepath = os.path.join(triplet_dir, "knowledge_graph.jsonl")
    with open(filepath, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"[Export] GraphRAG: 已导出 {len(triplets)} 条三元组")


def export_skill_documents(skills: list[dict]):
    """
    导出 Skills 技能知识文档

    Args:
        skills: [{"title": "技能标题", "content": "Markdown 内容"}]
    """
    skills_dir = os.path.join(DATAMIND_ROOT, "data", "skills")
    os.makedirs(skills_dir, exist_ok=True)
    for skill in skills:
        filepath = os.path.join(skills_dir, f"{skill['title']}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(skill["content"])
    print(f"[Export] Skills: 已导出 {len(skills)} 个技能文档到 {skills_dir}")


def export_database_tables(tables: dict):
    """
    导出 Database 表数据

    Args:
        tables: {
            "table_name": {
                "columns": {"col1": "TEXT", "col2": "INTEGER", ...},
                "rows": [{"col1": "val1", "col2": 123}, ...]
            }
        }
    """
    os.makedirs(STORAGE_DIR, exist_ok=True)
    db_path = os.path.join(STORAGE_DIR, "demo.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for table_name, table_data in tables.items():
        columns = table_data["columns"]
        col_defs = ", ".join(f"{name} {dtype}" for name, dtype in columns.items())
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({col_defs})")

        if table_data["rows"]:
            col_names = list(columns.keys())
            placeholders = ", ".join(["?"] * len(col_names))
            col_str = ", ".join(col_names)
            for row in table_data["rows"]:
                values = [row.get(c) for c in col_names]
                cursor.execute(
                    f"INSERT OR REPLACE INTO {table_name} ({col_str}) VALUES ({placeholders})",
                    values,
                )

    conn.commit()
    conn.close()
    print(f"[Export] Database: 已导出 {len(tables)} 张表到 {db_path}")


# ---- 使用示例 ----
if __name__ == "__main__":
    # RAG 方式 A: 导出原始文档
    export_rag_documents([
        {"title": "产品介绍", "content": "# 产品介绍\n\n这是一个...", "category": "产品"},
        {"title": "技术架构", "content": "# 技术架构\n\n系统采用...", "category": "技术"},
    ])

    # RAG 方式 B: 导出预分块数据（与方式 A 二选一）
    export_rag_chunks([
        {"text": "LlamaIndex 是一个 Python 框架...", "metadata": {"source": "技术文档.md"}},
        {"text": "向量检索通过语义相似度匹配...", "metadata": {"source": "技术文档.md"}},
    ])

    export_graph_triplets([
        {"subject": "DataMind", "relation": "使用", "object": "LlamaIndex"},
        {"subject": "LlamaIndex", "relation": "基于", "object": "Python"},
    ])

    export_skill_documents([
        {"title": "部署流程", "content": "# 部署流程\n\n## 适用场景\n..."},
        {"title": "故障排查手册", "content": "# 故障排查手册\n\n## 排查步骤\n..."},
    ])

    export_database_tables({
        "products": {
            "columns": {"id": "INTEGER PRIMARY KEY", "name": "TEXT", "price": "REAL"},
            "rows": [
                {"id": 1, "name": "笔记本电脑", "price": 6999.0},
                {"id": 2, "name": "机械键盘", "price": 399.0},
            ],
        }
    })
```

---

## 数据更新策略

| 模块 | 增量更新 | 全量重建 |
|------|---------|---------|
| RAG | 往 profile 目录新增文件后点击"重建索引" | 删除 `storage/{profile}/` 后重启 |
| GraphRAG | 当前仅支持全量重建 | 删除 `storage/{profile}/graph/` 后重启 |
| Skills | 往 `data/skills/` 新增 .md 文件后点击"重建索引" | 删除 skills_knowledge collection 后重启 |
| Database | 直接修改 `storage/{profile}/demo.db` 即时生效 | 删除 `.db` 文件后重启 |

注意：
- 切换 `DATA_PROFILE` 后，索引自动使用新 profile 的 `storage/` 目录，不需要手动删除
- RAG 方式 A 和 GraphRAG 方式 A 重建需要调用 LLM API（Embedding / 实体抽取），大量文档时会消耗较多 token
- RAG 方式 B（预分块）仅消耗 Embedding API token，不涉及 LLM 分块
- GraphRAG 方式 B（预构建三元组）不消耗任何 API token，直接导入图数据库
- Skills 知识索引仅消耗 Embedding API token
- 建议在数据稳定后再执行重建
