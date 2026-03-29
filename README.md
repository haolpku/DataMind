# DataMind

基于 LlamaIndex 的一体化智能助手，集成五大模块：

- **RAG** — 向量语义检索（Chroma）
- **GraphRAG** — 知识图谱检索（NetworkX）
- **Database** — 自然语言查数据库（SQLite NL2SQL）
- **Skills** — 自定义工具/技能（FunctionTool）
- **Memory** — 对话记忆（短期 + 长期）

Agent 会根据用户问题**自动选择**使用哪个工具，无需手动指定。

支持两种使用方式：**Web 界面**（带可视化管理面板）和**终端命令行**。

> 数据预处理人员请参阅 **[data.md](data.md)** — 说明各模块的数据格式规范和一键导入方式。
>
> 快速体验请参阅 **[demo.md](demo.md)** — 精心设计的示例问题，演示各模块如何被 Agent 自动调用。

---

## 快速开始

```bash
# 1. 创建并激活环境
conda create -n datamind python=3.12
conda activate datamind

# 2. 安装依赖
pip install -r requirements.txt

# 3. 编辑 config.py，填入你的 API Key 和 API Base

# 4. 将文档放入 data/ 目录
```

### 方式一：Web 界面（推荐）

```bash
python server.py
```

打开浏览器访问 **http://localhost:8000**

Web 界面功能：
- 左侧：流式对话（逐字输出）
- 右侧面板（点击顶部 Tab 切换）：
  - **Skills** — 查看所有已加载的技能工具及描述
  - **Memory** — 查看对话记忆内容、一键清空记忆
  - **RAG** — 查看/上传/删除知识库文档、重建向量索引
  - **GraphRAG** — 查看知识图谱的实体和关系、重建图谱
  - **Database** — 查看数据库表结构和数据

前端是纯 HTML + CSS + JS（`static/index.html`），不需要 Node.js 或 npm，由 FastAPI 直接提供服务。

### 方式二：终端命令行

```bash
python main.py
```

在终端中直接交互式对话，功能与 Web 界面完全一致，适合无图形界面的服务器环境。

---

首次运行会自动构建向量索引和知识图谱（需要调用 API），后续启动会直接加载已有索引。

---

## 项目结构

```
DataMind/
├── config.py              # API 配置 (LLM / Embedding / 路径 / Memory 参数)
├── server.py              # Web 入口: FastAPI 后端 + 前端页面
├── main.py                # 终端入口: 交互式命令行对话
├── requirements.txt       # Python 依赖
├── static/
│   └── index.html         # 前端页面 (纯 HTML/CSS/JS, 无需 npm)
├── modules/               # 各功能模块
│   ├── rag/               # RAG 向量检索
│   │   ├── indexer.py     #   文档加载 + Chroma 向量索引
│   │   └── retriever.py   #   检索引擎
│   ├── graphrag/          # GraphRAG 知识图谱检索
│   │   └── graph_rag.py   #   图谱构建 + 查询
│   ├── database/          # Database NL2SQL
│   │   └── database.py    #   SQLite 示例数据库 + NL2SQL
│   ├── skills/            # Skills 技能
│   │   ├── tools.py       #   工具型: 计算器/时间/换算等
│   │   └── knowledge.py   #   知识型: Markdown 技能文档检索
│   ├── memory/            # Memory 对话记忆
│   │   └── memory.py      #   短期 + 长期记忆管理
│   └── agent/             # Agent 智能调度
│       └── agent.py       #   整合所有工具的 FunctionAgent
├── data/                  # 数据目录
│   ├── *.txt/md/pdf/...   #   RAG 原始文档
│   ├── chunks/            #   RAG 预分块数据 (JSONL)
│   ├── triplets/          #   GraphRAG 预构建三元组 (JSONL)
│   └── skills/            #   知识型技能文档 (Markdown)
└── storage/               # 自动生成: 索引/图谱/数据库持久化
```

---

## 如何自定义你的 RAG 数据

RAG 模块会把 `data/` 目录下的所有文档加载、分块、向量化后存入 Chroma 向量数据库。

### 支持的文档格式

PDF, TXT, Markdown, DOCX, CSV, HTML, JSON, EPUB 等（LlamaIndex SimpleDirectoryReader 支持的所有格式）。

### 操作步骤

**通过 Web 界面**：点击 RAG 面板 → 上传文档 → 点击"重建索引"

**通过命令行**：

1. 将文件放入 `data/` 目录（支持子目录递归扫描）

```
data/
├── 公司手册.pdf
├── 技术文档/
│   ├── API说明.md
│   └── 架构设计.txt
└── FAQ.docx
```

2. 删除 `storage/` 目录，重新运行

```bash
rm -rf storage/
python main.py   # 或 python server.py
```

### 调整分块参数

编辑 `modules/rag/indexer.py` 中的 `build_index()` 函数：

```python
splitter = SentenceSplitter(
    chunk_size=512,     # 每个文本块的最大 token 数 (增大可保留更多上下文)
    chunk_overlap=64,   # 相邻块之间的重叠 token 数 (增大可避免关键信息被截断)
)
```

### 调整检索参数

编辑 `modules/rag/retriever.py`：

```python
def create_query_engine(index, similarity_top_k=3):  # 返回最相关的 3 个文档块
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        response_mode="compact",  # 可选: compact / refine / tree_summarize
    )
```

---

## 如何自定义你的 GraphRAG 数据

GraphRAG 从同一个 `data/` 目录读取文档，用 LLM 自动抽取实体和关系，构建知识图谱。

### 它和 RAG 有什么不同？

| | RAG (向量检索) | GraphRAG (图谱检索) |
|---|---|---|
| 适合的问题 | "X 是什么？" | "A 和 B 有什么关系？" |
| 检索方式 | 语义相似度匹配 | 实体关系遍历 |
| 强项 | 单跳事实查找 | 多跳推理、关系理解 |
| 数据存储 | 向量数据库 (Chroma) | 属性图 (NetworkX) |

### 操作步骤

**通过 Web 界面**：点击 GraphRAG 面板 → 查看实体和关系 → 点击"重建图谱"

**通过命令行**：删除 `storage/graph/` 目录后重新运行即可。

### 调整图抽取参数

编辑 `modules/graphrag/graph_rag.py` 中的 `build_graph_index()` 函数：

```python
kg_extractor = SimpleLLMPathExtractor(
    max_paths_per_chunk=10,  # 每个文本块最多抽取的关系路径数
)
```

如果需要更精确的抽取（预定义实体和关系类型），可以替换为 `SchemaLLMPathExtractor`：

```python
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

kg_extractor = SchemaLLMPathExtractor(
    possible_entities=["人物", "公司", "产品", "技术", "地点"],
    possible_relations=["开发了", "属于", "使用", "位于", "合作"],
)
```

### 可视化知识图谱

构建完成后，图谱会保存为 HTML 文件，可以用浏览器打开查看：

```
storage/graph/knowledge_graph.html
```

---

## 如何自定义你的 Database 数据

Database 模块使用 SQLite + SQLAlchemy，支持自然语言转 SQL 查询。

### 当前 Demo 数据

项目自带一个示例数据库（`storage/demo.db`），包含两张表：

**employees 员工表**:

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| name | VARCHAR(50) | 姓名 |
| department | VARCHAR(50) | 部门 |
| position | VARCHAR(50) | 职位 |
| salary | FLOAT | 工资 |
| city | VARCHAR(50) | 城市 |

**projects 项目表**:

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| project_name | VARCHAR(100) | 项目名称 |
| lead_employee_id | INTEGER | 负责人ID |
| budget | FLOAT | 预算 |
| status | VARCHAR(20) | 状态 |

**通过 Web 界面**：点击 Database 面板 → 查看表结构 → 点击"查看数据"展示表内容

### 替换为你自己的数据库

编辑 `modules/database/database.py`，修改 `init_demo_database()` 函数：

**方式 1: 修改表结构和数据**

```python
products = Table(
    "products", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(100)),
    Column("price", Float),
    Column("category", String(50)),
)

conn.execute(products.insert(), [
    {"id": 1, "name": "笔记本电脑", "price": 6999, "category": "电子产品"},
    # ...
])
```

**方式 2: 连接已有数据库**

```python
# SQLite 文件
engine = create_engine("sqlite:///path/to/your/database.db")

# MySQL
engine = create_engine("mysql+pymysql://user:password@host:3306/dbname")

# PostgreSQL
engine = create_engine("postgresql://user:password@host:5432/dbname")
```

然后更新 `create_sql_query_engine()` 中的表名列表：

```python
sql_database = SQLDatabase(engine, include_tables=["your_table1", "your_table2"])
query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["your_table1", "your_table2"],
)
```

同时更新 `modules/agent/agent.py` 中 `database_query` 工具的 `description`，告诉 Agent 你的数据库有哪些表和字段，这样 Agent 才能正确生成 SQL。

### 安全注意事项

- NL2SQL 会执行 LLM 生成的 SQL，存在安全风险
- 建议使用**只读数据库连接**
- 不要将此功能连接到包含敏感数据的生产数据库

---

## 如何添加自己的 Skills

Skills 是最灵活的扩展方式 — 任何 Python 函数都可以变成 Agent 的工具。

### 当前内置 Skills

| 技能 | 函数名 | 说明 |
|------|--------|------|
| 当前时间 | `get_current_time` | 获取日期和时间 |
| 计算器 | `calculator` | 精确数学计算 |
| 文本分析 | `analyze_text` | 统计字数、行数、段落数 |
| 单位换算 | `unit_convert` | 长度、重量、温度换算 |

**通过 Web 界面**：点击 Skills 面板即可查看所有已注册的技能和描述

### 添加新 Skill 的步骤

编辑 `modules/skills/tools.py`：

**第 1 步: 写一个 Python 函数**

```python
def my_new_skill(param1: str, param2: int = 10) -> str:
    """这里写工具描述 - Agent 靠这段文字判断何时调用此工具。
    param1: 参数1的说明
    param2: 参数2的说明，默认值为10
    """
    result = do_something(param1, param2)
    return f"结果: {result}"
```

**关键要求**:
- 函数的 **docstring 是最重要的** — Agent 完全靠它决定何时调用这个工具
- 参数需要有**类型标注**（`str`, `int`, `float` 等）
- 返回值是 `str`（Agent 会将返回值作为工具执行结果）
- docstring 中说清楚**什么场景该用**、**参数含义**

**第 2 步: 注册到 `get_all_skills()`**

```python
def get_all_skills() -> list:
    return [
        FunctionTool.from_defaults(fn=get_current_time),
        FunctionTool.from_defaults(fn=calculator),
        FunctionTool.from_defaults(fn=analyze_text),
        FunctionTool.from_defaults(fn=unit_convert),
        FunctionTool.from_defaults(fn=my_new_skill),   # <-- 加这一行
    ]
```

不需要改其他文件，重启 `python server.py` 或 `python main.py` 即可生效。

### 实用 Skill 示例

**网页搜索**:

```python
import requests

def web_search(query: str) -> str:
    """搜索互联网获取最新信息。当知识库和数据库都没有相关内容，
    或用户问的是实时信息（新闻、天气、股价等）时使用。
    query: 搜索关键词
    """
    resp = requests.get("https://your-search-api.com/search", params={"q": query})
    return resp.json().get("answer", "未找到结果")
```

**文件读取**:

```python
def read_file(file_path: str) -> str:
    """读取本地文件内容。当用户要求查看某个文件的内容时使用。
    file_path: 文件路径
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    if len(content) > 2000:
        content = content[:2000] + "\n... (已截断)"
    return content
```

**代码执行**:

```python
import subprocess

def run_python_code(code: str) -> str:
    """执行 Python 代码并返回输出。当用户需要运行代码或做复杂计算时使用。
    code: 要执行的 Python 代码
    """
    result = subprocess.run(
        ["python", "-c", code],
        capture_output=True, text=True, timeout=10,
    )
    output = result.stdout or result.stderr
    return output[:2000] if output else "(无输出)"
```

### Agent 如何决定调用哪个工具？

1. 收到用户问题
2. 读取所有工具的 `description`（即函数的 docstring）
3. LLM 判断哪个工具的描述最匹配用户意图
4. 自动提取参数并调用该工具
5. 将工具返回结果整合成最终回答

**写好 description 的技巧**:
- 说清楚**什么场景该用**（"当用户问实时信息时"）
- 说清楚**什么场景不该用**（避免和其他工具混淆）
- 参数说明要具体（"city: 城市名称，如'北京'"）

---

## Memory 是怎么工作的

Memory 模块提供对话记忆能力，让 Agent 能理解多轮对话的上下文。

**通过 Web 界面**：点击 Memory 面板 → 查看当前所有对话记忆 → 可一键清空

### 工作原理

```
用户消息 --> 存入短期记忆 (FIFO 队列)
                  |
                  v
         短期记忆满了？ --是--> 溢出的消息发送到长期记忆
                  |                    |
                  否                   v
                  |            LLM 提取关键信息
                  |            存为 MemoryBlock
                  v
          Agent 读取记忆:
          短期记忆(完整消息) + 长期记忆(关键信息摘要)
          一起作为上下文传给 LLM
```

### 短期记忆

- 是一个 **FIFO（先进先出）消息队列**
- 存储最近的对话消息（`ChatMessage` 对象）
- 有 token 上限，超出后最旧的消息被移出
- 默认占总 token 预算的 70%（`CHAT_HISTORY_TOKEN_RATIO = 0.7`）

### 长期记忆

- 当短期记忆溢出时自动触发
- 溢出的消息被发送到 `MemoryBlock` 处理
- MemoryBlock 使用 LLM 从旧消息中提取关键信息（事实、偏好、重要细节）
- 提取的信息以摘要形式长期保存
- 每次对话时，长期记忆摘要 + 短期记忆一起发送给 LLM

### 配置参数

在 `config.py` 中调整：

```python
MEMORY_TOKEN_LIMIT = 30000           # 短期+长期记忆的总 token 上限
CHAT_HISTORY_TOKEN_RATIO = 0.7       # 短期记忆占比 (0.7 = 70%)
```

- `MEMORY_TOKEN_LIMIT`: 总预算。30000 约等于 ~15000 个中文字的对话历史
- `CHAT_HISTORY_TOKEN_RATIO`: 短期 vs 长期的分配比例
  - 0.7 = 21000 tokens 给短期，9000 给长期
  - 调高 → 记住更多最近对话，但长期记忆减少
  - 调低 → 短期记忆更少，但能记住更多历史关键信息

### Memory 的生命周期

```
程序启动 → 创建 Memory 实例 (空的)
    ↓
用户提问 → 消息存入短期记忆
    ↓
Agent 回答 → 回答也存入短期记忆
    ↓
多轮对话... → 短期记忆逐渐填满
    ↓
短期记忆溢出 → 旧消息自动移入长期记忆 (LLM 提取摘要)
    ↓
程序退出 → 当前实现中记忆随程序结束而清空
```

注意：当前版本的 Memory **不会跨 session 持久化**。每次重启程序记忆会重新开始。如需跨 session 的持久化记忆，可以集成 Mem0 或 Zep。

### 多 Session 支持

如果需要为不同用户或场景使用独立记忆：

```python
from rag.memory import create_memory

memory_user_a = create_memory(session_id="user_a")
memory_user_b = create_memory(session_id="user_b")
```

---

## 配置说明

所有配置集中在 `config.py` 中：

```python
# LLM 配置 - 支持任何 OpenAI 兼容 API
LLM_API_BASE = "https://api.deepseek.com/v1"    # API 地址
LLM_API_KEY = "sk-xxx"                           # API Key
LLM_MODEL = "deepseek-chat"                      # 模型名

# Embedding 配置
USE_LOCAL_EMBEDDING = False                       # True = 本地模型, False = API
EMBEDDING_API_BASE = "https://api.deepseek.com/v1"
EMBEDDING_API_KEY = "sk-xxx"
EMBEDDING_MODEL = "text-embedding-3-small"

# 本地 Embedding (USE_LOCAL_EMBEDDING=True 时生效)
LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # CPU 可跑, ~100MB

# Memory 配置
MEMORY_TOKEN_LIMIT = 30000                        # 记忆总 token 预算
CHAT_HISTORY_TOKEN_RATIO = 0.7                    # 短期记忆占比
```

### 兼容的 API 服务

| 服务 | api_base | model 示例 |
|------|----------|-----------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 智谱 | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| 月之暗面 | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| 硅基流动 | `https://api.siliconflow.cn/v1` | `deepseek-ai/DeepSeek-V3` |

不需要 GPU，所有 LLM 推理和 Embedding 生成都通过 API 远程完成。

---

## 重建索引

**通过 Web 界面**：在 RAG 或 GraphRAG 面板中点击"重建索引"/"重建图谱"按钮。

**通过命令行**：

```bash
rm -rf storage/
python main.py   # 或 python server.py
```

这会重建向量索引、知识图谱，并重新创建示例数据库。

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
| Web 前端 | 纯 HTML/CSS/JS | 无需 npm, 零前端依赖 |
