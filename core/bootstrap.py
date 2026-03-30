"""
共享初始化逻辑: main.py / server.py / benchmark.py 统一调用

initialize(cfg) 一次性初始化所有模块，返回 AppState dataclass。
"""

from dataclasses import dataclass, field
from typing import Any

from config import Settings, settings as default_settings

from llama_index.core import Settings as LlamaSettings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from modules.rag.indexer import get_or_create_index
from modules.graphrag.graph_rag import get_or_create_graph_index
from modules.database.database import init_demo_database, create_sql_query_engine
from modules.skills.tools import get_all_skills
from modules.skills.knowledge import get_or_create_skill_index
from modules.agent.agent import create_agent


@dataclass
class AppState:
    llm: Any = None
    embed_model: Any = None
    vector_index: Any = None
    graph_index: Any = None
    db_engine: Any = None
    sql_query_engine: Any = None
    skills: list = field(default_factory=list)
    skill_index: Any = None
    agent: Any = None


def _create_llm(cfg: Settings) -> OpenAI:
    return OpenAI(
        api_base=cfg.llm_api_base,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
    )


def _create_embed_model(cfg: Settings):
    if cfg.use_local_embedding:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name=cfg.local_embedding_model)
        print(f"[INFO] 使用本地 Embedding: {cfg.local_embedding_model}")
    else:
        embed_model = OpenAIEmbedding(
            api_base=cfg.embedding_api_base,
            api_key=cfg.embedding_api_key,
            model_name=cfg.embedding_model,
        )
        print(f"[INFO] 使用远程 Embedding: {cfg.embedding_model}")
    return embed_model


def initialize(cfg: Settings = None) -> AppState:
    """一次性初始化所有模块，返回 AppState。

    Args:
        cfg: Settings 实例，为 None 时使用模块级 settings 单例。
    """
    if cfg is None:
        cfg = default_settings

    state = AppState()

    # 1. LLM + Embedding
    print("[INFO] 初始化配置...")
    state.llm = _create_llm(cfg)
    state.embed_model = _create_embed_model(cfg)
    LlamaSettings.llm = state.llm
    LlamaSettings.embed_model = state.embed_model
    print(f"[INFO] LLM: {cfg.llm_model} @ {cfg.llm_api_base}")

    # 2. RAG 向量索引
    print("\n[INFO] === 加载 RAG 向量索引 ===")
    state.vector_index = get_or_create_index()

    # 3. GraphRAG 图谱索引
    print("\n[INFO] === 加载 GraphRAG 图谱索引 ===")
    try:
        state.graph_index = get_or_create_graph_index()
    except Exception as e:
        print(f"[WARNING] GraphRAG 初始化失败: {e}")
        print("[WARNING] 将跳过 GraphRAG，其余功能正常使用")

    # 4. Database
    print("\n[INFO] === 初始化 Database ===")
    try:
        state.db_engine = init_demo_database()
        state.sql_query_engine = create_sql_query_engine(state.db_engine)
    except Exception as e:
        print(f"[WARNING] Database 初始化失败: {e}")

    # 5. Skills (工具型)
    print("\n[INFO] === 加载工具型 Skills ===")
    state.skills = get_all_skills()
    print(f"[Skills] 已加载 {len(state.skills)} 个工具型技能")

    # 6. Skills (知识型)
    print("\n[INFO] === 加载知识型 Skills ===")
    try:
        state.skill_index = get_or_create_skill_index()
    except Exception as e:
        print(f"[WARNING] 知识型 Skills 初始化失败: {e}")

    # 7. Agent
    print("\n[INFO] === 创建 Agent ===")
    state.agent = create_agent(
        vector_index=state.vector_index,
        graph_index=state.graph_index,
        sql_query_engine=state.sql_query_engine,
        skill_index=state.skill_index,
        extra_tools=state.skills,
        llm=state.llm,
    )

    print("[INFO] DataMind 初始化完成")
    return state


def recreate_agent(state: AppState) -> None:
    """用当前 state 中的组件重建 Agent（索引重建后调用）。"""
    state.agent = create_agent(
        vector_index=state.vector_index,
        graph_index=state.graph_index,
        sql_query_engine=state.sql_query_engine,
        skill_index=state.skill_index,
        extra_tools=state.skills,
        llm=state.llm,
    )
