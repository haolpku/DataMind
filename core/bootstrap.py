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


_MAX_TOOL_CALL_ID_LEN = 64


def _patch_long_tool_call_ids():
    """某些 OpenAI 兼容 API 生成的 tool_call_id 超过 64 字符上限，
    导致后续请求被拒绝。patch ToolCallBlock.__init__ 在创建时自动截断。"""
    from llama_index.core.base.llms.types import ToolCallBlock

    _orig_init = ToolCallBlock.__init__

    def _truncating_init(self, **kwargs):
        tcid = kwargs.get("tool_call_id")
        if tcid and len(tcid) > _MAX_TOOL_CALL_ID_LEN:
            kwargs["tool_call_id"] = tcid[:_MAX_TOOL_CALL_ID_LEN]
        _orig_init(self, **kwargs)

    ToolCallBlock.__init__ = _truncating_init


_patch_long_tool_call_ids()

from modules.rag.indexer import get_or_create_index
from modules.graphrag.graph_rag import get_or_create_graph_index
from modules.database.database import init_database, create_sql_query_engine
from modules.skills.tools import get_all_skills
from modules.skills.knowledge import get_or_create_skill_index
from modules.agent.agent import create_agent


@dataclass
class AppState:
    llm: Any = None
    embed_model: Any = None
    image_embed_model: Any = None
    multimodal_llm: Any = None
    vector_index: Any = None
    graph_index: Any = None
    db_engine: Any = None
    db_table_names: list = field(default_factory=list)
    sql_query_engine: Any = None
    skills: list = field(default_factory=list)
    skill_index: Any = None
    agent: Any = None
    last_retrieved_images: list = field(default_factory=list)


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


def _create_image_embed_model(cfg: Settings):
    """CLIP 模式时创建 ClipEmbedding，其他模式返回 None。"""
    if cfg.image_embedding_mode != "clip":
        return None
    from llama_index.embeddings.clip import ClipEmbedding
    model = ClipEmbedding(model_name=cfg.clip_model)
    print(f"[INFO] 使用 CLIP Embedding: {cfg.clip_model}")
    return model


def _create_multimodal_llm(cfg: Settings):
    """当 use_multimodal_llm=True 时创建 OpenAIMultiModal LLM，否则返回 None。"""
    if not cfg.use_multimodal_llm:
        return None
    from llama_index.multi_modal_llms.openai import OpenAIMultiModal
    model_name = cfg.vlm_model or cfg.llm_model
    mm_llm = OpenAIMultiModal(
        model=model_name,
        api_base=cfg.llm_api_base,
        api_key=cfg.llm_api_key,
        max_new_tokens=1024,
    )
    print(f"[INFO] 多模态 LLM: {model_name}")
    return mm_llm


def initialize(cfg: Settings = None) -> AppState:
    """一次性初始化所有模块，返回 AppState。

    Args:
        cfg: Settings 实例，为 None 时使用模块级 settings 单例。
    """
    if cfg is None:
        cfg = default_settings

    state = AppState()

    # 0. Profile info
    print(f"[INFO] Data profile: {cfg.data_profile}")
    print(f"[INFO] Data dir:     {cfg.data_dir}")
    print(f"[INFO] Storage dir:  {cfg.storage_dir}")

    # 1. LLM + Embedding (text + image)
    print("[INFO] 初始化配置...")
    state.llm = _create_llm(cfg)
    state.embed_model = _create_embed_model(cfg)
    state.image_embed_model = _create_image_embed_model(cfg)
    state.multimodal_llm = _create_multimodal_llm(cfg)
    LlamaSettings.llm = state.llm
    LlamaSettings.embed_model = state.embed_model
    print(f"[INFO] LLM: {cfg.llm_model} @ {cfg.llm_api_base}")
    if cfg.image_embedding_mode != "disabled":
        print(f"[INFO] Image embedding mode: {cfg.image_embedding_mode}")

    # 2. RAG 向量索引
    print("\n[INFO] === 加载 RAG 向量索引 ===")
    state.vector_index = get_or_create_index(
        image_embed_model=state.image_embed_model,
    )

    # 3. GraphRAG 图谱索引（可通过 ENABLE_GRAPHRAG=false 关闭）
    if cfg.enable_graphrag:
        print("\n[INFO] === 加载 GraphRAG 图谱索引 ===")
        try:
            state.graph_index = get_or_create_graph_index()
        except Exception as e:
            print(f"[WARNING] GraphRAG 初始化失败: {e}")
            print("[WARNING] 将跳过 GraphRAG，其余功能正常使用")
    else:
        print("\n[INFO] === GraphRAG 已禁用 (enable_graphrag=False)，跳过图谱索引 ===")
        state.graph_index = None

    # 4. Database
    print("\n[INFO] === 初始化 Database ===")
    try:
        state.db_engine, state.db_table_names = init_database()
        state.sql_query_engine = create_sql_query_engine(state.db_engine, state.db_table_names)
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
        db_table_names=state.db_table_names,
        multimodal_llm=state.multimodal_llm,
        app_state=state,
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
        db_table_names=state.db_table_names,
        multimodal_llm=state.multimodal_llm,
        app_state=state,
    )
