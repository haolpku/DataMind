"""
Agent 模块: 统一整合 RAG + GraphRAG + Database + Skills 的智能助手

FunctionAgent 根据用户问题自动选择工具:
- 文档检索 -> knowledge_search (向量 RAG)
- 关系/多跳推理 -> graph_search (GraphRAG)
- 数据查询 -> database_query (NL2SQL)
- 技能知识检索 -> skill_search (知识型 Skill)
- 自定义技能 -> calculator, get_current_time, ... (工具型 Skills)
- 普通聊天 -> 直接用 LLM 回答
"""

from __future__ import annotations

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex

from config import settings
from modules.rag.retriever import create_retriever_by_config, _is_multimodal_index


SYSTEM_PROMPT = """\
你是一个功能强大的智能助手，拥有多种工具。请根据用户问题自动选择最合适的工具:

数据检索工具:
- knowledge_search: 语义检索知识库文档，查找文档中的具体内容（包括图片的文字描述）
- graph_search: 知识图谱检索，理解实体间关系，多跳推理
- database_query: 数据库查询，统计、排序、筛选结构化数据
- skill_search: 检索技能知识库，查找操作流程、SOP、领域专业指南

实用技能工具:
- get_current_time: 获取当前日期和时间
- calculator: 精确数学计算
- analyze_text: 文本统计分析
- unit_convert: 单位换算

核心规则:
- 【最重要】收到任何与知识、文档、数据、图片相关的问题时，必须先调用工具检索，禁止直接用自身知识回答
- 当用户提到图片、图表或文件名（如 xxx.png）时，必须用 knowledge_search 检索，知识库已包含图片的文字描述
- 绝对不要要求用户上传图片或提供链接，知识库中已有所有图片的详细描述
- 系统会自动在你的回答下方展示检索到的相关图片，你不需要说"无法展示图片"之类的话，直接根据图片描述回答内容即可
- 根据问题类型选择合适的工具，可以组合使用多个工具
- 涉及操作流程、最佳实践、SOP 时优先用 skill_search
- 需要精确计算时用 calculator (不要自己算)
- 涉及时间日期用 get_current_time
- 只有明确的闲聊才不需要使用工具
- 用中文回答
"""


def _extract_image_paths(source_nodes) -> list[str]:
    """从检索结果节点中提取 image_path（profile 相对路径）。"""
    paths = []
    seen = set()
    for node_with_score in source_nodes:
        node = node_with_score.node
        img_path = node.metadata.get("image_path", "")
        if img_path and img_path not in seen:
            seen.add(img_path)
            paths.append(img_path)
    return paths


def _build_rag_query_engine(vector_index, llm, multimodal_llm=None):
    """根据索引类型和多模态配置构建 RAG query engine。

    - 普通 VectorStoreIndex → RetrieverQueryEngine (现有行为)
    - MultiModalVectorStoreIndex + multimodal_llm → SimpleMultiModalQueryEngine
    - MultiModalVectorStoreIndex 无 multimodal_llm → RetrieverQueryEngine (仅文本)
    """
    is_mm = _is_multimodal_index(vector_index)

    if is_mm and multimodal_llm is not None:
        from llama_index.core.query_engine.multi_modal import SimpleMultiModalQueryEngine
        retriever = create_retriever_by_config(vector_index, settings, llm)
        return SimpleMultiModalQueryEngine(
            retriever=retriever,
            multi_modal_llm=multimodal_llm,
        )

    retriever = create_retriever_by_config(vector_index, settings, llm)
    return RetrieverQueryEngine.from_args(
        retriever=retriever, llm=llm, response_mode="compact",
    )


def create_agent(
    vector_index=None,
    graph_index=None,
    sql_query_engine=None,
    skill_index=None,
    extra_tools: list = None,
    llm=None,
    db_table_names: list[str] | None = None,
    multimodal_llm=None,
    app_state=None,
):
    """创建整合了多种检索工具和技能的 Agent

    Args:
        vector_index: 向量索引 (RAG / MultiModalVectorStoreIndex)
        graph_index: 图谱索引 (GraphRAG)
        sql_query_engine: SQL 查询引擎 (Database)
        skill_index: 技能知识索引 (Knowledge Skills)
        extra_tools: 额外的自定义工具列表 (Tool Skills)
        llm: LLM 实例
        db_table_names: 数据库表名列表 (用于生成工具描述)
        multimodal_llm: 多模态 LLM (GPT-4o 等)，用于图文联合回答
        app_state: AppState 实例，用于记录检索到的图片路径
    """
    tools = []

    if vector_index is not None:
        query_engine = _build_rag_query_engine(vector_index, llm, multimodal_llm)

        async def _knowledge_search(query: str) -> str:
            """语义检索知识库文档。当用户问的问题涉及已上传文档的具体内容、功能说明、技术细节时使用。"""
            response = await query_engine.aquery(query)
            if app_state is not None:
                img_paths = _extract_image_paths(
                    getattr(response, "source_nodes", [])
                )
                app_state.last_retrieved_images = img_paths
            return str(response)

        rag_tool = FunctionTool.from_defaults(
            async_fn=_knowledge_search,
            name="knowledge_search",
            description=(
                "语义检索知识库文档。当用户问的问题涉及已上传文档的具体内容、"
                "功能说明、技术细节时使用。输入为用户的问题。"
            ),
        )
        tools.append(rag_tool)

    if graph_index is not None:
        from modules.graphrag.graph_rag import create_graph_query_engine
        graph_engine = create_graph_query_engine(graph_index)
        graph_tool = QueryEngineTool.from_defaults(
            query_engine=graph_engine,
            name="graph_search",
            description=(
                "知识图谱检索。通过实体和关系进行多跳推理查询。"
                "适用于理解实体之间的关系、查找关联实体等问题。输入为用户的问题。"
            ),
        )
        tools.append(graph_tool)

    if sql_query_engine is not None:
        tables_desc = ", ".join(db_table_names) if db_table_names else "unknown"
        db_tool = QueryEngineTool.from_defaults(
            query_engine=sql_query_engine,
            name="database_query",
            description=(
                "数据库查询工具。将自然语言转换为 SQL 查询，"
                "适用于统计、排序、筛选、聚合等结构化数据问题。"
                f"当前数据库包含以下表: {tables_desc}。"
            ),
        )
        tools.append(db_tool)

    if skill_index is not None:
        from modules.skills.knowledge import create_skill_query_engine
        skill_engine = create_skill_query_engine(skill_index)
        skill_tool = QueryEngineTool.from_defaults(
            query_engine=skill_engine,
            name="skill_search",
            description=(
                "检索技能知识库。当用户问题涉及操作流程、SOP、最佳实践、"
                "领域专业知识、使用指南时使用。输入为用户的问题。"
            ),
        )
        tools.append(skill_tool)

    if extra_tools:
        tools.extend(extra_tools)

    prompt = SYSTEM_PROMPT
    if not tools:
        prompt = "你是一个智能助手，用中文回答用户问题。"

    agent = FunctionAgent(
        tools=tools,
        llm=llm,
        system_prompt=prompt,
    )

    tool_names = [t.metadata.name for t in tools]
    print(f"[Agent] 已加载工具: {tool_names or '(无)'}")
    return agent
