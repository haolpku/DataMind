"""
Agent 模块: 统一整合 RAG + GraphRAG + Database + Skills 的智能助手

FunctionAgent 根据用户问题自动选择工具:
- 文档检索 -> knowledge_search (向量 RAG)
- 关系/多跳推理 -> graph_search (GraphRAG)
- 数据查询 -> database_query (NL2SQL)
- 自定义技能 -> calculator, get_current_time, ... (Skills)
- 普通聊天 -> 直接用 LLM 回答
"""

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core import VectorStoreIndex

from rag.retriever import create_query_engine


SYSTEM_PROMPT = """\
你是一个功能强大的智能助手，拥有多种工具。请根据用户问题自动选择最合适的工具:

数据检索工具:
- knowledge_search: 语义检索知识库文档，查找文档中的具体内容
- graph_search: 知识图谱检索，理解实体间关系，多跳推理
- database_query: 数据库查询，统计、排序、筛选结构化数据

实用技能工具:
- get_current_time: 获取当前日期和时间
- calculator: 精确数学计算
- analyze_text: 文本统计分析
- unit_convert: 单位换算

核心规则:
- 根据问题类型选择合适的工具，可以组合使用多个工具
- 需要精确计算时用 calculator (不要自己算)
- 涉及时间日期用 get_current_time
- 只有明确的闲聊才不需要使用工具
- 用中文回答
"""


def create_agent(
    vector_index: VectorStoreIndex = None,
    graph_index=None,
    sql_query_engine=None,
    extra_tools: list = None,
    llm=None,
):
    """创建整合了多种检索工具和技能的 Agent

    Args:
        vector_index: 向量索引 (RAG)
        graph_index: 图谱索引 (GraphRAG)
        sql_query_engine: SQL 查询引擎 (Database)
        extra_tools: 额外的自定义工具列表 (Skills)
        llm: LLM 实例
    """
    tools = []

    if vector_index is not None:
        query_engine = create_query_engine(vector_index)
        rag_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="knowledge_search",
            description=(
                "语义检索知识库文档。当用户问的问题涉及已上传文档的具体内容、"
                "功能说明、技术细节时使用。输入为用户的问题。"
            ),
        )
        tools.append(rag_tool)

    if graph_index is not None:
        from rag.graph_rag import create_graph_query_engine
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
        db_tool = QueryEngineTool.from_defaults(
            query_engine=sql_query_engine,
            name="database_query",
            description=(
                "数据库查询工具。将自然语言转换为 SQL 查询，"
                "适用于统计、排序、筛选、聚合等结构化数据问题。"
                "当前数据库包含 employees (员工表: id, name, department, position, salary, city) "
                "和 projects (项目表: id, project_name, lead_employee_id, budget, status) 两张表。"
            ),
        )
        tools.append(db_tool)

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
