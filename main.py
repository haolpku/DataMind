"""
DataMind 入口 (命令行模式)

功能:
  - RAG 向量检索 (Chroma)
  - GraphRAG 知识图谱检索 (NetworkX)
  - Database 自然语言查询 (SQLite NL2SQL)
  - Knowledge Skills 技能知识检索
  - 对话记忆 (短期+长期)

用法:
  1. 将文档放入 data/ 目录
  2. 在 config.py 中填入你的 API Key 和 API Base
  3. pip install -r requirements.txt
  4. python main.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from modules.rag.indexer import get_or_create_index
from modules.graphrag.graph_rag import get_or_create_graph_index
from modules.database.database import init_demo_database, create_sql_query_engine
from modules.skills.tools import get_all_skills
from modules.skills.knowledge import get_or_create_skill_index
from modules.agent.agent import create_agent
from modules.memory.memory import create_memory


def init_settings():
    """初始化全局 LLM 和 Embedding 配置"""
    llm = OpenAI(
        api_base=config.LLM_API_BASE,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL,
    )

    if config.USE_LOCAL_EMBEDDING:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name=config.LOCAL_EMBEDDING_MODEL)
        print(f"[INFO] 使用本地 Embedding: {config.LOCAL_EMBEDDING_MODEL}")
    else:
        embed_model = OpenAIEmbedding(
            api_base=config.EMBEDDING_API_BASE,
            api_key=config.EMBEDDING_API_KEY,
            model_name=config.EMBEDDING_MODEL,
        )
        print(f"[INFO] 使用远程 Embedding: {config.EMBEDDING_MODEL}")

    Settings.llm = llm
    Settings.embed_model = embed_model

    print(f"[INFO] LLM: {config.LLM_MODEL} @ {config.LLM_API_BASE}")
    return llm


async def chat_loop(agent, memory):
    """交互式对话循环"""
    print("\n" + "=" * 60)
    print("  DataMind 智能助手")
    print("  工具: RAG | GraphRAG | Database | Skills")
    print("-" * 60)
    print("  输入问题开始对话")
    print("  输入 'quit' 退出")
    print("  输入 'rebuild' 重建所有索引")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见!")
            break

        if user_input.lower() == "rebuild":
            print("[INFO] 请删除 storage/ 目录后重启程序以重建所有索引")
            continue

        try:
            response = await agent.run(user_input, memory=memory)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
            print("提示: 请检查 config.py 中的 API 配置是否正确\n")


def main():
    print("[INFO] 初始化配置...")
    llm = init_settings()

    # 1. 向量 RAG 索引
    print("\n[INFO] === 加载 RAG 向量索引 ===")
    vector_index = get_or_create_index()

    # 2. GraphRAG 图谱索引
    print("\n[INFO] === 加载 GraphRAG 图谱索引 ===")
    try:
        graph_index = get_or_create_graph_index()
    except Exception as e:
        print(f"[WARNING] GraphRAG 初始化失败: {e}")
        print("[WARNING] 将跳过 GraphRAG，其余功能正常使用")
        graph_index = None

    # 3. Database
    print("\n[INFO] === 初始化 Database ===")
    try:
        db_engine = init_demo_database()
        sql_query_engine = create_sql_query_engine(db_engine)
    except Exception as e:
        print(f"[WARNING] Database 初始化失败: {e}")
        sql_query_engine = None

    # 4. Skills (工具型)
    print("\n[INFO] === 加载工具型 Skills ===")
    skills = get_all_skills()
    print(f"[Skills] 已加载 {len(skills)} 个工具型技能")

    # 5. Skills (知识型)
    print("\n[INFO] === 加载知识型 Skills ===")
    try:
        skill_index = get_or_create_skill_index()
    except Exception as e:
        print(f"[WARNING] 知识型 Skills 初始化失败: {e}")
        skill_index = None

    # 6. 创建 Agent (整合所有工具)
    print("\n[INFO] === 创建 Agent ===")
    agent = create_agent(
        vector_index=vector_index,
        graph_index=graph_index,
        sql_query_engine=sql_query_engine,
        skill_index=skill_index,
        extra_tools=skills,
        llm=llm,
    )

    # 7. Memory
    memory = create_memory()

    asyncio.run(chat_loop(agent, memory))


if __name__ == "__main__":
    main()
