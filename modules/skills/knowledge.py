"""
知识型 Skills 模块: 从 Markdown 文件构建技能知识库

将 data/skills/ 目录下的 Markdown 文件索引到独立的 Chroma collection (skills_knowledge),
Agent 通过 skill_search 工具检索操作流程、SOP、领域专业知识。

与 RAG 知识库的区别:
  - RAG (rag_allinone collection): 通用文档知识库
  - Skills (skills_knowledge collection): 操作指南、最佳实践、SOP 等技能型知识
"""

import os
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

SKILLS_COLLECTION = "skills_knowledge"


def load_skill_documents(skills_dir: str = config.SKILLS_DIR):
    """从 data/skills/ 目录加载所有 Markdown 技能文件"""
    if not os.path.exists(skills_dir):
        return []

    md_files = [
        os.path.join(skills_dir, f)
        for f in os.listdir(skills_dir)
        if f.endswith(".md") and not f.startswith(".")
    ]

    if not md_files:
        return []

    reader = SimpleDirectoryReader(input_files=md_files)
    documents = reader.load_data()
    print(f"[Skills] 加载了 {len(documents)} 个技能文档 (来自 {len(md_files)} 个文件)")
    return documents


def build_skill_index(documents=None, persist_dir: str = config.STORAGE_DIR):
    """构建或加载技能知识的 Chroma 向量索引"""
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(SKILLS_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if documents:
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[splitter],
            show_progress=True,
        )
        print(f"[Skills] 技能知识索引构建完成，已持久化到: {persist_dir}")
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)
        print(f"[Skills] 从已有技能索引加载: {persist_dir}")

    return index


def get_or_create_skill_index():
    """如果技能索引已存在则加载，否则从 data/skills/ 构建"""
    chroma_client = chromadb.PersistentClient(path=config.STORAGE_DIR)
    collection = chroma_client.get_or_create_collection(SKILLS_COLLECTION)

    if collection.count() > 0:
        print(f"[Skills] 检测到已有技能索引 ({collection.count()} 条向量)，直接加载")
        return build_skill_index()

    docs = load_skill_documents()
    if not docs:
        print("[Skills] data/skills/ 目录为空，跳过技能知识索引")
        return None
    return build_skill_index(documents=docs)


def create_skill_query_engine(index: VectorStoreIndex, similarity_top_k: int = 3):
    """创建技能知识检索引擎"""
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        response_mode="compact",
    )
