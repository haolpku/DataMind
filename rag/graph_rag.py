"""
GraphRAG 模块: 基于知识图谱的检索

使用 PropertyGraphIndex + SimplePropertyGraphStore (NetworkX 后端)
从文档中抽取实体和关系，构建知识图谱，支持多跳推理查询。
"""

import os
from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

import config

GRAPH_STORAGE_DIR = os.path.join(config.STORAGE_DIR, "graph")


def build_graph_index(documents=None, data_dir: str = config.DATA_DIR):
    """从文档构建知识图谱索引"""
    if documents is None:
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            print("[WARNING] data 目录为空，无法构建图索引")
            return None
        reader = SimpleDirectoryReader(data_dir, recursive=True)
        documents = reader.load_data()
        print(f"[GraphRAG] 加载了 {len(documents)} 个文档")

    kg_extractor = SimpleLLMPathExtractor(max_paths_per_chunk=10)

    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=[kg_extractor],
        show_progress=True,
    )

    os.makedirs(GRAPH_STORAGE_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=GRAPH_STORAGE_DIR)
    print(f"[GraphRAG] 图索引构建完成，已持久化到: {GRAPH_STORAGE_DIR}")

    try:
        index.property_graph_store.save_networkx_graph(
            name=os.path.join(GRAPH_STORAGE_DIR, "knowledge_graph.html")
        )
        print(f"[GraphRAG] 知识图谱可视化已保存: {GRAPH_STORAGE_DIR}/knowledge_graph.html")
    except Exception:
        pass

    return index


def load_graph_index():
    """从持久化目录加载已有的图索引"""
    if not os.path.exists(GRAPH_STORAGE_DIR):
        return None
    try:
        storage_context = StorageContext.from_defaults(persist_dir=GRAPH_STORAGE_DIR)
        index = PropertyGraphIndex.from_existing(
            property_graph_store=storage_context.property_graph_store,
        )
        print(f"[GraphRAG] 从已有图索引加载: {GRAPH_STORAGE_DIR}")
        return index
    except Exception:
        return None


def get_or_create_graph_index():
    """如果图索引已存在则加载，否则构建"""
    index = load_graph_index()
    if index is not None:
        return index
    print("[GraphRAG] 未检测到已有图索引，开始构建...")
    return build_graph_index()


def create_graph_query_engine(index: PropertyGraphIndex):
    """创建图谱检索问答引擎"""
    return index.as_query_engine(
        include_text=True,
    )
