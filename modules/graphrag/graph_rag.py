"""
GraphRAG 模块: 基于知识图谱的检索

支持两种输入:
  方式 A: 原始文档 (data/) -> SimpleLLMPathExtractor 自动抽取实体/关系 -> 构建图索引
  方式 B: 预构建三元组 JSONL (data/triplets/*.jsonl) -> 直接导入图数据库，不涉及 LLM 抽取

使用 PropertyGraphIndex + SimplePropertyGraphStore (NetworkX 后端)
"""

import json
import os
import glob
from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.graph_stores.types import EntityNode, Relation

import config

GRAPH_STORAGE_DIR = os.path.join(config.STORAGE_DIR, "graph")
TRIPLETS_DIR = os.path.join(config.DATA_DIR, "triplets")


def load_triplets_from_file(triplets_dir: str = TRIPLETS_DIR):
    """从 data/triplets/ 目录加载预构建的三元组 JSONL 文件

    JSONL 格式 (每行一个 JSON):
        {"subject": "实体A", "relation": "关系", "object": "实体B"}

    可选字段:
        {"subject": "实体A", "relation": "关系", "object": "实体B",
         "subject_type": "Person", "object_type": "Organization"}

    Returns:
        (entities, relations): EntityNode 字典和 Relation 列表
    """
    if not os.path.exists(triplets_dir):
        return {}, []

    jsonl_files = glob.glob(os.path.join(triplets_dir, "*.jsonl"))
    if not jsonl_files:
        return {}, []

    entities = {}
    relations = []

    for filepath in jsonl_files:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARNING] {filepath} 第 {line_num} 行 JSON 解析失败，跳过")
                    continue

                subj = item.get("subject", "").strip()
                rel = item.get("relation", "").strip()
                obj = item.get("object", "").strip()
                if not subj or not rel or not obj:
                    continue

                subj_type = item.get("subject_type", "entity")
                obj_type = item.get("object_type", "entity")

                if subj not in entities:
                    entities[subj] = EntityNode(name=subj, label=subj_type)
                if obj not in entities:
                    entities[obj] = EntityNode(name=obj, label=obj_type)

                relations.append(Relation(
                    source_id=entities[subj].id,
                    target_id=entities[obj].id,
                    label=rel,
                ))

    if entities:
        print(f"[GraphRAG] 从三元组文件加载了 {len(entities)} 个实体, {len(relations)} 条关系 (来自 {len(jsonl_files)} 个文件)")
    return entities, relations


def build_graph_from_triplets(entities: dict, relations: list):
    """从预构建的三元组直接构建图索引（不涉及 LLM 抽取）"""
    graph_store = SimplePropertyGraphStore()
    graph_store.upsert_nodes(list(entities.values()))
    graph_store.upsert_relations(relations)

    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_kg_nodes=False,
    )

    os.makedirs(GRAPH_STORAGE_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=GRAPH_STORAGE_DIR)
    print(f"[GraphRAG] 从三元组构建图索引完成 ({len(entities)} 实体, {len(relations)} 关系)")

    try:
        graph_store.save_networkx_graph(
            name=os.path.join(GRAPH_STORAGE_DIR, "knowledge_graph.html")
        )
    except Exception:
        pass

    return index


def build_graph_index(documents=None, data_dir: str = config.DATA_DIR):
    """从文档构建知识图谱索引（方式 A: LLM 自动抽取实体/关系）"""
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
    """如果图索引已存在则加载，否则自动检测数据源并构建

    优先级:
      1. 已有索引 -> 直接加载
      2. data/triplets/*.jsonl 预构建三元组 -> 直接导入，不经过 LLM
      3. data/ 目录下的原始文档 -> LLM 自动抽取实体/关系
    """
    index = load_graph_index()
    if index is not None:
        return index

    entities, relations = load_triplets_from_file()
    if entities:
        print("[GraphRAG] 检测到预构建三元组，使用方式 B 构建图索引 (跳过 LLM 抽取)")
        return build_graph_from_triplets(entities, relations)

    print("[GraphRAG] 未检测到预构建三元组，使用方式 A 从文档构建图索引 (LLM 抽取)...")
    return build_graph_index()


def create_graph_query_engine(index: PropertyGraphIndex):
    """创建图谱检索问答引擎"""
    return index.as_query_engine(
        include_text=True,
    )
