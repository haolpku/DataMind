"""
文档索引模块: 加载文档 -> 分块 -> 构建 Chroma 向量索引

支持两种输入:
  方式 A: 原始文档 (data/ 目录) -> SentenceSplitter 自动分块 -> Embedding -> 存 Chroma
  方式 B: 预分块 JSONL (data/chunks/*.jsonl) -> 跳过分块 -> Embedding -> 存 Chroma
"""

import json
import os
import glob
import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

PRE_CHUNKED_DIR = os.path.join(config.DATA_DIR, "chunks")


def load_documents(data_dir: str = config.DATA_DIR):
    """从 data/ 目录加载所有文档 (支持 PDF, TXT, MD, DOCX 等)
    自动排除 chunks/ 和 triplets/ 子目录"""
    if not os.path.exists(data_dir):
        print(f"[WARNING] data 目录不存在: {data_dir}")
        return []

    exclude_dirs = {"chunks", "triplets"}
    files = []
    for item in os.listdir(data_dir):
        if item in exclude_dirs or item.startswith("."):
            continue
        full_path = os.path.join(data_dir, item)
        if os.path.isfile(full_path):
            files.append(full_path)
        elif os.path.isdir(full_path):
            for root, dirs, filenames in os.walk(full_path):
                for f in filenames:
                    if not f.startswith("."):
                        files.append(os.path.join(root, f))

    if not files:
        print(f"[WARNING] data 目录为空: {data_dir}")
        return []

    reader = SimpleDirectoryReader(input_files=files)
    documents = reader.load_data()
    print(f"[INFO] 加载了 {len(documents)} 个文档")
    return documents


def load_pre_chunked(chunks_dir: str = PRE_CHUNKED_DIR):
    """从 data/chunks/ 目录加载预分块的 JSONL 文件，直接构建 TextNode

    JSONL 格式 (每行一个 JSON):
        {"text": "chunk 内容", "metadata": {"source": "来源文件", ...}}

    metadata 字段可选，可以包含任意键值对作为检索时的过滤条件。
    """
    if not os.path.exists(chunks_dir):
        return []

    jsonl_files = glob.glob(os.path.join(chunks_dir, "*.jsonl"))
    if not jsonl_files:
        return []

    nodes = []
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

                text = item.get("text", "")
                if not text:
                    continue

                metadata = item.get("metadata", {})
                metadata["source_file"] = os.path.basename(filepath)
                node = TextNode(text=text, metadata=metadata)
                nodes.append(node)

    if nodes:
        print(f"[INFO] 从预分块文件加载了 {len(nodes)} 个 chunks (来自 {len(jsonl_files)} 个文件)")
    return nodes


def build_index(documents=None, nodes=None, persist_dir: str = config.STORAGE_DIR):
    """构建或加载 Chroma 向量索引

    Args:
        documents: 原始文档列表 (方式 A, 会经过 SentenceSplitter 分块)
        nodes: 预分块的 TextNode 列表 (方式 B, 跳过分块直接 Embedding)
        persist_dir: 持久化目录
    """
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("rag_allinone")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if nodes:
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        print(f"[INFO] 从预分块数据构建索引完成 ({len(nodes)} chunks)，已持久化到: {persist_dir}")
    elif documents:
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[splitter],
            show_progress=True,
        )
        print(f"[INFO] 从文档构建索引完成，已持久化到: {persist_dir}")
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)
        print(f"[INFO] 从已有索引加载: {persist_dir}")

    return index


def get_or_create_index():
    """如果索引已存在则加载，否则自动检测数据源并构建

    优先级:
      1. 已有索引 -> 直接加载
      2. data/chunks/*.jsonl 预分块数据 -> 跳过分块，直接 Embedding
      3. data/ 目录下的原始文档 -> SentenceSplitter 分块 + Embedding
    """
    chroma_client = chromadb.PersistentClient(path=config.STORAGE_DIR)
    collection = chroma_client.get_or_create_collection("rag_allinone")

    if collection.count() > 0:
        print(f"[INFO] 检测到已有索引 ({collection.count()} 条向量)，直接加载")
        return build_index()

    pre_chunked_nodes = load_pre_chunked()
    if pre_chunked_nodes:
        print("[INFO] 检测到预分块数据，使用方式 B 构建索引 (跳过分块)")
        return build_index(nodes=pre_chunked_nodes)

    print("[INFO] 未检测到预分块数据，使用方式 A 从文档构建索引...")
    docs = load_documents()
    if not docs:
        return None
    return build_index(documents=docs)
