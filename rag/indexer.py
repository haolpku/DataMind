"""
文档索引模块: 加载文档 -> 分块 -> 构建 Chroma 向量索引
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


def load_documents(data_dir: str = config.DATA_DIR):
    """从 data/ 目录加载所有文档 (支持 PDF, TXT, MD, DOCX 等)"""
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"[WARNING] data 目录为空或不存在: {data_dir}")
        print("请将你的文档放入 data/ 目录后重新运行")
        return []
    reader = SimpleDirectoryReader(data_dir, recursive=True)
    documents = reader.load_data()
    print(f"[INFO] 加载了 {len(documents)} 个文档")
    return documents


def build_index(documents=None, persist_dir: str = config.STORAGE_DIR):
    """构建或加载 Chroma 向量索引"""
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection("rag_allinone")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if documents:
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[splitter],
            show_progress=True,
        )
        print(f"[INFO] 索引构建完成，已持久化到: {persist_dir}")
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)
        print(f"[INFO] 从已有索引加载: {persist_dir}")

    return index


def get_or_create_index():
    """如果索引已存在则加载，否则从文档构建"""
    chroma_client = chromadb.PersistentClient(path=config.STORAGE_DIR)
    collection = chroma_client.get_or_create_collection("rag_allinone")

    if collection.count() > 0:
        print(f"[INFO] 检测到已有索引 ({collection.count()} 条向量)，直接加载")
        return build_index(documents=None)
    else:
        print("[INFO] 未检测到已有索引，开始构建...")
        docs = load_documents()
        if not docs:
            return None
        return build_index(documents=docs)
