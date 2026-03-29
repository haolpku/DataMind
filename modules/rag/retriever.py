"""
检索模块: 从索引中检索相关文档并生成回答
"""

from llama_index.core import VectorStoreIndex


def create_query_engine(index: VectorStoreIndex, similarity_top_k: int = 3):
    """创建检索问答引擎"""
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        response_mode="compact",
    )


def create_retriever(index: VectorStoreIndex, similarity_top_k: int = 5):
    """创建检索器 (仅返回相关片段，不生成回答)"""
    return index.as_retriever(similarity_top_k=similarity_top_k)
