"""
检索模块: 支持多种检索策略

- SimpleRetriever: 单 query 直接检索（默认行为）
- MultiQueryRetriever: LLM 拆解子查询 -> 并行检索 -> 去重合并

通过 create_retriever_by_config() 工厂函数根据配置选择策略。
保留 create_query_engine() 用于向后兼容。

多模态索引 (MultiModalVectorStoreIndex) 的 as_retriever() 额外接受
image_similarity_top_k 参数，由工厂函数自动处理。
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Union

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from config import settings

AnyIndex = Union[VectorStoreIndex, "MultiModalVectorStoreIndex"]


class BaseRetriever(ABC):
    @abstractmethod
    async def aretrieve(self, query: str) -> list[NodeWithScore]:
        """异步检索，返回节点列表"""


class SimpleRetriever(BaseRetriever):
    """单 query 直接向量检索（等同于原有行为）"""

    def __init__(self, index: AnyIndex, similarity_top_k: int = 3, image_similarity_top_k: int = 0):
        kwargs: dict = {"similarity_top_k": similarity_top_k}
        if image_similarity_top_k > 0 and _is_multimodal_index(index):
            kwargs["image_similarity_top_k"] = image_similarity_top_k
        self._retriever = index.as_retriever(**kwargs)

    async def aretrieve(self, query: str) -> list[NodeWithScore]:
        return await self._retriever.aretrieve(query)


MULTI_QUERY_PROMPT = """\
你是一个搜索助手。请将以下用户问题拆解为 {num} 个不同角度的子查询，用于从向量知识库中检索更全面的信息。
每个子查询独占一行，不要带编号或前缀，只输出子查询本身。

用户问题: {query}
"""


class MultiQueryRetriever(BaseRetriever):
    """LLM 拆解子查询 -> 并行检索 -> 去重合并（保留最高分）"""

    def __init__(
        self,
        index: AnyIndex,
        llm,
        num_queries: int = 3,
        similarity_top_k: int = 3,
        image_similarity_top_k: int = 0,
    ):
        self._index = index
        self._llm = llm
        self._num_queries = num_queries
        self._top_k = similarity_top_k
        self._img_top_k = image_similarity_top_k

    async def _generate_sub_queries(self, query: str) -> list[str]:
        prompt = MULTI_QUERY_PROMPT.format(num=self._num_queries, query=query)
        response = await self._llm.acomplete(prompt)
        lines = [
            line.strip()
            for line in str(response).strip().splitlines()
            if line.strip()
        ]
        return lines[: self._num_queries]

    @staticmethod
    def _merge_results(all_results: list[list[NodeWithScore]]) -> list[NodeWithScore]:
        best: dict[str, NodeWithScore] = {}
        for results in all_results:
            for node in results:
                nid = node.node.node_id
                if nid not in best or node.score > best[nid].score:
                    best[nid] = node
        return sorted(best.values(), key=lambda n: n.score, reverse=True)

    async def aretrieve(self, query: str) -> list[NodeWithScore]:
        sub_queries = await self._generate_sub_queries(query)
        if not sub_queries:
            sub_queries = [query]

        kwargs: dict = {"similarity_top_k": self._top_k}
        if self._img_top_k > 0 and _is_multimodal_index(self._index):
            kwargs["image_similarity_top_k"] = self._img_top_k
        retriever = self._index.as_retriever(**kwargs)
        tasks = [retriever.aretrieve(q) for q in sub_queries]
        all_results = await asyncio.gather(*tasks)

        return self._merge_results(all_results)


def _is_multimodal_index(index) -> bool:
    """检查索引是否为 MultiModalVectorStoreIndex。"""
    cls_name = type(index).__name__
    return cls_name == "MultiModalVectorStoreIndex"


def create_retriever_by_config(
    index: AnyIndex,
    cfg=None,
    llm=None,
) -> BaseRetriever:
    """工厂函数: 根据配置选择检索策略"""
    if cfg is None:
        cfg = settings

    img_top_k = getattr(cfg, "image_similarity_top_k", 0) if _is_multimodal_index(index) else 0

    if cfg.retriever_mode == "multi_query" and llm is not None:
        return MultiQueryRetriever(
            index, llm, cfg.multi_query_count, cfg.similarity_top_k,
            image_similarity_top_k=img_top_k,
        )
    return SimpleRetriever(index, cfg.similarity_top_k, image_similarity_top_k=img_top_k)


# ---- 向后兼容 ----

def create_query_engine(index: AnyIndex, similarity_top_k: int = 3):
    """创建检索问答引擎（保留用于向后兼容）"""
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        response_mode="compact",
    )


def create_retriever(index: AnyIndex, similarity_top_k: int = 5):
    """创建检索器 (仅返回相关片段，不生成回答)"""
    return index.as_retriever(similarity_top_k=similarity_top_k)
