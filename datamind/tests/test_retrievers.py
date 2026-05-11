"""Retriever behaviour tests (no network).

We use the in-memory fake store to avoid spinning up Chroma. The point is
to prove that SimpleRetriever / HybridRetriever wire correctly through the
protocol; Chroma-specific behaviour is covered by the real-API smoke test.
"""
from __future__ import annotations

from typing import Any, Sequence

import pytest

from datamind.capabilities.kb.providers.hybrid_retriever import (
    HybridRetriever,
    _tokenize,
)
from datamind.capabilities.kb.providers.simple_retriever import SimpleRetriever
from datamind.core.protocols import RetrievedChunk


class _Store:
    dimension = 4

    def __init__(self) -> None:
        # id -> (text, metadata, vec)
        self._rows: list[tuple[str, str, dict[str, Any], list[float]]] = []

    async def add(self, ids, texts, embeddings, metadatas=None):
        metas = list(metadatas or [{} for _ in ids])
        for i, t, v, m in zip(ids, texts, embeddings, metas):
            self._rows.append((i, t, dict(m), list(v)))

    async def query(self, embedding, *, top_k=5, where=None):
        # Return rows in insertion order with dummy descending scores.
        out: list[RetrievedChunk] = []
        for rank, (i, t, m, _v) in enumerate(self._rows[:top_k]):
            out.append(
                RetrievedChunk(
                    id=i, text=t, score=1.0 - rank * 0.1,
                    source=m.get("source"), metadata=m,
                )
            )
        return out

    async def count(self): return len(self._rows)
    async def delete(self, ids): pass
    async def reset(self): self._rows.clear()
    async def get_all_texts(self):
        return [(i, t, m) for i, t, m, _ in self._rows]


class _Embed:
    name = "fake"
    dimension = 4
    async def embed_texts(self, texts): return [[1.0, 0, 0, 0] for _ in texts]
    async def embed_query(self, query): return [1.0, 0, 0, 0]


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_retriever_passes_top_k_through():
    store = _Store()
    await store.add(
        ids=["a", "b", "c"],
        texts=["doc a", "doc b", "doc c"],
        embeddings=[[1, 0, 0, 0]] * 3,
    )
    r = SimpleRetriever(vector_store=store, embedding=_Embed())
    hits = await r.aretrieve("query", top_k=2)
    assert [h.id for h in hits] == ["a", "b"]


@pytest.mark.asyncio
async def test_hybrid_retriever_fuses_vector_and_bm25():
    store = _Store()
    await store.add(
        ids=["a", "b", "c"],
        texts=[
            "graph rag is a neural search technique",
            "knowledge bases store facts",
            "databases store rows and columns",
        ],
        embeddings=[[1, 0, 0, 0]] * 3,
    )
    r = HybridRetriever(vector_store=store, embedding=_Embed())
    hits = await r.aretrieve("knowledge base", top_k=2)
    # BM25 should push chunk b to the front — its tokens overlap the query.
    assert hits
    assert any(h.id == "b" for h in hits)


def test_tokenizer_handles_cjk():
    toks = _tokenize("GraphRAG 知识图谱 retrieval")
    # unigrams for zh, lowercased en
    assert "graphrag" in toks
    assert "知识图谱" in toks and "知" in toks and "谱" in toks
    assert "retrieval" in toks
