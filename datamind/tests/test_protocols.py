"""Protocol conformance tests.

`runtime_checkable` Protocols let us assert via isinstance that a mock
implementation satisfies the interface. These tests double as executable
docs — if you break a protocol, these fail first.
"""
from __future__ import annotations

from typing import Any, Sequence

import pytest

from datamind.core.protocols import (
    DatabaseDialect,
    EmbeddingProvider,
    Entity,
    GraphPath,
    GraphStore,
    GraphTriple,
    MemoryItem,
    MemoryStore,
    QueryResult,
    Retriever,
    RetrievedChunk,
    TableSchema,
    Edge,
)


# ---------------------------------------------------------------------------
# Mock implementations — minimal, just enough to satisfy each protocol.
# ---------------------------------------------------------------------------


class _MockEmbed:
    name = "mock"
    dimension = 4

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class _MockRetriever:
    async def aretrieve(self, query, *, top_k=5, filters=None):
        return [RetrievedChunk(id="1", text="hi", score=0.9)]


class _MockGraph:
    async def upsert_triples(self, triples): return None
    async def search_entities(self, query, *, top_k=5):
        return [Entity(id="e1", label=query, score=1.0)]
    async def traverse(self, start, *, max_hops=2, relation_filter=None):
        return [GraphPath(nodes=[start, "e2"], edges=[], score=1.0)]
    async def neighbors(self, entity, *, direction="both"):
        return []


class _MockDB:
    name = "mock"
    def build_engine(self, dsn, **kw): return object()
    async def list_tables(self, engine): return ["t"]
    async def describe(self, engine, table):
        return TableSchema(name=table, columns=[])
    async def execute_readonly(self, engine, sql, *, row_limit=1000, timeout_s=10.0):
        return QueryResult(columns=["a"], rows=[[1]], elapsed_ms=0.5)
    def is_destructive(self, sql): return sql.strip().upper().startswith(
        ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE")
    )


class _MockMemory:
    async def save(self, content, *, scope="profile", profile=None, session_id=None,
                   kind="fact", metadata=None):
        return "id1"
    async def recall(self, query, *, profile=None, session_id=None, top_k=8,
                     kinds=None, include_archived=False):
        return [MemoryItem(id="id1", scope="profile", profile=profile, content="x")]
    async def forget(self, item_id, *, hard=False): return True
    async def list_profiles(self): return ["default"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mock_embedding_satisfies_protocol():
    assert isinstance(_MockEmbed(), EmbeddingProvider)


def test_mock_retriever_satisfies_protocol():
    assert isinstance(_MockRetriever(), Retriever)


def test_mock_graph_satisfies_protocol():
    assert isinstance(_MockGraph(), GraphStore)


def test_mock_db_dialect_satisfies_protocol():
    assert isinstance(_MockDB(), DatabaseDialect)


def test_mock_memory_satisfies_protocol():
    assert isinstance(_MockMemory(), MemoryStore)


def test_destructive_sql_detection():
    db = _MockDB()
    assert db.is_destructive("DELETE FROM t")
    assert db.is_destructive("drop table t")
    assert db.is_destructive("INSERT INTO t VALUES (1)")
    assert not db.is_destructive("SELECT * FROM t")


@pytest.mark.asyncio
async def test_mock_retriever_returns_chunks():
    chunks = await _MockRetriever().aretrieve("hello", top_k=3)
    assert chunks and isinstance(chunks[0], RetrievedChunk)
    assert chunks[0].text == "hi"


@pytest.mark.asyncio
async def test_mock_embedding_dimensions_match():
    e = _MockEmbed()
    vecs = await e.embed_texts(["a", "b"])
    assert len(vecs) == 2
    assert all(len(v) == e.dimension for v in vecs)


@pytest.mark.asyncio
async def test_triple_upsert_and_traverse_roundtrip():
    g = _MockGraph()
    await g.upsert_triples([GraphTriple(subject="A", relation="rel", object="B")])
    paths = await g.traverse("A", max_hops=1)
    assert paths and paths[0].nodes[0] == "A"


def test_registries_are_compatible_with_protocols():
    """Registering a mock under an embedding_registry and retrieving it yields
    an EmbeddingProvider-compatible object. This is the integration point that
    future providers rely on.
    """
    from datamind.core.registry import Registry

    fake_reg: Registry = Registry("embedding_test")

    @fake_reg.register("mock")
    class M(_MockEmbed):
        pass

    inst = fake_reg.create("mock")
    assert isinstance(inst, EmbeddingProvider)
