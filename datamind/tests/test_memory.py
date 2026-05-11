"""Memory capability tests (no network)."""
from __future__ import annotations

from typing import Sequence

import pytest

from datamind.capabilities.memory import (
    MemoryService,
    ShortTermMemory,
)
from datamind.capabilities.memory.providers.sqlite_store import SQLiteMemoryStore
from datamind.core.protocols import MemoryStore


class _FakeEmbed:
    """Deterministic embedding: char-bag based, 8 dims.

    Similar strings get similar vectors without needing a real model.
    """

    name = "fake"
    dimension = 8

    @staticmethod
    def _vec(text: str) -> list[float]:
        buckets = [0.0] * 8
        for ch in text.lower():
            buckets[ord(ch) % 8] += 1.0
        total = sum(buckets) or 1.0
        return [b / total for b in buckets]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._vec(query)


# ---------------------------------------------------------------- short-term


@pytest.mark.asyncio
async def test_short_term_fifo_respects_max():
    st = ShortTermMemory(max_turns=3)
    await st.append("s", "user", "a")
    await st.append("s", "user", "b")
    await st.append("s", "user", "c")
    await st.append("s", "user", "d")
    recent = await st.recent("s")
    assert [t.content for t in recent] == ["b", "c", "d"]


@pytest.mark.asyncio
async def test_short_term_clear():
    st = ShortTermMemory(max_turns=3)
    await st.append("s", "user", "a")
    n = await st.clear("s")
    assert n == 1
    assert await st.recent("s") == []


# ---------------------------------------------------------------- long-term


@pytest.mark.asyncio
async def test_sqlite_store_satisfies_protocol(tmp_path):
    s = SQLiteMemoryStore(db_path=str(tmp_path / "m.db"), embedding=_FakeEmbed())
    assert isinstance(s, MemoryStore)


@pytest.mark.asyncio
async def test_save_and_recall_with_embedding(tmp_path):
    s = SQLiteMemoryStore(db_path=str(tmp_path / "m.db"), embedding=_FakeEmbed())
    await s.save("session:a", "user likes coffee")
    await s.save("session:a", "user dislikes onions")
    await s.save("session:a", "meeting at 3pm on Friday")
    hits = await s.recall("session:a", "coffee", top_k=2)
    assert hits and hits[0].content.startswith("user likes coffee")


@pytest.mark.asyncio
async def test_recall_namespace_isolation(tmp_path):
    s = SQLiteMemoryStore(db_path=str(tmp_path / "m.db"), embedding=_FakeEmbed())
    await s.save("session:a", "hello from a")
    await s.save("session:b", "hello from b")
    a = await s.recall("session:a", "hello", top_k=5)
    b = await s.recall("session:b", "hello", top_k=5)
    assert len(a) == 1 and a[0].content.endswith("from a")
    assert len(b) == 1 and b[0].content.endswith("from b")


@pytest.mark.asyncio
async def test_lexical_recall_when_no_embedding(tmp_path):
    s = SQLiteMemoryStore(db_path=str(tmp_path / "m.db"), embedding=None)
    await s.save("g", "the capital of France is Paris")
    await s.save("g", "pasta recipe requires tomatoes and basil")
    hits = await s.recall("g", "paris france", top_k=2)
    assert hits
    assert "Paris" in hits[0].content


@pytest.mark.asyncio
async def test_forget_and_list_namespaces(tmp_path):
    s = SQLiteMemoryStore(db_path=str(tmp_path / "m.db"), embedding=_FakeEmbed())
    rid = await s.save("n1", "foo")
    await s.save("n2", "bar")
    namespaces = await s.list_namespaces()
    assert set(namespaces) == {"n1", "n2"}
    ok = await s.forget("n1", rid)
    assert ok
    assert await s.count("n1") == 0


# ---------------------------------------------------------------- service


@pytest.mark.asyncio
async def test_service_combines_short_and_long(tmp_path):
    st = ShortTermMemory(max_turns=5)
    lt = SQLiteMemoryStore(db_path=str(tmp_path / "m.db"), embedding=_FakeEmbed())
    svc = MemoryService(short_term=st, long_term=lt)

    await svc.append_turn("s1", "user", "Hi there")
    await svc.append_turn("s1", "assistant", "Hello")
    turns = await svc.recent_turns("s1")
    assert len(turns) == 2

    rid = await svc.save("session:s1", "the user prefers Monday meetings")
    hits = await svc.recall("session:s1", "when does user like to meet")
    assert hits and "Monday" in hits[0]["content"]
    assert await svc.forget("session:s1", rid)
