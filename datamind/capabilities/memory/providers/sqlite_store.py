"""SQLite-backed long-term memory with embedding recall.

Schema:
    CREATE TABLE memory (
        id         TEXT PRIMARY KEY,
        namespace  TEXT NOT NULL,
        content    TEXT NOT NULL,
        metadata   TEXT,  -- JSON
        embedding  BLOB,  -- float32 packed
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_memory_ns ON memory(namespace);

Recall uses cosine similarity computed in Python. For corpora up to tens
of thousands of items this is fine; once you outgrow it, drop in a Redis
/ Postgres+pgvector / Qdrant provider under memory_registry.
"""
from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import struct
import time
import uuid
from pathlib import Path
from typing import Any

from datamind.core.errors import CapabilityError
from datamind.core.logging import get_logger
from datamind.core.protocols import EmbeddingProvider, MemoryItem
from datamind.core.registry import memory_registry

_log = get_logger("memory.sqlite")


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(data: bytes) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@memory_registry.register("sqlite")
class SQLiteMemoryStore:
    """Embedding-aware memory store in a single local SQLite file."""

    def __init__(
        self,
        *,
        dsn: str | None = None,
        db_path: str | Path | None = None,
        embedding: EmbeddingProvider | None = None,
    ) -> None:
        if dsn and not db_path:
            # Accept an SQLAlchemy-style URL for symmetry with the DB capability.
            if dsn.startswith("sqlite:///"):
                db_path = dsn[len("sqlite:///"):]
            else:
                db_path = dsn
        if not db_path:
            raise CapabilityError("memory", "sqlite memory store needs db_path or dsn")
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._embedding = embedding
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id         TEXT PRIMARY KEY,
                    namespace  TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    metadata   TEXT,
                    embedding  BLOB,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_ns ON memory(namespace);
                """
            )

    # ----------------------------------------------------------- async ops

    async def save(
        self,
        namespace: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        item_id = uuid.uuid4().hex
        ts = time.time()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        emb: bytes | None = None
        if self._embedding is not None and content.strip():
            vec = await self._embedding.embed_query(content)
            emb = _pack(vec)

        def _run() -> None:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO memory (id, namespace, content, metadata, embedding, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (item_id, namespace, content, meta_json, emb, ts),
                )

        await asyncio.to_thread(_run)
        return item_id

    async def recall(
        self,
        namespace: str,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        # If no embedding is available, fall back to substring ranking.
        if self._embedding is None:
            return await self._lexical_recall(namespace, query, top_k=top_k)
        qv = await self._embedding.embed_query(query)

        def _run() -> list[MemoryItem]:
            with self._conn() as conn:
                cur = conn.execute(
                    "SELECT id, content, metadata, embedding, created_at "
                    "FROM memory WHERE namespace = ?",
                    (namespace,),
                )
                rows = cur.fetchall()
            scored: list[tuple[float, MemoryItem]] = []
            for rid, content, meta_json, emb_blob, created_at in rows:
                score = _cosine(qv, _unpack(emb_blob)) if emb_blob else 0.0
                try:
                    meta = json.loads(meta_json) if meta_json else {}
                except json.JSONDecodeError:
                    meta = {}
                scored.append((
                    score,
                    MemoryItem(
                        id=rid,
                        namespace=namespace,
                        content=content,
                        score=score,
                        metadata=meta,
                        created_at=str(created_at),
                    ),
                ))
            scored.sort(key=lambda t: -t[0])
            return [item for _, item in scored[:top_k]]

        return await asyncio.to_thread(_run)

    async def _lexical_recall(self, namespace: str, query: str, *, top_k: int) -> list[MemoryItem]:
        q_terms = [t for t in query.lower().split() if t]

        def _run() -> list[MemoryItem]:
            with self._conn() as conn:
                cur = conn.execute(
                    "SELECT id, content, metadata, created_at FROM memory WHERE namespace = ?",
                    (namespace,),
                )
                rows = cur.fetchall()
            scored: list[tuple[float, MemoryItem]] = []
            for rid, content, meta_json, created_at in rows:
                lc = content.lower()
                hits = sum(1 for t in q_terms if t in lc)
                score = hits / max(len(q_terms), 1)
                try:
                    meta = json.loads(meta_json) if meta_json else {}
                except json.JSONDecodeError:
                    meta = {}
                scored.append((
                    score,
                    MemoryItem(
                        id=rid,
                        namespace=namespace,
                        content=content,
                        score=score,
                        metadata=meta,
                        created_at=str(created_at),
                    ),
                ))
            scored.sort(key=lambda t: -t[0])
            return [item for _, item in scored[:top_k]]

        return await asyncio.to_thread(_run)

    async def forget(self, namespace: str, item_id: str) -> bool:
        def _run() -> bool:
            with self._conn() as conn:
                cur = conn.execute(
                    "DELETE FROM memory WHERE namespace = ? AND id = ?",
                    (namespace, item_id),
                )
                return cur.rowcount > 0

        return await asyncio.to_thread(_run)

    async def list_namespaces(self) -> list[str]:
        def _run() -> list[str]:
            with self._conn() as conn:
                cur = conn.execute("SELECT DISTINCT namespace FROM memory ORDER BY namespace")
                return [r[0] for r in cur.fetchall()]

        return await asyncio.to_thread(_run)

    async def count(self, namespace: str | None = None) -> int:
        def _run() -> int:
            with self._conn() as conn:
                if namespace is None:
                    return int(conn.execute("SELECT COUNT(*) FROM memory").fetchone()[0])
                return int(
                    conn.execute(
                        "SELECT COUNT(*) FROM memory WHERE namespace = ?",
                        (namespace,),
                    ).fetchone()[0]
                )

        return await asyncio.to_thread(_run)
