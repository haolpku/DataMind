"""Capability protocols.

These are the contracts the agent and MCP servers talk to. Concrete provider
classes in `datamind.capabilities.<cap>.providers.*` must satisfy these,
and register themselves via `datamind.core.registry.<cap>_registry`.

Data-transfer objects (Pydantic models) are defined here too so providers
don't each invent their own shape. Keep them minimal — extensions belong
in provider-specific config, not in the protocol.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# DTOs — shared across capabilities
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    """A document fragment returned by a Retriever."""

    id: str
    text: str
    score: float = 0.0
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphTriple(BaseModel):
    subject: str
    relation: str
    object: str
    subject_type: str = "entity"
    object_type: str = "entity"
    confidence: float = 1.0
    source: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    id: str
    label: str
    type: str = "entity"
    score: float = 0.0
    properties: dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float = 1.0
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphPath(BaseModel):
    nodes: list[str]
    edges: list[Edge]
    score: float = 0.0


class ColumnSchema(BaseModel):
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False


class TableSchema(BaseModel):
    name: str
    columns: list[ColumnSchema]
    row_count_estimate: int | None = None


class QueryResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool = False
    elapsed_ms: float = 0.0


class MemoryItem(BaseModel):
    id: str
    namespace: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Capability protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Text (and optionally image) embedding backend.

    Providers are constructed via `embedding_registry.create(name, **cfg)` and
    should be cheap to instantiate but may lazily load heavy models.
    """

    name: str
    dimension: int

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, query: str) -> list[float]: ...

    # Optional hook — providers that support it can implement:
    # async def embed_images(self, paths: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persistent vector index.

    A VectorStore is created once per (profile, collection) pair. Indexer
    code writes chunks in, retrievers query. The contract is intentionally
    small — advanced features (metadata filters, hybrid search) live in
    retriever strategies, not here.
    """

    dimension: int

    async def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None: ...

    async def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...

    async def count(self) -> int: ...

    async def delete(self, ids: Sequence[str]) -> None: ...

    async def reset(self) -> None: ...

    async def get_all_texts(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Return (id, text, metadata) for every stored chunk.

        Used by lexical retrievers (BM25) that rebuild their index in memory
        on startup. Providers that can't enumerate may raise NotImplementedError.
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """KB retrieval strategy (simple / multi-query / hybrid / rerank)."""

    async def aretrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class GraphStore(Protocol):
    """Knowledge graph backend (NetworkX / Neo4j / ArangoDB / ...)."""

    async def upsert_triples(self, triples: Sequence[GraphTriple]) -> None: ...

    async def search_entities(self, query: str, *, top_k: int = 5) -> list[Entity]: ...

    async def traverse(
        self,
        start: str,
        *,
        max_hops: int = 2,
        relation_filter: list[str] | None = None,
    ) -> list[GraphPath]: ...

    async def neighbors(
        self,
        entity: str,
        *,
        direction: str = "both",
    ) -> list[Edge]: ...


@runtime_checkable
class DatabaseDialect(Protocol):
    """SQL backend (SQLite / MySQL / Postgres / ...).

    Built atop SQLAlchemy so dialects mostly differ in DSN parsing, quoting,
    and read-only enforcement. Each dialect must implement a safeguard
    (`is_destructive`) that runs BEFORE `execute_readonly`.
    """

    name: str

    def build_engine(self, dsn: str, **kwargs: Any) -> Any:
        """Return a SQLAlchemy Engine (or async equivalent)."""
        ...

    async def list_tables(self, engine: Any) -> list[str]: ...

    async def describe(self, engine: Any, table: str) -> TableSchema: ...

    async def execute_readonly(
        self,
        engine: Any,
        sql: str,
        *,
        row_limit: int = 1000,
        timeout_s: float = 10.0,
    ) -> QueryResult: ...

    def is_destructive(self, sql: str) -> bool:
        """True if `sql` would modify data or schema (INSERT/UPDATE/DELETE/DDL)."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Long-term memory storage (SQLite+embedding / Redis / Postgres / ...).

    `namespace` typically encodes session_id or user_id so the agent can
    scope recall. Providers are free to implement embedding-based search
    or lexical search — the contract only requires `recall(query) -> ranked
    list`.
    """

    async def save(
        self,
        namespace: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str: ...

    async def recall(
        self,
        namespace: str,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[MemoryItem]: ...

    async def forget(self, namespace: str, item_id: str) -> bool: ...

    async def list_namespaces(self) -> list[str]: ...
