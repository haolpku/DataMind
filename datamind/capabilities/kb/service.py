"""KB service: wires EmbeddingProvider + VectorStore + Retriever from Settings.

This is the thing agent code / MCP servers / tool handlers grab. It's
intentionally stateful but scoped to a single profile + collection, so
swapping profiles means building a new service (cheap — chroma is
persistent).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from datamind.capabilities.embedding import build_embedding
from datamind.config import LLMConfig, RetrievalConfig, Settings
from datamind.core.errors import ConfigError
from datamind.core.logging import get_logger
from datamind.core.protocols import EmbeddingProvider, Retriever, VectorStore
from datamind.core.registry import retriever_registry, vector_store_registry

# Importing providers populates the registries.
from . import providers  # noqa: F401
from .indexer import build_index, list_documents

_log = get_logger("kb.service")


class KBService:
    """Bundle of (embedding, vector store, retriever) scoped to one profile."""

    def __init__(
        self,
        *,
        embedding: EmbeddingProvider,
        vector_store: VectorStore,
        retriever: Retriever,
        data_dir: Path,
        retrieval_cfg: RetrievalConfig,
    ) -> None:
        self.embedding = embedding
        self.vector_store = vector_store
        self.retriever = retriever
        self.data_dir = data_dir
        self.retrieval_cfg = retrieval_cfg

    # ------------------------------------------------------------ behaviour

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        k = top_k or self.retrieval_cfg.top_k
        chunks = await self.retriever.aretrieve(query, top_k=k, filters=filters)
        return [c.model_dump() for c in chunks]

    async def count(self) -> int:
        return await self.vector_store.count()

    async def reindex(self) -> dict[str, Any]:
        await self.vector_store.reset()
        stats = await build_index(
            data_dir=self.data_dir,
            vector_store=self.vector_store,
            embedding=self.embedding,
            chunk_size=self.retrieval_cfg.chunk_size,
            chunk_overlap=self.retrieval_cfg.chunk_overlap,
        )
        # If the retriever keeps a lexical cache (hybrid), nudge it.
        rebuild = getattr(self.retriever, "rebuild_lexical", None)
        if callable(rebuild):
            await rebuild()
        return stats

    async def list_documents(self) -> list[dict[str, Any]]:
        return await list_documents(self.data_dir)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_kb_service(
    settings: Settings,
    *,
    llm_client: AsyncAnthropic | None = None,
    collection_name: str = "kb_default",
) -> KBService:
    """Build a KBService from the top-level Settings.

    `llm_client` is only consulted by the multi_query strategy; callers
    that don't use it can pass None.
    """
    embedding = build_embedding(settings.embedding, fallback_llm=settings.llm)
    storage_dir = settings.data.storage_dir
    storage_dir.mkdir(parents=True, exist_ok=True)
    vector_store = vector_store_registry.create(
        "chroma",
        persist_dir=str(storage_dir / "chroma"),
        collection_name=collection_name,
        dimension=embedding.dimension,
    )

    strategy = settings.retrieval.strategy
    if strategy == "simple":
        retriever = retriever_registry.create(
            "simple", vector_store=vector_store, embedding=embedding,
        )
    elif strategy == "multi_query":
        if llm_client is None:
            raise ConfigError(
                "multi_query retriever needs an llm_client; pass one or "
                "use another strategy via DATAMIND__RETRIEVAL__STRATEGY."
            )
        retriever = retriever_registry.create(
            "multi_query",
            vector_store=vector_store,
            embedding=embedding,
            llm_client=llm_client,
            llm_model=settings.llm.fallback_model or settings.llm.model,
        )
    elif strategy == "hybrid":
        retriever = retriever_registry.create(
            "hybrid", vector_store=vector_store, embedding=embedding,
        )
    else:
        raise ConfigError(
            f"Unknown retrieval strategy '{strategy}'. "
            f"Known: {retriever_registry.known()}"
        )

    return KBService(
        embedding=embedding,
        vector_store=vector_store,
        retriever=retriever,
        data_dir=settings.data.data_dir,
        retrieval_cfg=settings.retrieval,
    )


__all__ = ["KBService", "build_kb_service"]
