"""MemoryService — unifies short-term + long-term under one API.

Short-term (in-memory rolling window) is cheap and always-on.
Long-term (MemoryStore) uses whichever backend is configured (sqlite,
future: redis/postgres). An optional EmbeddingProvider powers semantic
recall; without it, recall falls back to lexical matching.

Namespace convention:
    "session:<session_id>"   — scoped to one conversation
    "user:<user_id>"         — cross-session but per-user
    "global"                 — system-wide knowledge
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic

from datamind.capabilities.embedding import build_embedding
from datamind.config import Settings
from datamind.core.logging import get_logger
from datamind.core.protocols import MemoryStore
from datamind.core.registry import memory_registry

# Import providers so registry is populated on module load.
from . import providers  # noqa: F401
from .extractor import extract_facts
from .short_term import ShortTermMemory, Turn

_log = get_logger("memory.service")


class MemoryService:
    def __init__(
        self,
        *,
        short_term: ShortTermMemory,
        long_term: MemoryStore,
        llm_client: AsyncAnthropic | None = None,
        llm_model: str | None = None,
    ) -> None:
        self.short_term = short_term
        self.long_term = long_term
        self._llm = llm_client
        self._model = llm_model

    # ----------------------------------------------------------- short-term

    async def append_turn(self, session_id: str, role: str, content: str, **metadata: Any) -> None:
        await self.short_term.append(session_id, role, content, **metadata)

    async def recent_turns(self, session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        return [t.to_dict() for t in await self.short_term.recent(session_id, limit=limit)]

    # ----------------------------------------------------------- long-term

    async def save(
        self,
        namespace: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return await self.long_term.save(namespace, content, metadata=metadata)

    async def recall(
        self,
        namespace: str,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        hits = await self.long_term.recall(namespace, query, top_k=top_k)
        return [h.model_dump() for h in hits]

    async def forget(self, namespace: str, item_id: str) -> bool:
        return await self.long_term.forget(namespace, item_id)

    async def list_namespaces(self) -> list[str]:
        return await self.long_term.list_namespaces()

    # -------------------------------------------------------- fact extraction

    async def extract_and_save(
        self,
        namespace: str,
        *,
        user_turn: str,
        assistant_turn: str,
    ) -> list[str]:
        """Call the fallback LLM to pull facts from the turn, persist them."""
        if not self._llm or not self._model:
            return []
        facts = await extract_facts(
            client=self._llm,
            model=self._model,
            user_turn=user_turn,
            assistant_turn=assistant_turn,
        )
        for f in facts:
            await self.save(namespace, f, metadata={"_source": "auto_extracted"})
        return facts


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_memory_service(
    settings: Settings,
    *,
    llm_client: AsyncAnthropic | None = None,
) -> MemoryService:
    # Short-term
    st = ShortTermMemory(max_turns=settings.memory.short_term_turns)

    # Long-term
    backend = settings.memory.backend
    kwargs: dict[str, Any] = {}
    if settings.memory.dsn:
        kwargs["dsn"] = settings.memory.dsn
    else:
        # Per-profile default file.
        kwargs["db_path"] = str(settings.data.storage_dir / "memory.db")

    # Embedding (optional — recall still works lexically without it)
    embedding = None
    has_creds = bool(settings.embedding.api_key or settings.llm.api_key)
    if settings.memory.long_term_enabled and has_creds:
        try:
            embedding = build_embedding(settings.embedding, fallback_llm=settings.llm)
        except Exception as exc:  # noqa: BLE001
            _log.warning("memory_embedding_disabled", extra={"err": repr(exc)})
    kwargs["embedding"] = embedding

    long_term = memory_registry.create(backend, **kwargs)

    model = settings.llm.fallback_model or settings.llm.model if llm_client else None
    return MemoryService(
        short_term=st,
        long_term=long_term,
        llm_client=llm_client,
        llm_model=model,
    )


__all__ = ["MemoryService", "build_memory_service", "Turn"]
