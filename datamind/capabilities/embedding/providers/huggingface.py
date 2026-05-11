"""Local HuggingFace embedding provider (optional extra).

Install via: pip install datamind[huggingface]

We import sentence-transformers lazily so the base install stays lean.
"""
from __future__ import annotations

import asyncio
from typing import Any, Sequence

from datamind.core.errors import ConfigError
from datamind.core.logging import get_logger
from datamind.core.registry import embedding_registry

_log = get_logger("embedding.huggingface")


@embedding_registry.register("huggingface")
class HuggingFaceEmbedding:
    """Run an embedding model locally via sentence-transformers."""

    name = "huggingface"

    def __init__(
        self,
        *,
        model: str = "BAAI/bge-small-zh-v1.5",
        device: str | None = None,
        batch_size: int = 32,
        # These two are accepted for interface symmetry with the openai
        # provider; they have no effect for the local backend.
        api_key: Any = None,
        api_base: Any = None,
        dimension: int | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ConfigError(
                "huggingface embedding requires sentence-transformers: "
                "pip install datamind[huggingface]"
            ) from exc

        self._model_name = model
        self._batch_size = batch_size
        self._model = SentenceTransformer(model, device=device)
        # Model.get_sentence_embedding_dimension() is the authoritative source.
        self.dimension = dimension or int(self._model.get_sentence_embedding_dimension())
        _log.info(
            "hf_embedding_loaded",
            extra={"model": model, "dim": self.dimension, "device": device},
        )

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        # sentence-transformers is sync; push to a thread so we don't block.
        vecs = await asyncio.to_thread(
            self._model.encode,
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return [v.tolist() for v in vecs]

    async def embed_query(self, query: str) -> list[float]:
        vecs = await self.embed_texts([query])
        return vecs[0]
