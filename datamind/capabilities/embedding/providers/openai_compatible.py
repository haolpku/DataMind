"""OpenAI-compatible embedding provider.

Works with:
- OpenAI (api.openai.com/v1)
- SiliconFlow, DeepSeek, Moonshot, 智谱 etc. — any provider that speaks the
  POST /v1/embeddings JSON shape `{"model": ..., "input": [...]}`.
- The gateway itself (http://35.220.164.252:3888) also exposes this endpoint
  so one API key can drive both LLM and embeddings.

We use httpx directly rather than the `openai` package: fewer deps, easier
to retry, and the Anthropic-style gateway sometimes has quirks around error
shapes that are simpler to handle raw.
"""
from __future__ import annotations

import asyncio
from typing import Any, Sequence

import httpx

from datamind.core.errors import ExternalServiceError
from datamind.core.logging import get_logger
from datamind.core.registry import embedding_registry

_log = get_logger("embedding.openai_compatible")

# Published dimensions for the most common models. If the runtime sees a
# different length we trust the server and update.
_KNOWN_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "BAAI/bge-large-zh-v1.5": 1024,
    "BAAI/bge-m3": 1024,
}


@embedding_registry.register("openai_compatible")
@embedding_registry.register("openai")
class OpenAICompatibleEmbedding:
    """Call any /v1/embeddings endpoint."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        api_base: str = "https://api.openai.com/v1",
        dimension: int | None = None,
        batch_size: int = 32,
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._model = model
        # Normalise the base URL so `/v1` always appears exactly once before
        # `/embeddings`. This lets users point at either
        # "https://api.openai.com/v1" or the gateway root "http://host:3888".
        base = api_base.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        self._api_base = base
        self._batch_size = batch_size
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        # Resolve dimension: explicit arg > known default > probe on first call.
        self.dimension = dimension or _KNOWN_DIMS.get(model, 0)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenAICompatibleEmbedding":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    # --------------------------------------------------------------- public

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        all_vecs: list[list[float]] = []
        # Batch for the API call — many providers cap input size.
        for i in range(0, len(texts), self._batch_size):
            batch = list(texts[i : i + self._batch_size])
            vecs = await self._call(batch)
            all_vecs.extend(vecs)
        return all_vecs

    async def embed_query(self, query: str) -> list[float]:
        vecs = await self._call([query])
        return vecs[0]

    # -------------------------------------------------------------- private

    async def _call(self, inputs: list[str]) -> list[list[float]]:
        url = f"{self._api_base}/embeddings"
        payload = {"model": self._model, "input": inputs}

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.post(url, json=payload)
                if resp.status_code >= 500 or resp.status_code == 429:
                    # retriable
                    raise ExternalServiceError(
                        "embedding",
                        f"HTTP {resp.status_code}: {resp.text[:200]}",
                        status_code=resp.status_code,
                    )
                resp.raise_for_status()
                body = resp.json()
                data = body.get("data") or []
                if not data:
                    raise ExternalServiceError(
                        "embedding",
                        f"empty data in response: {body!r}",
                    )
                vecs = [row["embedding"] for row in data]
                # Auto-detect dimension on first successful call.
                if not self.dimension and vecs:
                    self.dimension = len(vecs[0])
                    _log.info(
                        "embedding_dimension_detected",
                        extra={"model": self._model, "dim": self.dimension},
                    )
                return vecs
            except (httpx.HTTPError, ExternalServiceError) as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                break

        raise ExternalServiceError(
            "embedding",
            f"{type(last_exc).__name__}: {last_exc}",
            cause=last_exc,
        )
