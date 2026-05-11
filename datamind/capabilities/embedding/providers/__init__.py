"""Embedding providers.

Importing this package registers every built-in provider with
`embedding_registry`. Optional providers (huggingface) are imported
defensively — their dependencies are opt-in.
"""
from __future__ import annotations

# Always-on: only deps are httpx, already required.
from . import openai_compatible  # noqa: F401

# Optional: needs sentence-transformers. We try/except so a base install
# still loads cleanly and the user gets a clear error only when they
# actually ask for this provider via config.
try:
    from . import huggingface  # noqa: F401
except Exception:  # pragma: no cover — optional extra
    pass

__all__: list[str] = []
