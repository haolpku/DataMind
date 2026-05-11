"""Embedding providers. Concrete classes land in `providers/`."""

from .factory import build_embedding

__all__ = ["build_embedding"]
