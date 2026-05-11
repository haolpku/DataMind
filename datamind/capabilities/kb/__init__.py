"""Knowledge base (vector RAG). Retrieval strategies live in `providers/`."""

from .service import KBService, build_kb_service
from .tools import build_kb_tools

__all__ = ["KBService", "build_kb_service", "build_kb_tools"]
