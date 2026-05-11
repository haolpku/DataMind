"""Ingest capability — agent-driven additions to KB / DB / Graph."""
from .service import IngestService, build_ingest_service
from .tools import build_ingest_tools

__all__ = [
    "IngestService",
    "build_ingest_service",
    "build_ingest_tools",
]
