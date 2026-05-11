"""SQLite dialect.

Uses the built-in `sqlite:///` URL. If `dsn` is omitted, writes under
storage/<profile>/demo.db. We expose `PRAGMA query_only = ON` per-connection
as an extra belt-and-braces guard against writes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import text

from datamind.core.registry import db_registry

from ..base import BaseSQLDialect


@db_registry.register("sqlite")
class SQLiteDialect(BaseSQLDialect):
    name = "sqlite"

    def build_engine(self, dsn: str | None = None, **kwargs: Any):
        if not dsn:
            # Caller passes default_path via kwargs for profile-aware fallback.
            default_path = kwargs.pop("default_path", None)
            if not default_path:
                raise ValueError("SQLite dialect needs either a DSN or default_path")
            default_path = Path(default_path)
            default_path.parent.mkdir(parents=True, exist_ok=True)
            dsn = f"sqlite:///{default_path}"
        return super().build_engine(dsn, **kwargs)

    def _before_query(self, conn: Any) -> None:
        try:
            conn.execute(text("PRAGMA query_only = ON"))
        except Exception:  # pragma: no cover — non-fatal
            pass

    def _quote_ident(self, name: str) -> str:
        return '"' + name.replace('"', '""') + '"'
