"""Structured JSON logging with trace-id injection.

Design:
- Stdlib `logging` only — no heavyweight dependency.
- Each record is serialised as one-line JSON for ingestion into any log
  aggregator.
- A `contextvars.ContextVar` holds the active `RequestContext`; when set,
  `trace_id` / `session_id` / `profile` are automatically attached to
  every record emitted within that scope.

Usage:
    from datamind.core.logging import get_logger, bind_context, setup_logging
    setup_logging()                           # call once at process start
    with bind_context(ctx):                   # inside a request handler
        get_logger("kb").info("searching", extra={"query": "hi"})

The `extra=` kwarg on stdlib loggers is the blessed way to add
key-value fields; our formatter merges them into the JSON payload.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .context import RequestContext

_CTX_VAR: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar(
    "datamind_request_context", default=None
)

# stdlib LogRecord attributes we must NOT echo back into the JSON payload
# (they're already captured at top-level).
_RESERVED_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
        "taskName",
    }
)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        ctx = _CTX_VAR.get()
        if ctx is not None:
            payload["trace_id"] = ctx.trace_id
            payload["session_id"] = ctx.session_id
            payload["profile"] = ctx.profile
            if ctx.user_id:
                payload["user_id"] = ctx.user_id

        # Merge `extra=...` fields (logging adds them as attributes on record).
        for key, value in record.__dict__.items():
            if key in _RESERVED_ATTRS or key.startswith("_"):
                continue
            # Skip non-serialisable values defensively.
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: int | str = "INFO") -> None:
    """Install the JSON formatter on the root logger. Idempotent."""
    root = logging.getLogger()
    root.setLevel(level)
    # Drop any pre-existing handlers to avoid double-logging in tests.
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"datamind.{name}")


@contextmanager
def bind_context(ctx: RequestContext) -> Iterator[RequestContext]:
    """Attach `ctx` to the async task / thread-local stack of log records."""
    token = _CTX_VAR.set(ctx)
    try:
        yield ctx
    finally:
        _CTX_VAR.reset(token)


def current_context() -> RequestContext | None:
    """Return the context bound to the current task, if any."""
    return _CTX_VAR.get()
