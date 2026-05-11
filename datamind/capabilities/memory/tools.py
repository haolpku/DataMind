"""Memory tools exposed to the agent."""
from __future__ import annotations

from typing import Any

from datamind.core.tools import ToolSpec, tool_provider_registry

from .service import MemoryService


def build_memory_tools(memory: MemoryService, *, default_namespace: str | None = None) -> list[ToolSpec]:
    """Build memory tools; `default_namespace` is injected when the caller omits one."""
    fallback_ns = default_namespace or "global"

    async def _save(content: str, namespace: str | None = None, metadata: dict | None = None) -> dict:
        ns = namespace or fallback_ns
        item_id = await memory.save(ns, content, metadata=metadata)
        return {"namespace": ns, "id": item_id}

    async def _recall(query: str, namespace: str | None = None, top_k: int = 5) -> dict:
        ns = namespace or fallback_ns
        hits = await memory.recall(ns, query, top_k=top_k)
        return {"namespace": ns, "query": query, "count": len(hits), "results": hits}

    async def _forget(item_id: str, namespace: str | None = None) -> dict:
        ns = namespace or fallback_ns
        ok = await memory.forget(ns, item_id)
        return {"namespace": ns, "id": item_id, "deleted": ok}

    async def _list() -> dict:
        return {"namespaces": await memory.list_namespaces()}

    return [
        ToolSpec(
            name="memory_save",
            description=(
                "Persist a durable fact to long-term memory. "
                "Use for user preferences, decisions, domain facts the user confirms. "
                "The optional namespace lets you scope the fact to a session ('session:...') "
                "or a user ('user:...'); default is the active session."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The fact to remember."},
                    "namespace": {"type": "string", "description": "Optional explicit namespace."},
                    "metadata": {"type": "object", "additionalProperties": True},
                },
                "required": ["content"],
            },
            handler=_save,
            metadata={"group": "memory"},
        ),
        ToolSpec(
            name="memory_recall",
            description=(
                "Look up relevant facts from long-term memory by semantic similarity. "
                "Call this when the user asks about something they've told you before or when prior context would help."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "namespace": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                "required": ["query"],
            },
            handler=_recall,
            metadata={"group": "memory"},
        ),
        ToolSpec(
            name="memory_forget",
            description="Delete a specific memory item by id. Obtain the id from memory_recall first.",
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "namespace": {"type": "string"},
                },
                "required": ["item_id"],
            },
            handler=_forget,
            metadata={"group": "memory", "destructive": True},
        ),
        ToolSpec(
            name="memory_list_namespaces",
            description="List every namespace that currently has stored memories.",
            input_schema={"type": "object", "properties": {}},
            handler=_list,
            metadata={"group": "memory"},
        ),
    ]


@tool_provider_registry.register("memory")
class _MemoryToolProvider:
    def build(self, **services: Any) -> list[ToolSpec]:
        m = services.get("memory_service")
        if m is None:
            raise ValueError("memory tool provider requires 'memory_service'")
        ns = services.get("default_namespace")
        return build_memory_tools(m, default_namespace=ns)


__all__ = ["build_memory_tools"]
