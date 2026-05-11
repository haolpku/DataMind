"""Tool framework.

A Tool is anything the agent can call. We abstract it so that:

1. Tools have a typed schema (Anthropic tool_use format: name / description
   / input_schema).
2. Tools declare their dependencies (an `AppContext`-like services bag) via
   a factory function, so registering a tool doesn't instantiate it eagerly.
3. Tools can be grouped, filtered, and serialised to the Anthropic API in
   one call.

Design notes:
- We don't couple to Anthropic-specific types — `to_anthropic_tool()` is a
  single method on ToolSpec. Swapping providers later stays local.
- Tool handlers are async. Sync code should wrap itself via asyncio.to_thread.
- Errors raised from a handler become a tool_result with `is_error=True`;
  the agent loop in Phase 7 handles that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from .errors import ConfigError
from .registry import Registry

# Handler signature: async callable from JSON input -> JSON-serialisable output.
ToolHandler = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    """Declarative description of a tool + its async handler.

    `name` is unique across a single agent session. `input_schema` is a
    standard JSON Schema document (Draft 2020-12 subset) exactly as the
    Anthropic /v1/messages API expects.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    # Free-form metadata for UI / audit / grouping (e.g. {"group": "kb"}).
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Serialise to the JSON object the Anthropic API accepts in `tools=`."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@runtime_checkable
class ToolProvider(Protocol):
    """A provider that contributes one or more ToolSpecs.

    Used by MCP-server-style modules that want to expose a bundle of related
    tools (e.g. the KB server exposes kb_search + kb_list_documents + kb_reindex).
    """

    def build(self, **services: Any) -> list[ToolSpec]: ...


# Global registry — MCP-server-like bundles register themselves here.
tool_provider_registry: Registry = Registry("tool_provider")


class ToolRegistry:
    """A runtime-assembled collection of ToolSpecs, keyed by name."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def add(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ConfigError(f"Tool name collision: '{spec.name}' already registered")
        self._tools[spec.name] = spec

    def extend(self, specs: list[ToolSpec]) -> None:
        for s in specs:
            self.add(s)

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ConfigError(
                f"Unknown tool '{name}'. Available: {', '.join(sorted(self._tools)) or '<none>'}"
            )
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def as_anthropic_tools(self) -> list[dict[str, Any]]:
        return [t.to_anthropic_tool() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


__all__ = [
    "ToolSpec",
    "ToolHandler",
    "ToolProvider",
    "ToolRegistry",
    "tool_provider_registry",
]
