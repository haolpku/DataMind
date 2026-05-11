"""Tool framework tests (no external services)."""
from __future__ import annotations

import pytest

from datamind.core.errors import ConfigError
from datamind.core.tools import ToolRegistry, ToolSpec


async def _echo(value: str) -> dict:
    return {"value": value}


def _spec(name: str = "echo") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Echo value",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        handler=_echo,
    )


def test_tool_spec_serialises_to_anthropic_shape():
    s = _spec()
    out = s.to_anthropic_tool()
    assert out == {
        "name": "echo",
        "description": "Echo value",
        "input_schema": s.input_schema,
    }


def test_tool_registry_add_and_lookup():
    reg = ToolRegistry()
    reg.add(_spec("a"))
    reg.add(_spec("b"))
    assert set(reg.names()) == {"a", "b"}
    assert reg.get("a").name == "a"
    assert len(reg) == 2


def test_tool_registry_rejects_duplicates():
    reg = ToolRegistry()
    reg.add(_spec("a"))
    with pytest.raises(ConfigError):
        reg.add(_spec("a"))


def test_tool_registry_as_anthropic_tools_matches_spec():
    reg = ToolRegistry()
    reg.extend([_spec("x"), _spec("y")])
    out = reg.as_anthropic_tools()
    assert len(out) == 2
    for o in out:
        assert set(o.keys()) == {"name", "description", "input_schema"}


@pytest.mark.asyncio
async def test_handler_can_be_invoked():
    spec = _spec()
    result = await spec.handler(value="hi")
    assert result == {"value": "hi"}
