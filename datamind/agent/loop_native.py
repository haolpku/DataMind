"""Native tool-use loop on the `anthropic` Python SDK.

This is the default backend (`DATAMIND__AGENT__BACKEND=native`). Talks
directly to whatever `LLMConfig.api_base` points at — no local helper
process required. A straightforward implementation of the tool-use
protocol:

    1. Send [history + user_message] with tools=[...] to /v1/messages
    2. If stop_reason == "tool_use":
         - For each tool_use block:
             - Invoke ToolRegistry[name].handler(**input)
             - Append tool_result block
         - Loop to step 1 with the tool_result(s) appended as user message
    3. Otherwise: emit the final assistant text and return

Both `run_turn` (non-streaming) and `stream_turn` (async generator of
`AgentEvent`s) are exposed; the server uses `stream_turn` for SSE.

Hooks (`on_tool_start`, `on_tool_end`) fire at well-defined points — the
seam Phase 8 uses for audit logging and PreToolUse rejection.

Renamed from `AgentLoop` to `NativeAgentLoop` in Phase 11 when we added
the parallel `SdkAgentLoop` option. Shared types (`AgentEvent`,
`AgentLoopConfig`, protocol, hook signatures) moved to `base.py`.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic
from anthropic.types import Message

from datamind.core.logging import get_logger
from datamind.core.tools import ToolRegistry

from .base import AgentEvent, AgentLoopConfig, OnToolEnd, OnToolStart

_log = get_logger("agent.loop.native")


class NativeAgentLoop:
    """One instance per (client, tools) pair; safe to share across requests."""

    def __init__(
        self,
        *,
        client: AsyncAnthropic,
        tools: ToolRegistry,
        config: AgentLoopConfig,
        on_tool_start: OnToolStart | None = None,
        on_tool_end: OnToolEnd | None = None,
    ) -> None:
        self._client = client
        self._tools = tools
        self._cfg = config
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end

    # ----------------------------------------------------------- helpers

    async def _dispatch_tool(self, name: str, tool_input: dict) -> tuple[Any, Exception | None]:
        try:
            spec = self._tools.get(name)
        except Exception as exc:  # unknown tool
            return None, exc
        if self._on_tool_start:
            try:
                await self._on_tool_start(name, tool_input)
            except Exception as exc:  # hook failure is non-fatal
                _log.warning("on_tool_start_failed", extra={"err": repr(exc)})
        try:
            result = await spec.handler(**tool_input)
            err: Exception | None = None
        except Exception as exc:  # noqa: BLE001
            result = None
            err = exc
        if self._on_tool_end:
            try:
                await self._on_tool_end(name, tool_input, result, err)
            except Exception as exc:
                _log.warning("on_tool_end_failed", extra={"err": repr(exc)})
        return result, err

    @staticmethod
    def _block_to_dict(block: Any) -> dict[str, Any]:
        """Normalise Anthropic response blocks into plain dicts for history."""
        t = getattr(block, "type", None)
        if t == "text":
            return {"type": "text", "text": block.text}
        if t == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": dict(block.input or {}),
            }
        # Best effort — anything else we keep as JSON-stringified blob.
        try:
            return dict(block.to_dict())  # type: ignore[attr-defined]
        except AttributeError:
            return {"type": t or "unknown", "data": str(block)}

    def _tool_result_block(self, tool_use_id: str, result: Any, err: Exception | None) -> dict:
        if err is None:
            # Stringify unless already a string — the API accepts both but
            # models read plain text better.
            if isinstance(result, str):
                content = result
            else:
                try:
                    content = json.dumps(result, ensure_ascii=False)
                except (TypeError, ValueError):
                    content = str(result)
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "is_error": True,
            "content": f"{type(err).__name__}: {err}",
        }

    # ---------------------------------------------------------------- API

    async def run_turn(
        self,
        *,
        user_message: str,
        history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Run one user turn to completion. Returns {answer, history, usage}."""
        conv: list[dict] = list(history or [])
        conv.append({"role": "user", "content": user_message})

        total_input = 0
        total_output = 0
        total_cache_read = 0
        total_cache_create = 0

        for iteration in range(self._cfg.max_tool_turns):
            resp: Message = await self._client.messages.create(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                system=self._cfg.system_prompt or None,
                tools=self._tools.as_anthropic_tools() or None,
                messages=conv,
            )
            total_input += resp.usage.input_tokens
            total_output += resp.usage.output_tokens
            total_cache_read += getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            total_cache_create += getattr(resp.usage, "cache_creation_input_tokens", 0) or 0

            assistant_blocks = [self._block_to_dict(b) for b in resp.content]
            conv.append({"role": "assistant", "content": assistant_blocks})

            if resp.stop_reason != "tool_use":
                text = "".join(b["text"] for b in assistant_blocks if b.get("type") == "text")
                return {
                    "answer": text,
                    "history": conv,
                    "iterations": iteration + 1,
                    "stop_reason": resp.stop_reason,
                    "usage": {
                        "input_tokens": total_input,
                        "output_tokens": total_output,
                        "cache_read": total_cache_read,
                        "cache_create": total_cache_create,
                    },
                }

            # Dispatch every tool_use block, collect tool_result blocks.
            tool_results: list[dict] = []
            for b in assistant_blocks:
                if b.get("type") != "tool_use":
                    continue
                result, err = await self._dispatch_tool(b["name"], b["input"])
                tool_results.append(self._tool_result_block(b["id"], result, err))
            conv.append({"role": "user", "content": tool_results})

        # Hit the iteration cap.
        return {
            "answer": "（已达到工具调用上限，请重新提问或缩小范围）",
            "history": conv,
            "iterations": self._cfg.max_tool_turns,
            "stop_reason": "max_iterations",
            "usage": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cache_read": total_cache_read,
                "cache_create": total_cache_create,
            },
        }

    async def stream_turn(
        self,
        *,
        user_message: str,
        history: list[dict] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Like run_turn but yields AgentEvents as they happen.

        Streams token deltas from the assistant, emits tool_use and
        tool_result events, then a final 'done'.
        """
        conv: list[dict] = list(history or [])
        conv.append({"role": "user", "content": user_message})

        for iteration in range(self._cfg.max_tool_turns):
            async with self._client.messages.stream(
                model=self._cfg.model,
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                system=self._cfg.system_prompt or None,
                tools=self._tools.as_anthropic_tools() or None,
                messages=conv,
            ) as stream:
                async for text_delta in stream.text_stream:
                    if text_delta:
                        yield AgentEvent(type="text", data={"delta": text_delta})
                final = await stream.get_final_message()

            assistant_blocks = [self._block_to_dict(b) for b in final.content]
            conv.append({"role": "assistant", "content": assistant_blocks})

            if final.stop_reason != "tool_use":
                yield AgentEvent(
                    type="done",
                    data={
                        "iterations": iteration + 1,
                        "stop_reason": final.stop_reason,
                        "usage": {
                            "input_tokens": final.usage.input_tokens,
                            "output_tokens": final.usage.output_tokens,
                        },
                    },
                )
                return

            tool_results: list[dict] = []
            for b in assistant_blocks:
                if b.get("type") != "tool_use":
                    continue
                yield AgentEvent(
                    type="tool_use",
                    data={"name": b["name"], "input": b["input"], "id": b["id"]},
                )
                result, err = await self._dispatch_tool(b["name"], b["input"])
                tr = self._tool_result_block(b["id"], result, err)
                tool_results.append(tr)
                yield AgentEvent(
                    type="tool_result",
                    data={
                        "name": b["name"],
                        "is_error": bool(err),
                        "preview": tr["content"][:500] if isinstance(tr["content"], str) else None,
                    },
                )
            conv.append({"role": "user", "content": tool_results})

        yield AgentEvent(
            type="done",
            data={"stop_reason": "max_iterations", "iterations": self._cfg.max_tool_turns},
        )


__all__ = ["NativeAgentLoop"]
