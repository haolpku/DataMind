"""Probe whether claude-agent-sdk can talk to the model gateway via
claude-code-router (CCR) running on localhost.

Architecture:
    SDK (Python) ── stdio ──► claude CLI ── HTTP ──► CCR (localhost) ── HTTP ──► remote OpenAI-compatible gateway

CCR translates between Anthropic /v1/messages and OpenAI /v1/chat/completions.
This sidesteps the issue we hit with the official Anthropic-format gateway,
where some installed CLI binaries inject vendor headers that our gateway 401s on.

Pre-req: CCR must already be running. Default expectation:
    - $CCR_BASE = http://127.0.0.1:13456
    - config under $HOME/.claude-code-router/config.json points at the real model API.

If this exits 0 with a 'pong' line, the SDK route is viable.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    UserMessage,
    query,
)


def _load_env() -> tuple[str, str, str]:
    """Read CCR target + dummy auth + model name."""
    base = os.environ.get("CCR_BASE", "http://127.0.0.1:13456")
    key = os.environ.get("CCR_APIKEY") or os.environ.get("DATAMIND__LLM__API_KEY") or "dummy"
    model = os.environ.get("DATAMIND__LLM__MODEL", "claude-sonnet-4-6")
    return base, key, model


async def _run() -> int:
    base, key, model = _load_env()
    print(f"[hello_sdk_native] router  = {base}  (CCR — translates to OpenAI gateway)")
    print(f"[hello_sdk_native] model   = {model}")
    print(f"[hello_sdk_native] sdk     = claude-agent-sdk")

    def _stderr_relay(line: str) -> None:
        sys.stderr.write(f"[cli] {line}\n")
        sys.stderr.flush()

    options = ClaudeAgentOptions(
        model=model,
        system_prompt="You are a smoke test. Reply with exactly one word: pong",
        env={
            "ANTHROPIC_BASE_URL": base,
            "ANTHROPIC_API_KEY": key,
            "ANTHROPIC_AUTH_TOKEN": key,
            "DISABLE_TELEMETRY": "1",
            "DISABLE_AUTOUPDATER": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "NO_PROXY": "127.0.0.1,localhost",
        },
        permission_mode="bypassPermissions",
        max_turns=1,
        allowed_tools=[],
        load_timeout_ms=30000,
        stderr=_stderr_relay,
    )

    started = time.monotonic()
    saw_text = False
    try:
        async for msg in query(prompt="Reply with exactly one word: pong", options=options):
            elapsed = time.monotonic() - started
            tag = type(msg).__name__
            if isinstance(msg, SystemMessage):
                print(f"[{elapsed:5.1f}s] SystemMessage subtype={msg.subtype}")
                if msg.subtype in ("api_retry", "api_error"):
                    print(f"            .data = {getattr(msg, 'data', None)!r}"[:300])
            elif isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(f"[{elapsed:5.1f}s] text: {block.text!r}")
                        if "pong" in block.text.lower():
                            saw_text = True
            elif isinstance(msg, UserMessage):
                print(f"[{elapsed:5.1f}s] UserMessage (echo)")
            elif isinstance(msg, ResultMessage):
                print(
                    f"[{elapsed:5.1f}s] ResultMessage subtype={msg.subtype} "
                    f"is_error={msg.is_error} duration_ms={msg.duration_ms} "
                    f"cost_usd={msg.total_cost_usd}"
                )
            else:
                print(f"[{elapsed:5.1f}s] {tag}")
    except Exception as exc:
        print(f"[hello_sdk_native] ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    if saw_text:
        print("[hello_sdk_native] OK: SDK reached gateway via CCR, model replied 'pong'.")
        return 0
    print("[hello_sdk_native] FAILED: stream ended without 'pong' text.", file=sys.stderr)
    return 1


def main() -> None:
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
