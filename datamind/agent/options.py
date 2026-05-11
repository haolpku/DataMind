"""Agent assembly — wire all capabilities into a single DataMindAgent.

One function: `build_agent(settings)` returns a ready-to-use agent with
every tool registered. Cheap to call; builds each capability service once
and shares the Anthropic client across them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

from datamind.capabilities.db import DBService, build_db_service, build_db_tools
from datamind.capabilities.graph import GraphService, build_graph_service, build_graph_tools
from datamind.capabilities.ingest import (
    IngestService,
    build_ingest_service,
    build_ingest_tools,
)
from datamind.capabilities.kb import KBService, build_kb_service, build_kb_tools
from datamind.capabilities.memory import (
    MemoryService,
    build_memory_service,
    build_memory_tools,
)
from datamind.capabilities.skills import SkillsService, build_skills_service, build_skills_tools
from datamind.config import Settings
from datamind.core.logging import get_logger
from datamind.core.tools import ToolRegistry

from .base import AgentLoopConfig, AgentLoopProtocol
from .loop_native import NativeAgentLoop
from .prompts import build_system_prompt

_log = get_logger("agent.assemble")


@dataclass
class DataMindAgent:
    """Top-level handle exposing every piece a caller might want."""

    client: AsyncAnthropic
    tools: ToolRegistry
    loop: AgentLoopProtocolProtocol
    kb: KBService
    db: DBService
    graph: GraphService
    skills: SkillsService
    memory: MemoryService
    ingest: IngestService

    async def warmup(self) -> dict[str, Any]:
        """Load skills index, graph triplets, etc. Returns a stats dict."""
        info: dict[str, Any] = {}
        info["skills"] = await self.skills.load()
        info["graph"] = await self.graph.load_from_profile()
        info["kb_chunks"] = await self.kb.count()
        _log.info("agent_warmup", extra=info)
        return info


async def build_agent(
    settings: Settings,
    *,
    enable: set[str] | None = None,
    default_memory_namespace: str = "session:default",
) -> DataMindAgent:
    """Assemble every capability + the agent loop.

    `enable` lets you restrict which tool groups are active — handy in
    tests where e.g. you don't want the graph warmup to hit the filesystem.
    Defaults to everything.
    """
    active = enable or {"kb", "db", "graph", "skills", "memory", "ingest"}

    client = AsyncAnthropic(
        base_url=str(settings.llm.api_base),
        api_key=settings.llm.api_key.get_secret_value(),
        timeout=settings.llm.timeout_s,
    )

    tools = ToolRegistry()

    # KB
    kb = build_kb_service(settings, llm_client=client)
    if "kb" in active:
        tools.extend(build_kb_tools(kb))

    # DB
    db = build_db_service(settings, llm_client=client)
    if "db" in active:
        tools.extend(build_db_tools(db))

    # Graph
    graph = build_graph_service(settings)
    if "graph" in active:
        tools.extend(build_graph_tools(graph))

    # Skills
    skills = build_skills_service(settings)
    if "skills" in active:
        tools.extend(build_skills_tools(skills))

    # Memory
    memory = build_memory_service(settings, llm_client=client)
    if "memory" in active:
        tools.extend(build_memory_tools(memory, default_namespace=default_memory_namespace))

    # Ingest — agent-driven additions to KB / DB / Graph. Built last so it
    # can wire into already-constructed services. Tools are registered
    # under the "ingest" group, easy to disable wholesale via permissions.
    ingest = build_ingest_service(
        settings=settings,
        kb=kb,
        db=db,
        graph=graph,
        llm_client=client,
    )
    if "ingest" in active:
        tools.extend(build_ingest_tools(ingest))

    system = build_system_prompt(
        [tools.get(n) for n in tools.names()]
    )

    # Pick the agent-loop backend based on settings. Both satisfy
    # AgentLoopProtocol — everything downstream is backend-agnostic.
    loop_config = AgentLoopConfig(
        model=settings.llm.model,
        max_tokens=settings.llm.max_tokens,
        temperature=settings.llm.temperature,
        system_prompt=system,
        max_tool_turns=settings.agent.max_turns,
    )

    loop: AgentLoopProtocol
    if settings.agent.backend == "sdk":
        # Import locally so the `claude-agent-sdk` dependency is only
        # loaded when actually selected. Native users don't pay for it.
        from .loop_sdk import SdkAgentLoop  # noqa: PLC0415

        loop = SdkAgentLoop(
            tools=tools,
            config=loop_config,
            ccr_base_url=settings.agent.ccr_base_url,
            ccr_api_key=settings.agent.ccr_api_key.get_secret_value(),
        )
        _log.info("agent_loop_backend", extra={"backend": "sdk", "ccr": settings.agent.ccr_base_url})
    else:
        loop = NativeAgentLoop(
            client=client,
            tools=tools,
            config=loop_config,
        )
        _log.info("agent_loop_backend", extra={"backend": "native"})

    return DataMindAgent(
        client=client,
        tools=tools,
        loop=loop,
        kb=kb,
        db=db,
        graph=graph,
        skills=skills,
        memory=memory,
        ingest=ingest,
    )


__all__ = ["DataMindAgent", "build_agent"]
