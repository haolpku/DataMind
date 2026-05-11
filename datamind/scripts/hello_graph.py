"""End-to-end Graph smoke test. No LLM needed — it's a pure local store.

Exercises:
- upsert via triples
- entity search (exact + fuzzy)
- multi-hop traversal + relation filter
- neighbor lookup
- persist + reload
- tool-dispatch path (same as the agent loop will use)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys


async def _main() -> int:
    os.environ.setdefault("DATAMIND__DATA__PROFILE", "hello_graph_demo")
    os.environ.setdefault("DATAMIND__GRAPH__BACKEND", "networkx")
    # Graph is LLM-free; only the Settings root requires an API key.
    os.environ.setdefault(
        "DATAMIND__LLM__API_KEY",
        os.environ.get("DATAMIND__LLM__API_KEY", "sk-smoke-placeholder"),
    )

    from datamind.capabilities.graph import build_graph_service, build_graph_tools
    from datamind.config import Settings
    from datamind.core.logging import setup_logging
    from datamind.core.tools import ToolRegistry

    setup_logging("INFO")
    settings = Settings()
    settings.ensure_dirs()

    graph = build_graph_service(settings)
    # Start fresh every run.
    await graph.store.reset()

    await graph.upsert(
        [
            {"subject": "Ann", "relation": "works_at", "object": "Acme"},
            {"subject": "Bob", "relation": "works_at", "object": "Acme"},
            {"subject": "Ann", "relation": "leads", "object": "Project Alpha"},
            {"subject": "Bob", "relation": "leads", "object": "Project Beta"},
            {"subject": "Acme", "relation": "located_in", "object": "Shanghai"},
            {"subject": "Shanghai", "relation": "in_country", "object": "China"},
            {"subject": "Project Alpha", "relation": "depends_on", "object": "Project Beta"},
            {"subject": "Ann", "relation": "mentors", "object": "Bob", "confidence": 0.8},
        ]
    )
    print(f"[hello_graph] profile = {settings.data.profile}")
    print(f"[hello_graph] stats   = {graph.stats()}")

    tools = ToolRegistry()
    tools.extend(build_graph_tools(graph))
    print(f"[hello_graph] tools   = {tools.names()}")

    # --- search ---
    found = await tools.get("graph_search_entities").handler(query="shanghai", top_k=3)
    print("\n[hello_graph] search 'shanghai':")
    print(json.dumps(found, indent=2, ensure_ascii=False))

    # --- 2-hop traverse from Ann ---
    paths = await tools.get("graph_traverse").handler(start="Ann", max_hops=3)
    print(f"\n[hello_graph] traverse Ann (3 hops) -> {paths['count']} paths")
    for p in paths["paths"][:5]:
        chain = " -> ".join(p["nodes"])
        rels = " | ".join(e["relation"] for e in p["edges"])
        print(f"  score={p['score']:.2f}  {chain}   [{rels}]")

    # --- filtered traverse: only 'works_at' + 'located_in' ---
    filt = await tools.get("graph_traverse").handler(
        start="Ann",
        max_hops=3,
        relation_filter=["works_at", "located_in", "in_country"],
    )
    print("\n[hello_graph] filtered traverse (works_at/located_in/in_country):")
    for p in filt["paths"]:
        print(f"  {' -> '.join(p['nodes'])}")

    # --- neighbors of Acme ---
    nb = await tools.get("graph_neighbors").handler(entity="Acme", direction="both")
    print(f"\n[hello_graph] neighbors Acme: {nb['count']} edges")
    for e in nb["edges"]:
        print(f"  ({e['source']}) -[{e['relation']} w={e['weight']}]-> ({e['target']})")

    # --- persistence check ---
    # Reload a fresh service from disk and verify we still see Ann.
    graph2 = build_graph_service(settings)
    hits = await graph2.search_entities("Ann", top_k=1)
    assert hits, "persistence failed — Ann missing after reload"
    print("\n[hello_graph] persistence OK (reload saw Ann)")

    print("\n[hello_graph] OK")
    return 0


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
