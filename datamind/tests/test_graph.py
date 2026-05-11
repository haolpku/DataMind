"""Graph store tests (no network)."""
from __future__ import annotations

import pytest

from datamind.capabilities.graph.providers.networkx_store import NetworkXGraphStore
from datamind.core.protocols import GraphStore, GraphTriple


@pytest.fixture
def store(tmp_path):
    s = NetworkXGraphStore(persist_path=tmp_path / "g.json")
    return s


def _triples() -> list[GraphTriple]:
    # Small company-style graph.
    return [
        GraphTriple(subject="Ann", relation="works_at", object="Acme"),
        GraphTriple(subject="Bob", relation="works_at", object="Acme"),
        GraphTriple(subject="Acme", relation="located_in", object="Shanghai"),
        GraphTriple(subject="Ann", relation="leads", object="ProjectAlpha"),
        GraphTriple(subject="ProjectAlpha", relation="depends_on", object="ProjectBeta"),
        GraphTriple(subject="Bob", relation="leads", object="ProjectBeta"),
        GraphTriple(subject="Shanghai", relation="in_country", object="China", confidence=1.0),
    ]


@pytest.mark.asyncio
async def test_upsert_and_stats(store):
    await store.upsert_triples(_triples())
    stats = store.stats()
    assert stats["nodes"] >= 6 and stats["edges"] == 7


@pytest.mark.asyncio
async def test_search_entities_exact_and_fuzzy(store):
    await store.upsert_triples(_triples())
    hits = await store.search_entities("Acme", top_k=3)
    assert hits and hits[0].id == "Acme" and hits[0].score == 1.0

    # Fuzzy match — case-insensitive substring
    hits = await store.search_entities("shang", top_k=3)
    assert any(h.id == "Shanghai" for h in hits)


@pytest.mark.asyncio
async def test_traverse_respects_max_hops(store):
    await store.upsert_triples(_triples())
    # Ann -> Acme (1), Acme -> Shanghai (2), Shanghai -> China (3)
    one_hop = await store.traverse("Ann", max_hops=1)
    two_hop = await store.traverse("Ann", max_hops=2)
    three_hop = await store.traverse("Ann", max_hops=3)
    assert len(one_hop) <= len(two_hop) <= len(three_hop)
    # At least one 3-hop path should reach China.
    reachable = {p.nodes[-1] for p in three_hop}
    assert "China" in reachable
    # But a 1-hop path cannot.
    reachable_one = {p.nodes[-1] for p in one_hop}
    assert "China" not in reachable_one


@pytest.mark.asyncio
async def test_traverse_relation_filter(store):
    await store.upsert_triples(_triples())
    paths = await store.traverse("Ann", max_hops=3, relation_filter=["works_at", "located_in"])
    # From Ann, only works_at / located_in — so we should reach Shanghai but NOT ProjectAlpha.
    visited = {n for p in paths for n in p.nodes}
    assert "Shanghai" in visited
    assert "ProjectAlpha" not in visited


@pytest.mark.asyncio
async def test_neighbors_direction(store):
    await store.upsert_triples(_triples())
    out_edges = await store.neighbors("Ann", direction="out")
    in_edges = await store.neighbors("Ann", direction="in")
    both = await store.neighbors("Ann", direction="both")
    assert all(e.source == "Ann" for e in out_edges)
    assert all(e.target == "Ann" for e in in_edges)
    assert len(both) == len(out_edges) + len(in_edges)


@pytest.mark.asyncio
async def test_persist_and_reload_roundtrip(tmp_path):
    path = tmp_path / "g.json"
    s1 = NetworkXGraphStore(persist_path=path)
    await s1.upsert_triples(_triples())
    await s1.persist()

    s2 = NetworkXGraphStore(persist_path=path)
    hits = await s2.search_entities("ProjectBeta")
    assert hits and hits[0].id == "ProjectBeta"


def test_satisfies_protocol(store):
    assert isinstance(store, GraphStore)
