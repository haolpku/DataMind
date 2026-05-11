"""Capability implementations.

Each capability subpackage has a `providers/` module where concrete classes
register themselves with the corresponding registry. Importing the
`providers` package is enough to populate the registry — the agent layer
never imports providers directly.

Layout:
    capabilities/
      embedding/providers/   -> embedding_registry
      kb/providers/          -> retriever_registry (KB = retrieval over vector store)
      graph/providers/       -> graph_registry
      db/providers/          -> db_registry
      memory/providers/      -> memory_registry
"""
