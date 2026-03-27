"""Graph traversal utilities for multi-hop expansion."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GraphTraverser:
    """Expand entities through the knowledge graph for multi-hop retrieval."""

    def __init__(self, knowledge_store):
        self._store = knowledge_store

    def expand(self, entity_uri: str, max_hops: int = 2) -> list[dict]:
        """Return triples reachable within *max_hops* of *entity_uri*."""
        return self._store.get_triples(subject=entity_uri)
