"""Graph traversal utilities for multi-hop expansion."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TraversalResult:
    """Result of a multi-hop graph expansion."""

    edges: list[dict] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)


class GraphTraverser:
    """Expand entities through the knowledge graph for multi-hop retrieval."""

    def __init__(self, knowledge_store):
        self._store = knowledge_store

    def expand(
        self,
        entity_uris: str | list[str],
        max_hops: int = 2,
        min_confidence: float = 0.0,
    ) -> TraversalResult:
        """Return triples reachable within *max_hops* of the given entity URIs.

        Args:
            entity_uris: A single URI string or list of seed entity URIs.
            max_hops: Maximum number of hops to traverse.
            min_confidence: Minimum confidence threshold for included triples.

        Returns:
            TraversalResult with edges (triples) and nodes (visited entities).
        """
        if isinstance(entity_uris, str):
            entity_uris = [entity_uris]

        visited: set[str] = set()
        edges: list[dict] = []
        nodes: list[dict] = []
        frontier = [(uri, 0) for uri in entity_uris]

        while frontier:
            uri, hop = frontier.pop(0)
            if uri in visited or hop > max_hops:
                continue
            visited.add(uri)
            nodes.append({"uri": uri, "hop_distance": hop})

            # Get triples where this entity is the subject
            triples = self._store.get_triples(subject=uri)
            for t in triples:
                conf = t.get("confidence", 0)
                if conf is not None and conf >= min_confidence:
                    edges.append(t)
                    # Follow object if it looks like a URI (entity)
                    obj = t.get("object", "")
                    if isinstance(obj, str) and obj.startswith(("http://", "https://", "urn:")):
                        if obj not in visited and hop + 1 <= max_hops:
                            frontier.append((obj, hop + 1))

        return TraversalResult(edges=edges, nodes=nodes)
