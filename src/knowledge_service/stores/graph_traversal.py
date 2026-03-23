"""GraphTraverser — BFS graph traversal with Bayesian confidence propagation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from knowledge_service._utils import _rdf_value_to_str


@dataclass
class TraversalResult:
    """Result of multi-hop graph traversal."""

    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    paths: list[list[dict]] = field(default_factory=list)


class GraphTraverser:
    """BFS graph traversal with multiplicative path confidence and Noisy-OR combination."""

    _MAX_HOPS_CAP = 4

    def __init__(self, knowledge_store) -> None:
        self._ks = knowledge_store

    def expand(
        self,
        seed_uris: list[str],
        max_hops: int = 4,
        min_confidence: float = 0.1,
        max_nodes: int = 50,
    ) -> TraversalResult:
        """BFS expansion from seed entities with confidence propagation.

        Each path's confidence is the product of edge confidences along it.
        When multiple paths reach the same node, Noisy-OR combines them.
        """
        max_hops = min(max_hops, self._MAX_HOPS_CAP)

        # node_uri -> list of path confidences reaching it
        node_paths: dict[str, list[float]] = {}
        # node_uri -> minimum hop distance
        node_hops: dict[str, int] = {}
        # All discovered edges (stringified)
        all_edges: list[dict] = []
        # All paths (list of edge lists)
        all_path_lists: list[list[dict]] = []

        # Frontier: list of (uri, path_confidence, hop, path_edges)
        frontier: list[tuple[str, float, int, list[dict]]] = [
            (uri, 1.0, 0, []) for uri in seed_uris
        ]
        expanded: set[str] = set(seed_uris)

        for hop in range(1, max_hops + 1):
            next_frontier: list[tuple[str, float, int, list[dict]]] = []

            for node_uri, parent_conf, _, parent_path in frontier:
                if len(node_paths) >= max_nodes:
                    break

                # Get outgoing and incoming triples
                outgoing = self._ks.get_triples_by_subject(node_uri)
                incoming = self._ks.get_triples_by_object(node_uri, limit=None)

                for triple in outgoing:
                    neighbor_uri, edge = self._process_outgoing(triple, node_uri)
                    if edge is not None and neighbor_uri is None:
                        all_edges.append(edge)  # Record literal edges without expanding
                        continue
                    if neighbor_uri is None:
                        continue
                    self._maybe_add_neighbor(
                        neighbor_uri,
                        edge,
                        parent_conf,
                        hop,
                        parent_path,
                        node_paths,
                        node_hops,
                        all_edges,
                        all_path_lists,
                        next_frontier,
                        expanded,
                        min_confidence,
                        max_nodes,
                    )

                for triple in incoming:
                    neighbor_uri, edge = self._process_incoming(triple, node_uri)
                    if neighbor_uri is None:
                        continue
                    self._maybe_add_neighbor(
                        neighbor_uri,
                        edge,
                        parent_conf,
                        hop,
                        parent_path,
                        node_paths,
                        node_hops,
                        all_edges,
                        all_path_lists,
                        next_frontier,
                        expanded,
                        min_confidence,
                        max_nodes,
                    )

            if not next_frontier:
                break
            frontier = next_frontier

        # Build ranked nodes with Noisy-OR confidence
        nodes = []
        for uri, path_confs in node_paths.items():
            if uri in seed_uris:
                continue  # Don't include seeds in discovered nodes
            combined = self._noisy_or(path_confs)
            nodes.append(
                {
                    "uri": uri,
                    "confidence": combined,
                    "hop_distance": node_hops[uri],
                    "path_count": len(path_confs),
                }
            )

        nodes.sort(key=lambda n: n["confidence"], reverse=True)

        return TraversalResult(nodes=nodes, edges=all_edges, paths=all_path_lists)

    def _process_outgoing(self, triple: dict, source_uri: str):
        """Extract neighbor URI and edge dict from an outgoing triple."""
        obj = triple.get("object")
        if obj is None:
            return None, None
        obj_str = _rdf_value_to_str(obj)
        # Skip literals (not expandable as entities)
        if not obj_str.startswith(("http://", "https://", "urn:")):
            edge = self._make_edge(source_uri, triple)
            return None, edge  # Record edge but don't expand
        edge = self._make_edge(source_uri, triple)
        return obj_str, edge

    def _process_incoming(self, triple: dict, target_uri: str):
        """Extract neighbor URI and edge dict from an incoming triple."""
        subj = triple.get("subject")
        if subj is None:
            return None, None
        subj_str = _rdf_value_to_str(subj)
        if not subj_str.startswith(("http://", "https://", "urn:")):
            return None, None
        edge = {
            "subject": subj_str,
            "predicate": _rdf_value_to_str(triple.get("predicate")),
            "object": target_uri,
            "confidence": triple.get("confidence"),
            "knowledge_type": triple.get("knowledge_type", "Relationship"),
            "trust_tier": "verified" if "asserted" in triple.get("graph", "") else "extracted",
        }
        return subj_str, edge

    def _make_edge(self, source_uri: str, triple: dict) -> dict:
        return {
            "subject": source_uri,
            "predicate": _rdf_value_to_str(triple.get("predicate")),
            "object": _rdf_value_to_str(triple.get("object")),
            "confidence": triple.get("confidence"),
            "knowledge_type": triple.get("knowledge_type", "Relationship"),
            "trust_tier": "verified" if "asserted" in triple.get("graph", "") else "extracted",
        }

    def _maybe_add_neighbor(
        self,
        neighbor_uri,
        edge,
        parent_conf,
        hop,
        parent_path,
        node_paths,
        node_hops,
        all_edges,
        all_path_lists,
        next_frontier,
        expanded,
        min_confidence,
        max_nodes,
    ):
        edge_conf = edge.get("confidence") or 0.0
        path_conf = parent_conf * edge_conf
        if path_conf < min_confidence:
            return

        # If this is a brand-new node and we've hit the cap, skip entirely
        if neighbor_uri not in node_paths and len(node_paths) >= max_nodes:
            return

        all_edges.append(edge)
        new_path = parent_path + [edge]
        all_path_lists.append(new_path)

        if neighbor_uri not in node_paths:
            node_paths[neighbor_uri] = []
        node_paths[neighbor_uri].append(path_conf)

        if neighbor_uri not in node_hops or hop < node_hops[neighbor_uri]:
            node_hops[neighbor_uri] = hop

        if neighbor_uri not in expanded and len(node_paths) < max_nodes:
            expanded.add(neighbor_uri)
            next_frontier.append((neighbor_uri, path_conf, hop, new_path))

    @staticmethod
    def _noisy_or(confidences: list[float]) -> float:
        if not confidences:
            return 0.0
        failure_product = math.prod(1.0 - c for c in confidences)
        return 1.0 - failure_product
