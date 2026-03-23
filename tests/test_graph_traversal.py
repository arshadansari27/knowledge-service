import pytest
from unittest.mock import MagicMock
from pyoxigraph import NamedNode, Literal
from knowledge_service.stores.graph_traversal import GraphTraverser
from knowledge_service.ontology.namespaces import KS_GRAPH_EXTRACTED


def _make_knowledge_store(triples_by_subject=None, triples_by_object=None):
    """Create a mock KnowledgeStore for traversal tests."""
    ks = MagicMock()
    ks.get_triples_by_subject.side_effect = lambda uri, **kw: triples_by_subject.get(uri, [])
    ks.get_triples_by_object.side_effect = lambda uri, **kw: triples_by_object.get(uri, [])
    return ks


def _triple(subject_uri, predicate_uri, object_uri, confidence=0.8):
    """Create a triple dict matching KnowledgeStore output format."""
    return {
        "graph": KS_GRAPH_EXTRACTED,
        "subject": NamedNode(subject_uri),
        "predicate": NamedNode(predicate_uri),
        "object": NamedNode(object_uri),
        "confidence": confidence,
        "knowledge_type": "Claim",
        "valid_from": None,
        "valid_until": None,
    }


def _literal_triple(subject_uri, predicate_uri, literal_value, confidence=0.8):
    """Triple with a literal object (should not be expanded in BFS)."""
    return {
        "graph": KS_GRAPH_EXTRACTED,
        "subject": NamedNode(subject_uri),
        "predicate": NamedNode(predicate_uri),
        "object": Literal(literal_value),
        "confidence": confidence,
        "knowledge_type": "Claim",
        "valid_from": None,
        "valid_until": None,
    }


class TestBasicExpansion:
    def test_single_hop_discovers_neighbor(self):
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [_triple("http://e/a", "http://p/causes", "http://e/b")]
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=1)
        assert any(n["uri"] == "http://e/b" for n in result.nodes)

    def test_two_hop_chain(self):
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [_triple("http://e/a", "http://p/1", "http://e/b", 0.8)],
                "http://e/b": [_triple("http://e/b", "http://p/2", "http://e/c", 0.7)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2)
        uris = {n["uri"] for n in result.nodes}
        assert "http://e/b" in uris
        assert "http://e/c" in uris

    def test_empty_graph_returns_empty(self):
        ks = _make_knowledge_store(triples_by_subject={}, triples_by_object={})
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=3)
        assert result.nodes == []
        assert result.edges == []


class TestConfidencePropagation:
    def test_multiplicative_path_confidence(self):
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [_triple("http://e/a", "http://p/1", "http://e/b", 0.8)],
                "http://e/b": [_triple("http://e/b", "http://p/2", "http://e/c", 0.7)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2)
        node_c = next(n for n in result.nodes if n["uri"] == "http://e/c")
        assert node_c["confidence"] == pytest.approx(0.56, rel=0.01)

    def test_noisy_or_across_paths(self):
        """Node reachable via 2 independent paths has higher confidence."""
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [
                    _triple("http://e/a", "http://p/1", "http://e/b", 0.6),
                    _triple("http://e/a", "http://p/2", "http://e/c", 0.5),
                ],
                "http://e/b": [_triple("http://e/b", "http://p/3", "http://e/target", 0.7)],
                "http://e/c": [_triple("http://e/c", "http://p/4", "http://e/target", 0.8)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2)
        target = next(n for n in result.nodes if n["uri"] == "http://e/target")
        # Path 1: 0.6 * 0.7 = 0.42, Path 2: 0.5 * 0.8 = 0.40
        # Noisy-OR: 1 - (1-0.42)(1-0.40) = 1 - 0.58*0.60 = 0.652
        assert target["confidence"] > 0.42  # higher than either path alone
        assert target["path_count"] == 2


class TestPruningAndLimits:
    def test_low_confidence_path_pruned(self):
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [_triple("http://e/a", "http://p/1", "http://e/b", 0.2)],
                "http://e/b": [_triple("http://e/b", "http://p/2", "http://e/c", 0.3)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2, min_confidence=0.1)
        # 0.2 * 0.3 = 0.06 < 0.1, so c should not appear
        uris = {n["uri"] for n in result.nodes}
        assert "http://e/c" not in uris

    def test_cycle_does_not_loop(self):
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [_triple("http://e/a", "http://p/1", "http://e/b", 0.8)],
                "http://e/b": [_triple("http://e/b", "http://p/2", "http://e/a", 0.8)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=4)
        # Should complete without infinite loop
        assert len(result.nodes) <= 2

    def test_max_nodes_cap(self):
        # Create a fan-out where node a connects to 60 neighbors
        neighbors = {
            "http://e/a": [
                _triple("http://e/a", "http://p/1", f"http://e/n{i}", 0.9) for i in range(60)
            ]
        }
        ks = _make_knowledge_store(triples_by_subject=neighbors, triples_by_object={})
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2, max_nodes=10)
        assert len(result.nodes) <= 10

    def test_literal_objects_not_expanded(self):
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [_literal_triple("http://e/a", "http://p/has_value", "250%", 0.9)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2)
        # Literal is recorded as edge but not expanded
        assert len(result.edges) == 1
        assert len(result.nodes) == 0  # no URI neighbors found


class TestHopDistance:
    def test_hop_distance_is_minimum(self):
        """Node reachable at hop 1 and hop 2 should have hop_distance=1."""
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [
                    _triple("http://e/a", "http://p/1", "http://e/target", 0.5),
                    _triple("http://e/a", "http://p/2", "http://e/b", 0.8),
                ],
                "http://e/b": [_triple("http://e/b", "http://p/3", "http://e/target", 0.9)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2)
        target = next(n for n in result.nodes if n["uri"] == "http://e/target")
        assert target["hop_distance"] == 1
