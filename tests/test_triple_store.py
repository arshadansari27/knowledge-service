# tests/test_triple_store.py
import pytest
from knowledge_service.stores.triples import TripleStore
from knowledge_service.ontology.uri import KS, KS_DATA
from knowledge_service.ontology.namespaces import KS_GRAPH_EXTRACTED, KS_GRAPH_ONTOLOGY


@pytest.fixture
def store():
    return TripleStore(data_dir=None)


class TestInsert:
    def test_returns_hash_and_is_new(self, store):
        h, is_new = store.insert(
            f"{KS_DATA}dopamine",
            f"{KS}causes",
            f"{KS_DATA}alertness",
            confidence=0.8,
            knowledge_type="claim",
            valid_from=None,
            valid_until=None,
            graph=KS_GRAPH_EXTRACTED,
        )
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex
        assert is_new is True

    def test_idempotent(self, store):
        args = (
            f"{KS_DATA}a",
            f"{KS}b",
            f"{KS_DATA}c",
            0.8,
            "claim",
            None,
            None,
            KS_GRAPH_EXTRACTED,
        )
        h1, new1 = store.insert(*args)
        h2, new2 = store.insert(*args)
        assert h1 == h2
        assert new1 is True
        assert new2 is False

    def test_deterministic_hash(self, store):
        args = (
            f"{KS_DATA}x",
            f"{KS}y",
            f"{KS_DATA}z",
            0.9,
            "fact",
            None,
            None,
            KS_GRAPH_EXTRACTED,
        )
        h1, _ = store.insert(*args)
        store2 = TripleStore(data_dir=None)
        h2, _ = store2.insert(*args)
        assert h1 == h2

    def test_insert_literal_with_special_chars(self, store):
        """Literals containing quotes, backslashes, or newlines must not break SPARQL."""
        for obj in ['has "quotes"', "line1\nline2", "back\\slash", 'all "three"\nin\\one']:
            h, is_new = store.insert(
                f"{KS_DATA}entity",
                f"{KS}description",
                obj,
                confidence=0.7,
                knowledge_type="claim",
                valid_from=None,
                valid_until=None,
                graph=KS_GRAPH_EXTRACTED,
            )
            assert isinstance(h, str)
            assert is_new is True
            # Verify retrievable
            triples = store.get_triples(subject=f"{KS_DATA}entity", predicate=f"{KS}description")
            assert any(t["object"] == obj for t in triples)


class TestGetTriples:
    def test_by_subject(self, store):
        store.insert(
            f"{KS_DATA}a", f"{KS}p1", f"{KS_DATA}b", 0.8, "claim", None, None, KS_GRAPH_EXTRACTED
        )
        store.insert(
            f"{KS_DATA}a", f"{KS}p2", f"{KS_DATA}c", 0.9, "fact", None, None, KS_GRAPH_EXTRACTED
        )
        triples = store.get_triples(subject=f"{KS_DATA}a")
        assert len(triples) == 2

    def test_by_predicate(self, store):
        store.insert(
            f"{KS_DATA}a",
            f"{KS}causes",
            f"{KS_DATA}b",
            0.8,
            "claim",
            None,
            None,
            KS_GRAPH_EXTRACTED,
        )
        store.insert(
            f"{KS_DATA}c",
            f"{KS}causes",
            f"{KS_DATA}d",
            0.7,
            "claim",
            None,
            None,
            KS_GRAPH_EXTRACTED,
        )
        triples = store.get_triples(predicate=f"{KS}causes")
        assert len(triples) == 2

    def test_by_subject_and_predicate(self, store):
        store.insert(
            f"{KS_DATA}a", f"{KS}p1", f"{KS_DATA}b", 0.8, "claim", None, None, KS_GRAPH_EXTRACTED
        )
        store.insert(
            f"{KS_DATA}a", f"{KS}p2", f"{KS_DATA}c", 0.9, "fact", None, None, KS_GRAPH_EXTRACTED
        )
        triples = store.get_triples(subject=f"{KS_DATA}a", predicate=f"{KS}p1")
        assert len(triples) == 1

    def test_filter_by_graph(self, store):
        store.insert(
            f"{KS_DATA}a", f"{KS}p", f"{KS_DATA}b", 0.8, "claim", None, None, KS_GRAPH_EXTRACTED
        )
        assert len(store.get_triples(subject=f"{KS_DATA}a", graphs=[KS_GRAPH_EXTRACTED])) == 1
        assert len(store.get_triples(subject=f"{KS_DATA}a", graphs=[KS_GRAPH_ONTOLOGY])) == 0

    def test_includes_annotations(self, store):
        store.insert(
            f"{KS_DATA}a", f"{KS}p", f"{KS_DATA}b", 0.85, "claim", None, None, KS_GRAPH_EXTRACTED
        )
        triples = store.get_triples(subject=f"{KS_DATA}a")
        assert triples[0]["confidence"] == pytest.approx(0.85)
        assert triples[0]["knowledge_type"] == "claim"


class TestUpdateConfidence:
    def test_updates(self, store):
        store.insert(
            f"{KS_DATA}a", f"{KS}p", f"{KS_DATA}b", 0.5, "claim", None, None, KS_GRAPH_EXTRACTED
        )
        store.update_confidence(
            {"subject": f"{KS_DATA}a", "predicate": f"{KS}p", "object": f"{KS_DATA}b"}, 0.9
        )
        triples = store.get_triples(subject=f"{KS_DATA}a")
        assert triples[0]["confidence"] == pytest.approx(0.9)


class TestContradictions:
    def test_same_predicate_different_object(self, store):
        store.insert(
            f"{KS_DATA}a", f"{KS}revenue", "50M", 0.8, "claim", None, None, KS_GRAPH_EXTRACTED
        )
        store.insert(
            f"{KS_DATA}a", f"{KS}revenue", "60M", 0.9, "claim", None, None, KS_GRAPH_EXTRACTED
        )
        contras = store.find_contradictions(f"{KS_DATA}a", f"{KS}revenue", "60M")
        assert len(contras) >= 1
        assert any(c["object"] == "50M" for c in contras)
