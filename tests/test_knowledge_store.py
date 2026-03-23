import pytest
from pyoxigraph import NamedNode, Quad
from knowledge_service.stores.knowledge import KnowledgeStore
from knowledge_service.ontology.namespaces import (
    KS_GRAPH_EXTRACTED,
    KS_GRAPH_ASSERTED,
    KS_GRAPH_ONTOLOGY,
    KS_OPPOSITE_PREDICATE,
)


@pytest.fixture
def store():
    """In-memory KnowledgeStore for tests."""
    return KnowledgeStore(data_dir=None)  # None = in-memory


class TestInsertTriple:
    def test_insert_returns_triple_hash(self, store):
        h, _ = store.insert_triple(
            subject="http://knowledge.local/data/cold_exposure",
            predicate="http://knowledge.local/schema/increases",
            object_="http://knowledge.local/data/dopamine",
            confidence=0.7,
            knowledge_type="Claim",
        )
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_insert_same_triple_returns_same_hash(self, store):
        args = dict(
            subject="http://knowledge.local/data/x",
            predicate="http://knowledge.local/schema/y",
            object_="http://knowledge.local/data/z",
            confidence=0.8,
            knowledge_type="Fact",
        )
        h1, _ = store.insert_triple(**args)
        h2, _ = store.insert_triple(**args)
        assert h1 == h2

    def test_insert_triple_returns_is_new_true_for_new_triple(self, store):
        hash_, is_new = store.insert_triple("http://s/1", "http://p/1", "obj1", 0.8, "Claim")
        assert is_new is True

    def test_insert_triple_returns_is_new_false_for_duplicate(self, store):
        store.insert_triple("http://s/2", "http://p/2", "obj2", 0.8, "Claim")
        _, is_new = store.insert_triple("http://s/2", "http://p/2", "obj2", 0.8, "Claim")
        assert is_new is False

    def test_insert_stores_rdf_star_confidence(self, store):
        store.insert_triple(
            subject="http://knowledge.local/data/a",
            predicate="http://knowledge.local/schema/b",
            object_="http://knowledge.local/data/c",
            confidence=0.75,
            knowledge_type="Claim",
        )  # return value not needed here
        results = store.get_triples_by_subject("http://knowledge.local/data/a")
        assert len(results) == 1
        assert results[0]["confidence"] == pytest.approx(0.75)
        assert results[0]["knowledge_type"] == "Claim"
        assert results[0]["graph"] == KS_GRAPH_EXTRACTED


class TestQuery:
    def test_sparql_query(self, store):
        store.insert_triple(
            subject="http://knowledge.local/data/aegis",
            predicate="http://knowledge.local/schema/uses",
            object_="http://knowledge.local/data/pg16",
            confidence=0.99,
            knowledge_type="Fact",
        )
        # Triples are now in named graphs — use GRAPH ?g to find them
        results = store.query("""
            SELECT ?o WHERE {
                GRAPH ?g {
                    <http://knowledge.local/data/aegis>
                    <http://knowledge.local/schema/uses> ?o .
                }
            }
        """)
        assert len(results) == 1
        assert "pg16" in str(results[0]["o"])


class TestUpdateConfidence:
    def test_update_changes_confidence(self, store):
        store.insert_triple(
            subject="http://knowledge.local/data/x",
            predicate="http://knowledge.local/schema/y",
            object_="http://knowledge.local/data/z",
            confidence=0.5,
            knowledge_type="Claim",
        )
        store.update_confidence(
            subject="http://knowledge.local/data/x",
            predicate="http://knowledge.local/schema/y",
            object_="http://knowledge.local/data/z",
            new_confidence=0.9,
        )
        results = store.get_triples_by_subject("http://knowledge.local/data/x")
        assert results[0]["confidence"] == pytest.approx(0.9)


class TestFindContradictions:
    def test_finds_same_predicate_different_object(self, store):
        store.insert_triple(
            subject="http://knowledge.local/data/person",
            predicate="http://knowledge.local/schema/born_in",
            object_="http://knowledge.local/data/london",
            confidence=0.8,
            knowledge_type="Claim",
        )
        candidates = store.find_contradictions(
            subject="http://knowledge.local/data/person",
            predicate="http://knowledge.local/schema/born_in",
            object_="http://knowledge.local/data/paris",
        )
        assert len(candidates) == 1
        assert "london" in str(candidates[0]["object"])


class TestFindOppositePredContradictions:
    def test_finds_opposite_predicate_conflict(self):
        store = KnowledgeStore(data_dir=None)
        # Register that 'increases' and 'decreases' are opposites in the ontology graph
        store._store.add(
            Quad(
                NamedNode("http://ks/increases"),
                KS_OPPOSITE_PREDICATE,
                NamedNode("http://ks/decreases"),
                NamedNode(KS_GRAPH_ONTOLOGY),
            )
        )
        # Insert a triple: dopamine increases serotonin
        store.insert_triple(
            "http://ks/dopamine", "http://ks/increases", "http://ks/serotonin", 0.8, "Claim"
        )
        # Now check: does 'decreases' contradict 'increases' for same S-O?
        results = store.find_opposite_predicate_contradictions(
            "http://ks/dopamine", "http://ks/decreases", "http://ks/serotonin"
        )
        assert len(results) == 1
        assert results[0]["predicate_in_store"] == "http://ks/increases"
        assert results[0]["confidence"] == pytest.approx(0.8)

    def test_no_false_positives_when_no_opposite_stored(self):
        store = KnowledgeStore(data_dir=None)
        store.insert_triple("http://ks/a", "http://ks/foo", "http://ks/b", 0.8, "Claim")
        results = store.find_opposite_predicate_contradictions(
            "http://ks/a", "http://ks/bar", "http://ks/b"
        )
        assert results == []


class TestNamedGraphs:
    def test_insert_into_named_graph(self, store):
        """Insert with explicit graph, verify it appears in results."""
        store.insert_triple(
            subject="http://knowledge.local/data/x",
            predicate="http://knowledge.local/schema/y",
            object_="http://knowledge.local/data/z",
            confidence=0.9,
            knowledge_type="Fact",
            graph=KS_GRAPH_ASSERTED,
        )
        results = store.get_triples_by_subject("http://knowledge.local/data/x")
        assert len(results) == 1
        assert results[0]["graph"] == KS_GRAPH_ASSERTED

    def test_insert_default_graph_is_extracted(self, store):
        """Insert without graph, verify KS_GRAPH_EXTRACTED is used."""
        store.insert_triple(
            subject="http://knowledge.local/data/a",
            predicate="http://knowledge.local/schema/b",
            object_="http://knowledge.local/data/c",
            confidence=0.7,
            knowledge_type="Claim",
        )
        results = store.get_triples_by_subject("http://knowledge.local/data/a")
        assert len(results) == 1
        assert results[0]["graph"] == KS_GRAPH_EXTRACTED

    def test_get_triples_filters_by_graph(self, store):
        """Insert in different graphs, verify graph filter works."""
        store.insert_triple(
            subject="http://knowledge.local/data/thing",
            predicate="http://knowledge.local/schema/p1",
            object_="http://knowledge.local/data/o1",
            confidence=0.8,
            knowledge_type="Claim",
            graph=KS_GRAPH_ASSERTED,
        )
        store.insert_triple(
            subject="http://knowledge.local/data/thing",
            predicate="http://knowledge.local/schema/p2",
            object_="http://knowledge.local/data/o2",
            confidence=0.6,
            knowledge_type="Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        # No filter — both returned
        all_results = store.get_triples_by_subject("http://knowledge.local/data/thing")
        assert len(all_results) == 2

        # Filter to asserted only
        asserted = store.get_triples_by_subject(
            "http://knowledge.local/data/thing", graphs=[KS_GRAPH_ASSERTED]
        )
        assert len(asserted) == 1
        assert asserted[0]["graph"] == KS_GRAPH_ASSERTED

        # Filter to extracted only
        extracted = store.get_triples_by_subject(
            "http://knowledge.local/data/thing", graphs=[KS_GRAPH_EXTRACTED]
        )
        assert len(extracted) == 1
        assert extracted[0]["graph"] == KS_GRAPH_EXTRACTED

    def test_contradictions_span_graphs(self, store):
        """Contradictions are detected across different named graphs."""
        store.insert_triple(
            subject="http://knowledge.local/data/person",
            predicate="http://knowledge.local/schema/born_in",
            object_="http://knowledge.local/data/london",
            confidence=0.8,
            knowledge_type="Claim",
            graph=KS_GRAPH_ASSERTED,
        )
        # Check from extracted graph perspective
        candidates = store.find_contradictions(
            subject="http://knowledge.local/data/person",
            predicate="http://knowledge.local/schema/born_in",
            object_="http://knowledge.local/data/paris",
        )
        assert len(candidates) == 1
        assert "london" in str(candidates[0]["object"])


class TestGetTriplesByObject:
    def test_finds_triples_where_entity_is_object(self, store):
        store.insert_triple(
            "http://s/a",
            "http://p/causes",
            "http://o/target",
            0.8,
            "Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        results = store.get_triples_by_object("http://o/target")
        assert len(results) == 1
        assert "subject" in results[0]
        assert "graph" in results[0]

    def test_returns_empty_for_no_match(self, store):
        results = store.get_triples_by_object("http://o/nonexistent")
        assert results == []

    def test_respects_graph_filter(self, store):
        store.insert_triple(
            "http://s/1",
            "http://p/1",
            "http://o/shared",
            0.8,
            "Claim",
            graph=KS_GRAPH_ASSERTED,
        )
        store.insert_triple(
            "http://s/2",
            "http://p/2",
            "http://o/shared",
            0.7,
            "Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        results = store.get_triples_by_object("http://o/shared", graphs=[KS_GRAPH_ASSERTED])
        assert len(results) == 1


class TestFindConnectingTriples:
    def test_finds_direct_connection(self, store):
        store.insert_triple(
            "http://e/a",
            "http://p/causes",
            "http://e/b",
            0.8,
            "Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        results = store.find_connecting_triples("http://e/a", "http://e/b")
        assert len(results) >= 1

    def test_finds_reverse_connection(self, store):
        store.insert_triple(
            "http://e/b",
            "http://p/causes",
            "http://e/a",
            0.8,
            "Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        results = store.find_connecting_triples("http://e/a", "http://e/b")
        assert len(results) >= 1

    def test_returns_empty_when_not_connected(self, store):
        store.insert_triple(
            "http://e/x",
            "http://p/1",
            "http://e/y",
            0.8,
            "Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        results = store.find_connecting_triples("http://e/x", "http://e/z")
        assert results == []
