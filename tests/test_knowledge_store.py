import pytest
from knowledge_service.stores.knowledge import KnowledgeStore


@pytest.fixture
def store():
    """In-memory KnowledgeStore for tests."""
    return KnowledgeStore(data_dir=None)  # None = in-memory


class TestInsertTriple:
    def test_insert_returns_triple_hash(self, store):
        h = store.insert_triple(
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
        h1 = store.insert_triple(**args)
        h2 = store.insert_triple(**args)
        assert h1 == h2

    def test_insert_stores_rdf_star_confidence(self, store):
        store.insert_triple(
            subject="http://knowledge.local/data/a",
            predicate="http://knowledge.local/schema/b",
            object_="http://knowledge.local/data/c",
            confidence=0.75,
            knowledge_type="Claim",
        )
        results = store.get_triples_by_subject("http://knowledge.local/data/a")
        assert len(results) == 1
        assert results[0]["confidence"] == pytest.approx(0.75)
        assert results[0]["knowledge_type"] == "Claim"


class TestQuery:
    def test_sparql_query(self, store):
        store.insert_triple(
            subject="http://knowledge.local/data/aegis",
            predicate="http://knowledge.local/schema/uses",
            object_="http://knowledge.local/data/pg16",
            confidence=0.99,
            knowledge_type="Fact",
        )
        results = store.query("""
            SELECT ?o WHERE {
                <http://knowledge.local/data/aegis>
                <http://knowledge.local/schema/uses> ?o .
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
