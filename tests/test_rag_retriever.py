"""Unit tests for RAGRetriever — retrieval pipeline across stores."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from pyoxigraph import Literal, NamedNode

from knowledge_service.stores.rag import RAGRetriever, RetrievalContext


def _make_embedding_client():
    mock = AsyncMock()
    mock.embed.return_value = [0.1] * 768
    return mock


def _make_embedding_store(content_rows=None, entity_rows=None):
    mock = AsyncMock()
    mock.search.return_value = content_rows or []
    mock.search_entities.return_value = entity_rows or []
    return mock


def _make_knowledge_store(triples=None, contradictions=None):
    mock = MagicMock()
    mock.get_triples_by_subject.return_value = triples or []
    mock.find_contradictions.return_value = contradictions or []
    return mock


_CONTENT_ROW = {
    "id": "chunk-uuid-1",
    "chunk_text": "Relevant text about the topic",
    "chunk_index": 0,
    "content_id": "content-uuid-1",
    "url": "https://example.com/article",
    "title": "Test Article",
    "summary": "A summary",
    "source_type": "article",
    "tags": ["health"],
    "ingested_at": "2026-03-18T10:00:00Z",
    "similarity": 0.92,
}

_ENTITY_ROW = {
    "uri": "http://knowledge.local/data/dopamine",
    "label": "Dopamine",
    "rdf_type": "schema:ChemicalSubstance",
    "similarity": 0.85,
}


class TestRetrieveContentSearch:
    async def test_returns_retrieval_context(self):
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=_make_embedding_store(content_rows=[_CONTENT_ROW]),
            knowledge_store=_make_knowledge_store(),
        )
        ctx = await retriever.retrieve("test question", max_sources=5, min_confidence=0.0)
        assert isinstance(ctx, RetrievalContext)

    async def test_content_results_populated(self):
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=_make_embedding_store(content_rows=[_CONTENT_ROW]),
            knowledge_store=_make_knowledge_store(),
        )
        ctx = await retriever.retrieve("test question", max_sources=5, min_confidence=0.0)
        assert len(ctx.content_results) == 1
        assert ctx.content_results[0]["url"] == "https://example.com/article"

    async def test_embed_called_with_question(self):
        ec = _make_embedding_client()
        retriever = RAGRetriever(
            embedding_client=ec,
            embedding_store=_make_embedding_store(),
            knowledge_store=_make_knowledge_store(),
        )
        await retriever.retrieve("cold exposure dopamine", max_sources=5, min_confidence=0.0)
        ec.embed.assert_called_once_with("cold exposure dopamine")

    async def test_search_called_with_max_sources(self):
        es = _make_embedding_store()
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
            knowledge_store=_make_knowledge_store(),
        )
        await retriever.retrieve("q", max_sources=3, min_confidence=0.0)
        es.search.assert_called_once()
        call_kwargs = es.search.call_args
        assert call_kwargs.kwargs.get("limit") == 3 or call_kwargs[1].get("limit") == 3


class TestRetrieveEntityDiscovery:
    async def test_entities_found_populated(self):
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=_make_embedding_store(entity_rows=[_ENTITY_ROW]),
            knowledge_store=_make_knowledge_store(),
        )
        ctx = await retriever.retrieve("dopamine", max_sources=5, min_confidence=0.0)
        assert ctx.entities_found == ["http://knowledge.local/data/dopamine"]

    async def test_knowledge_triples_from_entity_lookup(self):
        triple = {
            "predicate": "http://knowledge.local/schema/increases",
            "object": "http://knowledge.local/data/dopamine",
            "confidence": 0.88,
            "knowledge_type": "Claim",
            "valid_from": None,
            "valid_until": None,
        }
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=_make_embedding_store(entity_rows=[_ENTITY_ROW]),
            knowledge_store=_make_knowledge_store(triples=[triple]),
        )
        ctx = await retriever.retrieve("dopamine", max_sources=5, min_confidence=0.0)
        assert len(ctx.knowledge_triples) == 1
        assert ctx.knowledge_triples[0]["confidence"] == 0.88


class TestRetrieveConfidenceFilter:
    async def test_filters_below_min_confidence(self):
        low = {
            "predicate": "p",
            "object": "o",
            "confidence": 0.2,
            "knowledge_type": "Claim",
            "valid_from": None,
            "valid_until": None,
        }
        high = {
            "predicate": "p2",
            "object": "o2",
            "confidence": 0.9,
            "knowledge_type": "Fact",
            "valid_from": None,
            "valid_until": None,
        }
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=_make_embedding_store(entity_rows=[_ENTITY_ROW]),
            knowledge_store=_make_knowledge_store(triples=[low, high]),
        )
        ctx = await retriever.retrieve("q", max_sources=5, min_confidence=0.5)
        assert len(ctx.knowledge_triples) == 1
        assert ctx.knowledge_triples[0]["confidence"] == 0.9


class TestRetrieveContradictions:
    async def test_contradictions_detected(self):
        triple = {
            "predicate": "http://ks/increases",
            "object": "http://ks/dopamine",
            "confidence": 0.8,
            "knowledge_type": "Claim",
            "valid_from": None,
            "valid_until": None,
        }
        contra = {"object": "http://ks/serotonin", "confidence": 0.3}
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=_make_embedding_store(entity_rows=[_ENTITY_ROW]),
            knowledge_store=_make_knowledge_store(triples=[triple], contradictions=[contra]),
        )
        ctx = await retriever.retrieve("q", max_sources=5, min_confidence=0.0)
        assert len(ctx.contradictions) == 1
        assert ctx.contradictions[0]["confidence"] == 0.3


class TestRetrieveEmpty:
    async def test_empty_when_no_content_or_entities(self):
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=_make_embedding_store(),
            knowledge_store=_make_knowledge_store(),
        )
        ctx = await retriever.retrieve("unknown topic", max_sources=5, min_confidence=0.0)
        assert ctx.content_results == []
        assert ctx.knowledge_triples == []
        assert ctx.contradictions == []
        assert ctx.entities_found == []


class TestRetrieveTripleSerialization:
    async def test_knowledge_triples_predicate_is_string(self):
        """Predicates in retrieval context must be plain strings, not NamedNode objects."""
        entity_row = {"uri": "http://ks/dopamine", "label": "Dopamine", "similarity": 0.9}
        triple = {
            "predicate": NamedNode("http://ks/increases"),
            "object": Literal("serotonin"),
            "confidence": 0.8,
            "knowledge_type": "Claim",
            "valid_from": None,
            "valid_until": None,
        }
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=_make_embedding_store(entity_rows=[entity_row]),
            knowledge_store=_make_knowledge_store(triples=[triple]),
        )
        ctx = await retriever.retrieve("test", max_sources=5, min_confidence=0.0)
        assert len(ctx.knowledge_triples) == 1
        t = ctx.knowledge_triples[0]
        assert isinstance(t["predicate"], str), f"Expected str, got {type(t['predicate'])}"
        assert isinstance(t["object"], str), f"Expected str, got {type(t['object'])}"
        assert t["predicate"] == "http://ks/increases"
        assert t["object"] == "serotonin"
