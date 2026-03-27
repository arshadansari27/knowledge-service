"""Unit tests for RAGRetriever — retrieval pipeline across stores."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from pyoxigraph import Literal, NamedNode

from knowledge_service.clients.classifier import QueryIntent
from knowledge_service.stores.rag import RAGRetriever, RetrievalContext


def _make_embedding_client():
    mock = AsyncMock()
    mock.embed.return_value = [0.1] * 768
    return mock


def _make_embedding_store(content_rows=None, entity_rows=None, predicate_rows=None):
    mock = AsyncMock()
    mock.search.return_value = content_rows or []
    mock.search_entities.return_value = entity_rows or []
    mock.search_predicates.return_value = predicate_rows or []
    return mock


def _make_knowledge_store(triples=None, contradictions=None):
    mock = MagicMock()
    mock.get_triples_by_subject.return_value = triples or []
    mock.get_triples_by_object.return_value = []
    mock.find_contradictions.return_value = contradictions or []
    mock.find_connecting_triples.return_value = []
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


class TestIntentDispatch:
    async def test_semantic_intent_uses_full_search(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        ks = _make_knowledge_store()
        retriever = RAGRetriever(ec, es, ks)
        intent = QueryIntent(intent="semantic", entities=[])
        await retriever.retrieve("find articles about stress", intent=intent)
        es.search.assert_called_once()

    async def test_entity_intent_resolves_entities(self):
        ec = _make_embedding_client()
        ec.embed_batch.return_value = [[0.1] * 768]
        es = _make_embedding_store()
        es.search_entities.return_value = [
            {"uri": "http://knowledge.local/data/dopamine", "similarity": 0.9}
        ]
        ks = _make_knowledge_store()
        retriever = RAGRetriever(ec, es, ks)
        intent = QueryIntent(intent="entity", entities=["dopamine"])
        context = await retriever.retrieve("what is dopamine?", intent=intent)
        assert len(context.entities_found) >= 1

    async def test_none_intent_defaults_to_semantic(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        ks = _make_knowledge_store()
        retriever = RAGRetriever(ec, es, ks)
        await retriever.retrieve("some question")
        es.search.assert_called_once()

    async def test_graph_intent_uses_traverser(self):
        """Graph intent should use multi-hop traversal."""
        ec = _make_embedding_client()
        ec.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]
        es = _make_embedding_store()
        es.search_entities.return_value = [
            {"uri": "http://knowledge.local/data/cortisol", "similarity": 0.9}
        ]
        ks = _make_knowledge_store()
        ks.get_triples.return_value = []
        retriever = RAGRetriever(ec, es, ks)
        intent = QueryIntent(intent="graph", entities=["cortisol", "inflammation"])
        await retriever.retrieve("how is cortisol connected to inflammation?", intent=intent)
        # GraphTraverser calls get_triples(subject=...) internally
        ks.get_triples.assert_called()

    async def test_entity_below_threshold_skipped(self):
        """Entity with similarity < 0.80 is skipped, falls back to semantic."""
        ec = _make_embedding_client()
        ec.embed_batch.return_value = [[0.1] * 768]
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        es.search_entities.return_value = [
            {"uri": "http://knowledge.local/data/weak_match", "similarity": 0.5}
        ]
        ks = _make_knowledge_store()
        retriever = RAGRetriever(ec, es, ks)
        intent = QueryIntent(intent="entity", entities=["weak_match"])
        await retriever.retrieve("what is weak_match?", intent=intent)
        # Low similarity entity skipped → falls back to semantic
        es.search.assert_called_once()

    async def test_graph_intent_returns_traversal_metadata(self):
        """Graph intent should populate traversal_depth and inferred_triples on context."""
        ec = _make_embedding_client()
        ec.embed_batch.return_value = [[0.1] * 768]
        es = _make_embedding_store()
        es.search_entities.return_value = [
            {"uri": "http://knowledge.local/data/dopamine", "similarity": 0.9}
        ]
        ks = _make_knowledge_store()
        retriever = RAGRetriever(ec, es, ks)
        intent = QueryIntent(intent="graph", entities=["dopamine"])
        context = await retriever.retrieve("what affects dopamine?", intent=intent)
        assert context.traversal_depth is not None
        # inferred_triples is None — ProbLog inference was removed
        assert context.inferred_triples is None


class TestGlobalIntent:
    async def test_global_intent_uses_community_summaries(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        ks = _make_knowledge_store()
        # Mock community store
        community_store = AsyncMock()
        community_store.get_all.return_value = [
            {
                "id": "c1",
                "level": 1,
                "label": "Health",
                "summary": "Health and biohacking topics",
                "member_entities": ["http://e/a"],
                "member_count": 3,
                "built_at": "2026-01-01",
            },
        ]
        retriever = RAGRetriever(ec, es, ks, community_store=community_store)
        intent = QueryIntent(intent="global", entities=[])
        context = await retriever.retrieve("what are the main themes?", intent=intent)
        community_store.get_all.assert_called_once()
        assert len(context.knowledge_triples) >= 1

    async def test_global_falls_back_to_semantic_without_communities(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        ks = _make_knowledge_store()
        community_store = AsyncMock()
        community_store.get_all.return_value = []
        retriever = RAGRetriever(ec, es, ks, community_store=community_store)
        intent = QueryIntent(intent="global", entities=[])
        await retriever.retrieve("what are the main themes?", intent=intent)
        # Falls back to semantic -- search called
        es.search.assert_called_once()

    async def test_global_falls_back_without_community_store(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        ks = _make_knowledge_store()
        retriever = RAGRetriever(ec, es, ks, community_store=None)
        intent = QueryIntent(intent="global", entities=[])
        await retriever.retrieve("what are the main themes?", intent=intent)
        # Falls back to semantic -- search called
        es.search.assert_called_once()

    async def test_global_includes_level0_fallback_when_no_keyword_match(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        ks = _make_knowledge_store()
        community_store = AsyncMock()
        community_store.get_all.return_value = [
            {
                "id": "c1",
                "level": 0,
                "label": "Neuroscience",
                "summary": "Brain chemistry and neurotransmitters",
                "member_entities": ["http://e/a"],
                "member_count": 5,
                "built_at": "2026-01-01",
            },
            {
                "id": "c2",
                "level": 1,
                "label": "Science",
                "summary": "Scientific research topics",
                "member_entities": ["http://e/b"],
                "member_count": 10,
                "built_at": "2026-01-01",
            },
        ]
        retriever = RAGRetriever(ec, es, ks, community_store=community_store)
        intent = QueryIntent(intent="global", entities=[])
        # Question words won't match level-0 summary, so fallback top-5 kicks in
        context = await retriever.retrieve("overview?", intent=intent)
        # Should include both: level-1 always included, level-0 via fallback
        assert len(context.knowledge_triples) == 2


_PREDICATE_ROW = {
    "uri": "http://knowledge.local/schema/sentry_issue",
    "label": "sentry_issue",
    "similarity": 0.85,
}

_PREDICATE_TRIPLE = {
    "graph": "http://knowledge.local/graph/extracted",
    "subject": NamedNode("http://knowledge.local/data/connection_error"),
    "predicate": NamedNode("http://knowledge.local/schema/sentry_issue"),
    "object": Literal("error"),
    "confidence": 0.9,
    "knowledge_type": "Claim",
    "valid_from": None,
    "valid_until": None,
}


class TestPredicateLookup:
    async def test_returns_triples_for_matching_predicate(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(predicate_rows=[_PREDICATE_ROW])
        ks = _make_knowledge_store()
        ks.get_triples_by_predicate.return_value = [_PREDICATE_TRIPLE]
        retriever = RAGRetriever(ec, es, ks)
        embedding = [0.1] * 768
        triples = await retriever._lookup_triples_by_predicate(embedding)
        assert len(triples) == 1
        assert triples[0]["subject"] == "http://knowledge.local/data/connection_error"
        assert triples[0]["predicate"] == "http://knowledge.local/schema/sentry_issue"
        assert triples[0]["object"] == "error"
        assert triples[0]["confidence"] == 0.9
        assert triples[0]["trust_tier"] == "extracted"

    async def test_filters_below_threshold(self):
        low_sim = {
            "uri": "http://knowledge.local/schema/weak",
            "label": "weak",
            "similarity": 0.5,
        }
        ec = _make_embedding_client()
        es = _make_embedding_store(predicate_rows=[low_sim])
        ks = _make_knowledge_store()
        retriever = RAGRetriever(ec, es, ks)
        embedding = [0.1] * 768
        triples = await retriever._lookup_triples_by_predicate(embedding)
        assert triples == []
        ks.get_triples_by_predicate.assert_not_called()

    async def test_empty_predicate_embeddings(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(predicate_rows=[])
        ks = _make_knowledge_store()
        retriever = RAGRetriever(ec, es, ks)
        embedding = [0.1] * 768
        triples = await retriever._lookup_triples_by_predicate(embedding)
        assert triples == []

    async def test_limits_to_top_n_by_confidence(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(predicate_rows=[_PREDICATE_ROW])
        ks = _make_knowledge_store()
        # Return 15 triples, should be limited to 10
        many_triples = []
        for i in range(15):
            t = dict(_PREDICATE_TRIPLE)
            t["confidence"] = round(0.5 + i * 0.03, 2)
            many_triples.append(t)
        ks.get_triples_by_predicate.return_value = many_triples
        retriever = RAGRetriever(ec, es, ks)
        embedding = [0.1] * 768
        triples = await retriever._lookup_triples_by_predicate(embedding)
        assert len(triples) == 10
        # Should be sorted by confidence descending
        confs = [t["confidence"] for t in triples]
        assert confs == sorted(confs, reverse=True)


class TestPredicateIntegration:
    async def test_semantic_includes_predicate_triples(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(predicate_rows=[_PREDICATE_ROW])
        ks = _make_knowledge_store()
        ks.get_triples_by_predicate.return_value = [_PREDICATE_TRIPLE]
        retriever = RAGRetriever(ec, es, ks)
        ctx = await retriever.retrieve("any sentry issues?", max_sources=5, min_confidence=0.0)
        assert len(ctx.knowledge_triples) == 1
        assert ctx.knowledge_triples[0]["predicate"] == "http://knowledge.local/schema/sentry_issue"

    async def test_entity_intent_includes_predicate_triples(self):
        ec = _make_embedding_client()
        ec.embed_batch.return_value = [[0.1] * 768]
        es = _make_embedding_store(predicate_rows=[_PREDICATE_ROW])
        es.search_entities.return_value = [
            {"uri": "http://knowledge.local/data/sentry", "similarity": 0.5}
        ]
        ks = _make_knowledge_store()
        ks.get_triples_by_predicate.return_value = [_PREDICATE_TRIPLE]
        retriever = RAGRetriever(ec, es, ks)
        intent = QueryIntent(intent="entity", entities=["sentry"])
        ctx = await retriever.retrieve("sentry issues?", intent=intent)
        # Entity resolution failed (similarity 0.5 < 0.80), fell back to semantic
        # but predicate lookup should still find sentry_issue triples
        assert len(ctx.knowledge_triples) == 1

    async def test_dedup_subject_and_predicate_triples(self):
        """Same triple from subject lookup and predicate lookup should not duplicate."""
        ec = _make_embedding_client()
        entity_row = {
            "uri": "http://knowledge.local/data/connection_error",
            "label": "connection_error",
            "similarity": 0.9,
        }
        subject_triple = {
            "predicate": NamedNode("http://knowledge.local/schema/sentry_issue"),
            "object": Literal("error"),
            "confidence": 0.9,
            "knowledge_type": "Claim",
            "valid_from": None,
            "valid_until": None,
            "graph": "http://knowledge.local/graph/extracted",
        }
        es = _make_embedding_store(entity_rows=[entity_row], predicate_rows=[_PREDICATE_ROW])
        ks = _make_knowledge_store(triples=[subject_triple])
        ks.get_triples_by_predicate.return_value = [_PREDICATE_TRIPLE]
        retriever = RAGRetriever(ec, es, ks)
        ctx = await retriever.retrieve("sentry", max_sources=5, min_confidence=0.0)
        # Should deduplicate — same (subject, predicate, object) from both lookups
        assert len(ctx.knowledge_triples) == 1

    async def test_entity_resolved_includes_predicate_triples(self):
        """When entity resolution succeeds, predicate triples are still included."""
        ec = _make_embedding_client()
        ec.embed_batch.return_value = [[0.1] * 768]
        entity_row = {
            "uri": "http://knowledge.local/data/my_service",
            "label": "my_service",
            "similarity": 0.9,
        }
        subject_triple = {
            "predicate": NamedNode("http://knowledge.local/schema/runs_on"),
            "object": Literal("kubernetes"),
            "confidence": 0.8,
            "knowledge_type": "Fact",
            "valid_from": None,
            "valid_until": None,
            "graph": "http://knowledge.local/graph/extracted",
        }
        es = _make_embedding_store(predicate_rows=[_PREDICATE_ROW])
        es.search_entities.return_value = [entity_row]
        ks = _make_knowledge_store(triples=[subject_triple])
        ks.get_triples_by_predicate.return_value = [_PREDICATE_TRIPLE]
        retriever = RAGRetriever(ec, es, ks)
        intent = QueryIntent(intent="entity", entities=["my_service"])
        ctx = await retriever.retrieve("my_service sentry issues?", intent=intent)
        # Both subject triple and predicate triple should be included
        assert len(ctx.knowledge_triples) == 2
        predicates = {t["predicate"] for t in ctx.knowledge_triples}
        assert "http://knowledge.local/schema/runs_on" in predicates
        assert "http://knowledge.local/schema/sentry_issue" in predicates
