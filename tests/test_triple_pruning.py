"""Tests for triple pruning: cap + rank the triples passed to the RAG prompt.

Root cause from the 2026-05-31 eval: graph-on mode flooded the prompt with ~97
triples (max 239), which tanked answer faithfulness. The retriever must cap the
knowledge_triples to a configurable maximum, keeping the highest-confidence ones.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from knowledge_service.stores.rag import RAGRetriever, RetrievalContext


def _make_embedding_client():
    mock = AsyncMock()
    mock.embed.return_value = [0.1] * 768
    mock.embed_batch.return_value = [[0.1] * 768]
    return mock


def _make_embedding_store(content_rows=None, entity_rows=None, predicate_rows=None):
    mock = AsyncMock()
    mock.search.return_value = content_rows or []
    mock.search_entities.return_value = entity_rows or []
    mock.search_predicates.return_value = predicate_rows or []
    return mock


def _triple(obj, conf):
    return {
        "subject": "http://knowledge.local/data/dopamine",
        "predicate": "http://knowledge.local/schema/affects",
        "object": obj,
        "confidence": conf,
        "graph": "http://knowledge.local/schema/graph/extracted",
        "knowledge_type": "Claim",
    }


_FIVE_TRIPLES = [
    _triple("a", 0.90),
    _triple("b", 0.50),
    _triple("c", 0.70),
    _triple("d", 0.30),
    _triple("e", 0.95),
]


def _make_knowledge_store(subject_triples):
    mock = MagicMock()

    def _get_triples(subject=None, predicate=None, object_=None, graphs=None):
        if subject is not None:
            return list(subject_triples)
        return []

    mock.get_triples.side_effect = _get_triples
    mock.find_contradictions.return_value = []
    return mock


_ENTITY_ROW = {"uri": "http://knowledge.local/data/dopamine", "similarity": 0.9}


class TestTriplePruning:
    async def test_caps_to_max_triples(self):
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=_make_embedding_store(entity_rows=[_ENTITY_ROW]),
            knowledge_store=_make_knowledge_store(_FIVE_TRIPLES),
            max_triples=2,
        )
        ctx = await retriever.retrieve("q", max_sources=5, min_confidence=0.0)
        assert isinstance(ctx, RetrievalContext)
        assert len(ctx.knowledge_triples) == 2

    async def test_keeps_highest_confidence(self):
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=_make_embedding_store(entity_rows=[_ENTITY_ROW]),
            knowledge_store=_make_knowledge_store(_FIVE_TRIPLES),
            max_triples=2,
        )
        ctx = await retriever.retrieve("q", max_sources=5, min_confidence=0.0)
        objs = {t["object"] for t in ctx.knowledge_triples}
        assert objs == {"e", "a"}  # confidences 0.95 and 0.90

    async def test_default_max_is_applied(self):
        # 5 triples, default cap is 15 -> all 5 pass through unpruned.
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=_make_embedding_store(entity_rows=[_ENTITY_ROW]),
            knowledge_store=_make_knowledge_store(_FIVE_TRIPLES),
        )
        ctx = await retriever.retrieve("q", max_sources=5, min_confidence=0.0)
        assert len(ctx.knowledge_triples) == 5

    def test_rank_and_cap_is_pure(self):
        capped = RAGRetriever._rank_and_cap_triples(_FIVE_TRIPLES, 3)
        assert [t["object"] for t in capped] == ["e", "a", "c"]  # 0.95, 0.90, 0.70

    def test_rank_and_cap_handles_missing_confidence(self):
        triples = [{"object": "x"}, {"object": "y", "confidence": 0.8}]
        capped = RAGRetriever._rank_and_cap_triples(triples, 5)
        assert capped[0]["object"] == "y"  # 0.8 outranks missing (treated as 0)
        assert len(capped) == 2
