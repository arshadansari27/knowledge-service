"""Tests for query-relevance triple ranking.

The 2026-05-31 eval showed pruning to top-15 *by confidence* still left graph-on
net-negative: the most confident triples are not the most relevant to the
question. This ranks candidate triples by cosine similarity of their rendered
text to the query embedding, so the few triples that reach the prompt are the
ones actually about the question.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from knowledge_service.stores.rag import RAGRetriever


def _triple(obj, conf):
    return {
        "subject": "http://knowledge.local/data/dopamine",
        "predicate": "http://knowledge.local/schema/affects",
        "object": obj,
        "confidence": conf,
        "graph": "http://knowledge.local/schema/graph/extracted",
    }


def _retriever_with_embeddings(text_to_vec):
    """Build a retriever whose embed_batch maps rendered triple text -> vector."""
    ec = AsyncMock()
    ec.embed.return_value = [1.0, 0.0, 0.0]

    async def _embed_batch(texts):
        return [text_to_vec(t) for t in texts]

    ec.embed_batch.side_effect = _embed_batch
    return RAGRetriever(
        embedding_client=ec,
        embedding_store=AsyncMock(),
        knowledge_store=MagicMock(),
    )


class TestTripleToText:
    def test_localizes_uris(self):
        t = _triple("http://knowledge.local/data/serotonin", 0.5)
        text = RAGRetriever._triple_to_text(t)
        assert text == "dopamine affects serotonin"

    def test_literal_object_kept_as_is(self):
        t = _triple("a calming effect", 0.5)
        assert RAGRetriever._triple_to_text(t) == "dopamine affects a calming effect"


class TestRelevanceRanking:
    async def test_relevant_triple_beats_more_confident_irrelevant_one(self):
        # query vector points at "alertness"; the alertness triple is LESS
        # confident but MUST win on relevance.
        query = [1.0, 0.0, 0.0]
        relevant = _triple("alertness", 0.30)
        irrelevant = _triple("shoelaces", 0.95)

        def vec(text):
            return [1.0, 0.0, 0.0] if "alertness" in text else [0.0, 1.0, 0.0]

        r = _retriever_with_embeddings(vec)
        ranked = await r._rank_triples_by_relevance([irrelevant, relevant], query, limit=1)
        assert len(ranked) == 1
        assert ranked[0]["object"] == "alertness"

    async def test_caps_to_limit(self):
        query = [1.0, 0.0, 0.0]
        triples = [_triple(f"obj{i}", 0.5) for i in range(5)]
        r = _retriever_with_embeddings(lambda t: [1.0, 0.0, 0.0])
        ranked = await r._rank_triples_by_relevance(triples, query, limit=3)
        assert len(ranked) == 3

    async def test_empty_input(self):
        r = _retriever_with_embeddings(lambda t: [1.0, 0.0, 0.0])
        assert await r._rank_triples_by_relevance([], [1.0, 0.0, 0.0], limit=5) == []

    async def test_falls_back_to_confidence_on_embed_failure(self):
        query = [1.0, 0.0, 0.0]
        lo = _triple("lo", 0.20)
        hi = _triple("hi", 0.90)
        ec = AsyncMock()
        ec.embed_batch.side_effect = RuntimeError("embed backend down")
        r = RAGRetriever(
            embedding_client=ec,
            embedding_store=AsyncMock(),
            knowledge_store=MagicMock(),
        )
        ranked = await r._rank_triples_by_relevance([lo, hi], query, limit=1)
        # graceful degradation: highest confidence kept
        assert ranked[0]["object"] == "hi"
