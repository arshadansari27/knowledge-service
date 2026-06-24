"""Tests for the retrieval_mode toggle on RAGRetriever and /api/ask."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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


def _make_knowledge_store():
    mock = MagicMock()
    mock.get_triples.return_value = []
    mock.find_contradictions.return_value = []
    return mock


_CONTENT_ROW = {
    "id": "chunk-1",
    "content_id": "content-1",
    "chunk_text": "text",
    "url": "https://example.com/a",
    "title": "A",
    "source_type": "article",
    "similarity": 0.9,
}


class TestChunksOnlyMode:
    async def test_chunks_only_returns_content_only(self):
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
            knowledge_store=_make_knowledge_store(),
        )
        ctx = await retriever.retrieve(
            "q", max_sources=5, min_confidence=0.0, retrieval_mode="chunks_only"
        )
        assert isinstance(ctx, RetrievalContext)
        assert len(ctx.content_results) == 1
        assert ctx.knowledge_triples == []
        assert ctx.contradictions == []

    async def test_chunks_only_skips_graph_calls(self):
        ks = _make_knowledge_store()
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
            knowledge_store=ks,
        )
        await retriever.retrieve(
            "q", max_sources=5, min_confidence=0.0, retrieval_mode="chunks_only"
        )
        es.search_entities.assert_not_called()
        es.search_predicates.assert_not_called()
        ks.get_triples.assert_not_called()
        ks.find_contradictions.assert_not_called()

    async def test_full_mode_is_default(self):
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
            knowledge_store=_make_knowledge_store(),
        )
        await retriever.retrieve("q", max_sources=5, min_confidence=0.0)
        es.search_entities.assert_called()
