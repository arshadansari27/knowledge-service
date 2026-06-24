"""Tests for the chunk-only RAGRetriever (graph layer removed)."""

from __future__ import annotations

from unittest.mock import AsyncMock

from knowledge_service.stores.rag import RAGRetriever, RetrievalContext


def _make_embedding_client():
    mock = AsyncMock()
    mock.embed.return_value = [0.1] * 768
    return mock


def _make_embedding_store(content_rows=None):
    mock = AsyncMock()
    mock.search.return_value = content_rows or []
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


class TestChunkRetrieval:
    async def test_returns_content_results(self):
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
        )
        ctx = await retriever.retrieve("q", max_sources=5, min_confidence=0.0)
        assert isinstance(ctx, RetrievalContext)
        assert len(ctx.content_results) == 1

    async def test_retrieve_ignores_legacy_mode_kwargs(self):
        # retrieve() still accepts intent/retrieval_mode for backward compat.
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
        )
        ctx = await retriever.retrieve(
            "q", max_sources=5, min_confidence=0.0, intent=None, retrieval_mode="chunks_only"
        )
        assert len(ctx.content_results) == 1
