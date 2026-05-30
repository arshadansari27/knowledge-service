"""Tests for _retrieved_ids dedup (prevents nDCG > 1 from chunk-level duplicates)."""

from __future__ import annotations

from knowledge_service.eval.runner import _retrieved_ids
from knowledge_service.stores.rag import RetrievalContext


def test_dedups_content_id_preserving_order():
    ctx = RetrievalContext(
        content_results=[
            {"content_id": "doc-1", "id": "chunk-a"},
            {"content_id": "doc-1", "id": "chunk-b"},  # same doc, second chunk
            {"content_id": "doc-2", "id": "chunk-c"},
        ]
    )
    assert _retrieved_ids(ctx) == ["doc-1", "doc-2"]


def test_falls_back_to_chunk_id_when_no_content_id():
    ctx = RetrievalContext(content_results=[{"id": "chunk-9"}])
    assert _retrieved_ids(ctx) == ["chunk-9"]


def test_skips_rows_without_any_id():
    ctx = RetrievalContext(content_results=[{"chunk_text": "x"}, {"content_id": "doc-1"}])
    assert _retrieved_ids(ctx) == ["doc-1"]
