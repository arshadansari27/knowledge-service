"""Pydantic models for API requests/responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ContentRequest(BaseModel):
    """Content ingestion request (graph/claims paths removed)."""

    url: str
    title: str | None = None
    summary: str | None = None
    raw_text: str | None = None
    source_type: str | None = None
    tags: list[str] = Field(default_factory=list)
    # Keep knowledge field for backward compat (will be ignored during ingest)
    knowledge: list[Any] = Field(default_factory=list)


class ContentAcceptedResponse(BaseModel):
    """Response after successful content ingest."""

    content_id: str
    job_id: str
    status: str = "accepted"
    chunks_total: int
    chunks_capped_from: int | None = None


class SearchResult(BaseModel):
    """Semantic search result — chunk from the content store."""

    content_id: str
    url: str
    title: str
    summary: str | None
    similarity: float | None  # Cosine similarity; None for BM25-only hits
    rrf_score: float | None = None  # Fused rank when hybrid search is active
    bm25_rank: int | None = None  # 0-based rank from BM25; None for vector-only
    source_type: str
    tags: list[str]
    ingested_at: datetime
    chunk_text: str
    chunk_index: int
    section_header: str | None = None
