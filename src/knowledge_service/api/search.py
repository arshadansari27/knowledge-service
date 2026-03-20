"""GET /api/search endpoint — semantic similarity search over ingested content."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from knowledge_service.models import SearchResult
from knowledge_service.stores.embedding import EmbeddingStore

router = APIRouter()


@router.get("/search", response_model=list[SearchResult])
async def get_search(
    request: Request,
    q: str = Query(..., description="Search query text"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results to return"),
    source_type: str | None = Query(None, description="Filter by source type"),
    tags: list[str] | None = Query(None, description="Filter by tags (all must match)"),
) -> list[SearchResult]:
    """Search ingested content by semantic similarity.

    Queries chunk embeddings in the content table, joined with content_metadata
    for filtering and metadata. Every result is a chunk.
    """
    embedding_client = request.app.state.embedding_client
    embedding_store = getattr(request.app.state, "embedding_store", None)

    if embedding_store is None:
        pg_pool = request.app.state.pg_pool
        embedding_store = EmbeddingStore(pg_pool)

    embedding = await embedding_client.embed(q)

    rows = await embedding_store.search(
        query_embedding=embedding,
        limit=limit,
        source_type=source_type,
        tags=tags,
    )

    return [
        SearchResult(
            content_id=str(row["content_id"]),
            url=row["url"],
            title=row["title"],
            summary=row.get("summary"),
            similarity=float(row["similarity"]),
            source_type=row["source_type"],
            tags=list(row["tags"]) if row["tags"] else [],
            ingested_at=row["ingested_at"],
            chunk_text=row["chunk_text"],
            chunk_index=row["chunk_index"],
        )
        for row in rows
    ]
