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

    Processing pipeline:
    1. Embed the query text via Ollama.
    2. Query EmbeddingStore for content rows ranked by cosine similarity.
    3. Return a list of SearchResult with similarity scores.
    """
    embedding_client = request.app.state.embedding_client
    pg_pool = request.app.state.pg_pool

    embedding = await embedding_client.embed(q)

    embedding_store = EmbeddingStore(pg_pool)
    rows = await embedding_store.search(
        query_embedding=embedding,
        limit=limit,
        source_type=source_type,
        tags=tags,
    )

    # ------------------------------------------------------------------
    # Step 3: Map to SearchResult
    # ------------------------------------------------------------------
    return [
        SearchResult(
            content_id=str(row["id"]),
            url=row["url"],
            title=row["title"],
            summary=row.get("summary"),
            similarity=float(row["similarity"]),
            source_type=row["source_type"],
            tags=list(row["tags"]) if row["tags"] else [],
            ingested_at=row["ingested_at"],
        )
        for row in rows
    ]
