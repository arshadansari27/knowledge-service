"""GET /api/search endpoint — semantic similarity search over ingested content."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from knowledge_service.models import SearchResult

router = APIRouter()


@router.get("/search", response_model=list[SearchResult])
async def get_search(
    request: Request,
    q: str = Query(..., description="Search query text"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results to return"),
    source_type: str | None = Query(None, description="Filter by source type"),
    tags: list[str] | None = Query(None, description="Filter by tags (all must match)"),
    content_id: str | None = Query(None, description="Scope search to a specific content item"),
) -> list[SearchResult]:
    """Search ingested content by semantic similarity.

    Queries chunk embeddings in the content table, joined with content_metadata
    for filtering and metadata. Every result is a chunk.
    """
    embedding_client = request.app.state.embedding_client
    stores = request.app.state.stores
    content_store = stores.content

    embedding = await embedding_client.embed(q)

    rows = await content_store.search(
        query_embedding=embedding,
        limit=limit,
        source_type=source_type,
        tags=tags,
        query_text=q,
        content_id=content_id,
    )

    results: list[SearchResult] = []
    for row in rows:
        # When hybrid (vector + BM25 + RRF) is active, ``similarity`` may be
        # None for BM25-only hits and ``rrf_score`` carries the fused rank.
        # When only vector search ran, both fields equal the cosine similarity.
        sim = row.get("similarity")
        rrf = row.get("rrf_score")
        if rrf is None and sim is not None:
            rrf = float(sim)
        results.append(
            SearchResult(
                content_id=str(row["content_id"]),
                url=row["url"],
                title=row["title"],
                summary=row.get("summary"),
                similarity=float(sim) if sim is not None else None,
                rrf_score=float(rrf) if rrf is not None else None,
                bm25_rank=row.get("bm25_rank"),
                source_type=row["source_type"],
                tags=list(row["tags"]) if row["tags"] else [],
                ingested_at=row["ingested_at"],
                chunk_text=row["chunk_text"],
                chunk_index=row["chunk_index"],
                section_header=row.get("section_header"),
            )
        )
    return results
