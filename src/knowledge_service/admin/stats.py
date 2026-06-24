"""Admin stats API endpoints (chunk-only — no knowledge graph)."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/stats/counts")
async def get_counts(request: Request) -> dict:
    """Corpus counts. triples/entities are always 0 (no knowledge graph)."""
    pg_pool = request.app.state.pg_pool
    async with pg_pool.acquire() as conn:
        content_count = await conn.fetchval("SELECT COUNT(*) FROM content")
    return {"triples": 0, "entities": 0, "content": content_count}


@router.get("/stats/content-items")
async def get_content_items(
    request: Request,
    source_type: str | None = Query(None, description="Filter by source_type"),
    limit: int = Query(200, ge=1, le=2000),
) -> list[dict]:
    pg_pool = request.app.state.pg_pool

    if source_type is None:
        sql = """
            SELECT id, url, title, source_type, tags, ingested_at
            FROM content_metadata
            ORDER BY ingested_at DESC
            LIMIT $1
        """
        params = (limit,)
    else:
        sql = """
            SELECT id, url, title, source_type, tags, ingested_at
            FROM content_metadata
            WHERE source_type = $1
            ORDER BY ingested_at DESC
            LIMIT $2
        """
        params = (source_type, limit)
    async with pg_pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    return [
        {
            "id": str(row["id"]),
            "url": row["url"],
            "title": row["title"],
            "source_type": row["source_type"],
            "tags": row["tags"],
            "ingested_at": row["ingested_at"].isoformat() if row["ingested_at"] else None,
        }
        for row in rows
    ]
