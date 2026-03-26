"""Admin API endpoint for ingestion jobs."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/jobs")
async def list_jobs(
    request: Request,
    limit: int = Query(default=50, le=200),
    status: str | None = Query(default=None),
):
    """List ingestion jobs in descending order."""
    pg_pool = request.app.state.pg_pool

    conditions: list[str] = []
    params: list = []

    if status:
        params.append(status)
        conditions.append(f"j.status = ${len(params)}")

    params.append(limit)
    limit_placeholder = f"${len(params)}"

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    sql = f"""
        SELECT j.id, j.content_id, j.status, j.chunks_total,
               j.chunks_embedded, j.chunks_extracted, j.chunks_failed,
               j.triples_created, j.entities_resolved, j.chunks_capped_from,
               j.error, j.created_at, j.updated_at,
               m.url, m.title
        FROM ingestion_jobs j
        JOIN content_metadata m ON j.content_id = m.id
        {where}
        ORDER BY j.created_at DESC
        LIMIT {limit_placeholder}
    """

    async with pg_pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    return [
        {
            "id": str(r["id"]),
            "content_id": str(r["content_id"]),
            "status": r["status"],
            "chunks_total": r["chunks_total"],
            "chunks_embedded": r["chunks_embedded"],
            "chunks_extracted": r["chunks_extracted"],
            "chunks_failed": r["chunks_failed"],
            "triples_created": r["triples_created"],
            "entities_resolved": r["entities_resolved"],
            "chunks_capped_from": r["chunks_capped_from"],
            "error": r["error"],
            "url": r["url"],
            "title": r["title"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
        }
        for r in rows
    ]
