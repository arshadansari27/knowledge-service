"""POST /api/content endpoint — async ingest content with embedded knowledge."""

from __future__ import annotations

import logging
from enum import StrEnum

import asyncpg.exceptions
import httpx

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from knowledge_service.chunking import chunk_text as split_into_chunks
from knowledge_service.config import settings
from knowledge_service.ingestion.worker import run_ingestion
from knowledge_service.models import (
    ContentAcceptedResponse,
    ContentRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Set by lifespan or tests; used for URL auto-fetch format detection + parsing.
_parser_registry = None


class JobPhase(StrEnum):
    EMBEDDING = "embedding"
    EXTRACTING = "extracting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


_CHUNK_SIZE = settings.chunk_size
_CHUNK_OVERLAP = settings.chunk_overlap
_MAX_CHUNKS = settings.max_chunks


# ---------------------------------------------------------------------------
# Synchronous acceptance phase
# ---------------------------------------------------------------------------


async def _accept_content_request(body: ContentRequest, stores) -> dict:
    """Synchronous phase: validate, upsert metadata, chunk, create job.

    Returns dict with content_id, job_id, chunks_total, chunks_capped_from,
    and chunk_records for the background worker.
    """
    content_store = stores.content
    pg_pool = stores.pg_pool

    # Step 1: Upsert content metadata
    content_id = await content_store.upsert_metadata(
        url=body.url,
        title=body.title,
        summary=body.summary or "",
        raw_text=body.raw_text or "",
        source_type=body.source_type,
        tags=body.tags,
        metadata=body.metadata,
    )

    # Step 1b: URL auto-fetch when no text provided
    if not body.raw_text and not body.summary and _parser_registry and body.url.startswith("http"):
        try:
            async with httpx.AsyncClient(
                timeout=settings.url_fetch_timeout,
                follow_redirects=True,
                headers={"User-Agent": "knowledge-service/1.0"},
            ) as http_client:
                resp = await http_client.get(body.url)
                resp.raise_for_status()

            content_type = resp.headers.get("content-type")
            fmt = _parser_registry.detect_format(
                content_type=content_type, url=body.url, data=resp.content
            )
            parser = _parser_registry.get_parser(fmt)
            if parser:
                parsed = await parser.parse(resp.content, content_type=content_type)
                updates: dict = {"raw_text": parsed.text}
                if parsed.title and not body.title:
                    updates["title"] = parsed.title
                body = body.model_copy(update=updates)
        except Exception as exc:
            logger.warning("URL auto-fetch failed for %s: %s", body.url, exc)
            return {"error": f"Failed to fetch URL: {exc}", "status_code": 422}

    # Step 2: Chunk the text
    text = body.raw_text or body.summary or body.title
    raw_chunks = split_into_chunks(text, chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)

    chunk_records: list[dict] = []
    for i, rc in enumerate(raw_chunks):
        chunk_records.append(
            {
                "chunk_index": i,
                "chunk_text": rc["chunk_text"],
                "char_start": rc["char_start"],
                "char_end": rc["char_end"],
                "section_header": rc.get("section_header"),
            }
        )

    # Step 3: Cap chunks
    chunks_capped_from = None
    if len(chunk_records) > _MAX_CHUNKS:
        chunks_capped_from = len(chunk_records)
        logger.warning(
            "Capping %d chunks to %d for url=%s", len(chunk_records), _MAX_CHUNKS, body.url
        )
        chunk_records = chunk_records[:_MAX_CHUNKS]

    # Step 4+5: Atomically create job if no active one exists
    async with pg_pool.acquire() as conn:
        try:
            job_row = await conn.fetchrow(
                """INSERT INTO ingestion_jobs (content_id, chunks_total, chunks_capped_from)
                   VALUES ($1::uuid, $2, $3)
                   RETURNING id""",
                content_id,
                len(chunk_records),
                chunks_capped_from,
            )
        except asyncpg.exceptions.UniqueViolationError:
            return {"conflict": True, "content_id": content_id}
    job_id = str(job_row["id"])

    return {
        "conflict": False,
        "content_id": content_id,
        "job_id": job_id,
        "chunks_total": len(chunk_records),
        "chunks_capped_from": chunks_capped_from,
        "chunk_records": chunk_records,
    }


# ---------------------------------------------------------------------------
# Background worker wrapper
# ---------------------------------------------------------------------------


async def _run_ingestion_worker(
    job_id: str,
    content_id: str,
    body: ContentRequest,
    chunk_records: list[dict],
    app_state,
) -> None:
    """Background worker: delegates to the new ingestion pipeline."""
    stores = app_state.stores
    embedding_client = app_state.embedding_client
    extraction_client = getattr(app_state, "extraction_client", None)
    entity_store = stores.entities
    engine = getattr(app_state, "inference_engine", None)
    nlp = getattr(app_state, "nlp", None)

    await run_ingestion(
        job_id=job_id,
        content_id=content_id,
        chunk_records=chunk_records,
        raw_text=body.raw_text,
        knowledge=list(body.knowledge) if body.knowledge else None,
        title=body.title,
        source_url=body.url,
        source_type=body.source_type or "",
        stores=stores,
        embedding_client=embedding_client,
        extraction_client=extraction_client,
        entity_store=entity_store,
        engine=engine,
        nlp=nlp,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/content")
async def post_content(request: Request, background_tasks: BackgroundTasks):
    """Ingest content asynchronously.

    Accepts a single ContentRequest or a list for batch processing.
    Returns 202 Accepted with job info. Processing happens in background.
    """
    raw = await request.json()
    stores = request.app.state.stores

    try:
        if isinstance(raw, list):
            results = []
            for item_raw in raw:
                try:
                    body = ContentRequest(**item_raw)
                except ValidationError as exc:
                    results.append({"error": exc.errors(), "status_code": 422})
                    continue
                result = await _accept_content_request(body, stores)
                if result.get("status_code") == 422:
                    results.append(result)
                    continue
                if result.get("conflict"):
                    results.append(
                        {
                            "error": "Active job exists for this content",
                            "content_id": result["content_id"],
                            "status_code": 409,
                        }
                    )
                    continue
                background_tasks.add_task(
                    _run_ingestion_worker,
                    result["job_id"],
                    result["content_id"],
                    body,
                    result["chunk_records"],
                    request.app.state,
                )
                results.append(
                    ContentAcceptedResponse(
                        content_id=result["content_id"],
                        job_id=result["job_id"],
                        chunks_total=result["chunks_total"],
                        chunks_capped_from=result["chunks_capped_from"],
                    ).model_dump()
                )
            return JSONResponse(results, status_code=202)

        body = ContentRequest(**raw)
        result = await _accept_content_request(body, stores)
        if result.get("status_code") == 422:
            return JSONResponse(status_code=422, content={"detail": result["error"]})
        if result.get("conflict"):
            return JSONResponse(
                status_code=409,
                content={"detail": "Active ingestion job exists for this content"},
            )
        background_tasks.add_task(
            _run_ingestion_worker,
            result["job_id"],
            result["content_id"],
            body,
            result["chunk_records"],
            request.app.state,
        )
        return JSONResponse(
            ContentAcceptedResponse(
                content_id=result["content_id"],
                job_id=result["job_id"],
                chunks_total=result["chunks_total"],
                chunks_capped_from=result["chunks_capped_from"],
            ).model_dump(),
            status_code=202,
        )
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})


@router.get("/content/{content_id}/status")
async def get_content_status(content_id: str, request: Request):
    """Return the latest ingestion job status for a content item."""
    stores = request.app.state.stores
    pg_pool = stores.pg_pool
    async with pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, content_id, status, chunks_total, chunks_embedded,
                      chunks_extracted, chunks_failed, triples_created,
                      entities_resolved, error, created_at, updated_at
               FROM ingestion_jobs
               WHERE content_id = $1::uuid
               ORDER BY created_at DESC LIMIT 1""",
            content_id,
        )
    if row is None:
        return JSONResponse(status_code=404, content={"detail": "No job found"})
    return {
        "content_id": str(row["content_id"]),
        "job_id": str(row["id"]),
        "status": row["status"],
        "chunks_total": row["chunks_total"],
        "chunks_embedded": row["chunks_embedded"],
        "chunks_extracted": row["chunks_extracted"],
        "chunks_failed": row["chunks_failed"],
        "triples_created": row["triples_created"],
        "entities_resolved": row["entities_resolved"],
        "error": row["error"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }
