"""POST /api/content endpoint — async ingest content with embedded knowledge."""

from __future__ import annotations

import json
import logging

import asyncpg.exceptions

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from knowledge_service._utils import _is_uri, is_object_entity
from knowledge_service.api._ingest import apply_uri_fallback, process_triple
from knowledge_service.chunking import chunk_text as split_into_chunks
from knowledge_service.config import settings
from knowledge_service.models import (
    ContentAcceptedResponse,
    ContentRequest,
    expand_to_triples,
)
from knowledge_service.stores.provenance import ProvenanceStore

router = APIRouter()
logger = logging.getLogger(__name__)

_CHUNK_SIZE = 4000
_CHUNK_OVERLAP = 200
_MAX_CHUNKS = 50
_EMBED_BATCH_SIZE = 20


async def _resolve_labels(
    item, entity_resolver, job_id: str | None = None, pg_pool=None
) -> tuple[int, object]:
    """Resolve entity labels in a knowledge item via embedding similarity.

    Returns (count_resolved, updated_item).
    """
    resolved = 0
    kt = item.knowledge_type.value

    if kt in ("Claim", "Fact", "Relationship"):
        if not _is_uri(item.subject):
            item.subject = await entity_resolver.resolve(
                item.subject, job_id=job_id, pg_pool=pg_pool
            )
            resolved += 1
        if not _is_uri(item.predicate):
            item.predicate = await entity_resolver.resolve_predicate(item.predicate)
            resolved += 1
        if not _is_uri(item.object) and is_object_entity(item):
            item.object = await entity_resolver.resolve(item.object, job_id=job_id, pg_pool=pg_pool)
            resolved += 1
    elif kt == "TemporalState":
        if not _is_uri(item.subject):
            item.subject = await entity_resolver.resolve(
                item.subject, job_id=job_id, pg_pool=pg_pool
            )
            resolved += 1
        if not _is_uri(item.property):
            item.property = await entity_resolver.resolve_predicate(item.property)
            resolved += 1
    elif kt == "Event":
        if not _is_uri(item.subject):
            item.subject = await entity_resolver.resolve(
                item.subject, job_id=job_id, pg_pool=pg_pool
            )
            resolved += 1

    return resolved, item


def _dedup_extracted_items(
    knowledge: list, chunk_ids: list[str | None]
) -> tuple[list, list[str | None]]:
    """Deduplicate knowledge items extracted from overlapping chunks.

    For triple-shaped items (Claim/Fact/Relationship), key on (subject, predicate, object).
    For TemporalState, key on (subject, property, value).
    For Entity, key on uri. For Event, key on (subject, occurred_at).
    For Conclusion, key on concludes text.
    Keeps the item with the highest confidence when duplicates are found.
    """
    seen: dict[tuple, int] = {}  # dedup_key -> index in deduped list
    deduped: list = []
    deduped_cids: list[str | None] = []

    for item, cid in zip(knowledge, chunk_ids):
        kt = item.knowledge_type.value
        if kt in ("Claim", "Fact", "Relationship"):
            key = (kt, item.subject.lower(), item.predicate.lower(), item.object.lower())
        elif kt == "TemporalState":
            key = (kt, item.subject.lower(), item.property.lower(), str(item.value).lower())
        elif kt == "Entity":
            key = (kt, item.uri.lower())
        elif kt == "Event":
            key = (kt, item.subject.lower(), str(item.occurred_at))
        elif kt == "Conclusion":
            key = (kt, item.concludes.lower())
        else:
            key = (kt, id(item))  # no dedup for unknown types

        if key in seen:
            idx = seen[key]
            if item.confidence > deduped[idx].confidence:
                deduped[idx] = item
                deduped_cids[idx] = cid
        else:
            seen[key] = len(deduped)
            deduped.append(item)
            deduped_cids.append(cid)

    return deduped, deduped_cids


# ---------------------------------------------------------------------------
# Synchronous acceptance phase
# ---------------------------------------------------------------------------


async def _accept_content_request(body: ContentRequest, pg_pool, embedding_store) -> dict:
    """Synchronous phase: validate, upsert metadata, chunk, create job.

    Returns dict with content_id, job_id, chunks_total, chunks_capped_from,
    and chunk_records for the background worker.
    """
    # Step 1: Upsert content metadata
    content_id = await embedding_store.insert_content_metadata(
        url=body.url,
        title=body.title,
        summary=body.summary or "",
        raw_text=body.raw_text or "",
        source_type=body.source_type,
        tags=body.tags,
        metadata=body.metadata,
    )

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
# Background worker
# ---------------------------------------------------------------------------


async def _run_ingestion_worker(
    job_id: str,
    content_id: str,
    body: ContentRequest,
    chunk_records: list[dict],
    app_state,
) -> None:
    """Background worker: embed, extract, process triples, update job progress."""
    pg_pool = app_state.pg_pool
    embedding_client = app_state.embedding_client
    extraction_client = app_state.extraction_client
    knowledge_store = app_state.knowledge_store
    reasoning_engine = app_state.reasoning_engine
    embedding_store = getattr(app_state, "embedding_store", None)
    entity_resolver = getattr(app_state, "entity_resolver", None)
    provenance_store = ProvenanceStore(pg_pool)

    if embedding_store is None:
        from knowledge_service.stores.embedding import EmbeddingStore

        embedding_store = EmbeddingStore(pg_pool)

    current_phase = "embedding"
    try:
        # --- Phase 1: Embedding ---
        async with pg_pool.acquire() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET status = 'embedding' WHERE id = $1::uuid", job_id
            )

        texts = [c["chunk_text"] for c in chunk_records]
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            batch_embeddings = await embedding_client.embed_batch(batch)
            embeddings.extend(batch_embeddings)
            async with pg_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE ingestion_jobs SET chunks_embedded = $1 WHERE id = $2::uuid",
                    len(embeddings),
                    job_id,
                )
        for rec, emb in zip(chunk_records, embeddings):
            rec["embedding"] = emb

        # Null out old provenance chunk_ids, delete old chunks, insert new
        async with pg_pool.acquire() as conn:
            await conn.execute(
                """UPDATE provenance SET chunk_id = NULL
                   WHERE chunk_id IN (SELECT id FROM content WHERE content_id = $1::uuid)""",
                content_id,
            )
        await embedding_store.delete_chunks(content_id)
        chunk_id_pairs = await embedding_store.insert_chunks(content_id, chunk_records)
        chunk_id_map = dict(chunk_id_pairs) if chunk_id_pairs else {}

        # --- Phase 2: Extraction ---
        current_phase = "extracting"
        async with pg_pool.acquire() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET status = 'extracting' WHERE id = $1::uuid", job_id
            )

        chunks_failed = 0
        if not body.knowledge and body.raw_text:
            knowledge = []
            chunk_ids_for_items: list[str | None] = []
            for chunk in chunk_records:
                cid = chunk_id_map.get(chunk["chunk_index"])
                items = await extraction_client.extract(
                    chunk["chunk_text"], title=body.title, source_type=body.source_type
                )
                if items is None:
                    chunks_failed += 1
                    async with pg_pool.acquire() as conn:
                        await conn.execute(
                            """UPDATE ingestion_jobs
                               SET chunks_extracted = chunks_extracted + 1,
                                   chunks_failed = chunks_failed + 1
                               WHERE id = $1::uuid""",
                            job_id,
                        )
                    continue
                for item in items:
                    knowledge.append(item)
                    chunk_ids_for_items.append(cid)
                async with pg_pool.acquire() as conn:
                    await conn.execute(
                        """UPDATE ingestion_jobs SET chunks_extracted = chunks_extracted + 1
                           WHERE id = $1::uuid""",
                        job_id,
                    )
            knowledge, chunk_ids_for_items = _dedup_extracted_items(knowledge, chunk_ids_for_items)
            extracted_by_llm = bool(knowledge)
        else:
            knowledge = list(body.knowledge)
            chunk_ids_for_items = [None] * len(knowledge)
            extracted_by_llm = False
            # Mark all chunks as "extracted" (no LLM needed)
            async with pg_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE ingestion_jobs SET chunks_extracted = $1 WHERE id = $2::uuid",
                    len(chunk_records),
                    job_id,
                )

        extractor = f"llm_{settings.llm_chat_model}" if extracted_by_llm else "api"

        # --- Phase 3: Processing ---
        current_phase = "processing"
        async with pg_pool.acquire() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET status = 'processing' WHERE id = $1::uuid", job_id
            )

        # Resolve entity labels
        entities_resolved = 0
        if entity_resolver is not None:
            for i, item in enumerate(knowledge):
                count, knowledge[i] = await _resolve_labels(
                    item, entity_resolver, job_id=job_id, pg_pool=pg_pool
                )
                entities_resolved += count

        # URI fallback
        for i, item in enumerate(knowledge):
            knowledge[i] = apply_uri_fallback(item)

        # Expand to triples and process
        triples_created = 0
        for i, item in enumerate(knowledge):
            for t in expand_to_triples(item):
                cid = chunk_ids_for_items[i] if i < len(chunk_ids_for_items) else None
                is_new, _contras, prov_failed = await process_triple(
                    t,
                    knowledge_store,
                    provenance_store,
                    reasoning_engine,
                    body.url,
                    body.source_type,
                    extractor,
                    chunk_id=cid,
                )
                if is_new:
                    triples_created += 1
                if prov_failed:
                    logger.warning(
                        "Provenance lost for triple in job %s from %s",
                        job_id,
                        body.url,
                    )

        # Log ingestion event
        async with pg_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ingestion_events (event_type, payload, source)
                   VALUES ($1, $2, $3)""",
                "content_ingested",
                json.dumps({"url": body.url, "triples_created": triples_created}),
                body.url,
            )

        # --- Done ---
        async with pg_pool.acquire() as conn:
            await conn.execute(
                """UPDATE ingestion_jobs
                   SET status = 'completed', triples_created = $1,
                       entities_resolved = $2, chunks_failed = $3
                   WHERE id = $4::uuid""",
                triples_created,
                entities_resolved,
                chunks_failed,
                job_id,
            )

    except Exception as exc:
        # Clean up orphaned chunks from Phase 1
        try:
            await embedding_store.delete_chunks(content_id)
        except Exception:
            logger.warning("Failed to clean up chunks for failed job %s", job_id)

        error_json = json.dumps(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "phase": current_phase,
            }
        )
        try:
            async with pg_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE ingestion_jobs SET status = 'failed', error = $1 WHERE id = $2::uuid",
                    error_json,
                    job_id,
                )
        except Exception:
            logger.exception("Failed to update job %s to failed status", job_id)
        logger.exception("Ingestion worker failed for job %s in phase %s", job_id, current_phase)


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
    pg_pool = request.app.state.pg_pool
    embedding_store = getattr(request.app.state, "embedding_store", None)
    if embedding_store is None:
        from knowledge_service.stores.embedding import EmbeddingStore

        embedding_store = EmbeddingStore(pg_pool)

    try:
        if isinstance(raw, list):
            results = []
            for item_raw in raw:
                try:
                    body = ContentRequest(**item_raw)
                except ValidationError as exc:
                    results.append({"error": exc.errors(), "status_code": 422})
                    continue
                result = await _accept_content_request(body, pg_pool, embedding_store)
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
        result = await _accept_content_request(body, pg_pool, embedding_store)
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
    pg_pool = request.app.state.pg_pool
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
