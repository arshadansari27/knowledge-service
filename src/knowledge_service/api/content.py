"""POST /api/content endpoint — ingest content with embedded knowledge."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from knowledge_service._utils import _is_uri
from knowledge_service.api._ingest import process_triple
from knowledge_service.clients.llm import (
    to_entity_uri,
    to_predicate_uri,
    resolve_predicate_synonym,
)
from knowledge_service.chunking import chunk_text as split_into_chunks
from knowledge_service.config import settings
from knowledge_service.models import ContentRequest, ContentResponse, expand_to_triples
from knowledge_service.stores.provenance import ProvenanceStore

router = APIRouter()

_CHUNK_SIZE = 4000
_CHUNK_OVERLAP = 200


async def _resolve_labels(item, entity_resolver) -> tuple[int, object]:
    """Resolve entity labels in a knowledge item via embedding similarity.

    Returns (count_resolved, updated_item).
    """
    resolved = 0
    kt = item.knowledge_type.value

    if kt in ("Claim", "Fact", "Relationship"):
        if not _is_uri(item.subject):
            item.subject = await entity_resolver.resolve(item.subject)
            resolved += 1
        if not _is_uri(item.predicate):
            item.predicate = await entity_resolver.resolve_predicate(item.predicate)
            resolved += 1
        if not _is_uri(item.object):
            item.object = await entity_resolver.resolve(item.object)
            resolved += 1
    elif kt == "TemporalState":
        if not _is_uri(item.subject):
            item.subject = await entity_resolver.resolve(item.subject)
            resolved += 1
        if not _is_uri(item.property):
            item.property = await entity_resolver.resolve_predicate(item.property)
            resolved += 1
    elif kt == "Event":
        if not _is_uri(item.subject):
            item.subject = await entity_resolver.resolve(item.subject)
            resolved += 1

    return resolved, item


def _apply_uri_fallback(item) -> object:
    """Ensure all subject/predicate/object fields are URIs.

    Called after entity resolution to catch any fields that weren't resolved
    (e.g., predicates, or when entity_resolver is None).
    """
    kt = item.knowledge_type.value

    if kt in ("Claim", "Fact", "Relationship"):
        if not _is_uri(item.subject):
            item.subject = to_entity_uri(item.subject)
        if not _is_uri(item.predicate):
            item.predicate = resolve_predicate_synonym(item.predicate)
            item.predicate = to_predicate_uri(item.predicate)
        obj = item.object
        if obj and not _is_uri(obj):
            item.object = to_entity_uri(obj)
    elif kt == "TemporalState":
        if not _is_uri(item.subject):
            item.subject = to_entity_uri(item.subject)
        if not _is_uri(item.property):
            item.property = resolve_predicate_synonym(item.property)
            item.property = to_predicate_uri(item.property)
    elif kt == "Event":
        if not _is_uri(item.subject):
            item.subject = to_entity_uri(item.subject)
    elif kt == "Entity":
        if not _is_uri(item.uri):
            item.uri = to_entity_uri(item.uri)
    elif kt == "Conclusion":
        pass  # Conclusion has no URI fields to normalize

    return item


async def _process_one_content_request(body: ContentRequest, request: Request) -> ContentResponse:
    """Process a single ContentRequest and return its response."""
    knowledge_store = request.app.state.knowledge_store
    pg_pool = request.app.state.pg_pool
    embedding_client = request.app.state.embedding_client
    extraction_client = request.app.state.extraction_client
    reasoning_engine = request.app.state.reasoning_engine
    provenance_store = ProvenanceStore(pg_pool)

    embedding_store = getattr(request.app.state, "embedding_store", None)
    entity_resolver = getattr(request.app.state, "entity_resolver", None)

    # Fall back to local EmbeddingStore if not on app.state
    if embedding_store is None:
        from knowledge_service.stores.embedding import EmbeddingStore

        embedding_store = EmbeddingStore(pg_pool)

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

    # Step 2: Chunk and embed
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

    # Embed chunks (batch for multiple, single for one)
    if len(chunk_records) == 1:
        embeddings = [await embedding_client.embed(chunk_records[0]["chunk_text"])]
    else:
        embeddings = await embedding_client.embed_batch([c["chunk_text"] for c in chunk_records])
    for rec, emb in zip(chunk_records, embeddings):
        rec["embedding"] = emb

    # Delete old chunks (re-ingestion) and insert new
    await embedding_store.delete_chunks(content_id)
    chunk_id_pairs = await embedding_store.insert_chunks(content_id, chunk_records)
    chunk_id_map = dict(chunk_id_pairs) if chunk_id_pairs else {}  # {chunk_index: chunk_id}

    # Step 2.5: Auto-extract knowledge per chunk if none provided
    if not body.knowledge and body.raw_text:
        knowledge_by_chunk: list[tuple[list, str | None]] = []
        for chunk in chunk_records:
            items = await extraction_client.extract(
                chunk["chunk_text"], title=body.title, source_type=body.source_type
            )
            cid = chunk_id_map.get(chunk["chunk_index"])
            knowledge_by_chunk.append((items, cid))
        knowledge = []
        chunk_ids_for_items: list[str | None] = []
        for items, cid in knowledge_by_chunk:
            for item in items:
                knowledge.append(item)
                chunk_ids_for_items.append(cid)
        extracted_by_llm = bool(knowledge)
    else:
        knowledge = list(body.knowledge)
        chunk_ids_for_items = [None] * len(knowledge)
        extracted_by_llm = False

    extractor = f"llm_{settings.llm_chat_model}" if extracted_by_llm else "api"

    # Step 2.75: Resolve entity labels through EntityResolver
    entities_resolved = 0
    if entity_resolver is not None:
        for i, item in enumerate(knowledge):
            count, knowledge[i] = await _resolve_labels(item, entity_resolver)
            entities_resolved += count

    # Step 2.8: Ensure all fields are proper URIs (fallback for unresolved fields)
    for i, item in enumerate(knowledge):
        knowledge[i] = _apply_uri_fallback(item)

    # Step 3: Expand all knowledge items to triples and process
    triples_created = 0
    contradictions_all: list[dict] = []

    for i, item in enumerate(knowledge):
        for t in expand_to_triples(item):
            cid = chunk_ids_for_items[i] if i < len(chunk_ids_for_items) else None
            is_new, contras = await process_triple(
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
            contradictions_all.extend(contras)

    # Step 4: Log ingestion event
    async with pg_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO ingestion_events (event_type, payload, source)
               VALUES ($1, $2, $3)""",
            "content_ingested",
            json.dumps({"url": body.url, "triples_created": triples_created}),
            body.url,
        )

    return ContentResponse(
        content_id=content_id,
        triples_created=triples_created,
        contradictions_detected=contradictions_all,
        entities_resolved=entities_resolved,
    )


@router.post("/content")
async def post_content(request: Request):
    """Ingest a piece of content and its associated knowledge items.

    Accepts a single ContentRequest or a list for batch processing.
    Returns a single ContentResponse or a list, matching the input shape.
    """
    raw = await request.json()

    try:
        if isinstance(raw, list):
            items = [ContentRequest(**item) for item in raw]
            results = await asyncio.gather(
                *[_process_one_content_request(item, request) for item in items]
            )
            return JSONResponse([r.model_dump() for r in results])

        body = ContentRequest(**raw)
        return await _process_one_content_request(body, request)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
