"""POST /api/content endpoint — ingest content with embedded knowledge."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge_service._utils import _is_uri
from knowledge_service.api._ingest import process_triple
from knowledge_service.config import settings
from knowledge_service.models import ContentRequest, ContentResponse, expand_to_triples

router = APIRouter()

_CHUNK_SIZE = 4000
_CHUNK_OVERLAP = 200
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


async def _resolve_labels(item, entity_resolver) -> tuple[int, object]:
    """Resolve entity labels in a knowledge item. Returns (count, updated_item)."""
    resolved = 0
    kt = item.knowledge_type.value

    if kt in ("Claim", "Fact", "Relationship"):
        if not _is_uri(item.subject):
            item.subject = await entity_resolver.resolve(item.subject)
            resolved += 1
        if not _is_uri(item.object) and " " not in item.object and len(item.object) <= 60:
            item.object = await entity_resolver.resolve(item.object)
            resolved += 1
    elif kt == "TemporalState":
        if not _is_uri(item.subject):
            item.subject = await entity_resolver.resolve(item.subject)
            resolved += 1
    elif kt == "Event":
        if not _is_uri(item.subject):
            item.subject = await entity_resolver.resolve(item.subject)
            resolved += 1

    return resolved, item


async def _process_one_content_request(body: ContentRequest, request: Request) -> ContentResponse:
    """Process a single ContentRequest and return its response."""
    knowledge_store = request.app.state.knowledge_store
    pg_pool = request.app.state.pg_pool
    embedding_client = request.app.state.embedding_client
    extraction_client = request.app.state.extraction_client
    reasoning_engine = request.app.state.reasoning_engine

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

    # Step 2: Always chunk and embed
    text = body.raw_text or body.summary or body.title
    if len(text) >= _CHUNK_SIZE:
        chunks_text = _splitter.split_text(text)
    else:
        chunks_text = [text]

    # Track char offsets
    chunk_records: list[dict] = []
    search_start = 0
    for i, ct in enumerate(chunks_text):
        char_start = text.find(ct[:100], search_start)
        if char_start == -1:
            char_start = search_start
        char_end = char_start + len(ct)
        search_start = max(search_start, char_start + 1)
        chunk_records.append(
            {
                "chunk_index": i,
                "chunk_text": ct,
                "char_start": char_start,
                "char_end": char_end,
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
    await embedding_store.insert_chunks(content_id, chunk_records)

    # Step 2.5: Auto-extract knowledge from raw_text if none provided
    if not body.knowledge and body.raw_text:
        knowledge = await extraction_client.extract(
            body.raw_text, title=body.title, source_type=body.source_type
        )
        extracted_by_llm = bool(knowledge)
    else:
        knowledge = list(body.knowledge)
        extracted_by_llm = False

    extractor = f"llm_{settings.llm_chat_model}" if extracted_by_llm else "api"

    # Step 2.75: Resolve entity labels through EntityResolver
    entities_resolved = 0
    if entity_resolver is not None:
        for i, item in enumerate(knowledge):
            count, knowledge[i] = await _resolve_labels(item, entity_resolver)
            entities_resolved += count

    # Step 3: Expand all knowledge items to triples and process
    triples_created = 0
    contradictions_all: list[dict] = []

    for item in knowledge:
        for t in expand_to_triples(item):
            is_new, contras = await process_triple(
                t,
                knowledge_store,
                pg_pool,
                reasoning_engine,
                body.url,
                body.source_type,
                extractor,
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
            results = [await _process_one_content_request(item, request) for item in items]
            return JSONResponse([r.model_dump() for r in results])

        body = ContentRequest(**raw)
        return await _process_one_content_request(body, request)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
