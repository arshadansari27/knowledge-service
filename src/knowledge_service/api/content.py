"""POST /api/content endpoint — ingest content with embedded knowledge."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request

from knowledge_service._utils import _is_uri
from knowledge_service.config import settings
from knowledge_service.models import ContentRequest, ContentResponse, expand_to_triples
from knowledge_service.stores.provenance import ProvenanceStore

router = APIRouter()


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


@router.post("/content", response_model=ContentResponse)
async def post_content(body: ContentRequest, request: Request) -> ContentResponse:
    """Ingest a piece of content and its associated knowledge items."""
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

    provenance_store = ProvenanceStore(pg_pool)

    # Step 1: Generate embedding for the content
    embed_text = body.raw_text or body.summary or body.title
    embedding = await embedding_client.embed(embed_text)

    # Step 2: Upsert content row in PostgreSQL
    content_id = await embedding_store.insert_content(
        url=body.url,
        title=body.title,
        summary=body.summary or "",
        raw_text=body.raw_text or "",
        source_type=body.source_type,
        tags=body.tags,
        embedding=embedding,
        metadata=body.metadata,
    )

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
            valid_from_dt = t["valid_from"]
            valid_until_dt = t["valid_until"]

            triple_hash: str = await asyncio.to_thread(
                knowledge_store.insert_triple,
                t["subject"],
                t["predicate"],
                t["object"],
                t["confidence"],
                t["knowledge_type"],
                valid_from_dt,
                valid_until_dt,
            )
            triples_created += 1

            # Check for contradictions
            contradictions: list[dict] = await asyncio.to_thread(
                knowledge_store.find_contradictions,
                t["subject"],
                t["predicate"],
                t["object"],
            )
            for c in contradictions:
                contradictions_all.append(
                    {
                        "subject": t["subject"],
                        "predicate": t["predicate"],
                        "existing_object": str(c.get("object", "")),
                        "existing_confidence": c.get("confidence"),
                        "new_object": t["object"],
                        "new_confidence": t["confidence"],
                    }
                )

            # Record provenance
            await provenance_store.insert(
                triple_hash=triple_hash,
                subject=t["subject"],
                predicate=t["predicate"],
                object_=t["object"],
                source_url=body.url,
                source_type=body.source_type,
                extractor=extractor,
                confidence=t["confidence"],
                metadata=body.metadata,
                valid_from=valid_from_dt,
                valid_until=valid_until_dt,
            )

            # Combine evidence if multiple sources exist for this triple
            prov_rows = await provenance_store.get_by_triple(triple_hash)
            if len(prov_rows) > 1:
                combined = reasoning_engine.combine_evidence([r["confidence"] for r in prov_rows])
                await asyncio.to_thread(
                    knowledge_store.update_confidence,
                    t["subject"],
                    t["predicate"],
                    t["object"],
                    combined,
                )

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
