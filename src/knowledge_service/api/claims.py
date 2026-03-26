"""POST /api/claims endpoint — ingest claims directly without content storage."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from knowledge_service.api._ingest import apply_uri_fallback, process_triple
from knowledge_service.models import ClaimsRequest, ClaimsResponse, expand_to_triples
from knowledge_service.stores.provenance import ProvenanceStore

logger = logging.getLogger(__name__)

router = APIRouter()


async def _process_one_claims_request(body: ClaimsRequest, request: Request) -> ClaimsResponse:
    """Process a single ClaimsRequest and return its response."""
    knowledge_store = request.app.state.knowledge_store
    pg_pool = request.app.state.pg_pool
    reasoning_engine = request.app.state.reasoning_engine
    provenance_store = ProvenanceStore(pg_pool)

    triples_created = 0
    contradictions_all: list[dict] = []

    for item in body.knowledge:
        item = apply_uri_fallback(item)
        for t in expand_to_triples(item):
            is_new, contras, prov_failed = await process_triple(
                t,
                knowledge_store,
                provenance_store,
                reasoning_engine,
                body.source_url,
                body.source_type,
                body.extractor,
            )
            if is_new:
                triples_created += 1
            if prov_failed:
                logger.warning(
                    "Provenance lost for triple from %s",
                    body.source_url,
                )
            contradictions_all.extend(contras)

    # Log ingestion event (parity with content pipeline)
    try:
        async with pg_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ingestion_events (event_type, payload, source)
                   VALUES ($1, $2, $3)""",
                "claims_ingested",
                json.dumps(
                    {
                        "source_url": body.source_url,
                        "extractor": body.extractor,
                        "triples_created": triples_created,
                    }
                ),
                body.source_url,
            )
    except Exception:
        logger.exception("Failed to log ingestion event for claims from %s", body.source_url)

    return ClaimsResponse(
        triples_created=triples_created,
        contradictions_detected=contradictions_all,
    )


@router.post("/claims")
async def post_claims(request: Request):
    """Ingest knowledge items directly without associated content.

    Accepts a single ClaimsRequest or a list for batch processing.
    Returns a single ClaimsResponse or a list, matching the input shape.
    """
    raw = await request.json()

    try:
        if isinstance(raw, list):
            items = [ClaimsRequest(**item) for item in raw]
            results = [await _process_one_claims_request(item, request) for item in items]
            return JSONResponse([r.model_dump() for r in results])

        body = ClaimsRequest(**raw)
        return await _process_one_claims_request(body, request)
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
