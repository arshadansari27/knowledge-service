"""POST /api/claims endpoint — ingest claims directly without content storage."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from knowledge_service.ingestion.pipeline import IngestContext, ingest_triple
from knowledge_service.models import ClaimsRequest, ClaimsResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def _process_one_claims_request(body: ClaimsRequest, request: Request) -> ClaimsResponse:
    """Process a single ClaimsRequest and return its response."""
    stores = request.app.state.stores
    engine = getattr(request.app.state, "inference_engine", None)
    drainer = getattr(request.app.state, "outbox_drainer", None)

    triples_created = 0
    contradictions_all: list[dict] = []
    thesis_breaks_all: list[dict] = []

    ctx = IngestContext.from_content(
        url=body.source_url,
        source_type=body.source_type,
        extractor=body.extractor,
    )

    for item in body.knowledge:
        # Each knowledge item has a to_triples() method (new model)
        if hasattr(item, "to_triples"):
            triples = item.to_triples()
        elif isinstance(item, dict) and "subject" in item and "predicate" in item:
            triples = [item]
        else:
            logger.warning("Skipping unrecognized knowledge item: %s", type(item))
            continue

        for t in triples:
            result = await ingest_triple(t, stores, ctx, engine=engine, drainer=drainer)
            if result.is_new:
                triples_created += 1
            contradictions_all.extend(result.contradictions)
            thesis_breaks_all.extend(result.thesis_breaks)

    # Log ingestion event (parity with content pipeline)
    try:
        async with stores.pg_pool.acquire() as conn:
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
        thesis_breaks=thesis_breaks_all,
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
