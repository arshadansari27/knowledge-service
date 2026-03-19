"""POST /api/claims endpoint — ingest claims directly without content storage."""

from __future__ import annotations

from fastapi import APIRouter, Request

from knowledge_service.api._ingest import process_triple
from knowledge_service.models import ClaimsRequest, ClaimsResponse, expand_to_triples

router = APIRouter()


@router.post("/claims", response_model=ClaimsResponse)
async def post_claims(body: ClaimsRequest, request: Request) -> ClaimsResponse:
    """Ingest knowledge items directly without associated content."""
    knowledge_store = request.app.state.knowledge_store
    pg_pool = request.app.state.pg_pool
    reasoning_engine = request.app.state.reasoning_engine

    triples_created = 0
    contradictions_all: list[dict] = []

    for item in body.knowledge:
        for t in expand_to_triples(item):
            is_new, contras = await process_triple(
                t,
                knowledge_store,
                pg_pool,
                reasoning_engine,
                body.source_url,
                body.source_type,
                body.extractor,
            )
            if is_new:
                triples_created += 1
            contradictions_all.extend(contras)

    return ClaimsResponse(
        triples_created=triples_created,
        contradictions_detected=contradictions_all,
    )
