"""POST /api/claims endpoint — ingest claims directly without content storage."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from knowledge_service.models import ClaimsRequest, ClaimsResponse, expand_to_triples
from knowledge_service.stores.provenance import ProvenanceStore

router = APIRouter()


@router.post("/claims", response_model=ClaimsResponse)
async def post_claims(body: ClaimsRequest, request: Request) -> ClaimsResponse:
    """Ingest knowledge items directly without associated content."""
    knowledge_store = request.app.state.knowledge_store
    pg_pool = request.app.state.pg_pool
    reasoning_engine = request.app.state.reasoning_engine

    provenance_store = ProvenanceStore(pg_pool)

    triples_created = 0
    contradictions_all: list[dict] = []

    for item in body.knowledge:
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
                source_url=body.source_url,
                source_type=body.source_type,
                extractor=body.extractor,
                confidence=t["confidence"],
                metadata={},
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

    return ClaimsResponse(
        triples_created=triples_created,
        contradictions_detected=contradictions_all,
    )
