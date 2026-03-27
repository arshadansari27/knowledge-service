"""Entity changes endpoint — what changed for an entity since a given date."""

import logging
from datetime import datetime

from fastapi import APIRouter, Query, Request

from knowledge_service.ontology.uri import to_entity_uri

logger = logging.getLogger(__name__)

router = APIRouter(tags=["changes"])


@router.get("/api/entity/{entity_id}/changes")
async def get_entity_changes(
    request: Request,
    entity_id: str,
    since: str = Query(..., description="ISO date (YYYY-MM-DD)"),
    limit: int = Query(20, ge=1, le=100),
):
    """Return recent changes for an entity, ranked by materiality * confidence."""
    stores = request.app.state.stores
    registry = request.app.state.domain_registry

    entity_uri = to_entity_uri(entity_id)
    since_dt = datetime.fromisoformat(since)

    # 1. Recent provenance rows
    recent = await stores.provenance.query_by_entity_and_time(entity_uri, since_dt)

    # 2. Enrich with triple metadata + materiality
    changes = []
    for row in recent:
        triples = stores.triples.get_triples(
            subject=row["subject"], predicate=row["predicate"], object_=row["object"]
        )
        triple_meta = triples[0] if triples else {}
        materiality = registry.get_materiality(row["predicate"])
        confidence = triple_meta.get("confidence") or row.get("confidence") or 0.0

        changes.append(
            {
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object"],
                "confidence": confidence,
                "knowledge_type": triple_meta.get("knowledge_type"),
                "materiality": materiality,
                "source_url": row.get("source_url"),
                "source_type": row.get("source_type"),
                "ingested_at": str(row["ingested_at"]),
            }
        )

    # 3. Sort by materiality * confidence
    changes.sort(key=lambda c: c["materiality"] * (c["confidence"] or 0), reverse=True)

    # 4. Check thesis breaks
    thesis_breaks = await stores.theses.find_breaks_for_entity(entity_uri, since_dt)

    return {
        "entity": entity_uri,
        "since": since,
        "changes": changes[:limit],
        "thesis_breaks": thesis_breaks,
    }
