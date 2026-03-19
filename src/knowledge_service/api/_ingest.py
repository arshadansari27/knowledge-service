"""Shared per-triple ingestion logic used by both /api/claims and /api/content."""

from __future__ import annotations

import asyncio

from knowledge_service._utils import _rdf_value_to_str
from knowledge_service.stores.provenance import ProvenanceStore


async def process_triple(
    t: dict,
    knowledge_store,
    pg_pool,
    reasoning_engine,
    source_url: str,
    source_type: str,
    extractor: str,
) -> tuple[bool, list[dict]]:
    """Insert one triple, detect contradictions, record provenance, combine evidence.

    Returns:
        (is_new, contradictions): is_new=True if this triple did not already exist.
    """
    provenance_store = ProvenanceStore(pg_pool)

    triple_hash, is_new = await asyncio.to_thread(
        knowledge_store.insert_triple,
        t["subject"],
        t["predicate"],
        t["object"],
        t["confidence"],
        t["knowledge_type"],
        t["valid_from"],
        t["valid_until"],
    )

    contradictions_raw: list[dict] = await asyncio.to_thread(
        knowledge_store.find_contradictions,
        t["subject"],
        t["predicate"],
        t["object"],
    )
    contradictions = [
        {
            "subject": t["subject"],
            "predicate": t["predicate"],
            "existing_object": _rdf_value_to_str(c.get("object", "")),
            "existing_confidence": c.get("confidence"),
            "new_object": t["object"],
            "new_confidence": t["confidence"],
        }
        for c in contradictions_raw
    ]

    opp_contradictions_raw: list[dict] = await asyncio.to_thread(
        knowledge_store.find_opposite_predicate_contradictions,
        t["subject"],
        t["predicate"],
        t["object"],
    )
    for c in opp_contradictions_raw:
        contradictions.append({
            "subject": t["subject"],
            "predicate": t["predicate"],
            "opposite_predicate_in_store": c["predicate_in_store"],
            "existing_confidence": c.get("confidence"),
            "new_object": t["object"],
            "new_confidence": t["confidence"],
        })

    await provenance_store.insert(
        triple_hash=triple_hash,
        subject=t["subject"],
        predicate=t["predicate"],
        object_=t["object"],
        source_url=source_url,
        source_type=source_type,
        extractor=extractor,
        confidence=t["confidence"],
        metadata={},
        valid_from=t["valid_from"],
        valid_until=t["valid_until"],
    )

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

    return is_new, contradictions
