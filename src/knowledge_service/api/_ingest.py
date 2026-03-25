"""Shared per-triple ingestion logic used by both /api/claims and /api/content."""

from __future__ import annotations

import asyncio
import logging

from knowledge_service._utils import _rdf_value_to_str
from knowledge_service.ontology.namespaces import KS_GRAPH_ASSERTED, KS_GRAPH_EXTRACTED
from knowledge_service.stores.provenance import ProvenanceStore

logger = logging.getLogger(__name__)

# Max confidence penalty when a contradiction exists at confidence=1.0
_CONTRADICTION_PENALTY_FACTOR = 0.5


def _extractor_to_graph(extractor: str) -> str:
    if extractor.startswith("llm_"):
        return KS_GRAPH_EXTRACTED
    return KS_GRAPH_ASSERTED


def _penalize_confidence(new_conf: float, existing_confs: list[float]) -> float:
    """Reduce confidence of a new triple that contradicts existing triples.

    Penalty = max(existing_confidence) * _CONTRADICTION_PENALTY_FACTOR.
    E.g., if existing has confidence 0.9 and penalty factor is 0.5,
    penalty is 0.45, so new_conf is multiplied by (1 - 0.45) = 0.55.
    """
    if not existing_confs:
        return new_conf
    max_existing = max(c for c in existing_confs if c is not None) if existing_confs else 0.0
    penalty = max_existing * _CONTRADICTION_PENALTY_FACTOR
    return round(new_conf * (1.0 - penalty), 4)


async def process_triple(
    t: dict,
    knowledge_store,
    provenance_store: ProvenanceStore,
    reasoning_engine,
    source_url: str,
    source_type: str,
    extractor: str,
    chunk_id: str | None = None,
) -> tuple[bool, list[dict]]:
    """Insert one triple, detect contradictions, record provenance, combine evidence.

    If contradictions are found, the new triple's confidence is penalized
    proportionally to the strongest contradicting triple's confidence.

    If the pyoxigraph insert succeeds but provenance fails, the triple is still
    in the graph (idempotent on retry) but the error is logged. Callers see
    is_new=False so the triple isn't double-counted.

    Returns:
        (is_new, contradictions): is_new=True if this triple did not already exist.
    """

    graph = _extractor_to_graph(extractor)

    triple_hash, is_new = await asyncio.to_thread(
        knowledge_store.insert_triple,
        t["subject"],
        t["predicate"],
        t["object"],
        t["confidence"],
        t["knowledge_type"],
        t["valid_from"],
        t["valid_until"],
        graph,
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
        contradictions.append(
            {
                "subject": t["subject"],
                "predicate": t["predicate"],
                "opposite_predicate_in_store": c["predicate_in_store"],
                "existing_confidence": c.get("confidence"),
                "new_object": t["object"],
                "new_confidence": t["confidence"],
            }
        )

    # Penalize confidence when contradictions exist
    effective_confidence = t["confidence"]
    if contradictions and is_new:
        existing_confs = [c.get("existing_confidence") for c in contradictions]
        existing_confs = [c for c in existing_confs if c is not None]
        effective_confidence = _penalize_confidence(t["confidence"], existing_confs)
        if effective_confidence != t["confidence"]:
            logger.info(
                "Contradiction penalty: triple %s confidence %.3f -> %.3f",
                triple_hash[:12],
                t["confidence"],
                effective_confidence,
            )
            try:
                await asyncio.to_thread(
                    knowledge_store.update_confidence,
                    t["subject"],
                    t["predicate"],
                    t["object"],
                    effective_confidence,
                )
            except Exception:
                logger.exception(
                    "Failed to apply contradiction penalty for triple %s",
                    triple_hash[:12],
                )

    try:
        await provenance_store.insert(
            triple_hash=triple_hash,
            subject=t["subject"],
            predicate=t["predicate"],
            object_=t["object"],
            source_url=source_url,
            source_type=source_type,
            extractor=extractor,
            confidence=effective_confidence,
            metadata={},
            valid_from=t["valid_from"],
            valid_until=t["valid_until"],
            chunk_id=chunk_id,
        )
    except Exception:
        logger.exception(
            "Failed to record provenance for triple %s from %s — "
            "triple exists in graph but provenance is missing",
            triple_hash[:12],
            source_url,
        )
        return is_new, contradictions

    prov_rows = await provenance_store.get_by_triple(triple_hash)
    if len(prov_rows) > 1:
        combined = reasoning_engine.combine_evidence([r["confidence"] for r in prov_rows])
        try:
            await asyncio.to_thread(
                knowledge_store.update_confidence,
                t["subject"],
                t["predicate"],
                t["object"],
                combined,
            )
        except Exception:
            logger.exception(
                "Failed to update combined confidence for triple %s",
                triple_hash[:12],
            )

    return is_new, contradictions
