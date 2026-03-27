"""Ingestion pipeline. Replaces the process_triple() god function."""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field

from pyoxigraph import Literal, NamedNode, Triple

from knowledge_service.ontology.uri import is_uri
from knowledge_service.ontology.namespaces import KS_GRAPH_EXTRACTED, KS_GRAPH_ASSERTED
from knowledge_service.reasoning.noisy_or import noisy_or

logger = logging.getLogger(__name__)

PENALTY_FACTOR = 0.5


@dataclass
class IngestContext:
    source_url: str
    source_type: str
    extractor: str
    graph: str
    chunk_id: str | None = None

    @classmethod
    def from_content(cls, url, source_type, extractor, chunk_id=None):
        graph = KS_GRAPH_ASSERTED if extractor == "api" else KS_GRAPH_EXTRACTED
        return cls(url, source_type, extractor, graph, chunk_id)


@dataclass
class IngestResult:
    is_new: bool
    delta: dict | None
    contradictions: list[dict]
    confidence: float
    thesis_breaks: list[dict] = field(default_factory=list)


def compute_hash(triple: dict) -> str:
    s = NamedNode(triple["subject"])
    p = NamedNode(triple["predicate"])
    o_val = triple["object"]
    o = NamedNode(o_val) if is_uri(o_val) else Literal(o_val)
    return hashlib.sha256(str(Triple(s, p, o)).encode()).hexdigest()


def apply_penalty(confidence: float, contradictions: list[dict]) -> float:
    if not contradictions:
        return confidence
    max_existing = max((c.get("existing_confidence", 0) for c in contradictions), default=0)
    return confidence * (1.0 - max_existing * PENALTY_FACTOR)


async def detect_delta(triple: dict, triple_store) -> dict | None:
    existing = await asyncio.to_thread(
        triple_store.get_triples,
        subject=triple["subject"],
        predicate=triple["predicate"],
    )
    if not existing:
        return None
    latest = existing[0]
    if latest["object"] == triple["object"]:
        return None
    return {
        "prior_value": latest["object"],
        "current_value": triple["object"],
        "description": f"changed from {latest['object']} to {triple['object']}",
    }


async def insert_triple(triple: dict, triple_store, graph: str) -> tuple[str, bool]:
    return await asyncio.to_thread(
        triple_store.insert,
        triple["subject"],
        triple["predicate"],
        triple["object"],
        triple["confidence"],
        triple["knowledge_type"],
        triple.get("valid_from"),
        triple.get("valid_until"),
        graph,
    )


async def detect_contradictions(triple: dict, triple_store) -> list[dict]:
    contras = await asyncio.to_thread(
        triple_store.find_contradictions,
        triple["subject"],
        triple["predicate"],
        triple["object"],
    )
    opp = await asyncio.to_thread(
        triple_store.find_opposite_contradictions,
        triple["subject"],
        triple["predicate"],
        triple["object"],
    )
    return contras + opp


async def combine_evidence(triple_hash: str, provenance_store) -> float:
    rows = await provenance_store.get_by_triple(triple_hash)
    if not rows:
        return 0.0
    confidences = [r["confidence"] for r in rows if r.get("confidence") is not None]
    if len(confidences) <= 1:
        return confidences[0] if confidences else 0.0
    return noisy_or(confidences)


async def check_thesis_impact(triple_hash, contradictions, stores) -> list[dict]:
    if not contradictions:
        return []
    affected_hashes = {triple_hash}
    for c in contradictions:
        h = c.get("existing_hash") or c.get("triple_hash")
        if h:
            affected_hashes.add(h)
    return await stores.theses.find_by_hashes(affected_hashes, status="active")


async def ingest_triple(triple: dict, stores, context: IngestContext) -> IngestResult:
    triple_hash = compute_hash(triple)

    delta = await detect_delta(triple, stores.triples)

    _, is_new = await insert_triple(triple, stores.triples, context.graph)

    contradictions = await detect_contradictions(triple, stores.triples)

    confidence = triple["confidence"]
    if contradictions and is_new:
        confidence = apply_penalty(confidence, contradictions)
        await asyncio.to_thread(
            stores.triples.update_confidence,
            triple,
            confidence,
        )

    await stores.provenance.insert(
        triple_hash,
        triple["subject"],
        triple["predicate"],
        triple["object"],
        context.source_url,
        context.source_type,
        context.extractor,
        confidence,
        {},
        triple.get("valid_from"),
        triple.get("valid_until"),
        context.chunk_id,
    )

    combined = await combine_evidence(triple_hash, stores.provenance)
    if combined != confidence:
        await asyncio.to_thread(
            stores.triples.update_confidence,
            triple,
            combined,
        )

    thesis_breaks = await check_thesis_impact(triple_hash, contradictions, stores)

    return IngestResult(is_new, delta, contradictions, combined, thesis_breaks)
