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
    inferred_triples: list[dict] = field(default_factory=list)


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


async def run_inference(
    triple: dict, engine, stores, context: IngestContext, drainer=None
) -> list[dict]:
    """Run inference engine and persist derived triples via the outbox."""
    if engine is None:
        return []

    from knowledge_service.ontology.namespaces import KS_GRAPH_INFERRED  # noqa: PLC0415

    derived_list = engine.run(triple)
    results = []

    for derived in derived_list:
        derived_hash = _derived_hash(derived)
        async with stores.pg_pool.acquire() as conn:
            async with conn.transaction():
                staged_id = await stores.outbox.stage(
                    conn,
                    operation="insert_inferred",
                    triple_hash=derived_hash,
                    subject=derived.subject,
                    predicate=derived.predicate,
                    object_=derived.object_,
                    confidence=derived.confidence,
                    knowledge_type="inferred",
                    graph=KS_GRAPH_INFERRED,
                    payload={
                        "derived_from": list(derived.derived_from),
                        "inference_method": derived.inference_method,
                    },
                )
                await stores.provenance.insert(
                    derived_hash,
                    derived.subject,
                    derived.predicate,
                    derived.object_,
                    context.source_url,
                    context.source_type,
                    f"inference:{derived.inference_method}",
                    derived.confidence,
                    {"derived_from": list(derived.derived_from)},
                    None,
                    None,
                    None,
                    conn=conn,
                )
        if drainer is not None:
            await drainer.drain_ids([staged_id])
        results.append(derived.to_dict())

    return results


def _derived_hash(derived) -> str:
    """SHA-256 of derived triple's canonical form, matching compute_hash()."""
    import hashlib  # noqa: PLC0415

    from pyoxigraph import Literal, NamedNode, Triple  # noqa: PLC0415

    s = NamedNode(derived.subject)
    p = NamedNode(derived.predicate)
    o = NamedNode(derived.object_) if is_uri(derived.object_) else Literal(derived.object_)
    return hashlib.sha256(str(Triple(s, p, o)).encode()).hexdigest()


def _remove_inferred_triple_with_annotations(raw_store, s, p, o, graph_node) -> None:
    """Remove an inferred triple and all its RDF-star annotations from the raw store.

    RDF-star inline annotations (inserted via SPARQL INSERT DATA) are stored as
    reification bnodes by pyoxigraph. We must find those bnodes via the rdf:reifies
    predicate and remove all associated quads before removing the base quad.

    Note: DELETE WHERE doesn't work with RDF-star in pyoxigraph — we must use the
    Python API to find and remove reification bnodes explicitly.
    """
    from pyoxigraph import NamedNode as NN
    from pyoxigraph import Quad

    reifies = NN("http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies")

    # Find all bnodes in the inferred graph that reify (s p o)
    for reif_quad in list(raw_store.quads_for_pattern(None, reifies, None, graph_node)):
        bnode = reif_quad.subject
        reified = reif_quad.object
        # reified is a Triple object — check it matches our SPO
        if (
            hasattr(reified, "subject")
            and reified.subject == s
            and reified.predicate == p
            and reified.object == o
        ):
            # Remove all annotation quads attached to this bnode
            for bnode_quad in list(raw_store.quads_for_pattern(bnode, None, None, graph_node)):
                raw_store.remove(bnode_quad)

    # Remove the base quad
    raw_store.remove(Quad(s, p, o, graph_node))


_MAX_RETRACT_DEPTH = 10


def retract_stale_inferences(
    triple_hash: str,
    triple_store,
    _seen: set[str] | None = None,
    _depth: int = 0,
) -> int:
    """Remove inferred triples that depend on a changed source triple.

    Synchronous — operates directly on the TripleStore. Wrap in asyncio.to_thread()
    at the call site. Includes cycle detection and depth limit.
    """
    if _depth >= _MAX_RETRACT_DEPTH:
        logger.warning("retract_stale_inferences: depth limit reached at hash %s", triple_hash)
        return 0

    if _seen is None:
        _seen = set()
    if triple_hash in _seen:
        return 0
    _seen.add(triple_hash)

    from knowledge_service.ontology.namespaces import KS_GRAPH_INFERRED
    from knowledge_service.ontology.uri import KS as KS_NS
    from pyoxigraph import Literal as Lit
    from pyoxigraph import NamedNode as NN

    rows = triple_store.query(f"""
        SELECT ?s ?p ?o WHERE {{
            GRAPH <{KS_GRAPH_INFERRED}> {{
                ?s ?p ?o .
                << ?s ?p ?o >> <{KS_NS}derivedFrom> "{triple_hash}" .
            }}
        }}
    """)
    if not rows:
        return 0

    removed = 0
    g = NN(KS_GRAPH_INFERRED)
    for row in rows:
        s_val = row["s"].value if hasattr(row["s"], "value") else str(row["s"])
        p_val = row["p"].value if hasattr(row["p"], "value") else str(row["p"])
        o_val = row["o"].value if hasattr(row["o"], "value") else str(row["o"])

        s = NN(s_val)
        p = NN(p_val)
        o = NN(o_val) if is_uri(o_val) else Lit(o_val)
        _remove_inferred_triple_with_annotations(triple_store.store, s, p, o, g)
        removed += 1

        # Cascade: retract inferences that depended on this inferred triple
        dep = {"subject": s_val, "predicate": p_val, "object": o_val}
        dep_hash = compute_hash(dep)
        removed += retract_stale_inferences(dep_hash, triple_store, _seen, _depth + 1)

    return removed


async def ingest_triple(
    triple: dict,
    stores,
    context: IngestContext,
    engine=None,
    drainer=None,
) -> IngestResult:
    """Ingest a single triple using the outbox pattern.

    Phase A (read-only): normalise + read prior state from pyoxigraph.
    Phase B (PG txn):    stage outbox + insert provenance. Commit is the
                         point of no return.
    Phase C (drain):     apply just-committed outbox rows to pyoxigraph.
    Phase D (derived):   contradictions + inference, each in its own B+C
                         cycle. Skippable under crash; re-ingest re-runs.
    """
    from knowledge_service._utils import is_object_entity  # noqa: PLC0415
    from knowledge_service.ontology.uri import to_entity_uri as _to_entity_uri  # noqa: PLC0415

    if is_object_entity(triple) and not is_uri(triple.get("object", "")):
        triple = {**triple, "object": _to_entity_uri(triple["object"])}

    triple_hash = compute_hash(triple)

    # Phase A
    delta = await detect_delta(triple, stores.triples)

    # Phase B: one PG txn for outbox + provenance
    staged_ids: list[int] = []
    async with stores.pg_pool.acquire() as conn:
        async with conn.transaction():
            insert_id = await stores.outbox.stage(
                conn,
                operation="insert",
                triple_hash=triple_hash,
                subject=triple["subject"],
                predicate=triple["predicate"],
                object_=triple["object"],
                confidence=triple["confidence"],
                knowledge_type=triple["knowledge_type"],
                valid_from=triple.get("valid_from"),
                valid_until=triple.get("valid_until"),
                graph=context.graph,
            )
            staged_ids.append(insert_id)

            if delta is not None:
                old_triple = {
                    "subject": triple["subject"],
                    "predicate": triple["predicate"],
                    "object": delta["prior_value"],
                }
                old_hash = compute_hash(old_triple)
                retract_id = await stores.outbox.stage(
                    conn,
                    operation="retract_inference",
                    triple_hash=old_hash,
                    subject=old_triple["subject"],
                    predicate=old_triple["predicate"],
                    object_=old_triple["object"],
                    graph=context.graph,
                )
                staged_ids.append(retract_id)

            await stores.provenance.insert(
                triple_hash,
                triple["subject"],
                triple["predicate"],
                triple["object"],
                context.source_url,
                context.source_type,
                context.extractor,
                triple["confidence"],
                {},
                triple.get("valid_from"),
                triple.get("valid_until"),
                context.chunk_id,
                conn=conn,
            )

    # Phase C
    applied = []
    if drainer is not None:
        applied = await drainer.drain_ids(staged_ids)
    is_new = any(a.operation == "insert" and a.is_new is True for a in applied)

    # Phase D: contradictions → penalty
    contradictions = await detect_contradictions(triple, stores.triples)
    confidence = triple["confidence"]
    if contradictions and is_new:
        confidence = apply_penalty(confidence, contradictions)
        if drainer is not None:
            async with stores.pg_pool.acquire() as conn:
                async with conn.transaction():
                    upd_id = await stores.outbox.stage(
                        conn,
                        operation="update_confidence",
                        triple_hash=triple_hash,
                        subject=triple["subject"],
                        predicate=triple["predicate"],
                        object_=triple["object"],
                        confidence=confidence,
                        graph=context.graph,
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
                        conn=conn,
                    )
            await drainer.drain_ids([upd_id])

    combined = await combine_evidence(triple_hash, stores.provenance)
    if combined != confidence and drainer is not None:
        async with stores.pg_pool.acquire() as conn:
            async with conn.transaction():
                comb_id = await stores.outbox.stage(
                    conn,
                    operation="update_confidence",
                    triple_hash=triple_hash,
                    subject=triple["subject"],
                    predicate=triple["predicate"],
                    object_=triple["object"],
                    confidence=combined,
                    graph=context.graph,
                )
        await drainer.drain_ids([comb_id])
        confidence = combined

    from knowledge_service.ontology.uri import to_entity_uri, to_predicate_uri  # noqa: PLC0415

    normalized = {
        **triple,
        "subject": to_entity_uri(triple["subject"]),
        "predicate": to_predicate_uri(triple["predicate"]),
        "confidence": confidence,
    }
    inferred = await run_inference(normalized, engine, stores, context, drainer=drainer)

    return IngestResult(is_new, delta, contradictions, confidence, inferred)
