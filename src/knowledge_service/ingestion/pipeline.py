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


async def check_thesis_impact(triple_hash, contradictions, stores) -> list[dict]:
    if not contradictions:
        return []
    affected_hashes = {triple_hash}
    for c in contradictions:
        h = c.get("existing_hash") or c.get("triple_hash")
        if h:
            affected_hashes.add(h)
    return await stores.theses.find_by_hashes(affected_hashes, status="active")


async def run_inference(triple: dict, engine, stores, context: IngestContext) -> list[dict]:
    """Run inference engine and persist derived triples with RDF-star annotations."""
    if engine is None:
        return []

    from knowledge_service.ontology.namespaces import KS_GRAPH_INFERRED
    from knowledge_service.ontology.uri import KS as KS_NS

    derived_list = engine.run(triple)
    results = []

    for derived in derived_list:
        derived_hash, _ = await asyncio.to_thread(
            stores.triples.insert,
            derived.subject,
            derived.predicate,
            derived.object_,
            derived.confidence,
            "inferred",
            None,
            None,
            KS_GRAPH_INFERRED,
        )

        # Add inference-specific RDF-star annotations (ks:derivedFrom + ks:inferenceMethod)
        obj_sparql = f"<{derived.object_}>" if is_uri(derived.object_) else f'"{derived.object_}"'
        quoted = f"<< <{derived.subject}> <{derived.predicate}> {obj_sparql} >>"

        annotation_lines = [
            f'{quoted} <{KS_NS}inferenceMethod> "{derived.inference_method}" .',
        ]
        for source_hash in derived.derived_from:
            annotation_lines.append(f'{quoted} <{KS_NS}derivedFrom> "{source_hash}" .')

        body = "\n                    ".join(annotation_lines)
        await asyncio.to_thread(
            stores.triples.store.update,
            f"""INSERT DATA {{
                GRAPH <{KS_GRAPH_INFERRED}> {{
                    {body}
                }}
            }}""",
        )

        # Provenance for inferred triple
        await stores.provenance.insert(
            derived_hash,
            derived.subject,
            derived.predicate,
            derived.object_,
            context.source_url,
            context.source_type,
            f"inference:{derived.inference_method}",
            derived.confidence,
            {"derived_from": derived.derived_from},
            None,
            None,
            None,
        )

        results.append(derived.to_dict())

    return results


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


async def ingest_triple(triple: dict, stores, context: IngestContext, engine=None) -> IngestResult:
    # Normalize entity objects to URIs early so the stored triple, hash,
    # provenance, and inference engine all see the same URI form.
    from knowledge_service._utils import is_object_entity  # noqa: PLC0415
    from knowledge_service.ontology.uri import to_entity_uri as _to_entity_uri  # noqa: PLC0415

    if is_object_entity(triple) and not is_uri(triple.get("object", "")):
        triple = {**triple, "object": _to_entity_uri(triple["object"])}

    triple_hash = compute_hash(triple)

    delta = await detect_delta(triple, stores.triples)

    # Retract stale inferences when a delta is detected
    if delta is not None:
        old_triple = {
            "subject": triple["subject"],
            "predicate": triple["predicate"],
            "object": delta["prior_value"],
        }
        old_hash = compute_hash(old_triple)
        await asyncio.to_thread(retract_stale_inferences, old_hash, stores.triples)

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

    # Run inference engine on URI-normalized triple. TripleStore.insert() normalizes
    # subjects/predicates to URIs but keeps non-URI objects as literals. The engine
    # needs the same normalized form so DerivedTriple.compute_hash() can create
    # NamedNode/Literal objects correctly. Object is already normalized above.
    from knowledge_service.ontology.uri import to_entity_uri, to_predicate_uri  # noqa: PLC0415

    normalized = {
        **triple,
        "subject": to_entity_uri(triple["subject"]),
        "predicate": to_predicate_uri(triple["predicate"]),
        "confidence": combined,
    }
    inferred = await run_inference(normalized, engine, stores, context)

    thesis_breaks = await check_thesis_impact(triple_hash, contradictions, stores)

    return IngestResult(is_new, delta, contradictions, combined, thesis_breaks, inferred)
