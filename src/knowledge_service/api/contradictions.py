"""GET /api/knowledge/contradictions endpoint."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from knowledge_service._utils import _triple_hash, _rdf_value_to_str
from knowledge_service.ontology.namespaces import KS_CONFIDENCE, KS_OPPOSITE_PREDICATE
from knowledge_service.stores.provenance import ProvenanceStore

router = APIRouter()


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class ContradictionResponse(BaseModel):
    claim_a: dict
    claim_b: dict
    contradiction_probability: float
    provenance_a: list[dict] = []
    provenance_b: list[dict] = []


def _serialise_provenance(rows: list[dict]) -> list[dict]:
    """Serialise any non-JSON-safe values (e.g. datetime) in provenance rows."""
    result = []
    for row in rows:
        result.append(
            {
                k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                for k, v in row.items()
            }
        )
    return result


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/knowledge/contradictions", response_model=list[ContradictionResponse])
async def get_contradictions(
    request: Request,
    min_confidence: float = Query(0.0, description="Minimum contradiction probability to include"),
) -> list[ContradictionResponse]:
    """Find all contradictions in the knowledge graph.

    A contradiction is a pair of triples that share the same subject and predicate
    but have different objects. The contradiction probability is computed as the
    product of both claims' individual confidence scores.

    Optional query parameter:
    - ``min_confidence``: filter out pairs whose contradiction probability is below
      this threshold (default: 0.0, i.e. return all pairs).
    """
    knowledge_store = request.app.state.knowledge_store
    pg_pool = request.app.state.pg_pool

    provenance_store = ProvenanceStore(pg_pool)

    sparql = f"""
        SELECT ?s ?p ?o1 ?o2 ?conf1 ?conf2 WHERE {{
            GRAPH ?g {{
                ?s ?p ?o1 .
                ?s ?p ?o2 .
            }}
            FILTER(?o1 != ?o2 && STR(?o1) < STR(?o2))
            GRAPH ?g {{
                << ?s ?p ?o1 >> <{KS_CONFIDENCE.value}> ?conf1 .
                << ?s ?p ?o2 >> <{KS_CONFIDENCE.value}> ?conf2 .
            }}
        }}
    """

    rows: list[dict] = await asyncio.to_thread(knowledge_store.query, sparql)

    # Pattern B: opposite predicates (e.g., increases vs decreases)
    opposite_sparql = f"""
        SELECT ?s ?p1 ?o ?p2 ?conf1 ?conf2 WHERE {{
            GRAPH ?gont {{
                ?p1 <{KS_OPPOSITE_PREDICATE.value}> ?p2 .
            }}
            GRAPH ?g {{
                ?s ?p1 ?o .
                ?s ?p2 ?o .
            }}
            GRAPH ?g {{
                << ?s ?p1 ?o >> <{KS_CONFIDENCE.value}> ?conf1 .
                << ?s ?p2 ?o >> <{KS_CONFIDENCE.value}> ?conf2 .
            }}
            FILTER(STR(?p1) < STR(?p2))
        }}
    """
    opposite_rows: list[dict] = await asyncio.to_thread(knowledge_store.query, opposite_sparql)

    # Convert opposite-predicate rows to same format as same-predicate rows
    for orow in opposite_rows:
        rows.append(
            {
                "s": orow.get("s"),
                "p": orow.get("p1"),  # use first predicate as the "predicate"
                "o1": orow.get("o"),
                "o2": orow.get("o"),  # same object, different predicates
                "conf1": orow.get("conf1"),
                "conf2": orow.get("conf2"),
                "_opposite_p2": orow.get("p2"),
            }
        )

    results: list[ContradictionResponse] = []
    for row in rows:
        s_str = _rdf_value_to_str(row.get("s"))
        p_str = _rdf_value_to_str(row.get("p"))
        o1_str = _rdf_value_to_str(row.get("o1"))
        o2_str = _rdf_value_to_str(row.get("o2"))
        try:
            conf1 = float(_rdf_value_to_str(row.get("conf1")) or "0.0")
        except (ValueError, TypeError):
            continue
        try:
            conf2 = float(_rdf_value_to_str(row.get("conf2")) or "0.0")
        except (ValueError, TypeError):
            continue

        contradiction_probability = conf1 * conf2

        if contradiction_probability < min_confidence:
            continue

        # For opposite-predicate contradictions, use the second predicate for claim_b
        p2_val = row.get("_opposite_p2")
        p_b_str = _rdf_value_to_str(p2_val) if p2_val else p_str

        # Look up provenance for both triples (use correct predicate for each)
        hash_a = _triple_hash(s_str, p_str, o1_str)
        hash_b = _triple_hash(s_str, p_b_str, o2_str)

        prov_a_rows = await provenance_store.get_by_triple(hash_a)
        prov_b_rows = await provenance_store.get_by_triple(hash_b)

        results.append(
            ContradictionResponse(
                claim_a={
                    "subject": s_str,
                    "predicate": p_str,
                    "object": o1_str,
                    "confidence": conf1,
                },
                claim_b={
                    "subject": s_str,
                    "predicate": p_b_str,
                    "object": o2_str,
                    "confidence": conf2,
                },
                contradiction_probability=contradiction_probability,
                provenance_a=_serialise_provenance(prov_a_rows),
                provenance_b=_serialise_provenance(prov_b_rows),
            )
        )

    return results
