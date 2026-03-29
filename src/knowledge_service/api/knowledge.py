"""GET /api/knowledge/query and POST /api/knowledge/sparql endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from knowledge_service._utils import (
    _is_uri,
    _triple_hash,
    _rdf_value_to_str,
    sanitize_sparql_string,
)
from knowledge_service.ontology.namespaces import (
    KS,
    KS_CONFIDENCE,
    KS_KNOWLEDGE_TYPE,
    KS_VALID_FROM,
    KS_VALID_UNTIL,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class KnowledgeResult(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float
    knowledge_type: str
    valid_from: str | None = None
    valid_until: str | None = None
    provenance: list[dict] = []
    source: str | None = None


class SparqlQueryBody(BaseModel):
    query: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/knowledge/query", response_model=list[KnowledgeResult])
async def get_knowledge_query(
    request: Request,
    subject: str | None = Query(None, description="Filter by subject URI"),
    predicate: str | None = Query(None, description="Filter by predicate URI"),
    object: str | None = Query(None, alias="object", description="Filter by object URI or literal"),
) -> list[KnowledgeResult]:
    """Query the knowledge graph with optional subject/predicate/object filters.

    At least one parameter must be provided. Results are enriched with
    provenance data from PostgreSQL.
    """
    if subject is None and predicate is None and object is None:
        raise HTTPException(
            status_code=422,
            detail="At least one of subject, predicate, or object must be provided.",
        )

    stores = request.app.state.stores
    triple_store = stores.triples
    provenance_store = stores.provenance

    # Build SPARQL filters
    filters = []
    if subject:
        if not _is_uri(subject):
            raise HTTPException(status_code=422, detail="subject must be a valid URI")
        filters.append(f"FILTER(?s = <{sanitize_sparql_string(subject)}>)")
    if predicate:
        if not _is_uri(predicate):
            raise HTTPException(status_code=422, detail="predicate must be a valid URI")
        filters.append(f"FILTER(?p = <{sanitize_sparql_string(predicate)}>)")
    if object:
        if _is_uri(object):
            filters.append(f"FILTER(?o = <{sanitize_sparql_string(object)}>)")
        else:
            filters.append(f'FILTER(?o = "{sanitize_sparql_string(object)}")')

    sparql = f"""
        SELECT ?s ?p ?o ?conf ?ktype ?vfrom ?vuntil WHERE {{
            GRAPH ?g {{
                ?s ?p ?o .
            }}
            OPTIONAL {{
                GRAPH ?g {{ << ?s ?p ?o >> <{KS_CONFIDENCE.value}> ?conf . }}
            }}
            OPTIONAL {{
                GRAPH ?g {{ << ?s ?p ?o >> <{KS_KNOWLEDGE_TYPE.value}> ?ktype . }}
            }}
            OPTIONAL {{
                GRAPH ?g {{ << ?s ?p ?o >> <{KS_VALID_FROM.value}> ?vfrom . }}
            }}
            OPTIONAL {{
                GRAPH ?g {{ << ?s ?p ?o >> <{KS_VALID_UNTIL.value}> ?vuntil . }}
            }}
            FILTER(BOUND(?conf))
            {" ".join(filters)}
        }}
    """

    try:
        rows: list[dict] = await asyncio.to_thread(triple_store.query, sparql)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"SPARQL query failed: {exc}") from exc

    results: list[KnowledgeResult] = []
    for row in rows:
        s_str = _rdf_value_to_str(row.get("s"))
        p_str = _rdf_value_to_str(row.get("p"))
        o_str = _rdf_value_to_str(row.get("o"))
        conf_val = row.get("conf")
        ktype_val = row.get("ktype")
        vfrom_val = row.get("vfrom")
        vuntil_val = row.get("vuntil")

        confidence = float(_rdf_value_to_str(conf_val)) if conf_val is not None else 0.0
        ktype_str = _rdf_value_to_str(ktype_val) if ktype_val is not None else ""
        knowledge_type = ktype_str[len(KS) :] if ktype_str.startswith(KS) else ktype_str

        # Look up provenance by triple hash
        th = _triple_hash(s_str, p_str, o_str)
        provenance_rows = await provenance_store.get_by_triple(th)

        serialised_provenance: list[dict] = []
        for prow in provenance_rows:
            serialised_provenance.append(
                {
                    k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                    for k, v in prow.items()
                }
            )

        results.append(
            KnowledgeResult(
                subject=s_str,
                predicate=p_str,
                object=o_str,
                confidence=confidence,
                knowledge_type=knowledge_type,
                valid_from=_rdf_value_to_str(vfrom_val) if vfrom_val else None,
                valid_until=_rdf_value_to_str(vuntil_val) if vuntil_val else None,
                provenance=serialised_provenance,
            )
        )

    return results


@router.post("/knowledge/sparql")
async def post_knowledge_sparql(
    body: SparqlQueryBody | None = None,
    request: Request = None,  # type: ignore[assignment]
) -> list[dict]:
    """Execute a raw SPARQL SELECT query against the knowledge graph.

    Accepts either JSON body ``{"query": "..."}`` or raw SPARQL with
    Content-Type ``application/sparql-query``.
    """
    stores = request.app.state.stores
    triple_store = stores.triples

    if body is not None:
        sparql = body.query
    else:
        # Accept raw SPARQL body (application/sparql-query)
        raw = await request.body()
        sparql = raw.decode("utf-8")

    if not sparql or not sparql.strip():
        raise HTTPException(status_code=422, detail="No SPARQL query provided.")

    try:
        rows: list[dict] = await asyncio.to_thread(triple_store.query, sparql)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"SPARQL query failed: {exc}") from exc

    serialised: list[dict] = []
    for row in rows:
        serialised.append({k: _rdf_value_to_str(v) for k, v in row.items()})

    return serialised
