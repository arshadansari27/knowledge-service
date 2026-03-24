"""Admin stats and knowledge browsing API endpoints."""

from __future__ import annotations

import asyncio
import re

from fastapi import APIRouter, HTTPException, Query, Request

from knowledge_service.ontology.namespaces import (
    KS,
    KS_CONFIDENCE,
    KS_KNOWLEDGE_TYPE,
    KS_VALID_FROM,
    KS_VALID_UNTIL,
)

router = APIRouter()


def _rdf_value_to_str(value: object) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _sanitize_sparql_string(value: str) -> str:
    """Sanitize a string for safe inclusion in SPARQL queries."""
    return re.sub(r'["\\\n\r]', "", value)


@router.get("/stats/counts")
async def get_counts(request: Request) -> dict:
    knowledge_store = request.app.state.knowledge_store
    pg_pool = request.app.state.pg_pool

    sparql = f"""
        SELECT (COUNT(*) AS ?cnt) WHERE {{
            GRAPH ?g {{
                ?s ?p ?o .
            }}
            GRAPH ?g {{
                << ?s ?p ?o >> <{KS_CONFIDENCE.value}> ?conf .
            }}
        }}
    """
    triple_rows = await asyncio.to_thread(knowledge_store.query, sparql)
    triple_count = int(_rdf_value_to_str(triple_rows[0]["cnt"])) if triple_rows else 0

    async with pg_pool.acquire() as conn:
        entity_count = await conn.fetchval("SELECT COUNT(*) FROM entity_embeddings")
        content_count = await conn.fetchval("SELECT COUNT(*) FROM content")
        event_count = await conn.fetchval("SELECT COUNT(*) FROM ingestion_events")

    return {
        "triples": triple_count,
        "entities": entity_count,
        "content": content_count,
        "events": event_count,
    }


@router.get("/stats/confidence")
async def get_confidence_distribution(request: Request) -> dict:
    knowledge_store = request.app.state.knowledge_store

    sparql = f"""
        SELECT ?conf WHERE {{
            GRAPH ?g {{
                ?s ?p ?o .
            }}
            GRAPH ?g {{
                << ?s ?p ?o >> <{KS_CONFIDENCE.value}> ?conf .
            }}
        }}
    """
    rows = await asyncio.to_thread(knowledge_store.query, sparql)

    low = medium = high = 0
    for row in rows:
        conf = float(_rdf_value_to_str(row["conf"]))
        if conf < 0.3:
            low += 1
        elif conf < 0.7:
            medium += 1
        else:
            high += 1

    return {"low": low, "medium": medium, "high": high}


@router.get("/stats/types")
async def get_type_breakdown(request: Request) -> dict:
    knowledge_store = request.app.state.knowledge_store

    sparql = f"""
        SELECT ?ktype (COUNT(*) AS ?cnt) WHERE {{
            GRAPH ?g {{
                ?s ?p ?o .
            }}
            GRAPH ?g {{
                << ?s ?p ?o >> <{KS_KNOWLEDGE_TYPE.value}> ?ktype .
            }}
        }}
        GROUP BY ?ktype
    """
    rows = await asyncio.to_thread(knowledge_store.query, sparql)

    result = {}
    for row in rows:
        ktype = _rdf_value_to_str(row["ktype"])
        if ktype.startswith(KS):
            ktype = ktype[len(KS) :]
        result[ktype] = int(_rdf_value_to_str(row["cnt"]))

    return result


@router.get("/stats/activity")
async def get_recent_activity(request: Request) -> list[dict]:
    pg_pool = request.app.state.pg_pool

    sql = """
        SELECT id, event_type, source, payload->>'url' AS url, created_at
        FROM ingestion_events
        ORDER BY created_at DESC
        LIMIT 20
    """
    async with pg_pool.acquire() as conn:
        rows = await conn.fetch(sql)

    return [
        {
            "id": str(row["id"]),
            "event_type": row["event_type"],
            "source": row["source"],
            "url": row.get("url"),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
    ]


@router.get("/stats/content-items")
async def get_content_items(request: Request) -> list[dict]:
    pg_pool = request.app.state.pg_pool

    sql = """
        SELECT id, url, title, source_type, tags, ingested_at
        FROM content
        ORDER BY ingested_at DESC
        LIMIT 200
    """
    async with pg_pool.acquire() as conn:
        rows = await conn.fetch(sql)

    return [
        {
            "id": str(row["id"]),
            "url": row["url"],
            "title": row["title"],
            "source_type": row["source_type"],
            "tags": row["tags"],
            "ingested_at": row["ingested_at"].isoformat() if row["ingested_at"] else None,
        }
        for row in rows
    ]


@router.get("/knowledge/triples")
async def browse_triples(
    request: Request,
    q: str | None = Query(None, description="Text search across subject/predicate/object"),
    subject: str | None = Query(None, description="Exact subject URI filter"),
    knowledge_type: str | None = Query(None, description="Filter by knowledge type"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    max_confidence: float = Query(1.0, ge=0.0, le=1.0),
    sort: str = Query("recency", pattern="^(recency|confidence)$"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict:
    knowledge_store = request.app.state.knowledge_store

    valid_types = {
        "Claim",
        "Fact",
        "Event",
        "Entity",
        "Relationship",
        "Conclusion",
        "TemporalState",
    }
    if knowledge_type and knowledge_type not in valid_types:
        raise HTTPException(
            status_code=422, detail=f"Invalid knowledge_type. Must be one of: {valid_types}"
        )

    filters = []
    if subject:
        safe_subj = _sanitize_sparql_string(subject)
        filters.append(f'FILTER(STR(?s) = "{safe_subj}")')
    elif q:
        safe_q = _sanitize_sparql_string(q)
        filters.append(
            f'FILTER(CONTAINS(LCASE(STR(?s)), LCASE("{safe_q}")) || '
            f'CONTAINS(LCASE(STR(?p)), LCASE("{safe_q}")) || '
            f'CONTAINS(LCASE(STR(?o)), LCASE("{safe_q}")))'
        )
    if knowledge_type:
        filters.append(f"FILTER(?ktype = <{KS}{knowledge_type}>)")

    filters.append(
        f"FILTER(xsd:float(?conf) >= {min_confidence} && xsd:float(?conf) <= {max_confidence})"
    )

    filter_clause = "\n            ".join(filters)
    order_clause = (
        "ORDER BY DESC(xsd:float(?conf))" if sort == "confidence" else "ORDER BY DESC(?s)"
    )

    count_sparql = f"""
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT (COUNT(*) AS ?cnt) WHERE {{
            GRAPH ?g {{
                ?s ?p ?o .
            }}
            GRAPH ?g {{
                << ?s ?p ?o >> <{KS_CONFIDENCE.value}> ?conf .
            }}
            OPTIONAL {{
                GRAPH ?g {{ << ?s ?p ?o >> <{KS_KNOWLEDGE_TYPE.value}> ?ktype . }}
            }}
            {filter_clause}
        }}
    """

    data_sparql = f"""
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT ?s ?p ?o ?conf ?ktype ?vfrom ?vuntil WHERE {{
            GRAPH ?g {{
                ?s ?p ?o .
            }}
            GRAPH ?g {{
                << ?s ?p ?o >> <{KS_CONFIDENCE.value}> ?conf .
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
            {filter_clause}
        }}
        {order_clause}
        LIMIT {limit}
        OFFSET {offset}
    """

    count_rows, data_rows = await asyncio.gather(
        asyncio.to_thread(knowledge_store.query, count_sparql),
        asyncio.to_thread(knowledge_store.query, data_sparql),
    )

    total = int(_rdf_value_to_str(count_rows[0]["cnt"])) if count_rows else 0

    items = []
    for row in data_rows:
        ktype = _rdf_value_to_str(row.get("ktype"))
        if ktype.startswith(KS):
            ktype = ktype[len(KS) :]

        items.append(
            {
                "subject": _rdf_value_to_str(row.get("s")),
                "predicate": _rdf_value_to_str(row.get("p")),
                "object": _rdf_value_to_str(row.get("o")),
                "confidence": float(_rdf_value_to_str(row.get("conf"))) if row.get("conf") else 0.0,
                "knowledge_type": ktype,
                "valid_from": _rdf_value_to_str(row.get("vfrom")) if row.get("vfrom") else None,
                "valid_until": _rdf_value_to_str(row.get("vuntil")) if row.get("vuntil") else None,
            }
        )

    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/stats/gaps")
async def get_gaps(request: Request):
    """Detect knowledge gaps: isolated entities and thin communities."""
    community_store = getattr(request.app.state, "community_store", None)
    pg_pool = getattr(request.app.state, "pg_pool", None)

    if not community_store or not pg_pool:
        return {"error": "Community or embedding store not available"}

    # All entities in the system (direct SQL, not similarity search)
    async with pg_pool.acquire() as conn:
        rows = await conn.fetch("SELECT uri FROM entity_embeddings")
    all_entities = {r["uri"] for r in rows}

    # Entities in communities
    community_entities = await community_store.get_member_entities()

    isolated = sorted(all_entities - community_entities)

    # Thin communities (<=2 members)
    all_communities = await community_store.get_all()
    thin = [
        {"id": str(c.get("id", "")), "label": c.get("label", ""), "member_count": c["member_count"]}
        for c in all_communities
        if c["member_count"] <= 2
    ]

    total = len(all_entities)
    in_communities = len(all_entities & community_entities)
    coverage = in_communities / total if total > 0 else 0.0

    return {
        "isolated_entities": isolated,
        "thin_communities": thin,
        "total_entities": total,
        "entities_in_communities": in_communities,
        "community_coverage": round(coverage, 2),
    }
