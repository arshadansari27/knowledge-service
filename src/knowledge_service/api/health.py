"""Health check endpoint for the Knowledge Service API."""

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health_check(request: Request):
    """Check the health of all service dependencies.

    Returns overall status ('ok' or 'degraded') and per-component statuses
    for pyoxigraph, PostgreSQL, and the LLM API.
    """
    components = {}

    # Check pyoxigraph
    try:
        request.app.state.knowledge_store.query("SELECT (1 AS ?x) WHERE {}")
        components["oxigraph"] = "ok"
    except Exception as e:
        components["oxigraph"] = f"error: {e}"

    # Check PostgreSQL
    try:
        async with request.app.state.pg_pool.acquire() as conn:
            await conn.execute("SELECT 1")
        components["postgresql"] = "ok"
    except Exception as e:
        components["postgresql"] = f"error: {e}"

    # Check LLM API (via EmbeddingClient's underlying httpx client)
    try:
        resp = await request.app.state.embedding_client._client.get("/v1/models")
        components["llm"] = "ok" if resp.status_code == 200 else f"status: {resp.status_code}"
    except Exception as e:
        components["llm"] = f"error: {e}"

    # Check NLP / spaCy entity linker
    nlp_status = getattr(request.app.state, "nlp_status", None)
    if nlp_status is not None:
        components["nlp"] = nlp_status

    all_ok = all(v == "ok" for v in components.values())
    return {"status": "ok" if all_ok else "degraded", "components": components}
