"""Admin page routes — serves Jinja2 templates for the admin panel."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

_TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))


@router.get("/admin", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "active": "dashboard"})


@router.get("/admin/knowledge", response_class=HTMLResponse)
async def knowledge_explorer(request: Request):
    return templates.TemplateResponse("knowledge.html", {"request": request, "active": "knowledge"})


@router.get("/admin/knowledge/entity/{uri:path}", response_class=HTMLResponse)
async def entity_detail(request: Request, uri: str):
    uri = unquote(uri)
    embedding_store = getattr(request.app.state, "embedding_store", None)
    entity_info = None
    label = uri.split("/")[-1]
    if embedding_store:
        entity_info = await embedding_store.get_entity_by_uri(uri)
        if entity_info:
            label = entity_info.get("label", label)
    return templates.TemplateResponse(
        "entity.html",
        {"request": request, "active": "knowledge", "uri": uri, "label": label, "entity_info": entity_info},
    )


@router.get("/admin/knowledge/content", response_class=HTMLResponse)
async def content_list(request: Request):
    return templates.TemplateResponse("content_list.html", {"request": request, "active": "knowledge"})


@router.get("/admin/knowledge/content/{content_id}", response_class=HTMLResponse)
async def content_detail(request: Request, content_id: str):
    import uuid as _uuid

    try:
        cid = _uuid.UUID(content_id)
    except ValueError:
        return HTMLResponse("<h1>Not Found</h1>", status_code=404)

    pg_pool = request.app.state.pg_pool
    async with pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, url, title, summary, raw_text, source_type, tags, ingested_at FROM content WHERE id = $1",
            cid,
        )
    if row is None:
        return HTMLResponse("<h1>Not Found</h1>", status_code=404)
    content = dict(row)
    content["id"] = str(content["id"])
    content["ingested_at"] = content["ingested_at"].isoformat() if content["ingested_at"] else ""
    return templates.TemplateResponse(
        "content_detail.html",
        {"request": request, "active": "knowledge", "content": content},
    )


@router.get("/admin/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "active": "chat"})


@router.get("/admin/contradictions", response_class=HTMLResponse)
async def contradictions_page(request: Request):
    return templates.TemplateResponse("contradictions.html", {"request": request, "active": "contradictions"})


@router.post("/admin/chat/send", response_class=HTMLResponse)
async def chat_send(request: Request):
    """Process a chat question and return an HTML partial with the answer."""
    form = await request.form()
    question = str(form.get("question", "")).strip()
    if not question:
        return HTMLResponse('<div class="text-red-400 p-2">Please enter a question.</div>')

    retriever = request.app.state.rag_retriever
    rag_client = request.app.state.rag_client

    context = await retriever.retrieve(question, max_sources=5, min_confidence=0.0)
    raw_answer = await rag_client.answer(question, context)

    seen_urls: set[str] = set()
    sources = []
    for row in context.content_results:
        url = row.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            sources.append({
                "url": url,
                "title": row.get("title", ""),
                "source_type": row.get("source_type", ""),
            })

    confidences = [t["confidence"] for t in context.knowledge_triples if t.get("confidence") is not None]
    confidence = max(confidences) if confidences else None
    knowledge_types = sorted({t["knowledge_type"] for t in context.knowledge_triples if t.get("knowledge_type")})

    return templates.TemplateResponse(
        "partials/chat_message.html",
        {
            "request": request,
            "answer": raw_answer.answer,
            "confidence": confidence,
            "sources": sources,
            "knowledge_types": knowledge_types,
        },
    )
