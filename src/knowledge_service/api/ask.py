"""POST /api/ask endpoint — chunk-only multi-doc synthesis (no knowledge graph)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()

_MAX_QUESTION_LEN = 4000


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=_MAX_QUESTION_LEN)
    max_sources: int = Field(5, ge=1, le=20)
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)


class SourceInfo(BaseModel):
    url: str
    title: str
    source_type: str


class AskResponse(BaseModel):
    answer: str
    confidence: float | None
    sources: list[SourceInfo]
    # ponytail: graph-only fields (knowledge_types_used, contradictions, evidence,
    # intent, traversal_depth) removed. ask is now chunk-only synthesis. These
    # fields will be fully deleted in phase 3 when the graph layer is removed.
    knowledge_types_used: list[str] = []
    contradictions: list = []
    evidence: list = []
    intent: str | None = None
    traversal_depth: int | None = None


@router.post("/ask", response_model=AskResponse)
async def post_ask(body: AskRequest, request: Request) -> AskResponse:
    """Answer a question via chunk-only multi-doc synthesis (no knowledge graph).

    Retrieve top-k chunks, synthesize an answer with citations. No intent
    classification, no graph/triples, no contradictions.
    """
    retriever = request.app.state.rag_retriever
    rag_client = request.app.state.rag_client

    # Retrieve chunks only (no intent, no graph path).
    context = await retriever.retrieve(
        body.question,
        max_sources=body.max_sources,
        min_confidence=body.min_confidence,
        intent=None,
        retrieval_mode="chunks_only",
    )

    try:
        raw_answer = await rag_client.answer(body.question, context)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM service error: {exc}") from exc

    # Sources: deduplicated from content results
    seen_urls: set[str] = set()
    sources: list[SourceInfo] = []
    for row in context.content_results:
        url = row.get("url", "")
        if url not in seen_urls:
            seen_urls.add(url)
            sources.append(
                SourceInfo(
                    url=url,
                    title=row.get("title", ""),
                    source_type=row.get("source_type", ""),
                )
            )

    # ponytail: confidence, knowledge_types, contradictions, evidence are removed.
    # They depended on graph/triples which are now gone. In phase 3, these fields
    # are fully deleted from AskResponse; for now they're empty stubs for backward compat.

    return AskResponse(
        answer=raw_answer.answer,
        confidence=None,
        sources=sources,
    )
