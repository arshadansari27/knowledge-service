"""POST /api/ask endpoint — RAG-powered question answering."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()

_MAX_QUESTION_LEN = 4000


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=_MAX_QUESTION_LEN)
    max_sources: int = Field(5, ge=1, le=100)
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)


class SourceInfo(BaseModel):
    url: str
    title: str
    source_type: str


class ContradictionInfo(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float | None


class AskResponse(BaseModel):
    answer: str
    confidence: float | None
    sources: list[SourceInfo]
    knowledge_types_used: list[str]
    contradictions: list[ContradictionInfo]


@router.post("/ask", response_model=AskResponse)
async def post_ask(body: AskRequest, request: Request) -> AskResponse:
    """Answer a natural language question using the knowledge base."""
    retriever = request.app.state.rag_retriever
    rag_client = request.app.state.rag_client

    context = await retriever.retrieve(
        body.question,
        max_sources=body.max_sources,
        min_confidence=body.min_confidence,
    )

    try:
        raw_answer = await rag_client.answer(body.question, context)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM service error: {exc}") from exc

    # Confidence: max from knowledge triples, null if none
    confidences = [
        t["confidence"] for t in context.knowledge_triples if t.get("confidence") is not None
    ]
    confidence = max(confidences) if confidences else None

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

    # Knowledge types: computed from retrieval context
    knowledge_types_used = sorted(
        {t["knowledge_type"] for t in context.knowledge_triples if t.get("knowledge_type")}
    )

    # Contradictions
    contradictions = [
        ContradictionInfo(
            subject=c.get("subject", ""),
            predicate=c.get("predicate", ""),
            object=c.get("object", ""),
            confidence=c.get("confidence"),
        )
        for c in context.contradictions
    ]

    return AskResponse(
        answer=raw_answer.answer,
        confidence=confidence,
        sources=sources,
        knowledge_types_used=knowledge_types_used,
        contradictions=contradictions,
    )
