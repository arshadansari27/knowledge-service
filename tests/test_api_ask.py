"""Integration tests for POST /api/ask endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from knowledge_service.clients.rag import RAGAnswer
from knowledge_service.main import create_app
from knowledge_service.stores.rag import RetrievalContext
from tests.conftest import make_test_session_cookie


def _make_rag_retriever(context=None):
    if context is None:
        context = RetrievalContext(
            content_results=[
                {
                    "id": "uuid-1",
                    "url": "https://example.com/article",
                    "title": "Test Article",
                    "summary": "A summary",
                    "source_type": "article",
                    "tags": [],
                    "ingested_at": "2026-03-18T10:00:00Z",
                    "similarity": 0.92,
                }
            ],
            knowledge_triples=[
                {
                    "subject": "http://ks/s",
                    "predicate": "http://ks/p",
                    "object": "http://ks/o",
                    "confidence": 0.88,
                    "knowledge_type": "Claim",
                    "valid_from": None,
                    "valid_until": None,
                }
            ],
            contradictions=[],
            entities_found=["http://ks/s"],
        )
    mock = AsyncMock()
    mock.retrieve.return_value = context
    return mock


def _make_rag_client(answer="Test answer", source_urls=None):
    mock = AsyncMock()
    mock.answer.return_value = RAGAnswer(
        answer=answer,
        source_urls_cited=source_urls or ["https://example.com/article"],
    )
    return mock


@pytest.fixture
async def client():
    app = create_app(use_lifespan=False)
    app.state.rag_retriever = _make_rag_retriever()
    app.state.rag_client = _make_rag_client()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", cookies={"ks_session": make_test_session_cookie()}) as c:
        yield c


class TestPostAskBasic:
    async def test_returns_200(self, client):
        response = await client.post("/api/ask", json={"question": "test?"})
        assert response.status_code == 200

    async def test_response_has_answer(self, client):
        response = await client.post("/api/ask", json={"question": "test?"})
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Test answer"

    async def test_response_has_confidence(self, client):
        response = await client.post("/api/ask", json={"question": "test?"})
        data = response.json()
        assert "confidence" in data
        assert data["confidence"] == pytest.approx(0.88)

    async def test_response_has_sources(self, client):
        response = await client.post("/api/ask", json={"question": "test?"})
        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) == 1
        assert data["sources"][0]["url"] == "https://example.com/article"

    async def test_response_has_knowledge_types(self, client):
        response = await client.post("/api/ask", json={"question": "test?"})
        data = response.json()
        assert "knowledge_types_used" in data
        assert "Claim" in data["knowledge_types_used"]

    async def test_response_has_contradictions(self, client):
        response = await client.post("/api/ask", json={"question": "test?"})
        data = response.json()
        assert "contradictions" in data
        assert data["contradictions"] == []


class TestPostAskValidation:
    async def test_missing_question_returns_422(self, client):
        response = await client.post("/api/ask", json={})
        assert response.status_code == 422

    async def test_empty_question_returns_422(self, client):
        response = await client.post("/api/ask", json={"question": ""})
        assert response.status_code == 422

    async def test_question_too_long_returns_422(self, client):
        response = await client.post("/api/ask", json={"question": "x" * 4001})
        assert response.status_code == 422


class TestPostAskLLMFailure:
    async def test_returns_502_on_llm_error(self):
        app = create_app(use_lifespan=False)
        app.state.rag_retriever = _make_rag_retriever()
        failing_client = AsyncMock()
        failing_client.answer.side_effect = Exception("LLM connection refused")
        app.state.rag_client = failing_client

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test", cookies={"ks_session": make_test_session_cookie()}) as c:
            response = await c.post("/api/ask", json={"question": "test?"})

        assert response.status_code == 502


class TestPostAskNullConfidence:
    async def test_null_confidence_when_no_triples(self):
        app = create_app(use_lifespan=False)
        app.state.rag_retriever = _make_rag_retriever(
            context=RetrievalContext(
                content_results=[
                    {
                        "id": "uuid-1",
                        "url": "https://example.com",
                        "title": "T",
                        "summary": "S",
                        "source_type": "article",
                        "tags": [],
                        "ingested_at": "2026-03-18T10:00:00Z",
                        "similarity": 0.9,
                    }
                ],
            )
        )
        app.state.rag_client = _make_rag_client()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test", cookies={"ks_session": make_test_session_cookie()}) as c:
            response = await c.post("/api/ask", json={"question": "q"})

        data = response.json()
        assert data["confidence"] is None
        assert data["knowledge_types_used"] == []


class TestPostAskContradictions:
    async def test_contradictions_in_response(self):
        ctx = RetrievalContext(
            content_results=[],
            knowledge_triples=[
                {
                    "subject": "http://ks/s",
                    "predicate": "http://ks/p",
                    "object": "http://ks/o",
                    "confidence": 0.8,
                    "knowledge_type": "Claim",
                    "valid_from": None,
                    "valid_until": None,
                }
            ],
            contradictions=[
                {
                    "subject": "http://ks/s",
                    "predicate": "http://ks/p2",
                    "object": "http://ks/o2",
                    "confidence": 0.3,
                }
            ],
            entities_found=[],
        )
        app = create_app(use_lifespan=False)
        app.state.rag_retriever = _make_rag_retriever(context=ctx)
        app.state.rag_client = _make_rag_client()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test", cookies={"ks_session": make_test_session_cookie()}) as c:
            response = await c.post("/api/ask", json={"question": "q"})

        data = response.json()
        assert len(data["contradictions"]) == 1
        assert data["contradictions"][0]["subject"] == "http://ks/s"
