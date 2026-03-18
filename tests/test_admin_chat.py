"""Tests for admin chat functionality."""

import pytest
from unittest.mock import AsyncMock
from dataclasses import dataclass, field

from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from knowledge_service.admin.routes import router as admin_router


@dataclass
class MockRetrievalContext:
    content_results: list = field(default_factory=list)
    knowledge_triples: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)
    entities_found: list = field(default_factory=list)


@dataclass
class MockRawAnswer:
    answer: str = "Test answer about knowledge."


@pytest.fixture
def chat_app():
    app = FastAPI()
    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = MockRetrievalContext(
        content_results=[{"url": "http://example.com", "title": "Test", "source_type": "article"}],
        knowledge_triples=[{"knowledge_type": "Claim", "confidence": 0.8}],
    )
    app.state.rag_retriever = mock_retriever
    mock_rag_client = AsyncMock()
    mock_rag_client.answer.return_value = MockRawAnswer()
    app.state.rag_client = mock_rag_client
    app.include_router(admin_router)
    return app


@pytest.fixture
async def chat_client(chat_app):
    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_chat_page_renders(chat_client):
    resp = await chat_client.get("/admin/chat")
    assert resp.status_code == 200
    assert "Chat" in resp.text


async def test_chat_send_returns_html(chat_client):
    resp = await chat_client.post(
        "/admin/chat/send", data={"question": "What do I know about Python?"}
    )
    assert resp.status_code == 200
    assert "Test answer about knowledge" in resp.text
    assert "text/html" in resp.headers["content-type"]


async def test_chat_send_includes_sources(chat_client):
    resp = await chat_client.post("/admin/chat/send", data={"question": "Tell me something"})
    assert resp.status_code == 200
    assert "Sources" in resp.text or "example.com" in resp.text
