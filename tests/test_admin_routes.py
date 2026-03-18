"""Tests for admin page routes."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from knowledge_service.admin.routes import router as admin_router
from knowledge_service.admin.stats import router as stats_router


@pytest.fixture
def admin_app():
    app = FastAPI()
    app.state.knowledge_store = MagicMock()
    app.state.knowledge_store.query.return_value = []
    pool = AsyncMock()
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=0)
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    app.state.pg_pool = pool
    app.state.embedding_store = None
    app.include_router(admin_router)
    app.include_router(stats_router, prefix="/api/admin")
    return app


@pytest.fixture
async def admin_client(admin_app):
    transport = ASGITransport(app=admin_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_dashboard_renders(admin_client):
    resp = await admin_client.get("/admin")
    assert resp.status_code == 200
    assert "Dashboard" in resp.text


async def test_knowledge_page_renders(admin_client):
    resp = await admin_client.get("/admin/knowledge")
    assert resp.status_code == 200
    assert "Knowledge" in resp.text


async def test_chat_page_renders(admin_client):
    resp = await admin_client.get("/admin/chat")
    assert resp.status_code == 200
    assert "Chat" in resp.text


async def test_contradictions_page_renders(admin_client):
    resp = await admin_client.get("/admin/contradictions")
    assert resp.status_code == 200
    assert "Contradictions" in resp.text


async def test_content_list_renders(admin_client):
    resp = await admin_client.get("/admin/knowledge/content")
    assert resp.status_code == 200
    assert "Content" in resp.text
