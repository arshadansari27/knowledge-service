"""Tests for admin stats API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from knowledge_service.admin.stats import router as stats_router


@pytest.fixture
def mock_knowledge_store():
    store = MagicMock()
    store.query.return_value = []
    return store


@pytest.fixture
def mock_pg_pool():
    pool = MagicMock()
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=0)
    conn.fetch = AsyncMock(return_value=[])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = cm
    return pool, conn


@pytest.fixture
def stats_app(mock_knowledge_store, mock_pg_pool):
    pool, _conn = mock_pg_pool
    app = FastAPI()
    app.include_router(stats_router, prefix="/api/admin")
    app.state.knowledge_store = mock_knowledge_store
    app.state.pg_pool = pool
    return app


@pytest.fixture
async def stats_client(stats_app):
    transport = ASGITransport(app=stats_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_counts_endpoint(stats_client, mock_pg_pool):
    _pool, conn = mock_pg_pool
    conn.fetchval.return_value = 42

    resp = await stats_client.get("/api/admin/stats/counts")
    assert resp.status_code == 200
    data = resp.json()
    assert "triples" in data
    assert "entities" in data
    assert "content" in data


async def test_content_items_default(stats_client, mock_pg_pool):
    _pool, conn = mock_pg_pool
    resp = await stats_client.get("/api/admin/stats/content-items")
    assert resp.status_code == 200
    sql, *args = conn.fetch.call_args.args
    assert "WHERE source_type" not in sql
    assert "LIMIT $1" in sql
    assert args == [200]


async def test_content_items_filters_by_source_type(stats_client, mock_pg_pool):
    _pool, conn = mock_pg_pool
    resp = await stats_client.get(
        "/api/admin/stats/content-items",
        params={"source_type": "reference", "limit": 500},
    )
    assert resp.status_code == 200
    sql, *args = conn.fetch.call_args.args
    assert "WHERE source_type = $1" in sql
    assert "LIMIT $2" in sql
    assert args == ["reference", 500]


async def test_content_items_limit_validation(stats_client):
    resp = await stats_client.get("/api/admin/stats/content-items", params={"limit": 5000})
    assert resp.status_code == 422
    resp = await stats_client.get("/api/admin/stats/content-items", params={"limit": 0})
    assert resp.status_code == 422
