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


async def test_confidence_endpoint(stats_client):
    resp = await stats_client.get("/api/admin/stats/confidence")
    assert resp.status_code == 200
    data = resp.json()
    assert "low" in data
    assert "medium" in data
    assert "high" in data


async def test_types_endpoint(stats_client):
    resp = await stats_client.get("/api/admin/stats/types")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


async def test_triples_browse_endpoint(stats_client):
    resp = await stats_client.get("/api/admin/knowledge/triples")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data


async def test_triples_browse_with_filters(stats_client):
    resp = await stats_client.get(
        "/api/admin/knowledge/triples",
        params={"q": "test", "knowledge_type": "Claim", "min_confidence": 0.5, "limit": 10},
    )
    assert resp.status_code == 200


async def test_triples_browse_no_duplicates(stats_client):
    """browse_triples must not return duplicate rows for the same triple."""
    resp = await stats_client.get("/api/admin/knowledge/triples")
    assert resp.status_code == 200
    data = resp.json()
    items = data["items"]
    # Check uniqueness by (subject, predicate, object) tuple
    seen = set()
    for item in items:
        key = (item["subject"], item["predicate"], item["object"])
        assert key not in seen, f"Duplicate triple in browse results: {key}"
        seen.add(key)
