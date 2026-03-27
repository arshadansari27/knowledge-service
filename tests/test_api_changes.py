"""Tests for GET /api/entity/{entity_id}/changes endpoint."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from knowledge_service.api.changes import router as changes_router


def _make_stores_mock():
    stores = MagicMock()
    stores.provenance = AsyncMock()
    stores.provenance.query_by_entity_and_time.return_value = [
        {
            "triple_hash": "h1",
            "subject": "http://knowledge.local/data/acme",
            "predicate": "http://knowledge.local/schema/revenue",
            "object": "50M",
            "source_url": "http://test.com",
            "source_type": "article",
            "confidence": 0.85,
            "ingested_at": "2026-03-26T10:00:00Z",
        },
    ]
    stores.triples = MagicMock()
    stores.triples.get_triples.return_value = [
        {"confidence": 0.85, "knowledge_type": "Claim", "object": "50M"}
    ]
    stores.theses = AsyncMock()
    stores.theses.find_breaks_for_entity.return_value = []
    return stores


def _create_test_app():
    app = FastAPI()
    app.include_router(changes_router)
    stores = _make_stores_mock()
    app.state.stores = stores
    registry = MagicMock()
    registry.get_materiality.return_value = 0.8
    app.state.domain_registry = registry
    return app


@pytest.fixture
async def client():
    app = _create_test_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestGetChanges:
    async def test_returns_changes(self, client):
        resp = await client.get("/api/entity/acme/changes?since=2026-03-25")
        assert resp.status_code == 200
        data = resp.json()
        assert "changes" in data
        assert len(data["changes"]) >= 1
        assert data["changes"][0]["predicate"] == "http://knowledge.local/schema/revenue"

    async def test_includes_materiality(self, client):
        resp = await client.get("/api/entity/acme/changes?since=2026-03-25")
        data = resp.json()
        assert data["changes"][0]["materiality"] == 0.8

    async def test_includes_thesis_breaks(self, client):
        resp = await client.get("/api/entity/acme/changes?since=2026-03-25")
        data = resp.json()
        assert "thesis_breaks" in data

    async def test_empty_changes(self, client):
        client._transport.app.state.stores.provenance.query_by_entity_and_time.return_value = []
        resp = await client.get("/api/entity/acme/changes?since=2026-03-25")
        assert resp.status_code == 200
        data = resp.json()
        assert data["changes"] == []

    async def test_limit_parameter(self, client):
        resp = await client.get("/api/entity/acme/changes?since=2026-03-25&limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["changes"]) <= 1

    async def test_since_required(self, client):
        resp = await client.get("/api/entity/acme/changes")
        assert resp.status_code == 422
