"""Tests for thesis API endpoints and admin views."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from knowledge_service.api.theses import router as theses_router
from knowledge_service.admin.theses import router as admin_router


def _make_mock_stores():
    stores = MagicMock()
    stores.theses = AsyncMock()
    stores.theses.create.return_value = "thesis-uuid-1"
    stores.theses.get.return_value = {
        "id": "thesis-uuid-1",
        "name": "Test",
        "description": "Desc",
        "status": "draft",
        "claims": [],
    }
    stores.theses.list.return_value = [{"id": "thesis-uuid-1", "name": "Test", "status": "draft"}]
    stores.theses.find_by_hashes.return_value = []
    return stores


def _make_app(stores=None):
    app = FastAPI()
    app.include_router(theses_router)
    app.include_router(admin_router)
    app.state.stores = stores or _make_mock_stores()
    app.state.extraction_client = None  # No LLM in tests
    return app


@pytest.fixture
async def client():
    app = _make_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestCreateThesis:
    async def test_creates_draft(self, client):
        resp = await client.post(
            "/api/theses", json={"name": "Bull case", "description": "ACME grows 30%"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "thesis-uuid-1"
        assert data["status"] == "draft"

    async def test_with_extraction(self):
        stores = _make_mock_stores()
        extraction_client = AsyncMock()
        extraction_client.decompose_thesis.return_value = [
            {"subject": "acme", "predicate": "grows", "object": "30%", "confidence": 0.8}
        ]
        app = _make_app(stores)
        app.state.extraction_client = extraction_client
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/api/theses", json={"name": "Bull", "description": "ACME grows"})
        data = resp.json()
        assert len(data["claims"]) == 1
        stores.theses.add_claim.assert_called_once()


class TestListTheses:
    async def test_returns_list(self, client):
        resp = await client.get("/api/theses")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1

    async def test_filter_by_status(self, client):
        resp = await client.get("/api/theses?status=active")
        assert resp.status_code == 200


class TestGetThesis:
    async def test_returns_thesis(self, client):
        resp = await client.get("/api/theses/thesis-uuid-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "thesis-uuid-1"

    async def test_404_when_not_found(self, client):
        client._transport.app.state.stores.theses.get.return_value = None
        resp = await client.get("/api/theses/nonexistent")
        assert resp.status_code == 404


class TestPatchThesis:
    async def test_change_status(self, client):
        resp = await client.patch("/api/theses/thesis-uuid-1", json={"status": "active"})
        assert resp.status_code == 200

    async def test_add_claims(self, client):
        resp = await client.patch(
            "/api/theses/thesis-uuid-1",
            json={
                "add_claims": [
                    {"triple_hash": "h1", "subject": "s", "predicate": "p", "object": "o"}
                ]
            },
        )
        assert resp.status_code == 200


class TestDeleteThesis:
    async def test_archives(self, client):
        resp = await client.delete("/api/theses/thesis-uuid-1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "archived"


class TestGetBreaks:
    async def test_returns_breaks(self, client):
        resp = await client.get("/api/theses/thesis-uuid-1/breaks")
        assert resp.status_code == 200
        assert "breaks" in resp.json()


class TestAdminEndpoints:
    async def test_admin_list(self, client):
        resp = await client.get("/api/admin/theses")
        assert resp.status_code == 200
        data = resp.json()
        assert "theses" in data
        assert "count" in data

    async def test_admin_activate(self, client):
        resp = await client.post("/api/admin/theses/thesis-uuid-1/activate")
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    async def test_admin_archive(self, client):
        resp = await client.post("/api/admin/theses/thesis-uuid-1/archive")
        assert resp.status_code == 200
        assert resp.json()["status"] == "archived"
