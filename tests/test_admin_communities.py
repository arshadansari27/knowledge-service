"""Tests for admin community rebuild and gaps endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from knowledge_service.admin.communities import router as communities_router
from knowledge_service.admin.stats import router as stats_router


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
def mock_community_store():
    store = AsyncMock()
    store.replace_all.return_value = 3
    store.get_all.return_value = [
        {
            "id": "c1",
            "level": 0,
            "label": "Neuro",
            "member_count": 5,
            "member_entities": ["http://e/a"],
        },
        {
            "id": "c2",
            "level": 0,
            "label": "Solo",
            "member_count": 1,
            "member_entities": ["http://e/b"],
        },
        {
            "id": "c3",
            "level": 1,
            "label": "Science",
            "member_count": 10,
            "member_entities": ["http://e/a"],
        },
    ]
    store.get_member_entities.return_value = {"http://e/a"}
    return store


@pytest.fixture
def communities_app(mock_pg_pool, mock_community_store):
    pool, _conn = mock_pg_pool
    app = FastAPI()
    app.include_router(communities_router, prefix="/api/admin")
    app.include_router(stats_router, prefix="/api/admin")

    ks = MagicMock()
    ks.query.return_value = []

    rag_client = MagicMock()
    rag_client._client = AsyncMock()
    rag_client._model = "test-rag-model"

    app.state.knowledge_store = ks
    app.state.community_store = mock_community_store
    app.state.pg_pool = pool
    app.state.rag_client = rag_client
    return app


@pytest.fixture
async def communities_client(communities_app):
    transport = ASGITransport(app=communities_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestRebuildCommunities:
    async def test_rebuild_triggers_detect_and_store(
        self, communities_client, mock_community_store
    ):
        """Rebuild endpoint should detect communities and store them."""
        with (
            patch("knowledge_service.admin.communities.CommunityDetector") as MockDetector,
            patch("knowledge_service.admin.communities.CommunitySummarizer") as MockSummarizer,
        ):
            MockDetector.return_value.detect.return_value = [
                {"level": 0, "member_entities": ["http://e/a"], "member_count": 1},
                {"level": 1, "member_entities": ["http://e/a", "http://e/b"], "member_count": 2},
            ]
            mock_summarizer_instance = AsyncMock()
            mock_summarizer_instance.summarize_one.side_effect = lambda c: {
                **c,
                "label": "Test",
                "summary": "Test summary",
            }
            MockSummarizer.return_value = mock_summarizer_instance

            resp = await communities_client.post("/api/admin/rebuild-communities")

        assert resp.status_code == 200
        data = resp.json()
        assert data["communities_built"] == 3  # replace_all mock returns 3
        assert "levels" in data
        assert "duration_seconds" in data
        mock_community_store.replace_all.assert_called_once()

    async def test_rebuild_returns_level_breakdown(self, communities_client):
        """Rebuild response should include per-level counts."""
        with (
            patch("knowledge_service.admin.communities.CommunityDetector") as MockDetector,
            patch("knowledge_service.admin.communities.CommunitySummarizer") as MockSummarizer,
        ):
            MockDetector.return_value.detect.return_value = [
                {"level": 0, "member_entities": ["http://e/a"], "member_count": 1},
                {"level": 0, "member_entities": ["http://e/b"], "member_count": 1},
                {"level": 1, "member_entities": ["http://e/a", "http://e/b"], "member_count": 2},
            ]
            mock_summarizer_instance = AsyncMock()
            mock_summarizer_instance.summarize_one.side_effect = lambda c: {
                **c,
                "label": "L",
                "summary": "S",
            }
            MockSummarizer.return_value = mock_summarizer_instance

            resp = await communities_client.post("/api/admin/rebuild-communities")

        data = resp.json()
        assert data["levels"]["level_0"] == 2
        assert data["levels"]["level_1"] == 1


class TestGapsEndpoint:
    async def test_gaps_returns_isolated_entities(
        self, communities_client, mock_pg_pool, mock_community_store
    ):
        """Gaps endpoint should identify entities not in any community."""
        _, conn = mock_pg_pool
        conn.fetch.return_value = [
            {"uri": "http://e/a"},
            {"uri": "http://e/b"},
            {"uri": "http://e/c"},
        ]
        mock_community_store.get_member_entities.return_value = {"http://e/a"}

        resp = await communities_client.get("/api/admin/stats/gaps")
        assert resp.status_code == 200
        data = resp.json()
        assert "http://e/b" in data["isolated_entities"]
        assert "http://e/c" in data["isolated_entities"]
        assert "http://e/a" not in data["isolated_entities"]

    async def test_gaps_returns_thin_communities(self, communities_client, mock_community_store):
        """Gaps endpoint should report communities with <=2 members."""
        resp = await communities_client.get("/api/admin/stats/gaps")
        data = resp.json()
        thin_ids = [t["id"] for t in data["thin_communities"]]
        assert "c2" in thin_ids  # member_count=1

    async def test_gaps_returns_coverage(
        self, communities_client, mock_pg_pool, mock_community_store
    ):
        """Gaps endpoint should report community coverage ratio."""
        _, conn = mock_pg_pool
        conn.fetch.return_value = [
            {"uri": "http://e/a"},
            {"uri": "http://e/b"},
        ]
        mock_community_store.get_member_entities.return_value = {"http://e/a"}

        resp = await communities_client.get("/api/admin/stats/gaps")
        data = resp.json()
        assert data["total_entities"] == 2
        assert data["entities_in_communities"] == 1
        assert data["community_coverage"] == 0.5
