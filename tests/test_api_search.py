"""Integration tests for GET /api/search endpoint.

All external dependencies (PostgreSQL, Ollama) are mocked — no real services required.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from knowledge_service.main import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

_SAMPLE_ROW = {
    "id": "content-uuid-1234",
    "url": "https://example.com/article",
    "title": "Test Article",
    "summary": "A test article summary",
    "source_type": "article",
    "tags": ["python", "testing"],
    "ingested_at": _NOW,
    "similarity": 0.92,
}


def _make_pg_pool_mock(rows: list[dict] | None = None):
    """Build a mock asyncpg pool whose .acquire() works as an async context manager."""
    if rows is None:
        rows = [_SAMPLE_ROW]

    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "SELECT 1"
    mock_conn.fetchrow.return_value = {"id": "content-uuid-1234"}
    mock_conn.fetch.return_value = rows

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    mock_pool = MagicMock()
    mock_pool.acquire = _acquire
    return mock_pool


def _make_knowledge_store_mock():
    mock_ks = MagicMock()
    mock_ks.insert_triple.return_value = "abc123deadbeef"
    mock_ks.find_contradictions.return_value = []
    return mock_ks


def _make_embedding_client_mock():
    """Build a mock EmbeddingClient."""
    mock = AsyncMock()
    mock.embed.return_value = [0.1] * 768
    mock.embed_batch.return_value = [[0.1] * 768]
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """Create test client with all external dependencies mocked."""
    app = create_app(use_lifespan=False)
    app.state.knowledge_store = _make_knowledge_store_mock()
    app.state.pg_pool = _make_pg_pool_mock()
    app.state.embedding_client = _make_embedding_client_mock()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
async def empty_client():
    """Create test client that returns empty search results."""
    app = create_app(use_lifespan=False)
    app.state.knowledge_store = _make_knowledge_store_mock()
    app.state.pg_pool = _make_pg_pool_mock(rows=[])
    app.state.embedding_client = _make_embedding_client_mock()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Tests: basic response structure
# ---------------------------------------------------------------------------


class TestGetSearchBasic:
    async def test_returns_200(self, client):
        response = await client.get("/api/search", params={"q": "test query"})
        assert response.status_code == 200

    async def test_response_is_list(self, client):
        response = await client.get("/api/search", params={"q": "test query"})
        data = response.json()
        assert isinstance(data, list)

    async def test_result_has_required_fields(self, client):
        response = await client.get("/api/search", params={"q": "test query"})
        data = response.json()
        assert len(data) == 1
        result = data[0]
        assert "content_id" in result
        assert "url" in result
        assert "title" in result
        assert "summary" in result
        assert "similarity" in result
        assert "source_type" in result
        assert "tags" in result
        assert "ingested_at" in result

    async def test_result_values_match_row(self, client):
        response = await client.get("/api/search", params={"q": "test query"})
        data = response.json()
        result = data[0]
        assert result["content_id"] == "content-uuid-1234"
        assert result["url"] == "https://example.com/article"
        assert result["title"] == "Test Article"
        assert result["summary"] == "A test article summary"
        assert result["source_type"] == "article"
        assert result["tags"] == ["python", "testing"]


# ---------------------------------------------------------------------------
# Tests: similarity scores
# ---------------------------------------------------------------------------


class TestGetSearchSimilarity:
    async def test_similarity_score_present(self, client):
        response = await client.get("/api/search", params={"q": "test"})
        data = response.json()
        assert len(data) == 1
        assert data[0]["similarity"] == pytest.approx(0.92, abs=1e-6)

    async def test_similarity_is_float(self, client):
        response = await client.get("/api/search", params={"q": "test"})
        data = response.json()
        assert isinstance(data[0]["similarity"], float)

    async def test_multiple_results_have_similarity(self):
        rows = [
            {**_SAMPLE_ROW, "id": "uuid-1", "url": "https://example.com/1", "similarity": 0.95},
            {**_SAMPLE_ROW, "id": "uuid-2", "url": "https://example.com/2", "similarity": 0.80},
        ]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(rows=rows)
        app.state.embedding_client = _make_embedding_client_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.get("/api/search", params={"q": "multiple"})

        data = response.json()
        assert len(data) == 2
        assert data[0]["similarity"] == pytest.approx(0.95, abs=1e-6)
        assert data[1]["similarity"] == pytest.approx(0.80, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: empty results
# ---------------------------------------------------------------------------


class TestGetSearchEmpty:
    async def test_empty_results_return_empty_list(self, empty_client):
        response = await empty_client.get("/api/search", params={"q": "no match"})
        assert response.status_code == 200
        assert response.json() == []

    async def test_empty_results_is_list_type(self, empty_client):
        response = await empty_client.get("/api/search", params={"q": "nothing"})
        assert isinstance(response.json(), list)


# ---------------------------------------------------------------------------
# Tests: query parameter validation
# ---------------------------------------------------------------------------


class TestGetSearchValidation:
    async def test_missing_q_returns_422(self, client):
        response = await client.get("/api/search")
        assert response.status_code == 422

    async def test_default_limit_is_ten(self):
        """Verify embedding client is called and limit defaults to 10."""
        app = create_app(use_lifespan=False)
        mock_ec = _make_embedding_client_mock()
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(rows=[])
        app.state.embedding_client = mock_ec

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.get("/api/search", params={"q": "test"})

        assert response.status_code == 200
        mock_ec.embed.assert_called_once()

    async def test_custom_limit_accepted(self, client):
        response = await client.get("/api/search", params={"q": "test", "limit": 5})
        assert response.status_code == 200

    async def test_source_type_filter_accepted(self, client):
        response = await client.get("/api/search", params={"q": "test", "source_type": "article"})
        assert response.status_code == 200

    async def test_tags_filter_accepted(self, client):
        response = await client.get("/api/search", params={"q": "test", "tags": ["python"]})
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Tests: Ollama embedding call
# ---------------------------------------------------------------------------


class TestGetSearchEmbedding:
    async def test_embedding_client_called_with_query(self):
        """Verify EmbeddingClient.embed() is called with the query text."""
        app = create_app(use_lifespan=False)
        mock_ec = _make_embedding_client_mock()
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(rows=[])
        app.state.embedding_client = mock_ec

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            await c.get("/api/search", params={"q": "semantic query"})

        mock_ec.embed.assert_called_once_with("semantic query")

    async def test_embedding_client_called_once_per_request(self):
        app = create_app(use_lifespan=False)
        mock_ec = _make_embedding_client_mock()
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(rows=[])
        app.state.embedding_client = mock_ec

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            await c.get("/api/search", params={"q": "first"})
            await c.get("/api/search", params={"q": "second"})

        assert mock_ec.embed.call_count == 2


# ---------------------------------------------------------------------------
# Tests: null summary handling
# ---------------------------------------------------------------------------


class TestGetSearchNullSummary:
    async def test_null_summary_allowed(self):
        rows = [{**_SAMPLE_ROW, "summary": None}]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(rows=rows)
        app.state.embedding_client = _make_embedding_client_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.get("/api/search", params={"q": "test"})

        assert response.status_code == 200
        data = response.json()
        assert data[0]["summary"] is None

    async def test_empty_tags_list(self):
        rows = [{**_SAMPLE_ROW, "tags": []}]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(rows=rows)
        app.state.embedding_client = _make_embedding_client_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.get("/api/search", params={"q": "test"})

        assert response.status_code == 200
        data = response.json()
        assert data[0]["tags"] == []
