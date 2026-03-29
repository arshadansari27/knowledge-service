import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from knowledge_service.main import create_app
from tests.conftest import make_test_session_cookie


def _make_pg_pool_mock():
    """Build a mock asyncpg pool whose .acquire() works as an async context manager."""
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "SELECT 1"

    # asynccontextmanager-based acquire so `async with pool.acquire() as conn` works
    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    mock_pool = MagicMock()
    mock_pool.acquire = _acquire
    return mock_pool


@pytest.fixture
async def client():
    """Create test client with mocked dependencies."""
    app = create_app(use_lifespan=False)

    # Mock app state to avoid needing real services
    app.state.knowledge_store = MagicMock()
    app.state.knowledge_store.query.return_value = [{"x": 1}]

    app.state.pg_pool = _make_pg_pool_mock()

    mock_llm = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_llm.get.return_value = mock_resp
    app.state.embedding_client = MagicMock()
    app.state.embedding_client._client = mock_llm
    app.state.reasoning_engine = MagicMock()

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        cookies={"ks_session": make_test_session_cookie()},
    ) as c:
        yield c


class TestHealth:
    async def test_health_returns_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert "components" in data

    async def test_health_components_present(self, client):
        response = await client.get("/health")
        data = response.json()
        assert "oxigraph" in data["components"]
        assert "postgresql" in data["components"]
        assert "llm" in data["components"]

    async def test_health_all_ok_when_services_healthy(self, client):
        response = await client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["components"]["oxigraph"] == "ok"
        assert data["components"]["postgresql"] == "ok"
        assert data["components"]["llm"] == "ok"

    async def test_health_includes_nlp_when_status_set(self):
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = MagicMock()
        app.state.knowledge_store.query.return_value = [{"x": 1}]
        app.state.pg_pool = _make_pg_pool_mock()
        mock_llm = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_llm.get.return_value = mock_resp
        app.state.embedding_client = MagicMock()
        app.state.embedding_client._client = mock_llm
        app.state.nlp_status = "ok"

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/health")
        data = response.json()
        assert data["components"]["nlp"] == "ok"

    async def test_health_nlp_unavailable_causes_degraded(self):
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = MagicMock()
        app.state.knowledge_store.query.return_value = [{"x": 1}]
        app.state.pg_pool = _make_pg_pool_mock()
        mock_llm = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_llm.get.return_value = mock_resp
        app.state.embedding_client = MagicMock()
        app.state.embedding_client._client = mock_llm
        app.state.nlp_status = "unavailable: spaCy not loaded"

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/health")
        data = response.json()
        assert data["status"] == "degraded"
        assert "unavailable" in data["components"]["nlp"]

    async def test_health_no_nlp_key_when_status_not_set(self, client):
        response = await client.get("/health")
        data = response.json()
        assert "nlp" not in data["components"]

    async def test_health_degraded_when_oxigraph_fails(self):
        app = create_app(use_lifespan=False)

        # oxigraph raises an error
        app.state.knowledge_store = MagicMock()
        app.state.knowledge_store.query.side_effect = RuntimeError("store unavailable")

        app.state.pg_pool = _make_pg_pool_mock()

        mock_llm = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_llm.get.return_value = mock_resp
        app.state.embedding_client = MagicMock()
        app.state.embedding_client._client = mock_llm
        app.state.reasoning_engine = MagicMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/health")
            data = response.json()
            assert data["status"] == "degraded"
            assert data["components"]["oxigraph"].startswith("error:")
