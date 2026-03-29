"""Tests for URL auto-fetch in POST /api/content endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport

from knowledge_service.main import create_app
from knowledge_service.parsing import ParserRegistry
from knowledge_service.parsing.text import TextParser
from tests.conftest import make_test_session_cookie


# ---------------------------------------------------------------------------
# Helpers (mirrors test_api_content.py patterns)
# ---------------------------------------------------------------------------


def _make_pg_pool_mock():
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "INSERT 0 1"

    async def _fetchrow(sql, *args):
        if "ingestion_jobs" in sql and "INSERT" in sql:
            return {"id": "job-uuid-1234"}
        return {"id": "content-uuid-1234"}

    mock_conn.fetchrow.side_effect = _fetchrow
    mock_conn.fetch.return_value = []

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    mock_pool = MagicMock()
    mock_pool.acquire = _acquire
    return mock_pool


def _make_content_store_mock():
    mock = AsyncMock()
    mock.upsert_metadata.return_value = "content-uuid-1234"
    mock.delete_chunks.return_value = None

    async def _insert_chunks(content_id, chunks):
        return [(c["chunk_index"], f"chunk-uuid-{c['chunk_index']}") for c in chunks]

    mock.insert_chunks.side_effect = _insert_chunks
    return mock


def _make_app_with_mocks(parser_registry=None):
    """Create test app with mocked stores and optional parser_registry."""
    import knowledge_service.api.content as content_mod

    app = create_app(use_lifespan=False)

    mock_pg = _make_pg_pool_mock()
    mock_content = _make_content_store_mock()

    stores = MagicMock()
    stores.triples = MagicMock()
    stores.triples.insert.return_value = ("abc123", True)
    stores.triples.find_contradictions.return_value = []
    stores.triples.find_opposite_contradictions.return_value = []
    stores.triples.get_triples.return_value = []
    stores.content = mock_content
    stores.entities = AsyncMock()
    stores.provenance = AsyncMock()
    stores.provenance.get_by_triple.return_value = []
    stores.provenance.insert.return_value = None
    stores.theses = AsyncMock()
    stores.theses.find_by_hashes.return_value = []
    stores.pg_pool = mock_pg
    app.state.stores = stores

    mock_ec = AsyncMock()
    mock_ec.embed.return_value = [0.1] * 768
    mock_ec.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]
    app.state.embedding_client = mock_ec
    app.state.extraction_client = AsyncMock()
    app.state.extraction_client.extract.return_value = []
    app.state.knowledge_store = stores.triples
    app.state.pg_pool = mock_pg
    app.state.reasoning_engine = None

    # Set module-level parser_registry
    content_mod._parser_registry = parser_registry

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestURLAutoFetch:
    async def test_url_with_raw_text_skips_fetch(self):
        """When raw_text is provided, URL fetch should not happen — returns 202."""
        registry = ParserRegistry()
        registry.register(TextParser())
        app = _make_app_with_mocks(parser_registry=registry)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            resp = await client.post(
                "/api/content",
                json={
                    "url": "https://example.com/article",
                    "title": "Test",
                    "raw_text": "Existing text content",
                    "source_type": "article",
                },
            )

        assert resp.status_code == 202
        data = resp.json()
        assert "content_id" in data

    async def test_url_without_raw_text_returns_422_on_fetch_fail(self):
        """When URL fetch fails (network error), return 422."""
        registry = ParserRegistry()
        registry.register(TextParser())
        app = _make_app_with_mocks(parser_registry=registry)

        # Patch httpx.AsyncClient to simulate a network failure
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Connection refused")

        mock_http_client = AsyncMock()
        mock_http_client.get.return_value = mock_response
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=False)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            with patch(
                "knowledge_service.api.content.httpx.AsyncClient", return_value=mock_http_client
            ):
                resp = await client.post(
                    "/api/content",
                    json={
                        "url": "https://example.com/article.txt",
                        "title": "Test",
                        "source_type": "article",
                    },
                )

        assert resp.status_code == 422
        data = resp.json()
        assert "detail" in data

    async def test_url_without_raw_text_no_registry_falls_through(self):
        """When no parser_registry is set, URL fetch is skipped gracefully."""
        app = _make_app_with_mocks(parser_registry=None)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            resp = await client.post(
                "/api/content",
                json={
                    "url": "https://example.com/article",
                    "title": "Test",
                    "source_type": "article",
                },
            )

        # Without parser_registry, no fetch attempted; uses title as text fallback
        assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Tests: SSRF protection
# ---------------------------------------------------------------------------


class TestIsUrlSafe:
    """Unit tests for the _is_url_safe helper."""

    def test_blocks_localhost(self):
        from knowledge_service.api.content import _is_url_safe

        assert _is_url_safe("http://localhost/secret") is False
        assert _is_url_safe("http://127.0.0.1/secret") is False

    def test_blocks_private_ips(self):
        from knowledge_service.api.content import _is_url_safe

        assert _is_url_safe("http://10.0.0.1/metadata") is False
        assert _is_url_safe("http://192.168.1.1/admin") is False
        assert _is_url_safe("http://169.254.169.254/latest/meta-data/") is False
        assert _is_url_safe("http://172.16.0.1/internal") is False

    def test_blocks_zero_address(self):
        from knowledge_service.api.content import _is_url_safe

        assert _is_url_safe("http://0.0.0.0/") is False

    def test_allows_public_urls(self):
        from knowledge_service.api.content import _is_url_safe

        assert _is_url_safe("https://example.com/article") is True
        assert _is_url_safe("https://en.wikipedia.org/wiki/Test") is True

    def test_rejects_empty_url(self):
        from knowledge_service.api.content import _is_url_safe

        assert _is_url_safe("") is False
