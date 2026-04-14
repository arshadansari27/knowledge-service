"""Tests for POST /api/content/upload file upload endpoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from httpx import AsyncClient, ASGITransport

from knowledge_service.main import create_app
from knowledge_service.parsing import ParserRegistry
from knowledge_service.parsing.text import TextParser
from tests.conftest import make_test_session_cookie


# ---------------------------------------------------------------------------
# Helpers
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
    mock.replace_chunks.side_effect = _insert_chunks
    return mock


def _make_app_with_parser_registry():
    """Create test app with parser_registry set."""
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

    # Set up parser registry with text parser
    registry = ParserRegistry()
    registry.register(TextParser())
    content_mod._parser_registry = registry

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFileUpload:
    async def test_upload_text_file(self):
        """Upload a .txt file returns 202."""
        app = _make_app_with_parser_registry()
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            resp = await client.post(
                "/api/content/upload",
                files={"file": ("test.txt", b"Hello world, this is test content.", "text/plain")},
            )

        assert resp.status_code == 202
        data = resp.json()
        assert "content_id" in data
        assert "job_id" in data
        assert data["status"] == "accepted"

    async def test_upload_pdf_file(self):
        """Upload a PDF file returns 202 (with a mock PDF parser)."""
        import knowledge_service.api.content as content_mod

        app = _make_app_with_parser_registry()

        # Add a mock PDF parser to the registry
        mock_pdf_parser = AsyncMock()
        mock_pdf_parser.supported_formats = {"pdf"}
        mock_pdf_parser.parse.return_value = MagicMock(
            text="Extracted PDF content here",
            title="PDF Title",
            metadata={},
            source_format="pdf",
        )
        content_mod._parser_registry.register(mock_pdf_parser)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            # Send bytes that look like a PDF header
            resp = await client.post(
                "/api/content/upload",
                files={"file": ("sample.pdf", b"%PDF-1.4 fake content", "application/pdf")},
            )

        assert resp.status_code == 202
        data = resp.json()
        assert "content_id" in data

    async def test_upload_too_large_returns_413(self):
        """File exceeding max_upload_size returns 413."""
        app = _make_app_with_parser_registry()
        transport = ASGITransport(app=app)

        # Create data larger than 50MB
        large_data = b"x" * (50 * 1024 * 1024 + 1)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            resp = await client.post(
                "/api/content/upload",
                files={"file": ("big.txt", large_data, "text/plain")},
            )

        assert resp.status_code == 413
        data = resp.json()
        assert "detail" in data

    async def test_upload_with_metadata(self):
        """Upload with title and source_type metadata returns 202."""
        app = _make_app_with_parser_registry()
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            resp = await client.post(
                "/api/content/upload",
                files={"file": ("notes.txt", b"Some research notes content.", "text/plain")},
                data={
                    "title": "My Research Notes",
                    "source_type": "research",
                    "tags": '["science", "notes"]',
                    "url": "https://example.com/notes",
                },
            )

        assert resp.status_code == 202
        data = resp.json()
        assert "content_id" in data
        assert data["status"] == "accepted"
