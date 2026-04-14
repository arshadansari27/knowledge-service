"""Tests for async ingestion jobs: models, worker, chunk cap, status endpoint."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient

from knowledge_service.main import create_app
from knowledge_service.models import ContentAcceptedResponse, ContentRequest, IngestionJobStatus
from tests.conftest import make_test_session_cookie


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestContentAcceptedResponse:
    def test_minimal(self):
        r = ContentAcceptedResponse(content_id="abc", job_id="def", chunks_total=5)
        assert r.status == "accepted"
        assert r.chunks_capped_from is None

    def test_with_cap(self):
        r = ContentAcceptedResponse(
            content_id="abc", job_id="def", chunks_total=50, chunks_capped_from=337
        )
        assert r.chunks_capped_from == 337


class TestIngestionJobStatus:
    def test_all_fields(self):
        s = IngestionJobStatus(
            content_id="a",
            job_id="b",
            status="extracting",
            chunks_total=10,
            chunks_embedded=10,
            chunks_extracted=3,
            chunks_failed=0,
            triples_created=5,
            entities_resolved=2,
            error=None,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:01:00Z",
        )
        assert s.status == "extracting"


# ---------------------------------------------------------------------------
# Worker tests
# ---------------------------------------------------------------------------


def _make_worker_stores():
    """Build mock stores for the background worker."""
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "UPDATE 1"
    mock_conn.fetchrow.return_value = None
    mock_conn.fetch.return_value = []

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    mock_pool = MagicMock()
    mock_pool.acquire = _acquire

    stores = MagicMock()
    stores.pg_pool = mock_pool
    stores.triples = MagicMock()
    stores.triples.insert.return_value = ("hash", True)
    stores.triples.find_contradictions.return_value = []
    stores.triples.find_opposite_contradictions.return_value = []
    stores.triples.get_triples.return_value = []
    stores.triples.update_confidence.return_value = None
    stores.content = AsyncMock()
    stores.content.delete_chunks.return_value = None
    stores.content.insert_chunks.return_value = [(0, "chunk-uuid-0")]
    stores.content.replace_chunks.return_value = [(0, "chunk-uuid-0")]
    stores.entities = AsyncMock()
    stores.provenance = AsyncMock()
    stores.provenance.get_by_triple.return_value = []
    stores.provenance.insert.return_value = None
    stores.theses = AsyncMock()
    stores.theses.find_by_hashes.return_value = []

    return stores, mock_conn


class TestIngestionWorker:
    async def test_worker_updates_status_to_completed(self):
        from knowledge_service.ingestion.worker import run_ingestion

        stores, conn = _make_worker_stores()
        embedding_client = AsyncMock()
        embedding_client.embed_batch.return_value = [[0.1] * 768]

        chunks = [
            {
                "chunk_index": 0,
                "chunk_text": "Short text.",
                "char_start": 0,
                "char_end": 11,
                "section_header": None,
            }
        ]

        await run_ingestion(
            job_id="job-1",
            content_id="content-1",
            chunk_records=chunks,
            raw_text="Short text.",
            knowledge=None,
            title="Test",
            source_url="http://test.com",
            source_type="article",
            stores=stores,
            embedding_client=embedding_client,
            extraction_client=None,
            entity_store=stores.entities,
        )

        calls = [str(c) for c in conn.execute.call_args_list]
        statuses = [c for c in calls if "ingestion_jobs" in c and "status" in c]
        assert any("embedding" in s for s in statuses)
        assert any("completed" in s for s in statuses)

    async def test_worker_handles_extraction_failure(self):
        from knowledge_service.ingestion.worker import run_ingestion

        stores, conn = _make_worker_stores()
        embedding_client = AsyncMock()
        embedding_client.embed_batch.return_value = [[0.1] * 768]
        extraction_client = AsyncMock()
        extraction_client.extract.return_value = None  # LLM failure

        chunks = [
            {
                "chunk_index": 0,
                "chunk_text": "Some text.",
                "char_start": 0,
                "char_end": 10,
                "section_header": None,
            }
        ]

        await run_ingestion(
            job_id="job-1",
            content_id="content-1",
            chunk_records=chunks,
            raw_text="Some text.",
            knowledge=None,
            title="Test",
            source_url="http://test.com",
            source_type="article",
            stores=stores,
            embedding_client=embedding_client,
            extraction_client=extraction_client,
            entity_store=stores.entities,
        )

        calls = [str(c) for c in conn.execute.call_args_list]
        assert any("completed" in s for s in calls)

    async def test_worker_marks_job_failed_when_embed_phase_crashes(self):
        """If EmbedPhase.replace_chunks raises, the worker must mark the job
        'failed' so the partial-unique index unblocks re-ingest, and must
        NOT proceed to extract/process phases."""
        from knowledge_service.ingestion.worker import run_ingestion

        stores, conn = _make_worker_stores()
        stores.content.replace_chunks.side_effect = RuntimeError("DB insert crashed")

        embedding_client = AsyncMock()
        embedding_client.embed_batch.return_value = [[0.1] * 768]
        extraction_client = AsyncMock()
        extraction_client.extract = AsyncMock()

        chunks = [
            {
                "chunk_index": 0,
                "chunk_text": "text",
                "char_start": 0,
                "char_end": 4,
                "section_header": None,
            }
        ]

        # run_ingestion swallows the exception via its own try/except
        await run_ingestion(
            job_id="job-1",
            content_id="content-1",
            chunk_records=chunks,
            raw_text="text",
            knowledge=None,
            title="Test",
            source_url="http://test.com",
            source_type="article",
            stores=stores,
            embedding_client=embedding_client,
            extraction_client=extraction_client,
            entity_store=stores.entities,
        )

        executed = [str(c) for c in conn.execute.call_args_list]
        assert any("status = 'failed'" in s for s in executed), (
            f"Expected job to be marked failed; got: {executed}"
        )
        # Extraction must not have been attempted — failure was in embed
        extraction_client.extract.assert_not_called()


# ---------------------------------------------------------------------------
# Chunk cap test
# ---------------------------------------------------------------------------


class TestChunkCap:
    async def test_chunks_capped_at_max(self):
        from knowledge_service.api.content import _MAX_CHUNKS, _accept_content_request

        body = ContentRequest(
            url="http://test.com/big",
            title="Big doc",
            source_type="article",
            raw_text="A" * 300_000,
        )

        mock_conn = AsyncMock()
        mock_conn.fetchrow.side_effect = [
            {"id": "job-uuid"},  # job insert
        ]

        @asynccontextmanager
        async def _acquire():
            yield mock_conn

        mock_pool = MagicMock()
        mock_pool.acquire = _acquire

        mock_content = AsyncMock()
        mock_content.upsert_metadata.return_value = "content-uuid"

        # _accept_content_request now takes (body, stores)
        stores = MagicMock()
        stores.content = mock_content
        stores.pg_pool = mock_pool

        result = await _accept_content_request(body, stores)

        assert not result["conflict"]
        assert result["chunks_total"] == _MAX_CHUNKS
        assert result["chunks_capped_from"] is not None
        assert result["chunks_capped_from"] > _MAX_CHUNKS
        assert len(result["chunk_records"]) == _MAX_CHUNKS


# ---------------------------------------------------------------------------
# Status endpoint tests
# ---------------------------------------------------------------------------


class TestContentStatusEndpoint:
    async def test_returns_latest_job(self):
        app = create_app(use_lifespan=False)
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            "id": "job-uuid",
            "content_id": "content-uuid",
            "status": "extracting",
            "chunks_total": 10,
            "chunks_embedded": 10,
            "chunks_extracted": 3,
            "chunks_failed": 0,
            "triples_created": 0,
            "entities_resolved": 0,
            "error": None,
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }

        @asynccontextmanager
        async def _acquire():
            yield mock_conn

        mock_pool = MagicMock()
        mock_pool.acquire = _acquire

        stores = MagicMock()
        stores.pg_pool = mock_pool
        app.state.stores = stores
        app.state.pg_pool = mock_pool

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            resp = await client.get("/api/content/content-uuid/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "extracting"
        assert data["chunks_embedded"] == 10

    async def test_returns_404_when_no_job(self):
        app = create_app(use_lifespan=False)
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None

        @asynccontextmanager
        async def _acquire():
            yield mock_conn

        mock_pool = MagicMock()
        mock_pool.acquire = _acquire

        stores = MagicMock()
        stores.pg_pool = mock_pool
        app.state.stores = stores
        app.state.pg_pool = mock_pool

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            resp = await client.get("/api/content/nonexistent-uuid/status")

        assert resp.status_code == 404
