# tests/test_ingestion_worker.py
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from knowledge_service.ingestion.worker import JobTracker, run_ingestion


class TestJobTracker:
    async def test_complete_sets_status(self):
        pool = MagicMock()
        conn = AsyncMock()
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _acquire():
            yield conn

        pool.acquire = _acquire
        tracker = JobTracker("job-id", pool)
        await tracker.complete(triples_created=5, entities_resolved=3, chunks_failed=0)
        conn.execute.assert_called()

    async def test_fail_sets_error(self):
        pool = MagicMock()
        conn = AsyncMock()
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _acquire():
            yield conn

        pool.acquire = _acquire
        tracker = JobTracker("job-id", pool)
        await tracker.fail(Exception("boom"))
        call_args = conn.execute.call_args
        assert "failed" in str(call_args).lower()


def _make_mock_pool():
    """Create a mock asyncpg pool with working acquire context manager."""
    pool = MagicMock()
    conn = AsyncMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool, conn


class TestRunIngestionWithNlp:
    """Test that run_ingestion is now embed-only (NLP/extraction phases are skipped)."""

    async def test_run_ingestion_is_embed_only(self):
        # ponytail: ingest is embed-only — chunk + embed, no graph phases.
        pool, conn = _make_mock_pool()

        stores = MagicMock()
        stores.pg_pool = pool
        stores.content = AsyncMock()
        stores.content.replace_chunks = AsyncMock(return_value=[(0, "chunk-uuid-0")])

        embedding_client = AsyncMock()
        embedding_client.embed_batch = AsyncMock(return_value=[[0.1] * 768])

        chunk_records = [{"chunk_text": "Test sentence.", "chunk_index": 0}]

        await run_ingestion(
            job_id="test-job-id",
            content_id="test-content-id",
            chunk_records=chunk_records,
            stores=stores,
            embedding_client=embedding_client,
        )

        # Job marked complete (not failed)
        calls = conn.execute.call_args_list
        final_sql = str(calls[-1])
        assert "completed" in final_sql.lower()
