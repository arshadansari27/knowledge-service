# tests/test_ingestion_worker.py
from unittest.mock import AsyncMock, MagicMock

from knowledge_service.ingestion.worker import JobTracker


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
