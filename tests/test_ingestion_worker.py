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
    """Test that run_ingestion integrates NLP pre-pass and coreference."""

    async def test_run_ingestion_accepts_nlp_pipeline(self):
        """Verify run_ingestion works with a mock nlp pipeline without crashing."""
        pool, conn = _make_mock_pool()

        # Mock stores
        stores = MagicMock()
        stores.pg_pool = pool
        stores.content = AsyncMock()
        stores.content.delete_chunks = AsyncMock()
        stores.content.insert_chunks = AsyncMock(return_value=[(0, "chunk-uuid-0")])

        # Mock embedding client
        embedding_client = AsyncMock()
        embedding_client.embed_batch = AsyncMock(return_value=[[0.1] * 768])

        # Mock extraction client — returns one entity item
        extraction_client = AsyncMock()
        extraction_client.extract = AsyncMock(
            return_value=[
                {
                    "knowledge_type": "Entity",
                    "uri": "test_entity",
                    "rdf_type": "schema:Thing",
                    "label": "test_entity",
                    "properties": {},
                    "confidence": 0.9,
                }
            ]
        )
        # _call_llm used by CoreferencePhase tier 2
        extraction_client._call_llm = AsyncMock(return_value=None)

        # Mock spaCy nlp pipeline — returns doc with empty ents and one sentence
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_sent = MagicMock()
        mock_doc.sents = iter([mock_sent])
        mock_nlp = MagicMock(return_value=mock_doc)

        chunk_records = [{"chunk_text": "Test sentence.", "chunk_index": 0}]

        await run_ingestion(
            job_id="test-job-id",
            content_id="test-content-id",
            chunk_records=chunk_records,
            raw_text="Test sentence.",
            knowledge=None,
            title="Test",
            source_url="http://example.com",
            source_type="article",
            stores=stores,
            embedding_client=embedding_client,
            extraction_client=extraction_client,
            nlp=mock_nlp,
        )

        # Verify NLP was called on the chunk
        mock_nlp.assert_called_once_with("Test sentence.")

        # Verify extraction was called
        extraction_client.extract.assert_called_once()

        # Verify job was marked complete (not failed)
        # The last execute call should be the complete update
        calls = conn.execute.call_args_list
        final_sql = str(calls[-1])
        assert "completed" in final_sql.lower()
