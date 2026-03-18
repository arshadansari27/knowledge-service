import pytest
from unittest.mock import AsyncMock, MagicMock
from knowledge_service.stores.provenance import ProvenanceStore


@pytest.fixture
def mock_pool():
    """Mock asyncpg pool.

    asyncpg's pool.acquire() is used as ``async with pool.acquire() as conn``.
    That means acquire() must return an object that supports __aenter__ /
    __aexit__ directly (an async context manager), NOT an awaitable.
    We use MagicMock for the pool so acquire() is a regular callable that
    returns an AsyncMock context manager.
    """
    conn = AsyncMock()
    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)

    pool = MagicMock()
    pool.acquire.return_value = acquire_ctx
    return pool, conn


@pytest.fixture
def store(mock_pool):
    pool, _ = mock_pool
    return ProvenanceStore(pool)


class TestInsert:
    async def test_insert_calls_execute_with_upsert(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = None
        await store.insert(
            triple_hash="abc123",
            subject="http://ex.org/s",
            predicate="http://ex.org/p",
            object_="http://ex.org/o",
            source_url="https://example.com/article",
            source_type="article",
            extractor="ollama_llama3",
            confidence=0.7,
        )
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO provenance" in sql
        assert "ON CONFLICT" in sql

    async def test_insert_passes_correct_params(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = None
        await store.insert(
            triple_hash="abc123",
            subject="http://ex.org/s",
            predicate="http://ex.org/p",
            object_="http://ex.org/o",
            source_url="https://example.com/article",
            source_type="article",
            extractor="ollama_llama3",
            confidence=0.7,
        )
        args = conn.execute.call_args[0]
        # Should pass the params after the SQL
        assert "abc123" in args
        assert 0.7 in args

    async def test_insert_with_metadata(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = None
        metadata = {"page": 42, "section": "intro"}
        await store.insert(
            triple_hash="abc123",
            subject="http://ex.org/s",
            predicate="http://ex.org/p",
            object_="http://ex.org/o",
            source_url="https://example.com/article",
            source_type="article",
            extractor="ollama_llama3",
            confidence=0.7,
            metadata=metadata,
        )
        conn.execute.assert_called_once()
        args = conn.execute.call_args[0]
        # metadata dict should appear in the args (serialised as JSON string or dict)
        assert any(metadata == a or (isinstance(a, str) and "page" in a) for a in args)

    async def test_insert_without_metadata_uses_empty_dict(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = None
        await store.insert(
            triple_hash="xyz",
            subject="http://ex.org/s",
            predicate="http://ex.org/p",
            object_="http://ex.org/o",
            source_url="https://example.com",
            source_type="webpage",
            extractor="manual",
            confidence=1.0,
        )
        conn.execute.assert_called_once()
        # Should not raise — default metadata is handled internally


class TestGetByTriple:
    async def test_get_by_triple_queries_correctly(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {
                "triple_hash": "abc",
                "source_url": "https://a.com",
                "confidence": 0.7,
                "source_type": "article",
                "extractor": "manual",
                "ingested_at": "2025-01-01",
            }
        ]
        rows = await store.get_by_triple("abc")
        conn.fetch.assert_called_once()
        assert len(rows) == 1
        assert rows[0]["confidence"] == 0.7

    async def test_get_by_triple_passes_hash_param(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.get_by_triple("deadbeef")
        args = conn.fetch.call_args[0]
        assert "deadbeef" in args

    async def test_get_by_triple_returns_empty_list_when_no_rows(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        rows = await store.get_by_triple("nonexistent")
        assert rows == []


class TestGetBySource:
    async def test_get_by_source_queries_source_url(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {"triple_hash": "abc", "source_url": "https://example.com", "confidence": 0.9}
        ]
        rows = await store.get_by_source("https://example.com")
        conn.fetch.assert_called_once()
        sql = conn.fetch.call_args[0][0]
        assert "source_url" in sql
        assert len(rows) == 1

    async def test_get_by_source_passes_url_param(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.get_by_source("https://example.com/article")
        args = conn.fetch.call_args[0]
        assert "https://example.com/article" in args


class TestQueryByConfidence:
    async def test_filters_by_min_confidence(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [{"triple_hash": "high", "confidence": 0.8}]
        await store.query_by_confidence(min_confidence=0.5)
        conn.fetch.assert_called_once()
        sql = conn.fetch.call_args[0][0]
        assert "confidence" in sql

    async def test_filters_by_max_confidence(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.query_by_confidence(max_confidence=0.5)
        conn.fetch.assert_called_once()
        sql = conn.fetch.call_args[0][0]
        assert "confidence" in sql

    async def test_passes_both_confidence_bounds_as_params(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.query_by_confidence(min_confidence=0.3, max_confidence=0.9)
        args = conn.fetch.call_args[0]
        assert 0.3 in args
        assert 0.9 in args

    async def test_default_bounds_cover_full_range(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.query_by_confidence()
        args = conn.fetch.call_args[0]
        assert 0.0 in args
        assert 1.0 in args


class TestQueryByTime:
    async def test_query_by_time_with_since(self, store, mock_pool):
        from datetime import datetime, timezone

        _, conn = mock_pool
        conn.fetch.return_value = []
        since = datetime(2025, 1, 1, tzinfo=timezone.utc)
        await store.query_by_time(since=since)
        conn.fetch.assert_called_once()
        sql = conn.fetch.call_args[0][0]
        assert "ingested_at" in sql

    async def test_query_by_time_with_until(self, store, mock_pool):
        from datetime import datetime, timezone

        _, conn = mock_pool
        conn.fetch.return_value = []
        until = datetime(2025, 12, 31, tzinfo=timezone.utc)
        await store.query_by_time(until=until)
        conn.fetch.assert_called_once()
        sql = conn.fetch.call_args[0][0]
        assert "ingested_at" in sql

    async def test_query_by_time_passes_datetime_params(self, store, mock_pool):
        from datetime import datetime, timezone

        _, conn = mock_pool
        conn.fetch.return_value = []
        since = datetime(2025, 1, 1, tzinfo=timezone.utc)
        until = datetime(2025, 6, 1, tzinfo=timezone.utc)
        await store.query_by_time(since=since, until=until)
        args = conn.fetch.call_args[0]
        assert since in args
        assert until in args

    async def test_query_by_time_no_filters_returns_all(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [{"triple_hash": "x"}]
        rows = await store.query_by_time()
        conn.fetch.assert_called_once()
        assert len(rows) == 1
