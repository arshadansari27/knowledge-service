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


