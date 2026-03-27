# tests/test_content_store.py
from unittest.mock import AsyncMock, MagicMock
from knowledge_service.stores.content import ContentStore


def _make_pool():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {"id": "content-uuid-1234"}
    mock_conn.fetchval.return_value = "content-uuid-1234"
    mock_conn.fetch.return_value = []
    mock_conn.execute.return_value = "DELETE 0"

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    pool = MagicMock()
    pool.acquire = _acquire
    return pool, mock_conn


class TestUpsertMetadata:
    async def test_returns_content_id(self):
        pool, conn = _make_pool()
        store = ContentStore(pool)
        cid = await store.upsert_metadata(
            url="http://example.com",
            title="Test",
            summary=None,
            raw_text="hello",
            source_type="article",
            tags=None,
            metadata=None,
        )
        assert cid == "content-uuid-1234"
        conn.fetchval.assert_called_once()


class TestInsertChunks:
    async def test_returns_chunk_id_pairs(self):
        pool, conn = _make_pool()
        conn.fetch.return_value = [
            {"chunk_index": 0, "id": "chunk-uuid-0"},
            {"chunk_index": 1, "id": "chunk-uuid-1"},
        ]
        store = ContentStore(pool)
        chunks = [
            {
                "chunk_index": 0,
                "chunk_text": "hello",
                "embedding": [0.1] * 768,
                "char_start": 0,
                "char_end": 5,
                "section_header": None,
            },
            {
                "chunk_index": 1,
                "chunk_text": "world",
                "embedding": [0.2] * 768,
                "char_start": 6,
                "char_end": 11,
                "section_header": None,
            },
        ]
        pairs = await store.insert_chunks("content-uuid-1234", chunks)
        assert len(pairs) == 2
        assert pairs[0] == (0, "chunk-uuid-0")


class TestDeleteChunks:
    async def test_calls_execute(self):
        pool, conn = _make_pool()
        store = ContentStore(pool)
        await store.delete_chunks("content-uuid-1234")
        conn.execute.assert_called_once()
