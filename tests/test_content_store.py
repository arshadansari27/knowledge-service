# tests/test_content_store.py
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from knowledge_service.stores.content import ContentStore


def _make_pool():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {"id": "content-uuid-1234"}
    mock_conn.fetchval.return_value = "content-uuid-1234"
    mock_conn.fetch.return_value = []
    mock_conn.execute.return_value = "DELETE 0"

    txn_state = {"entered": False, "committed": False, "rolled_back": False}

    @asynccontextmanager
    async def _transaction():
        txn_state["entered"] = True
        try:
            yield
        except BaseException:
            txn_state["rolled_back"] = True
            raise
        else:
            txn_state["committed"] = True

    mock_conn.transaction = _transaction
    mock_conn.txn_state = txn_state

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


_SAMPLE_CHUNKS = [
    {
        "chunk_index": 0,
        "chunk_text": "hello",
        "embedding": [0.1] * 768,
        "char_start": 0,
        "char_end": 5,
        "section_header": None,
    },
]


class TestReplaceChunks:
    async def test_delete_and_insert_share_a_transaction(self):
        pool, conn = _make_pool()
        conn.fetch.return_value = [{"chunk_index": 0, "id": "chunk-uuid-0"}]
        store = ContentStore(pool)

        pairs = await store.replace_chunks("content-uuid-1234", _SAMPLE_CHUNKS)

        assert pairs == [(0, "chunk-uuid-0")]
        assert conn.txn_state["entered"] is True
        assert conn.txn_state["committed"] is True
        assert conn.txn_state["rolled_back"] is False
        # Delete was issued on the same connection, not via a fresh acquire
        conn.execute.assert_called_once()
        conn.fetch.assert_called_once()

    async def test_insert_failure_rolls_back_delete(self):
        """If the INSERT fails, the DELETE must roll back too — otherwise
        prior chunks would be wiped and provenance.chunk_id set to NULL
        across every triple linked to this content."""
        pool, conn = _make_pool()
        conn.fetch.side_effect = RuntimeError("insert failed")
        store = ContentStore(pool)

        with pytest.raises(RuntimeError, match="insert failed"):
            await store.replace_chunks("content-uuid-1234", _SAMPLE_CHUNKS)

        assert conn.txn_state["entered"] is True
        assert conn.txn_state["rolled_back"] is True
        assert conn.txn_state["committed"] is False

    async def test_empty_chunks_still_deletes_in_transaction(self):
        pool, conn = _make_pool()
        store = ContentStore(pool)

        pairs = await store.replace_chunks("content-uuid-1234", [])

        assert pairs == []
        assert conn.txn_state["entered"] is True
        assert conn.txn_state["committed"] is True
        conn.execute.assert_called_once()
        conn.fetch.assert_not_called()
