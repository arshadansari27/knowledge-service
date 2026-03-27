from unittest.mock import AsyncMock, MagicMock
from contextlib import asynccontextmanager
from knowledge_service.stores.theses import ThesisStore


def _make_pool():
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {"id": "thesis-uuid-1"}
    mock_conn.fetchval.return_value = "thesis-uuid-1"
    mock_conn.fetch.return_value = []
    mock_conn.execute.return_value = "INSERT 0 1"

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    pool = MagicMock()
    pool.acquire = _acquire
    return pool, mock_conn


class TestCreate:
    async def test_returns_id(self):
        pool, conn = _make_pool()
        store = ThesisStore(pool)
        tid = await store.create("Bull case ACME", "ACME revenue will grow 30%")
        assert tid == "thesis-uuid-1"


class TestAddClaim:
    async def test_inserts_claim(self):
        pool, conn = _make_pool()
        store = ThesisStore(pool)
        await store.add_claim("thesis-uuid-1", "hash123", "s", "p", "o", "supporting")
        conn.execute.assert_called_once()


class TestRemoveClaim:
    async def test_removes_claim(self):
        pool, conn = _make_pool()
        store = ThesisStore(pool)
        await store.remove_claim("thesis-uuid-1", "hash123")
        conn.execute.assert_called_once()


class TestFindByHashes:
    async def test_returns_matching_theses(self):
        pool, conn = _make_pool()
        conn.fetch.return_value = [
            {
                "thesis_id": "t1",
                "name": "Bull case",
                "triple_hash": "hash123",
                "subject": "s",
                "predicate": "p",
                "object": "o",
                "role": "supporting",
            }
        ]
        store = ThesisStore(pool)
        results = await store.find_by_hashes({"hash123", "hash456"}, status="active")
        assert len(results) == 1
        assert results[0]["thesis_id"] == "t1"

    async def test_empty_when_no_match(self):
        pool, conn = _make_pool()
        conn.fetch.return_value = []
        store = ThesisStore(pool)
        results = await store.find_by_hashes({"hash999"}, status="active")
        assert results == []

    async def test_empty_hashes_returns_empty(self):
        pool, conn = _make_pool()
        store = ThesisStore(pool)
        results = await store.find_by_hashes(set(), status="active")
        assert results == []


class TestList:
    async def test_with_status_filter(self):
        pool, conn = _make_pool()
        conn.fetch.return_value = [{"id": "t1", "name": "Test", "status": "active"}]
        store = ThesisStore(pool)
        results = await store.list(status="active")
        assert len(results) == 1

    async def test_without_filter(self):
        pool, conn = _make_pool()
        conn.fetch.return_value = []
        store = ThesisStore(pool)
        results = await store.list()
        assert results == []
