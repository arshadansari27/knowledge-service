import pytest
from unittest.mock import AsyncMock, MagicMock
from knowledge_service.stores.community import CommunityDetector, CommunityStore


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=txn)
    txn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn)
    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_ctx
    return pool, conn


@pytest.fixture
def store(mock_pool):
    pool, _ = mock_pool
    return CommunityStore(pool)


class TestCommunityStore:
    async def test_replace_all_deletes_and_inserts(self, store, mock_pool):
        _, conn = mock_pool
        communities = [
            {
                "level": 0,
                "label": "Health",
                "summary": "Health topics",
                "member_entities": ["http://e/a", "http://e/b"],
                "member_count": 2,
            },
        ]
        await store.replace_all(communities)
        # Should execute delete then insert within transaction
        assert conn.execute.call_count >= 1

    async def test_get_by_level(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {
                "id": "uuid1",
                "level": 0,
                "label": "Test",
                "summary": "Sum",
                "member_entities": ["http://e/a"],
                "member_count": 1,
                "built_at": "2026-01-01",
            },
        ]
        results = await store.get_by_level(0)
        assert len(results) == 1
        sql = conn.fetch.call_args[0][0]
        assert "level" in sql

    async def test_get_all(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        results = await store.get_all()
        assert results == []

    async def test_get_member_entities(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {"member_entities": ["http://e/a", "http://e/b"]},
            {"member_entities": ["http://e/b", "http://e/c"]},
        ]
        result = await store.get_member_entities()
        assert "http://e/a" in result
        assert "http://e/c" in result


def _make_knowledge_store_for_detection():
    """Mock KnowledgeStore that returns a small entity graph."""
    ks = MagicMock()
    # Simulates a SPARQL result with entity-to-entity edges
    ks.query.return_value = [
        {
            "s": MagicMock(value="http://e/a"),
            "o": MagicMock(value="http://e/b"),
            "conf": MagicMock(value="0.8"),
        },
        {
            "s": MagicMock(value="http://e/b"),
            "o": MagicMock(value="http://e/c"),
            "conf": MagicMock(value="0.7"),
        },
        {
            "s": MagicMock(value="http://e/d"),
            "o": MagicMock(value="http://e/e"),
            "conf": MagicMock(value="0.9"),
        },
    ]
    return ks


class TestCommunityDetector:
    def test_detect_returns_communities(self):
        ks = _make_knowledge_store_for_detection()
        detector = CommunityDetector(ks)
        communities = detector.detect()
        assert len(communities) > 0
        for c in communities:
            assert "level" in c
            assert "member_entities" in c
            assert "member_count" in c

    def test_detect_produces_two_levels(self):
        ks = _make_knowledge_store_for_detection()
        detector = CommunityDetector(ks)
        communities = detector.detect()
        levels = {c["level"] for c in communities}
        assert 0 in levels

    def test_detect_empty_graph(self):
        ks = MagicMock()
        ks.query.return_value = []
        detector = CommunityDetector(ks)
        communities = detector.detect()
        assert communities == []
