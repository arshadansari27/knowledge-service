import pytest
from unittest.mock import AsyncMock, MagicMock

from knowledge_service.stores.embedding import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_single_list(self):
        results = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        fused = reciprocal_rank_fusion(results, key="id", k=60, limit=3)
        assert len(fused) == 3
        assert fused[0]["id"] == "a"

    def test_two_lists_overlap_ranks_higher(self):
        list1 = [{"id": "shared"}, {"id": "only_vec"}]
        list2 = [{"id": "shared"}, {"id": "only_bm25"}]
        fused = reciprocal_rank_fusion(list1, list2, key="id", k=60, limit=3)
        assert fused[0]["id"] == "shared"

    def test_deduplication(self):
        list1 = [{"id": "x", "source": "vec"}]
        list2 = [{"id": "x", "source": "bm25"}]
        fused = reciprocal_rank_fusion(list1, list2, key="id", k=60, limit=10)
        assert len(fused) == 1

    def test_respects_limit(self):
        list1 = [{"id": str(i)} for i in range(20)]
        fused = reciprocal_rank_fusion(list1, key="id", k=60, limit=5)
        assert len(fused) == 5

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([], [], key="id", k=60, limit=10)
        assert fused == []

    def test_rrf_score_injected(self):
        list1 = [{"id": "a", "similarity": 0.9}]
        list2 = [{"id": "a", "similarity": 0.5}]
        fused = reciprocal_rank_fusion(list1, list2, key="id", k=60, limit=10)
        assert "similarity" in fused[0]
        assert fused[0]["similarity"] == pytest.approx(2.0 / 61, rel=1e-3)


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_ctx
    return pool, conn


@pytest.fixture
def store(mock_pool):
    from knowledge_service.stores.embedding import EmbeddingStore

    pool, _ = mock_pool
    return EmbeddingStore(pool)


class TestSearchBM25:
    async def test_returns_matching_rows(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {
                "id": "chunk-1",
                "chunk_text": "cold exposure increases dopamine",
                "chunk_index": 0,
                "content_id": "meta-1",
                "url": "http://example.com",
                "title": "Test",
                "summary": None,
                "source_type": "article",
                "tags": [],
                "ingested_at": "2026-01-01",
                "similarity": 0.5,
            }
        ]
        results = await store.search_bm25("dopamine", limit=10)
        assert len(results) == 1
        assert results[0]["id"] == "chunk-1"
        sql = conn.fetch.call_args[0][0]
        assert "tsv @@ plainto_tsquery" in sql
        assert "ts_rank" in sql

    async def test_empty_query_returns_empty(self, store, mock_pool):
        results = await store.search_bm25("", limit=10)
        assert results == []

    async def test_stop_word_only_handled(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        results = await store.search_bm25("the", limit=10)
        assert isinstance(results, list)

    async def test_special_characters_handled(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        results = await store.search_bm25("hello!@#$% world", limit=10)
        assert isinstance(results, list)

    async def test_filters_by_source_type(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search_bm25("test", limit=10, source_type="article")
        sql = conn.fetch.call_args[0][0]
        assert "source_type" in sql
