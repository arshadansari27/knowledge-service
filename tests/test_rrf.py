"""PR2: RRF must preserve real cosine similarity and expose rrf_score separately."""

from __future__ import annotations

from knowledge_service.stores.content import reciprocal_rank_fusion


def _row(chunk_id: str, similarity: float) -> dict:
    return {
        "id": chunk_id,
        "chunk_text": f"text for {chunk_id}",
        "similarity": similarity,
        "content_id": "cid",
        "url": "http://example.com",
        "title": "T",
    }


class TestRRF:
    def test_vector_only_hit_preserves_cosine_as_similarity(self):
        vector = [_row("a", 0.87), _row("b", 0.74)]
        bm25: list[dict] = []
        fused = reciprocal_rank_fusion(vector, bm25, key="id", k=60, limit=10)
        by_id = {r["id"]: r for r in fused}
        assert by_id["a"]["similarity"] == 0.87
        assert by_id["b"]["similarity"] == 0.74
        # rrf_score is the fused rank score — distinct from similarity
        assert 0 < by_id["a"]["rrf_score"] < 1
        assert by_id["a"]["bm25_rank"] is None

    def test_bm25_only_hit_has_null_similarity_not_rrf_score(self):
        vector: list[dict] = []
        bm25 = [_row("x", 0.5)]  # ts_rank from BM25 (not a similarity)
        fused = reciprocal_rank_fusion(vector, bm25, key="id", k=60, limit=10)
        assert len(fused) == 1
        # similarity is null because no vector comparison was made.
        # Previous behavior lied by returning 1/61 here.
        assert fused[0]["similarity"] is None
        assert fused[0]["bm25_rank"] == 0
        assert fused[0]["rrf_score"] == 1 / 61

    def test_chunk_in_both_lists_sums_rrf_and_keeps_cosine(self):
        vector = [_row("overlap", 0.9), _row("v_only", 0.5)]
        bm25 = [_row("overlap", 0.3), _row("b_only", 0.2)]
        fused = reciprocal_rank_fusion(vector, bm25, key="id", k=60, limit=10)
        by_id = {r["id"]: r for r in fused}
        # "overlap" is rank 0 in both — rrf score should be 2/(60+1) = 2/61
        assert by_id["overlap"]["rrf_score"] == 2 / 61
        # And it still exposes the real cosine, not the ts_rank
        assert by_id["overlap"]["similarity"] == 0.9
        assert by_id["overlap"]["bm25_rank"] == 0
        # Overlap should win the ranking
        assert fused[0]["id"] == "overlap"

    def test_no_field_is_literally_one_over_sixty_one_for_a_real_cosine(self):
        """Regression guard against the 'every hit has similarity=0.01639' bug.

        Prior implementation overwrote similarity with rrf_score, so every
        single-list hit returned similarity==1/61. This test proves the cosine
        is preserved untouched.
        """
        vector = [_row("solo", 0.42)]
        bm25: list[dict] = []
        fused = reciprocal_rank_fusion(vector, bm25, key="id", k=60, limit=10)
        assert fused[0]["similarity"] == 0.42
        assert fused[0]["similarity"] != 1 / 61
