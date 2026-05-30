"""Unit tests for pure retrieval metrics."""

from __future__ import annotations

import math

from knowledge_service.eval.metrics import (
    dcg,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0

    def test_half_relevant_retrieved(self):
        assert recall_at_k(["a", "x"], {"a", "b"}, k=5) == 0.5

    def test_k_truncates(self):
        assert recall_at_k(["x", "b"], {"a", "b"}, k=1) == 0.0

    def test_no_relevant_set_is_zero(self):
        assert recall_at_k(["a"], set(), k=5) == 0.0


class TestPrecisionAtK:
    def test_all_retrieved_relevant(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0

    def test_half_precision(self):
        assert precision_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5

    def test_empty_retrieved_is_zero(self):
        assert precision_at_k([], {"a"}, k=5) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "x"], {"a"}) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a"], {"a"}) == 0.5

    def test_no_hit_is_zero(self):
        assert mrr(["x", "y"], {"a"}) == 0.0


class TestNDCG:
    def test_dcg_single_hit_first(self):
        assert dcg(["a"], {"a"}) == 1.0

    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_reversed_is_less_than_one(self):
        val = ndcg_at_k(["x", "a"], {"a"}, k=2)
        assert 0.0 < val < 1.0
        assert math.isclose(val, (1 / math.log2(3)) / 1.0)
