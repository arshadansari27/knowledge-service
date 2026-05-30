"""Tests for report aggregation + summary-table formatting (pure)."""

from __future__ import annotations

import pytest

from knowledge_service.eval.report import QueryResult, aggregate, format_summary_table


def _qr(mode, qtype, **metrics):
    base = {
        "recall_at_k": 0.0,
        "precision_at_k": 0.0,
        "mrr": 0.0,
        "ndcg_at_k": 0.0,
        "faithfulness": 0.0,
        "correctness": 0.0,
    }
    base.update(metrics)
    return QueryResult(query_id=f"{mode}-{qtype}", mode=mode, query_type=qtype, metrics=base)


class TestAggregate:
    def test_averages_within_mode_and_type(self):
        results = [
            _qr("full", "semantic", recall_at_k=1.0, correctness=0.8),
            _qr("full", "semantic", recall_at_k=0.0, correctness=0.4),
        ]
        cell = aggregate(results)[("full", "semantic")]
        assert cell["recall_at_k"] == pytest.approx(0.5)
        assert cell["correctness"] == pytest.approx(0.6)
        assert cell["count"] == 2

    def test_separates_modes(self):
        results = [
            _qr("full", "gtd", recall_at_k=1.0),
            _qr("chunks_only", "gtd", recall_at_k=0.0),
        ]
        agg = aggregate(results)
        assert agg[("full", "gtd")]["recall_at_k"] == 1.0
        assert agg[("chunks_only", "gtd")]["recall_at_k"] == 0.0


class TestFormatSummaryTable:
    def test_table_has_header_and_rows(self):
        agg = aggregate([_qr("full", "semantic", recall_at_k=1.0)])
        table = format_summary_table(agg)
        assert "mode" in table
        assert "query_type" in table
        assert "full" in table
        assert "semantic" in table
