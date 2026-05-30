"""Unit test for the runner's pure scoring seam."""

from __future__ import annotations

from knowledge_service.eval.golden import GoldenItem
from knowledge_service.eval.judge import JudgeScore
from knowledge_service.eval.runner import score_query
from knowledge_service.stores.rag import RetrievalContext


def test_score_query_combines_retrieval_and_judge():
    item = GoldenItem(
        id="q1",
        question="What is X?",
        query_type="entity",
        relevant_source_ids=["content-1"],
        reference_answer="X is a thing.",
    )
    ctx = RetrievalContext(
        content_results=[{"content_id": "content-1", "chunk_text": "X is a thing."}]
    )
    result = score_query(
        item=item,
        mode="full",
        context=ctx,
        generated_answer="X is a thing.",
        judge_score=JudgeScore(faithfulness=0.9, correctness=0.8, rationale="ok"),
        k=5,
    )
    assert result.mode == "full"
    assert result.query_type == "entity"
    assert result.metrics["recall_at_k"] == 1.0
    assert result.metrics["precision_at_k"] == 1.0
    assert result.metrics["faithfulness"] == 0.9
    assert result.metrics["correctness"] == 0.8


def test_score_query_falls_back_to_id_when_no_content_id():
    item = GoldenItem(
        id="q2",
        question="Q",
        query_type="semantic",
        relevant_source_ids=["chunk-9"],
        reference_answer="A",
    )
    ctx = RetrievalContext(content_results=[{"id": "chunk-9", "chunk_text": "A"}])
    result = score_query(
        item=item,
        mode="chunks_only",
        context=ctx,
        generated_answer="A",
        judge_score=JudgeScore(0.0, 0.0, ""),
        k=5,
    )
    assert result.metrics["recall_at_k"] == 1.0
