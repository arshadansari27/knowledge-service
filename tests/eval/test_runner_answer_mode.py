"""Tests that run_eval threads answer_mode through to the RAG client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_service.eval.golden import GoldenItem
from knowledge_service.eval.runner import run_eval


class TestAnswerModeThreading:
    async def test_run_eval_passes_answer_mode_to_client(self):
        item = GoldenItem(
            id="q1",
            question="What is dopamine?",
            reference_answer="A neurotransmitter.",
            relevant_source_ids=["doc-1"],
            query_type="entity",
        )
        retriever = AsyncMock()
        retriever.classify.return_value = MagicMock(intent="entity")
        retriever.retrieve.return_value = MagicMock(
            content_results=[], knowledge_triples=[], contradictions=[]
        )
        rag_client = AsyncMock()
        rag_client.answer_auto.return_value = MagicMock(answer="generated")
        judge = AsyncMock()
        judge.score_one.return_value = MagicMock(faithfulness=1.0, correctness=1.0, rationale="ok")
        pool = AsyncMock()
        knowledge_store = MagicMock()

        with (
            patch("knowledge_service.eval.runner._build_components") as mock_build,
            patch("knowledge_service.eval.runner.Judge") as mock_judge_cls,
            patch("knowledge_service.eval.runner.load_golden") as mock_load,
        ):
            mock_build.return_value = (pool, knowledge_store, retriever, rag_client, (rag_client,))
            mock_judge_cls.return_value = judge
            mock_load.return_value = [item]
            await run_eval(modes=["full"], k=5, golden_path=MagicMock(), answer_mode="verify")
            rag_client.answer_auto.assert_awaited()
            assert rag_client.answer_auto.await_args.args[2] == "verify"
