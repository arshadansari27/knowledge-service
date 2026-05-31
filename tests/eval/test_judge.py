"""Tests for the LLM-as-judge module (HTTP mocked)."""

from __future__ import annotations

import json

from knowledge_service.eval.judge import Judge, JudgeScore, parse_judge_response


class TestParseJudgeResponse:
    def test_parses_scores_and_rationale(self):
        raw = json.dumps({"faithfulness": 0.9, "correctness": 0.8, "rationale": "well grounded"})
        score = parse_judge_response(raw)
        assert score == JudgeScore(faithfulness=0.9, correctness=0.8, rationale="well grounded")

    def test_parses_with_markdown_fence(self):
        raw = '```json\n{"faithfulness": 1, "correctness": 0, "rationale": "x"}\n```'
        score = parse_judge_response(raw)
        assert score.faithfulness == 1.0
        assert score.correctness == 0.0

    def test_clamps_out_of_range(self):
        raw = json.dumps({"faithfulness": 2.5, "correctness": -1, "rationale": ""})
        score = parse_judge_response(raw)
        assert score.faithfulness == 1.0
        assert score.correctness == 0.0

    def test_unparseable_returns_zero_with_rationale(self):
        score = parse_judge_response("the model rambled with no json")
        assert score.faithfulness == 0.0
        assert score.correctness == 0.0
        assert "unparseable" in score.rationale.lower()


class TestJudge:
    async def test_calls_chat_completions_and_returns_score(self, httpx_mock):
        httpx_mock.add_response(
            url="https://litellm.example/v1/chat/completions",
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"faithfulness": 0.7, "correctness": 0.6, "rationale": "ok"}
                            ),
                        }
                    }
                ]
            },
        )
        judge = Judge(base_url="https://litellm.example", model="kimi-k2.5", api_key="k")
        score = await judge.score_one(
            question="What is X?",
            reference_answer="X is a thing.",
            generated_answer="X is a thing.",
            retrieved_context="some context",
        )
        await judge.close()
        assert isinstance(score, JudgeScore)
        assert score.faithfulness == 0.7
        assert score.correctness == 0.6

    async def test_strips_trailing_v1_in_base_url(self, httpx_mock):
        # base_url already ending in /v1 must not become /v1/v1/...
        httpx_mock.add_response(
            url="https://litellm.example/v1/chat/completions",
            json={
                "choices": [
                    {"message": {"content": json.dumps({"faithfulness": 1, "correctness": 1})}}
                ]
            },
        )
        judge = Judge(base_url="https://litellm.example/v1", model="kimi-k2.5", api_key="k")
        score = await judge.score_one(
            question="q", reference_answer="r", generated_answer="g", retrieved_context="c"
        )
        await judge.close()
        assert score.faithfulness == 1.0
