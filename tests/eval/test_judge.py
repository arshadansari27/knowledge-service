"""Tests for the Claude-as-judge module (HTTP mocked)."""

from __future__ import annotations

import json

import pytest

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
    async def test_calls_anthropic_and_returns_score(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.anthropic.com/v1/messages",
            json={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"faithfulness": 0.7, "correctness": 0.6, "rationale": "ok"}
                        ),
                    }
                ]
            },
        )
        judge = Judge(base_url="https://api.anthropic.com", model="claude-opus-4-8", api_key="k")
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

    async def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="api key"):
            Judge(base_url="https://api.anthropic.com", model="claude-opus-4-8", api_key="")
