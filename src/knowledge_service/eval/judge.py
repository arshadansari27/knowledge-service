"""Claude-as-judge: scores generated answers for faithfulness + correctness.

The system-under-test is qwen3; the judge is Claude so the SUT does not grade
itself. Uses the Anthropic Messages API directly (httpx), not the OpenAI shim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from knowledge_service._utils import _extract_json

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """You are a strict evaluator of a question-answering system.

Score the GENERATED ANSWER on two axes from 0.0 to 1.0:
- "faithfulness": is the answer grounded in the RETRIEVED CONTEXT, with no
  fabricated claims? 1.0 = fully grounded, 0.0 = fabricated/unsupported.
- "correctness": does the answer match the REFERENCE ANSWER semantically?
  1.0 = equivalent, 0.0 = wrong or missing.

Return ONLY a JSON object:
{{"faithfulness": <float>, "correctness": <float>, "rationale": "<one sentence>"}}

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RETRIEVED CONTEXT:
{retrieved_context}

GENERATED ANSWER:
{generated_answer}
"""


@dataclass(frozen=True)
class JudgeScore:
    faithfulness: float
    correctness: float
    rationale: str


def _clamp(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def parse_judge_response(raw: str) -> JudgeScore:
    """Parse the judge's JSON reply, tolerating fences/think-tags via _extract_json."""
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return JudgeScore(0.0, 0.0, "unparseable judge response")
    return JudgeScore(
        faithfulness=_clamp(parsed.get("faithfulness")),
        correctness=_clamp(parsed.get("correctness")),
        rationale=str(parsed.get("rationale", "")),
    )


class Judge:
    """Anthropic Messages API client for answer scoring."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        if not api_key:
            raise ValueError("Judge requires an Anthropic api key (eval_judge_api_key)")
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )

    async def score_one(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
        retrieved_context: str,
    ) -> JudgeScore:
        prompt = _JUDGE_PROMPT.format(
            question=question,
            reference_answer=reference_answer,
            retrieved_context=retrieved_context,
            generated_answer=generated_answer,
        )
        resp = await self._client.post(
            "/v1/messages",
            json={
                "model": self._model,
                "max_tokens": 512,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        blocks = resp.json().get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return parse_judge_response(text)

    async def close(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()
