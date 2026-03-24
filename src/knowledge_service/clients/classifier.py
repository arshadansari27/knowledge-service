"""QueryClassifier — LLM-based question intent classification."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from freeform LLM output.

    Handles markdown fences, thinking tags, and trailing text.
    """
    # Strip markdown code fences
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    # Strip qwen3 thinking tags
    stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL).strip()
    # Try direct parse first
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass
    # Find the first {...} block
    match = re.search(r"\{[^{}]*\}", stripped)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


_VALID_INTENTS = {"semantic", "entity", "graph", "global"}

_CLASSIFICATION_PROMPT = """Classify this question into one category:
- "semantic": searching for documents about a topic (e.g., "find articles about stress management")
- "entity": asking about a specific thing (e.g., "what is dopamine?", "tell me about PostgreSQL")
- "graph": asking about relationships between things (e.g., "how is cortisol connected to inflammation?", "what causes dopamine release?")
- "global": asking about themes, summaries, or overviews across the entire knowledge base (e.g., "what are the main topics?", "summarize what I know about health", "what areas have I collected knowledge on?")

Also extract any named entities mentioned in the question.

Return JSON: {{"intent": "semantic|entity|graph|global", "entities": ["entity1", "entity2"]}}

Question: {question}"""


@dataclass
class QueryIntent:
    """Classified question intent with extracted entity names."""

    intent: str  # "semantic", "entity", "graph", or "global"
    entities: list[str] = field(default_factory=list)


class QueryClassifier:
    """Classifies questions into retrieval intent types via LLM."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self._model = model
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        )

    async def classify(self, question: str) -> QueryIntent:
        """Classify a question into a retrieval intent.

        Returns QueryIntent with intent and extracted entities.
        Falls back to 'semantic' on any failure.
        """
        prompt = _CLASSIFICATION_PROMPT.format(question=question)
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("QueryClassifier: LLM call failed, defaulting to semantic: %s", exc)
            return QueryIntent(intent="semantic")

        raw = response.json()["choices"][0]["message"]["content"]
        parsed = _extract_json(raw)
        if parsed is None:
            logger.warning("QueryClassifier: bad JSON response, defaulting to semantic")
            return QueryIntent(intent="semantic")

        intent = parsed.get("intent", "semantic")
        if intent not in _VALID_INTENTS:
            logger.warning("QueryClassifier: invalid intent '%s', defaulting to semantic", intent)
            intent = "semantic"

        entities = parsed.get("entities", [])
        if not isinstance(entities, list):
            entities = []

        logger.info(
            "QueryClassifier: question='%s' → intent=%s, entities=%s",
            question[:80],
            intent,
            entities,
        )
        return QueryIntent(intent=intent, entities=entities)

    async def close(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()
