"""QueryClassifier — LLM-based question intent classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from knowledge_service._utils import _extract_json
from knowledge_service.clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


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


class QueryClassifier(BaseLLMClient):
    """Classifies questions into retrieval intent types via LLM."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        super().__init__(base_url, model, api_key, read_timeout=30.0)

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
