"""OpenAI-compatible clients for embeddings and knowledge extraction."""

from __future__ import annotations

import asyncio
import logging

import httpx
from pydantic import TypeAdapter, ValidationError

from knowledge_service.clients.base import BaseLLMClient
from knowledge_service.ontology.registry import DomainRegistry

logger = logging.getLogger(__name__)

# Extraction-call retry policy. Homelab qwen3 is occasionally overloaded during
# batch ingestion bursts (daily briefing); a single 5xx/timeout was previously
# enough to drop a whole chunk's extraction to zero triples. Exponential
# backoff: 1s, 2s between attempts. Keep small — caller already has a 600s
# read timeout, and every retry multiplies worst-case latency.
_EXTRACT_MAX_RETRIES = 2


class LLMClientError(RuntimeError):
    """Raised when the LLM API returns an error or unexpected response."""


class EmbeddingClient(BaseLLMClient):
    """HTTP client wrapping the OpenAI-compatible embeddings API."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        # 120 s read budget. The shared Ollama instance hosting
        # ``nomic-embed-text`` runs alongside other models on a busy node;
        # 32-chunk batch requests routinely exceed 30 s during ingestion
        # bursts. Prod metrics 2026-04 showed ~225 ingestion jobs failing per
        # 14 d in the embedding phase from read timeouts at the previous
        # 30 s budget.
        super().__init__(base_url, model, api_key, read_timeout=120.0)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text."""
        vectors = await self._request([text])
        return vectors[0]

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        When batch_size is set, splits into sub-batches to avoid overwhelming
        the embedding endpoint. Default (None) sends all texts in one request.
        """
        if not texts:
            return []
        if batch_size is None or batch_size >= len(texts):
            return await self._request(texts)
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(await self._request(batch))
        return results

    async def _request(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                "/v1/embeddings",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(f"LLM API returned {exc.response.status_code}") from exc
        except httpx.TimeoutException as exc:
            raise LLMClientError(f"LLM API request timed out: {exc}") from exc
        data = response.json()
        return [item["embedding"] for item in data["data"]]


class ExtractionClient(BaseLLMClient):
    """Single-pass extraction using DomainRegistry for domain-aware prompts.

    Extracts entities, events, and relations in a single LLM call per chunk.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        registry: DomainRegistry | None = None,
    ) -> None:
        super().__init__(base_url, model, api_key, read_timeout=600.0)
        self._registry = registry
        self._prompt_builder = None
        if registry is not None:
            from knowledge_service.clients.prompt_builder import PromptBuilder  # noqa: PLC0415

            self._prompt_builder = PromptBuilder(registry)

    async def _post_chat(self, prompt: str) -> str | None:
        """POST the prompt to /v1/chat/completions with retry on transient errors.

        Retries on HTTP 5xx and timeouts (`_EXTRACT_MAX_RETRIES` extra attempts,
        exponential backoff 1s, 2s, ...). 4xx responses and non-HTTP errors are
        not retried — they represent deterministic failures.

        Returns the raw assistant message content on success, or None if every
        attempt failed.
        """
        last_status: int | None = None
        last_timeout: httpx.TimeoutException | None = None

        for attempt in range(_EXTRACT_MAX_RETRIES + 1):
            try:
                response = await self._client.post(
                    "/v1/chat/completions",
                    json={
                        "model": self._model,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as exc:
                last_status = exc.response.status_code
                if last_status < 500 or attempt >= _EXTRACT_MAX_RETRIES:
                    break
                logger.warning(
                    "ExtractionClient: LLM API returned %s, retrying (attempt %d/%d)",
                    last_status,
                    attempt + 1,
                    _EXTRACT_MAX_RETRIES,
                )
            except httpx.TimeoutException as exc:
                last_timeout = exc
                if attempt >= _EXTRACT_MAX_RETRIES:
                    break
                logger.warning(
                    "ExtractionClient: LLM API timed out, retrying (attempt %d/%d)",
                    attempt + 1,
                    _EXTRACT_MAX_RETRIES,
                )
            await asyncio.sleep(2**attempt)

        if last_status is not None:
            logger.warning("ExtractionClient: LLM API returned %s", last_status)
        elif last_timeout is not None:
            logger.warning("ExtractionClient: LLM API request timed out: %s", last_timeout)
        return None

    async def _call_llm_combined(self, prompt: str) -> list[dict] | None:
        """Send a combined prompt and return merged entity + relation dicts.

        Accepts responses in either format:
        - {"entities": [...], "relations": [...]}  (combined format)
        - {"items": [...]}  (legacy format)
        Returns None on failure.
        """
        raw_text = await self._post_chat(prompt)
        if raw_text is None:
            return None

        from knowledge_service._utils import _extract_json  # noqa: PLC0415

        parsed = _extract_json(raw_text)
        if parsed is None:
            logger.warning("ExtractionClient: could not parse JSON from response")
            return None

        # Accept both combined and legacy format
        if "entities" in parsed or "relations" in parsed:
            entities = parsed.get("entities", [])
            relations = parsed.get("relations", [])
            # qwen3 occasionally returns a dict instead of a list — coerce to list
            if not isinstance(entities, list):
                entities = list(entities.values()) if isinstance(entities, dict) else []
            if not isinstance(relations, list):
                relations = list(relations.values()) if isinstance(relations, dict) else []
            return entities + relations

        return parsed.get("items", [])

    async def extract(
        self,
        text: str,
        title: str | None = None,
        source_type: str | None = None,
        domains: list[str] | None = None,
        entity_hints: list[dict] | None = None,
    ) -> list | None:
        """Extract KnowledgeInput items from raw text using single-pass extraction.

        Extracts entities, events, and relations in a single LLM call.
        Returns None if LLM call failed (distinguishable from [] = nothing found).
        """
        items, _rejected = await self.extract_with_stats(
            text=text,
            title=title,
            source_type=source_type,
            domains=domains,
            entity_hints=entity_hints,
        )
        return items

    async def extract_with_stats(
        self,
        text: str,
        title: str | None = None,
        source_type: str | None = None,
        domains: list[str] | None = None,
        entity_hints: list[dict] | None = None,
    ) -> tuple[list | None, int]:
        """Extract + return (items, rejected_count).

        rejected_count = items the LLM returned that failed KnowledgeInput schema
        validation. This is the signal to look at when chunks_extracted>0 but
        triples_created==0 — it means the model is emitting shapes we can't use.
        """
        from knowledge_service.models import KnowledgeInput  # noqa: PLC0415

        adapter = TypeAdapter(KnowledgeInput)

        if self._prompt_builder:
            prompt = self._prompt_builder.build_combined_prompt(
                text, title, source_type, entity_hints=entity_hints, domains=domains
            )
        else:
            prompt = _build_combined_extraction_prompt_fallback(
                text, title, source_type, entity_hints=entity_hints
            )

        raw = await self._call_llm_combined(prompt)
        if raw is None:
            return None, 0

        items: list = []
        rejected = 0
        for item_dict in raw:
            try:
                items.append(adapter.validate_python(item_dict))
            except ValidationError as exc:
                rejected += 1
                logger.warning(
                    "ExtractionClient: rejected item (schema): %s -- error: %s",
                    item_dict,
                    exc.errors()[:3],
                )

        if raw and not items:
            logger.warning(
                "ExtractionClient: LLM returned %d items, all rejected as schema-invalid",
                len(raw),
            )

        return items, rejected


# ---------------------------------------------------------------------------
# Fallback prompt builders (used when no DomainRegistry is available)
# ---------------------------------------------------------------------------

_MAX_TEXT_CHARS = 4000


CANONICAL_PREDICATES: tuple[str, ...] = (
    "causes",
    "increases",
    "decreases",
    "inhibits",
    "activates",
    "is_a",
    "part_of",
    "located_in",
    "created_by",
    "depends_on",
    "related_to",
    "contains",
    "precedes",
    "follows",
    "has_property",
    "used_for",
    "produced_by",
    "associated_with",
)
"""Canonical predicate names matching ontology/domains/base.ttl.

Single source of truth shared between the LLM relation-extraction fallback
prompt and ``main.py``'s predicate-embedding seed.
"""

_FALLBACK_PREDICATES = ", ".join(CANONICAL_PREDICATES)


def _build_combined_extraction_prompt_fallback(
    text: str,
    title: str | None,
    source_type: str | None,
    entity_hints: list[dict] | None = None,
) -> str:
    """Build a single-pass combined extraction prompt (no DomainRegistry)."""
    context = ""
    if title:
        context += f"Title: {title}\n"
    if source_type:
        context += f"Source type: {source_type}\n"
    if entity_hints:
        context += "\nNLP-detected entities (confirm, correct, or add to these):\n"
        for hint in entity_hints:
            context += f"- {hint['text']} ({hint['label']})\n"
    return f"""{context}You are a knowledge extraction system. Extract entities, events, AND relationships from the text below.
Return ONLY a JSON object: {{"entities": [...], "relations": [...]}}

## Step 1: Extract Entities and Events

Each entity/event item must have a knowledge_type field:
- Entity: uri, rdf_type (e.g. "schema:Person", "schema:Thing"), label, properties (dict), confidence
- Event: subject, occurred_at (YYYY-MM-DD), confidence, properties (dict)

Entity naming rules:
- Use canonical, well-known names: "dopamine" not "the neurotransmitter dopamine"
- Use singular form: "neuron" not "neurons"
- Use lowercase snake_case: "cold_exposure" not "Cold Exposure"
- Be specific: "vitamin_d3" not "vitamin_d" when the text specifies D3
- The uri and label should both use the snake_case form

## Step 2: Extract Relationships Using Those Entities

Each relation item must have a knowledge_type field:
- Claim: subject, predicate, object, object_type, confidence (0.0-0.89)
- Fact: subject, predicate, object, object_type, confidence (0.9-1.0) for verified facts
- Relationship: subject, predicate, object, object_type, confidence

Preferred predicates (use these when applicable):
{_FALLBACK_PREDICATES}
Only invent a new predicate if none of the above fit.

Use entities from Step 1 as subjects and objects. For object values, include object_type ("entity" or "literal"):
- "entity": the object is a thing/concept
- "literal": the object is a measurement, description, or date

Use Claim for uncertain assertions, Fact for high-confidence verifiable statements.

If nothing found, return {{"entities": [], "relations": []}}

Text:
---
{text[:_MAX_TEXT_CHARS]}
---"""
