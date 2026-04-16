"""OpenAI-compatible clients for embeddings and knowledge extraction."""

from __future__ import annotations

import logging

import httpx
from pydantic import TypeAdapter, ValidationError

from knowledge_service.clients.base import BaseLLMClient
from knowledge_service.ontology.registry import DomainRegistry

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when the LLM API returns an error or unexpected response."""


class EmbeddingClient(BaseLLMClient):
    """HTTP client wrapping the OpenAI-compatible embeddings API."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        super().__init__(base_url, model, api_key, read_timeout=30.0)

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

    async def _call_llm(self, prompt: str) -> list[dict] | None:
        """Send a prompt to the LLM and return parsed item dicts.

        Returns None on HTTP errors, timeouts, or JSON parse failures (distinguishable
        from an empty list which means "LLM responded but found nothing").
        Raises no exceptions.
        """
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("ExtractionClient: LLM API returned %s", exc.response.status_code)
            return None
        except httpx.TimeoutException as exc:
            logger.warning("ExtractionClient: LLM API request timed out: %s", exc)
            return None

        raw = response.json()["choices"][0]["message"]["content"]
        from knowledge_service._utils import _extract_json  # noqa: PLC0415

        parsed = _extract_json(raw)
        if parsed is None:
            logger.warning(
                "ExtractionClient: could not parse JSON from response (first 200 chars: %s)",
                raw[:200],
            )
            return None

        return parsed.get("items", [])

    async def _call_llm_combined(self, prompt: str) -> list[dict] | None:
        """Send a combined prompt and return merged entity + relation dicts.

        Accepts responses in either format:
        - {"entities": [...], "relations": [...]}  (combined format)
        - {"items": [...]}  (legacy format)
        Returns None on failure.
        """
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("ExtractionClient: LLM API returned %s", exc.response.status_code)
            return None
        except httpx.TimeoutException as exc:
            logger.warning("ExtractionClient: LLM API request timed out: %s", exc)
            return None

        raw_text = response.json()["choices"][0]["message"]["content"]
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

    async def call_raw(self, prompt: str) -> list[dict] | None:
        """Public interface for sending a raw prompt to the LLM.

        Returns parsed item dicts, None on failure. Used by CoreferencePhase.
        """
        return await self._call_llm(prompt)

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

    async def decompose_thesis(self, statement: str) -> list[dict] | None:
        """Decompose a thesis statement into constituent claims."""
        if self._prompt_builder:
            prompt = self._prompt_builder.build_thesis_decomposition_prompt(statement)
        else:
            prompt = (
                f"Decompose the following statement into constituent claims.\n"
                f'Return ONLY a JSON object: {{"items": [...]}}\n\n'
                f"Each item must be a triple with: subject, predicate, object, "
                f"confidence (0.0-1.0)\n\nStatement: {statement}"
            )
        return await self._call_llm(prompt)


# ---------------------------------------------------------------------------
# Fallback prompt builders (used when no DomainRegistry is available)
# ---------------------------------------------------------------------------

_MAX_TEXT_CHARS = 4000


def _build_entity_extraction_prompt_fallback(
    text: str,
    title: str | None,
    source_type: str | None,
    entity_hints: list[dict] | None = None,
) -> str:
    """Build the phase-1 prompt that extracts Entity and Event items only."""
    context = ""
    if title:
        context += f"Title: {title}\n"
    if source_type:
        context += f"Source type: {source_type}\n"
    if entity_hints:
        context += "\nNLP-detected entities (confirm, correct, or add to these):\n"
        for hint in entity_hints:
            context += f"- {hint['text']} ({hint['label']})\n"
    return f"""{context}Extract entities and events from the text below.
Return ONLY a JSON object: {{"items": [...]}}

Each item must have a knowledge_type field. Supported types and required fields:
- Entity: uri, rdf_type (e.g. "schema:Person", "schema:Thing"), label, properties (dict), confidence
- Event: subject, occurred_at (YYYY-MM-DD), confidence, properties (dict)

Entity naming rules:
- Use canonical, well-known names: "dopamine" not "the neurotransmitter dopamine"
- Use singular form: "neuron" not "neurons"
- Use lowercase snake_case: "cold_exposure" not "Cold Exposure"
- Be specific: "vitamin_d3" not "vitamin_d" when the text specifies D3
- The uri and label should both use the snake_case form

Extract every distinct entity and event mentioned. If nothing found, return {{"items": []}}

Example:
Text: "Regular cold water immersion has been shown to increase dopamine levels by up to 250%."
Output: {{"items": [
  {{"knowledge_type": "Entity", "uri": "cold_water_immersion", "rdf_type": "schema:Thing", "label": "cold_water_immersion", "properties": {{}}, "confidence": 0.95}},
  {{"knowledge_type": "Entity", "uri": "dopamine", "rdf_type": "schema:Thing", "label": "dopamine", "properties": {{}}, "confidence": 0.95}}
]}}

Text:
---
{text[:_MAX_TEXT_CHARS]}
---"""


_FALLBACK_PREDICATES = (
    "causes, increases, decreases, inhibits, activates, is_a, part_of, located_in, "
    "created_by, depends_on, related_to, contains, precedes, follows, has_property, "
    "used_for, produced_by, associated_with"
)


def _build_relation_extraction_prompt_fallback(
    text: str,
    title: str | None,
    source_type: str | None,
    entities: list[str],
) -> str:
    """Build the phase-2 prompt that extracts relations constrained to known entities."""
    context = ""
    if title:
        context += f"Title: {title}\n"
    if source_type:
        context += f"Source type: {source_type}\n"
    entity_list = ", ".join(entities)
    return f"""{context}Extract relationships and claims from the text below.
Return ONLY a JSON object: {{"items": [...]}}

Known entities: [{entity_list}]
Prefer these entities as subjects and objects. If the text clearly mentions an entity not in this list, you may use it — follow the entity naming rules below.

Each item must have a knowledge_type field. Supported types and required fields:
- Claim: subject, predicate, object, object_type, confidence (0.0-0.89)
- Fact: subject, predicate, object, object_type, confidence (0.9-1.0) for verified facts
- Relationship: subject, predicate, object, object_type, confidence
- TemporalState: subject, property, value, valid_from (YYYY-MM-DD), valid_until (YYYY-MM-DD), confidence
- Conclusion: concludes (text), derived_from (list of identifiers), inference_method, confidence

Preferred predicates (use these when applicable):
{_FALLBACK_PREDICATES}
Only invent a new predicate if none of the above fit.

Entity naming rules (for consistency with the entity list above):
- Use canonical, well-known names: "dopamine" not "the neurotransmitter dopamine"
- Use singular form: "neuron" not "neurons"
- Use lowercase snake_case: "cold_exposure" not "Cold Exposure"
- Be specific: "vitamin_d3" not "vitamin_d" when the text specifies D3

For object values, include object_type ("entity" or "literal"):
- "entity": the object is a thing/concept from the entity list above
- "literal": the object is a measurement, description, or date (e.g. "250%", "2024-01-15")

Use Claim for uncertain assertions, Fact for high-confidence verifiable statements.
Extract 3-8 items. If nothing found, return {{"items": []}}

Example:
Text: "Regular cold water immersion has been shown to increase dopamine levels by up to 250%."
Known entities: [cold_water_immersion, dopamine]
Output: {{"items": [
  {{"knowledge_type": "Claim", "subject": "cold_water_immersion", "predicate": "increases", "object": "dopamine", "object_type": "entity", "confidence": 0.75}},
  {{"knowledge_type": "Claim", "subject": "cold_water_immersion", "predicate": "has_property", "object": "250% dopamine increase", "object_type": "literal", "confidence": 0.7}}
]}}

Text:
---
{text[:_MAX_TEXT_CHARS]}
---"""


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
- TemporalState: subject, property, value, valid_from (YYYY-MM-DD), valid_until (YYYY-MM-DD), confidence
- Conclusion: concludes (text), derived_from (list of identifiers), inference_method, confidence

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
