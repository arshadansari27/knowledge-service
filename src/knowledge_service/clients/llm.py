"""OpenAI-compatible clients for embeddings and knowledge extraction."""

from __future__ import annotations

import logging
import re

import httpx
from pydantic import TypeAdapter, ValidationError

from knowledge_service.ontology.namespaces import KS, KS_DATA

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when the LLM API returns an error or unexpected response."""


class EmbeddingClient:
    """HTTP client wrapping the OpenAI-compatible embeddings API."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self._model = model
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self._client = httpx.AsyncClient(
            base_url=url,
            headers=headers,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text."""
        vectors = await self._request([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single request."""
        return await self._request(texts)

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

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if not self._client.is_closed:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# URI normalisation helpers for ExtractionClient
# ---------------------------------------------------------------------------

_KS_ENTITY = KS_DATA
_KS_PREDICATE = KS
_MAX_TEXT_CHARS = 4000

# Canonical predicates and their known synonyms.  Used by the extraction prompt
# to guide the LLM, and by predicate resolution for fast exact-match lookup.
CANONICAL_PREDICATES: list[str] = [
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
]

_CANONICAL_SET: frozenset[str] = frozenset(CANONICAL_PREDICATES)

PREDICATE_SYNONYMS: dict[str, str] = {
    # increases
    "boosts": "increases",
    "enhances": "increases",
    "elevates": "increases",
    "raises": "increases",
    "improves": "increases",
    "upregulates": "increases",
    "amplifies": "increases",
    "stimulates": "increases",
    # decreases
    "reduces": "decreases",
    "lowers": "decreases",
    "diminishes": "decreases",
    "suppresses": "decreases",
    "downregulates": "decreases",
    "attenuates": "decreases",
    # causes
    "leads_to": "causes",
    "results_in": "causes",
    "triggers": "causes",
    "induces": "causes",
    "produces": "causes",
    # inhibits
    "blocks": "inhibits",
    "prevents": "inhibits",
    # activates
    "promotes": "activates",
    "enables": "activates",
    # is_a
    "type_of": "is_a",
    "instance_of": "is_a",
    "kind_of": "is_a",
    # part_of
    "component_of": "part_of",
    "belongs_to": "part_of",
    "member_of": "part_of",
    # located_in
    "based_in": "located_in",
    "situated_in": "located_in",
    # created_by
    "authored_by": "created_by",
    "made_by": "created_by",
    "built_by": "created_by",
    "developed_by": "created_by",
    # depends_on
    "requires": "depends_on",
    "needs": "depends_on",
    # related_to
    "connected_to": "related_to",
    "linked_to": "related_to",
    # contains
    "includes": "contains",
    "has": "contains",
    "comprises": "contains",
    # precedes
    "before": "precedes",
    "prior_to": "precedes",
    # follows
    "after": "follows",
    "subsequent_to": "follows",
}


def resolve_predicate_synonym(predicate: str) -> str:
    """Resolve a predicate to its canonical form via exact synonym lookup.

    Returns the canonical predicate name if found, otherwise the original.
    """
    slug = re.sub(r"[^\w]", "_", predicate.lower().strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if slug in PREDICATE_SYNONYMS:
        return PREDICATE_SYNONYMS[slug]
    if slug in _CANONICAL_SET:
        return slug
    return predicate


def to_entity_uri(value: str) -> str:
    if value.startswith(("http://", "https://", "urn:")):
        return value
    slug = re.sub(r"[^\w]", "_", value.lower().strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return f"{_KS_ENTITY}{slug}"


def to_predicate_uri(value: str) -> str:
    if value.startswith(("http://", "https://", "urn:")):
        return value
    slug = re.sub(r"[^\w]", "_", value.lower().strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return f"{_KS_PREDICATE}{slug}"


def _build_entity_extraction_prompt(text: str, title: str | None, source_type: str | None) -> str:
    """Build the phase-1 prompt that extracts Entity and Event items only."""
    context = ""
    if title:
        context += f"Title: {title}\n"
    if source_type:
        context += f"Source type: {source_type}\n"
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


def _build_relation_extraction_prompt(
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
    predicates_csv = ", ".join(CANONICAL_PREDICATES)
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
{predicates_csv}
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


class ExtractionClient:
    """HTTP client that calls the OpenAI-compatible chat API to extract knowledge items.

    Uses a two-phase extraction strategy:
      Phase 1: Extract Entity and Event items
      Phase 2: Extract relations (Claim, Fact, Relationship, TemporalState, Conclusion)
               constrained to the entities discovered in phase 1
    """

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self._model = model
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self._client = httpx.AsyncClient(
            base_url=url,
            headers=headers,
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=10.0, pool=5.0),
        )

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    @property
    def model(self) -> str:
        return self._model

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
        from knowledge_service._utils import _extract_json

        parsed = _extract_json(raw)
        if parsed is None:
            logger.warning("ExtractionClient: could not parse JSON from response")
            return None

        return parsed.get("items", [])

    async def extract(
        self,
        text: str,
        title: str | None = None,
        source_type: str | None = None,
    ) -> list | None:
        """Extract KnowledgeInput items from raw text using two-phase extraction.

        Phase 1: entities/events. Phase 2: relations constrained to phase 1 entities.
        Returns None if phase 1 LLM call failed (distinguishable from [] = nothing found).
        If phase 2 fails → return phase 1 results only.
        """
        from knowledge_service.models import KnowledgeInput  # noqa: PLC0415

        adapter = TypeAdapter(KnowledgeInput)

        # --- Phase 1: Entity/Event extraction ---
        phase1_prompt = _build_entity_extraction_prompt(text, title, source_type)
        phase1_raw = await self._call_llm(phase1_prompt)
        if phase1_raw is None:
            return None
        if not phase1_raw:
            return []

        phase1_items = []
        for item_dict in phase1_raw:
            try:
                phase1_items.append(adapter.validate_python(item_dict))
            except ValidationError as exc:
                logger.warning("ExtractionClient: skipping invalid item %s: %s", item_dict, exc)

        # Collect entity names from phase 1 results
        entity_names: list[str] = []
        for item in phase1_items:
            kt = item.knowledge_type.value
            if kt == "Entity":
                entity_names.append(item.label)
            elif kt == "Event":
                entity_names.append(item.subject)

        # If no entities found, skip phase 2
        if not entity_names:
            return phase1_items

        # --- Phase 2: Relation extraction constrained to phase 1 entities ---
        phase2_prompt = _build_relation_extraction_prompt(text, title, source_type, entity_names)
        phase2_raw = await self._call_llm(phase2_prompt)
        if not phase2_raw:
            return phase1_items

        phase2_items = []
        for item_dict in phase2_raw:
            try:
                phase2_items.append(adapter.validate_python(item_dict))
            except ValidationError as exc:
                logger.warning("ExtractionClient: skipping invalid item %s: %s", item_dict, exc)

        return phase1_items + phase2_items

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if not self._client.is_closed:
            await self._client.aclose()
