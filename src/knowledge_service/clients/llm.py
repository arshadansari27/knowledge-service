"""OpenAI-compatible clients for embeddings and knowledge extraction."""

from __future__ import annotations

import json
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
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
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
    if slug in CANONICAL_PREDICATES:
        return slug
    return predicate


def _to_entity_uri(value: str) -> str:
    if value.startswith(("http://", "https://", "urn:")):
        return value
    slug = re.sub(r"[^\w]", "_", value.lower().strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return f"{_KS_ENTITY}{slug}"


def _to_predicate_uri(value: str) -> str:
    if value.startswith(("http://", "https://", "urn:")):
        return value
    slug = re.sub(r"[^\w]", "_", value.lower().strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return f"{_KS_PREDICATE}{slug}"


def _is_object_entity(item: dict) -> bool:
    """Decide whether the object field is an entity reference (vs a literal).

    Uses the LLM-provided ``object_type`` hint when available, falling back
    to the legacy heuristic (no spaces and <=60 chars).
    """
    obj_type = item.get("object_type", "")
    if obj_type == "entity":
        return True
    if obj_type == "literal":
        return False
    # Legacy heuristic fallback
    obj = item.get("object", "")
    return bool(obj) and " " not in obj and len(obj) <= 60


def _normalize_item_uris(item: dict) -> dict:
    """Normalize subject/predicate string fields to URIs based on knowledge_type.

    Also resolves predicate synonyms and strips the ``object_type`` helper field.
    """
    kt = item.get("knowledge_type", "")
    item = dict(item)
    if kt in ("Claim", "Fact", "Relationship"):
        if "subject" in item:
            item["subject"] = _to_entity_uri(item["subject"])
        if "predicate" in item:
            item["predicate"] = resolve_predicate_synonym(item["predicate"])
            item["predicate"] = _to_predicate_uri(item["predicate"])
        if _is_object_entity(item):
            item["object"] = _to_entity_uri(item["object"])
    elif kt == "TemporalState":
        if "subject" in item:
            item["subject"] = _to_entity_uri(item["subject"])
        if "property" in item:
            item["property"] = resolve_predicate_synonym(item["property"])
            item["property"] = _to_predicate_uri(item["property"])
    elif kt == "Event":
        if "subject" in item:
            item["subject"] = _to_entity_uri(item["subject"])
    # Strip object_type — not part of the Pydantic model
    item.pop("object_type", None)
    return item


def _build_extraction_prompt(text: str, title: str | None, source_type: str | None) -> str:
    context = ""
    if title:
        context += f"Title: {title}\n"
    if source_type:
        context += f"Source type: {source_type}\n"
    predicates_csv = ", ".join(CANONICAL_PREDICATES)
    return f"""{context}Extract structured knowledge from the text below.
Return ONLY a JSON object: {{"items": [...]}}

Each item must have a knowledge_type field. Supported types and required fields:
- Claim: subject, predicate, object, object_type, confidence (0.0-0.89)
- Fact: subject, predicate, object, object_type, confidence (0.9-1.0) for verified facts
- Relationship: subject, predicate, object, object_type, confidence
- Event: subject, occurred_at (YYYY-MM-DD), confidence, properties (dict)
- Entity: uri, rdf_type (e.g. "schema:Person"), label, properties (dict), confidence
- TemporalState: subject, property, value, valid_from (YYYY-MM-DD), valid_until (YYYY-MM-DD), confidence
- Conclusion: concludes (text), derived_from (list of identifiers), inference_method, confidence

Preferred predicates (use these when applicable):
{predicates_csv}
Only invent a new predicate if none of the above fit.

Entity naming rules:
- Use canonical, well-known names: "dopamine" not "the neurotransmitter dopamine"
- Use singular form: "neuron" not "neurons"
- Use lowercase snake_case: "cold_exposure" not "Cold Exposure"
- Be specific: "vitamin_d3" not "vitamin_d" when the text specifies D3

For object values, include object_type ("entity" or "literal"):
- "entity": the object is a thing/concept (e.g. "dopamine", "postgresql")
- "literal": the object is a measurement, description, or date (e.g. "250%", "2024-01-15")

Use Claim for uncertain assertions, Fact for high-confidence verifiable statements.
Extract 3-8 items. If nothing found, return {{"items": []}}

Example:
Text: "Regular cold water immersion has been shown to increase dopamine levels by up to 250%."
Output: {{"items": [
  {{"knowledge_type": "Claim", "subject": "cold_water_immersion", "predicate": "increases", "object": "dopamine", "object_type": "entity", "confidence": 0.75}},
  {{"knowledge_type": "Claim", "subject": "cold_water_immersion", "predicate": "has_property", "object": "250% dopamine increase", "object_type": "literal", "confidence": 0.7}}
]}}

Text:
---
{text[:_MAX_TEXT_CHARS]}
---"""


class ExtractionClient:
    """HTTP client that calls the OpenAI-compatible chat API to extract knowledge items."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self._model = model
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )

    async def extract(
        self,
        text: str,
        title: str | None = None,
        source_type: str | None = None,
    ) -> list:
        """Extract KnowledgeInput items from raw text. Returns [] on any failure."""
        from knowledge_service.models import KnowledgeInput  # noqa: PLC0415

        prompt = _build_extraction_prompt(text, title, source_type)
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("ExtractionClient: LLM API returned %s", exc.response.status_code)
            return []
        except httpx.TimeoutException as exc:
            logger.warning("ExtractionClient: LLM API request timed out: %s", exc)
            return []

        raw = response.json()["choices"][0]["message"]["content"]
        # Strip markdown fences that some models wrap around JSON
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.warning("ExtractionClient: could not parse JSON response: %s", exc)
            return []

        adapter = TypeAdapter(KnowledgeInput)
        result = []
        for item_dict in parsed.get("items", []):
            try:
                result.append(adapter.validate_python(item_dict))
            except ValidationError as exc:
                logger.warning("ExtractionClient: skipping invalid item %s: %s", item_dict, exc)
        return result

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if not self._client.is_closed:
            await self._client.aclose()
