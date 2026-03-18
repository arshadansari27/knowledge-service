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


def _normalize_item_uris(item: dict) -> dict:
    """Normalize subject/predicate string fields to URIs based on knowledge_type."""
    kt = item.get("knowledge_type", "")
    item = dict(item)
    if kt in ("Claim", "Fact", "Relationship"):
        if "subject" in item:
            item["subject"] = _to_entity_uri(item["subject"])
        if "predicate" in item:
            item["predicate"] = _to_predicate_uri(item["predicate"])
        # Leave object as-is if it looks like a literal (has spaces or is long)
        obj = item.get("object", "")
        if obj and " " not in obj and len(obj) <= 60:
            item["object"] = _to_entity_uri(obj)
    elif kt == "TemporalState":
        if "subject" in item:
            item["subject"] = _to_entity_uri(item["subject"])
        if "property" in item:
            item["property"] = _to_predicate_uri(item["property"])
    elif kt == "Event":
        if "subject" in item:
            item["subject"] = _to_entity_uri(item["subject"])
    return item


def _build_extraction_prompt(text: str, title: str | None, source_type: str | None) -> str:
    context = ""
    if title:
        context += f"Title: {title}\n"
    if source_type:
        context += f"Source type: {source_type}\n"
    return f"""{context}Extract structured knowledge from the text below.
Return ONLY a JSON object: {{"items": [...]}}

Each item must have knowledge_type (Claim, Fact, or Relationship) and:
- Claim/Fact/Relationship: subject, predicate, object (short snake_case identifiers), confidence (0.0-1.0)
- Use Claim for uncertain assertions (confidence < 0.9), Fact for high-confidence verifiable statements (confidence >= 0.9)
- Extract 3-8 items. If nothing found, return {{"items": []}}

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
            item_dict = _normalize_item_uris(item_dict)
            try:
                result.append(adapter.validate_python(item_dict))
            except ValidationError as exc:
                logger.warning("ExtractionClient: skipping invalid item %s: %s", item_dict, exc)
        return result

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if not self._client.is_closed:
            await self._client.aclose()
