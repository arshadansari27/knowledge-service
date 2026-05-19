from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any

from pydantic import BaseModel, Discriminator, Field, Tag, field_validator, model_validator

from knowledge_service.ontology.uri import KS, RDF_TYPE, RDFS_LABEL, to_entity_uri, to_predicate_uri


# --- Knowledge types ---


PropertyValue = str | list[str]


class TripleInput(BaseModel):
    """Universal knowledge unit. Replaces Claim, Fact, Relationship, TemporalState."""

    subject: str
    predicate: str
    object: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    knowledge_type: str = "claim"
    valid_from: date | None = None
    valid_until: date | None = None

    def to_triples(self) -> list[dict]:
        return [
            {
                "subject": to_entity_uri(self.subject),
                "predicate": to_predicate_uri(self.predicate),
                "object": self.object,
                "confidence": self.confidence,
                "knowledge_type": self.knowledge_type,
                "valid_from": self.valid_from,
                "valid_until": self.valid_until,
            }
        ]


class EventInput(BaseModel):
    """Timestamped occurrence. Expands to N triples.

    ``occurred_at`` accepts either a date or a string. Strings that don't parse
    as ISO dates (e.g. qwen3's "some_25_years_ago") are coerced to ``None``;
    ``to_triples()`` returns an empty list in that case — an event without a
    timestamp is not worth emitting.
    """

    subject: str
    occurred_at: date | None = None
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    knowledge_type: str = "event"

    @field_validator("occurred_at", mode="before")
    @classmethod
    def _coerce_unparseable_date(cls, value: Any) -> Any:
        if value is None or isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return value

    def to_triples(self) -> list[dict]:
        if self.occurred_at is None:
            return []
        uri = to_entity_uri(self.subject)
        triples = [
            {
                "subject": uri,
                "predicate": f"{KS}occurredAt",
                "object": self.occurred_at.isoformat(),
                "confidence": self.confidence,
                "knowledge_type": "event",
                "valid_from": None,
                "valid_until": None,
            }
        ]
        for key, value in self.properties.items():
            predicate = to_predicate_uri(key)
            for object_ in _expand_property_value(value):
                triples.append(
                    {
                        "subject": uri,
                        "predicate": predicate,
                        "object": object_,
                        "confidence": self.confidence,
                        "knowledge_type": "event",
                        "valid_from": None,
                        "valid_until": None,
                    }
                )
        return triples


class EntityInput(BaseModel):
    """Thing with type, label, properties. Expands to 2+ triples."""

    uri: str
    rdf_type: str
    label: str
    properties: dict[str, PropertyValue] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.95)
    knowledge_type: str = "entity"

    @model_validator(mode="before")
    @classmethod
    def _lift_misplaced_fields(cls, data: Any) -> Any:
        """qwen3:14b occasionally nests ``rdf_type`` and ``label`` inside
        ``properties``. When the top-level field is missing, lift the value
        out of ``properties`` so validation succeeds."""
        if not isinstance(data, dict):
            return data
        properties = data.get("properties")
        if not isinstance(properties, dict):
            return data
        for field_name in ("rdf_type", "label"):
            if field_name in properties and not data.get(field_name):
                data[field_name] = properties.pop(field_name)
        return data

    def to_triples(self) -> list[dict]:
        entity_uri = to_entity_uri(self.uri)
        triples = [
            {
                "subject": entity_uri,
                "predicate": RDF_TYPE,
                "object": self.rdf_type,
                "confidence": self.confidence,
                "knowledge_type": "entity",
                "valid_from": None,
                "valid_until": None,
            },
            {
                "subject": entity_uri,
                "predicate": RDFS_LABEL,
                "object": self.label,
                "confidence": self.confidence,
                "knowledge_type": "entity",
                "valid_from": None,
                "valid_until": None,
            },
        ]
        for key, value in self.properties.items():
            predicate = to_predicate_uri(key)
            for object_ in _expand_property_value(value):
                triples.append(
                    {
                        "subject": entity_uri,
                        "predicate": predicate,
                        "object": object_,
                        "confidence": self.confidence,
                        "knowledge_type": "entity",
                        "valid_from": None,
                        "valid_until": None,
                    }
                )
        return triples


def _expand_property_value(value: PropertyValue) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [value]


def _route_knowledge_input(value: Any) -> str | None:
    """Discriminator callable: pick a union member by ``knowledge_type``.

    Routes ``Entity`` / ``Event`` (case-insensitive) to their own members; every
    other label (``Claim``, ``Fact``, ``Relationship``, ``TemporalFact``,
    ``TemporalState``, …) routes to TripleInput. Without this, Pydantic
    reports per-member errors for the whole union and TripleInput's three
    missing-field errors drown out the real reason for rejection.

    When ``knowledge_type`` is missing or empty we fall back to shape
    detection so untagged payloads accepted by the pre-discriminator union
    keep working (the public API on ``/api/claims`` does not require the tag).
    """
    if isinstance(value, dict):
        tag = value.get("knowledge_type")
        if not isinstance(tag, str) or not tag.strip():
            if "occurred_at" in value:
                return "event"
            if "rdf_type" in value or (
                "uri" in value and "label" in value and "predicate" not in value
            ):
                return "entity"
            return "triple"
    else:
        tag = getattr(value, "knowledge_type", None)
        if not isinstance(tag, str):
            return "triple"
    normalised = tag.strip().lower()
    if normalised == "entity":
        return "entity"
    if normalised == "event":
        return "event"
    return "triple"


KnowledgeInput = Annotated[
    Annotated[TripleInput, Tag("triple")]
    | Annotated[EventInput, Tag("event")]
    | Annotated[EntityInput, Tag("entity")],
    Discriminator(_route_knowledge_input),
]


# --- Request/Response models ---


class ContentRequest(BaseModel):
    url: str
    title: str | None = None
    summary: str | None = None
    raw_text: str | None = None
    source_type: str | None = None
    tags: list[str] = []
    metadata: dict = {}
    knowledge: list[KnowledgeInput] = []
    domains: list[str] | None = None  # optional domain hint for extraction


class ClaimsRequest(BaseModel):
    source_url: str
    source_type: str
    extractor: str
    knowledge: list[KnowledgeInput] = []


class ContentAcceptedResponse(BaseModel):
    content_id: str
    job_id: str
    status: str = "accepted"
    chunks_total: int
    chunks_capped_from: int | None = None


class IngestionJobStatus(BaseModel):
    content_id: str
    job_id: str
    status: str
    chunks_total: int
    chunks_embedded: int
    chunks_extracted: int
    chunks_failed: int
    triples_created: int
    entities_resolved: int
    error: str | None
    created_at: str
    updated_at: str


class ClaimsResponse(BaseModel):
    triples_created: int
    contradictions_detected: list[dict] = []


class SearchResult(BaseModel):
    content_id: str
    url: str
    title: str
    summary: str | None
    # Real cosine similarity from the vector stage. None when the chunk only
    # surfaced via BM25 (no vector comparison was made for it).
    similarity: float | None
    # Fused rank score used for ordering when hybrid (vector + BM25) is active.
    # Equal to ``similarity`` when only vector search ran.
    rrf_score: float | None = None
    # 0-based rank from the BM25 stage; null for vector-only hits.
    bm25_rank: int | None = None
    source_type: str
    tags: list[str]
    ingested_at: datetime
    chunk_text: str
    chunk_index: int
    section_header: str | None = None


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]
