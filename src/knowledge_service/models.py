from __future__ import annotations

import hashlib
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field, model_validator
from datetime import date, datetime
from enum import Enum


class KnowledgeType(str, Enum):
    CLAIM = "Claim"
    FACT = "Fact"
    EVENT = "Event"
    ENTITY = "Entity"
    RELATIONSHIP = "Relationship"
    CONCLUSION = "Conclusion"
    TEMPORAL_STATE = "TemporalState"


# --- Triple-shaped types (Claims, Facts, Relationships) ---


class TripleInput(BaseModel):
    """Base for knowledge that maps to a single S-P-O triple."""

    subject: str
    predicate: str
    object: str
    object_type: Literal["entity", "literal"] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: date | None = None
    valid_until: date | None = None


class ClaimInput(TripleInput):
    """Probabilistic assertion from consumed content."""

    knowledge_type: Literal[KnowledgeType.CLAIM] = KnowledgeType.CLAIM


class FactInput(TripleInput):
    """Verified truth from an authoritative source."""

    knowledge_type: Literal[KnowledgeType.FACT] = KnowledgeType.FACT
    confidence: float = Field(ge=0.9, le=1.0, default=0.99)  # Facts require high confidence


class RelationshipInput(TripleInput):
    """Typed connection between entities."""

    knowledge_type: Literal[KnowledgeType.RELATIONSHIP] = KnowledgeType.RELATIONSHIP


# --- Structured types (Events, Entities, Conclusions, TemporalStates) ---


class EventInput(BaseModel):
    """A timestamped occurrence. Produces multiple triples."""

    knowledge_type: Literal[KnowledgeType.EVENT] = KnowledgeType.EVENT
    subject: str  # What happened (URI)
    occurred_at: date
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    properties: dict[str, str] = {}  # Additional properties (amount, currency, etc.)


class EntityInput(BaseModel):
    """A thing that exists. Typed via external ontologies."""

    knowledge_type: Literal[KnowledgeType.ENTITY] = KnowledgeType.ENTITY
    uri: str  # Entity URI
    rdf_type: str  # e.g., "schema:SoftwareApplication"
    label: str  # Human-readable name
    properties: dict[str, str] = {}  # schema:name, skos:broader, etc.
    confidence: float = Field(ge=0.0, le=1.0, default=0.95)


class ConclusionInput(BaseModel):
    """Derived knowledge with reasoning chain preserved."""

    knowledge_type: Literal[KnowledgeType.CONCLUSION] = KnowledgeType.CONCLUSION
    concludes: str  # Text of the conclusion
    derived_from: list[str]  # Triple hashes or URIs of supporting evidence
    inference_method: str  # "bayesian_combination", "manual", etc.
    confidence: float = Field(ge=0.0, le=1.0)


class TemporalStateInput(BaseModel):
    """Property that changes over time. validUntil is mandatory."""

    knowledge_type: Literal[KnowledgeType.TEMPORAL_STATE] = KnowledgeType.TEMPORAL_STATE
    subject: str
    property: str  # What property is being tracked
    value: str
    valid_from: date
    valid_until: date  # Mandatory — distinguishes from regular claims
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    @model_validator(mode="after")
    def valid_until_after_valid_from(self):
        if self.valid_until < self.valid_from:
            raise ValueError("valid_until must be >= valid_from")
        return self


# --- Discriminated union for polymorphic ingestion ---

KnowledgeInput = Annotated[
    ClaimInput
    | FactInput
    | RelationshipInput
    | EventInput
    | EntityInput
    | ConclusionInput
    | TemporalStateInput,
    Field(discriminator="knowledge_type"),
]


# --- Expand any KnowledgeInput to RDF triples ---

# Namespace constants (duplicated from ontology.namespaces to avoid circular imports)
_KS = "http://knowledge.local/schema/"
_KS_DATA = "http://knowledge.local/data/"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


def _ensure_uri(value: str, fallback_ns: str = _KS) -> str:
    from urllib.parse import quote

    if value.startswith(("http://", "https://", "urn:")):
        return value
    slug = quote(value, safe="/_-:.~")
    return f"{fallback_ns}{slug}"


def expand_to_triples(
    item: Union[
        ClaimInput,
        FactInput,
        RelationshipInput,
        EventInput,
        EntityInput,
        ConclusionInput,
        TemporalStateInput,
    ],
) -> list[dict]:
    """Expand any KnowledgeInput into a list of triple dicts ready for insert_triple.

    Each dict has keys: subject, predicate, object, confidence, knowledge_type,
    valid_from, valid_until.
    """
    if isinstance(item, TripleInput):
        return [
            {
                "subject": _ensure_uri(item.subject, _KS_DATA),
                "predicate": _ensure_uri(item.predicate),
                "object": item.object,
                "confidence": item.confidence,
                "knowledge_type": item.knowledge_type.value,
                "valid_from": item.valid_from,
                "valid_until": item.valid_until,
            }
        ]

    if isinstance(item, EventInput):
        subject = _ensure_uri(item.subject, _KS_DATA)
        triples = [
            {
                "subject": subject,
                "predicate": f"{_KS}occurredAt",
                "object": item.occurred_at.isoformat(),
                "confidence": item.confidence,
                "knowledge_type": KnowledgeType.EVENT.value,
                "valid_from": None,
                "valid_until": None,
            }
        ]
        for key, value in item.properties.items():
            triples.append(
                {
                    "subject": subject,
                    "predicate": _ensure_uri(key),
                    "object": value,
                    "confidence": item.confidence,
                    "knowledge_type": KnowledgeType.EVENT.value,
                    "valid_from": None,
                    "valid_until": None,
                }
            )
        return triples

    if isinstance(item, EntityInput):
        triples = [
            {
                "subject": _ensure_uri(item.uri, _KS_DATA),
                "predicate": _RDF_TYPE,
                "object": _ensure_uri(item.rdf_type),
                "confidence": item.confidence,
                "knowledge_type": KnowledgeType.ENTITY.value,
                "valid_from": None,
                "valid_until": None,
            },
            {
                "subject": _ensure_uri(item.uri, _KS_DATA),
                "predicate": _RDFS_LABEL,
                "object": item.label,
                "confidence": item.confidence,
                "knowledge_type": KnowledgeType.ENTITY.value,
                "valid_from": None,
                "valid_until": None,
            },
        ]
        for key, value in item.properties.items():
            triples.append(
                {
                    "subject": _ensure_uri(item.uri, _KS_DATA),
                    "predicate": _ensure_uri(key),
                    "object": value,
                    "confidence": item.confidence,
                    "knowledge_type": KnowledgeType.ENTITY.value,
                    "valid_from": None,
                    "valid_until": None,
                }
            )
        return triples

    if isinstance(item, ConclusionInput):
        cid = hashlib.sha256(item.concludes.encode()).hexdigest()[:12]
        uri = f"{_KS_DATA}conclusion/{cid}"
        triples = [
            {
                "subject": uri,
                "predicate": f"{_KS}concludes",
                "object": item.concludes,
                "confidence": item.confidence,
                "knowledge_type": KnowledgeType.CONCLUSION.value,
                "valid_from": None,
                "valid_until": None,
            }
        ]
        for evidence in item.derived_from:
            triples.append(
                {
                    "subject": uri,
                    "predicate": f"{_KS}derivedFrom",
                    "object": evidence,
                    "confidence": item.confidence,
                    "knowledge_type": KnowledgeType.CONCLUSION.value,
                    "valid_from": None,
                    "valid_until": None,
                }
            )
        triples.append(
            {
                "subject": uri,
                "predicate": f"{_KS}inferenceMethod",
                "object": item.inference_method,
                "confidence": item.confidence,
                "knowledge_type": KnowledgeType.CONCLUSION.value,
                "valid_from": None,
                "valid_until": None,
            }
        )
        return triples

    if isinstance(item, TemporalStateInput):
        return [
            {
                "subject": _ensure_uri(item.subject, _KS_DATA),
                "predicate": _ensure_uri(item.property),
                "object": item.value,
                "confidence": item.confidence,
                "knowledge_type": KnowledgeType.TEMPORAL_STATE.value,
                "valid_from": item.valid_from,
                "valid_until": item.valid_until,
            }
        ]

    return []


# --- Request/Response models ---


class ContentRequest(BaseModel):
    url: str
    title: str
    summary: str | None = None
    raw_text: str | None = None
    source_type: str
    tags: list[str] = []
    metadata: dict = {}
    knowledge: list[KnowledgeInput] = []  # Type-specific, not generic "claims"


class ClaimsRequest(BaseModel):
    source_url: str
    source_type: str
    extractor: str
    knowledge: list[KnowledgeInput] = []  # Type-specific


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
    similarity: float
    source_type: str
    tags: list[str]
    ingested_at: datetime
    chunk_text: str
    chunk_index: int
    section_header: str | None = None


class HealthResponse(BaseModel):
    status: str
    components: dict[str, str]
