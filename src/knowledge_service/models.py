from __future__ import annotations

from datetime import date, datetime
from pydantic import BaseModel, Field

from knowledge_service.ontology.uri import KS, RDF_TYPE, RDFS_LABEL, to_entity_uri, to_predicate_uri


# --- Knowledge types ---


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
    """Timestamped occurrence. Expands to N triples."""

    subject: str
    occurred_at: date
    properties: dict[str, str] = {}
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    def to_triples(self) -> list[dict]:
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
            triples.append(
                {
                    "subject": uri,
                    "predicate": to_predicate_uri(key),
                    "object": value,
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
    properties: dict[str, str] = {}
    confidence: float = Field(ge=0.0, le=1.0, default=0.95)

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
            triples.append(
                {
                    "subject": entity_uri,
                    "predicate": to_predicate_uri(key),
                    "object": value,
                    "confidence": self.confidence,
                    "knowledge_type": "entity",
                    "valid_from": None,
                    "valid_until": None,
                }
            )
        return triples


# Union type — no Pydantic discriminator
KnowledgeInput = TripleInput | EventInput | EntityInput


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
