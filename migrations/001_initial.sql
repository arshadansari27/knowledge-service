-- Knowledge Service — Initial Schema
-- gen_random_uuid() is built-in since PostgreSQL 13, no extension needed

CREATE EXTENSION IF NOT EXISTS vector;

-- Provenance: tracks where each triple came from
-- Triple components are denormalized here so provenance is independently
-- queryable without the RDF store (for audit, debugging, contradiction reports).
CREATE TABLE provenance (
    triple_hash     TEXT        NOT NULL,
    subject         TEXT        NOT NULL,
    predicate       TEXT        NOT NULL,
    object          TEXT        NOT NULL,
    source_url      TEXT        NOT NULL,
    source_type     TEXT        NOT NULL,
    extractor       TEXT        NOT NULL,
    confidence      FLOAT       NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    valid_from      TIMESTAMPTZ,
    valid_until     TIMESTAMPTZ,
    metadata        JSONB       DEFAULT '{}',

    PRIMARY KEY (triple_hash, source_url)
);

CREATE INDEX idx_provenance_confidence ON provenance (confidence);
CREATE INDEX idx_provenance_ingested ON provenance (ingested_at);
CREATE INDEX idx_provenance_valid_range ON provenance (valid_from, valid_until);
CREATE INDEX idx_provenance_source_type ON provenance (source_type);

-- Content: raw ingested content with embeddings
CREATE TABLE content (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    url             TEXT        UNIQUE,
    title           TEXT,
    summary         TEXT,
    raw_text        TEXT,
    source_type     TEXT        NOT NULL,
    tags            TEXT[]      DEFAULT '{}',
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding       vector(768),
    -- Source-specific fields live in metadata JSONB:
    -- article: {"author", "publication", "word_count"}
    -- video:   {"channel", "duration_seconds", "has_transcript"}
    -- paper:   {"authors", "doi", "journal", "abstract"}
    metadata        JSONB       DEFAULT '{}'
);

CREATE INDEX idx_content_embedding ON content
    USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops);
CREATE INDEX idx_content_source_type ON content (source_type);
CREATE INDEX idx_content_ingested ON content (ingested_at);
CREATE INDEX idx_content_tags ON content USING gin (tags);

-- Ingestion events: append-only audit log
CREATE TABLE ingestion_events (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type      TEXT        NOT NULL,
    payload         JSONB       NOT NULL,
    source          TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    processed_at    TIMESTAMPTZ
);

CREATE INDEX idx_ingestion_events_type ON ingestion_events (event_type);
CREATE INDEX idx_ingestion_events_created ON ingestion_events (created_at);

-- Entity embeddings: for entity resolution on write path
CREATE TABLE entity_embeddings (
    uri             TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    rdf_type        TEXT DEFAULT '',
    embedding       vector(768),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_entity_embedding ON entity_embeddings
    USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops);
