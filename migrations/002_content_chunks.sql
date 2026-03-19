-- 002_content_chunks.sql
-- Restructure content storage: metadata in content_metadata, chunks in content.
-- Existing content data is dropped (acceptable at v0.1.x).

DROP TABLE IF EXISTS content;

CREATE TABLE content_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    summary TEXT,
    raw_text TEXT,
    source_type TEXT NOT NULL,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL REFERENCES content_metadata(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(768),
    char_start INTEGER,
    char_end INTEGER,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(content_id, chunk_index)
);

CREATE INDEX idx_content_embedding ON content
    USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops);
CREATE INDEX idx_content_content_id ON content (content_id);
