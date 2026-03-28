-- migrations/012_entity_aliases.sql
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias       TEXT PRIMARY KEY,
    canonical   TEXT NOT NULL,  -- stores URI, e.g. http://knowledge.local/data/narendra_modi
    source      TEXT NOT NULL,  -- "spacy_linking" or "llm_coreference"
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical ON entity_aliases(canonical);

ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS source_format TEXT,
    ADD COLUMN IF NOT EXISTS entities_linked INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS entities_coref INTEGER DEFAULT 0;
