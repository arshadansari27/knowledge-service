CREATE TABLE ingestion_jobs (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id        UUID        NOT NULL REFERENCES content_metadata(id) ON DELETE CASCADE,
    status            TEXT        NOT NULL DEFAULT 'accepted',
    chunks_total      INTEGER     NOT NULL DEFAULT 0,
    chunks_embedded   INTEGER     NOT NULL DEFAULT 0,
    chunks_extracted  INTEGER     NOT NULL DEFAULT 0,
    chunks_failed     INTEGER     NOT NULL DEFAULT 0,
    triples_created   INTEGER     NOT NULL DEFAULT 0,
    entities_resolved INTEGER     NOT NULL DEFAULT 0,
    chunks_capped_from INTEGER,
    error             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_ingestion_jobs_content ON ingestion_jobs (content_id);
CREATE INDEX idx_ingestion_jobs_status ON ingestion_jobs (status);
CREATE INDEX idx_ingestion_jobs_created ON ingestion_jobs (created_at DESC);

-- Only one active (non-terminal) job per content_id at a time
CREATE UNIQUE INDEX idx_ingestion_jobs_active ON ingestion_jobs (content_id)
    WHERE status NOT IN ('completed', 'failed');

-- Auto-update updated_at on every row change
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = now(); RETURN NEW; END; $$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ingestion_jobs_updated
    BEFORE UPDATE ON ingestion_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
