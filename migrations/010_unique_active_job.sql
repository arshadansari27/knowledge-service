-- Partial unique index: only one non-terminal job per content_id
CREATE UNIQUE INDEX IF NOT EXISTS idx_ingestion_jobs_active_content
ON ingestion_jobs (content_id)
WHERE status NOT IN ('completed', 'failed');
