ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS chunks_skipped INTEGER DEFAULT 0;
