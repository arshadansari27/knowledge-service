-- Drop tables, columns, and indexes whose feature code was removed in earlier
-- refactors but the schema artefacts were left behind.
--
--   ingestion_events:           legacy event log; last write 2026-04-16, no
--                               readers or writers in current code (audit
--                               2026-04-30).
--   ingestion_jobs.federation_enriched:
--                               column added in migration 009 for a federation
--                               enrichment counter; never written by any code
--                               path (max() = 0 in prod) and never read.
--   idx_ingestion_jobs_active:  duplicate of idx_ingestion_jobs_active_content
--                               (same column, same predicate). The 010
--                               migration created the second index but did not
--                               drop this one.

DROP TABLE IF EXISTS ingestion_events;

ALTER TABLE ingestion_jobs DROP COLUMN IF EXISTS federation_enriched;

DROP INDEX IF EXISTS idx_ingestion_jobs_active;
