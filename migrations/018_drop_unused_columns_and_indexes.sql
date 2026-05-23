-- Drop columns and indexes that audit (2026-05-23) identified as orphan.
-- ALL writers were removed in the prior PR (#81) — this migration cannot land
-- before that code change is on main, otherwise prod will try to write to
-- columns that no longer exist.
--
-- Columns:
--   ingestion_jobs.source_format        — never written by any code path
--                                         (no JobTracker key, no UPDATE).
--   ingestion_jobs.entities_linked      — JobTracker wrote it after NLP; no
--                                         reader anywhere. Write dropped in #81.
--   ingestion_jobs.entities_coref       — JobTracker wrote it after coreference;
--                                         no reader. Write dropped in #81.
--   ingestion_jobs.chunks_skipped       — ExtractPhase always returned 0;
--                                         column was structurally always 0.
--                                         Write + display dropped in #81.
--   entity_aliases.source               — hard-coded to "spacy_linking" by
--                                         CoreferencePhase; no reader. Write
--                                         dropped in #81.
--   content_metadata.metadata           — API accepted a JSONB metadata dict
--                                         and ContentStore wrote it; no SELECT
--                                         path ever surfaced it. API field +
--                                         write dropped in #81.
--
-- Indexes:
--   idx_provenance_confidence           — no WHERE/ORDER BY on provenance
--                                         .confidence anywhere in code.
--                                         Confidence filtering lives in
--                                         pyoxigraph SPARQL annotations.
--   idx_provenance_source_type          — no WHERE provenance.source_type =
--                                         in any query path.
--   idx_provenance_valid_range          — valid_from/valid_until written and
--                                         read via SELECT *, never appear in
--                                         a WHERE predicate.
--   idx_entity_aliases_canonical        — no reverse "what aliases point at
--                                         this canonical URI" lookup is
--                                         implemented; only forward
--                                         alias-PK lookup is used.

ALTER TABLE ingestion_jobs DROP COLUMN IF EXISTS source_format;
ALTER TABLE ingestion_jobs DROP COLUMN IF EXISTS entities_linked;
ALTER TABLE ingestion_jobs DROP COLUMN IF EXISTS entities_coref;
ALTER TABLE ingestion_jobs DROP COLUMN IF EXISTS chunks_skipped;

ALTER TABLE entity_aliases DROP COLUMN IF EXISTS source;

ALTER TABLE content_metadata DROP COLUMN IF EXISTS metadata;

DROP INDEX IF EXISTS idx_provenance_confidence;
DROP INDEX IF EXISTS idx_provenance_source_type;
DROP INDEX IF EXISTS idx_provenance_valid_range;
DROP INDEX IF EXISTS idx_entity_aliases_canonical;
