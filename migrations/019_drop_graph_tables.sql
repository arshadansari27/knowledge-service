-- Drop the knowledge-graph tables left vestigial after the graph layer was
-- removed in the no-graph redesign (KS PRs #87 embed-only ingest, #89 graph
-- code deletion). The chunk-only system uses only content, content_metadata,
-- and ingestion_jobs. Nothing reads or writes these tables anymore — the code
-- that created and seeded them (TripleStore, EntityStore + canonical-predicate
-- seeding, the triple-processing pipeline, the outbox drainer) is deleted.
--
--   triple_outbox        — outbox queue for triple writes (drainer removed)
--   provenance           — per-triple provenance (graph)
--   entity_embeddings    — entity-resolution vectors (EntityStore removed)
--   predicate_embeddings — canonical-predicate seed vectors (seeding removed)
--   entity_aliases       — entity alias map (graph)

DROP TABLE IF EXISTS triple_outbox CASCADE;
DROP TABLE IF EXISTS provenance CASCADE;
DROP TABLE IF EXISTS entity_embeddings CASCADE;
DROP TABLE IF EXISTS predicate_embeddings CASCADE;
DROP TABLE IF EXISTS entity_aliases CASCADE;
