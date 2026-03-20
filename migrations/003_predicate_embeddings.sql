-- Predicate embeddings for semantic predicate resolution.
-- Analogous to entity_embeddings but for predicates (verbs/relationships).

CREATE TABLE IF NOT EXISTS predicate_embeddings (
    uri         TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    embedding   vector(768),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_predicate_embeddings_hnsw
    ON predicate_embeddings
    USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops);
