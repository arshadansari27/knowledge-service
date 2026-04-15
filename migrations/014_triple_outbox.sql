-- Outbox table for coordinating pyoxigraph writes with PG transactions.
-- See docs/superpowers/specs/2026-04-15-processphase-2pc-outbox-design.md

CREATE TABLE IF NOT EXISTS triple_outbox (
    id              BIGSERIAL PRIMARY KEY,
    triple_hash     TEXT NOT NULL,
    operation       TEXT NOT NULL,
    subject         TEXT NOT NULL,
    predicate       TEXT NOT NULL,
    object          TEXT NOT NULL,
    confidence      DOUBLE PRECISION,
    knowledge_type  TEXT,
    valid_from      TIMESTAMPTZ,
    valid_until     TIMESTAMPTZ,
    graph           TEXT NOT NULL,
    payload         JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    applied_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_outbox_pending
    ON triple_outbox (id) WHERE applied_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_outbox_hash
    ON triple_outbox (triple_hash);
