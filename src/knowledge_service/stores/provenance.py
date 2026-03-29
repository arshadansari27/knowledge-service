"""ProvenanceStore: asyncpg-backed store for the provenance PostgreSQL table.

Schema (from migrations/001_initial.sql):
    CREATE TABLE provenance (
        triple_hash     TEXT        NOT NULL,
        subject         TEXT        NOT NULL,
        predicate       TEXT        NOT NULL,
        object          TEXT        NOT NULL,
        source_url      TEXT        NOT NULL,
        source_type     TEXT        NOT NULL,
        extractor       TEXT        NOT NULL,
        confidence      FLOAT       NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
        ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
        valid_from      TIMESTAMPTZ,
        valid_until     TIMESTAMPTZ,
        metadata        JSONB       DEFAULT '{}',
        PRIMARY KEY (triple_hash, source_url)
    );
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any


class ProvenanceStore:
    """Wraps an asyncpg connection pool for provenance table operations."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def insert(
        self,
        triple_hash: str,
        subject: str,
        predicate: str,
        object_: str,
        source_url: str,
        source_type: str,
        extractor: str,
        confidence: float,
        metadata: dict | None = None,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        chunk_id: str | None = None,
    ) -> None:
        """Upsert a provenance record.

        On conflict (triple_hash, source_url) the confidence, metadata,
        temporal bounds, and chunk_id are updated to the new values,
        leaving ingested_at unchanged.
        """
        metadata_json = json.dumps(metadata if metadata is not None else {})

        sql = """
            INSERT INTO provenance (
                triple_hash, subject, predicate, object, source_url,
                source_type, extractor, confidence, metadata,
                valid_from, valid_until, chunk_id
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (triple_hash, source_url) DO UPDATE SET
                confidence  = EXCLUDED.confidence,
                metadata    = EXCLUDED.metadata,
                valid_from  = EXCLUDED.valid_from,
                valid_until = EXCLUDED.valid_until,
                chunk_id    = EXCLUDED.chunk_id
        """

        async with self._pool.acquire() as conn:
            await conn.execute(
                sql,
                triple_hash,
                subject,
                predicate,
                object_,
                source_url,
                source_type,
                extractor,
                confidence,
                metadata_json,
                valid_from,
                valid_until,
                chunk_id,
            )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_by_triple(self, triple_hash: str) -> list[dict]:
        """Return all provenance rows for a given triple hash."""
        sql = "SELECT * FROM provenance WHERE triple_hash = $1"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, triple_hash)
        return [dict(row) for row in rows]

    async def get_by_triples(self, triple_hashes: list[str]) -> dict[str, list[dict]]:
        """Return provenance rows for multiple triple hashes in a single query."""
        if not triple_hashes:
            return {}
        sql = "SELECT * FROM provenance WHERE triple_hash = ANY($1::text[])"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, triple_hashes)
        result: dict[str, list[dict]] = {}
        for row in rows:
            h = row["triple_hash"]
            result.setdefault(h, []).append(dict(row))
        return result

    async def query_by_entity_and_time(self, entity_uri: str, since: datetime) -> list[dict]:
        """Return provenance rows where subject = entity_uri AND ingested_at >= since."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT triple_hash, subject, predicate, object, source_url, source_type,
                          confidence, ingested_at
                   FROM provenance
                   WHERE subject = $1 AND ingested_at >= $2
                   ORDER BY ingested_at DESC""",
                entity_uri,
                since,
            )
        return [dict(r) for r in rows]
