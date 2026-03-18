"""EmbeddingStore: asyncpg-backed store for pgvector semantic similarity search.

Manages two tables (schema from migrations/001_initial.sql):

    content:
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid()
        url             TEXT UNIQUE
        title           TEXT
        summary         TEXT
        raw_text        TEXT
        source_type     TEXT NOT NULL
        tags            TEXT[] DEFAULT '{}'
        ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        embedding       vector(768)
        metadata        JSONB DEFAULT '{}'

    entity_embeddings:
        uri             TEXT PRIMARY KEY
        label           TEXT NOT NULL
        rdf_type        TEXT DEFAULT ''
        embedding       vector(768)
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()

Both tables have HNSW indexes on (embedding::halfvec(768)) using halfvec_cosine_ops.
Queries must cast to halfvec to exploit those indexes.
"""

from __future__ import annotations

import json
from typing import Any


class EmbeddingStore:
    """Wraps an asyncpg connection pool for pgvector embedding operations."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vector_to_str(embedding: list[float]) -> str:
        """Convert a Python list of floats to the pgvector literal string '[a,b,c,...]'."""
        return "[" + ",".join(str(v) for v in embedding) + "]"

    # ------------------------------------------------------------------
    # Content table operations
    # ------------------------------------------------------------------

    async def insert_content(
        self,
        url: str,
        title: str,
        summary: str,
        raw_text: str,
        source_type: str,
        tags: list[str],
        embedding: list[float],
        metadata: dict,
    ) -> str:
        """Upsert a content row and return its UUID.

        On conflict (url) the existing row is updated with fresh values for all
        mutable columns, leaving id and ingested_at unchanged.
        """
        embedding_str = self._vector_to_str(embedding)
        metadata_json = json.dumps(metadata)

        sql = """
            INSERT INTO content (
                url, title, summary, raw_text, source_type,
                tags, embedding, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7::vector(768), $8)
            ON CONFLICT (url) DO UPDATE SET
                title      = EXCLUDED.title,
                summary    = EXCLUDED.summary,
                raw_text   = EXCLUDED.raw_text,
                source_type = EXCLUDED.source_type,
                tags       = EXCLUDED.tags,
                embedding  = EXCLUDED.embedding,
                metadata   = EXCLUDED.metadata
            RETURNING id
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql,
                url,
                title,
                summary,
                raw_text,
                source_type,
                tags,
                embedding_str,
                metadata_json,
            )
        return str(row["id"])

    async def search(
        self,
        query_embedding: list[float],
        limit: int,
        source_type: str | None = None,
        tags: list[str] | None = None,
        min_date: Any | None = None,
    ) -> list[dict]:
        """Return content rows ranked by cosine similarity to query_embedding.

        Optional filters:
          source_type — restrict to a single source type
          tags        — restrict to rows that contain ALL given tags (array contains)

        The halfvec cast ensures the HNSW index is used.
        """
        embedding_str = self._vector_to_str(query_embedding)

        conditions: list[str] = []
        params: list[Any] = [embedding_str]

        if source_type is not None:
            params.append(source_type)
            conditions.append(f"source_type = ${len(params)}")

        if tags is not None:
            params.append(tags)
            conditions.append(f"tags @> ${len(params)}")

        if min_date is not None:
            params.append(min_date)
            conditions.append(f"ingested_at >= ${len(params)}")

        params.append(limit)
        limit_placeholder = f"${len(params)}"

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = f"""
            SELECT
                id, url, title, summary, source_type, tags, ingested_at,
                1 - (embedding::halfvec(768) <=> $1::halfvec(768)) AS similarity
            FROM content
            {where_clause}
            ORDER BY embedding::halfvec(768) <=> $1::halfvec(768)
            LIMIT {limit_placeholder}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Entity embeddings table operations
    # ------------------------------------------------------------------

    async def insert_entity_embedding(
        self,
        uri: str,
        label: str,
        rdf_type: str,
        embedding: list[float],
    ) -> None:
        """Upsert an entity embedding row.

        On conflict (uri) the label, rdf_type, and embedding are updated.
        """
        embedding_str = self._vector_to_str(embedding)

        sql = """
            INSERT INTO entity_embeddings (uri, label, rdf_type, embedding)
            VALUES ($1, $2, $3, $4::vector(768))
            ON CONFLICT (uri) DO UPDATE SET
                label     = EXCLUDED.label,
                rdf_type  = EXCLUDED.rdf_type,
                embedding = EXCLUDED.embedding
        """

        async with self._pool.acquire() as conn:
            await conn.execute(sql, uri, label, rdf_type, embedding_str)

    async def search_entities(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[dict]:
        """Return entity rows ranked by cosine similarity to query_embedding.

        The halfvec cast ensures the HNSW index is used.
        """
        embedding_str = self._vector_to_str(query_embedding)

        sql = """
            SELECT
                uri, label, rdf_type,
                1 - (embedding::halfvec(768) <=> $1::halfvec(768)) AS similarity
            FROM entity_embeddings
            ORDER BY embedding::halfvec(768) <=> $1::halfvec(768)
            LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, embedding_str, limit)
        return [dict(row) for row in rows]

    async def get_entity_by_uri(self, uri: str) -> dict | None:
        """Look up an entity by its URI. Returns {label, rdf_type} or None."""
        sql = "SELECT label, rdf_type FROM entity_embeddings WHERE uri = $1"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, uri)
        if row is None:
            return None
        return dict(row)
