"""ContentStore: asyncpg-backed store for content metadata and chunks.

Split from EmbeddingStore — manages only the content_metadata and content (chunks) tables.
Entity and predicate embeddings are handled by EntityStore.

Tables:

    content_metadata:
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid()
        url             TEXT UNIQUE NOT NULL
        title           TEXT
        summary         TEXT
        raw_text        TEXT
        source_type     TEXT NOT NULL
        tags            TEXT[] DEFAULT '{}'
        ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        metadata        JSONB DEFAULT '{}'

    content (chunks):
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid()
        content_id      UUID NOT NULL REFERENCES content_metadata(id) ON DELETE CASCADE
        chunk_index     INTEGER NOT NULL
        chunk_text      TEXT NOT NULL
        embedding       vector(768)
        char_start      INTEGER
        char_end        INTEGER
        created_at      TIMESTAMPTZ DEFAULT now()
        UNIQUE(content_id, chunk_index)
"""

from __future__ import annotations

import json
from typing import Any


def reciprocal_rank_fusion(
    *result_lists: list[dict],
    key: str = "id",
    k: int = 60,
    limit: int = 10,
) -> list[dict]:
    """Fuse multiple ranked result lists via Reciprocal Rank Fusion.

    Each item's score = sum(1 / (k + rank + 1)) across all lists it appears in.
    Items appearing in multiple lists score higher than single-list items.
    The fused RRF score replaces the 'similarity' field in the returned dicts.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            item_key = str(item[key])
            scores[item_key] = scores.get(item_key, 0.0) + 1.0 / (k + rank + 1)
            items[item_key] = item
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    fused = []
    for item_key, score in ranked:
        result = dict(items[item_key])
        result["similarity"] = score
        fused.append(result)
    return fused


class ContentStore:
    """Wraps an asyncpg connection pool for content metadata and chunk operations."""

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
    # Content metadata table operations
    # ------------------------------------------------------------------

    async def upsert_metadata(
        self,
        url: str,
        title: str | None,
        summary: str | None,
        raw_text: str | None,
        source_type: str,
        tags: list[str] | None,
        metadata: dict | None,
    ) -> str:
        """Upsert a content_metadata row and return its UUID.

        On conflict (url) the existing row is updated with fresh values,
        leaving id and ingested_at unchanged.
        """
        metadata_json = json.dumps(metadata or {})

        sql = """
            INSERT INTO content_metadata (
                url, title, summary, raw_text, source_type, tags, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (url) DO UPDATE SET
                title       = EXCLUDED.title,
                summary     = EXCLUDED.summary,
                raw_text    = EXCLUDED.raw_text,
                source_type = EXCLUDED.source_type,
                tags        = EXCLUDED.tags,
                metadata    = EXCLUDED.metadata
            RETURNING id
        """

        async with self._pool.acquire() as conn:
            content_id = await conn.fetchval(
                sql,
                url,
                title,
                summary,
                raw_text,
                source_type,
                tags or [],
                metadata_json,
            )
        return str(content_id)

    # ------------------------------------------------------------------
    # Content (chunks) table operations
    # ------------------------------------------------------------------

    async def delete_chunks(self, content_id: str) -> None:
        """Delete all chunks for a given content_id."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM content WHERE content_id = $1",
                content_id,
            )

    async def insert_chunks(
        self,
        content_id: str,
        chunks: list[dict],
    ) -> list[tuple[int, str]]:
        """Insert chunk rows. Returns list of (chunk_index, chunk_id).

        Each dict must have: chunk_index, chunk_text, embedding, char_start, char_end.
        Optional: section_header.
        """
        if not chunks:
            return []

        placeholders = []
        values: list = []
        for i, chunk in enumerate(chunks):
            base = i * 7
            placeholders.append(
                f"(${base + 1}, ${base + 2}, ${base + 3},"
                f" ${base + 4}::vector(768), ${base + 5}, ${base + 6}, ${base + 7})"
            )
            values.extend(
                [
                    content_id,
                    chunk["chunk_index"],
                    chunk["chunk_text"],
                    self._vector_to_str(chunk["embedding"]),
                    chunk["char_start"],
                    chunk["char_end"],
                    chunk.get("section_header"),
                ]
            )

        sql = f"""
            INSERT INTO content (
                content_id, chunk_index, chunk_text, embedding,
                char_start, char_end, section_header
            )
            VALUES {", ".join(placeholders)}
            RETURNING chunk_index, id
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *values)
        return [(row["chunk_index"], str(row["id"])) for row in rows]

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, str]:
        """Return {chunk_id: chunk_text} for the given IDs."""
        if not chunk_ids:
            return {}
        sql = "SELECT id, chunk_text FROM content WHERE id = ANY($1::uuid[])"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, chunk_ids)
        return {str(r["id"]): r["chunk_text"] for r in rows}

    # ------------------------------------------------------------------
    # Search operations
    # ------------------------------------------------------------------

    async def search(
        self,
        query_embedding: list[float],
        limit: int,
        source_type: str | None = None,
        tags: list[str] | None = None,
        min_date: Any | None = None,
        query_text: str | None = None,
    ) -> list[dict]:
        """Return chunk rows ranked by cosine similarity, joined with content metadata.

        When query_text is provided, also runs BM25 full-text search and fuses
        both result sets via Reciprocal Rank Fusion (RRF). Vector search is
        overfetched (limit * 3) to give RRF a richer candidate pool.

        Optional filters:
          source_type -- restrict to a single source type
          tags        -- restrict to rows that contain ALL given tags
          min_date    -- restrict to rows ingested on or after this date
          query_text  -- enable hybrid mode (vector + BM25 via RRF)
        """
        overfetch = limit * 3 if query_text else limit

        embedding_str = self._vector_to_str(query_embedding)

        conditions: list[str] = []
        params: list[Any] = [embedding_str]

        if source_type is not None:
            params.append(source_type)
            conditions.append(f"m.source_type = ${len(params)}")

        if tags is not None:
            params.append(tags)
            conditions.append(f"m.tags @> ${len(params)}")

        if min_date is not None:
            params.append(min_date)
            conditions.append(f"m.ingested_at >= ${len(params)}")

        params.append(overfetch)
        limit_placeholder = f"${len(params)}"

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = f"""
            SELECT
                c.id, c.chunk_text, c.chunk_index, c.section_header,
                m.id AS content_id, m.url, m.title, m.summary,
                m.source_type, m.tags, m.ingested_at,
                1 - (c.embedding::halfvec(768) <=> $1::halfvec(768)) AS similarity
            FROM content c
            JOIN content_metadata m ON c.content_id = m.id
            {where_clause}
            ORDER BY c.embedding::halfvec(768) <=> $1::halfvec(768)
            LIMIT {limit_placeholder}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        vector_results = [dict(row) for row in rows]

        if query_text:
            bm25_results = await self._search_bm25(
                query_text, overfetch, source_type, tags, min_date
            )
            return reciprocal_rank_fusion(vector_results, bm25_results, key="id", k=60, limit=limit)

        return vector_results

    async def get_chunks(self, content_id: str) -> list[dict] | None:
        """Return all chunks for a content item, ordered by chunk_index.

        Returns None if the content_id doesn't exist.
        """
        sql = """
            SELECT c.chunk_index, c.chunk_text, c.section_header, c.char_start, c.char_end
            FROM content c
            WHERE c.content_id = $1::uuid
            ORDER BY c.chunk_index
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, content_id)
        if not rows:
            return None
        return [dict(row) for row in rows]

    async def _search_bm25(
        self,
        query_text: str,
        limit: int,
        source_type: str | None = None,
        tags: list[str] | None = None,
        min_date: Any | None = None,
    ) -> list[dict]:
        """Full-text search using PostgreSQL tsvector/tsquery.

        Returns the same dict shape as search() for RRF compatibility.
        Uses plainto_tsquery for safe natural-language query parsing.
        """
        if not query_text or not query_text.strip():
            return []

        conditions: list[str] = ["c.tsv @@ plainto_tsquery('english', $1)"]
        params: list[Any] = [query_text]

        if source_type is not None:
            params.append(source_type)
            conditions.append(f"m.source_type = ${len(params)}")

        if tags is not None:
            params.append(tags)
            conditions.append(f"m.tags @> ${len(params)}")

        if min_date is not None:
            params.append(min_date)
            conditions.append(f"m.ingested_at >= ${len(params)}")

        params.append(limit)
        limit_placeholder = f"${len(params)}"

        where_clause = f"WHERE {' AND '.join(conditions)}"

        sql = f"""
            SELECT
                c.id, c.chunk_text, c.chunk_index, c.section_header,
                m.id AS content_id, m.url, m.title, m.summary,
                m.source_type, m.tags, m.ingested_at,
                ts_rank(c.tsv, plainto_tsquery('english', $1)) AS similarity
            FROM content c
            JOIN content_metadata m ON c.content_id = m.id
            {where_clause}
            ORDER BY ts_rank(c.tsv, plainto_tsquery('english', $1)) DESC
            LIMIT {limit_placeholder}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [dict(row) for row in rows]
