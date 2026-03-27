"""EntityStore: entity and predicate embedding storage with resolution.

Combines entity/predicate embedding CRUD (formerly in EmbeddingStore) with
resolution logic (formerly in EntityResolver). Uses LRU caching to avoid
redundant embedding calls for repeated labels.
"""

from __future__ import annotations

import logging
from typing import Any

from cachetools import LRUCache

from knowledge_service.ontology.uri import to_entity_uri, to_predicate_uri

logger = logging.getLogger(__name__)

ENTITY_SIMILARITY_THRESHOLD = 0.85
PREDICATE_SIMILARITY_THRESHOLD = 0.90
_CACHE_SIZE = 1024


class EntityStore:
    """Manages entity and predicate embeddings with deduplication via similarity search."""

    def __init__(self, pool: Any, embedding_client: Any) -> None:
        self._pool = pool
        self._embedding_client = embedding_client
        self._entity_cache: LRUCache[str, str] = LRUCache(maxsize=_CACHE_SIZE)
        self._predicate_cache: LRUCache[str, str] = LRUCache(maxsize=_CACHE_SIZE)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vector_to_str(embedding: list[float]) -> str:
        """Convert a Python list of floats to the pgvector literal string '[a,b,c,...]'."""
        return "[" + ",".join(str(v) for v in embedding) + "]"

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    async def resolve_entity(self, label: str, rdf_type: str | None = None) -> str:
        """Resolve a concept label to a canonical entity URI.

        1. Check LRU cache
        2. Embed the label
        3. Search existing entity embeddings
        4. If similarity >= threshold, return existing URI
        5. Otherwise create new URI and store embedding
        """
        cache_key = label.lower()
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        embedding = await self._embedding_client.embed(label)

        candidates = await self.search_entities(query_embedding=embedding, limit=3)
        for candidate in candidates:
            if candidate["similarity"] >= ENTITY_SIMILARITY_THRESHOLD:
                uri = candidate["uri"]
                self._entity_cache[cache_key] = uri
                return uri

        # No match — create new entity
        uri = to_entity_uri(label)
        await self.insert_entity_embedding(
            uri=uri, label=label, rdf_type=rdf_type or "", embedding=embedding
        )
        self._entity_cache[cache_key] = uri
        return uri

    async def resolve_predicate(self, label: str) -> str:
        """Resolve a predicate label to a canonical predicate URI.

        1. Check LRU cache
        2. Embed the label and search predicate_embeddings
        3. If similarity >= threshold, return existing URI
        4. Otherwise create new predicate URI and store embedding
        """
        cache_key = label.lower()
        if cache_key in self._predicate_cache:
            return self._predicate_cache[cache_key]

        embedding = await self._embedding_client.embed(label)

        candidates = await self.search_predicates(query_embedding=embedding, limit=3)
        for candidate in candidates:
            if candidate["similarity"] >= PREDICATE_SIMILARITY_THRESHOLD:
                uri = candidate["uri"]
                self._predicate_cache[cache_key] = uri
                return uri

        # No match — create new predicate
        uri = to_predicate_uri(label)
        await self.insert_predicate_embedding(uri=uri, label=label, embedding=embedding)
        self._predicate_cache[cache_key] = uri
        return uri

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

    # ------------------------------------------------------------------
    # Predicate embeddings table operations
    # ------------------------------------------------------------------

    async def insert_predicate_embedding(
        self,
        uri: str,
        label: str,
        embedding: list[float],
    ) -> None:
        """Upsert a predicate embedding row."""
        embedding_str = self._vector_to_str(embedding)

        sql = """
            INSERT INTO predicate_embeddings (uri, label, embedding)
            VALUES ($1, $2, $3::vector(768))
            ON CONFLICT (uri) DO UPDATE SET
                label     = EXCLUDED.label,
                embedding = EXCLUDED.embedding
        """

        async with self._pool.acquire() as conn:
            await conn.execute(sql, uri, label, embedding_str)

    async def search_predicates(
        self,
        query_embedding: list[float],
        limit: int,
    ) -> list[dict]:
        """Return predicate rows ranked by cosine similarity."""
        embedding_str = self._vector_to_str(query_embedding)

        sql = """
            SELECT
                uri, label,
                1 - (embedding::halfvec(768) <=> $1::halfvec(768)) AS similarity
            FROM predicate_embeddings
            ORDER BY embedding::halfvec(768) <=> $1::halfvec(768)
            LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, embedding_str, limit)
        return [dict(row) for row in rows]
