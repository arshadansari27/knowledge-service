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
        # Lazy predicate-seed state: None=never tried, True=done, False=failed-retry-on-next-use
        self._predicate_seed_status: bool | None = None
        self._predicate_seed_spec: list[tuple[str, str]] | None = None

    def set_predicate_seed(self, entries: list[tuple[str, str]]) -> None:
        """Register the canonical (uri, label) pairs to seed on first predicate resolution.

        Called at startup. Seeding itself is deferred to the first resolve_predicate()
        call (or can be triggered explicitly via ensure_predicates_seeded()). If the
        embedding backend is down at startup we don't block — the next lookup retries.
        """
        self._predicate_seed_spec = list(entries)
        self._predicate_seed_status = None

    async def ensure_predicates_seeded(self) -> bool:
        """Seed canonical predicate embeddings if not already done.

        Returns True on success or if already seeded, False on failure (caller may retry).
        Safe to call concurrently — races are harmless because insert_predicate_embedding
        is an upsert.
        """
        if self._predicate_seed_status is True:
            return True
        if not self._predicate_seed_spec:
            return True
        uris = [u for u, _ in self._predicate_seed_spec]
        labels = [label for _, label in self._predicate_seed_spec]
        try:
            embeddings = await self._embedding_client.embed_batch(labels)
        except Exception as exc:
            self._predicate_seed_status = False
            logger.warning("Predicate seed deferred — embedding backend unavailable: %s", exc)
            return False
        for uri, label, embedding in zip(uris, labels, embeddings):
            await self.insert_predicate_embedding(uri=uri, label=label, embedding=embedding)
        self._predicate_seed_status = True
        logger.info("Seeded %d canonical predicate embeddings (lazy)", len(labels))
        return True

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
        2. Check entity_aliases table
        3. Embed the label
        4. Search existing entity embeddings
        5. If similarity >= threshold, return existing URI
        6. Otherwise create new URI and store embedding
        """
        cache_key = label.lower()
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        # Check alias table before incurring embedding cost
        async with self._pool.acquire() as conn:
            alias_row = await conn.fetchrow(
                "SELECT canonical FROM entity_aliases WHERE alias = $1",
                cache_key,
            )
        if alias_row is not None:
            uri = alias_row["canonical"]
            self._entity_cache[cache_key] = uri
            return uri

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

        # Ensure canonical vocabulary is present before similarity lookup.
        # First-call-after-startup pays the seeding cost; subsequent calls are no-ops.
        await self.ensure_predicates_seeded()

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
