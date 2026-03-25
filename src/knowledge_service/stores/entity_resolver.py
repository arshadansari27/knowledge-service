"""EntityResolver: resolves concept labels to canonical entity URIs.

On the write path, this ensures "PostgreSQL" from different sources links to
the same entity URI, preventing vocabulary drift across ingested content.
Uses embedding similarity to detect near-duplicate labels.
"""

from __future__ import annotations

import asyncio
import logging
import re

from knowledge_service.ontology.namespaces import KS, KS_DATA, OWL

logger = logging.getLogger(__name__)

_CACHE_MAX_SIZE = 1000


class EntityResolver:
    """Resolves concept labels to entity URIs, reusing existing entities when possible."""

    SIMILARITY_THRESHOLD = 0.85
    PREDICATE_SIMILARITY_THRESHOLD = 0.90

    def __init__(self, knowledge_store, embedding_store, embedding_client, federation_client=None):
        self._knowledge_store = knowledge_store
        self._embedding_store = embedding_store
        self._embedding_client = embedding_client
        self._federation_client = federation_client
        # In-memory caches to avoid redundant embedding calls for the same label
        self._entity_cache: dict[str, str] = {}  # label_lower -> URI
        self._predicate_cache: dict[str, str] = {}  # label_lower -> URI

    def _slugify(self, label: str) -> str:
        """Convert label to URI-safe slug."""
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        return slug

    def clear_cache(self) -> None:
        """Clear resolution caches. Call between unrelated ingestion batches if needed."""
        self._entity_cache.clear()
        self._predicate_cache.clear()

    async def resolve(self, label: str, rdf_type: str | None = None) -> str:
        """Resolve a concept label to an entity URI.

        1. Check in-memory cache for previously resolved label
        2. Embed the label
        3. Search for similar existing entities
        4. If match >= threshold, return existing URI
        5. If federation_client is set, look up external sources
        6. If no match anywhere, create new local URI
        """
        cache_key = label.lower()
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        embedding = await self._embedding_client.embed(label)

        # Search existing entity embeddings
        candidates = await self._embedding_store.search_entities(query_embedding=embedding, limit=3)

        # Check for high-confidence match
        for candidate in candidates:
            if candidate["similarity"] >= self.SIMILARITY_THRESHOLD:
                uri = candidate["uri"]
                self._cache_entity(cache_key, uri)
                return uri

        # Try federation if available
        if self._federation_client is not None:
            try:
                ext = await self._federation_client.lookup_entity(label)
            except Exception:
                logger.debug("Federation lookup failed for '%s'", label)
                ext = None

            if ext and ext.get("uri"):
                uri = f"{KS_DATA}{self._slugify(label)}"
                resolved_type = ext.get("rdf_type") or rdf_type or ""

                # Store owl:sameAs triple
                await asyncio.to_thread(
                    self._knowledge_store.insert_triple,
                    uri,
                    f"{OWL}sameAs",
                    ext["uri"],
                    1.0,
                    "Entity",
                )

                # Store embedding with resolved type
                await self._embedding_store.insert_entity_embedding(
                    uri=uri, label=label, rdf_type=resolved_type, embedding=embedding
                )
                self._cache_entity(cache_key, uri)
                return uri

        # No match — create new entity
        uri = f"{KS_DATA}{self._slugify(label)}"
        await self._embedding_store.insert_entity_embedding(
            uri=uri, label=label, rdf_type=rdf_type or "", embedding=embedding
        )
        self._cache_entity(cache_key, uri)
        return uri

    async def resolve_predicate(self, label: str) -> str:
        """Resolve a predicate label to a canonical predicate URI.

        1. Check in-memory cache for previously resolved label
        2. Check synonym lookup for exact match (no embedding call needed)
        3. Embed the label and search predicate_embeddings
        4. If match >= threshold (0.90), return existing URI
        5. If no match, create new predicate URI and store embedding
        """
        cache_key = label.lower()
        if cache_key in self._predicate_cache:
            return self._predicate_cache[cache_key]

        from knowledge_service.clients.llm import (  # noqa: PLC0415
            _CANONICAL_SET,
            resolve_predicate_synonym,
        )

        # Fast path: exact synonym or canonical match (no embedding call needed)
        resolved = resolve_predicate_synonym(label)
        slug = re.sub(r"[^\w]", "_", resolved.lower().strip())
        slug = re.sub(r"_+", "_", slug).strip("_")
        if slug in _CANONICAL_SET:
            uri = f"{KS}{slug}"
            self._cache_predicate(cache_key, uri)
            return uri

        # Embed and search
        embedding = await self._embedding_client.embed(label)
        candidates = await self._embedding_store.search_predicates(
            query_embedding=embedding, limit=3
        )

        for candidate in candidates:
            if candidate["similarity"] >= self.PREDICATE_SIMILARITY_THRESHOLD:
                uri = candidate["uri"]
                self._cache_predicate(cache_key, uri)
                return uri

        # No match — create new predicate
        slug = self._slugify(label)
        uri = f"{KS}{slug}"
        await self._embedding_store.insert_predicate_embedding(
            uri=uri, label=label, embedding=embedding
        )
        self._cache_predicate(cache_key, uri)
        return uri

    def _cache_entity(self, key: str, uri: str) -> None:
        if len(self._entity_cache) >= _CACHE_MAX_SIZE:
            self._entity_cache.clear()
        self._entity_cache[key] = uri

    def _cache_predicate(self, key: str, uri: str) -> None:
        if len(self._predicate_cache) >= _CACHE_MAX_SIZE:
            self._predicate_cache.clear()
        self._predicate_cache[key] = uri
