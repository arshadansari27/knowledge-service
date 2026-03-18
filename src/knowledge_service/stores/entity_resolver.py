"""EntityResolver: resolves concept labels to canonical entity URIs.

On the write path, this ensures "PostgreSQL" from different sources links to
the same entity URI, preventing vocabulary drift across ingested content.
Uses embedding similarity to detect near-duplicate labels.
"""

from __future__ import annotations

import asyncio
import logging
import re

from knowledge_service.ontology.namespaces import KS_DATA, OWL

logger = logging.getLogger(__name__)


class EntityResolver:
    """Resolves concept labels to entity URIs, reusing existing entities when possible."""

    SIMILARITY_THRESHOLD = 0.85

    def __init__(self, knowledge_store, embedding_store, embedding_client, federation_client=None):
        self._knowledge_store = knowledge_store
        self._embedding_store = embedding_store
        self._embedding_client = embedding_client
        self._federation_client = federation_client

    def _slugify(self, label: str) -> str:
        """Convert label to URI-safe slug."""
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        return slug

    async def resolve(self, label: str, rdf_type: str | None = None) -> str:
        """Resolve a concept label to an entity URI.

        1. Embed the label
        2. Search for similar existing entities
        3. If match >= threshold, return existing URI
        4. If federation_client is set, look up external sources
        5. If no match anywhere, create new local URI
        """
        embedding = await self._embedding_client.embed(label)

        # Search existing entity embeddings
        candidates = await self._embedding_store.search_entities(query_embedding=embedding, limit=3)

        # Check for high-confidence match
        for candidate in candidates:
            if candidate["similarity"] >= self.SIMILARITY_THRESHOLD:
                return candidate["uri"]

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
                return uri

        # No match — create new entity
        uri = f"{KS_DATA}{self._slugify(label)}"
        await self._embedding_store.insert_entity_embedding(
            uri=uri, label=label, rdf_type=rdf_type or "", embedding=embedding
        )
        return uri
