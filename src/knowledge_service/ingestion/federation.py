"""Background federation enrichment — looks up entities on DBpedia/Wikidata after ingestion."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from knowledge_service.ontology.namespaces import KS_GRAPH_ASSERTED

logger = logging.getLogger(__name__)

_OWL_SAME_AS = "http://www.w3.org/2002/07/owl#sameAs"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDFS_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"


@dataclass
class FederationResult:
    entities_enriched: int = 0
    entities_skipped: int = 0
    errors: int = 0


class FederationPhase:
    """Enrich extracted entities with external knowledge from DBpedia/Wikidata."""

    def __init__(
        self,
        federation_client: Any,
        triple_store: Any,
        max_lookups: int = 10,
        delay: float = 1.0,
    ) -> None:
        self._client = federation_client
        self._store = triple_store
        self._max_lookups = max_lookups
        self._delay = delay

    async def run(self, entities: list[dict]) -> FederationResult:
        """Enrich entities with external URIs and metadata.

        Args:
            entities: List of dicts with 'label' and 'uri' keys.

        Returns:
            FederationResult with counts.
        """
        result = FederationResult()
        lookups = 0

        for entity in entities:
            if lookups >= self._max_lookups:
                break

            label = entity.get("label", "")
            local_uri = entity.get("uri", "")
            if not label or not local_uri:
                continue

            # Skip if already has owl:sameAs
            existing = self._store.get_triples(subject=local_uri, predicate=_OWL_SAME_AS)
            if existing:
                result.entities_skipped += 1
                continue

            try:
                match = await self._client.lookup_entity(label)
                lookups += 1
            except Exception:
                logger.warning("Federation lookup failed for %s", label, exc_info=True)
                result.errors += 1
                continue

            if match is None:
                continue

            # Insert owl:sameAs triple
            external_uri = match["uri"]
            self._store.insert(
                subject=local_uri,
                predicate=_OWL_SAME_AS,
                object_=external_uri,
                graph=KS_GRAPH_ASSERTED,
                confidence=0.95,
                knowledge_type="Fact",
            )

            # Insert rdf:type if available
            if match.get("rdf_type"):
                self._store.insert(
                    subject=local_uri,
                    predicate=_RDF_TYPE,
                    object_=match["rdf_type"],
                    graph=KS_GRAPH_ASSERTED,
                    confidence=0.9,
                    knowledge_type="Fact",
                )

            # Insert description if available
            if match.get("description"):
                self._store.insert(
                    subject=local_uri,
                    predicate=_RDFS_COMMENT,
                    object_=match["description"],
                    graph=KS_GRAPH_ASSERTED,
                    confidence=0.9,
                    knowledge_type="Fact",
                )

            result.entities_enriched += 1
            logger.info("Federation: %s -> %s", label, external_uri)

            if self._delay > 0 and lookups < self._max_lookups:
                await asyncio.sleep(self._delay)

        return result
