"""RAGRetriever — orchestrates intent-based retrieval across content store and knowledge graph."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from knowledge_service._utils import _rdf_value_to_str
from knowledge_service.clients.classifier import QueryIntent
from knowledge_service.clients.llm import to_entity_uri
from knowledge_service.ontology.namespaces import KS_GRAPH_ASSERTED

logger = logging.getLogger(__name__)

_ENTITY_MATCH_THRESHOLD = 0.80


@dataclass
class RetrievalContext:
    content_results: list[dict] = field(default_factory=list)
    knowledge_triples: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    entities_found: list[str] = field(default_factory=list)


class RAGRetriever:
    def __init__(self, embedding_client, embedding_store, knowledge_store) -> None:
        self._embedding_client = embedding_client
        self._embedding_store = embedding_store
        self._knowledge_store = knowledge_store

    async def retrieve(
        self,
        question: str,
        max_sources: int = 5,
        min_confidence: float = 0.0,
        intent: QueryIntent | None = None,
    ) -> RetrievalContext:
        embedding = await self._embedding_client.embed(question)

        if intent is None or intent.intent == "semantic":
            return await self._retrieve_semantic(question, embedding, max_sources, min_confidence)
        elif intent.intent == "entity":
            return await self._retrieve_entity(
                question, embedding, intent.entities, max_sources, min_confidence
            )
        elif intent.intent == "graph":
            return await self._retrieve_graph(
                question, embedding, intent.entities, max_sources, min_confidence
            )
        else:
            return await self._retrieve_semantic(question, embedding, max_sources, min_confidence)

    # --- Strategy: semantic (current behavior) ---

    async def _retrieve_semantic(
        self, question, embedding, max_sources, min_confidence
    ) -> RetrievalContext:
        content_results = await self._embedding_store.search(
            query_embedding=embedding, limit=max_sources, query_text=question
        )
        entity_rows = await self._embedding_store.search_entities(
            query_embedding=embedding, limit=3
        )
        entities_found = [row["uri"] for row in entity_rows]
        triples = await self._lookup_triples_by_subject(entities_found)
        filtered = self._filter_by_confidence(triples, min_confidence)
        contradictions = await self._detect_contradictions(filtered)
        return RetrievalContext(
            content_results=content_results,
            knowledge_triples=filtered,
            contradictions=contradictions,
            entities_found=entities_found,
        )

    # --- Strategy: entity ---

    async def _retrieve_entity(
        self, question, embedding, entity_names, max_sources, min_confidence
    ) -> RetrievalContext:
        resolved_uris = await self._resolve_entity_names(entity_names)
        if not resolved_uris:
            # Fallback: no entities resolved, use semantic
            return await self._retrieve_semantic(question, embedding, max_sources, min_confidence)
        triples = await self._lookup_triples_by_subject(resolved_uris)
        filtered = self._filter_by_confidence(triples, min_confidence)
        contradictions = await self._detect_contradictions(filtered)
        # Light content search for supporting text
        content_results = await self._embedding_store.search(
            query_embedding=embedding, limit=3, query_text=question
        )
        return RetrievalContext(
            content_results=content_results,
            knowledge_triples=filtered,
            contradictions=contradictions,
            entities_found=resolved_uris,
        )

    # --- Strategy: graph ---

    async def _retrieve_graph(
        self, question, embedding, entity_names, max_sources, min_confidence
    ) -> RetrievalContext:
        resolved_uris = await self._resolve_entity_names(entity_names)
        if not resolved_uris:
            return await self._retrieve_semantic(question, embedding, max_sources, min_confidence)
        # Bidirectional triple lookup
        triples = await self._lookup_triples_by_subject(resolved_uris)
        obj_triples = await self._lookup_triples_by_object(resolved_uris)
        all_triples = triples + obj_triples
        # Connecting paths between entities (if 2+)
        if len(resolved_uris) >= 2:
            connections = await asyncio.to_thread(
                self._knowledge_store.find_connecting_triples,
                resolved_uris[0],
                resolved_uris[1],
            )
            for c in connections:
                c["knowledge_type"] = c.get("knowledge_type", "Relationship")
                c["trust_tier"] = "extracted"
            all_triples.extend(connections)
        filtered = self._filter_by_confidence(all_triples, min_confidence)
        contradictions = await self._detect_contradictions(filtered)
        content_results = await self._embedding_store.search(
            query_embedding=embedding, limit=3, query_text=question
        )
        return RetrievalContext(
            content_results=content_results,
            knowledge_triples=filtered,
            contradictions=contradictions,
            entities_found=resolved_uris,
        )

    # --- Shared helpers ---

    async def _resolve_entity_names(self, names: list[str]) -> list[str]:
        """Resolve entity names to URIs via embedding similarity or slug fallback."""
        if not names:
            return []
        embeddings = await self._embedding_client.embed_batch(names)
        resolved = []
        for name, emb in zip(names, embeddings):
            rows = await self._embedding_store.search_entities(query_embedding=emb, limit=1)
            if rows and rows[0].get("similarity", 0) >= _ENTITY_MATCH_THRESHOLD:
                resolved.append(rows[0]["uri"])
            else:
                # Slug fallback: check if triples exist for this URI
                slug_uri = to_entity_uri(name)
                triples = await asyncio.to_thread(
                    self._knowledge_store.get_triples_by_subject, slug_uri
                )
                if triples:
                    resolved.append(slug_uri)
                else:
                    logger.info("Could not resolve entity '%s', skipping", name)
        return resolved

    async def _lookup_triples_by_subject(self, uris: list[str]) -> list[dict]:
        all_triples = []
        for uri in uris:
            triples = await asyncio.to_thread(self._knowledge_store.get_triples_by_subject, uri)
            for t in triples:
                t["subject"] = uri
                t["predicate"] = _rdf_value_to_str(t.get("predicate"))
                t["object"] = _rdf_value_to_str(t.get("object"))
                graph = t.get("graph", "")
                t["trust_tier"] = "verified" if graph == KS_GRAPH_ASSERTED else "extracted"
            all_triples.extend(triples)
        return all_triples

    async def _lookup_triples_by_object(self, uris: list[str]) -> list[dict]:
        all_triples = []
        for uri in uris:
            triples = await asyncio.to_thread(self._knowledge_store.get_triples_by_object, uri)
            for t in triples:
                t["object"] = uri
                t["subject"] = _rdf_value_to_str(t.get("subject"))
                t["predicate"] = _rdf_value_to_str(t.get("predicate"))
                graph = t.get("graph", "")
                t["trust_tier"] = "verified" if graph == KS_GRAPH_ASSERTED else "extracted"
            all_triples.extend(triples)
        return all_triples

    @staticmethod
    def _filter_by_confidence(triples, min_confidence):
        return [
            t
            for t in triples
            if t.get("confidence") is not None and t["confidence"] >= min_confidence
        ]

    async def _detect_contradictions(self, triples):
        contradictions = []
        seen = set()
        for t in triples:
            s, p, o = t["subject"], t["predicate"], t["object"]
            key = (s, p)
            if key in seen:
                continue
            seen.add(key)
            contras = await asyncio.to_thread(self._knowledge_store.find_contradictions, s, p, o)
            for c in contras:
                contradictions.append(
                    {
                        "subject": s,
                        "predicate": p,
                        "object": str(c["object"]),
                        "confidence": c.get("confidence"),
                    }
                )
        return contradictions
