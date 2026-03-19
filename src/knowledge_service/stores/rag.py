"""RAGRetriever — orchestrates hybrid retrieval across content store and knowledge graph."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from knowledge_service._utils import _rdf_value_to_str


@dataclass
class RetrievalContext:
    """Assembled retrieval results ready for prompt construction."""

    content_results: list[dict] = field(default_factory=list)
    knowledge_triples: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    entities_found: list[str] = field(default_factory=list)


class RAGRetriever:
    """Orchestrates retrieval across embedding store and knowledge graph.

    Pipeline:
    1. Embed the question
    2. Semantic search over content
    3. Entity discovery via entity embeddings
    4. Knowledge graph lookup per discovered entity
    5. Filter triples below min_confidence
    6. Contradiction detection
    """

    def __init__(self, embedding_client, embedding_store, knowledge_store) -> None:
        self._embedding_client = embedding_client
        self._embedding_store = embedding_store
        self._knowledge_store = knowledge_store

    async def retrieve(
        self,
        question: str,
        max_sources: int = 5,
        min_confidence: float = 0.0,
    ) -> RetrievalContext:
        # Step 1: Embed the question
        embedding = await self._embedding_client.embed(question)

        # Step 2: Content search
        content_results = await self._embedding_store.search(
            query_embedding=embedding, limit=max_sources
        )

        # Step 3: Entity discovery
        entity_rows = await self._embedding_store.search_entities(
            query_embedding=embedding, limit=5
        )
        entities_found = [row["uri"] for row in entity_rows]

        # Step 4: Knowledge graph lookup (synchronous — use asyncio.to_thread)
        all_triples: list[dict] = []
        for uri in entities_found:
            triples = await asyncio.to_thread(self._knowledge_store.get_triples_by_subject, uri)
            for t in triples:
                t["subject"] = uri
                # Stringify pyoxigraph RDF terms so downstream consumers get plain strings
                t["predicate"] = _rdf_value_to_str(t.get("predicate"))
                t["object"] = _rdf_value_to_str(t.get("object"))
            all_triples.extend(triples)

        # Step 5: Filter by min_confidence
        filtered_triples = [
            t
            for t in all_triples
            if t.get("confidence") is not None and t["confidence"] >= min_confidence
        ]

        # Step 6: Contradiction detection
        contradictions: list[dict] = []
        seen = set()
        for t in filtered_triples:
            s = t["subject"]
            p = t["predicate"]  # already a str
            o = t["object"]     # already a str
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

        return RetrievalContext(
            content_results=content_results,
            knowledge_triples=filtered_triples,
            contradictions=contradictions,
            entities_found=entities_found,
        )
