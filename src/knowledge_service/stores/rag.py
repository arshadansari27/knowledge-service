"""RAGRetriever — orchestrates intent-based retrieval across content store and knowledge graph."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from knowledge_service._utils import _rdf_value_to_str
from knowledge_service.clients.classifier import QueryIntent
from knowledge_service.clients.llm import to_entity_uri
from knowledge_service.ontology.namespaces import KS_GRAPH_ASSERTED
from knowledge_service.stores.graph_traversal import GraphTraverser

logger = logging.getLogger(__name__)

_ENTITY_MATCH_THRESHOLD = 0.80
_PREDICATE_MATCH_THRESHOLD = 0.80
_PREDICATE_TRIPLE_LIMIT = 10


@dataclass
class RetrievalContext:
    content_results: list[dict] = field(default_factory=list)
    knowledge_triples: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    entities_found: list[str] = field(default_factory=list)
    traversal_depth: int | None = None
    inferred_triples: int | None = None


class RAGRetriever:
    def __init__(
        self, embedding_client, embedding_store, knowledge_store, community_store=None
    ) -> None:
        self._embedding_client = embedding_client
        self._embedding_store = embedding_store
        self._knowledge_store = knowledge_store
        self._community_store = community_store
        self._graph_traverser = GraphTraverser(knowledge_store)

    async def retrieve(
        self,
        question: str,
        max_sources: int = 5,
        min_confidence: float = 0.0,
        intent: QueryIntent | None = None,
        use_reasoning: bool = False,
        reasoning_engine=None,
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
                question,
                embedding,
                intent.entities,
                max_sources,
                min_confidence,
                use_reasoning=use_reasoning,
                reasoning_engine=reasoning_engine,
            )
        elif intent.intent == "global":
            return await self._retrieve_global(question, embedding, max_sources, min_confidence)
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
        self,
        question,
        embedding,
        entity_names,
        max_sources,
        min_confidence,
        use_reasoning=False,
        reasoning_engine=None,
    ) -> RetrievalContext:
        resolved_uris = await self._resolve_entity_names(entity_names)
        if not resolved_uris:
            return await self._retrieve_semantic(question, embedding, max_sources, min_confidence)

        # Multi-hop BFS traversal with confidence propagation
        traversal = await asyncio.to_thread(
            self._graph_traverser.expand,
            resolved_uris,
            max_hops=4,
            min_confidence=max(min_confidence, 0.1),
        )

        # Use traversal edges as knowledge triples
        filtered = self._filter_by_confidence(traversal.edges, min_confidence)

        # Use traversal node URIs as entities found
        entities_found = resolved_uris + [n["uri"] for n in traversal.nodes[:10]]

        contradictions = await self._detect_contradictions(filtered)
        content_results = await self._embedding_store.search(
            query_embedding=embedding, limit=3, query_text=question
        )

        # Traversal metadata
        traversal_depth = max((n["hop_distance"] for n in traversal.nodes), default=0)

        # Optional ProbLog inference
        inferred_count = 0
        if use_reasoning and reasoning_engine and filtered:
            inferred = await self._run_problog_inference(
                filtered[:50], resolved_uris, reasoning_engine
            )
            filtered.extend(inferred)
            inferred_count = len(inferred)

        return RetrievalContext(
            content_results=content_results,
            knowledge_triples=filtered,
            contradictions=contradictions,
            entities_found=entities_found,
            traversal_depth=traversal_depth,
            inferred_triples=inferred_count,
        )

    # --- Strategy: global ---

    async def _retrieve_global(
        self, question, embedding, max_sources, min_confidence
    ) -> RetrievalContext:
        """Global strategy: use community summaries for corpus-level questions."""
        if not self._community_store:
            return await self._retrieve_semantic(question, embedding, max_sources, min_confidence)

        communities = await self._community_store.get_all()
        if not communities:
            return await self._retrieve_semantic(question, embedding, max_sources, min_confidence)

        # Build knowledge triples from community summaries
        question_words = set(question.lower().split())
        triples = []
        level0_keyword_matched = False

        for c in communities:
            # Level 1 (coarse): always include
            # Level 0 (fine): include if keyword match
            if c["level"] == 1:
                include = True
            else:
                summary_words = set((c.get("summary") or "").lower().split())
                matched = bool(question_words & summary_words)
                if matched:
                    level0_keyword_matched = True
                include = matched

            if include and c.get("summary"):
                triples.append(
                    {
                        "subject": f"community_{c.get('id', 'unknown')}",
                        "predicate": "has_summary",
                        "object": c["summary"],
                        "confidence": 1.0,
                        "knowledge_type": "Community",
                        "trust_tier": "computed",
                    }
                )

        # If no level-0 matched by keyword, add top 5 by member count
        if not level0_keyword_matched:
            level0 = [c for c in communities if c["level"] == 0 and c.get("summary")]
            for c in level0[:5]:
                triples.append(
                    {
                        "subject": f"community_{c.get('id', 'unknown')}",
                        "predicate": "has_summary",
                        "object": c["summary"],
                        "confidence": 1.0,
                        "knowledge_type": "Community",
                        "trust_tier": "computed",
                    }
                )

        # Light content search for grounding
        content_results = await self._embedding_store.search(
            query_embedding=embedding, limit=3, query_text=question
        )

        return RetrievalContext(
            content_results=content_results,
            knowledge_triples=triples,
            contradictions=[],
            entities_found=[],
        )

    # --- ProbLog inference ---

    @staticmethod
    def _uri_to_local(uri: str) -> str:
        """Strip URI to local name for ProbLog atom compatibility.

        ProbLog rules use short predicate names (causes, increases) but graph
        data uses full URIs (http://knowledge.local/schema/causes). This
        extracts the local name so ProbLog rules can match.
        """
        if "/" in uri:
            return uri.rsplit("/", 1)[-1]
        return uri

    async def _run_problog_inference(self, edges, seed_uris, reasoning_engine):
        """Run ProbLog causal/indirect inference on discovered subgraph."""
        from knowledge_service.reasoning.engine import _to_atom

        claims = []
        for e in edges:
            claims.append(
                (
                    self._uri_to_local(e["subject"]),
                    self._uri_to_local(e["predicate"]),
                    self._uri_to_local(e["object"]),
                    e.get("confidence") or 0.5,
                    {"knowledge_type": e.get("knowledge_type", "Claim")},
                )
            )

        inferred = []
        for seed in seed_uris:
            seed_atom = _to_atom(self._uri_to_local(seed))
            for query_template in [
                f"causal_propagation({seed_atom}, C)",
                f"indirect_link({seed_atom}, P, C)",
            ]:
                results = await asyncio.to_thread(reasoning_engine.infer, query_template, claims)
                for r in results:
                    if r.probability > 0.05:
                        inferred.append(
                            {
                                "subject": seed,
                                "predicate": r.query,
                                "object": str(r.probability),
                                "confidence": r.probability,
                                "knowledge_type": "Conclusion",
                                "trust_tier": "inferred",
                            }
                        )
        return inferred

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

    async def _lookup_triples_by_predicate(
        self, embedding, limit=_PREDICATE_TRIPLE_LIMIT
    ) -> list[dict]:
        """Find triples by predicate similarity to the query embedding."""
        pred_rows = await self._embedding_store.search_predicates(
            query_embedding=embedding, limit=3
        )
        matched_uris = [
            r["uri"] for r in pred_rows if r.get("similarity", 0) >= _PREDICATE_MATCH_THRESHOLD
        ]
        if not matched_uris:
            return []

        all_triples = []
        for uri in matched_uris:
            triples = await asyncio.to_thread(
                self._knowledge_store.get_triples_by_predicate, uri
            )
            for t in triples:
                t["subject"] = _rdf_value_to_str(t.get("subject"))
                t["predicate"] = _rdf_value_to_str(t.get("predicate"))
                t["object"] = _rdf_value_to_str(t.get("object"))
                graph = t.get("graph", "")
                t["trust_tier"] = "verified" if graph == KS_GRAPH_ASSERTED else "extracted"
            all_triples.extend(triples)

        all_triples.sort(key=lambda t: t.get("confidence") or 0, reverse=True)
        return all_triples[:limit]

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
