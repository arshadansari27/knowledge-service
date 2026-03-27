"""RAGRetriever — orchestrates intent-based retrieval across content store and knowledge graph."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import httpx

from knowledge_service._utils import _extract_json, _rdf_value_to_str
from knowledge_service.ontology.namespaces import KS_GRAPH_ASSERTED, KS_GRAPH_FEDERATED
from knowledge_service.ontology.uri import to_entity_uri

logger = logging.getLogger(__name__)

_ENTITY_MATCH_THRESHOLD = 0.80
_PREDICATE_MATCH_THRESHOLD = 0.80
_PREDICATE_TRIPLE_LIMIT = 10

_VALID_INTENTS = {"semantic", "entity", "graph", "global"}

_CLASSIFICATION_PROMPT = """Classify this question into one category:
- "semantic": searching for documents about a topic (e.g., "find articles about stress management")
- "entity": asking about a specific thing (e.g., "what is dopamine?", "tell me about PostgreSQL")
- "graph": asking about relationships between things (e.g., "how is cortisol connected to inflammation?", "what causes dopamine release?")
- "global": asking about themes, summaries, or overviews across the entire knowledge base (e.g., "what are the main topics?", "summarize what I know about health", "what areas have I collected knowledge on?")

Also extract any named entities mentioned in the question.

Return JSON: {{"intent": "semantic|entity|graph|global", "entities": ["entity1", "entity2"]}}

Question: {question}"""


@dataclass
class QueryIntent:
    """Classified question intent with extracted entity names."""

    intent: str  # "semantic", "entity", "graph", or "global"
    entities: list[str] = field(default_factory=list)


@dataclass
class _TraversalResult:
    """Result of a multi-hop graph expansion."""

    edges: list[dict] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)


@dataclass
class RetrievalContext:
    content_results: list[dict] = field(default_factory=list)
    knowledge_triples: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    entities_found: list[str] = field(default_factory=list)
    traversal_depth: int | None = None
    inferred_triples: int | None = None


def _expand_graph(
    knowledge_store,
    entity_uris: str | list[str],
    max_hops: int = 2,
    min_confidence: float = 0.0,
) -> _TraversalResult:
    """BFS expansion through the knowledge graph for multi-hop retrieval."""
    if isinstance(entity_uris, str):
        entity_uris = [entity_uris]

    visited: set[str] = set()
    edges: list[dict] = []
    nodes: list[dict] = []
    frontier = [(uri, 0) for uri in entity_uris]

    while frontier:
        uri, hop = frontier.pop(0)
        if uri in visited or hop > max_hops:
            continue
        visited.add(uri)
        nodes.append({"uri": uri, "hop_distance": hop})

        triples = knowledge_store.get_triples(subject=uri)
        for t in triples:
            conf = t.get("confidence", 0)
            if conf is not None and conf >= min_confidence:
                edges.append(t)
                obj = t.get("object", "")
                if isinstance(obj, str) and obj.startswith(("http://", "https://", "urn:")):
                    if obj not in visited and hop + 1 <= max_hops:
                        frontier.append((obj, hop + 1))

    return _TraversalResult(edges=edges, nodes=nodes)


class RAGRetriever:
    def __init__(
        self,
        embedding_client,
        embedding_store,
        knowledge_store,
        community_store=None,
        entity_store=None,
        classify_client=None,
    ) -> None:
        self._embedding_client = embedding_client
        self._embedding_store = embedding_store  # ContentStore (search, get_chunks_by_ids)
        self._knowledge_store = knowledge_store  # TripleStore
        self._community_store = community_store
        # entity_store has search_entities/search_predicates; fall back to embedding_store
        # for backward compat (old EmbeddingStore had all methods)
        self._entity_store = entity_store or embedding_store
        self._classify_client = classify_client  # BaseLLMClient for query classification

    async def classify(self, question: str) -> QueryIntent:
        """Classify a question into a retrieval intent via LLM.

        Falls back to ``QueryIntent(intent="semantic")`` on any failure.
        """
        if self._classify_client is None:
            return QueryIntent(intent="semantic")

        prompt = _CLASSIFICATION_PROMPT.format(question=question)
        try:
            response = await self._classify_client.client.post(
                "/v1/chat/completions",
                json={
                    "model": self._classify_client.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("QueryClassifier: LLM call failed, defaulting to semantic: %s", exc)
            return QueryIntent(intent="semantic")

        raw = response.json()["choices"][0]["message"]["content"]
        parsed = _extract_json(raw)
        if parsed is None:
            logger.warning("QueryClassifier: bad JSON response, defaulting to semantic")
            return QueryIntent(intent="semantic")

        intent_str = parsed.get("intent", "semantic")
        if intent_str not in _VALID_INTENTS:
            logger.warning(
                "QueryClassifier: invalid intent '%s', defaulting to semantic", intent_str
            )
            intent_str = "semantic"

        entities = parsed.get("entities", [])
        if not isinstance(entities, list):
            entities = []

        logger.info(
            "QueryClassifier: question='%s' → intent=%s, entities=%s",
            question[:80],
            intent_str,
            entities,
        )
        return QueryIntent(intent=intent_str, entities=entities)

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
                question,
                embedding,
                intent.entities,
                max_sources,
                min_confidence,
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
        entity_rows = await self._entity_store.search_entities(query_embedding=embedding, limit=3)
        entities_found = [row["uri"] for row in entity_rows]
        triples = await self._lookup_triples_by_subject(entities_found)
        predicate_triples = await self._lookup_triples_by_predicate(embedding)
        merged = self._deduplicate_triples(triples + predicate_triples)
        filtered = self._filter_by_confidence(merged, min_confidence)
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
        predicate_triples = await self._lookup_triples_by_predicate(embedding)
        merged = self._deduplicate_triples(triples + predicate_triples)
        filtered = self._filter_by_confidence(merged, min_confidence)
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
    ) -> RetrievalContext:
        resolved_uris = await self._resolve_entity_names(entity_names)
        if not resolved_uris:
            return await self._retrieve_semantic(question, embedding, max_sources, min_confidence)

        # Multi-hop BFS traversal with confidence propagation
        traversal = await asyncio.to_thread(
            _expand_graph,
            self._knowledge_store,
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

        return RetrievalContext(
            content_results=content_results,
            knowledge_triples=filtered,
            contradictions=contradictions,
            entities_found=entities_found,
            traversal_depth=traversal_depth,
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

    # --- Shared helpers ---

    async def _resolve_entity_names(self, names: list[str]) -> list[str]:
        """Resolve entity names to URIs via embedding similarity or slug fallback."""
        if not names:
            return []
        embeddings = await self._embedding_client.embed_batch(names)
        resolved = []
        for name, emb in zip(names, embeddings):
            rows = await self._entity_store.search_entities(query_embedding=emb, limit=1)
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
                if graph == KS_GRAPH_ASSERTED:
                    t["trust_tier"] = "verified"
                elif graph == KS_GRAPH_FEDERATED:
                    t["trust_tier"] = "federated"
                else:
                    t["trust_tier"] = "extracted"
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
                if graph == KS_GRAPH_ASSERTED:
                    t["trust_tier"] = "verified"
                elif graph == KS_GRAPH_FEDERATED:
                    t["trust_tier"] = "federated"
                else:
                    t["trust_tier"] = "extracted"
            all_triples.extend(triples)
        return all_triples

    async def _lookup_triples_by_predicate(
        self, embedding, limit=_PREDICATE_TRIPLE_LIMIT
    ) -> list[dict]:
        """Find triples by predicate similarity to the query embedding."""
        pred_rows = await self._entity_store.search_predicates(query_embedding=embedding, limit=3)
        matched_uris = [
            r["uri"] for r in pred_rows if r.get("similarity", 0) >= _PREDICATE_MATCH_THRESHOLD
        ]
        if not matched_uris:
            return []

        all_triples = []
        for uri in matched_uris:
            triples = await asyncio.to_thread(self._knowledge_store.get_triples_by_predicate, uri)
            for t in triples:
                t["subject"] = _rdf_value_to_str(t.get("subject"))
                t["predicate"] = _rdf_value_to_str(t.get("predicate"))
                t["object"] = _rdf_value_to_str(t.get("object"))
                graph = t.get("graph", "")
                if graph == KS_GRAPH_ASSERTED:
                    t["trust_tier"] = "verified"
                elif graph == KS_GRAPH_FEDERATED:
                    t["trust_tier"] = "federated"
                else:
                    t["trust_tier"] = "extracted"
            all_triples.extend(triples)

        all_triples.sort(key=lambda t: t.get("confidence") or 0, reverse=True)
        return all_triples[:limit]

    @staticmethod
    def _deduplicate_triples(triples: list[dict]) -> list[dict]:
        """Deduplicate triples by (subject, predicate, object). First occurrence wins."""
        seen: set[tuple[str, str, str]] = set()
        result = []
        for t in triples:
            key = (t.get("subject", ""), t.get("predicate", ""), t.get("object", ""))
            if key not in seen:
                seen.add(key)
                result.append(t)
        return result

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
