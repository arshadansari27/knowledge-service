"""RAGRetriever — orchestrates intent-based retrieval across content store and knowledge graph."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field

import httpx

from knowledge_service._utils import _extract_json
from knowledge_service.ontology.namespaces import KS_GRAPH_ASSERTED
from knowledge_service.ontology.uri import to_entity_uri

logger = logging.getLogger(__name__)

_ENTITY_MATCH_THRESHOLD = 0.80
_PREDICATE_MATCH_THRESHOLD = 0.80
_PREDICATE_TRIPLE_LIMIT = 10

_VALID_INTENTS = {"semantic", "entity", "graph"}

_CLASSIFICATION_PROMPT = """Classify this question into one category:
- "semantic": searching for documents about a topic (e.g., "find articles about stress management")
- "entity": asking about a specific thing (e.g., "what is dopamine?", "tell me about PostgreSQL")
- "graph": asking about relationships between things (e.g., "how is cortisol connected to inflammation?", "what causes dopamine release?")

Also extract any named entities mentioned in the question.

Return JSON: {{"intent": "semantic|entity|graph", "entities": ["entity1", "entity2"]}}

Question: {question}"""


@dataclass
class QueryIntent:
    """Classified question intent with extracted entity names."""

    intent: str  # "semantic", "entity", or "graph"
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
    traversal_depth: int | None = None


_MAX_TRAVERSAL_EDGES = 200
_DEFAULT_MAX_TRIPLES = 15


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
        if len(edges) >= _MAX_TRAVERSAL_EDGES:
            break
        uri, hop = frontier.pop(0)
        if uri in visited or hop > max_hops:
            continue
        visited.add(uri)
        nodes.append({"uri": uri, "hop_distance": hop})

        triples = knowledge_store.get_triples(subject=uri)
        for t in triples:
            if len(edges) >= _MAX_TRAVERSAL_EDGES:
                break
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
        entity_store=None,
        classify_client=None,
        max_triples: int = _DEFAULT_MAX_TRIPLES,
    ) -> None:
        self._embedding_client = embedding_client
        self._embedding_store = embedding_store  # ContentStore (search, get_chunks_by_ids)
        self._knowledge_store = knowledge_store  # TripleStore
        # entity_store has search_entities/search_predicates; fall back to embedding_store
        # for backward compat (old EmbeddingStore had all methods)
        self._entity_store = entity_store or embedding_store
        self._classify_client = classify_client  # BaseLLMClient for query classification
        # Cap on triples passed to the RAG prompt — prevents the triple-flood that
        # the 2026-05-31 eval showed harms answer faithfulness.
        self._max_triples = max_triples

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
        retrieval_mode: str = "full",
    ) -> RetrievalContext:
        embedding = await self._embedding_client.embed(question)

        if retrieval_mode == "chunks_only":
            content_results = await self._embedding_store.search(
                query_embedding=embedding, limit=max_sources, query_text=question
            )
            return RetrievalContext(content_results=content_results)

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
        triples = await self._lookup_triples_by_subject([row["uri"] for row in entity_rows])
        predicate_triples = await self._lookup_triples_by_predicate(embedding)
        merged = self._deduplicate_triples(triples + predicate_triples)
        filtered = self._filter_by_confidence(merged, min_confidence)
        capped = await self._rank_triples_by_relevance(filtered, embedding, self._max_triples)
        contradictions = await self._detect_contradictions(capped)
        return RetrievalContext(
            content_results=content_results,
            knowledge_triples=capped,
            contradictions=contradictions,
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
        capped = await self._rank_triples_by_relevance(filtered, embedding, self._max_triples)
        contradictions = await self._detect_contradictions(capped)
        # Light content search for supporting text
        content_results = await self._embedding_store.search(
            query_embedding=embedding, limit=3, query_text=question
        )
        return RetrievalContext(
            content_results=content_results,
            knowledge_triples=capped,
            contradictions=contradictions,
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

        # Use traversal edges as knowledge triples (relevance-ranked to the budget)
        filtered = self._filter_by_confidence(traversal.edges, min_confidence)
        capped = await self._rank_triples_by_relevance(filtered, embedding, self._max_triples)
        contradictions = await self._detect_contradictions(capped)
        content_results = await self._embedding_store.search(
            query_embedding=embedding, limit=3, query_text=question
        )

        # Traversal metadata
        traversal_depth = max((n["hop_distance"] for n in traversal.nodes), default=0)

        return RetrievalContext(
            content_results=content_results,
            knowledge_triples=capped,
            contradictions=contradictions,
            traversal_depth=traversal_depth,
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
                    self._knowledge_store.get_triples, subject=slug_uri
                )
                if triples:
                    resolved.append(slug_uri)
                else:
                    logger.info("Could not resolve entity '%s', skipping", name)
        return resolved

    async def _lookup_triples_by_subject(self, uris: list[str]) -> list[dict]:
        all_triples = []
        for uri in uris:
            triples = await asyncio.to_thread(self._knowledge_store.get_triples, subject=uri)
            for t in triples:
                graph = t.get("graph", "")
                if graph == KS_GRAPH_ASSERTED:
                    t["trust_tier"] = "verified"
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
            triples = await asyncio.to_thread(self._knowledge_store.get_triples, predicate=uri)
            for t in triples:
                graph = t.get("graph", "")
                if graph == KS_GRAPH_ASSERTED:
                    t["trust_tier"] = "verified"
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

    @staticmethod
    def _rank_and_cap_triples(triples: list[dict], limit: int) -> list[dict]:
        """Keep the top-``limit`` triples by confidence (desc).

        Triples without a confidence sort as 0. This bounds how much graph context
        reaches the RAG prompt: the eval showed an unbounded triple set (~97 avg)
        degrades answer quality versus a small, high-confidence set. Used as the
        fallback when relevance ranking is unavailable (embedding backend down).
        """
        ranked = sorted(triples, key=lambda t: t.get("confidence") or 0, reverse=True)
        return ranked[:limit]

    @staticmethod
    def _localize(term: str) -> str:
        """Render a URI as its human label (last path/hash segment, de-slugged).

        ``http://knowledge.local/data/cold_exposure`` -> ``cold exposure``.
        Literals (non-URIs) are returned unchanged.
        """
        if not isinstance(term, str):
            return str(term)
        if term.startswith(("http://", "https://", "urn:")):
            tail = term.rstrip("/").replace("#", "/").rsplit("/", 1)[-1]
            return tail.replace("_", " ")
        return term

    @classmethod
    def _triple_to_text(cls, triple: dict) -> str:
        """Render a triple as a natural phrase for embedding (subject pred object)."""
        return (
            f"{cls._localize(triple.get('subject', ''))} "
            f"{cls._localize(triple.get('predicate', ''))} "
            f"{cls._localize(triple.get('object', ''))}"
        ).strip()

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity of two equal-length vectors (0.0 on degenerate input)."""
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    async def _rank_triples_by_relevance(
        self, triples: list[dict], query_embedding: list[float], limit: int
    ) -> list[dict]:
        """Keep the top-``limit`` triples most relevant to the query.

        Each triple is rendered to text and embedded; triples are ranked by cosine
        similarity of that embedding to the question embedding. This fixes the
        failure the 2026-05-31 eval exposed: capping by *confidence* alone still
        floods the prompt with confident-but-irrelevant facts. Relevance ranking
        puts the few triples actually about the question in front of the LLM.

        Degrades gracefully to confidence ranking if the embedding backend fails,
        so a transient embedding outage never drops graph context entirely.
        """
        if not triples:
            return []
        try:
            texts = [self._triple_to_text(t) for t in triples]
            embeddings = await self._embedding_client.embed_batch(texts)
        except Exception as exc:  # embedding backend down / malformed response
            logger.warning("Triple relevance ranking failed (%s); falling back to confidence", exc)
            return self._rank_and_cap_triples(triples, limit)

        # Defensive: a wrong-length batch would let zip() silently DROP triples
        # (losing graph context with no error). Treat any length mismatch as a
        # backend fault and fall back to confidence ranking instead.
        if len(embeddings) != len(triples):
            logger.warning(
                "embed_batch returned %d vectors for %d triples; "
                "falling back to confidence ranking",
                len(embeddings),
                len(triples),
            )
            return self._rank_and_cap_triples(triples, limit)

        scored = [(self._cosine(query_embedding, emb), t) for t, emb in zip(triples, embeddings)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [t for _, t in scored[:limit]]

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
