# Query Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Classify user questions by intent (semantic/entity/graph), then route to the optimal retrieval strategy in RAGRetriever.

**Architecture:** New `QueryClassifier` LLM client classifies questions into 3 intents. `RAGRetriever.retrieve()` dispatches to intent-specific strategy methods. New `get_triples_by_object()` and `find_connecting_triples()` methods on KnowledgeStore for graph-mode retrieval.

**Tech Stack:** OpenAI-compatible chat API (qwen3:14b), pyoxigraph SPARQL, asyncpg

**Spec:** `docs/superpowers/specs/2026-03-23-query-routing-design.md`

**Status:** COMPLETE — merged as PR #15, version 0.1.25. 458 tests passing. Deployed and verified in production.

**Production verification:**
- `"find information about cold exposure benefits"` → classified as `semantic`
- `"what is dopamine?"` → classified as `entity`, returned Entity knowledge type
- `"how is cold exposure connected to dopamine?"` → classified as `graph`, returned Claim + Entity + Fact + Relationship types
- Classifier exception handling broadened to `httpx.HTTPError` (catches ConnectError, NetworkError)

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/knowledge_service/clients/classifier.py` | QueryClassifier + QueryIntent |
| Modify | `src/knowledge_service/stores/knowledge.py` | Add `get_triples_by_object()`, `find_connecting_triples()` |
| Modify | `src/knowledge_service/stores/rag.py` | Add `intent` param, 3 strategy methods |
| Modify | `src/knowledge_service/api/ask.py` | Classify before retrieve, `intent` in response |
| Modify | `src/knowledge_service/main.py` | Initialize QueryClassifier |
| Create | `tests/test_classifier.py` | Classifier tests |
| Modify | `tests/test_knowledge_store.py` | get_triples_by_object + find_connecting tests |
| Modify | `tests/test_rag_retriever.py` | Strategy dispatch tests |
| Modify | `tests/test_api_ask.py` | Intent in response tests |

---

## Task 1: QueryClassifier client

**Files:**
- Create: `src/knowledge_service/clients/classifier.py`
- Create: `tests/test_classifier.py`

- [x] **Step 1: Write failing tests**

Create `tests/test_classifier.py`:

```python
import json
import pytest
from knowledge_service.clients.classifier import QueryClassifier, QueryIntent

_BASE = "http://llm-test"
_KEY = "sk-test"
_CHAT_URL = f"{_BASE}/v1/chat/completions"


def _make_response(intent: str, entities: list[str]) -> dict:
    return {
        "choices": [{"message": {"content": json.dumps({"intent": intent, "entities": entities})}}]
    }


class TestClassify:
    async def test_returns_semantic_intent(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_response("semantic", []))
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("find articles about stress")
        assert result.intent == "semantic"
        assert result.entities == []
        await c.close()

    async def test_returns_entity_intent_with_entities(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_response("entity", ["dopamine"]))
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("what is dopamine?")
        assert result.intent == "entity"
        assert "dopamine" in result.entities
        await c.close()

    async def test_returns_graph_intent(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL, json=_make_response("graph", ["cortisol", "inflammation"])
        )
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("how is cortisol connected to inflammation?")
        assert result.intent == "graph"
        assert len(result.entities) == 2
        await c.close()

    async def test_falls_back_to_semantic_on_http_error(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, status_code=500)
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("some question")
        assert result.intent == "semantic"
        assert result.entities == []
        await c.close()

    async def test_falls_back_to_semantic_on_bad_json(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json={"choices": [{"message": {"content": "not json {{"}}]},
        )
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("some question")
        assert result.intent == "semantic"
        await c.close()

    async def test_falls_back_on_invalid_intent(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_response("unknown_type", []))
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("question")
        assert result.intent == "semantic"
        await c.close()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [x] **Step 3: Implement QueryClassifier**

Create `src/knowledge_service/clients/classifier.py`:

```python
"""QueryClassifier — LLM-based question intent classification."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

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


class QueryClassifier:
    """Classifies questions into retrieval intent types via LLM."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self._model = model
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        )

    async def classify(self, question: str) -> QueryIntent:
        """Classify a question into a retrieval intent.

        Returns QueryIntent with intent and extracted entities.
        Falls back to 'semantic' on any failure.
        """
        prompt = _CLASSIFICATION_PROMPT.format(question=question)
        try:
            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
        except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
            logger.warning("QueryClassifier: LLM call failed, defaulting to semantic: %s", exc)
            return QueryIntent(intent="semantic")

        raw = response.json()["choices"][0]["message"]["content"]
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
        stripped = re.sub(r"\n?```\s*$", "", stripped)

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("QueryClassifier: bad JSON response, defaulting to semantic")
            return QueryIntent(intent="semantic")

        intent = parsed.get("intent", "semantic")
        if intent not in _VALID_INTENTS:
            logger.warning("QueryClassifier: invalid intent '%s', defaulting to semantic", intent)
            intent = "semantic"

        entities = parsed.get("entities", [])
        if not isinstance(entities, list):
            entities = []

        logger.info("QueryClassifier: question='%s' → intent=%s, entities=%s", question[:80], intent, entities)
        return QueryIntent(intent=intent, entities=entities)

    async def close(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()
```

- [x] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: ALL PASS

- [x] **Step 5: Commit**

```bash
git add src/knowledge_service/clients/classifier.py tests/test_classifier.py
git commit -m "feat: add QueryClassifier for LLM-based intent classification"
```

---

## Task 2: KnowledgeStore — get_triples_by_object and find_connecting_triples

**Files:**
- Modify: `src/knowledge_service/stores/knowledge.py` (after `get_triples_by_predicate`, ~line 370)
- Modify: `tests/test_knowledge_store.py`

- [x] **Step 1: Write failing tests**

Add to `tests/test_knowledge_store.py`:

```python
class TestGetTriplesByObject:
    def test_finds_triples_where_entity_is_object(self, store):
        store.insert_triple(
            "http://s/a", "http://p/causes", "http://o/target", 0.8, "Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        results = store.get_triples_by_object("http://o/target")
        assert len(results) == 1
        assert "subject" in results[0]
        assert "graph" in results[0]

    def test_returns_empty_for_no_match(self, store):
        results = store.get_triples_by_object("http://o/nonexistent")
        assert results == []

    def test_respects_graph_filter(self, store):
        store.insert_triple(
            "http://s/1", "http://p/1", "http://o/shared", 0.8, "Claim",
            graph=KS_GRAPH_ASSERTED,
        )
        store.insert_triple(
            "http://s/2", "http://p/2", "http://o/shared", 0.7, "Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        results = store.get_triples_by_object(
            "http://o/shared", graphs=[KS_GRAPH_ASSERTED]
        )
        assert len(results) == 1


class TestFindConnectingTriples:
    def test_finds_direct_connection(self, store):
        store.insert_triple(
            "http://e/a", "http://p/causes", "http://e/b", 0.8, "Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        results = store.find_connecting_triples("http://e/a", "http://e/b")
        assert len(results) >= 1

    def test_finds_reverse_connection(self, store):
        store.insert_triple(
            "http://e/b", "http://p/causes", "http://e/a", 0.8, "Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        results = store.find_connecting_triples("http://e/a", "http://e/b")
        assert len(results) >= 1

    def test_returns_empty_when_not_connected(self, store):
        store.insert_triple(
            "http://e/x", "http://p/1", "http://e/y", 0.8, "Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        results = store.find_connecting_triples("http://e/x", "http://e/z")
        assert results == []
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_knowledge_store.py::TestGetTriplesByObject -v`
Expected: FAIL — `AttributeError: 'KnowledgeStore' has no attribute 'get_triples_by_object'`

- [x] **Step 3: Implement get_triples_by_object**

First, add the import at the top of `knowledge.py`:
```python
from knowledge_service._utils import _rdf_value_to_str
```

Then add to `KnowledgeStore` class, after `get_triples_by_predicate()`:

```python
def get_triples_by_object(
    self,
    object_uri: str,
    graphs: list[str] | None = None,
) -> list[dict]:
    """Get all annotated triples where the given URI appears as the object."""
    graph_filter = ""
    if graphs:
        values = " ".join(f"<{g}>" for g in graphs)
        graph_filter = f"VALUES ?g {{ {values} }}"

    obj_sparql = _sparql_object(object_uri)

    sparql = f"""
        SELECT ?g ?s ?p ?conf ?ktype ?vfrom ?vuntil WHERE {{
            {graph_filter}
            GRAPH ?g {{
                ?s ?p {obj_sparql} .
            }}
            OPTIONAL {{
                GRAPH ?g {{
                    << ?s ?p {obj_sparql} >>
                        <{KS_CONFIDENCE.value}> ?conf .
                }}
            }}
            OPTIONAL {{
                GRAPH ?g {{
                    << ?s ?p {obj_sparql} >>
                        <{KS_KNOWLEDGE_TYPE.value}> ?ktype .
                }}
            }}
            OPTIONAL {{
                GRAPH ?g {{
                    << ?s ?p {obj_sparql} >>
                        <{KS_VALID_FROM.value}> ?vfrom .
                }}
            }}
            OPTIONAL {{
                GRAPH ?g {{
                    << ?s ?p {obj_sparql} >>
                        <{KS_VALID_UNTIL.value}> ?vuntil .
                }}
            }}
            FILTER(BOUND(?conf))
        }}
        ORDER BY DESC(?conf)
        LIMIT 20
    """
    query_result = self._store.query(sparql)
    results = []
    for solution in query_result:
        row = {
            "graph": solution["g"].value,
            "subject": solution["s"],
            "predicate": solution["p"],
            "object": NamedNode(object_uri) if _is_uri(object_uri) else Literal(object_uri),
            "confidence": float(solution["conf"].value) if solution["conf"] else None,
            "knowledge_type": _strip_ks_prefix(solution["ktype"].value)
            if solution["ktype"]
            else None,
            "valid_from": solution["vfrom"].value if solution["vfrom"] else None,
            "valid_until": solution["vuntil"].value if solution["vuntil"] else None,
        }
        results.append(row)
    return results
```

- [x] **Step 4: Implement find_connecting_triples**

```python
def find_connecting_triples(
    self,
    entity_a: str,
    entity_b: str,
    graphs: list[str] | None = None,
) -> list[dict]:
    """Find 1-2 hop connecting triples between two entities."""
    graph_filter = ""
    if graphs:
        values = " ".join(f"<{g}>" for g in graphs)
        graph_filter = f"VALUES ?g {{ {values} }}"

    # 1-hop: A directly relates to B (either direction)
    sparql_1hop = f"""
        SELECT ?g ?s ?p ?o ?conf WHERE {{
            {graph_filter}
            GRAPH ?g {{
                {{ <{entity_a}> ?p <{entity_b}> . BIND(<{entity_a}> AS ?s) BIND(<{entity_b}> AS ?o) }}
                UNION
                {{ <{entity_b}> ?p <{entity_a}> . BIND(<{entity_b}> AS ?s) BIND(<{entity_a}> AS ?o) }}
            }}
            OPTIONAL {{
                GRAPH ?g {{
                    << ?s ?p ?o >> <{KS_CONFIDENCE.value}> ?conf .
                }}
            }}
        }}
    """
    results_1hop = self._store.query(sparql_1hop)
    results = []
    for sol in results_1hop:
        results.append({
            "subject": _rdf_value_to_str(sol["s"]),
            "predicate": _rdf_value_to_str(sol["p"]),
            "object": _rdf_value_to_str(sol["o"]),
            "confidence": float(sol["conf"].value) if sol.get("conf") and sol["conf"] else None,
        })

    if results:
        return results

    # 2-hop: A -> mid -> B (only if 1-hop found nothing)
    sparql_2hop = f"""
        SELECT ?g ?mid ?p1 ?p2 ?conf1 ?conf2 WHERE {{
            {graph_filter}
            GRAPH ?g {{
                <{entity_a}> ?p1 ?mid .
                ?mid ?p2 <{entity_b}> .
            }}
            FILTER(?mid != <{entity_a}> && ?mid != <{entity_b}>)
            OPTIONAL {{
                GRAPH ?g {{
                    << <{entity_a}> ?p1 ?mid >> <{KS_CONFIDENCE.value}> ?conf1 .
                }}
            }}
            OPTIONAL {{
                GRAPH ?g {{
                    << ?mid ?p2 <{entity_b}> >> <{KS_CONFIDENCE.value}> ?conf2 .
                }}
            }}
        }}
        LIMIT 10
    """
    results_2hop = self._store.query(sparql_2hop)
    for sol in results_2hop:
        mid = _rdf_value_to_str(sol["mid"])
        results.append({
            "subject": entity_a,
            "predicate": _rdf_value_to_str(sol["p1"]),
            "object": mid,
            "confidence": float(sol["conf1"].value) if sol.get("conf1") and sol["conf1"] else None,
        })
        results.append({
            "subject": mid,
            "predicate": _rdf_value_to_str(sol["p2"]),
            "object": entity_b,
            "confidence": float(sol["conf2"].value) if sol.get("conf2") and sol["conf2"] else None,
        })

    return results
```

Note: add `from knowledge_service._utils import _rdf_value_to_str` at the top of `knowledge.py`.

- [x] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_knowledge_store.py -v`
Expected: ALL PASS

- [x] **Step 6: Commit**

```bash
git add src/knowledge_service/stores/knowledge.py tests/test_knowledge_store.py
git commit -m "feat: add get_triples_by_object and find_connecting_triples"
```

---

## Task 3: RAGRetriever strategy dispatch

**Files:**
- Modify: `src/knowledge_service/stores/rag.py`
- Modify: `tests/test_rag_retriever.py`

- [x] **Step 1: Write failing tests for intent-based dispatch**

Add to `tests/test_rag_retriever.py`:

Use the existing helper functions pattern from the test file (`_make_embedding_client`, `_make_embedding_store`, `_make_knowledge_store`):

```python
from knowledge_service.clients.classifier import QueryIntent


class TestIntentDispatch:
    async def test_semantic_intent_uses_full_search(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        ks = _make_knowledge_store()
        retriever = RAGRetriever(ec, es, ks)
        intent = QueryIntent(intent="semantic", entities=[])
        context = await retriever.retrieve("find articles about stress", intent=intent)
        es.search.assert_called_once()

    async def test_entity_intent_resolves_entities(self):
        ec = _make_embedding_client()
        ec.embed_batch.return_value = [[0.1] * 768]
        es = _make_embedding_store()
        es.search_entities.return_value = [
            {"uri": "http://knowledge.local/data/dopamine", "similarity": 0.9}
        ]
        ks = _make_knowledge_store()
        retriever = RAGRetriever(ec, es, ks)
        intent = QueryIntent(intent="entity", entities=["dopamine"])
        context = await retriever.retrieve("what is dopamine?", intent=intent)
        assert len(context.entities_found) >= 1

    async def test_none_intent_defaults_to_semantic(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        ks = _make_knowledge_store()
        retriever = RAGRetriever(ec, es, ks)
        context = await retriever.retrieve("some question")
        es.search.assert_called_once()

    async def test_graph_intent_uses_bidirectional_lookup(self):
        ec = _make_embedding_client()
        ec.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]
        es = _make_embedding_store()
        es.search_entities.return_value = [
            {"uri": "http://knowledge.local/data/cortisol", "similarity": 0.9}
        ]
        ks = _make_knowledge_store()
        ks.get_triples_by_object.return_value = []
        ks.find_connecting_triples.return_value = []
        retriever = RAGRetriever(ec, es, ks)
        intent = QueryIntent(intent="graph", entities=["cortisol", "inflammation"])
        context = await retriever.retrieve("how is cortisol connected to inflammation?", intent=intent)
        # Should call get_triples_by_object (bidirectional)
        ks.get_triples_by_object.assert_called()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rag_retriever.py::TestIntentDispatch -v`
Expected: FAIL — `retrieve() got unexpected keyword argument 'intent'`

- [x] **Step 3: Refactor RAGRetriever with strategy dispatch**

Rewrite `src/knowledge_service/stores/rag.py`. The existing `retrieve()` body becomes `_retrieve_semantic()`. Add `intent` parameter and dispatch logic:

```python
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
            return await self._retrieve_semantic(
                question, embedding, max_sources, min_confidence
            )
        elif intent.intent == "entity":
            return await self._retrieve_entity(
                question, embedding, intent.entities, max_sources, min_confidence
            )
        elif intent.intent == "graph":
            return await self._retrieve_graph(
                question, embedding, intent.entities, max_sources, min_confidence
            )
        else:
            return await self._retrieve_semantic(
                question, embedding, max_sources, min_confidence
            )

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
            return await self._retrieve_semantic(
                question, embedding, max_sources, min_confidence
            )
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
            return await self._retrieve_semantic(
                question, embedding, max_sources, min_confidence
            )
        # Bidirectional triple lookup
        triples = await self._lookup_triples_by_subject(resolved_uris)
        obj_triples = await self._lookup_triples_by_object(resolved_uris)
        all_triples = triples + obj_triples
        # Connecting paths between entities (if 2+)
        if len(resolved_uris) >= 2:
            connections = await asyncio.to_thread(
                self._knowledge_store.find_connecting_triples,
                resolved_uris[0], resolved_uris[1],
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
            rows = await self._embedding_store.search_entities(
                query_embedding=emb, limit=1
            )
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
            triples = await asyncio.to_thread(
                self._knowledge_store.get_triples_by_subject, uri
            )
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
            triples = await asyncio.to_thread(
                self._knowledge_store.get_triples_by_object, uri
            )
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
            t for t in triples
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
            contras = await asyncio.to_thread(
                self._knowledge_store.find_contradictions, s, p, o
            )
            for c in contras:
                contradictions.append({
                    "subject": s, "predicate": p,
                    "object": str(c["object"]), "confidence": c.get("confidence"),
                })
        return contradictions
```

- [x] **Step 4: Run tests**

Run: `uv run pytest tests/test_rag_retriever.py -v`
Expected: ALL PASS

- [x] **Step 5: Commit**

```bash
git add src/knowledge_service/stores/rag.py tests/test_rag_retriever.py
git commit -m "feat: intent-based retrieval strategies in RAGRetriever"
```

---

## Task 4: Wire into ask.py and main.py

**Files:**
- Modify: `src/knowledge_service/api/ask.py:40-46,49-60`
- Modify: `src/knowledge_service/main.py`
- Modify: `tests/test_api_ask.py`

- [x] **Step 1: Add intent field to AskResponse**

In `ask.py`, add to `AskResponse` (line 46):

```python
class AskResponse(BaseModel):
    answer: str
    confidence: float | None
    sources: list[SourceInfo]
    knowledge_types_used: list[str]
    contradictions: list[ContradictionInfo]
    evidence: list[EvidenceSnippet] = []
    intent: str | None = None
```

- [x] **Step 2: Add classification to post_ask**

In `post_ask()`, before `retriever.retrieve()` (around line 56):

```python
# Classify query intent
classifier = getattr(request.app.state, "query_classifier", None)
intent = None
if classifier:
    intent = await classifier.classify(body.question)

context = await retriever.retrieve(
    body.question,
    max_sources=body.max_sources,
    min_confidence=body.min_confidence,
    intent=intent,
)
```

At the return, add `intent=intent.intent if intent else None`.

- [x] **Step 3: Initialize classifier in main.py**

In `main.py` lifespan, after the reasoning engine initialization, add:

```python
from knowledge_service.clients.classifier import QueryClassifier  # noqa: PLC0415

app.state.query_classifier = QueryClassifier(
    base_url=settings.llm_base_url,
    model=settings.llm_chat_model,
    api_key=settings.llm_api_key,
)
```

Add `await app.state.query_classifier.close()` in the shutdown section.

- [x] **Step 4: Add test for intent in response**

Add to `tests/test_api_ask.py`:

```python
class TestAskIntent:
    async def test_response_includes_intent_field(self, client):
        response = await client.post("/api/ask", json={"question": "test question"})
        assert response.status_code == 200
        data = response.json()
        assert "intent" in data
```

- [x] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [x] **Step 6: Lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`

- [x] **Step 7: Commit**

```bash
git add src/knowledge_service/api/ask.py src/knowledge_service/main.py tests/test_api_ask.py
git commit -m "feat: wire query classification into /api/ask pipeline"
```

---

## Task 5: Final integration test and lint

- [x] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [x] **Step 2: Lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`
If needed: `uv run ruff format .`

- [x] **Step 3: Commit if needed**

```bash
git add -A && git commit -m "chore: lint fixes for query routing"
```
