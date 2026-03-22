# Query Intent Classification and Routing

**Date:** 2026-03-23
**Phase:** 7 of 9 (KG-RAG improvement roadmap)
**Scope:** Classify user questions by intent, then route to the optimal retrieval strategy

---

## Context

Currently every question goes through the same retrieval pipeline: embed → hybrid BM25+vector → entity discovery → graph lookup → contradictions → LLM answer. This is wasteful — a question like "what is dopamine?" doesn't need heavy content search, and "find articles about stress" doesn't need graph traversal.

Query routing classifies the question first, then chooses the retrieval strategy that best serves it. This improves both retrieval quality (right context for the question type) and efficiency (skip unnecessary retrieval steps).

**Dependencies:** Phase 4 (BM25 hybrid search) — provides the keyword retrieval path that `semantic` intent routes to.

---

## Design

### Query Classifier

New component: `QueryClassifier` in `src/knowledge_service/clients/classifier.py`.

**Input:** question string
**Output:** `QueryIntent` dataclass

```python
@dataclass
class QueryIntent:
    intent: Literal["semantic", "entity", "graph"]
    entities: list[str]
```

**Intent types:**

| Intent | When | Example questions |
|--------|------|-------------------|
| `semantic` | Searching for documents about a topic | "find articles about stress management", "what research exists on cold exposure" |
| `entity` | Asking about a specific thing | "what is dopamine?", "tell me about PostgreSQL", "describe cold water immersion" |
| `graph` | Asking about relationships between things | "how is cortisol connected to inflammation?", "what causes dopamine release?", "what depends on PostgreSQL?" |

**Classification prompt:**

```
Classify this question into one category:
- "semantic": searching for documents about a topic (e.g., "find articles about stress management")
- "entity": asking about a specific thing (e.g., "what is dopamine?", "tell me about PostgreSQL")
- "graph": asking about relationships between things (e.g., "how is cortisol connected to inflammation?", "what causes dopamine release?")

Also extract any named entities mentioned in the question.

Return JSON: {"intent": "semantic|entity|graph", "entities": ["entity1", "entity2"]}

Question: {question}
```

Uses the same LLM endpoint as other clients (qwen3:14b via LiteLLM). Small prompt, fast response.

**Fallback:** If classification fails (LLM error, bad JSON, invalid intent), default to `semantic` — the current behavior. This ensures the system never breaks on classifier failure.

### Retrieval Strategies

`RAGRetriever.retrieve()` gains an optional `intent: QueryIntent | None` parameter. Dispatches to a private strategy method based on intent. If `None`, defaults to `semantic`.

#### `semantic` strategy (current behavior, minor tune)

1. Embed question
2. Hybrid search (vector + BM25) — `max_sources` content chunks
3. Entity discovery (top 3 entities via embedding similarity)
4. Graph lookup for discovered entities (`get_triples_by_subject`)
5. Contradiction detection

Same as today but entity discovery reduced from 5 → 3 since content chunks are the focus.

#### `entity` strategy

1. Embed question
2. Resolve entity names from classifier to URIs:
   - Batch-embed entity names via `embedding_client.embed_batch(entity_names)` (1 network call)
   - For each embedding, call `embedding_store.search_entities(embedding, limit=1)`
   - Accept match only if similarity >= 0.80 (slightly lower than EntityResolver's 0.85 to be more permissive at query time)
   - For unmatched names, construct URI slug via `to_entity_uri()` from `clients/llm.py` and check `get_triples_by_subject()` — if non-empty, use it
   - Skip entities that can't be resolved
3. Get all triples for resolved entities (`get_triples_by_subject`)
4. Light hybrid search (top 3 chunks) for supporting text
5. Contradiction detection

Key difference: graph triples are primary context, content chunks are secondary.

**Latency:** 1 embed_batch call + N search_entities calls + graph lookups. ~2-4s total.

#### `graph` strategy

1. Embed question
2. Resolve entity names to URIs (same as entity strategy above)
3. For each entity, get triples as **both subject and object** (bidirectional):
   - `get_triples_by_subject(entity)` — what does this entity do?
   - `get_triples_by_object(entity)` — what affects this entity? (NEW method)
4. If 2+ entities resolved, find connecting path via `find_connecting_triples()` (NEW method, see below)
5. Light hybrid search (top 3 chunks) for supporting text
6. Contradiction detection

Key difference: relationship traversal is primary. Requires new `get_triples_by_object()` and `find_connecting_triples()` methods.

**Latency:** Same as entity strategy + bidirectional lookups + optional path query. ~3-5s total.

### New KnowledgeStore methods

#### `get_triples_by_object(object_uri, graphs=None)`

Find triples where an entity appears as the **object**. Same pattern as `get_triples_by_subject()` but matching `?s ?p <entity>`:

```sparql
SELECT ?g ?s ?p ?conf ?ktype ?vfrom ?vuntil WHERE {
    {graph_filter}
    GRAPH ?g {
        ?s ?p <{object_uri}> .
    }
    OPTIONAL {
        GRAPH ?g {
            << ?s ?p <{object_uri}> >>
                <{KS_CONFIDENCE}> ?conf .
        }
    }
    OPTIONAL {
        GRAPH ?g {
            << ?s ?p <{object_uri}> >>
                <{KS_KNOWLEDGE_TYPE}> ?ktype .
        }
    }
    OPTIONAL {
        GRAPH ?g {
            << ?s ?p <{object_uri}> >>
                <{KS_VALID_FROM}> ?vfrom .
        }
    }
    OPTIONAL {
        GRAPH ?g {
            << ?s ?p <{object_uri}> >>
                <{KS_VALID_UNTIL}> ?vuntil .
        }
    }
    FILTER(BOUND(?conf))
}
ORDER BY DESC(?conf)
LIMIT 20
```

Returns dict with keys: `graph`, `subject`, `predicate`, `confidence`, `knowledge_type`, `valid_from`, `valid_until`. Note: `subject` and `predicate` are pyoxigraph terms — callers must apply `_rdf_value_to_str()` before passing to downstream consumers.

Gains optional `graphs: list[str] | None` filter via `VALUES ?g { ... }` (same as other methods).

#### `find_connecting_triples(entity_a_uri, entity_b_uri, graphs=None)`

Find 1-2 hop paths between two entities. Returns connecting triples.

**1-hop query** (A directly relates to B):
```sparql
SELECT ?g ?p ?conf WHERE {
    GRAPH ?g {
        { <{entity_a}> ?p <{entity_b}> . }
        UNION
        { <{entity_b}> ?p <{entity_a}> . }
    }
    OPTIONAL {
        GRAPH ?g {
            << <{entity_a}> ?p <{entity_b}> >>
                <{KS_CONFIDENCE}> ?conf .
        }
    }
    FILTER(BOUND(?conf) || true)
}
```

**2-hop query** (A → intermediate → B, only if 1-hop returns nothing):
```sparql
SELECT ?g ?mid ?p1 ?p2 ?conf1 ?conf2 WHERE {
    GRAPH ?g {
        <{entity_a}> ?p1 ?mid .
        ?mid ?p2 <{entity_b}> .
    }
    FILTER(?mid != <{entity_a}> && ?mid != <{entity_b}>)
    OPTIONAL {
        GRAPH ?g {
            << <{entity_a}> ?p1 ?mid >> <{KS_CONFIDENCE}> ?conf1 .
        }
    }
    OPTIONAL {
        GRAPH ?g {
            << ?mid ?p2 <{entity_b}> >> <{KS_CONFIDENCE}> ?conf2 .
        }
    }
}
LIMIT 10
```

Returns a list of connecting triple dicts. For 1-hop: `{"subject": A, "predicate": p, "object": B, "confidence": conf}`. For 2-hop: two triple dicts per path (A→mid, mid→B).

If both queries return nothing, returns empty list (entities are not connected within 2 hops).

### Integration

#### ask.py changes

Before retrieval, classify the question:

```python
classifier = getattr(request.app.state, "query_classifier", None)
intent = await classifier.classify(body.question) if classifier else None
context = await retriever.retrieve(body.question, intent=intent, ...)
```

Uses `getattr` guard so tests without a classifier work (intent=None → semantic fallback).

#### AskResponse gains `intent` field

```python
class AskResponse(BaseModel):
    answer: str
    confidence: float | None
    sources: list[SourceInfo]
    knowledge_types_used: list[str]
    contradictions: list[ContradictionInfo]
    evidence: list[EvidenceSnippet] = []
    intent: str | None = None  # NEW
```

Populated from `intent.intent` if classification succeeded.

#### main.py startup

```python
app.state.query_classifier = QueryClassifier(
    base_url=settings.llm_base_url,
    model=settings.llm_chat_model,
    api_key=settings.llm_api_key,
)
```

### What does NOT change

- `/api/search` — stays pure hybrid search, no classification
- `RAGClient.answer()` — still gets `RetrievalContext`, same prompt building
- Evidence snippets, contradictions, knowledge types — unchanged
- Admin panel — unchanged
- `RetrievalContext` dataclass — unchanged

---

## File changes summary

| File | Change |
|------|--------|
| `src/knowledge_service/clients/classifier.py` | NEW: QueryClassifier + QueryIntent |
| `src/knowledge_service/stores/rag.py` | Add `intent` param, 3 strategy methods |
| `src/knowledge_service/stores/knowledge.py` | Add `get_triples_by_object()` |
| `src/knowledge_service/api/ask.py` | Classify before retrieve, add `intent` to response |
| `src/knowledge_service/main.py` | Initialize QueryClassifier at startup |
| `src/knowledge_service/api/ask.py` | `AskResponse` gains `intent` field (model is defined in ask.py, not models.py) |
| `tests/test_classifier.py` | NEW: classifier tests |
| `tests/test_rag_retriever.py` | Strategy dispatch tests |
| `tests/test_knowledge_store.py` | `get_triples_by_object` tests |

## Constraints

- Classification adds ~1-3s latency per question (1 LLM call)
- No caching of classification results (questions are unique)
- No training/fine-tuning — prompt-based classification only
- `semantic` is always the safe fallback
- No new API endpoints — classification is internal to `/api/ask`
- `get_triples_by_object()` may return many results for popular entities — limited to top 20 by confidence (ORDER BY DESC + LIMIT in SPARQL)
- RAG prompt (`build_rag_prompt`) works unchanged — all strategies produce the same `RetrievalContext` shape. The ratio of triples-to-content differs by intent but the prompt handles both uniformly. No intent-specific prompt changes needed.
- Classification result is logged at INFO level for observability (helps debug misclassification)
- Entity names from classifier go through `_rdf_value_to_str()` after graph lookups, same as existing retriever code

## Tests

- Test classifier returns valid intent for each type (semantic, entity, graph questions)
- Test classifier extracts entity names from question
- Test classifier falls back to semantic on LLM error
- Test classifier falls back to semantic on invalid JSON
- Test `get_triples_by_object()` returns triples where entity is object
- Test `get_triples_by_object()` respects graph filter
- Test `retrieve()` with semantic intent uses full hybrid search
- Test `retrieve()` with entity intent prioritizes graph triples
- Test `retrieve()` with graph intent uses bidirectional traversal
- Test `retrieve()` with None intent defaults to semantic
- Test `/api/ask` response includes `intent` field
- Test `/api/ask` works when classifier is not set on app.state
- Test graph strategy with single entity (skips path query)
- Test graph strategy when no connecting path exists (returns empty connections)
- Test `get_triples_by_object()` with no matching triples (returns empty)
- Test entity resolution with low similarity (below 0.80 threshold, skipped)
