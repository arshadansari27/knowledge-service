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
2. Use entity names from classifier to resolve URIs directly:
   - Try exact match via `search_entities()` with embedding of each entity name
   - Fall back to URI slug lookup
3. Get all triples for resolved entities (`get_triples_by_subject`)
4. Light hybrid search (top 3 chunks) for supporting text
5. Contradiction detection

Key difference: graph triples are primary context, content chunks are secondary.

#### `graph` strategy

1. Embed question
2. Resolve entity names from classifier to URIs (same as entity strategy)
3. For each entity, get triples as **both subject and object** (bidirectional):
   - `get_triples_by_subject(entity)` — what does this entity do?
   - `get_triples_by_object(entity)` — what affects this entity? (NEW method)
4. If 2+ entities found, find connecting triples via SPARQL path query (1-2 hop)
5. Light hybrid search (top 3 chunks) for supporting text
6. Contradiction detection

Key difference: relationship traversal is primary. Requires new `get_triples_by_object()` method.

### New KnowledgeStore method: `get_triples_by_object()`

For graph mode, need to find triples where an entity appears as the **object**. Same pattern as `get_triples_by_subject()` but matching `?s ?p <entity>`:

```sparql
SELECT ?g ?s ?p ?conf ?ktype ?vfrom ?vuntil WHERE {
    GRAPH ?g {
        ?s ?p <{object_uri}> .
    }
    OPTIONAL {
        GRAPH ?g {
            << ?s ?p <{object_uri}> >>
                <ks:confidence> ?conf .
        }
    }
    ... (same OPTIONAL pattern for ktype, vfrom, vuntil)
    FILTER(BOUND(?conf))
}
```

Returns same dict shape as `get_triples_by_subject()` with `graph`, `subject`, `predicate`, `confidence`, `knowledge_type`, etc.

Gains optional `graphs: list[str] | None` filter (same as `get_triples_by_subject()`).

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
| `src/knowledge_service/models.py` | `AskResponse` gains `intent` field (or update in ask.py) |
| `tests/test_classifier.py` | NEW: classifier tests |
| `tests/test_rag_retriever.py` | Strategy dispatch tests |
| `tests/test_knowledge_store.py` | `get_triples_by_object` tests |

## Constraints

- Classification adds ~1-3s latency per question (1 LLM call)
- No caching of classification results (questions are unique)
- No training/fine-tuning — prompt-based classification only
- `semantic` is always the safe fallback
- No new API endpoints — classification is internal to `/api/ask`
- `get_triples_by_object()` may return many results for popular entities — limit to top 20 by confidence

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
