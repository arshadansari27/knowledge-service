# Multi-Hop Graph Retrieval with Bayesian Confidence Propagation

**Date:** 2026-03-23
**Phase:** 8 of 9 (KG-RAG improvement roadmap)
**Scope:** BFS graph traversal up to 4 hops with multiplicative path confidence, Noisy-OR across paths, and optional ProbLog inference on discovered subgraphs

---

## Context

Current graph retrieval is limited to 1-hop (subject/object lookups) and 2-hop path finding between entity pairs. Questions like "what indirectly causes X?" or "tell me everything related to Y within a few hops" cannot be answered because the system doesn't traverse beyond immediate neighbors.

Phase 7 (query routing) added the `graph` intent strategy that uses `get_triples_by_subject`, `get_triples_by_object`, and `find_connecting_triples`. This phase replaces those shallow lookups in the graph strategy with deep BFS traversal.

**Dependencies:** Phase 7 (query routing) provides the `graph` intent that triggers deep traversal. Phases 1-3 provide the confidence annotations and ProbLog rules used for propagation.

---

## Design

### GraphTraverser

New component in `src/knowledge_service/stores/graph_traversal.py`. Pure synchronous graph traversal over KnowledgeStore — no LLM calls, no async I/O.

#### Core method: `expand()`

```python
def expand(
    self,
    seed_uris: list[str],
    max_hops: int = 4,
    min_confidence: float = 0.1,
    max_nodes: int = 50,
) -> TraversalResult:
```

**Algorithm:**
1. Initialize frontier = seed URIs (hop 0, path confidence 1.0)
2. Maintain visited set (strings) to prevent cycles (each node expanded only once)
3. For each hop (1 to `min(max_hops, 4)`):
   - For each node in current frontier:
     - Get outgoing triples via `knowledge_store.get_triples_by_subject(uri)`
     - Get incoming triples via `knowledge_store.get_triples_by_object(uri)`
     - **Normalize pyoxigraph terms:** `subject`, `predicate`, `object` fields from KnowledgeStore are pyoxigraph `NamedNode`/`Literal` objects. Apply `_rdf_value_to_str()` to convert to plain strings before use in visited set, frontier, and edge dicts.
     - **Filter out literal objects:** Only URI-valued objects go into the BFS frontier. Literal values (measurements, dates, strings) are recorded as edges but not expanded.
   - For each discovered edge to a URI neighbor:
     - For each known path to the current node, compute: new_path_confidence = path_confidence x edge_confidence
     - If new_path_confidence < min_confidence, prune (don't add to next frontier)
     - Add target node to next frontier (if not in visited)
     - Record the edge and all new paths
   - Stop early if max_nodes reached
4. After all hops, for each discovered node:
   - Collect all paths that reach it from any seed
   - Combine via Noisy-OR: `propagated_confidence = 1 - product(1 - path_conf)`
   - Set `hop_distance` to the minimum hop count across all paths to this node
5. Rank nodes by propagated confidence
6. Return TraversalResult

**Pruning:** Paths below `min_confidence` (default 0.1) are not expanded further. A 4-hop chain of 0.3-confidence edges: 0.3^4 = 0.008, pruned at hop 2 (0.3^2 = 0.09 < 0.1). This prevents fan-out explosion.

**Cycle handling:** Visited set prevents infinite loops. A node can be reached via multiple paths (all paths tracked for Noisy-OR) but expanded only once.

**Max nodes:** Safety cap at 50 nodes to bound traversal cost. Once reached, expansion stops regardless of remaining hops.

**Max hops cap:** `max_hops` is clamped to 4 inside `expand()`: `max_hops = min(max_hops, 4)`. Prevents callers from requesting unbounded traversal.

**LIMIT 20 on `get_triples_by_object`:** The existing method has `LIMIT 20` in SPARQL. For traversal, this limit should be removed or raised. Add a `limit: int | None = None` parameter to `get_triples_by_object()` — when `None`, no LIMIT clause. `GraphTraverser` calls with `limit=None`; existing callers continue to get the default 20.

**Latency estimate:** Worst case: 50 nodes x 2 SPARQL queries each = 100 queries. pyoxigraph in-memory queries take ~0.1-0.5ms each. Total: ~10-50ms for traversal. ProbLog inference on 50 facts: ~50-200ms. Total worst case: ~250ms — well within acceptable latency.

#### TraversalResult dataclass

```python
@dataclass
class TraversalResult:
    nodes: list[dict]
    # {"uri": str, "confidence": float, "hop_distance": int, "path_count": int}

    edges: list[dict]
    # {"subject": str, "predicate": str, "object": str,
    #  "confidence": float, "knowledge_type": str, "trust_tier": str}

    paths: list[list[dict]]
    # Each path is a list of edge dicts from seed to discovered node
```

`edges` uses the same dict shape as `knowledge_triples` in `RetrievalContext` — can be used directly.

`nodes` provides ranked entity URIs with propagated confidence — used for `entities_found` in `RetrievalContext`.

### Confidence propagation

**Per-path:** Multiplicative. If path is A -[0.8]-> B -[0.7]-> C, path confidence = 0.56.

**Across paths:** Noisy-OR via `ReasoningEngine.combine_evidence()`. If node C is reachable via two independent paths with confidences 0.56 and 0.42, propagated confidence = 1 - (1 - 0.56)(1 - 0.42) = 1 - 0.44 x 0.58 = 0.745.

**Edge confidence source:** `confidence` from RDF-star annotations (already present on every triple). Edges without confidence annotations are skipped (same filter as existing retrieval).

### Integration with RAGRetriever

#### Modified `_retrieve_graph()` strategy

Replace the current shallow traversal in `_retrieve_graph()`:

**Current:**
1. Resolve entities → `_lookup_triples_by_subject` + `_lookup_triples_by_object` + `find_connecting_triples`

**New:**
1. Resolve entities to URIs (existing `_resolve_entity_names`)
2. `GraphTraverser.expand(resolved_uris, max_hops=4)` via `asyncio.to_thread`
3. `TraversalResult.edges` → `knowledge_triples` in RetrievalContext
4. `TraversalResult.nodes` URIs → `entities_found`
5. Light hybrid search (top 3 chunks) for supporting text
6. Contradiction detection on traversed edges

**`semantic` and `entity` strategies remain unchanged** — they don't need deep traversal.

#### RAGRetriever constructor

`GraphTraverser` constructed internally using existing `self._knowledge_store`:
```python
self._graph_traverser = GraphTraverser(knowledge_store)
```

### Optional ProbLog inference

When `use_reasoning=True` on the `/api/ask` request and intent is `graph`:

1. Take top 50 edges from `TraversalResult` (by propagated confidence)
2. Convert to ProbLog 5-tuples: `(subject, predicate, object, confidence, {"knowledge_type": ...})`
3. Run `ReasoningEngine.infer()` with ProbLog queries (uppercase variables for unknowns):
   - `causal_propagation({_to_atom(seed_uri)}, C)` for each seed entity
   - `indirect_link({_to_atom(seed_uri)}, P, C)` for each seed entity

   Example: for seed `http://knowledge.local/data/cold_exposure`, the query is `causal_propagation('http://knowledge.local/data/cold_exposure', C)`. ProbLog will ground `C` against all matching facts and return each grounding with its probability.
4. Append inference results as additional triples with:
   - `knowledge_type: "Conclusion"`
   - `trust_tier: "inferred"`
   - `confidence:` ProbLog-computed probability

When `use_reasoning=False` (default) or intent is not `graph`: no ProbLog call.

### API changes

#### AskRequest gains `use_reasoning`

```python
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    max_sources: int = Field(5, ge=1, le=100)
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)
    use_reasoning: bool = Field(False)  # NEW
```

#### AskResponse gains traversal metadata

```python
class AskResponse(BaseModel):
    # ... existing fields ...
    intent: str | None = None
    traversal_depth: int | None = None   # NEW: max hop distance reached
    inferred_triples: int | None = None  # NEW: count of ProbLog-derived triples
```

Both nullable with defaults. Backward compatible.

---

## File changes summary

| File | Change |
|------|--------|
| `src/knowledge_service/stores/graph_traversal.py` | NEW: GraphTraverser + TraversalResult |
| `src/knowledge_service/stores/rag.py` | Replace graph strategy with GraphTraverser, add ProbLog integration |
| `src/knowledge_service/api/ask.py` | `use_reasoning` param, `traversal_depth`/`inferred_triples` in response |
| `tests/test_graph_traversal.py` | NEW: traversal tests |
| `tests/test_rag_retriever.py` | Updated graph strategy tests |
| `tests/test_api_ask.py` | New response field tests |

## Constraints

- Max 4 hops (hardcoded upper limit — `max_hops` parameter capped)
- Max 50 nodes per traversal (safety cap)
- Min confidence 0.1 for path pruning (prevents noise)
- ProbLog inference limited to top 50 edges (ProbLog doesn't scale beyond ~1000 facts)
- `use_reasoning` defaults to False (opt-in due to latency)
- `GraphTraverser` is synchronous — called via `asyncio.to_thread` from RAGRetriever
- Existing `find_connecting_triples()` remains on KnowledgeStore (not removed — keeps backward compatibility, may be used by future callers)

## Tests

- Test BFS expansion from single seed: discovers 1-hop, 2-hop, 3-hop nodes
- Test confidence propagation: 2-hop chain has multiplicative confidence
- Test Noisy-OR across paths: node reachable via 2 paths has higher confidence than either path alone
- Test pruning: low-confidence paths don't expand beyond threshold
- Test cycle handling: cycles don't cause infinite loop
- Test max_nodes cap: traversal stops at 50 nodes
- Test empty graph: no triples returns empty result
- Test graph strategy uses GraphTraverser instead of shallow lookups
- Test `use_reasoning=True` produces inferred triples
- Test `use_reasoning=False` produces no inferred triples
- Test `traversal_depth` and `inferred_triples` in AskResponse
- Test backward compat: existing graph tests still pass
