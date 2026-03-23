# Multi-Hop Graph Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add BFS graph traversal up to 4 hops with Bayesian confidence propagation (multiplicative per-path, Noisy-OR across paths) and optional ProbLog inference on discovered subgraphs.

**Architecture:** New `GraphTraverser` class performs synchronous BFS over KnowledgeStore. The `graph` strategy in RAGRetriever replaces shallow lookups with `GraphTraverser.expand()`. Optional ProbLog inference runs `causal_propagation` and `indirect_link` queries on the discovered subgraph when `use_reasoning=True`.

**Tech Stack:** pyoxigraph SPARQL, ProbLog, ReasoningEngine (Noisy-OR)

**Spec:** `docs/superpowers/specs/2026-03-23-multihop-retrieval-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/knowledge_service/stores/graph_traversal.py` | GraphTraverser + TraversalResult |
| Modify | `src/knowledge_service/stores/knowledge.py:366-421` | Add `limit` param to `get_triples_by_object` |
| Modify | `src/knowledge_service/stores/rag.py:100-133` | Replace graph strategy with GraphTraverser |
| Modify | `src/knowledge_service/api/ask.py:13-47` | `use_reasoning`, `traversal_depth`, `inferred_triples` |
| Create | `tests/test_graph_traversal.py` | Traversal tests |
| Modify | `tests/test_rag_retriever.py` | Updated graph strategy tests |
| Modify | `tests/test_api_ask.py` | New response field tests |

---

## Task 1: Add `limit` parameter to `get_triples_by_object`

**Files:**
- Modify: `src/knowledge_service/stores/knowledge.py:366-421`
- Modify: `tests/test_knowledge_store.py`

- [ ] **Step 1: Update `get_triples_by_object` signature and SPARQL**

Add `limit: int | None = 20` parameter. When `None`, omit the `LIMIT` clause:

```python
def get_triples_by_object(
    self,
    object_uri: str,
    graphs: list[str] | None = None,
    limit: int | None = 20,
) -> list[dict]:
```

Change the SPARQL from hardcoded `LIMIT 20` to:

```python
limit_clause = f"LIMIT {limit}" if limit is not None else ""

sparql = f"""
    ...
    ORDER BY DESC(?conf)
    {limit_clause}
"""
```

- [ ] **Step 2: Run existing tests**

Run: `uv run pytest tests/test_knowledge_store.py -v`
Expected: ALL PASS (default limit=20 preserves behavior)

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_service/stores/knowledge.py
git commit -m "feat: add configurable limit to get_triples_by_object"
```

---

## Task 2: GraphTraverser — core BFS with confidence propagation

**Files:**
- Create: `src/knowledge_service/stores/graph_traversal.py`
- Create: `tests/test_graph_traversal.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_graph_traversal.py`:

```python
import pytest
from unittest.mock import MagicMock
from pyoxigraph import NamedNode, Literal
from knowledge_service.stores.graph_traversal import GraphTraverser, TraversalResult
from knowledge_service.ontology.namespaces import KS_GRAPH_EXTRACTED


def _make_knowledge_store(triples_by_subject=None, triples_by_object=None):
    """Create a mock KnowledgeStore for traversal tests."""
    ks = MagicMock()
    ks.get_triples_by_subject.side_effect = lambda uri, **kw: triples_by_subject.get(uri, [])
    ks.get_triples_by_object.side_effect = lambda uri, **kw: triples_by_object.get(uri, [])
    return ks


def _triple(subject_uri, predicate_uri, object_uri, confidence=0.8):
    """Create a triple dict matching KnowledgeStore output format."""
    return {
        "graph": KS_GRAPH_EXTRACTED,
        "subject": NamedNode(subject_uri),
        "predicate": NamedNode(predicate_uri),
        "object": NamedNode(object_uri),
        "confidence": confidence,
        "knowledge_type": "Claim",
        "valid_from": None,
        "valid_until": None,
    }


def _literal_triple(subject_uri, predicate_uri, literal_value, confidence=0.8):
    """Triple with a literal object (should not be expanded in BFS)."""
    return {
        "graph": KS_GRAPH_EXTRACTED,
        "subject": NamedNode(subject_uri),
        "predicate": NamedNode(predicate_uri),
        "object": Literal(literal_value),
        "confidence": confidence,
        "knowledge_type": "Claim",
        "valid_from": None,
        "valid_until": None,
    }


class TestBasicExpansion:
    def test_single_hop_discovers_neighbor(self):
        ks = _make_knowledge_store(
            triples_by_subject={"http://e/a": [_triple("http://e/a", "http://p/causes", "http://e/b")]},
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=1)
        assert any(n["uri"] == "http://e/b" for n in result.nodes)

    def test_two_hop_chain(self):
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [_triple("http://e/a", "http://p/1", "http://e/b", 0.8)],
                "http://e/b": [_triple("http://e/b", "http://p/2", "http://e/c", 0.7)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2)
        uris = {n["uri"] for n in result.nodes}
        assert "http://e/b" in uris
        assert "http://e/c" in uris

    def test_empty_graph_returns_empty(self):
        ks = _make_knowledge_store(triples_by_subject={}, triples_by_object={})
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=3)
        assert result.nodes == []
        assert result.edges == []


class TestConfidencePropagation:
    def test_multiplicative_path_confidence(self):
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [_triple("http://e/a", "http://p/1", "http://e/b", 0.8)],
                "http://e/b": [_triple("http://e/b", "http://p/2", "http://e/c", 0.7)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2)
        node_c = next(n for n in result.nodes if n["uri"] == "http://e/c")
        assert node_c["confidence"] == pytest.approx(0.56, rel=0.01)

    def test_noisy_or_across_paths(self):
        """Node reachable via 2 independent paths has higher confidence."""
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [
                    _triple("http://e/a", "http://p/1", "http://e/b", 0.6),
                    _triple("http://e/a", "http://p/2", "http://e/c", 0.5),
                ],
                "http://e/b": [_triple("http://e/b", "http://p/3", "http://e/target", 0.7)],
                "http://e/c": [_triple("http://e/c", "http://p/4", "http://e/target", 0.8)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2)
        target = next(n for n in result.nodes if n["uri"] == "http://e/target")
        # Path 1: 0.6 * 0.7 = 0.42, Path 2: 0.5 * 0.8 = 0.40
        # Noisy-OR: 1 - (1-0.42)(1-0.40) = 1 - 0.58*0.60 = 0.652
        assert target["confidence"] > 0.42  # higher than either path alone
        assert target["path_count"] == 2


class TestPruningAndLimits:
    def test_low_confidence_path_pruned(self):
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [_triple("http://e/a", "http://p/1", "http://e/b", 0.2)],
                "http://e/b": [_triple("http://e/b", "http://p/2", "http://e/c", 0.3)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2, min_confidence=0.1)
        # 0.2 * 0.3 = 0.06 < 0.1, so c should not appear
        uris = {n["uri"] for n in result.nodes}
        assert "http://e/c" not in uris

    def test_cycle_does_not_loop(self):
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [_triple("http://e/a", "http://p/1", "http://e/b", 0.8)],
                "http://e/b": [_triple("http://e/b", "http://p/2", "http://e/a", 0.8)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=4)
        # Should complete without infinite loop
        assert len(result.nodes) <= 2

    def test_max_nodes_cap(self):
        # Create a fan-out where node a connects to 60 neighbors
        neighbors = {
            "http://e/a": [
                _triple("http://e/a", "http://p/1", f"http://e/n{i}", 0.9) for i in range(60)
            ]
        }
        ks = _make_knowledge_store(triples_by_subject=neighbors, triples_by_object={})
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2, max_nodes=10)
        assert len(result.nodes) <= 10

    def test_literal_objects_not_expanded(self):
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [_literal_triple("http://e/a", "http://p/has_value", "250%", 0.9)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2)
        # Literal is recorded as edge but not expanded
        assert len(result.edges) == 1
        assert len(result.nodes) == 0  # no URI neighbors found


class TestHopDistance:
    def test_hop_distance_is_minimum(self):
        """Node reachable at hop 1 and hop 2 should have hop_distance=1."""
        ks = _make_knowledge_store(
            triples_by_subject={
                "http://e/a": [
                    _triple("http://e/a", "http://p/1", "http://e/target", 0.5),
                    _triple("http://e/a", "http://p/2", "http://e/b", 0.8),
                ],
                "http://e/b": [_triple("http://e/b", "http://p/3", "http://e/target", 0.9)],
            },
            triples_by_object={},
        )
        traverser = GraphTraverser(ks)
        result = traverser.expand(["http://e/a"], max_hops=2)
        target = next(n for n in result.nodes if n["uri"] == "http://e/target")
        assert target["hop_distance"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph_traversal.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement GraphTraverser**

Create `src/knowledge_service/stores/graph_traversal.py`:

```python
"""GraphTraverser — BFS graph traversal with Bayesian confidence propagation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from knowledge_service._utils import _rdf_value_to_str


@dataclass
class TraversalResult:
    """Result of multi-hop graph traversal."""

    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    paths: list[list[dict]] = field(default_factory=list)


class GraphTraverser:
    """BFS graph traversal with multiplicative path confidence and Noisy-OR combination."""

    _MAX_HOPS_CAP = 4

    def __init__(self, knowledge_store) -> None:
        self._ks = knowledge_store

    def expand(
        self,
        seed_uris: list[str],
        max_hops: int = 4,
        min_confidence: float = 0.1,
        max_nodes: int = 50,
    ) -> TraversalResult:
        """BFS expansion from seed entities with confidence propagation.

        Each path's confidence is the product of edge confidences along it.
        When multiple paths reach the same node, Noisy-OR combines them.
        """
        max_hops = min(max_hops, self._MAX_HOPS_CAP)

        # node_uri -> list of path confidences reaching it
        node_paths: dict[str, list[float]] = {}
        # node_uri -> minimum hop distance
        node_hops: dict[str, int] = {}
        # All discovered edges (stringified)
        all_edges: list[dict] = []
        # All paths (list of edge lists)
        all_path_lists: list[list[dict]] = []

        # Frontier: list of (uri, path_confidence, hop, path_edges)
        frontier: list[tuple[str, float, int, list[dict]]] = [
            (uri, 1.0, 0, []) for uri in seed_uris
        ]
        expanded: set[str] = set(seed_uris)

        for hop in range(1, max_hops + 1):
            next_frontier: list[tuple[str, float, int, list[dict]]] = []

            for node_uri, parent_conf, _, parent_path in frontier:
                if len(node_paths) >= max_nodes:
                    break

                # Get outgoing and incoming triples
                outgoing = self._ks.get_triples_by_subject(node_uri)
                incoming = self._ks.get_triples_by_object(node_uri, limit=None)

                for triple in outgoing:
                    neighbor_uri, edge = self._process_outgoing(triple, node_uri)
                    if edge is not None and neighbor_uri is None:
                        all_edges.append(edge)  # Record literal edges without expanding
                        continue
                    if neighbor_uri is None:
                        continue
                    self._maybe_add_neighbor(
                        neighbor_uri, edge, parent_conf, hop, parent_path,
                        node_paths, node_hops, all_edges, all_path_lists,
                        next_frontier, expanded, min_confidence, max_nodes,
                    )

                for triple in incoming:
                    neighbor_uri, edge = self._process_incoming(triple, node_uri)
                    if neighbor_uri is None:
                        continue
                    self._maybe_add_neighbor(
                        neighbor_uri, edge, parent_conf, hop, parent_path,
                        node_paths, node_hops, all_edges, all_path_lists,
                        next_frontier, expanded, min_confidence, max_nodes,
                    )

            if not next_frontier:
                break
            frontier = next_frontier

        # Build ranked nodes with Noisy-OR confidence
        nodes = []
        for uri, path_confs in node_paths.items():
            if uri in seed_uris:
                continue  # Don't include seeds in discovered nodes
            combined = self._noisy_or(path_confs)
            nodes.append({
                "uri": uri,
                "confidence": combined,
                "hop_distance": node_hops[uri],
                "path_count": len(path_confs),
            })

        nodes.sort(key=lambda n: n["confidence"], reverse=True)

        return TraversalResult(nodes=nodes, edges=all_edges, paths=all_path_lists)

    def _process_outgoing(self, triple: dict, source_uri: str):
        """Extract neighbor URI and edge dict from an outgoing triple."""
        obj = triple.get("object")
        if obj is None:
            return None, None
        obj_str = _rdf_value_to_str(obj)
        # Skip literals (not expandable as entities)
        if not obj_str.startswith(("http://", "https://", "urn:")):
            edge = self._make_edge(source_uri, triple)
            return None, edge  # Record edge but don't expand
        edge = self._make_edge(source_uri, triple)
        return obj_str, edge

    def _process_incoming(self, triple: dict, target_uri: str):
        """Extract neighbor URI and edge dict from an incoming triple."""
        subj = triple.get("subject")
        if subj is None:
            return None, None
        subj_str = _rdf_value_to_str(subj)
        if not subj_str.startswith(("http://", "https://", "urn:")):
            return None, None
        edge = {
            "subject": subj_str,
            "predicate": _rdf_value_to_str(triple.get("predicate")),
            "object": target_uri,
            "confidence": triple.get("confidence"),
            "knowledge_type": triple.get("knowledge_type", "Relationship"),
            "trust_tier": "verified"
            if "asserted" in triple.get("graph", "")
            else "extracted",
        }
        return subj_str, edge

    def _make_edge(self, source_uri: str, triple: dict) -> dict:
        return {
            "subject": source_uri,
            "predicate": _rdf_value_to_str(triple.get("predicate")),
            "object": _rdf_value_to_str(triple.get("object")),
            "confidence": triple.get("confidence"),
            "knowledge_type": triple.get("knowledge_type", "Relationship"),
            "trust_tier": "verified"
            if "asserted" in triple.get("graph", "")
            else "extracted",
        }

    def _maybe_add_neighbor(
        self, neighbor_uri, edge, parent_conf, hop, parent_path,
        node_paths, node_hops, all_edges, all_path_lists,
        next_frontier, expanded, min_confidence, max_nodes,
    ):
        edge_conf = edge.get("confidence") or 0.0
        path_conf = parent_conf * edge_conf
        if path_conf < min_confidence:
            return

        all_edges.append(edge)
        new_path = parent_path + [edge]
        all_path_lists.append(new_path)

        if neighbor_uri not in node_paths:
            node_paths[neighbor_uri] = []
        node_paths[neighbor_uri].append(path_conf)

        if neighbor_uri not in node_hops or hop < node_hops[neighbor_uri]:
            node_hops[neighbor_uri] = hop

        if neighbor_uri not in expanded and len(node_paths) < max_nodes:
            expanded.add(neighbor_uri)
            next_frontier.append((neighbor_uri, path_conf, hop, new_path))

    @staticmethod
    def _noisy_or(confidences: list[float]) -> float:
        if not confidences:
            return 0.0
        failure_product = math.prod(1.0 - c for c in confidences)
        return 1.0 - failure_product
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph_traversal.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/stores/graph_traversal.py tests/test_graph_traversal.py
git commit -m "feat: GraphTraverser — BFS with Bayesian confidence propagation"
```

---

## Task 3: Replace graph strategy with GraphTraverser

**Files:**
- Modify: `src/knowledge_service/stores/rag.py:100-133`
- Modify: `tests/test_rag_retriever.py`

- [ ] **Step 1: Update `_retrieve_graph` to use GraphTraverser**

In `rag.py`, add to `__init__`:

```python
from knowledge_service.stores.graph_traversal import GraphTraverser

self._graph_traverser = GraphTraverser(knowledge_store)
```

**Also update `retrieve()` signature** to accept and pass through `use_reasoning` and `reasoning_engine`:

```python
async def retrieve(
    self,
    question: str,
    max_sources: int = 5,
    min_confidence: float = 0.0,
    intent: QueryIntent | None = None,
    use_reasoning: bool = False,
    reasoning_engine=None,
) -> RetrievalContext:
```

In the dispatch, pass to `_retrieve_graph`:
```python
elif intent.intent == "graph":
    return await self._retrieve_graph(
        question, embedding, intent.entities, max_sources, min_confidence,
        use_reasoning=use_reasoning, reasoning_engine=reasoning_engine,
    )
```

Replace `_retrieve_graph` body (lines 102-133):

```python
async def _retrieve_graph(
    self, question, embedding, entity_names, max_sources, min_confidence,
    use_reasoning=False, reasoning_engine=None,
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

    ctx = RetrievalContext(
        content_results=content_results,
        knowledge_triples=filtered,
        contradictions=contradictions,
        entities_found=entities_found,
    )
    ctx.traversal_depth = traversal_depth
    ctx.inferred_triples = inferred_count
    return ctx
```

**Add traversal metadata fields to RetrievalContext** (in `rag.py`):

```python
@dataclass
class RetrievalContext:
    content_results: list[dict] = field(default_factory=list)
    knowledge_triples: list[dict] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    entities_found: list[str] = field(default_factory=list)
    traversal_depth: int | None = None   # NEW: max hop distance in graph traversal
    inferred_triples: int | None = None  # NEW: count of ProbLog-derived triples
```

**Wire in `post_ask()`** (in `ask.py`): after building the response, add:

```python
traversal_depth=getattr(context, "traversal_depth", None),
inferred_triples=getattr(context, "inferred_triples", None),
```

This uses `getattr` so existing callers that don't set these fields get `None` safely.

- [ ] **Step 2: Update mock in test_rag_retriever.py**

In `_make_knowledge_store`, the mock already has `get_triples_by_object` and `find_connecting_triples`. No change needed — `GraphTraverser` calls the real methods on the mock.

However, update the `test_graph_intent_uses_bidirectional_lookup` test since the graph strategy no longer calls `get_triples_by_object` directly (GraphTraverser does):

```python
async def test_graph_intent_uses_traverser(self):
    """Graph intent should use multi-hop traversal."""
    ec = _make_embedding_client()
    ec.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]
    es = _make_embedding_store()
    es.search_entities.return_value = [
        {"uri": "http://knowledge.local/data/cortisol", "similarity": 0.9}
    ]
    ks = _make_knowledge_store()
    retriever = RAGRetriever(ec, es, ks)
    intent = QueryIntent(intent="graph", entities=["cortisol", "inflammation"])
    context = await retriever.retrieve("how is cortisol connected to inflammation?", intent=intent)
    # GraphTraverser calls get_triples_by_subject internally
    ks.get_triples_by_subject.assert_called()
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_service/stores/rag.py tests/test_rag_retriever.py
git commit -m "feat: replace shallow graph lookups with multi-hop GraphTraverser"
```

---

## Task 4: Optional ProbLog inference + API changes

**Files:**
- Modify: `src/knowledge_service/api/ask.py:13-47,49-end`
- Modify: `src/knowledge_service/stores/rag.py`
- Modify: `tests/test_api_ask.py`

- [ ] **Step 1: Add `use_reasoning` to AskRequest, metadata to AskResponse**

In `ask.py`:

```python
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=_MAX_QUESTION_LEN)
    max_sources: int = Field(5, ge=1, le=100)
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)
    use_reasoning: bool = Field(False)  # NEW


class AskResponse(BaseModel):
    answer: str
    confidence: float | None
    sources: list[SourceInfo]
    knowledge_types_used: list[str]
    contradictions: list[ContradictionInfo]
    evidence: list[EvidenceSnippet] = []
    intent: str | None = None
    traversal_depth: int | None = None   # NEW
    inferred_triples: int | None = None  # NEW
```

- [ ] **Step 2: Pass `use_reasoning` through to retriever**

In `post_ask()`, pass to retrieve and capture traversal metadata:

```python
context = await retriever.retrieve(
    body.question,
    max_sources=body.max_sources,
    min_confidence=body.min_confidence,
    intent=intent,
    use_reasoning=body.use_reasoning,
    reasoning_engine=reasoning_engine,
)
```

Add `traversal_depth` and `inferred_triples` to the response from `context` metadata (stored as attributes on RetrievalContext or returned separately).

- [ ] **Step 3: Add ProbLog integration to RAGRetriever**

In `rag.py`, update `_retrieve_graph` to accept `use_reasoning` and `reasoning_engine`:

After BFS traversal, if `use_reasoning` is True:

```python
if use_reasoning and reasoning_engine:
    inferred = await self._run_problog_inference(
        traversal.edges[:50], resolved_uris, reasoning_engine
    )
    filtered.extend(inferred)
```

New helper:

```python
async def _run_problog_inference(self, edges, seed_uris, reasoning_engine):
    """Run ProbLog causal/indirect inference on discovered subgraph."""
    from knowledge_service.reasoning.engine import _to_atom

    claims = []
    for e in edges:
        claims.append((
            e["subject"], e["predicate"], e["object"],
            e.get("confidence") or 0.5,
            {"knowledge_type": e.get("knowledge_type", "Claim")},
        ))

    inferred = []
    for seed in seed_uris:
        seed_atom = _to_atom(seed)
        for query_template in [
            f"causal_propagation({seed_atom}, C)",
            f"indirect_link({seed_atom}, P, C)",
        ]:
            results = await asyncio.to_thread(
                reasoning_engine.infer, query_template, claims
            )
            for r in results:
                if r.probability > 0.05:
                    inferred.append({
                        "subject": seed,
                        "predicate": r.query,
                        "object": str(r.probability),
                        "confidence": r.probability,
                        "knowledge_type": "Conclusion",
                        "trust_tier": "inferred",
                    })
    return inferred
```

- [ ] **Step 4: Add tests**

In `tests/test_api_ask.py`:

```python
class TestAskTraversalMetadata:
    async def test_response_includes_traversal_fields(self, client):
        response = await client.post("/api/ask", json={"question": "test"})
        data = response.json()
        assert "traversal_depth" in data
        assert "inferred_triples" in data

    async def test_use_reasoning_parameter_accepted(self, client):
        response = await client.post(
            "/api/ask", json={"question": "test", "use_reasoning": False}
        )
        assert response.status_code == 200
```

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_service/api/ask.py src/knowledge_service/stores/rag.py tests/test_api_ask.py
git commit -m "feat: optional ProbLog inference on traversed subgraph, traversal metadata in API"
```

---

## Task 5: Final integration test and lint

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`
If needed: `uv run ruff format .`

- [ ] **Step 3: Commit if needed**

```bash
git add -A && git commit -m "chore: lint fixes for multi-hop retrieval"
```
