# Foundation Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 12 ProbLog reasoning rules, 4-tier named graph trust boundaries, and chunk-level provenance tracking to the knowledge-service.

**Architecture:** Three independent changes: (1) new ProbLog rule files + engine integration for structural reasoning, (2) pyoxigraph named graphs with SPARQL rewrites for trust-tiered storage, (3) provenance table + extraction pipeline changes for chunk-level evidence tracing.

**Tech Stack:** ProbLog, pyoxigraph (SPARQL 1.2, named graphs, RDF-star), PostgreSQL/asyncpg, FastAPI

**Spec:** `docs/superpowers/specs/2026-03-21-foundation-improvements-design.md`

---

## File Map

### Phase 1: ProbLog Rules

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/knowledge_service/reasoning/rules/base.pl` | Add `inverse_holds/3`, `corroborated/3` |
| Create | `src/knowledge_service/reasoning/rules/inference_chains.pl` | `indirect_link/3`, `causal_propagation/2` |
| Create | `src/knowledge_service/reasoning/rules/confidence.pl` | `high_confidence/3`, `contested/3`, `authoritative/3` |
| Modify | `src/knowledge_service/reasoning/rules/temporal.pl` | Replace placeholder with 3 real rules |
| Modify | `src/knowledge_service/reasoning/engine.py` | Glob-load `.pl` files, 5-tuple claims, metadata fact emission |
| Modify | `src/knowledge_service/ontology/schema.ttl` | Add `ks:inversePredicate`, fix `contains/part_of` |
| Modify | `src/knowledge_service/ontology/namespaces.py` | Add `KS_INVERSE_PREDICATE` constant |
| Modify | `tests/test_reasoning_engine.py` | Tests for all new rules and engine changes |

### Phase 2: Named Graphs

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/knowledge_service/ontology/namespaces.py` | Add `KS_GRAPH_*` constants |
| Modify | `src/knowledge_service/stores/knowledge.py` | Named graph support in all methods, SPARQL rewrites |
| Modify | `src/knowledge_service/ontology/bootstrap.py` | Load into `ks:graph/ontology` with idempotency guard |
| Modify | `src/knowledge_service/api/_ingest.py` | Map extractor to graph URI |
| Modify | `src/knowledge_service/stores/rag.py` | Trust-tier labeling in retrieval context |
| Modify | `src/knowledge_service/clients/rag.py` | Trust tier labels in RAG prompt |
| Create | `src/knowledge_service/stores/graph_migration.py` | One-time migration: default graph → named graphs |
| Modify | `src/knowledge_service/main.py` | Call graph migration at startup |
| Modify | `tests/test_knowledge_store.py` | Named graph tests |
| Modify | `tests/test_rag_retriever.py` | Trust tier labeling tests |

### Phase 3: Chunk Provenance

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `migrations/004_add_chunk_provenance.sql` | `chunk_id` column + index |
| Modify | `src/knowledge_service/stores/provenance.py` | `chunk_id` in insert/read |
| Modify | `src/knowledge_service/stores/embedding.py` | `insert_chunks` returns IDs, `get_chunks_by_ids` |
| Modify | `src/knowledge_service/api/_ingest.py` | Pass `chunk_id` to `process_triple` |
| Modify | `src/knowledge_service/api/content.py` | Per-chunk extraction with chunk IDs |
| Modify | `src/knowledge_service/api/ask.py` | `EvidenceSnippet` in response |
| Modify | `tests/test_provenance_store.py` | chunk_id tests |
| Modify | `tests/test_api_ask.py` | Evidence snippet tests |

---

## Phase 1: ProbLog Rules

### Task 1: New rule files (inference_chains.pl, confidence.pl)

**Files:**
- Create: `src/knowledge_service/reasoning/rules/inference_chains.pl`
- Create: `src/knowledge_service/reasoning/rules/confidence.pl`

- [ ] **Step 1: Create inference_chains.pl**

```prolog
% Transitive link (bounded to 2-hop to prevent runaway)
indirect_link(A, P, C) :-
    claims(A, P, B, _),
    claims(B, P, C, _),
    A \= C.

% Cross-predicate causal chains: A causes B, B increases/decreases C
causal_propagation(A, C) :-
    claims(A, causes, B, _),
    claims(B, increases, C, _).

causal_propagation(A, C) :-
    claims(A, causes, B, _),
    claims(B, decreases, C, _).
```

- [ ] **Step 2: Create confidence.pl**

```prolog
% High-confidence: supported and no value conflicts
high_confidence(S, P, O) :-
    supported(S, P, O),
    \+ value_conflict(S, P, _, _).

% Contested: supported but has conflicting values
contested(S, P, O) :-
    supported(S, P, O),
    value_conflict(S, P, O, _).

% Fact overrides claim when both exist
authoritative(S, P, O) :-
    claims(S, P, O, _),
    claim_type(S, P, O, fact).
```

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_service/reasoning/rules/inference_chains.pl src/knowledge_service/reasoning/rules/confidence.pl
git commit -m "feat: add inference chain and confidence ProbLog rules"
```

### Task 2: Update base.pl and temporal.pl

**Files:**
- Modify: `src/knowledge_service/reasoning/rules/base.pl:17` (append after `supported`)
- Modify: `src/knowledge_service/reasoning/rules/temporal.pl` (replace entirely)

- [ ] **Step 1: Append inverse_holds and corroborated to base.pl**

Add after line 17 (the `supported` rule):

```prolog

% Inverse predicate inference: if A contains B, then B part_of A
inverse_holds(S, P2, O) :-
    claims(O, P1, S, _),
    inverse(P1, P2).

% Multi-source corroboration: claim from 2+ independent sources
corroborated(S, P, O) :-
    claims(S, P, O, Src1),
    claims(S, P, O, Src2),
    Src1 \= Src2.
```

- [ ] **Step 2: Replace temporal.pl**

Replace entire contents of `temporal.pl`:

```prolog
% Expired: valid_until has passed
expired(S, P, O) :-
    claims(S, P, O, _),
    valid_until(S, P, O, Until),
    current_date(Now),
    Now > Until.

% Currently valid: has temporal bounds and not expired
currently_valid(S, P, O) :-
    claims(S, P, O, _),
    valid_from(S, P, O, From),
    current_date(Now),
    Now >= From,
    \+ expired(S, P, O).

% Temporal supersedes: newer temporal state replaces older for same S-P
supersedes(S, P, O_new, O_old) :-
    claims(S, P, O_new, _),
    claims(S, P, O_old, _),
    valid_from(S, P, O_new, F1),
    valid_from(S, P, O_old, F2),
    F1 > F2,
    O_new \= O_old.
```

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_service/reasoning/rules/base.pl src/knowledge_service/reasoning/rules/temporal.pl
git commit -m "feat: expand base.pl with inverse/corroboration, replace temporal.pl with real rules"
```

### Task 3: Update schema.ttl with ks:inversePredicate

**Files:**
- Modify: `src/knowledge_service/ontology/schema.ttl:27` (add property)
- Modify: `src/knowledge_service/ontology/schema.ttl:53` (fix contains/part_of)
- Modify: `src/knowledge_service/ontology/namespaces.py:64` (add constant)

- [ ] **Step 1: Add ks:inversePredicate property to schema.ttl**

After line 27 (`ks:oppositePredicate`), add:

```turtle
ks:inversePredicate    rdf:type rdf:Property ; rdfs:label "inverse predicate" .
```

- [ ] **Step 2: Fix contains/part_of — replace oppositePredicate with inversePredicate**

Change line 53 from:
```turtle
ks:contains        ks:oppositePredicate ks:part_of .
```
To:
```turtle
ks:contains        ks:inversePredicate ks:part_of .
```

- [ ] **Step 3: Add KS_INVERSE_PREDICATE constant to namespaces.py**

After line 64 (`KS_OPPOSITE_PREDICATE`), add:

```python
KS_INVERSE_PREDICATE = ks("inversePredicate")
```

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_service/ontology/schema.ttl src/knowledge_service/ontology/namespaces.py
git commit -m "feat: add ks:inversePredicate, fix contains/part_of from opposite to inverse"
```

### Task 4: ReasoningEngine glob-loading and 5-tuple support

**Files:**
- Modify: `src/knowledge_service/reasoning/engine.py:41-46` (glob-load)
- Modify: `src/knowledge_service/reasoning/engine.py:175-222` (infer 5-tuple)
- Modify: `src/knowledge_service/reasoning/engine.py:69-173` (check_contradiction 5-tuple)

- [ ] **Step 1: Write failing test for glob-loading**

Add to `tests/test_reasoning_engine.py`:

```python
class TestGlobLoading:
    def test_loads_all_pl_files_including_new(self, engine):
        """Engine should load all .pl files via glob, including new ones."""
        assert "indirect_link" in engine._all_rules
        assert "causal_propagation" in engine._all_rules
        assert "high_confidence" in engine._all_rules
        assert "authoritative" in engine._all_rules
        assert "currently_valid" in engine._all_rules
        assert "inverse_holds" in engine._all_rules
        assert "corroborated" in engine._all_rules
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reasoning_engine.py::TestGlobLoading -v`
Expected: FAIL — `indirect_link` not in `_all_rules` (only 3 files loaded)

- [ ] **Step 3: Implement glob-loading in engine.py**

Replace `__init__` (lines 41-46) with:

```python
def __init__(self, rules_dir: str | Path) -> None:
    self._rules_dir = Path(rules_dir)
    parts = []
    for pl_file in sorted(self._rules_dir.glob("*.pl")):
        parts.append(pl_file.read_text(encoding="utf-8"))
    self._all_rules: str = "\n".join(parts)
    self._base_rules: str = self._load_rules("base.pl")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_reasoning_engine.py::TestGlobLoading -v`
Expected: PASS

- [ ] **Step 5: Write failing test for 5-tuple infer with metadata**

Add to `tests/test_reasoning_engine.py`:

```python
class TestInferWithMetadata:
    def test_infer_accepts_5_tuple_claims(self, engine):
        """infer() should accept 5-tuples with metadata dict."""
        claims = [
            ("cold_exposure", "increases", "dopamine", 0.7, {"knowledge_type": "fact"}),
        ]
        results = engine.infer(
            query="authoritative(cold_exposure, increases, dopamine)",
            claims=claims,
        )
        assert len(results) >= 1
        assert results[0].probability > 0

    def test_infer_backward_compat_4_tuple(self, engine):
        """infer() should still work with 4-tuples (no metadata)."""
        claims = [("a", "b", "c", 0.5)]
        results = engine.infer(query="supported(a, b, c)", claims=claims)
        assert len(results) >= 1
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_reasoning_engine.py::TestInferWithMetadata -v`
Expected: FAIL — `ValueError: too many values to unpack`

- [ ] **Step 7: Implement 5-tuple support in infer()**

Replace `infer()` method (lines 175-222). Key changes:
- Accept `list[tuple]` (4 or 5 elements)
- Unpack with fallback: `meta = claim[4] if len(claim) > 4 else {}`
- After claims loop, emit metadata facts:

```python
def infer(
    self,
    query: str,
    claims: list[tuple],
) -> list[InferenceResult]:
    from datetime import date as date_type

    program_parts: list[str] = [self._all_rules, ""]

    # Inject current_date/1
    program_parts.append(f"current_date('{date_type.today().isoformat()}').")
    program_parts.append("")

    for idx, claim in enumerate(claims):
        subj, pred, obj, conf = claim[0], claim[1], claim[2], claim[3]
        meta = claim[4] if len(claim) > 4 else {}
        s_atom = _to_atom(subj)
        p_atom = _to_atom(pred)
        o_atom = _to_atom(obj)
        program_parts.append(f"{conf}::claims({s_atom}, {p_atom}, {o_atom}, source{idx}).")

        if meta.get("knowledge_type"):
            kt_atom = _to_atom(meta["knowledge_type"])
            program_parts.append(f"claim_type({s_atom}, {p_atom}, {o_atom}, {kt_atom}).")
        if meta.get("valid_from"):
            program_parts.append(
                f"valid_from({s_atom}, {p_atom}, {o_atom}, '{meta['valid_from']}')."
            )
        if meta.get("valid_until"):
            program_parts.append(
                f"valid_until({s_atom}, {p_atom}, {o_atom}, '{meta['valid_until']}')."
            )

    program_parts.append("")
    program_parts.append(f"query({query}).")

    program = "\n".join(program_parts)

    try:
        db = PrologString(program)
        raw_results = get_evaluatable().create_from(db).evaluate()
    except Exception as exc:
        logger.warning("ProbLog inference failed, using Python fallback: %s", exc)
        return self._fallback_infer(query, claims)

    results: list[InferenceResult] = []
    for term, prob in raw_results.items():
        results.append(InferenceResult(query=str(term), probability=float(prob)))

    results.sort(key=lambda r: r.probability, reverse=True)
    return results
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_reasoning_engine.py::TestInferWithMetadata -v`
Expected: PASS

- [ ] **Step 9: Apply same 5-tuple change to check_contradiction()**

Update `check_contradiction()` (lines 69-173):
- Change signature to accept `tuple` (4 or 5 elements)
- Unpack with: `subj, pred, obj, conf = claim[0:4]`
- After emitting claims, emit metadata facts same way as `infer()`
- Add `current_date/1` injection

- [ ] **Step 10: Update _fallback_infer for 5-tuple compat**

In `_fallback_infer` (line 236), change unpack from:
```python
conf for (subj, pred, obj, conf) in claims
```
To:
```python
claim[3] for claim in claims if claim[0] == qs and claim[1] == qp and claim[2] == qo
```

- [ ] **Step 11: Run full reasoning test suite**

Run: `uv run pytest tests/test_reasoning_engine.py -v`
Expected: ALL PASS (existing tests still work with 4-tuples)

- [ ] **Step 12: Commit**

```bash
git add src/knowledge_service/reasoning/engine.py tests/test_reasoning_engine.py
git commit -m "feat: glob-load ProbLog rules, support 5-tuple claims with metadata"
```

### Task 5: Test all new ProbLog rules

**Files:**
- Modify: `tests/test_reasoning_engine.py`

- [ ] **Step 1: Write tests for inference chains**

```python
class TestInferenceChains:
    def test_indirect_link_2hop(self, engine):
        """A->B, B->C via same predicate => indirect_link(A, P, C)."""
        claims = [
            ("a", "causes", "b", 0.8),
            ("b", "causes", "c", 0.7),
        ]
        results = engine.infer("indirect_link(a, causes, c)", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0

    def test_causal_propagation_increases(self, engine):
        """A causes B, B increases C => causal_propagation(A, C)."""
        claims = [
            ("stress", "causes", "cortisol", 0.8),
            ("cortisol", "increases", "inflammation", 0.7),
        ]
        results = engine.infer("causal_propagation(stress, inflammation)", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0

    def test_no_indirect_link_self(self, engine):
        """A->A should not produce indirect_link(A, P, A)."""
        claims = [("a", "causes", "a", 0.8)]
        results = engine.infer("indirect_link(a, causes, a)", claims=claims)
        assert len(results) == 0 or results[0].probability == pytest.approx(0.0)
```

- [ ] **Step 2: Write tests for confidence rules**

```python
class TestConfidenceRules:
    def test_high_confidence_no_conflict(self, engine):
        claims = [("a", "b", "c", 0.9)]
        results = engine.infer("high_confidence(a, b, c)", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0

    def test_contested_with_conflict(self, engine):
        claims = [
            ("a", "b", "c", 0.9),
            ("a", "b", "d", 0.7),
        ]
        results = engine.infer("contested(a, b, c)", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0

    def test_authoritative_fact(self, engine):
        claims = [("a", "b", "c", 0.95, {"knowledge_type": "fact"})]
        results = engine.infer("authoritative(a, b, c)", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0
```

- [ ] **Step 3: Write tests for temporal rules**

```python
class TestTemporalRules:
    def test_expired_claim(self, engine):
        claims = [("btc", "has_property", "50000", 0.9, {"valid_until": "2020-01-01"})]
        results = engine.infer("expired(btc, has_property, '50000')", claims=claims)
        assert len(results) >= 1
        assert results[0].probability > 0

    def test_currently_valid(self, engine):
        claims = [("btc", "has_property", "100000", 0.9, {"valid_from": "2020-01-01"})]
        results = engine.infer(
            "currently_valid(btc, has_property, '100000')", claims=claims
        )
        assert len(results) >= 1
        assert results[0].probability > 0
```

- [ ] **Step 4: Write test for inverse_holds**

```python
class TestInverseHolds:
    def test_inverse_holds_contains_part_of(self, engine):
        """If A contains B and inverse(contains, part_of) => inverse_holds(B, part_of, A)."""
        claims = [("body", "contains", "heart", 0.95)]
        # Manually inject inverse/2 fact since we don't load from schema.ttl in tests
        import types
        original_all_rules = engine._all_rules
        engine._all_rules = original_all_rules + "\ninverse(contains, part_of).\n"
        try:
            results = engine.infer("inverse_holds(heart, part_of, body)", claims=claims)
            assert len(results) >= 1
            assert results[0].probability > 0
        finally:
            engine._all_rules = original_all_rules
```

- [ ] **Step 5: Run all new tests**

Run: `uv run pytest tests/test_reasoning_engine.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_reasoning_engine.py
git commit -m "test: comprehensive tests for all new ProbLog rules"
```

---

## Phase 2: Named Graphs

### Task 6: Add graph constants to namespaces.py

**Files:**
- Modify: `src/knowledge_service/ontology/namespaces.py:64`

- [ ] **Step 1: Add named graph constants**

After `KS_OPPOSITE_PREDICATE` (line 64), add:

```python
KS_INVERSE_PREDICATE = ks("inversePredicate")

# Named graphs for trust-tier separation
KS_GRAPH_ONTOLOGY = f"{KS}graph/ontology"
KS_GRAPH_ASSERTED = f"{KS}graph/asserted"
KS_GRAPH_EXTRACTED = f"{KS}graph/extracted"
KS_GRAPH_INFERRED = f"{KS}graph/inferred"
```

(Note: `KS_INVERSE_PREDICATE` may already exist from Task 3. If so, just add the graph constants.)

- [ ] **Step 2: Commit**

```bash
git add src/knowledge_service/ontology/namespaces.py
git commit -m "feat: add named graph URI constants to namespaces"
```

### Task 7: Rewrite KnowledgeStore for named graphs

**Files:**
- Modify: `src/knowledge_service/stores/knowledge.py`
- Modify: `tests/test_knowledge_store.py`

This is the largest task. Every SPARQL query and `quads_for_pattern` call needs updating.

- [ ] **Step 1: Write failing test for insert_triple with graph parameter**

Add to `tests/test_knowledge_store.py`:

```python
from knowledge_service.ontology.namespaces import KS_GRAPH_ASSERTED, KS_GRAPH_EXTRACTED

class TestNamedGraphs:
    def test_insert_into_named_graph(self, store):
        store.insert_triple(
            "http://s/1", "http://p/1", "http://o/1", 0.8, "Claim",
            graph=KS_GRAPH_ASSERTED,
        )
        results = store.get_triples_by_subject("http://s/1")
        assert len(results) == 1
        assert results[0]["graph"] == KS_GRAPH_ASSERTED

    def test_insert_default_graph_is_extracted(self, store):
        store.insert_triple("http://s/2", "http://p/2", "http://o/2", 0.8, "Claim")
        results = store.get_triples_by_subject("http://s/2")
        assert len(results) == 1
        assert results[0]["graph"] == KS_GRAPH_EXTRACTED

    def test_get_triples_filters_by_graph(self, store):
        store.insert_triple(
            "http://s/3", "http://p/3", "http://o/3a", 0.8, "Claim",
            graph=KS_GRAPH_ASSERTED,
        )
        store.insert_triple(
            "http://s/3", "http://p/3b", "http://o/3b", 0.7, "Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        asserted = store.get_triples_by_subject("http://s/3", graphs=[KS_GRAPH_ASSERTED])
        assert len(asserted) == 1
        all_graphs = store.get_triples_by_subject("http://s/3")
        assert len(all_graphs) == 2

    def test_contradictions_span_graphs(self, store):
        store.insert_triple(
            "http://s/4", "http://p/4", "http://o/a", 0.8, "Claim",
            graph=KS_GRAPH_ASSERTED,
        )
        store.insert_triple(
            "http://s/4", "http://p/4", "http://o/b", 0.7, "Claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        contras = store.find_contradictions("http://s/4", "http://p/4", "http://o/a")
        assert len(contras) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_knowledge_store.py::TestNamedGraphs -v`
Expected: FAIL — `insert_triple() got an unexpected keyword argument 'graph'`

- [ ] **Step 3: Implement named graph support in KnowledgeStore**

Update imports at top of `knowledge.py` — add:
```python
from knowledge_service.ontology.namespaces import KS_GRAPH_EXTRACTED
```

**3a. `insert_triple()`** — add `graph` parameter, use `Quad` with named graph:

```python
def insert_triple(
    self,
    subject: str,
    predicate: str,
    object_: str,
    confidence: float,
    knowledge_type: str,
    valid_from: date | datetime | None = None,
    valid_until: date | datetime | None = None,
    graph: str | None = None,
) -> tuple[str, bool]:
```

Change `self._store.add(Quad(s, p, o))` to:
```python
graph_uri = graph or KS_GRAPH_EXTRACTED
graph_node = NamedNode(graph_uri)
self._store.add(Quad(s, p, o, graph_node))
```

Change `quads_for_pattern` idempotency check — keep `None` for graph to check all graphs:
```python
existing_reifications = list(self._store.quads_for_pattern(None, RDF_REIFIES, triple, None))
```

Change SPARQL `INSERT DATA` for annotations to use `GRAPH`:
```python
sparql = f"""
    INSERT DATA {{
        GRAPH <{graph_uri}> {{
            << <{subject}> <{predicate}> {obj_sparql} >>
                <{KS_CONFIDENCE.value}> {conf_literal} .
            << <{subject}> <{predicate}> {obj_sparql} >>
                <{KS_KNOWLEDGE_TYPE.value}> {type_uri} .
        }}
    }}
"""
```

Same for temporal annotation SPARQL — wrap in `GRAPH <{graph_uri}> { ... }`.

**3b. `get_triples_by_subject()`** — add `graphs` parameter, use `GRAPH ?g`:

```python
def get_triples_by_subject(self, subject: str, graphs: list[str] | None = None) -> list[dict]:
```

Rewrite SPARQL:
```python
graph_filter = ""
if graphs:
    values = " ".join(f"<{g}>" for g in graphs)
    graph_filter = f"VALUES ?g {{ {values} }}"

sparql = f"""
    SELECT ?g ?p ?o ?conf ?ktype ?vfrom ?vuntil WHERE {{
        {graph_filter}
        GRAPH ?g {{
            <{subject}> ?p ?o .
        }}
        OPTIONAL {{
            GRAPH ?g {{
                << <{subject}> ?p ?o >>
                    <{KS_CONFIDENCE.value}> ?conf .
            }}
        }}
        OPTIONAL {{
            GRAPH ?g {{
                << <{subject}> ?p ?o >>
                    <{KS_KNOWLEDGE_TYPE.value}> ?ktype .
            }}
        }}
        OPTIONAL {{
            GRAPH ?g {{
                << <{subject}> ?p ?o >>
                    <{KS_VALID_FROM.value}> ?vfrom .
            }}
        }}
        OPTIONAL {{
            GRAPH ?g {{
                << <{subject}> ?p ?o >>
                    <{KS_VALID_UNTIL.value}> ?vuntil .
            }}
        }}
        FILTER(BOUND(?conf))
    }}
"""
```

Add `"graph": solution["g"].value` to each result dict.

**3c. `get_triples_by_predicate()`** — same pattern as 3b but with `?s <predicate> ?o` inside `GRAPH ?g`.

**3d. `find_contradictions()`** — wrap triple pattern in `GRAPH ?g`:

```python
sparql = f"""
    SELECT ?o ?conf WHERE {{
        GRAPH ?g {{
            <{subject}> <{predicate}> ?o .
        }}
        FILTER(?o != {obj_sparql})
        OPTIONAL {{
            GRAPH ?g {{
                << <{subject}> <{predicate}> ?o >>
                    <{KS_CONFIDENCE.value}> ?conf .
            }}
        }}
    }}
"""
```

**3e. `find_opposite_predicate_contradictions()`** — the UNION for `oppositePredicate` lookups must reach `ks:graph/ontology`. Rewrite:

```python
sparql = f"""
    SELECT DISTINCT ?p_stored ?conf WHERE {{
        GRAPH ?gont {{
            {{
                <{predicate}> <{KS_OPPOSITE_PREDICATE.value}> ?p_stored .
            }} UNION {{
                ?p_stored <{KS_OPPOSITE_PREDICATE.value}> <{predicate}> .
            }}
        }}
        GRAPH ?g {{
            <{subject}> ?p_stored {obj_sparql} .
        }}
        OPTIONAL {{
            GRAPH ?g {{
                << <{subject}> ?p_stored {obj_sparql} >>
                    <{KS_CONFIDENCE.value}> ?conf .
            }}
        }}
    }}
"""
```

- [ ] **Step 4: Run named graph tests**

Run: `uv run pytest tests/test_knowledge_store.py::TestNamedGraphs -v`
Expected: PASS

- [ ] **Step 5: Update existing tests that use bare SPARQL or default graph**

Existing tests will break because:
- `TestQuery.test_sparql_query` uses bare SPARQL without `GRAPH` clause — update to use `GRAPH ?g { ... }` pattern
- `TestFindOppositePredContradictions` manually adds `oppositePredicate` quad to default graph — update to add to `NamedNode(KS_GRAPH_ONTOLOGY)` instead
- Any test using `insert_triple` without `graph=` now inserts into `KS_GRAPH_EXTRACTED` — their SPARQL queries must use `GRAPH ?g`

Update all affected tests to use the named graph patterns.

- [ ] **Step 6: Run full knowledge store test suite**

Run: `uv run pytest tests/test_knowledge_store.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_service/stores/knowledge.py tests/test_knowledge_store.py
git commit -m "feat: named graph support in KnowledgeStore with SPARQL rewrites"
```

### Task 8: Update bootstrap.py for named graph loading

**Files:**
- Modify: `src/knowledge_service/ontology/bootstrap.py`

- [ ] **Step 1: Implement named graph loading with idempotency**

Replace `bootstrap_ontology()`:

```python
from pyoxigraph import NamedNode, Store
from knowledge_service.ontology.namespaces import KS_GRAPH_ONTOLOGY


def bootstrap_ontology(store: Store) -> int:
    schema_path = Path(__file__).parent / "schema.ttl"
    graph_node = NamedNode(KS_GRAPH_ONTOLOGY)

    # Idempotency: skip if ontology graph already has triples
    existing = list(store.quads_for_pattern(None, None, None, graph_node))
    if existing:
        return 0

    initial_count = len(list(store.quads_for_pattern(None, None, None, None)))
    with open(schema_path, "rb") as f:
        store.load(f, "text/turtle", to_graph=graph_node)
    final_count = len(list(store.quads_for_pattern(None, None, None, None)))
    return final_count - initial_count
```

- [ ] **Step 2: Run existing tests**

Run: `uv run pytest tests/ -v -k "bootstrap or ontology"`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_service/ontology/bootstrap.py
git commit -m "feat: bootstrap ontology into ks:graph/ontology named graph"
```

### Task 9: Update _ingest.py to map extractor to graph

**Files:**
- Modify: `src/knowledge_service/api/_ingest.py:11-19`

- [ ] **Step 1: Add graph mapping logic**

Add import and helper at top of `_ingest.py`:

```python
from knowledge_service.ontology.namespaces import KS_GRAPH_ASSERTED, KS_GRAPH_EXTRACTED


def _extractor_to_graph(extractor: str) -> str:
    if extractor.startswith("llm_"):
        return KS_GRAPH_EXTRACTED
    return KS_GRAPH_ASSERTED
```

- [ ] **Step 2: Pass graph to insert_triple**

In `process_triple()`, change the `knowledge_store.insert_triple` call to include `graph`:

```python
graph = _extractor_to_graph(extractor)

triple_hash, is_new = await asyncio.to_thread(
    knowledge_store.insert_triple,
    t["subject"],
    t["predicate"],
    t["object"],
    t["confidence"],
    t["knowledge_type"],
    t["valid_from"],
    t["valid_until"],
    graph,
)
```

- [ ] **Step 3: Run ingest tests**

Run: `uv run pytest tests/test_ingest.py tests/test_api_content.py tests/test_api_claims.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_service/api/_ingest.py
git commit -m "feat: map extractor to named graph in process_triple"
```

### Task 10: Update RAG retriever and prompt for trust tiers

**Files:**
- Modify: `src/knowledge_service/stores/rag.py:62-67`
- Modify: `src/knowledge_service/clients/rag.py:46-55`

- [ ] **Step 1: Add graph/trust_tier to retrieval context triples**

In `rag.py`, after stringifying predicate/object (line 66), add trust tier:

```python
from knowledge_service.ontology.namespaces import KS_GRAPH_ASSERTED

# ... inside the for uri loop, after existing t["object"] = ...
graph = t.get("graph", "")
t["trust_tier"] = "verified" if graph == KS_GRAPH_ASSERTED else "extracted"
```

- [ ] **Step 2: Update RAG prompt to show trust tier**

In `clients/rag.py`, change the knowledge triples section (line 54):

```python
sections.append(f"- [{t.get('trust_tier', 'unknown')}] {s} -> {p} -> {o} ({ktype}, confidence: {conf})")
```

- [ ] **Step 3: Run RAG tests**

Run: `uv run pytest tests/test_rag_retriever.py tests/test_rag_client.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_service/stores/rag.py src/knowledge_service/clients/rag.py
git commit -m "feat: trust tier labels in RAG retrieval context and prompt"
```

### Task 11: Graph migration script

**Files:**
- Create: `src/knowledge_service/stores/graph_migration.py`
- Modify: `src/knowledge_service/main.py` (call migration at startup)

- [ ] **Step 1: Create graph_migration.py**

```python
"""One-time migration: move triples from default graph to named graphs."""

from __future__ import annotations

import hashlib
import logging

from pyoxigraph import Literal, NamedNode, Quad, Store, Triple

from knowledge_service.ontology.namespaces import (
    KS,
    KS_GRAPH_ASSERTED,
    KS_GRAPH_EXTRACTED,
    KS_GRAPH_ONTOLOGY,
)

logger = logging.getLogger(__name__)

MIGRATION_MARKER_S = NamedNode(f"{KS}migration/named_graphs")
MIGRATION_MARKER_P = NamedNode(f"{KS}completedAt")


def _is_migration_done(store: Store) -> bool:
    quads = list(store.quads_for_pattern(
        MIGRATION_MARKER_S, MIGRATION_MARKER_P, None, NamedNode(KS_GRAPH_ONTOLOGY)
    ))
    return len(quads) > 0


def _triple_hash(triple: Triple) -> str:
    return hashlib.sha256(str(triple).encode()).hexdigest()


async def migrate_to_named_graphs(store: Store, pg_pool) -> int:
    """Migrate triples from default graph to named graphs.

    Returns number of triples migrated, or 0 if already done.
    """
    if _is_migration_done(store):
        logger.info("Named graph migration already completed, skipping")
        return 0

    # Phase A: Read all triples from default graph
    from pyoxigraph import DefaultGraph
    default_quads = list(store.quads_for_pattern(None, None, None, DefaultGraph()))
    if not default_quads:
        logger.info("No triples in default graph, nothing to migrate")
        return 0

    logger.info("Migrating %d quads from default graph to named graphs", len(default_quads))

    # Identify ontology triples (KS namespace subjects that are class/property definitions)
    ontology_subjects = set()
    for q in default_quads:
        s_val = q.subject.value if hasattr(q.subject, "value") else str(q.subject)
        if s_val.startswith(KS):
            ontology_subjects.add(s_val)

    # Batch lookup provenance extractors
    triple_hashes = {}
    for q in default_quads:
        t = Triple(q.subject, q.predicate, q.object)
        triple_hashes[_triple_hash(t)] = q

    extractor_map = {}
    if pg_pool is not None:
        async with pg_pool.acquire() as conn:
            hashes = list(triple_hashes.keys())
            # Batch in chunks of 500
            for i in range(0, len(hashes), 500):
                batch = hashes[i : i + 500]
                rows = await conn.fetch(
                    "SELECT DISTINCT triple_hash, extractor FROM provenance WHERE triple_hash = ANY($1)",
                    batch,
                )
                for row in rows:
                    extractor_map[row["triple_hash"]] = row["extractor"]

    # Phase A: Copy to named graphs
    migrated = 0
    for q in default_quads:
        s_val = q.subject.value if hasattr(q.subject, "value") else str(q.subject)
        t = Triple(q.subject, q.predicate, q.object)
        th = _triple_hash(t)

        if s_val in ontology_subjects:
            target = KS_GRAPH_ONTOLOGY
        elif extractor_map.get(th, "").startswith("llm_"):
            target = KS_GRAPH_EXTRACTED
        else:
            target = KS_GRAPH_ASSERTED

        store.add(Quad(q.subject, q.predicate, q.object, NamedNode(target)))
        migrated += 1

    # Phase B: Delete from default graph
    for q in default_quads:
        store.remove(q)

    # Write completion marker
    from datetime import datetime

    store.add(Quad(
        MIGRATION_MARKER_S,
        MIGRATION_MARKER_P,
        Literal(datetime.now().isoformat()),
        NamedNode(KS_GRAPH_ONTOLOGY),
    ))

    store.flush()
    logger.info("Named graph migration complete: %d triples migrated", migrated)
    return migrated
```

- [ ] **Step 2: Wire migration into main.py lifespan**

In `main.py`, after the line where `bootstrap_ontology` is called and after `pg_pool` is created, add:

```python
from knowledge_service.stores.graph_migration import migrate_to_named_graphs
await migrate_to_named_graphs(knowledge_store.store, pg_pool)
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_service/stores/graph_migration.py src/knowledge_service/main.py
git commit -m "feat: one-time migration script for default graph to named graphs"
```

---

## Phase 3: Chunk Provenance

### Task 12: SQL migration for chunk_id column

**Files:**
- Create: `migrations/004_add_chunk_provenance.sql`

- [ ] **Step 1: Create migration file**

```sql
ALTER TABLE provenance ADD COLUMN chunk_id UUID REFERENCES content(id) ON DELETE SET NULL;
CREATE INDEX idx_provenance_chunk_id ON provenance(chunk_id);
```

- [ ] **Step 2: Commit**

```bash
git add migrations/004_add_chunk_provenance.sql
git commit -m "feat: add chunk_id column to provenance table"
```

### Task 13: Update ProvenanceStore for chunk_id

**Files:**
- Modify: `src/knowledge_service/stores/provenance.py:38-87` (insert)
- Modify: `src/knowledge_service/stores/provenance.py:93-98` (get_by_triple)
- Modify: `tests/test_provenance_store.py`

- [ ] **Step 1: Write failing test for chunk_id in provenance**

Add to `tests/test_provenance_store.py`:

```python
class TestChunkProvenance:
    async def test_insert_with_chunk_id(self, provenance_store):
        await provenance_store.insert(
            triple_hash="abc123",
            subject="http://s/1",
            predicate="http://p/1",
            object_="http://o/1",
            source_url="http://example.com",
            source_type="article",
            extractor="llm_qwen",
            confidence=0.8,
            chunk_id="550e8400-e29b-41d4-a716-446655440000",
        )
        rows = await provenance_store.get_by_triple("abc123")
        assert len(rows) == 1
        assert str(rows[0]["chunk_id"]) == "550e8400-e29b-41d4-a716-446655440000"

    async def test_insert_without_chunk_id(self, provenance_store):
        await provenance_store.insert(
            triple_hash="def456",
            subject="http://s/2",
            predicate="http://p/2",
            object_="http://o/2",
            source_url="http://example.com/2",
            source_type="article",
            extractor="api",
            confidence=0.9,
        )
        rows = await provenance_store.get_by_triple("def456")
        assert len(rows) == 1
        assert rows[0]["chunk_id"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_provenance_store.py::TestChunkProvenance -v`
Expected: FAIL — `insert() got an unexpected keyword argument 'chunk_id'`

- [ ] **Step 3: Add chunk_id to ProvenanceStore.insert()**

In `provenance.py`, update `insert()`:

Add `chunk_id: str | None = None` parameter after `valid_until`.

Update SQL to include `chunk_id`:
```python
sql = """
    INSERT INTO provenance (
        triple_hash, subject, predicate, object, source_url,
        source_type, extractor, confidence, metadata,
        valid_from, valid_until, chunk_id
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
    ON CONFLICT (triple_hash, source_url) DO UPDATE SET
        confidence  = EXCLUDED.confidence,
        metadata    = EXCLUDED.metadata,
        valid_from  = EXCLUDED.valid_from,
        valid_until = EXCLUDED.valid_until,
        chunk_id    = EXCLUDED.chunk_id
"""
```

Add `chunk_id` as the 12th parameter in `conn.execute()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_provenance_store.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/stores/provenance.py tests/test_provenance_store.py
git commit -m "feat: add chunk_id support to ProvenanceStore"
```

### Task 14: Update EmbeddingStore — return chunk IDs, add get_chunks_by_ids

**Files:**
- Modify: `src/knowledge_service/stores/embedding.py:120-150`
- Modify: `tests/test_embedding_store.py`

- [ ] **Step 1: Update insert_chunks to return IDs**

Change `insert_chunks` return type and use `fetchrow` with `RETURNING id`:

```python
async def insert_chunks(
    self,
    content_id: str,
    chunks: list[dict],
) -> list[tuple[int, str]]:
    """Insert chunk rows. Returns list of (chunk_index, chunk_id)."""
    if not chunks:
        return []

    sql = """
        INSERT INTO content (
            content_id, chunk_index, chunk_text, embedding, char_start, char_end
        )
        VALUES ($1, $2, $3, $4::vector(768), $5, $6)
        RETURNING id
    """

    results = []
    async with self._pool.acquire() as conn:
        for chunk in chunks:
            embedding_str = self._vector_to_str(chunk["embedding"])
            row = await conn.fetchrow(
                sql,
                content_id,
                chunk["chunk_index"],
                chunk["chunk_text"],
                embedding_str,
                chunk["char_start"],
                chunk["char_end"],
            )
            results.append((chunk["chunk_index"], str(row["id"])))
    return results
```

- [ ] **Step 2: Add get_chunks_by_ids method**

```python
async def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, str]:
    """Return {chunk_id: chunk_text} for the given IDs."""
    if not chunk_ids:
        return {}
    sql = "SELECT id, chunk_text FROM content WHERE id = ANY($1::uuid[])"
    async with self._pool.acquire() as conn:
        rows = await conn.fetch(sql, chunk_ids)
    return {str(r["id"]): r["chunk_text"] for r in rows}
```

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_service/stores/embedding.py
git commit -m "feat: insert_chunks returns IDs, add get_chunks_by_ids"
```

### Task 15: Update process_triple and content.py for chunk_id

**Files:**
- Modify: `src/knowledge_service/api/_ingest.py:11-19` (add chunk_id param)
- Modify: `src/knowledge_service/api/content.py:97-219` (per-chunk extraction)

- [ ] **Step 1: Add chunk_id to process_triple**

In `_ingest.py`, add `chunk_id: str | None = None` parameter to `process_triple()`:

```python
async def process_triple(
    t: dict,
    knowledge_store,
    pg_pool,
    reasoning_engine,
    source_url: str,
    source_type: str,
    extractor: str,
    chunk_id: str | None = None,
) -> tuple[bool, list[dict]]:
```

Pass it through to `provenance_store.insert()`:
```python
await provenance_store.insert(
    ...,
    chunk_id=chunk_id,
)
```

- [ ] **Step 2: Update content.py for per-chunk extraction with chunk IDs**

In `_process_one_content_request()`, after chunks are inserted (around line 159-160), capture returned IDs:

```python
await embedding_store.delete_chunks(content_id)
chunk_id_pairs = await embedding_store.insert_chunks(content_id, chunk_records)
chunk_id_map = dict(chunk_id_pairs)  # {chunk_index: chunk_id}
```

Replace the auto-extraction block (around lines 163-171):

```python
if not body.knowledge and body.raw_text:
    knowledge_by_chunk: list[tuple[list, str | None]] = []
    for chunk in chunk_records:
        items = await extraction_client.extract(
            chunk["chunk_text"], title=body.title, source_type=body.source_type
        )
        cid = chunk_id_map.get(chunk["chunk_index"])
        knowledge_by_chunk.append((items, cid))
    knowledge = []
    chunk_ids_for_items: list[str | None] = []
    for items, cid in knowledge_by_chunk:
        for item in items:
            knowledge.append(item)
            chunk_ids_for_items.append(cid)
    extracted_by_llm = bool(knowledge)
else:
    knowledge = list(body.knowledge)
    chunk_ids_for_items = [None] * len(knowledge)
    extracted_by_llm = False
```

Then in the triple processing loop (around line 189-202), pass chunk_id:

```python
item_idx = 0
for i, item in enumerate(knowledge):
    for t in expand_to_triples(item):
        cid = chunk_ids_for_items[i] if i < len(chunk_ids_for_items) else None
        is_new, contras = await process_triple(
            t, knowledge_store, pg_pool, reasoning_engine,
            body.url, body.source_type, extractor,
            chunk_id=cid,
        )
        if is_new:
            triples_created += 1
        contradictions_all.extend(contras)
```

- [ ] **Step 3: Run content ingestion tests**

Run: `uv run pytest tests/test_api_content.py tests/test_ingest.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_service/api/_ingest.py src/knowledge_service/api/content.py
git commit -m "feat: per-chunk extraction with chunk_id provenance tracking"
```

### Task 16: Update /api/ask with evidence snippets

**Files:**
- Modify: `src/knowledge_service/api/ask.py`
- Modify: `tests/test_api_ask.py`

- [ ] **Step 1: Write failing test for evidence in ask response**

Add to `tests/test_api_ask.py`:

```python
class TestAskEvidence:
    async def test_ask_returns_evidence_field(self, client):
        response = await client.post("/api/ask", json={"question": "test question"})
        assert response.status_code == 200
        data = response.json()
        assert "evidence" in data
        assert isinstance(data["evidence"], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_ask.py::TestAskEvidence -v`
Expected: FAIL — `evidence` not in response

- [ ] **Step 3: Add EvidenceSnippet model and evidence assembly**

In `ask.py`, add models:

```python
class EvidenceSnippet(BaseModel):
    triple_subject: str
    triple_predicate: str
    triple_object: str
    chunk_text: str
    source_url: str


class AskResponse(BaseModel):
    answer: str
    confidence: float | None
    sources: list[SourceInfo]
    knowledge_types_used: list[str]
    contradictions: list[ContradictionInfo]
    evidence: list[EvidenceSnippet] = []
```

In `post_ask()`, after building `sources`, add evidence assembly:

```python
# Evidence: look up chunk text for knowledge triples
evidence: list[EvidenceSnippet] = []
pg_pool = getattr(request.app.state, "pg_pool", None)
embedding_store = getattr(request.app.state, "embedding_store", None)

if pg_pool and embedding_store and context.knowledge_triples:
    provenance_store = ProvenanceStore(pg_pool)
    from knowledge_service._utils import _triple_hash

    chunk_ids_to_fetch: list[str] = []
    triple_prov_map: dict[str, list[dict]] = {}

    for t in context.knowledge_triples:
        th = _triple_hash(t["subject"], t["predicate"], t["object"])
        prov_rows = await provenance_store.get_by_triple(th)
        for row in prov_rows:
            if row.get("chunk_id"):
                chunk_ids_to_fetch.append(str(row["chunk_id"]))
        triple_prov_map[th] = prov_rows

    if chunk_ids_to_fetch:
        chunk_texts = await embedding_store.get_chunks_by_ids(chunk_ids_to_fetch)
        for t in context.knowledge_triples:
            th = _triple_hash(t["subject"], t["predicate"], t["object"])
            for row in triple_prov_map.get(th, []):
                cid = str(row["chunk_id"]) if row.get("chunk_id") else None
                if cid and cid in chunk_texts:
                    evidence.append(EvidenceSnippet(
                        triple_subject=t["subject"],
                        triple_predicate=t["predicate"],
                        triple_object=t["object"],
                        chunk_text=chunk_texts[cid],
                        source_url=row.get("source_url", ""),
                    ))
```

Add `evidence=evidence` to the `AskResponse` return.

Note: `_triple_hash` already exists in `_utils.py` (line 18) — import it from there. No new function needed.

- [ ] **Step 4: Run ask tests**

Run: `uv run pytest tests/test_api_ask.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/api/ask.py src/knowledge_service/_utils.py tests/test_api_ask.py
git commit -m "feat: evidence snippets with chunk text in /api/ask response"
```

### Task 17: Final integration test

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Run lint**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No errors

- [ ] **Step 3: Fix any lint issues**

Run: `uv run ruff format .` if needed.

- [ ] **Step 4: Final commit if lint fixes were needed**

```bash
git add -A
git commit -m "chore: lint fixes for foundation improvements"
```
