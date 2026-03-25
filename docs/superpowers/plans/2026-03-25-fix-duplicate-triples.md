# Fix Duplicate Triple Reification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the broken idempotency guard in `insert_triple` that creates duplicate RDF-star annotation blank nodes, fix the broken `update_confidence` that silently fails, and add defensive DISTINCT to browse queries.

**Architecture:** The root cause is that pyoxigraph's SPARQL 1.2 `<< s p o >>` INSERT DATA creates reification blank nodes that are NOT discoverable via the Python API `quads_for_pattern(None, rdf:reifies, Triple, None)`. The fix switches all reification checks/mutations from the Python API to SPARQL queries using `<< >>` syntax, matching pyoxigraph's internal representation.

**Tech Stack:** pyoxigraph, SPARQL 1.2, FastAPI, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/knowledge_service/stores/knowledge.py` | Modify | Fix `insert_triple` idempotency guard (line ~161), fix `update_confidence` (line ~583) |
| `src/knowledge_service/admin/stats.py` | Modify | Add DISTINCT to `browse_triples` data+count queries (line ~239) |
| `tests/test_knowledge_store.py` | Modify | Add tests proving no duplicate annotations, test update_confidence actually works |
| `tests/test_admin_stats.py` | Modify | Add test that browse_triples returns no duplicates |

---

### Task 1: Fix `insert_triple` idempotency guard

**Files:**
- Test: `tests/test_knowledge_store.py`
- Modify: `src/knowledge_service/stores/knowledge.py:155-164`

- [ ] **Step 1: Write failing test that proves duplicate annotations are created**

In `tests/test_knowledge_store.py`, add to `TestInsertTriple`:

```python
def test_insert_triple_twice_does_not_duplicate_annotations(self, store):
    """Re-inserting same triple must not create duplicate RDF-star annotations."""
    args = dict(
        subject="http://knowledge.local/data/cold",
        predicate="http://knowledge.local/schema/increases",
        object_="http://knowledge.local/data/dopamine",
        confidence=0.9,
        knowledge_type="Claim",
    )
    store.insert_triple(**args)
    store.insert_triple(**args)

    results = store.get_triples_by_subject("http://knowledge.local/data/cold")
    assert len(results) == 1, f"Expected 1 result, got {len(results)} (duplicate annotations)"

def test_insert_triple_twice_with_literal_object_no_duplicates(self, store):
    """Idempotency guard must also work for literal (non-URI) objects."""
    args = dict(
        subject="http://knowledge.local/data/cold",
        predicate="http://knowledge.local/schema/dopamine_increase",
        object_="up to 250 percent",
        confidence=0.95,
        knowledge_type="Entity",
    )
    store.insert_triple(**args)
    store.insert_triple(**args)

    results = store.get_triples_by_subject("http://knowledge.local/data/cold")
    assert len(results) == 1, f"Expected 1 result, got {len(results)} (duplicate annotations)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_knowledge_store.py::TestInsertTriple::test_insert_triple_twice_does_not_duplicate_annotations -v`
Expected: FAIL — `assert 2 == 1` (or similar, proving duplicates exist)

- [ ] **Step 3: Fix the idempotency guard in `insert_triple`**

In `src/knowledge_service/stores/knowledge.py`, replace lines ~160-164:

The full modified section from line ~155 to ~182 should become:

```python
        graph_uri = graph or KS_GRAPH_EXTRACTED
        graph_node = NamedNode(graph_uri)

        # Insert base triple into named graph (idempotent — pyoxigraph deduplicates quads)
        self._store.add(Quad(s, p, o, graph_node))

        # Check if annotations already exist using SPARQL (matches pyoxigraph's
        # internal reification representation, unlike the Python API which cannot
        # find reification blank nodes created by SPARQL INSERT DATA with << >>).
        obj_sparql = _sparql_object(object_)
        ask_sparql = f"""
            ASK {{
                GRAPH ?g {{
                    << <{subject}> <{predicate}> {obj_sparql} >>
                        <{KS_CONFIDENCE.value}> ?conf .
                }}
            }}
        """
        if self._store.query(ask_sparql):
            return triple_hash, False

        # Insert RDF-star annotations via SPARQL UPDATE into the same named graph
        conf_literal = f'"{confidence}"^^<{XSD}float>'
        type_uri = f"<{KS}{knowledge_type}>"
```

Key changes: `obj_sparql` is computed once before the guard (no longer duplicated at line 167). The `RDF_REIFIES` Python API check is replaced with SPARQL ASK.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_knowledge_store.py::TestInsertTriple::test_insert_triple_twice_does_not_duplicate_annotations -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/test_knowledge_store.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_knowledge_store.py src/knowledge_service/stores/knowledge.py
git commit -m "fix: use SPARQL ASK for insert_triple idempotency guard

The Python API quads_for_pattern(None, rdf:reifies, Triple) cannot find
reification blank nodes created by SPARQL INSERT DATA with << s p o >>
syntax. Switch to SPARQL ASK which uses the same internal representation,
preventing duplicate annotation blank nodes on re-ingestion."
```

---

### Task 2: Fix `update_confidence`

**Files:**
- Test: `tests/test_knowledge_store.py`
- Modify: `src/knowledge_service/stores/knowledge.py:566-600`

- [ ] **Step 1: Write failing test that proves update_confidence actually changes the value**

The existing `test_update_changes_confidence` may be silently passing if `get_triples_by_subject` returns the original value (the update silently fails, but the query picks up the original annotation). Add a more rigorous test:

```python
def test_update_confidence_changes_value_verified_by_sparql(self, store):
    """Verify update_confidence actually modifies the stored confidence."""
    store.insert_triple(
        subject="http://knowledge.local/data/a",
        predicate="http://knowledge.local/schema/b",
        object_="http://knowledge.local/data/c",
        confidence=0.5,
        knowledge_type="Claim",
    )
    store.update_confidence(
        subject="http://knowledge.local/data/a",
        predicate="http://knowledge.local/schema/b",
        object_="http://knowledge.local/data/c",
        new_confidence=0.95,
    )
    # Query directly via SPARQL to verify the stored value
    results = store.query("""
        SELECT ?conf WHERE {
            GRAPH ?g {
                << <http://knowledge.local/data/a>
                   <http://knowledge.local/schema/b>
                   <http://knowledge.local/data/c> >>
                    <http://knowledge.local/schema/confidence> ?conf .
            }
        }
    """)
    assert len(results) == 1, f"Expected 1 confidence annotation, got {len(results)}"
    assert float(str(results[0]["conf"])) == pytest.approx(0.95)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_knowledge_store.py::TestUpdateConfidence::test_update_confidence_changes_value_verified_by_sparql -v`
Expected: FAIL — either confidence is still 0.5, or there are 2 confidence annotations

- [ ] **Step 3: Rewrite `update_confidence` to use SPARQL**

Replace the entire `update_confidence` method body in `knowledge.py`. Update the docstring too — the old one references the Python API approach.

**Risk note:** SPARQL DELETE/INSERT with `GRAPH ?g` variable + RDF-star `<< >>` is untested in this codebase. If pyoxigraph rejects the variable `?g` in this context, fall back to: (1) SELECT the graph URI first, (2) use concrete graph URIs in separate DELETE DATA / INSERT DATA statements.

```python
def update_confidence(
    self,
    subject: str,
    predicate: str,
    object_: str,
    new_confidence: float,
) -> None:
    """Update the confidence annotation on a triple.

    Uses SPARQL DELETE/INSERT with << >> syntax to match pyoxigraph's
    internal reification representation. Silently does nothing if the
    triple does not exist.
    """
    obj_sparql = _sparql_object(object_)
    conf_literal = f'"{new_confidence}"^^<{XSD}float>'

    # Use SPARQL DELETE/INSERT to update confidence annotation.
    # The << s p o >> syntax matches pyoxigraph's internal reification.
    sparql = f"""
        DELETE {{
            GRAPH ?g {{
                << <{subject}> <{predicate}> {obj_sparql} >>
                    <{KS_CONFIDENCE.value}> ?old_conf .
            }}
        }}
        INSERT {{
            GRAPH ?g {{
                << <{subject}> <{predicate}> {obj_sparql} >>
                    <{KS_CONFIDENCE.value}> {conf_literal} .
            }}
        }}
        WHERE {{
            GRAPH ?g {{
                << <{subject}> <{predicate}> {obj_sparql} >>
                    <{KS_CONFIDENCE.value}> ?old_conf .
            }}
        }}
    """
    self._store.update(sparql)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_knowledge_store.py::TestUpdateConfidence::test_update_confidence_changes_value_verified_by_sparql -v`
Expected: PASS

- [ ] **Step 5: Add test for update_confidence on non-existent triple**

```python
def test_update_confidence_on_nonexistent_triple_is_noop(self, store):
    """Updating confidence on a triple that doesn't exist should silently do nothing."""
    # Should not raise
    store.update_confidence(
        subject="http://knowledge.local/data/nonexistent",
        predicate="http://knowledge.local/schema/foo",
        object_="http://knowledge.local/data/bar",
        new_confidence=0.99,
    )
    results = store.get_triples_by_subject("http://knowledge.local/data/nonexistent")
    assert len(results) == 0
```

- [ ] **Step 6: Run all update_confidence tests**

Run: `uv run pytest tests/test_knowledge_store.py::TestUpdateConfidence -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_knowledge_store.py src/knowledge_service/stores/knowledge.py
git commit -m "fix: rewrite update_confidence to use SPARQL DELETE/INSERT

The Python API approach (quads_for_pattern with rdf:reifies) silently
failed because pyoxigraph SPARQL-created reifications aren't visible
via the Python quad API. Use SPARQL DELETE/INSERT with << >> syntax."
```

---

### Task 3: Add DISTINCT to browse_triples query

**Files:**
- Test: `tests/test_admin_stats.py`
- Modify: `src/knowledge_service/admin/stats.py:223-262`

- [ ] **Step 1: Write test that browse_triples returns no duplicates**

```python
async def test_triples_browse_no_duplicates(stats_client):
    """browse_triples must not return duplicate rows for the same triple."""
    resp = await stats_client.get("/api/admin/knowledge/triples")
    assert resp.status_code == 200
    data = resp.json()
    items = data["items"]
    # Check uniqueness by (subject, predicate, object) tuple
    seen = set()
    for item in items:
        key = (item["subject"], item["predicate"], item["object"])
        assert key not in seen, f"Duplicate triple in browse results: {key}"
        seen.add(key)
```

- [ ] **Step 2: Run test (should pass since test store is fresh, but validates the contract)**

Run: `uv run pytest tests/test_admin_stats.py::test_triples_browse_no_duplicates -v`
Expected: PASS (fresh in-memory store has no dupes)

- [ ] **Step 3: Add DISTINCT to the SPARQL queries**

In `src/knowledge_service/admin/stats.py`, change the data query (line ~239):

```python
# Before:
SELECT ?s ?p ?o ?conf ?ktype ?vfrom ?vuntil WHERE {

# After:
SELECT DISTINCT ?s ?p ?o ?conf ?ktype ?vfrom ?vuntil WHERE {
```

And the count query (line ~225):

```python
# Before:
SELECT (COUNT(*) AS ?cnt) WHERE {

# After:
SELECT (COUNT(*) AS ?cnt) WHERE {
    SELECT DISTINCT ?s ?p ?o WHERE {
```

For the count query, wrap the inner part in a subquery so COUNT operates on distinct triples:

Note: `?conf` must NOT be in the DISTINCT projection (otherwise different confidence values for the same triple would be counted separately). The confidence filter is inside the subquery WHERE, so `?conf` is bound there but not projected.

```python
count_sparql = f"""
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT (COUNT(*) AS ?cnt) WHERE {{
        SELECT DISTINCT ?s ?p ?o WHERE {{
            GRAPH ?g {{
                ?s ?p ?o .
            }}
            GRAPH ?g {{
                << ?s ?p ?o >> <{KS_CONFIDENCE.value}> ?conf .
            }}
            OPTIONAL {{
                GRAPH ?g {{ << ?s ?p ?o >> <{KS_KNOWLEDGE_TYPE.value}> ?ktype . }}
            }}
            {filter_clause}
        }}
    }}
"""
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_admin_stats.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_admin_stats.py src/knowledge_service/admin/stats.py
git commit -m "fix: add DISTINCT to browse_triples SPARQL queries

Defensive measure to prevent duplicate rows in the admin triple
browser, even if duplicate reification blank nodes exist in storage."
```

---

### Task 4: Clean up dead code

After Tasks 1 and 2, the `RDF_REIFIES` constant (line 43) and its usage are no longer needed. Check if any other code references it before removing.

- [ ] **Step 1: Search for `RDF_REIFIES` usage**

Run: `grep -rn RDF_REIFIES src/`
If only the constant definition remains (no usage), remove the line:

```python
RDF_REIFIES = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies")
```

- [ ] **Step 2: Commit if removed**

```bash
git add src/knowledge_service/stores/knowledge.py
git commit -m "chore: remove unused RDF_REIFIES constant"
```

---

### Task 5: Lint check and full test suite

- [ ] **Step 1: Run ruff**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No errors

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS

- [ ] **Step 3: Fix any issues, then commit if needed**

---

### Task 6: Production cleanup

This is a manual step after deploying the fix. With only 91 triples and 4 content items, the simplest approach is to wipe the oxigraph store and let n8n re-ingest:

- [ ] **Step 1: Document the cleanup approach**

After deploying the fixed image:
```bash
# SSH into the swarm node or use docker context
docker --context swarm-baa service update --force aegis_knowledge
# The oxigraph store is a Docker volume — to wipe it:
# 1. Scale down: docker --context swarm-baa service scale aegis_knowledge=0
# 2. Remove volume: docker --context swarm-baa volume rm aegis_knowledge_oxigraph
# 3. Scale up: docker --context swarm-baa service scale aegis_knowledge=1
# 4. Re-trigger n8n workflows to re-ingest content
```

Alternatively, if the data volume is small enough, just re-deploy and let future ingestions be clean (duplicates won't grow, and DISTINCT in queries hides existing ones).

- [ ] **Step 2: Deploy and verify**

After PR merge and CI build:
```bash
docker --context swarm-baa service update --image arshadansari27/knowledge-service:latest aegis_knowledge
# Verify:
curl -s https://knowledge.hikmahtech.in/api/admin/stats/counts -H "X-API-Key: <ADMIN_PASSWORD>"
```
