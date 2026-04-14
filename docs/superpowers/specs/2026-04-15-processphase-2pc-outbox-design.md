# ProcessPhase 2PC via Outbox — Design Spec

Date: 2026-04-15
Status: Approved (pending user review of spec document)
Scope: `ProcessPhase` in `src/knowledge_service/ingestion/phases.py` and the per-triple flow in `src/knowledge_service/ingestion/pipeline.py`.

## 1. Problem

`ProcessPhase.run()` writes to two stores with no coordination:

- **pyoxigraph** via `TripleStore` (triples, RDF-star annotations, inferred triples, confidence updates)
- **PostgreSQL** via `ProvenanceStore` (provenance rows) and entity/predicate embeddings

A mid-phase crash (SIGTERM, OOM, exception) can leave either of two partial states:

1. A triple in pyoxigraph with no corresponding provenance row in PostgreSQL.
2. A provenance row in PostgreSQL whose triple was never written to pyoxigraph.

Both are silent data-quality bugs: search/RAG readers will happily return partial state. The current code order is pyoxigraph-first, then PG, so case (1) is the likely failure mode today.

The spec author does not want (1) OR (2). Option A (strongly consistent) is more invasive than the problem warrants. Option C (periodic reconciler) leaves user-visible drift during the reconciliation interval. This design implements **Option B (outbox-driven eventual-but-bounded consistency)**, which:

- Never produces case (1) — pyoxigraph writes are projected from durable PG rows.
- Produces case (2) only for a bounded window between PG commit and outbox drain, and only in the "triple not yet visible" direction (the less dangerous one — readers see fewer facts, not phantom facts).
- Adds a single recovery mechanism (the drainer) that runs on startup and after each commit.

## 2. Invariant

> Every row in `provenance` references a triple that is either (a) already durable in pyoxigraph, or (b) present as an unapplied entry in `triple_outbox`. Never the inverse — pyoxigraph is never the sole holder of a triple whose provenance has not been committed.

**Convergence:** Any `triple_outbox` row with `applied_at IS NULL` will be drained either (a) synchronously by the per-transaction drainer immediately after commit, or (b) by the startup drainer on the next application start. No external reconciler is required.

**Reader semantics during the drain window:** A reader of `/api/ask` or SPARQL may see a `provenance` row for a triple whose SPARQL query returns nothing. This is bounded by the drain latency and is self-healing. Under the prior code, the inverse was possible (pyoxigraph triple with no provenance), which is strictly worse because RAG context would cite a source we have no record of.

## 3. Schema

New migration file: `migrations/NNNN_triple_outbox.sql` (NNNN = next available migration number).

```sql
CREATE TABLE triple_outbox (
    id              BIGSERIAL PRIMARY KEY,
    triple_hash     TEXT NOT NULL,
    operation       TEXT NOT NULL,
    subject         TEXT NOT NULL,
    predicate       TEXT NOT NULL,
    object          TEXT NOT NULL,
    confidence      DOUBLE PRECISION,
    knowledge_type  TEXT,
    valid_from      TIMESTAMPTZ,
    valid_until     TIMESTAMPTZ,
    graph           TEXT NOT NULL,
    payload         JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    applied_at      TIMESTAMPTZ
);

CREATE INDEX idx_outbox_pending ON triple_outbox (id) WHERE applied_at IS NULL;
CREATE INDEX idx_outbox_hash    ON triple_outbox (triple_hash);
```

**`operation` enum (string-typed):**

| operation            | Meaning                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `insert`             | Insert a base triple into `graph` with `confidence` + `knowledge_type`. |
| `update_confidence`  | Update the `ks:confidence` RDF-star annotation of an existing triple.   |
| `insert_inferred`    | Insert a derived triple into `ks:graph/inferred` plus RDF-star `ks:derivedFrom` / `ks:inferenceMethod` annotations carried in `payload`. |
| `retract_inference`  | Remove inferred triples whose `ks:derivedFrom` chain includes `triple_hash`, using the existing `retract_stale_inferences` cascade. |

**`payload` shape:**

- For `insert_inferred`: `{"derived_from": ["<hash1>", ...], "inference_method": "<method>"}`
- For other operations: may be `NULL` or `{}`.

No new columns on existing tables. No FK from `triple_outbox` to any other table — the outbox is append-only and can contain references to triples whose prior state has been rewritten.

## 4. Per-Triple Flow Restructure

The current `ingest_triple` in `pipeline.py` (lines 287–362) is restructured into four phases.

### 4.1 Phase A — Plan (reads only)

- Normalize object URI (unchanged from today).
- Compute `triple_hash` (unchanged).
- `detect_delta(triple, triple_store)` — pyoxigraph read, returns prior-value delta if any.

No writes in this phase.

### 4.2 Phase B — Base commit (one PG transaction)

```python
async with pg_pool.acquire() as conn, conn.transaction():
    await outbox.stage(
        conn,
        operation="insert",
        triple_hash=triple_hash,
        subject=triple["subject"],
        predicate=triple["predicate"],
        object=triple["object"],
        confidence=triple["confidence"],
        knowledge_type=triple["knowledge_type"],
        valid_from=triple.get("valid_from"),
        valid_until=triple.get("valid_until"),
        graph=context.graph,
    )
    if delta is not None:
        old_hash = compute_hash({
            "subject": triple["subject"],
            "predicate": triple["predicate"],
            "object": delta["prior_value"],
        })
        await outbox.stage(
            conn,
            operation="retract_inference",
            triple_hash=old_hash,
            subject=triple["subject"],
            predicate=triple["predicate"],
            object=delta["prior_value"],
            graph=KS_GRAPH_INFERRED,
        )
    await provenance.insert(
        conn,  # all provenance.insert call sites accept an explicit conn to join the txn
        triple_hash, triple["subject"], triple["predicate"], triple["object"],
        context.source_url, context.source_type, context.extractor,
        triple["confidence"], {},
        triple.get("valid_from"), triple.get("valid_until"), context.chunk_id,
    )
```

Commit of this transaction is the **point of no return** — pyoxigraph is now owed the writes recorded in the outbox rows we just inserted.

Provenance is inserted with the pre-contradiction (input) confidence. The post-contradiction penalty is applied in Phase D as an additional `update_confidence` outbox entry + a provenance-row update. This keeps Phase B self-contained and avoids a read of pyoxigraph (contradictions) inside the commit path.

### 4.3 Phase C — Synchronous drain of just-committed entries

Immediately after Phase B commits, the drainer replays the outbox rows produced by this transaction to pyoxigraph:

```python
await drainer.drain_ids([row_id_1, row_id_2, ...])
```

Failure here is non-fatal for data integrity — the rows remain `applied_at IS NULL` and will be picked up by the next call or the startup drainer. We log and continue.

### 4.4 Phase D — Derived work (each in its own small B/C cycle)

Derived work needs to read pyoxigraph AFTER Phase C has made the base triple visible.

1. **Contradictions + penalty**
   - Read: `detect_contradictions(triple, triple_store)` — pyoxigraph query.
   - `is_new` is captured from the Phase C drain: `OutboxDrainer.drain_ids` returns a list of `(id, operation, is_new)` tuples for `insert` operations, where `is_new` is propagated from `TripleStore.insert`. Re-ingestion of the same triple yields `is_new=False` and the penalty branch is skipped — matching the current code's guard.
   - If contradictions found and `is_new`:
     - Compute `penalty_confidence = apply_penalty(...)`.
     - Run a new PG txn (B'): stage `update_confidence` outbox + update the provenance row's `confidence` column. Commit. Drain.
   - Combine evidence across all provenance rows via Noisy-OR; if it differs from the post-penalty confidence, run another B'/C cycle.

2. **Inference**
   - Run `engine.run(normalized_triple)`.
   - For each derived triple: run a PG txn (B''): stage `insert_inferred` outbox (with `payload={"derived_from": [...], "inference_method": ...}`) + insert an inference provenance row. Commit. Drain.

3. **Thesis impact** — read-only against PG, unchanged.

**Crash tolerance of Phase D:** If the worker crashes anywhere inside Phase D, the base triple is durable in both stores (Phase B + C complete). Derived work is skipped. Re-ingestion of the same content re-runs `ingest_triple`, which:

- Re-runs Phase B: outbox `insert` is idempotent at the pyoxigraph layer (content-addressed hash); provenance insert hits the existing `(triple_hash, source_url)` UNIQUE — this constraint must be verified to exist (see §8 Open items).
- Re-runs Phase D: `engine.run()` is deterministic; inferred-triple inserts are content-addressed and RDF-star annotations are guarded by SPARQL ASK per `lesson_pyoxigraph_rdfstar`.

## 5. Drainer

New module: `src/knowledge_service/ingestion/outbox.py`.

### 5.1 `OutboxStore` (staging)

```python
class OutboxStore:
    async def stage(self, conn, *, operation, triple_hash, subject, predicate, object,
                    confidence=None, knowledge_type=None, valid_from=None,
                    valid_until=None, graph, payload=None) -> int:
        """Insert a pending row inside the caller's transaction. Returns row id."""
```

### 5.2 `OutboxDrainer`

```python
class OutboxDrainer:
    def __init__(self, pg_pool, triple_store):
        ...

    async def drain_ids(self, ids: list[int]) -> int:
        """Drain a specific set of ids (used post-commit). Returns count applied."""

    async def drain_pending(self, limit: int | None = None) -> int:
        """Drain all pending rows (used at startup). Returns count applied."""
```

Both methods use:
- `SELECT ... FROM triple_outbox WHERE applied_at IS NULL AND id = ANY($1) FOR UPDATE SKIP LOCKED`
  (or without `id = ANY($1)` for `drain_pending`).
- For each row, dispatch by `operation`:

| operation            | Applied via                                                              |
|----------------------|--------------------------------------------------------------------------|
| `insert`             | `triple_store.insert(subject, predicate, object, confidence, knowledge_type, valid_from, valid_until, graph)` — wrapped in `asyncio.to_thread`. |
| `update_confidence`  | `triple_store.update_confidence(triple_dict, confidence)` — wrapped in `asyncio.to_thread`. |
| `insert_inferred`    | `triple_store.insert(...)` into `KS_GRAPH_INFERRED` + SPARQL `INSERT DATA` for `ks:derivedFrom` / `ks:inferenceMethod` annotations. SPARQL ASK guard for idempotency on the annotations (per `lesson_pyoxigraph_rdfstar`). |
| `retract_inference`  | `retract_stale_inferences(triple_hash, triple_store)` — wrapped in `asyncio.to_thread`. |

- After the pyoxigraph side succeeds: `UPDATE triple_outbox SET applied_at = NOW() WHERE id = $1` in the same connection.
- If the pyoxigraph call raises, the row is left with `applied_at IS NULL` and the drainer logs the exception and continues to the next row. The next invocation will retry.

### 5.3 Idempotency guarantees for each operation

- `insert`: `TripleStore.insert` already returns `(hash, is_new)` — re-inserting the same content-addressed triple is a no-op. (Verified in `stores/triples.py`; the test suite covers this.)
- `insert_inferred`: SPARQL ASK before inserting annotations. The triple itself is content-addressed; the annotations must not accumulate on re-apply.
- `update_confidence`: Setting the annotation to the target value is idempotent — running it twice produces the same final state.
- `retract_inference`: The underlying `retract_stale_inferences` queries for inferred rows with `ks:derivedFrom "<hash>"`; when they've already been retracted, the query returns zero rows and the call is a no-op.

### 5.4 Invocation points

1. **After each Phase B / B' / B'' commit:** `drainer.drain_ids([ids])` — synchronous, inline.
2. **Application startup:** in `main.py` `lifespan`, after migrations run and before the app begins serving traffic, call `drainer.drain_pending()` once. This recovers from any prior-process crash that left rows pending.

The existing startup janitor (which marks non-terminal `ingestion_jobs` as failed) is untouched. The outbox drainer is a distinct recovery mechanism for the per-triple two-store write layer.

## 6. Stores wiring

- `Stores` dataclass in `src/knowledge_service/stores/__init__.py` gains:
  ```python
  outbox: OutboxStore
  ```
- `main.py` `lifespan` constructs `OutboxStore(pg_pool)` and an `OutboxDrainer(pg_pool, triple_store)`, stashing the drainer on `app.state.outbox_drainer`. The drainer is also passed into `ingest_triple` so post-commit drains can run inline.
- `ProvenanceStore.insert` gains an optional `conn` parameter so the call can join the caller's PG transaction. If `conn is None`, it acquires one from the pool (backward compatible). All other `ProvenanceStore` methods are unchanged. If an existing method signature makes threading `conn` through awkward (e.g., due to helpers), the implementation may add a parallel `insert_in_txn(conn, ...)` method rather than altering the existing signature — this is an implementation detail, not a design constraint.

## 7. Crash-injection tests

New file: `tests/test_ingestion_crash_injection.py`. Uses the existing in-memory pyoxigraph fixture pattern (`TripleStore(data_dir=None)`) and the existing PG test scaffolding (asyncpg against a test database or the CI-provided pool). If no PG fixture exists in `tests/`, this test file uses a lightweight async mock that tracks inserted rows and supports raising on a target call — this is sufficient because the invariants we test are about ordering and idempotency, not PG-specific semantics.

### 7.1 Test cases

1. **`test_crash_after_base_commit_before_drain`**
   - Patch `OutboxDrainer.drain_ids` to raise after the PG txn commits.
   - Run `ingest_triple`.
   - Assert: provenance has 1 row; `triple_outbox` has 1 row with `applied_at IS NULL`; pyoxigraph has 0 triples.
   - Un-patch, call `drainer.drain_pending()`.
   - Assert: pyoxigraph has the triple; outbox row has `applied_at IS NOT NULL`.

2. **`test_crash_mid_drain`**
   - Stage 3 outbox rows directly.
   - Patch `triple_store.insert` to raise on the 2nd call.
   - Call `drain_pending()`.
   - Assert: row 1 `applied_at` set; rows 2–3 pending; pyoxigraph has 1 triple.
   - Un-patch, call `drain_pending()` again.
   - Assert: all three rows applied; pyoxigraph has 3 triples.

3. **`test_crash_before_base_commit`**
   - Patch `outbox.stage` to raise on the 2nd staging call inside the PG txn.
   - Run `ingest_triple`.
   - Assert: PG has no provenance row; PG has no outbox row (txn rolled back); pyoxigraph has no triple.

4. **`test_idempotent_redrain`**
   - Ingest one triple normally (including drain).
   - Reset the outbox row's `applied_at` to `NULL` directly in the DB (simulating a partial drain-then-marker-failure).
   - Call `drain_pending()`.
   - Assert: no exception; pyoxigraph triple count unchanged; outbox row `applied_at` re-set.

5. **`test_startup_drainer`**
   - Insert pending outbox rows directly.
   - Construct a fresh `OutboxDrainer` (simulating process restart).
   - Call `drain_pending()`.
   - Assert: all rows applied; pyoxigraph converged.

6. **`test_inferred_triple_crash_recovery`**
   - Ingest a triple whose inference engine derives N triples.
   - Patch the drain on inference entries to raise.
   - Assert: base triple is durable; inferred outbox rows pending; inferred pyoxigraph triples absent.
   - Un-patch, drain.
   - Assert: inferred triples present with RDF-star annotations.

### 7.2 Non-goals for this PR

- No property-based tests of arbitrary crash points — the 6 cases above cover the state-machine boundaries.
- No stress/load test of the drainer under concurrency — `SKIP LOCKED` is the mechanism; we assert correctness via the deterministic cases above.

## 8. Open items (flag-only, not blockers)

- **Provenance UNIQUE constraint:** Re-ingest idempotency depends on `provenance` having a UNIQUE constraint on `(triple_hash, source_url)` (or equivalent). Implementation will verify and add a migration if missing. If absent, re-ingest would produce duplicate provenance rows — still correct per the Noisy-OR combiner, but wasteful.

## 9. Files changed

| File                                                        | Change  |
|-------------------------------------------------------------|---------|
| `migrations/NNNN_triple_outbox.sql`                         | New     |
| `src/knowledge_service/ingestion/outbox.py`                 | New     |
| `src/knowledge_service/ingestion/pipeline.py`               | Restructured `ingest_triple` + helpers |
| `src/knowledge_service/stores/__init__.py`                  | Add `outbox` to `Stores` dataclass     |
| `src/knowledge_service/stores/provenance.py`                | Accept optional `conn` in `insert`     |
| `src/knowledge_service/main.py`                             | Construct drainer; call `drain_pending()` in lifespan |
| `CLAUDE.md`                                                 | New "ProcessPhase consistency" section |
| `tests/test_ingestion_crash_injection.py`                   | New     |
| `tests/test_pipeline.py` (existing)                         | Adjust for new ordering if assertions depend on it |

## 10. Explicitly out of scope

- `EmbedPhase` — handled in another PR per the user.
- Stuck-job janitor in `main.py` lifespan — handled separately per the user.
- Inference engine internals (`reasoning/engine.py`) — no rule changes.
- Entity/predicate resolution (`stores/entities.py`) — no transactional changes.
- Any change to `ingestion_jobs` semantics or `JobTracker`.
- Cross-document 2PC — this PR only addresses per-triple two-store coordination.
