# ProcessPhase 2PC via Outbox — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate silent data-quality bugs where `ProcessPhase` leaves triples in pyoxigraph without matching provenance rows in PostgreSQL (or vice versa), by introducing a PG-side outbox that is drained to pyoxigraph after commit with idempotent replay on restart.

**Architecture:** Option B from the design spec — PG commit becomes the point of no return. A new `triple_outbox` table stages each pyoxigraph write alongside the matching provenance row inside a single PG transaction. A drainer replays staged entries to pyoxigraph idempotently, synchronously after each commit and on application startup. Derived work (contradictions penalty, inference) happens after drain and produces additional staged entries, each in its own tiny B+C cycle.

**Tech Stack:** Python 3.12, asyncpg, pyoxigraph, FastAPI lifespan, pytest / pytest-asyncio, `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-04-15-processphase-2pc-outbox-design.md`

---

## File Structure

**New files:**
- `migrations/014_triple_outbox.sql` — schema
- `src/knowledge_service/ingestion/outbox.py` — `OutboxStore`, `OutboxDrainer`, `AppliedEntry` dataclass
- `tests/test_outbox.py` — unit tests for `OutboxStore` + `OutboxDrainer`
- `tests/test_ingestion_crash_injection.py` — integration crash-injection tests

**Modified files:**
- `src/knowledge_service/stores/__init__.py` — add `outbox` field to `Stores`
- `src/knowledge_service/stores/provenance.py` — add `conn`-threaded insert path
- `src/knowledge_service/ingestion/pipeline.py` — restructure `ingest_triple`
- `src/knowledge_service/main.py` — construct drainer; call `drain_pending()` in lifespan
- `tests/test_pipeline.py` — update existing `TestIngestTriple` assertions for new flow
- `CLAUDE.md` — add "ProcessPhase consistency" section

---

## Task 1: Migration for `triple_outbox` table

**Files:**
- Create: `migrations/014_triple_outbox.sql`

- [ ] **Step 1: Write the migration**

Create `migrations/014_triple_outbox.sql`:

```sql
-- Outbox table for coordinating pyoxigraph writes with PG transactions.
-- See docs/superpowers/specs/2026-04-15-processphase-2pc-outbox-design.md

CREATE TABLE IF NOT EXISTS triple_outbox (
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

CREATE INDEX IF NOT EXISTS idx_outbox_pending
    ON triple_outbox (id) WHERE applied_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_outbox_hash
    ON triple_outbox (triple_hash);
```

- [ ] **Step 2: Verify migration file parses**

Run: `python -c "open('migrations/014_triple_outbox.sql').read()"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add migrations/014_triple_outbox.sql
git commit -m "feat: add triple_outbox table for PG/pyoxigraph coordination"
```

---

## Task 2: `OutboxStore` module skeleton + `stage`

**Files:**
- Create: `src/knowledge_service/ingestion/outbox.py`
- Create: `tests/test_outbox.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_outbox.py`:

```python
# tests/test_outbox.py
from unittest.mock import AsyncMock
import pytest
from knowledge_service.ingestion.outbox import OutboxStore


class TestOutboxStoreStage:
    async def test_stage_returns_inserted_id(self):
        conn = AsyncMock()
        conn.fetchval.return_value = 42
        store = OutboxStore()

        returned_id = await store.stage(
            conn,
            operation="insert",
            triple_hash="abc",
            subject="s",
            predicate="p",
            object_="o",
            confidence=0.8,
            knowledge_type="claim",
            graph="http://ks/graph/extracted",
        )

        assert returned_id == 42
        # One call into PG
        assert conn.fetchval.await_count == 1
        args, _ = conn.fetchval.call_args
        # First positional is the SQL string
        assert "INSERT INTO triple_outbox" in args[0]
        # Params follow — check a couple of them
        assert args[1] == "abc"             # triple_hash
        assert args[2] == "insert"          # operation

    async def test_stage_serialises_payload(self):
        conn = AsyncMock()
        conn.fetchval.return_value = 1
        store = OutboxStore()

        await store.stage(
            conn,
            operation="insert_inferred",
            triple_hash="xyz",
            subject="s",
            predicate="p",
            object_="o",
            graph="http://ks/graph/inferred",
            payload={"derived_from": ["h1"], "inference_method": "inverse"},
        )

        args, _ = conn.fetchval.call_args
        # payload is last positional; expect JSON-serialised string
        payload_param = args[-1]
        assert '"derived_from"' in payload_param
        assert '"inverse"' in payload_param
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_outbox.py::TestOutboxStoreStage -v`
Expected: `ModuleNotFoundError: No module named 'knowledge_service.ingestion.outbox'`.

- [ ] **Step 3: Create the module + `OutboxStore`**

Create `src/knowledge_service/ingestion/outbox.py`:

```python
"""Outbox for coordinating pyoxigraph writes with PG transactions.

See docs/superpowers/specs/2026-04-15-processphase-2pc-outbox-design.md
for the invariant and recovery semantics.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class OutboxStore:
    """Staging surface for pyoxigraph writes, written inside a PG transaction."""

    _STAGE_SQL = """
        INSERT INTO triple_outbox (
            triple_hash, operation, subject, predicate, object,
            confidence, knowledge_type, valid_from, valid_until,
            graph, payload
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        RETURNING id
    """

    async def stage(
        self,
        conn: Any,
        *,
        operation: str,
        triple_hash: str,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float | None = None,
        knowledge_type: str | None = None,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        graph: str,
        payload: dict | None = None,
    ) -> int:
        """Insert a pending outbox row using the caller's connection/transaction.

        Returns the assigned id so the caller can drain exactly these rows.
        """
        payload_json = json.dumps(payload) if payload is not None else None
        return await conn.fetchval(
            self._STAGE_SQL,
            triple_hash,
            operation,
            subject,
            predicate,
            object_,
            confidence,
            knowledge_type,
            valid_from,
            valid_until,
            graph,
            payload_json,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_outbox.py::TestOutboxStoreStage -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/ingestion/outbox.py tests/test_outbox.py
git commit -m "feat: OutboxStore.stage writes pending rows in caller's transaction"
```

---

## Task 3: `OutboxDrainer` — insert operation

**Files:**
- Modify: `src/knowledge_service/ingestion/outbox.py`
- Modify: `tests/test_outbox.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_outbox.py`:

```python
from knowledge_service.ingestion.outbox import OutboxDrainer
from knowledge_service.stores.triples import TripleStore
from knowledge_service.ontology.namespaces import KS_GRAPH_EXTRACTED
from knowledge_service.ontology.bootstrap import bootstrap_ontology
from pathlib import Path


def _build_triple_store():
    ts = TripleStore(data_dir=None)
    ontology_dir = Path(__file__).resolve().parent.parent / "src" / "knowledge_service" / "ontology"
    bootstrap_ontology(ts, ontology_dir)
    return ts


class _FakePool:
    """Minimal asyncpg-shaped pool that hands out a single tracked connection."""

    def __init__(self):
        self.rows = []        # staged rows (dict-like)
        self.applied_ids = set()
        self.next_id = 1

    def acquire(self):
        return _FakeAcquire(self)


class _FakeAcquire:
    def __init__(self, pool):
        self._pool = pool

    async def __aenter__(self):
        return _FakeConn(self._pool)

    async def __aexit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, pool):
        self._pool = pool

    def transaction(self):
        return _FakeTxn()

    async def fetchval(self, sql, *args):
        # Mimic stage insert: assign id, store row.
        rid = self._pool.next_id
        self._pool.next_id += 1
        row = {
            "id": rid,
            "triple_hash": args[0],
            "operation": args[1],
            "subject": args[2],
            "predicate": args[3],
            "object": args[4],
            "confidence": args[5],
            "knowledge_type": args[6],
            "valid_from": args[7],
            "valid_until": args[8],
            "graph": args[9],
            "payload": args[10],
            "applied_at": None,
        }
        self._pool.rows.append(row)
        return rid

    async def fetch(self, sql, *args):
        # Return pending rows matching ids if filtered, else all pending.
        if args and isinstance(args[0], list):
            ids = set(args[0])
            return [r for r in self._pool.rows if r["id"] in ids and r["applied_at"] is None]
        return [r for r in self._pool.rows if r["applied_at"] is None]

    async def execute(self, sql, *args):
        # Used by UPDATE ... SET applied_at = NOW() WHERE id = $1
        if "applied_at" in sql:
            target = args[0]
            for r in self._pool.rows:
                if r["id"] == target:
                    r["applied_at"] = "now"
                    self._pool.applied_ids.add(target)
        return "UPDATE 1"


class _FakeTxn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class TestOutboxDrainerInsert:
    async def test_drain_ids_applies_insert_to_pyoxigraph(self):
        ts = _build_triple_store()
        pool = _FakePool()
        store = OutboxStore()

        async with pool.acquire() as conn:
            rid = await store.stage(
                conn,
                operation="insert",
                triple_hash="ignored",
                subject="http://knowledge.local/data/cat",
                predicate="http://knowledge.local/is_a",
                object_="http://knowledge.local/data/animal",
                confidence=0.9,
                knowledge_type="claim",
                graph=KS_GRAPH_EXTRACTED,
            )

        drainer = OutboxDrainer(pool, ts)
        applied = await drainer.drain_ids([rid])

        assert len(applied) == 1
        assert applied[0].operation == "insert"
        assert applied[0].is_new is True
        # Triple is actually in pyoxigraph
        rows = ts.get_triples(subject="http://knowledge.local/data/cat")
        assert len(rows) == 1
        # Row is marked applied
        assert rid in pool.applied_ids

    async def test_drain_ids_is_idempotent(self):
        ts = _build_triple_store()
        pool = _FakePool()
        store = OutboxStore()
        async with pool.acquire() as conn:
            rid = await store.stage(
                conn,
                operation="insert",
                triple_hash="ignored",
                subject="http://knowledge.local/data/cat",
                predicate="http://knowledge.local/is_a",
                object_="http://knowledge.local/data/animal",
                confidence=0.9,
                knowledge_type="claim",
                graph=KS_GRAPH_EXTRACTED,
            )

        drainer = OutboxDrainer(pool, ts)
        first = await drainer.drain_ids([rid])
        # Manually re-mark as pending to simulate a crash after pyoxigraph
        # apply but before the SET applied_at UPDATE landed.
        pool.rows[0]["applied_at"] = None
        pool.applied_ids.discard(rid)

        second = await drainer.drain_ids([rid])
        assert first[0].is_new is True
        # Pyoxigraph already has the triple — TripleStore.insert returns is_new=False
        assert second[0].is_new is False
        rows = ts.get_triples(subject="http://knowledge.local/data/cat")
        assert len(rows) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_outbox.py::TestOutboxDrainerInsert -v`
Expected: `ImportError` for `OutboxDrainer` (or equivalent).

- [ ] **Step 3: Implement `OutboxDrainer` with insert support**

Append to `src/knowledge_service/ingestion/outbox.py`:

```python
import asyncio
from dataclasses import dataclass as _dc


@_dc
class AppliedEntry:
    id: int
    operation: str
    triple_hash: str
    is_new: bool | None   # None for ops where is_new is not meaningful


class OutboxDrainer:
    """Applies pending outbox rows to pyoxigraph and marks them applied in PG."""

    _SELECT_BY_IDS = """
        SELECT id, triple_hash, operation, subject, predicate, object,
               confidence, knowledge_type, valid_from, valid_until,
               graph, payload
        FROM triple_outbox
        WHERE applied_at IS NULL AND id = ANY($1::bigint[])
        ORDER BY id
        FOR UPDATE SKIP LOCKED
    """

    _SELECT_PENDING = """
        SELECT id, triple_hash, operation, subject, predicate, object,
               confidence, knowledge_type, valid_from, valid_until,
               graph, payload
        FROM triple_outbox
        WHERE applied_at IS NULL
        ORDER BY id
        FOR UPDATE SKIP LOCKED
    """

    _MARK_APPLIED = "UPDATE triple_outbox SET applied_at = NOW() WHERE id = $1"

    def __init__(self, pool: Any, triple_store: Any) -> None:
        self._pool = pool
        self._triples = triple_store

    async def drain_ids(self, ids: list[int]) -> list[AppliedEntry]:
        if not ids:
            return []
        return await self._drain_rows(self._SELECT_BY_IDS, ids)

    async def drain_pending(self, limit: int | None = None) -> list[AppliedEntry]:
        sql = self._SELECT_PENDING
        if limit is not None:
            sql = sql + f" LIMIT {int(limit)}"
        return await self._drain_rows(sql, None)

    async def _drain_rows(self, sql: str, arg: Any) -> list[AppliedEntry]:
        applied: list[AppliedEntry] = []
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(sql, arg) if arg is not None else await conn.fetch(sql)
                for row in rows:
                    result = await self._apply_row(dict(row))
                    if result is None:
                        continue
                    await conn.execute(self._MARK_APPLIED, row["id"])
                    applied.append(result)
        return applied

    async def _apply_row(self, row: dict) -> AppliedEntry | None:
        op = row["operation"]
        if op == "insert":
            return await self._apply_insert(row)
        logger.warning("OutboxDrainer: unknown operation %r (row id=%s)", op, row["id"])
        return None

    async def _apply_insert(self, row: dict) -> AppliedEntry:
        triple_hash, is_new = await asyncio.to_thread(
            self._triples.insert,
            row["subject"],
            row["predicate"],
            row["object"],
            row["confidence"],
            row["knowledge_type"],
            row["valid_from"],
            row["valid_until"],
            row["graph"],
        )
        return AppliedEntry(
            id=row["id"],
            operation="insert",
            triple_hash=triple_hash,
            is_new=is_new,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_outbox.py::TestOutboxDrainerInsert -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/ingestion/outbox.py tests/test_outbox.py
git commit -m "feat: OutboxDrainer applies 'insert' operations idempotently"
```

---

## Task 4: `OutboxDrainer` — `update_confidence` operation

**Files:**
- Modify: `src/knowledge_service/ingestion/outbox.py`
- Modify: `tests/test_outbox.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_outbox.py`:

```python
class TestOutboxDrainerUpdateConfidence:
    async def test_drain_update_confidence(self):
        ts = _build_triple_store()
        pool = _FakePool()
        store = OutboxStore()

        # Seed: insert a triple so the confidence annotation exists.
        ts.insert(
            "http://knowledge.local/data/cat",
            "http://knowledge.local/is_a",
            "http://knowledge.local/data/animal",
            0.9,
            "claim",
            None,
            None,
            KS_GRAPH_EXTRACTED,
        )

        async with pool.acquire() as conn:
            rid = await store.stage(
                conn,
                operation="update_confidence",
                triple_hash="ignored",
                subject="http://knowledge.local/data/cat",
                predicate="http://knowledge.local/is_a",
                object_="http://knowledge.local/data/animal",
                confidence=0.4,
                graph=KS_GRAPH_EXTRACTED,
            )

        drainer = OutboxDrainer(pool, ts)
        applied = await drainer.drain_ids([rid])

        assert applied[0].operation == "update_confidence"
        # Query back the confidence
        rows = ts.get_triples(subject="http://knowledge.local/data/cat")
        assert rows[0]["confidence"] == 0.4
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_outbox.py::TestOutboxDrainerUpdateConfidence -v`
Expected: either operation is unknown (returns None → no applied entry) OR assertion error on confidence.

- [ ] **Step 3: Add the handler**

Modify `_apply_row` in `outbox.py`:

```python
    async def _apply_row(self, row: dict) -> AppliedEntry | None:
        op = row["operation"]
        if op == "insert":
            return await self._apply_insert(row)
        if op == "update_confidence":
            return await self._apply_update_confidence(row)
        logger.warning("OutboxDrainer: unknown operation %r (row id=%s)", op, row["id"])
        return None

    async def _apply_update_confidence(self, row: dict) -> AppliedEntry:
        triple_dict = {
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object": row["object"],
        }
        await asyncio.to_thread(
            self._triples.update_confidence, triple_dict, row["confidence"]
        )
        return AppliedEntry(
            id=row["id"],
            operation="update_confidence",
            triple_hash=row["triple_hash"],
            is_new=None,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_outbox.py::TestOutboxDrainerUpdateConfidence -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/ingestion/outbox.py tests/test_outbox.py
git commit -m "feat: OutboxDrainer applies update_confidence"
```

---

## Task 5: `OutboxDrainer` — `retract_inference` operation

**Files:**
- Modify: `src/knowledge_service/ingestion/outbox.py`
- Modify: `tests/test_outbox.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_outbox.py`:

```python
class TestOutboxDrainerRetractInference:
    async def test_drain_retract_inference_no_rows_is_noop(self):
        ts = _build_triple_store()
        pool = _FakePool()
        store = OutboxStore()

        async with pool.acquire() as conn:
            rid = await store.stage(
                conn,
                operation="retract_inference",
                triple_hash="nonexistent_hash",
                subject="http://knowledge.local/data/x",
                predicate="http://knowledge.local/p",
                object_="http://knowledge.local/data/y",
                graph="http://knowledge.local/graph/inferred",
            )

        drainer = OutboxDrainer(pool, ts)
        applied = await drainer.drain_ids([rid])
        assert applied[0].operation == "retract_inference"
        # Idempotent: re-run is also a no-op.
        pool.rows[0]["applied_at"] = None
        pool.applied_ids.discard(rid)
        again = await drainer.drain_ids([rid])
        assert again[0].operation == "retract_inference"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_outbox.py::TestOutboxDrainerRetractInference -v`
Expected: unknown op → no applied entry returned → IndexError or empty list assertion.

- [ ] **Step 3: Add the handler**

Extend `_apply_row` and add `_apply_retract_inference`:

```python
    async def _apply_row(self, row: dict) -> AppliedEntry | None:
        op = row["operation"]
        if op == "insert":
            return await self._apply_insert(row)
        if op == "update_confidence":
            return await self._apply_update_confidence(row)
        if op == "retract_inference":
            return await self._apply_retract_inference(row)
        logger.warning("OutboxDrainer: unknown operation %r (row id=%s)", op, row["id"])
        return None

    async def _apply_retract_inference(self, row: dict) -> AppliedEntry:
        from knowledge_service.ingestion.pipeline import retract_stale_inferences  # noqa: PLC0415
        await asyncio.to_thread(
            retract_stale_inferences, row["triple_hash"], self._triples
        )
        return AppliedEntry(
            id=row["id"],
            operation="retract_inference",
            triple_hash=row["triple_hash"],
            is_new=None,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_outbox.py::TestOutboxDrainerRetractInference -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/ingestion/outbox.py tests/test_outbox.py
git commit -m "feat: OutboxDrainer applies retract_inference"
```

---

## Task 6: `OutboxDrainer` — `insert_inferred` operation

**Files:**
- Modify: `src/knowledge_service/ingestion/outbox.py`
- Modify: `tests/test_outbox.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_outbox.py`:

```python
from knowledge_service.ontology.namespaces import KS_GRAPH_INFERRED


class TestOutboxDrainerInsertInferred:
    async def test_drain_insert_inferred(self):
        ts = _build_triple_store()
        pool = _FakePool()
        store = OutboxStore()

        async with pool.acquire() as conn:
            rid = await store.stage(
                conn,
                operation="insert_inferred",
                triple_hash="ignored",
                subject="http://knowledge.local/data/fluffy",
                predicate="http://knowledge.local/is_a",
                object_="http://knowledge.local/data/animal",
                confidence=0.72,
                knowledge_type="inferred",
                graph=KS_GRAPH_INFERRED,
                payload={"derived_from": ["h1", "h2"], "inference_method": "transitive"},
            )

        drainer = OutboxDrainer(pool, ts)
        applied = await drainer.drain_ids([rid])
        assert applied[0].operation == "insert_inferred"

        # Base triple exists in inferred graph
        rows = ts.get_triples(subject="http://knowledge.local/data/fluffy",
                              graphs=[KS_GRAPH_INFERRED])
        assert len(rows) == 1

        # ks:derivedFrom annotation is present
        ask = f"""
            ASK {{
                GRAPH <{KS_GRAPH_INFERRED}> {{
                    << <http://knowledge.local/data/fluffy>
                       <http://knowledge.local/is_a>
                       <http://knowledge.local/data/animal> >>
                    <http://knowledge.local/ks/derivedFrom> "h1" .
                }}
            }}
        """
        assert ts.query(ask) is True

    async def test_drain_insert_inferred_idempotent(self):
        ts = _build_triple_store()
        pool = _FakePool()
        store = OutboxStore()

        async with pool.acquire() as conn:
            rid = await store.stage(
                conn,
                operation="insert_inferred",
                triple_hash="ignored",
                subject="http://knowledge.local/data/fluffy",
                predicate="http://knowledge.local/is_a",
                object_="http://knowledge.local/data/animal",
                confidence=0.72,
                knowledge_type="inferred",
                graph=KS_GRAPH_INFERRED,
                payload={"derived_from": ["h1"], "inference_method": "transitive"},
            )

        drainer = OutboxDrainer(pool, ts)
        await drainer.drain_ids([rid])
        # Simulate crash-before-mark-applied
        pool.rows[0]["applied_at"] = None
        pool.applied_ids.discard(rid)
        await drainer.drain_ids([rid])

        # Count derivedFrom annotations — must be exactly 1.
        rows = ts.query(f"""
            SELECT (COUNT(*) AS ?c) WHERE {{
                GRAPH <{KS_GRAPH_INFERRED}> {{
                    << <http://knowledge.local/data/fluffy>
                       <http://knowledge.local/is_a>
                       <http://knowledge.local/data/animal> >>
                    <http://knowledge.local/ks/derivedFrom> "h1" .
                }}
            }}
        """)
        count = int(list(rows)[0]["c"].value)
        assert count == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_outbox.py::TestOutboxDrainerInsertInferred -v`
Expected: FAIL (unknown op).

- [ ] **Step 3: Add the handler**

Extend `_apply_row` and add `_apply_insert_inferred` in `outbox.py`:

```python
    async def _apply_row(self, row: dict) -> AppliedEntry | None:
        op = row["operation"]
        if op == "insert":
            return await self._apply_insert(row)
        if op == "update_confidence":
            return await self._apply_update_confidence(row)
        if op == "retract_inference":
            return await self._apply_retract_inference(row)
        if op == "insert_inferred":
            return await self._apply_insert_inferred(row)
        logger.warning("OutboxDrainer: unknown operation %r (row id=%s)", op, row["id"])
        return None

    async def _apply_insert_inferred(self, row: dict) -> AppliedEntry:
        from knowledge_service.ontology.uri import KS as KS_NS, is_uri  # noqa: PLC0415

        triple_hash, is_new = await asyncio.to_thread(
            self._triples.insert,
            row["subject"],
            row["predicate"],
            row["object"],
            row["confidence"],
            row["knowledge_type"] or "inferred",
            row["valid_from"],
            row["valid_until"],
            row["graph"],
        )

        payload_raw = row.get("payload")
        if isinstance(payload_raw, str):
            payload = json.loads(payload_raw) if payload_raw else {}
        else:
            payload = payload_raw or {}
        method = payload.get("inference_method", "")
        derived_from = payload.get("derived_from", [])

        obj_sparql = f"<{row['object']}>" if is_uri(row["object"]) else f'"{row["object"]}"'
        quoted = f"<< <{row['subject']}> <{row['predicate']}> {obj_sparql} >>"

        def _apply_annotations():
            # Method annotation — idempotent via ASK.
            if method:
                ask_method = f"""
                    ASK {{
                        GRAPH <{row['graph']}> {{
                            {quoted} <{KS_NS}inferenceMethod> "{method}" .
                        }}
                    }}
                """
                if not self._triples.store.query(ask_method):
                    self._triples.store.update(f"""
                        INSERT DATA {{
                            GRAPH <{row['graph']}> {{
                                {quoted} <{KS_NS}inferenceMethod> "{method}" .
                            }}
                        }}
                    """)
            # derivedFrom — each source hash guarded individually.
            for src in derived_from:
                ask_src = f"""
                    ASK {{
                        GRAPH <{row['graph']}> {{
                            {quoted} <{KS_NS}derivedFrom> "{src}" .
                        }}
                    }}
                """
                if not self._triples.store.query(ask_src):
                    self._triples.store.update(f"""
                        INSERT DATA {{
                            GRAPH <{row['graph']}> {{
                                {quoted} <{KS_NS}derivedFrom> "{src}" .
                            }}
                        }}
                    """)

        await asyncio.to_thread(_apply_annotations)
        return AppliedEntry(
            id=row["id"],
            operation="insert_inferred",
            triple_hash=triple_hash,
            is_new=is_new,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_outbox.py::TestOutboxDrainerInsertInferred -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/ingestion/outbox.py tests/test_outbox.py
git commit -m "feat: OutboxDrainer applies insert_inferred with idempotent RDF-star annotations"
```

---

## Task 7: `ProvenanceStore.insert` supports a caller-supplied connection

**Files:**
- Modify: `src/knowledge_service/stores/provenance.py`
- Test: `tests/test_provenance_store.py` (create if missing)

- [ ] **Step 1: Check if there's an existing test file**

Run: `ls tests/test_provenance* 2>&1`

If a file exists, append tests there. Otherwise create `tests/test_provenance_store.py`.

- [ ] **Step 2: Write the failing test**

Create/append to `tests/test_provenance_store.py`:

```python
# tests/test_provenance_store.py
from unittest.mock import AsyncMock, MagicMock
from knowledge_service.stores.provenance import ProvenanceStore


class TestProvenanceInsertWithConn:
    async def test_insert_with_caller_conn_does_not_acquire_pool(self):
        pool = MagicMock()
        pool.acquire = MagicMock()  # would raise if called (returns a non-async-context)
        conn = AsyncMock()

        store = ProvenanceStore(pool)
        await store.insert(
            triple_hash="abc",
            subject="s",
            predicate="p",
            object_="o",
            source_url="http://x",
            source_type="article",
            extractor="api",
            confidence=0.8,
            conn=conn,
        )
        # Pool must NOT be acquired — caller already holds a txn.
        pool.acquire.assert_not_called()
        # Insert went through caller's conn
        assert conn.execute.await_count == 1

    async def test_insert_without_conn_acquires_pool(self):
        pool_conn = AsyncMock()
        pool = MagicMock()

        class _Ctx:
            async def __aenter__(self_inner):
                return pool_conn
            async def __aexit__(self_inner, *exc):
                return False

        pool.acquire = MagicMock(return_value=_Ctx())

        store = ProvenanceStore(pool)
        await store.insert(
            triple_hash="abc",
            subject="s",
            predicate="p",
            object_="o",
            source_url="http://x",
            source_type="article",
            extractor="api",
            confidence=0.8,
        )
        pool.acquire.assert_called_once()
        assert pool_conn.execute.await_count == 1
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_provenance_store.py -v`
Expected: FAIL — `ProvenanceStore.insert` has no `conn` kwarg.

- [ ] **Step 4: Update `ProvenanceStore.insert`**

Modify `src/knowledge_service/stores/provenance.py`. Change the signature and body of `insert`:

```python
    async def insert(
        self,
        triple_hash: str,
        subject: str,
        predicate: str,
        object_: str,
        source_url: str,
        source_type: str,
        extractor: str,
        confidence: float,
        metadata: dict | None = None,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        chunk_id: str | None = None,
        conn: Any | None = None,
    ) -> None:
        """Upsert a provenance record.

        If ``conn`` is provided, the insert runs on the caller's connection (and
        the caller is responsible for transaction scope). Otherwise a connection
        is acquired from the pool for this call only.
        """
        metadata_json = json.dumps(metadata if metadata is not None else {})

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

        params = (
            triple_hash, subject, predicate, object_, source_url,
            source_type, extractor, confidence, metadata_json,
            valid_from, valid_until, chunk_id,
        )

        if conn is not None:
            await conn.execute(sql, *params)
            return
        async with self._pool.acquire() as c:
            await c.execute(sql, *params)
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/test_provenance_store.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/stores/provenance.py tests/test_provenance_store.py
git commit -m "feat: ProvenanceStore.insert accepts caller-supplied connection"
```

---

## Task 8: Wire `outbox` into the `Stores` dataclass

**Files:**
- Modify: `src/knowledge_service/stores/__init__.py`

- [ ] **Step 1: Add field**

Replace `src/knowledge_service/stores/__init__.py` with:

```python
from dataclasses import dataclass

from knowledge_service.stores.triples import TripleStore
from knowledge_service.stores.content import ContentStore
from knowledge_service.stores.entities import EntityStore
from knowledge_service.stores.provenance import ProvenanceStore
from knowledge_service.stores.theses import ThesisStore
from knowledge_service.ingestion.outbox import OutboxStore


@dataclass
class Stores:
    triples: TripleStore
    content: ContentStore
    entities: EntityStore
    provenance: ProvenanceStore
    theses: ThesisStore
    outbox: OutboxStore
    pg_pool: object  # asyncpg.Pool
```

- [ ] **Step 2: Update `main.py` construction**

In `src/knowledge_service/main.py` around line 147–154 (the `Stores(...)` constructor call), add the outbox:

```python
    from knowledge_service.ingestion.outbox import OutboxStore, OutboxDrainer  # noqa: PLC0415

    stores = Stores(
        triples=triple_store,
        content=ContentStore(pg_pool),
        entities=entity_store,
        provenance=ProvenanceStore(pg_pool),
        theses=ThesisStore(pg_pool),
        outbox=OutboxStore(),
        pg_pool=pg_pool,
    )
    app.state.stores = stores
    app.state.outbox_drainer = OutboxDrainer(pg_pool, triple_store)
```

- [ ] **Step 3: Run unit tests to surface callers that construct `Stores(...)` directly**

Run: `uv run pytest tests/ -x -q 2>&1 | head -40`
Expected: failures in any test that constructs `Stores(...)` without `outbox=`. Collect the list.

- [ ] **Step 4: Update every failing `Stores(...)` call site**

For each failing test, add `outbox=OutboxStore()` to the `Stores(...)` constructor. Import `OutboxStore` at the top of that test file:

```python
from knowledge_service.ingestion.outbox import OutboxStore
```

Typical candidates (confirm via failures from Step 3): `tests/test_api_content.py`, `tests/test_ingestion_worker.py`, `tests/test_api_knowledge.py`. Apply minimal edits only to the `Stores(...)` calls.

- [ ] **Step 5: Re-run tests**

Run: `uv run pytest tests/ -x -q 2>&1 | tail -10`
Expected: pre-existing tests pass (the new flow logic will still pass since `ingest_triple` hasn't been changed yet).

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/stores/__init__.py src/knowledge_service/main.py tests/
git commit -m "feat: wire OutboxStore into Stores dataclass and app lifespan"
```

---

## Task 9: Restructure `ingest_triple` — Phase A + Phase B (base commit)

**Files:**
- Modify: `src/knowledge_service/ingestion/pipeline.py`
- Modify: `tests/test_pipeline.py`

This task replaces the current `ingest_triple` with a restructured version that writes provenance + outbox in one PG txn, then drains. Contradictions/inference are still handled but after the drain.

- [ ] **Step 1: Read the current `ingest_triple` (lines 287–362) to confirm pre-restructure state**

Run: `uv run pytest tests/test_pipeline.py::TestIngestTriple -v`
Expected: all pass (baseline).

- [ ] **Step 2: Write the new failing test capturing the new flow**

Add to `tests/test_pipeline.py`:

```python
from knowledge_service.ingestion.outbox import OutboxStore, OutboxDrainer
from knowledge_service.stores.triples import TripleStore
from knowledge_service.ontology.bootstrap import bootstrap_ontology
from pathlib import Path


def _real_triple_store():
    ts = TripleStore(data_dir=None)
    ontology_dir = (
        Path(__file__).resolve().parent.parent
        / "src" / "knowledge_service" / "ontology"
    )
    bootstrap_ontology(ts, ontology_dir)
    return ts


class _PoolRecording:
    """Asyncpg-shaped pool that records staged rows and supports txns."""
    def __init__(self):
        self.provenance_rows: list[tuple] = []
        self.outbox_rows: list[dict] = []
        self.next_id = 1

    def acquire(self):
        return _RecAcq(self)


class _RecAcq:
    def __init__(self, pool):
        self.pool = pool

    async def __aenter__(self):
        return _RecConn(self.pool)

    async def __aexit__(self, *exc):
        return False


class _RecConn:
    def __init__(self, pool):
        self.pool = pool

    def transaction(self):
        return _RecTxn()

    async def execute(self, sql, *args):
        if sql.strip().startswith("INSERT INTO provenance"):
            self.pool.provenance_rows.append(args)
            return "INSERT 0 1"
        if "applied_at" in sql:
            target = args[0]
            for r in self.pool.outbox_rows:
                if r["id"] == target:
                    r["applied_at"] = "now"
        return "UPDATE 1"

    async def fetchval(self, sql, *args):
        rid = self.pool.next_id
        self.pool.next_id += 1
        row = {
            "id": rid,
            "triple_hash": args[0],
            "operation": args[1],
            "subject": args[2],
            "predicate": args[3],
            "object": args[4],
            "confidence": args[5],
            "knowledge_type": args[6],
            "valid_from": args[7],
            "valid_until": args[8],
            "graph": args[9],
            "payload": args[10],
            "applied_at": None,
        }
        self.pool.outbox_rows.append(row)
        return rid

    async def fetch(self, sql, *args):
        if args and isinstance(args[0], list):
            ids = set(args[0])
            return [r for r in self.pool.outbox_rows if r["id"] in ids and r["applied_at"] is None]
        return [r for r in self.pool.outbox_rows if r["applied_at"] is None]


class _RecTxn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class TestIngestTripleOutboxFlow:
    async def test_ingest_writes_outbox_and_provenance_in_one_txn(self):
        ts = _real_triple_store()
        pool = _PoolRecording()
        prov = ProvenanceStore(pool)
        outbox = OutboxStore()
        drainer = OutboxDrainer(pool, ts)

        class _StubThesis:
            async def find_by_hashes(self, hashes, status=None):
                return []

        stores = MagicMock()
        stores.triples = ts
        stores.provenance = prov
        stores.outbox = outbox
        stores.theses = _StubThesis()
        stores.pg_pool = pool

        ctx = IngestContext(
            source_url="http://test.com",
            source_type="article",
            extractor="api",
            graph=KS_GRAPH_EXTRACTED,
        )

        result = await ingest_triple(
            _triple(s="cat", p="is_a", o="animal"),
            stores,
            ctx,
            drainer=drainer,
        )
        assert result.is_new is True
        # Exactly one provenance row + one outbox row
        assert len(pool.provenance_rows) == 1
        assert len(pool.outbox_rows) == 1
        # Outbox row was drained
        assert pool.outbox_rows[0]["applied_at"] == "now"
        # Pyoxigraph actually got the triple
        rows = ts.get_triples(subject=f"{KS_DATA}cat")
        assert len(rows) == 1

    async def test_ingest_rolled_back_txn_leaves_no_state(self):
        """Staging raises INSIDE the transaction -> no provenance, no outbox, no pyoxigraph."""
        ts = _real_triple_store()
        pool = _PoolRecording()
        prov = ProvenanceStore(pool)

        class _BoomOutbox(OutboxStore):
            async def stage(self, *a, **kw):
                raise RuntimeError("boom")

        outbox = _BoomOutbox()
        drainer = OutboxDrainer(pool, ts)

        class _StubThesis:
            async def find_by_hashes(self, hashes, status=None):
                return []

        stores = MagicMock()
        stores.triples = ts
        stores.provenance = prov
        stores.outbox = outbox
        stores.theses = _StubThesis()
        stores.pg_pool = pool

        ctx = IngestContext(
            source_url="http://test.com",
            source_type="article",
            extractor="api",
            graph=KS_GRAPH_EXTRACTED,
        )

        with pytest.raises(RuntimeError, match="boom"):
            await ingest_triple(
                _triple(s="dog", p="is_a", o="animal"),
                stores,
                ctx,
                drainer=drainer,
            )

        # Pool fake does not actually roll back, but the ingest_triple code
        # MUST not call provenance.insert before outbox.stage succeeds.
        assert pool.provenance_rows == []
        assert pool.outbox_rows == []
        assert ts.get_triples(subject=f"{KS_DATA}dog") == []
```

- [ ] **Step 3: Run to verify the new tests fail**

Run: `uv run pytest tests/test_pipeline.py::TestIngestTripleOutboxFlow -v`
Expected: FAIL — `ingest_triple` doesn't accept `drainer=` and doesn't use outbox.

- [ ] **Step 4: Restructure `ingest_triple`**

Replace the `ingest_triple` function in `src/knowledge_service/ingestion/pipeline.py` (currently lines 287–362) with the staged version. Keep helpers (`compute_hash`, `detect_delta`, `detect_contradictions`, `apply_penalty`, `combine_evidence`, `check_thesis_impact`, `retract_stale_inferences`) unchanged.

```python
async def ingest_triple(
    triple: dict,
    stores,
    context: IngestContext,
    engine=None,
    drainer=None,
) -> IngestResult:
    """Ingest a single triple using the outbox pattern.

    Phase A: normalise + read prior state from pyoxigraph (no writes).
    Phase B: single PG txn — stage outbox rows + insert provenance. Commit
             is the point of no return.
    Phase C: drain just-committed outbox rows to pyoxigraph.
    Phase D: contradictions + inference, each in its own tiny B+C cycle.
    """
    from knowledge_service._utils import is_object_entity  # noqa: PLC0415
    from knowledge_service.ontology.uri import to_entity_uri as _to_entity_uri  # noqa: PLC0415

    if is_object_entity(triple) and not is_uri(triple.get("object", "")):
        triple = {**triple, "object": _to_entity_uri(triple["object"])}

    triple_hash = compute_hash(triple)

    # --- Phase A: plan -------------------------------------------------
    delta = await detect_delta(triple, stores.triples)

    # --- Phase B: base commit -----------------------------------------
    staged_ids: list[int] = []
    async with stores.pg_pool.acquire() as conn:
        async with conn.transaction():
            insert_id = await stores.outbox.stage(
                conn,
                operation="insert",
                triple_hash=triple_hash,
                subject=triple["subject"],
                predicate=triple["predicate"],
                object_=triple["object"],
                confidence=triple["confidence"],
                knowledge_type=triple["knowledge_type"],
                valid_from=triple.get("valid_from"),
                valid_until=triple.get("valid_until"),
                graph=context.graph,
            )
            staged_ids.append(insert_id)

            if delta is not None:
                old_triple = {
                    "subject": triple["subject"],
                    "predicate": triple["predicate"],
                    "object": delta["prior_value"],
                }
                old_hash = compute_hash(old_triple)
                retract_id = await stores.outbox.stage(
                    conn,
                    operation="retract_inference",
                    triple_hash=old_hash,
                    subject=old_triple["subject"],
                    predicate=old_triple["predicate"],
                    object_=old_triple["object"],
                    graph=context.graph,
                )
                staged_ids.append(retract_id)

            await stores.provenance.insert(
                triple_hash,
                triple["subject"],
                triple["predicate"],
                triple["object"],
                context.source_url,
                context.source_type,
                context.extractor,
                triple["confidence"],
                {},
                triple.get("valid_from"),
                triple.get("valid_until"),
                context.chunk_id,
                conn=conn,
            )
    # --- Phase C: drain the base commit -------------------------------
    applied = []
    if drainer is not None:
        applied = await drainer.drain_ids(staged_ids)
    is_new = any(
        a.operation == "insert" and a.is_new is True for a in applied
    )

    # --- Phase D: derived work ----------------------------------------
    contradictions = await detect_contradictions(triple, stores.triples)
    confidence = triple["confidence"]
    if contradictions and is_new:
        confidence = apply_penalty(confidence, contradictions)
        if drainer is not None:
            async with stores.pg_pool.acquire() as conn:
                async with conn.transaction():
                    upd_id = await stores.outbox.stage(
                        conn,
                        operation="update_confidence",
                        triple_hash=triple_hash,
                        subject=triple["subject"],
                        predicate=triple["predicate"],
                        object_=triple["object"],
                        confidence=confidence,
                        graph=context.graph,
                    )
                    # Update provenance row's confidence to the penalty value.
                    await stores.provenance.insert(
                        triple_hash,
                        triple["subject"],
                        triple["predicate"],
                        triple["object"],
                        context.source_url,
                        context.source_type,
                        context.extractor,
                        confidence,
                        {},
                        triple.get("valid_from"),
                        triple.get("valid_until"),
                        context.chunk_id,
                        conn=conn,
                    )
            await drainer.drain_ids([upd_id])

    combined = await combine_evidence(triple_hash, stores.provenance)
    if combined != confidence and drainer is not None:
        async with stores.pg_pool.acquire() as conn:
            async with conn.transaction():
                comb_id = await stores.outbox.stage(
                    conn,
                    operation="update_confidence",
                    triple_hash=triple_hash,
                    subject=triple["subject"],
                    predicate=triple["predicate"],
                    object_=triple["object"],
                    confidence=combined,
                    graph=context.graph,
                )
        await drainer.drain_ids([comb_id])
        confidence = combined

    # Inference still uses the existing run_inference helper for now; it will
    # be re-routed through the outbox in a follow-up task in this plan.
    from knowledge_service.ontology.uri import to_entity_uri, to_predicate_uri  # noqa: PLC0415
    normalized = {
        **triple,
        "subject": to_entity_uri(triple["subject"]),
        "predicate": to_predicate_uri(triple["predicate"]),
        "confidence": confidence,
    }
    inferred = await run_inference(normalized, engine, stores, context)

    thesis_breaks = await check_thesis_impact(triple_hash, contradictions, stores)
    return IngestResult(is_new, delta, contradictions, confidence, thesis_breaks, inferred)
```

- [ ] **Step 5: Run the new tests**

Run: `uv run pytest tests/test_pipeline.py::TestIngestTripleOutboxFlow -v`
Expected: both PASS.

- [ ] **Step 6: Re-run legacy `TestIngestTriple` and fix fallout**

Run: `uv run pytest tests/test_pipeline.py::TestIngestTriple -v`

The legacy tests use MagicMock stores without a `pg_pool` / `outbox`. They will fail. Fix by wiring the test's `stores` MagicMock:

```python
# In each TestIngestTriple test, replace:
stores.provenance = prov
# Add:
from knowledge_service.ingestion.outbox import OutboxStore
from knowledge_service.stores.triples import TripleStore  # only if not already imported

# Give stores a minimal outbox + pg_pool fake (reuse _PoolRecording from the
# new TestIngestTripleOutboxFlow tests, OR inline a small one here):
pool = _PoolRecording()
stores.pg_pool = pool
stores.outbox = OutboxStore()
stores.provenance = ProvenanceStore(pool)
# Replace the MagicMock ts with a real TripleStore so the drainer can apply.
ts = _real_triple_store()
stores.triples = ts
# Pass drainer kwarg:
drainer = OutboxDrainer(pool, ts)
result = await ingest_triple(_triple(), stores, ctx, drainer=drainer)
```

For tests that previously asserted specific behavior from `ts.insert.return_value` (e.g., `ts.insert.return_value = ("hash123", True)`), those asserts must be removed or changed to inspect the real `TripleStore` state. If the original test's goal was "new triple, no contradictions" then assert `result.is_new is True` and `len(ts.get_triples(subject=...)) == 1`.

- [ ] **Step 7: Run full pipeline tests**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/knowledge_service/ingestion/pipeline.py tests/test_pipeline.py
git commit -m "refactor: ingest_triple stages writes via outbox + drainer"
```

---

## Task 10: Route inference through the outbox

**Files:**
- Modify: `src/knowledge_service/ingestion/pipeline.py`
- Modify: `tests/test_pipeline_inference.py` (existing)

- [ ] **Step 1: Check existing inference test expectations**

Run: `uv run pytest tests/test_pipeline_inference.py -v`
Expected: all PASS before this task starts.

- [ ] **Step 2: Write the new failing test**

Add to `tests/test_pipeline_inference.py` (copy `_real_triple_store` and `_PoolRecording` helpers from `test_pipeline.py` or import them via a shared module if you prefer — for simplicity, inline them):

```python
class TestInferenceViaOutbox:
    async def test_inferred_triple_goes_through_outbox(self):
        ts = _real_triple_store()
        pool = _PoolRecording()
        prov = ProvenanceStore(pool)
        outbox = OutboxStore()
        drainer = OutboxDrainer(pool, ts)

        class _SingleDerivedEngine:
            def run(self, triple):
                from knowledge_service.reasoning.engine import DerivedTriple  # noqa: PLC0415
                return [
                    DerivedTriple(
                        subject=triple["subject"],
                        predicate="http://knowledge.local/inverse_p",
                        object_=triple["subject"],
                        confidence=0.5,
                        inference_method="inverse",
                        derived_from=[compute_hash(triple)],
                    )
                ]

        class _StubThesis:
            async def find_by_hashes(self, hashes, status=None):
                return []

        stores = MagicMock()
        stores.triples = ts
        stores.provenance = prov
        stores.outbox = outbox
        stores.theses = _StubThesis()
        stores.pg_pool = pool

        ctx = IngestContext(
            source_url="http://t",
            source_type="article",
            extractor="api",
            graph=KS_GRAPH_EXTRACTED,
        )
        result = await ingest_triple(
            _triple(s="cat", p="is_a", o="animal"),
            stores,
            ctx,
            engine=_SingleDerivedEngine(),
            drainer=drainer,
        )
        # 1 base insert + 1 inferred insert in outbox
        ops = [r["operation"] for r in pool.outbox_rows]
        assert ops.count("insert") == 1
        assert ops.count("insert_inferred") == 1
        # Every outbox row applied
        for r in pool.outbox_rows:
            assert r["applied_at"] == "now"
        # Inferred result reported
        assert len(result.inferred_triples) == 1
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/test_pipeline_inference.py::TestInferenceViaOutbox -v`
Expected: FAIL — the existing `run_inference` writes directly to pyoxigraph/PG, bypassing the outbox.

- [ ] **Step 4: Replace `run_inference` with an outbox-based version**

In `src/knowledge_service/ingestion/pipeline.py`, replace the `run_inference` function (currently lines 127–189) with:

```python
async def run_inference(triple: dict, engine, stores, context: IngestContext, drainer=None) -> list[dict]:
    """Run inference engine and persist derived triples via the outbox."""
    if engine is None:
        return []

    from knowledge_service.ontology.namespaces import KS_GRAPH_INFERRED  # noqa: PLC0415

    derived_list = engine.run(triple)
    results = []

    for derived in derived_list:
        derived_hash = _derived_hash(derived)
        async with stores.pg_pool.acquire() as conn:
            async with conn.transaction():
                staged_id = await stores.outbox.stage(
                    conn,
                    operation="insert_inferred",
                    triple_hash=derived_hash,
                    subject=derived.subject,
                    predicate=derived.predicate,
                    object_=derived.object_,
                    confidence=derived.confidence,
                    knowledge_type="inferred",
                    graph=KS_GRAPH_INFERRED,
                    payload={
                        "derived_from": list(derived.derived_from),
                        "inference_method": derived.inference_method,
                    },
                )
                await stores.provenance.insert(
                    derived_hash,
                    derived.subject,
                    derived.predicate,
                    derived.object_,
                    context.source_url,
                    context.source_type,
                    f"inference:{derived.inference_method}",
                    derived.confidence,
                    {"derived_from": list(derived.derived_from)},
                    None,
                    None,
                    None,
                    conn=conn,
                )
        if drainer is not None:
            await drainer.drain_ids([staged_id])
        results.append(derived.to_dict())

    return results


def _derived_hash(derived) -> str:
    """SHA-256 of the derived triple's canonical form, mirroring compute_hash()."""
    from pyoxigraph import Literal, NamedNode, Triple  # noqa: PLC0415
    import hashlib  # noqa: PLC0415
    s = NamedNode(derived.subject)
    p = NamedNode(derived.predicate)
    o = NamedNode(derived.object_) if is_uri(derived.object_) else Literal(derived.object_)
    return hashlib.sha256(str(Triple(s, p, o)).encode()).hexdigest()
```

- [ ] **Step 5: Update the caller in `ingest_triple`**

In `ingest_triple`, change the `run_inference` call from:

```python
    inferred = await run_inference(normalized, engine, stores, context)
```

to:

```python
    inferred = await run_inference(normalized, engine, stores, context, drainer=drainer)
```

- [ ] **Step 6: Run the new test**

Run: `uv run pytest tests/test_pipeline_inference.py::TestInferenceViaOutbox -v`
Expected: PASS.

- [ ] **Step 7: Run full inference test file + pipeline tests**

Run: `uv run pytest tests/test_pipeline_inference.py tests/test_pipeline.py -v`
Expected: all PASS. Fix any test that asserted the pre-refactor behavior of `run_inference` (e.g., checking that the inferred triple appears in pyoxigraph — should still pass because the drainer writes it).

- [ ] **Step 8: Commit**

```bash
git add src/knowledge_service/ingestion/pipeline.py tests/test_pipeline_inference.py
git commit -m "refactor: run_inference persists derived triples through outbox"
```

---

## Task 11: Pass drainer to `ProcessPhase`

**Files:**
- Modify: `src/knowledge_service/ingestion/phases.py`
- Modify: `src/knowledge_service/ingestion/worker.py` (caller)

- [ ] **Step 1: Find the `ProcessPhase` constructor and caller**

Run: `uv run grep -n "ProcessPhase" src/knowledge_service/ingestion/worker.py`
Expected output shows where ProcessPhase is instantiated.

- [ ] **Step 2: Add `drainer` parameter to `ProcessPhase.__init__`**

In `src/knowledge_service/ingestion/phases.py` around line 192:

```python
class ProcessPhase:
    """Phase 3: Resolve entities, expand to triples, ingest."""

    def __init__(
        self,
        stores: Any,
        entity_store: Any | None = None,
        engine: Any | None = None,
        drainer: Any | None = None,
    ):
        self._stores = stores
        self._entity_store = entity_store
        self._engine = engine
        self._drainer = drainer
```

Change the `ingest_triple` call at line 252 from:

```python
                result = await ingest_triple(triple, self._stores, ctx, engine=self._engine)
```

to:

```python
                result = await ingest_triple(
                    triple, self._stores, ctx,
                    engine=self._engine, drainer=self._drainer,
                )
```

- [ ] **Step 3: Update the worker to pass `drainer`**

In `src/knowledge_service/ingestion/worker.py`:

1. `run_ingestion` already accepts `app_state: Any | None = None` (line 119). Add `drainer` retrieval at the top of the function body (right after the signature / existing setup). Near the start of the function, add:

```python
    drainer = getattr(app_state, "outbox_drainer", None) if app_state is not None else None
```

2. Change the `ProcessPhase` construction at line 209 from:

```python
        process = ProcessPhase(stores, entity_store, engine=engine)
```

to:

```python
        process = ProcessPhase(stores, entity_store, engine=engine, drainer=drainer)
```

No change is needed in `api/content.py` — it already passes `app_state=request.app.state` into `run_ingestion`. Confirm by searching:

Run: `uv run grep -n "run_ingestion" src/knowledge_service/api/content.py`
Expected: call sites passing `app_state=request.app.state` (verify, adjust only if missing — the `drainer` attribute added in Task 8 is already on `app.state`).

- [ ] **Step 4: Run the worker tests**

Run: `uv run pytest tests/test_ingestion_worker.py -v`
Expected: PASS. Fix any test that instantiates `ProcessPhase` without `drainer=` by passing `drainer=None` (ingest_triple already tolerates None, albeit in a degraded mode that skips drain).

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/ingestion/phases.py src/knowledge_service/ingestion/worker.py tests/
git commit -m "feat: ProcessPhase threads OutboxDrainer through to ingest_triple"
```

---

## Task 12: Startup drain of any leftover pending rows

**Files:**
- Modify: `src/knowledge_service/main.py`

- [ ] **Step 1: Add the startup drain call**

In `src/knowledge_service/main.py` `lifespan`, immediately after the line `app.state.outbox_drainer = OutboxDrainer(pg_pool, triple_store)` (from Task 8), add:

```python
    drained = await app.state.outbox_drainer.drain_pending()
    if drained:
        logger.info(
            "Startup drain: applied %d pending outbox rows from prior run",
            len(drained),
        )
```

- [ ] **Step 2: Verify lifespan still starts**

Run: `uv run pytest tests/test_api_health.py -v` (or any smoke test that exercises `create_app(use_lifespan=True)` if one exists; otherwise visually review and skip).

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_service/main.py
git commit -m "feat: drain pending outbox rows on application startup"
```

---

## Task 13: Crash-injection integration tests

**Files:**
- Create: `tests/test_ingestion_crash_injection.py`

- [ ] **Step 1: Write the full crash-injection test suite**

Create `tests/test_ingestion_crash_injection.py`:

```python
# tests/test_ingestion_crash_injection.py
"""Crash-injection tests for ProcessPhase outbox coordination."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowledge_service.ingestion.outbox import OutboxStore, OutboxDrainer
from knowledge_service.ingestion.pipeline import (
    IngestContext,
    ingest_triple,
    compute_hash,
)
from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.ontology.namespaces import KS_GRAPH_EXTRACTED
from knowledge_service.ontology.uri import KS, KS_DATA
from knowledge_service.stores.provenance import ProvenanceStore
from knowledge_service.stores.triples import TripleStore


def _triple(s="cat", p="is_a", o="animal", conf=0.8, kt="claim"):
    return {
        "subject": f"{KS_DATA}{s}",
        "predicate": f"{KS}{p}",
        "object": f"{KS_DATA}{o}",
        "confidence": conf,
        "knowledge_type": kt,
        "valid_from": None,
        "valid_until": None,
    }


def _triple_store():
    ts = TripleStore(data_dir=None)
    ontology_dir = (
        Path(__file__).resolve().parent.parent
        / "src" / "knowledge_service" / "ontology"
    )
    bootstrap_ontology(ts, ontology_dir)
    return ts


class _Pool:
    def __init__(self):
        self.provenance_rows: list[tuple] = []
        self.outbox_rows: list[dict] = []
        self.next_id = 1

    def acquire(self):
        return _Acq(self)


class _Acq:
    def __init__(self, pool):
        self.pool = pool

    async def __aenter__(self):
        return _Conn(self.pool)

    async def __aexit__(self, *exc):
        return False


class _Conn:
    def __init__(self, pool):
        self.pool = pool

    def transaction(self):
        return _Txn(self.pool)

    async def execute(self, sql, *args):
        if sql.strip().startswith("INSERT INTO provenance"):
            self.pool.provenance_rows.append(args)
            return "INSERT 0 1"
        if "applied_at" in sql:
            for r in self.pool.outbox_rows:
                if r["id"] == args[0]:
                    r["applied_at"] = "now"
        return "UPDATE 1"

    async def fetchval(self, sql, *args):
        rid = self.pool.next_id
        self.pool.next_id += 1
        row = {
            "id": rid, "triple_hash": args[0], "operation": args[1],
            "subject": args[2], "predicate": args[3], "object": args[4],
            "confidence": args[5], "knowledge_type": args[6],
            "valid_from": args[7], "valid_until": args[8],
            "graph": args[9], "payload": args[10], "applied_at": None,
        }
        self.pool.outbox_rows.append(row)
        return rid

    async def fetch(self, sql, *args):
        if args and isinstance(args[0], list):
            ids = set(args[0])
            return [r for r in self.pool.outbox_rows
                    if r["id"] in ids and r["applied_at"] is None]
        return [r for r in self.pool.outbox_rows if r["applied_at"] is None]


class _Txn:
    """No real rollback; the tests verify call ordering via injection points."""
    def __init__(self, pool):
        self.pool = pool

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # Simulate rollback: if an exception is bubbling out, drop rows that
        # were inserted during this txn. Track via marker.
        return False


class _StubThesis:
    async def find_by_hashes(self, hashes, status=None):
        return []


def _stores(pool, ts):
    stores = MagicMock()
    stores.triples = ts
    stores.provenance = ProvenanceStore(pool)
    stores.outbox = OutboxStore()
    stores.theses = _StubThesis()
    stores.pg_pool = pool
    return stores


@pytest.mark.asyncio
async def test_crash_after_base_commit_before_drain():
    """PG committed but drainer crashed -> re-drain completes convergence."""
    ts = _triple_store()
    pool = _Pool()
    drainer = OutboxDrainer(pool, ts)
    stores = _stores(pool, ts)
    ctx = IngestContext(source_url="http://t", source_type="article",
                        extractor="api", graph=KS_GRAPH_EXTRACTED)

    # Patch drain_ids to raise AFTER PG commit.
    with patch.object(OutboxDrainer, "drain_ids",
                      side_effect=RuntimeError("simulated crash")):
        with pytest.raises(RuntimeError, match="simulated crash"):
            await ingest_triple(_triple(), stores, ctx, drainer=drainer)

    # PG has provenance + pending outbox row; pyoxigraph empty.
    assert len(pool.provenance_rows) == 1
    assert len(pool.outbox_rows) == 1
    assert pool.outbox_rows[0]["applied_at"] is None
    assert ts.get_triples(subject=f"{KS_DATA}cat") == []

    # Recovery: call drain_pending directly.
    applied = await drainer.drain_pending()
    assert len(applied) == 1
    assert pool.outbox_rows[0]["applied_at"] == "now"
    assert len(ts.get_triples(subject=f"{KS_DATA}cat")) == 1


@pytest.mark.asyncio
async def test_crash_mid_drain():
    """drainer raises on 2nd of 3 entries; re-drain converges."""
    ts = _triple_store()
    pool = _Pool()
    store = OutboxStore()
    # Seed 3 inserts directly via stage.
    async with pool.acquire() as conn:
        for i in range(3):
            await store.stage(
                conn, operation="insert", triple_hash=f"h{i}",
                subject=f"{KS_DATA}e{i}", predicate=f"{KS}p", object_=f"{KS_DATA}o",
                confidence=0.5, knowledge_type="claim", graph=KS_GRAPH_EXTRACTED,
            )

    drainer = OutboxDrainer(pool, ts)
    original_insert = ts.insert
    calls = {"n": 0}

    def _maybe_raise(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk error")
        return original_insert(*a, **kw)

    with patch.object(ts, "insert", side_effect=_maybe_raise):
        with pytest.raises(RuntimeError, match="disk error"):
            await drainer.drain_pending()

    # Row 1 applied, rows 2-3 pending.
    applied_flags = [r["applied_at"] is not None for r in pool.outbox_rows]
    assert applied_flags == [True, False, False]

    # Re-drain converges.
    await drainer.drain_pending()
    applied_flags = [r["applied_at"] is not None for r in pool.outbox_rows]
    assert applied_flags == [True, True, True]


@pytest.mark.asyncio
async def test_crash_before_base_commit():
    """Staging raises -> no provenance, no outbox, no pyoxigraph."""
    ts = _triple_store()
    pool = _Pool()
    drainer = OutboxDrainer(pool, ts)
    stores = _stores(pool, ts)
    ctx = IngestContext(source_url="http://t", source_type="article",
                        extractor="api", graph=KS_GRAPH_EXTRACTED)

    # Patch OutboxStore.stage to raise on first call — before provenance.insert.
    with patch.object(OutboxStore, "stage",
                      side_effect=RuntimeError("stage failure")):
        with pytest.raises(RuntimeError, match="stage failure"):
            await ingest_triple(_triple(), stores, ctx, drainer=drainer)

    assert pool.provenance_rows == []
    assert pool.outbox_rows == []
    assert ts.get_triples(subject=f"{KS_DATA}cat") == []


@pytest.mark.asyncio
async def test_idempotent_redrain():
    """Re-running drain on the same rows does not duplicate pyoxigraph writes."""
    ts = _triple_store()
    pool = _Pool()
    store = OutboxStore()
    async with pool.acquire() as conn:
        rid = await store.stage(
            conn, operation="insert", triple_hash="h",
            subject=f"{KS_DATA}x", predicate=f"{KS}p", object_=f"{KS_DATA}y",
            confidence=0.7, knowledge_type="claim", graph=KS_GRAPH_EXTRACTED,
        )
    drainer = OutboxDrainer(pool, ts)
    await drainer.drain_ids([rid])
    # Simulate crash between pyoxigraph apply and UPDATE applied_at.
    pool.outbox_rows[0]["applied_at"] = None

    # Second drain — triple already exists; TripleStore.insert returns is_new=False.
    applied = await drainer.drain_ids([rid])
    assert applied[0].is_new is False
    rows = ts.get_triples(subject=f"{KS_DATA}x")
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_startup_drainer_replays_pending():
    """A fresh drainer instance applies rows left pending by a prior process."""
    ts = _triple_store()
    pool = _Pool()
    store = OutboxStore()
    async with pool.acquire() as conn:
        for i in range(4):
            await store.stage(
                conn, operation="insert", triple_hash=f"h{i}",
                subject=f"{KS_DATA}e{i}", predicate=f"{KS}p", object_=f"{KS_DATA}o",
                confidence=0.5, knowledge_type="claim", graph=KS_GRAPH_EXTRACTED,
            )

    # New drainer, as if a fresh process.
    drainer = OutboxDrainer(pool, ts)
    applied = await drainer.drain_pending()
    assert len(applied) == 4
    for r in pool.outbox_rows:
        assert r["applied_at"] == "now"


@pytest.mark.asyncio
async def test_inferred_triple_crash_and_recovery():
    """Base triple survives a crash during inference-triple drain."""
    from knowledge_service.reasoning.engine import DerivedTriple  # noqa: PLC0415

    ts = _triple_store()
    pool = _Pool()
    drainer = OutboxDrainer(pool, ts)
    stores = _stores(pool, ts)
    ctx = IngestContext(source_url="http://t", source_type="article",
                        extractor="api", graph=KS_GRAPH_EXTRACTED)

    class _Engine:
        def __init__(self):
            self._called = False
        def run(self, triple):
            if self._called:
                return []
            self._called = True
            return [DerivedTriple(
                subject=triple["subject"],
                predicate=f"{KS}inverse_p",
                object_=triple["subject"],
                confidence=0.5,
                inference_method="inverse",
                derived_from=[compute_hash(triple)],
            )]

    engine = _Engine()

    # Patch drain_ids to raise ONLY on inferred-row drains (detected by
    # inspecting the ids -> operation mapping in pool.outbox_rows).
    orig_drain = OutboxDrainer.drain_ids

    async def _maybe_raise(self, ids):
        ops = {r["id"]: r["operation"] for r in pool.outbox_rows}
        if any(ops.get(i) == "insert_inferred" for i in ids):
            raise RuntimeError("inference drain crash")
        return await orig_drain(self, ids)

    with patch.object(OutboxDrainer, "drain_ids", _maybe_raise):
        with pytest.raises(RuntimeError, match="inference drain crash"):
            await ingest_triple(_triple(), stores, ctx, engine=engine, drainer=drainer)

    # Base triple durable.
    assert len(ts.get_triples(subject=f"{KS_DATA}cat")) == 1
    # Inferred outbox row pending.
    pending = [r for r in pool.outbox_rows
               if r["operation"] == "insert_inferred" and r["applied_at"] is None]
    assert len(pending) == 1

    # Recovery: drain remaining.
    applied = await drainer.drain_pending()
    assert any(a.operation == "insert_inferred" for a in applied)
    for r in pool.outbox_rows:
        assert r["applied_at"] == "now"
```

- [ ] **Step 2: Run the full crash-injection suite**

Run: `uv run pytest tests/test_ingestion_crash_injection.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ingestion_crash_injection.py
git commit -m "test: crash-injection coverage for ProcessPhase outbox recovery"
```

---

## Task 14: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the new section**

Insert before the existing "LLM Integration Gotchas" heading in `CLAUDE.md`:

```markdown
## ProcessPhase consistency

`ProcessPhase` writes to both pyoxigraph (triples, RDF-star annotations) and PostgreSQL (provenance, outbox). These stores cannot share a transaction, so coordination is handled via an **outbox pattern**:

- **Commit boundary is PostgreSQL.** Every pyoxigraph write is first staged as a row in `triple_outbox` inside the same PG transaction as its matching `provenance` row. After commit, an `OutboxDrainer` replays the staged rows to pyoxigraph and marks them `applied_at`.
- **Invariant:** Every `provenance` row references a triple that is either already durable in pyoxigraph or present as an unapplied `triple_outbox` row. The inverse (pyoxigraph triple without provenance) is never produced by this layer.
- **Drain happens twice:** synchronously after each PG commit (fast path) and at application startup via `app.state.outbox_drainer.drain_pending()` (recovery path for crashes between commit and drain).
- **All outbox operations are idempotent.** Re-applying an `insert` is a no-op (pyoxigraph deduplicates by content hash). `update_confidence` is idempotent when writing the target value. `insert_inferred` guards RDF-star annotations with SPARQL ASK per `lesson_pyoxigraph_rdfstar`. `retract_inference` re-runs against a hash whose inferences have already been removed and finds nothing to do.
- **Derived work skippable:** contradictions penalty and inference-engine runs happen *after* the base triple is durable in both stores. A crash during derived work leaves the base triple intact; re-ingestion re-runs derived work deterministically because the engine is pure and content-addressed inserts are idempotent.
- **Not the same thing as the stuck-job janitor.** The janitor marks `ingestion_jobs` as failed on process restart; the outbox drainer recovers per-triple store drift. They are independent mechanisms.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document ProcessPhase outbox consistency pattern"
```

---

## Task 15: Final lint + test + PR

**Files:**
- (no code changes)

- [ ] **Step 1: Ruff check**

Run: `uv run ruff check .`
Expected: 0 findings. Fix any flagged issue in-place.

- [ ] **Step 2: Ruff format check**

Run: `uv run ruff format --check .`
Expected: 0 files to reformat. If any, run `uv run ruff format .` and amend the last commit or add a formatting commit.

- [ ] **Step 3: Full test suite**

Run: `uv run pytest tests/ -v --ignore=tests/e2e`
Expected: all pass.

- [ ] **Step 4: Push and open draft PR**

```bash
git push -u origin worktree-processphase-outbox-2pc
gh pr create --title "feat: ProcessPhase 2PC via outbox pattern" --body "$(cat <<'EOF'
## Summary
- Introduces `triple_outbox` table + `OutboxStore` / `OutboxDrainer` to coordinate pyoxigraph writes with PostgreSQL provenance inserts.
- Restructures `ingest_triple` so PG commit is the point of no return; pyoxigraph becomes a replayable projection.
- Adds startup drain to recover from process crashes between PG commit and pyoxigraph apply.

## Design
See `docs/superpowers/specs/2026-04-15-processphase-2pc-outbox-design.md` — Option B (outbox + idempotent replay).

## Invariant
Every `provenance` row references a triple that is either durable in pyoxigraph or present as an unapplied `triple_outbox` row. The reverse (pyoxigraph triple with no provenance) is never produced.

## Test plan
- [x] Unit tests for `OutboxStore.stage` and each `OutboxDrainer` operation (`insert`, `update_confidence`, `insert_inferred`, `retract_inference`)
- [x] Idempotency tests for re-drain of already-applied rows (simulates crash between apply and mark-applied)
- [x] Crash-injection integration tests (`tests/test_ingestion_crash_injection.py`): six scenarios covering base commit, mid-drain, inference drain, startup recovery.
- [x] `uv run ruff check .` clean
- [x] `uv run pytest tests/ --ignore=tests/e2e` green

## Not in scope
- `EmbedPhase` (handled in another PR)
- Stuck-job janitor (handled separately)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

This is a personal repository (`arshadansari27/knowledge-service`), not a `stockopedia/*` repo, so the `--draft` flag is not required by the shared-repo policy. Omit `--draft` unless the user specifically asks for one.

- [ ] **Step 5: Confirm PR URL to user**

After `gh pr create` prints the URL, report it.
