# tests/test_outbox.py
from pathlib import Path
from unittest.mock import AsyncMock

from knowledge_service.ingestion.outbox import OutboxDrainer, OutboxStore
from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.ontology.namespaces import KS_GRAPH_EXTRACTED, KS_GRAPH_INFERRED
from knowledge_service.stores.triples import TripleStore


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
        assert args[1] == "abc"  # triple_hash
        assert args[2] == "insert"  # operation

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


def _build_triple_store():
    ts = TripleStore(data_dir=None)
    ontology_dir = Path(__file__).resolve().parent.parent / "src" / "knowledge_service" / "ontology"
    bootstrap_ontology(ts, ontology_dir)
    return ts


class _FakePool:
    """Minimal asyncpg-shaped pool that hands out a single tracked connection."""

    def __init__(self):
        self.rows = []  # staged rows (dict-like)
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
        rows = ts.get_triples(
            subject="http://knowledge.local/data/fluffy", graphs=[KS_GRAPH_INFERRED]
        )
        assert len(rows) == 1

        # ks:derivedFrom annotation is present
        ask = f"""
            ASK {{
                GRAPH <{KS_GRAPH_INFERRED}> {{
                    << <http://knowledge.local/data/fluffy>
                       <http://knowledge.local/is_a>
                       <http://knowledge.local/data/animal> >>
                    <http://knowledge.local/schema/derivedFrom> "h1" .
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
                    <http://knowledge.local/schema/derivedFrom> "h1" .
                }}
            }}
        """)
        count = int(list(rows)[0]["c"].value)
        assert count == 1
