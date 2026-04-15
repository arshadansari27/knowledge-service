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
    ontology_dir = Path(__file__).resolve().parent.parent / "src" / "knowledge_service" / "ontology"
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
        return _Txn()

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


class _Txn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
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


async def test_crash_after_base_commit_before_drain():
    ts = _triple_store()
    pool = _Pool()
    drainer = OutboxDrainer(pool, ts)
    stores = _stores(pool, ts)
    ctx = IngestContext(
        source_url="http://t", source_type="article", extractor="api", graph=KS_GRAPH_EXTRACTED
    )

    with patch.object(OutboxDrainer, "drain_ids", side_effect=RuntimeError("simulated crash")):
        with pytest.raises(RuntimeError, match="simulated crash"):
            await ingest_triple(_triple(), stores, ctx, drainer=drainer)

    assert len(pool.provenance_rows) == 1
    assert len(pool.outbox_rows) == 1
    assert pool.outbox_rows[0]["applied_at"] is None
    assert ts.get_triples(subject=f"{KS_DATA}cat") == []

    applied = await drainer.drain_pending()
    assert len(applied) == 1
    assert pool.outbox_rows[0]["applied_at"] == "now"
    assert len(ts.get_triples(subject=f"{KS_DATA}cat")) == 1


async def test_crash_mid_drain():
    ts = _triple_store()
    pool = _Pool()
    store = OutboxStore()
    async with pool.acquire() as conn:
        for i in range(3):
            await store.stage(
                conn,
                operation="insert",
                triple_hash=f"h{i}",
                subject=f"{KS_DATA}e{i}",
                predicate=f"{KS}p",
                object_=f"{KS_DATA}o",
                confidence=0.5,
                knowledge_type="claim",
                graph=KS_GRAPH_EXTRACTED,
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

    applied_flags = [r["applied_at"] is not None for r in pool.outbox_rows]
    assert applied_flags == [True, False, False]

    await drainer.drain_pending()
    applied_flags = [r["applied_at"] is not None for r in pool.outbox_rows]
    assert applied_flags == [True, True, True]


async def test_crash_before_base_commit():
    ts = _triple_store()
    pool = _Pool()
    drainer = OutboxDrainer(pool, ts)
    stores = _stores(pool, ts)
    ctx = IngestContext(
        source_url="http://t", source_type="article", extractor="api", graph=KS_GRAPH_EXTRACTED
    )

    with patch.object(OutboxStore, "stage", side_effect=RuntimeError("stage failure")):
        with pytest.raises(RuntimeError, match="stage failure"):
            await ingest_triple(_triple(), stores, ctx, drainer=drainer)

    assert pool.provenance_rows == []
    assert pool.outbox_rows == []
    assert ts.get_triples(subject=f"{KS_DATA}cat") == []


async def test_idempotent_redrain():
    ts = _triple_store()
    pool = _Pool()
    store = OutboxStore()
    async with pool.acquire() as conn:
        rid = await store.stage(
            conn,
            operation="insert",
            triple_hash="h",
            subject=f"{KS_DATA}x",
            predicate=f"{KS}p",
            object_=f"{KS_DATA}y",
            confidence=0.7,
            knowledge_type="claim",
            graph=KS_GRAPH_EXTRACTED,
        )
    drainer = OutboxDrainer(pool, ts)
    await drainer.drain_ids([rid])
    pool.outbox_rows[0]["applied_at"] = None

    applied = await drainer.drain_ids([rid])
    assert applied[0].is_new is False
    rows = ts.get_triples(subject=f"{KS_DATA}x")
    assert len(rows) == 1


async def test_startup_drainer_replays_pending():
    ts = _triple_store()
    pool = _Pool()
    store = OutboxStore()
    async with pool.acquire() as conn:
        for i in range(4):
            await store.stage(
                conn,
                operation="insert",
                triple_hash=f"h{i}",
                subject=f"{KS_DATA}e{i}",
                predicate=f"{KS}p",
                object_=f"{KS_DATA}o",
                confidence=0.5,
                knowledge_type="claim",
                graph=KS_GRAPH_EXTRACTED,
            )

    drainer = OutboxDrainer(pool, ts)
    applied = await drainer.drain_pending()
    assert len(applied) == 4
    for r in pool.outbox_rows:
        assert r["applied_at"] == "now"


async def test_inferred_triple_crash_and_recovery():
    from knowledge_service.reasoning.engine import DerivedTriple  # noqa: PLC0415

    ts = _triple_store()
    pool = _Pool()
    drainer = OutboxDrainer(pool, ts)
    stores = _stores(pool, ts)
    ctx = IngestContext(
        source_url="http://t", source_type="article", extractor="api", graph=KS_GRAPH_EXTRACTED
    )

    class _Engine:
        def __init__(self):
            self._called = False

        def run(self, triple):
            if self._called:
                return []
            self._called = True
            return [
                DerivedTriple(
                    subject=triple["subject"],
                    predicate=f"{KS}inverse_p",
                    object_=triple["subject"],
                    confidence=0.5,
                    inference_method="inverse",
                    derived_from=[compute_hash(triple)],
                    depth=0,
                )
            ]

    engine = _Engine()
    orig_drain = OutboxDrainer.drain_ids

    async def _maybe_raise(self, ids):
        ops = {r["id"]: r["operation"] for r in pool.outbox_rows}
        if any(ops.get(i) == "insert_inferred" for i in ids):
            raise RuntimeError("inference drain crash")
        return await orig_drain(self, ids)

    with patch.object(OutboxDrainer, "drain_ids", _maybe_raise):
        with pytest.raises(RuntimeError, match="inference drain crash"):
            await ingest_triple(_triple(), stores, ctx, engine=engine, drainer=drainer)

    assert len(ts.get_triples(subject=f"{KS_DATA}cat")) == 1
    pending = [
        r
        for r in pool.outbox_rows
        if r["operation"] == "insert_inferred" and r["applied_at"] is None
    ]
    assert len(pending) == 1

    applied = await drainer.drain_pending()
    assert any(a.operation == "insert_inferred" for a in applied)
    for r in pool.outbox_rows:
        assert r["applied_at"] == "now"
