# tests/test_pipeline.py
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from knowledge_service.ingestion.pipeline import (
    ingest_triple,
    detect_delta,
    apply_penalty,
    combine_evidence,
    compute_hash,
    IngestContext,
)
from knowledge_service.ingestion.outbox import OutboxStore, OutboxDrainer
from knowledge_service.stores.provenance import ProvenanceStore
from knowledge_service.stores.triples import TripleStore
from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.ontology.uri import KS, KS_DATA
from knowledge_service.ontology.namespaces import KS_GRAPH_EXTRACTED


def _triple(s="a", p="b", o="c", conf=0.8, kt="claim"):
    return {
        "subject": f"{KS_DATA}{s}",
        "predicate": f"{KS}{p}",
        "object": o,
        "confidence": conf,
        "knowledge_type": kt,
        "valid_from": None,
        "valid_until": None,
    }


class TestComputeHash:
    def test_deterministic(self):
        t = _triple()
        assert compute_hash(t) == compute_hash(t)

    def test_different_triples_different_hash(self):
        assert compute_hash(_triple(o="x")) != compute_hash(_triple(o="y"))


class TestApplyPenalty:
    def test_no_contradictions(self):
        assert apply_penalty(0.8, []) == pytest.approx(0.8)

    def test_with_contradiction(self):
        contras = [{"existing_confidence": 0.9}]
        result = apply_penalty(0.8, contras)
        # 0.8 * (1 - 0.9 * 0.5) = 0.8 * 0.55 = 0.44
        assert result == pytest.approx(0.44)


class TestCombineEvidence:
    async def test_single_source(self):
        prov = AsyncMock()
        prov.get_by_triple.return_value = [{"confidence": 0.8}]
        result = await combine_evidence("hash123", prov)
        assert result == pytest.approx(0.8)

    async def test_multiple_sources(self):
        prov = AsyncMock()
        prov.get_by_triple.return_value = [
            {"confidence": 0.7},
            {"confidence": 0.8},
        ]
        result = await combine_evidence("hash123", prov)
        # noisy_or([0.7, 0.8]) = 0.94
        assert result == pytest.approx(0.94)


class TestDetectDelta:
    async def test_no_prior(self):
        ts = MagicMock()
        ts.get_triples.return_value = []
        delta = await detect_delta(_triple(), ts)
        assert delta is None

    async def test_same_value(self):
        ts = MagicMock()
        ts.get_triples.return_value = [{"object": "c", "confidence": 0.7}]
        delta = await detect_delta(_triple(o="c"), ts)
        assert delta is None

    async def test_different_value(self):
        ts = MagicMock()
        ts.get_triples.return_value = [{"object": "old_value", "confidence": 0.7}]
        delta = await detect_delta(_triple(o="new_value"), ts)
        assert delta is not None
        assert delta["prior_value"] == "old_value"
        assert delta["current_value"] == "new_value"


def _real_triple_store():
    ts = TripleStore(data_dir=None)
    ontology_dir = Path(__file__).resolve().parent.parent / "src" / "knowledge_service" / "ontology"
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
        # Outbox drain: fetch by id list
        if args and isinstance(args[0], list):
            ids = set(args[0])
            return [r for r in self.pool.outbox_rows if r["id"] in ids and r["applied_at"] is None]
        # Provenance lookup: SELECT * FROM provenance WHERE triple_hash = $1
        if args and "provenance" in sql and "triple_hash" in sql:
            target_hash = args[0]
            results = []
            for row_args in self.pool.provenance_rows:
                # row_args is the tuple of params:
                # (triple_hash, subject, predicate, object, source_url,
                #  source_type, extractor, confidence, metadata_json,
                #  valid_from, valid_until, chunk_id)
                if row_args[0] == target_hash:
                    results.append({"confidence": row_args[7]})
            return results
        return [r for r in self.pool.outbox_rows if r["applied_at"] is None]


class _RecTxn:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _StubThesis:
    async def find_by_hashes(self, hashes, status=None):
        return []


class TestIngestTriple:
    async def test_new_triple_no_contradictions(self):
        ts = _real_triple_store()
        pool = _PoolRecording()
        prov = ProvenanceStore(pool)
        outbox = OutboxStore()
        drainer = OutboxDrainer(pool, ts)

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
        result = await ingest_triple(_triple(), stores, ctx, drainer=drainer)
        assert result.is_new is True
        assert result.contradictions == []
        assert result.delta is None


class TestIngestTripleOutboxFlow:
    async def test_ingest_writes_outbox_and_provenance_in_one_txn(self):
        ts = _real_triple_store()
        pool = _PoolRecording()
        prov = ProvenanceStore(pool)
        outbox = OutboxStore()
        drainer = OutboxDrainer(pool, ts)

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
        assert len(pool.provenance_rows) == 1
        assert len(pool.outbox_rows) == 1
        assert pool.outbox_rows[0]["applied_at"] == "now"
        rows = ts.get_triples(subject=f"{KS_DATA}cat")
        assert len(rows) == 1

    async def test_ingest_rolled_back_txn_leaves_no_state(self):
        ts = _real_triple_store()
        pool = _PoolRecording()
        prov = ProvenanceStore(pool)

        class _BoomOutbox(OutboxStore):
            async def stage(self, *a, **kw):
                raise RuntimeError("boom")

        outbox = _BoomOutbox()
        drainer = OutboxDrainer(pool, ts)

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

        assert pool.provenance_rows == []
        assert pool.outbox_rows == []
        assert ts.get_triples(subject=f"{KS_DATA}dog") == []
