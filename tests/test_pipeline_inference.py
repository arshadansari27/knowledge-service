# tests/test_pipeline_inference.py
import pytest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from knowledge_service.ingestion.pipeline import (
    ingest_triple,
    IngestContext,
    IngestResult,
    compute_hash,
)
from knowledge_service.ingestion.outbox import OutboxStore, OutboxDrainer
from knowledge_service.reasoning.engine import (
    InferenceEngine,
    InverseRule,
    TransitiveRule,
    TypeInheritanceRule,
)
from knowledge_service.stores.triples import TripleStore
from knowledge_service.stores import Stores
from knowledge_service.stores.provenance import ProvenanceStore
from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.ontology.uri import KS, KS_DATA
from knowledge_service.ontology.namespaces import KS_GRAPH_INFERRED, KS_GRAPH_EXTRACTED
from tests.test_pipeline import _real_triple_store, _PoolRecording, _triple


@pytest.fixture
def ts_with_ontology():
    ts = TripleStore(data_dir=None)
    ontology_dir = Path(__file__).resolve().parent.parent / "src" / "knowledge_service" / "ontology"
    bootstrap_ontology(ts, ontology_dir)
    return ts


def _make_stateful_pool():
    """Return (pool, drainer_factory) with a stateful outbox-aware pool."""
    outbox_rows: list[dict] = []
    next_id = [1]

    class _Conn:
        async def execute(self, sql, *args):
            if "applied_at" in sql:
                target = args[0]
                for r in outbox_rows:
                    if r["id"] == target:
                        r["applied_at"] = "now"
            return "OK"

        async def fetchval(self, sql, *args):
            rid = next_id[0]
            next_id[0] += 1
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
            outbox_rows.append(row)
            return rid

        async def fetch(self, sql, *args):
            if args and isinstance(args[0], list):
                ids = set(args[0])
                return [r for r in outbox_rows if r["id"] in ids and r["applied_at"] is None]
            return []

        def transaction(self):
            return _TxnCM()

    class _TxnCM:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    _conn = _Conn()

    @asynccontextmanager
    async def _acquire():
        yield _conn

    mock_pool = MagicMock()
    mock_pool.acquire = _acquire
    return mock_pool


@pytest.fixture
def mock_stores(ts_with_ontology):
    provenance = AsyncMock()
    provenance.get_by_triple.return_value = []
    provenance.insert = AsyncMock()
    theses = AsyncMock()
    theses.find_by_hashes = AsyncMock(return_value=[])

    mock_pool = _make_stateful_pool()

    stores = MagicMock(spec=Stores)
    stores.triples = ts_with_ontology
    stores.provenance = provenance
    stores.theses = theses
    stores.pg_pool = mock_pool
    stores.outbox = OutboxStore()
    return stores


@pytest.fixture
def drainer(mock_stores):
    return OutboxDrainer(mock_stores.pg_pool, mock_stores.triples)


@pytest.fixture
def engine(ts_with_ontology):
    rules = [InverseRule(), TransitiveRule(), TypeInheritanceRule()]
    eng = InferenceEngine(ts_with_ontology, rules, max_depth=3)
    eng.configure()
    return eng


class TestPipelineInference:
    async def test_ingest_triggers_inference(self, mock_stores, engine):
        ctx = IngestContext.from_content("http://example.com", "article", "api")
        triple = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
            "knowledge_type": "claim",
            "valid_from": None,
            "valid_until": None,
        }
        result = await ingest_triple(triple, mock_stores, ctx, engine=engine)
        assert isinstance(result, IngestResult)
        assert len(result.inferred_triples) >= 1
        inv = [t for t in result.inferred_triples if t.get("predicate", "").endswith("part_of")]
        assert len(inv) == 1

    async def test_inferred_in_correct_graph(self, mock_stores, engine, drainer):
        ctx = IngestContext.from_content("http://example.com", "article", "api")
        triple = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
            "knowledge_type": "claim",
            "valid_from": None,
            "valid_until": None,
        }
        await ingest_triple(triple, mock_stores, ctx, engine=engine, drainer=drainer)
        inferred = mock_stores.triples.get_triples(
            subject=f"{KS_DATA}b",
            predicate=f"{KS}part_of",
            graphs=[KS_GRAPH_INFERRED],
        )
        assert len(inferred) >= 1
        assert inferred[0]["graph"] == KS_GRAPH_INFERRED

    async def test_inferred_provenance_written(self, mock_stores, engine):
        ctx = IngestContext.from_content("http://example.com", "article", "api")
        triple = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
            "knowledge_type": "claim",
            "valid_from": None,
            "valid_until": None,
        }
        await ingest_triple(triple, mock_stores, ctx, engine=engine)
        prov_calls = mock_stores.provenance.insert.call_args_list
        inference_calls = [c for c in prov_calls if "inference:" in str(c)]
        assert len(inference_calls) >= 1

    async def test_no_engine_no_inference(self, mock_stores):
        ctx = IngestContext.from_content("http://example.com", "article", "api")
        triple = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
            "knowledge_type": "claim",
            "valid_from": None,
            "valid_until": None,
        }
        result = await ingest_triple(triple, mock_stores, ctx, engine=None)
        assert result.inferred_triples == []


class TestInferredAnnotations:
    async def test_derived_from_annotation(self, mock_stores, engine, drainer):
        ts = mock_stores.triples
        ctx = IngestContext.from_content("http://example.com", "article", "api")
        triple = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
            "knowledge_type": "claim",
            "valid_from": None,
            "valid_until": None,
        }
        await ingest_triple(triple, mock_stores, ctx, engine=engine, drainer=drainer)
        rows = ts.query(f"""
            SELECT ?derived_hash WHERE {{
                GRAPH <{KS_GRAPH_INFERRED}> {{
                    << <{KS_DATA}b> <{KS}part_of> <{KS_DATA}a> >>
                        <{KS}derivedFrom> ?derived_hash .
                }}
            }}
        """)
        assert len(rows) >= 1

    async def test_inference_method_annotation(self, mock_stores, engine, drainer):
        ts = mock_stores.triples
        ctx = IngestContext.from_content("http://example.com", "article", "api")
        triple = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
            "knowledge_type": "claim",
            "valid_from": None,
            "valid_until": None,
        }
        await ingest_triple(triple, mock_stores, ctx, engine=engine, drainer=drainer)
        rows = ts.query(f"""
            SELECT ?method WHERE {{
                GRAPH <{KS_GRAPH_INFERRED}> {{
                    << <{KS_DATA}b> <{KS}part_of> <{KS_DATA}a> >>
                        <{KS}inferenceMethod> ?method .
                }}
            }}
        """)
        assert len(rows) == 1
        method_val = (
            rows[0]["method"].value
            if hasattr(rows[0]["method"], "value")
            else str(rows[0]["method"])
        )
        assert method_val == "inverse"


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
                        predicate="http://knowledge.local/schema/inverse_p",
                        object_=triple["subject"],
                        confidence=0.5,
                        inference_method="inverse",
                        derived_from=[compute_hash(triple)],
                        depth=0,
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
        # Base insert + inferred insert_inferred
        ops = [r["operation"] for r in pool.outbox_rows]
        assert ops.count("insert") == 1
        assert ops.count("insert_inferred") == 1
        # Every outbox row applied
        for r in pool.outbox_rows:
            assert r["applied_at"] == "now"
        # Inferred result reported
        assert len(result.inferred_triples) == 1


class TestRetraction:
    async def test_retraction_removes_annotations(self, mock_stores, engine, drainer):
        """Retracting a stale inferred triple must remove its RDF-star annotation quads.

        Steps:
        1. Ingest A contains B  → engine derives B part_of A (inferred)
        2. Verify the inferred triple exists in ks:graph/inferred
        3. Ingest A contains C (same subject+predicate, different object = delta)
        4. Verify the OLD inferred triple (B part_of A) is gone
        5. Verify no orphaned annotation quads remain for the old inferred triple
        """
        from pyoxigraph import NamedNode as NN

        ts = mock_stores.triples
        ctx = IngestContext.from_content("http://example.com", "article", "api")

        triple_ab = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
            "knowledge_type": "claim",
            "valid_from": None,
            "valid_until": None,
        }
        await ingest_triple(triple_ab, mock_stores, ctx, engine=engine, drainer=drainer)

        # Step 2: Verify inferred triple B part_of A exists
        inferred_before = ts.get_triples(
            subject=f"{KS_DATA}b",
            predicate=f"{KS}part_of",
            graphs=[KS_GRAPH_INFERRED],
        )
        assert len(inferred_before) >= 1, "Expected B part_of A to be inferred"

        # Collect annotation quads before retraction
        reifies_nn = NN("http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies")
        graph_node = NN(KS_GRAPH_INFERRED)
        s_nn = NN(f"{KS_DATA}b")
        p_nn = NN(f"{KS}part_of")
        o_nn = NN(f"{KS_DATA}a")

        annotation_quads_before = []
        for reif_quad in ts.store.quads_for_pattern(None, reifies_nn, None, graph_node):
            reified = reif_quad.object
            if (
                hasattr(reified, "subject")
                and reified.subject == s_nn
                and reified.predicate == p_nn
                and reified.object == o_nn
            ):
                bnode = reif_quad.subject
                annotation_quads_before.extend(
                    list(ts.store.quads_for_pattern(bnode, None, None, graph_node))
                )
        assert len(annotation_quads_before) >= 1, "Expected annotation quads before retraction"

        # Step 3: Ingest A contains C (delta — replaces B)
        triple_ac = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}c",
            "confidence": 0.9,
            "knowledge_type": "claim",
            "valid_from": None,
            "valid_until": None,
        }
        await ingest_triple(triple_ac, mock_stores, ctx, engine=engine, drainer=drainer)

        # Step 4: Old inferred triple (B part_of A) should be gone
        inferred_after = ts.get_triples(
            subject=f"{KS_DATA}b",
            predicate=f"{KS}part_of",
            graphs=[KS_GRAPH_INFERRED],
        )
        assert len(inferred_after) == 0, (
            "Old inferred triple B part_of A should have been retracted"
        )

        # Step 5: No orphaned annotation quads for the old inferred triple
        orphaned_quads = []
        for reif_quad in ts.store.quads_for_pattern(None, reifies_nn, None, graph_node):
            reified = reif_quad.object
            if (
                hasattr(reified, "subject")
                and reified.subject == s_nn
                and reified.predicate == p_nn
                and reified.object == o_nn
            ):
                bnode = reif_quad.subject
                orphaned_quads.extend(
                    list(ts.store.quads_for_pattern(bnode, None, None, graph_node))
                )
        assert len(orphaned_quads) == 0, (
            f"Found {len(orphaned_quads)} orphaned annotation quads after retraction"
        )
