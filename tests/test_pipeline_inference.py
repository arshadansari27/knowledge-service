# tests/test_pipeline_inference.py
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from knowledge_service.ingestion.pipeline import ingest_triple, IngestContext, IngestResult
from knowledge_service.reasoning.engine import (
    InferenceEngine,
    InverseRule,
    TransitiveRule,
    TypeInheritanceRule,
)
from knowledge_service.stores.triples import TripleStore
from knowledge_service.stores import Stores
from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.ontology.uri import KS, KS_DATA
from knowledge_service.ontology.namespaces import KS_GRAPH_INFERRED


@pytest.fixture
def ts_with_ontology():
    ts = TripleStore(data_dir=None)
    ontology_dir = Path(__file__).resolve().parent.parent / "src" / "knowledge_service" / "ontology"
    bootstrap_ontology(ts, ontology_dir)
    return ts


@pytest.fixture
def mock_stores(ts_with_ontology):
    provenance = AsyncMock()
    provenance.get_by_triple.return_value = []
    provenance.insert = AsyncMock()
    theses = AsyncMock()
    theses.find_by_hashes = AsyncMock(return_value=[])
    stores = MagicMock(spec=Stores)
    stores.triples = ts_with_ontology
    stores.provenance = provenance
    stores.theses = theses
    return stores


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

    async def test_inferred_in_correct_graph(self, mock_stores, engine):
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
    async def test_derived_from_annotation(self, mock_stores, engine):
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
        await ingest_triple(triple, mock_stores, ctx, engine=engine)
        rows = ts.query(f"""
            SELECT ?derived_hash WHERE {{
                GRAPH <{KS_GRAPH_INFERRED}> {{
                    << <{KS_DATA}b> <{KS}part_of> <{KS_DATA}a> >>
                        <{KS}derivedFrom> ?derived_hash .
                }}
            }}
        """)
        assert len(rows) >= 1

    async def test_inference_method_annotation(self, mock_stores, engine):
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
        await ingest_triple(triple, mock_stores, ctx, engine=engine)
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
