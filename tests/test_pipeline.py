# tests/test_pipeline.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from knowledge_service.ingestion.pipeline import (
    ingest_triple,
    detect_delta,
    apply_penalty,
    combine_evidence,
    compute_hash,
    IngestContext,
)
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


class TestIngestTriple:
    async def test_new_triple_no_contradictions(self):
        ts = MagicMock()
        ts.insert.return_value = ("hash123", True)
        ts.get_triples.return_value = []
        ts.find_contradictions.return_value = []
        ts.find_opposite_contradictions.return_value = []

        prov = AsyncMock()
        prov.insert.return_value = None
        prov.get_by_triple.return_value = [{"confidence": 0.8}]

        stores = MagicMock()
        stores.triples = ts
        stores.provenance = prov
        stores.theses = AsyncMock()
        stores.theses.find_by_hashes.return_value = []

        ctx = IngestContext(
            source_url="http://test.com",
            source_type="article",
            extractor="api",
            graph=KS_GRAPH_EXTRACTED,
        )
        result = await ingest_triple(_triple(), stores, ctx)
        assert result.is_new is True
        assert result.contradictions == []
        assert result.delta is None
