"""Tests for the shared per-triple ingestion pipeline."""

from unittest.mock import AsyncMock, MagicMock

from knowledge_service.api._ingest import _penalize_confidence, process_triple
from knowledge_service.api.content import _dedup_extracted_items


def _make_ks(is_new=True, contradictions=None):
    mock = MagicMock()
    mock.insert_triple.return_value = ("abc123", is_new)
    mock.find_contradictions.return_value = contradictions or []
    mock.find_opposite_predicate_contradictions = MagicMock(return_value=[])
    mock.update_confidence = MagicMock()
    return mock


def _make_provenance_store(prov_rows=None):
    mock = AsyncMock()
    mock.insert.return_value = None
    mock.get_by_triple.return_value = prov_rows or []
    return mock


def _make_engine():
    m = MagicMock()
    m.combine_evidence.return_value = 0.9
    return m


async def test_process_triple_returns_true_for_new():
    ks = _make_ks(is_new=True)
    is_new, contras = await process_triple(
        {
            "subject": "http://s",
            "predicate": "http://p",
            "object": "v",
            "confidence": 0.8,
            "knowledge_type": "Claim",
            "valid_from": None,
            "valid_until": None,
        },
        ks,
        _make_provenance_store(),
        _make_engine(),
        "http://src",
        "article",
        "manual",
    )
    assert is_new is True
    assert contras == []


async def test_process_triple_returns_false_for_existing():
    ks = _make_ks(is_new=False)
    is_new, _ = await process_triple(
        {
            "subject": "http://s",
            "predicate": "http://p",
            "object": "v",
            "confidence": 0.8,
            "knowledge_type": "Claim",
            "valid_from": None,
            "valid_until": None,
        },
        ks,
        _make_provenance_store(),
        _make_engine(),
        "http://src",
        "article",
        "manual",
    )
    assert is_new is False


async def test_process_triple_returns_contradictions():
    ks = _make_ks(is_new=True, contradictions=[{"object": "other", "confidence": 0.6}])
    _, contras = await process_triple(
        {
            "subject": "http://s",
            "predicate": "http://p",
            "object": "v",
            "confidence": 0.8,
            "knowledge_type": "Claim",
            "valid_from": None,
            "valid_until": None,
        },
        ks,
        _make_provenance_store(),
        _make_engine(),
        "http://src",
        "article",
        "manual",
    )
    assert len(contras) == 1
    assert contras[0]["existing_object"] == "other"


async def test_process_triple_combines_evidence_for_multiple_sources():
    ks = _make_ks(is_new=True)
    prov_store = _make_provenance_store(
        prov_rows=[
            {"confidence": 0.7, "triple_hash": "abc123", "source_url": "http://a"},
            {"confidence": 0.6, "triple_hash": "abc123", "source_url": "http://b"},
        ]
    )
    engine = _make_engine()
    await process_triple(
        {
            "subject": "http://s",
            "predicate": "http://p",
            "object": "v",
            "confidence": 0.8,
            "knowledge_type": "Claim",
            "valid_from": None,
            "valid_until": None,
        },
        ks,
        prov_store,
        engine,
        "http://src",
        "article",
        "manual",
    )
    engine.combine_evidence.assert_called_once()
    ks.update_confidence.assert_called_once()


async def test_process_triple_detects_opposite_predicate_contradiction():
    ks = _make_ks(is_new=True)
    ks.find_opposite_predicate_contradictions = MagicMock(
        return_value=[{"predicate_in_store": "http://ks/decreases", "confidence": 0.7}]
    )
    prov_store = _make_provenance_store()
    engine = _make_engine()
    t = {
        "subject": "http://s",
        "predicate": "http://ks/increases",
        "object": "http://o",
        "confidence": 0.8,
        "knowledge_type": "Claim",
        "valid_from": None,
        "valid_until": None,
    }
    _, contras = await process_triple(t, ks, prov_store, engine, "http://src", "article", "manual")
    # Should have detected the opposite-predicate contradiction
    opp = [c for c in contras if "opposite_predicate_in_store" in c]
    assert len(opp) == 1
    assert opp[0]["opposite_predicate_in_store"] == "http://ks/decreases"


# ---------------------------------------------------------------------------
# Tests: _penalize_confidence
# ---------------------------------------------------------------------------


class TestPenalizeConfidence:
    def test_no_existing_confs_returns_original(self):
        assert _penalize_confidence(0.8, []) == 0.8

    def test_all_none_confs_returns_original(self):
        assert _penalize_confidence(0.8, [None, None]) == 0.8

    def test_penalizes_proportionally(self):
        # existing=0.9, penalty_factor=0.5 → penalty=0.45 → 0.8 * 0.55 = 0.44
        result = _penalize_confidence(0.8, [0.9])
        assert result == 0.44

    def test_max_existing_used(self):
        # max(0.6, 0.9) = 0.9
        result = _penalize_confidence(0.8, [0.6, 0.9])
        assert result == _penalize_confidence(0.8, [0.9])

    def test_zero_existing_no_penalty(self):
        assert _penalize_confidence(0.8, [0.0]) == 0.8

    def test_full_confidence_existing(self):
        # existing=1.0 → penalty=0.5 → 0.8 * 0.5 = 0.4
        assert _penalize_confidence(0.8, [1.0]) == 0.4


# ---------------------------------------------------------------------------
# Tests: _dedup_extracted_items
# ---------------------------------------------------------------------------


class TestDedupExtractedItems:
    def _make_claim(self, subject, predicate, obj, confidence=0.7):
        from knowledge_service.models import ClaimInput

        return ClaimInput(
            subject=subject,
            predicate=predicate,
            object=obj,
            object_type="entity",
            confidence=confidence,
        )

    def _make_entity(self, uri, label, confidence=0.9):
        from knowledge_service.models import EntityInput

        return EntityInput(
            uri=uri,
            rdf_type="schema:Thing",
            label=label,
            properties={},
            confidence=confidence,
        )

    def test_no_duplicates_preserved(self):
        items = [self._make_claim("a", "p", "b"), self._make_claim("c", "p", "d")]
        cids = ["c1", "c2"]
        result, result_cids = _dedup_extracted_items(items, cids)
        assert len(result) == 2

    def test_exact_duplicate_keeps_higher_confidence(self):
        low = self._make_claim("a", "p", "b", confidence=0.5)
        high = self._make_claim("a", "p", "b", confidence=0.9)
        result, result_cids = _dedup_extracted_items([low, high], ["c1", "c2"])
        assert len(result) == 1
        assert result[0].confidence == 0.9
        assert result_cids[0] == "c2"

    def test_case_insensitive_dedup(self):
        a = self._make_claim("Dopamine", "Increases", "Alertness", confidence=0.7)
        b = self._make_claim("dopamine", "increases", "alertness", confidence=0.8)
        result, _ = _dedup_extracted_items([a, b], ["c1", "c2"])
        assert len(result) == 1
        assert result[0].confidence == 0.8

    def test_entity_dedup_by_uri(self):
        a = self._make_entity("dopamine", "dopamine", confidence=0.7)
        b = self._make_entity("dopamine", "dopamine", confidence=0.9)
        result, _ = _dedup_extracted_items([a, b], ["c1", "c2"])
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_different_types_not_deduped(self):
        claim = self._make_claim("a", "p", "b")
        entity = self._make_entity("a", "a")
        result, _ = _dedup_extracted_items([claim, entity], ["c1", "c2"])
        assert len(result) == 2
