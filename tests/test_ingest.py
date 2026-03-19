"""Tests for the shared per-triple ingestion pipeline."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from knowledge_service.api._ingest import process_triple


def _make_ks(is_new=True, contradictions=None):
    mock = MagicMock()
    mock.insert_triple.return_value = ("abc123", is_new)
    mock.find_contradictions.return_value = contradictions or []
    mock.find_opposite_predicate_contradictions = MagicMock(return_value=[])
    mock.update_confidence = MagicMock()
    return mock


def _make_pool(prov_rows=None):
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "INSERT 0 1"
    mock_conn.fetch.return_value = prov_rows or []

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    mock_pool = MagicMock()
    mock_pool.acquire = _acquire
    return mock_pool


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
        _make_pool(),
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
        _make_pool(),
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
        _make_pool(),
        _make_engine(),
        "http://src",
        "article",
        "manual",
    )
    assert len(contras) == 1
    assert contras[0]["existing_object"] == "other"


async def test_process_triple_combines_evidence_for_multiple_sources():
    ks = _make_ks(is_new=True)
    pool = _make_pool(
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
        pool,
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
    pool = _make_pool()
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
    _, contras = await process_triple(t, ks, pool, engine, "http://src", "article", "manual")
    # Should have detected the opposite-predicate contradiction
    opp = [c for c in contras if "opposite_predicate_in_store" in c]
    assert len(opp) == 1
    assert opp[0]["opposite_predicate_in_store"] == "http://ks/decreases"
