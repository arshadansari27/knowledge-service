"""Integration tests for GET /api/knowledge/contradictions.

All external dependencies (PostgreSQL, pyoxigraph KnowledgeStore) are mocked —
no real services are required.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from knowledge_service.main import create_app
from tests.conftest import make_test_session_cookie


# ---------------------------------------------------------------------------
# Helpers / fake RDF types
# ---------------------------------------------------------------------------


class _FakeNamedNode:
    """Minimal stand-in for pyoxigraph.NamedNode."""

    def __init__(self, iri: str) -> None:
        self.value = iri

    def __str__(self) -> str:
        return f"<{self.value}>"


class _FakeLiteral:
    """Minimal stand-in for pyoxigraph.Literal."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return f'"{self.value}"'


# Sample URIs
_SUBJECT = "http://example.com/subject"
_PREDICATE = "http://example.com/predicate"
_OBJECT_A = "http://example.com/objectA"
_OBJECT_B = "http://example.com/objectB"
_CONF_A = "0.8"
_CONF_B = "0.6"

# A single contradiction candidate row as returned by KnowledgeStore.query
_CONTRADICTION_ROW = {
    "s": _FakeNamedNode(_SUBJECT),
    "p": _FakeNamedNode(_PREDICATE),
    "o1": _FakeNamedNode(_OBJECT_A),
    "o2": _FakeNamedNode(_OBJECT_B),
    "conf1": _FakeLiteral(_CONF_A),
    "conf2": _FakeLiteral(_CONF_B),
}

_SAMPLE_PROVENANCE_ROW = {
    "triple_hash": "abc123",
    "subject": _SUBJECT,
    "predicate": _PREDICATE,
    "object": _OBJECT_A,
    "source_url": "https://example.com/article",
    "source_type": "article",
    "extractor": "manual",
    "confidence": 0.8,
    "ingested_at": "2024-01-01T00:00:00+00:00",
    "valid_from": None,
    "valid_until": None,
    "metadata": "{}",
}


def _make_knowledge_store_mock(rows: list[dict] | None = None) -> MagicMock:
    if rows is None:
        rows = [_CONTRADICTION_ROW]
    mock_ks = MagicMock()
    # The endpoint makes two query() calls: first for same-predicate contradictions,
    # second for opposite-predicate contradictions. Return rows for the first,
    # empty for subsequent (opposite predicates query).
    call_count = {"n": 0}

    def _query_side_effect(sparql):
        call_count["n"] += 1
        if call_count["n"] % 2 == 1:  # odd calls = same-predicate query
            return rows
        return []  # even calls = opposite-predicate query

    mock_ks.query.side_effect = _query_side_effect
    return mock_ks


def _make_pg_pool_mock(provenance_rows: list[dict] | None = None) -> MagicMock:
    if provenance_rows is None:
        provenance_rows = [_SAMPLE_PROVENANCE_ROW]

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = provenance_rows

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    mock_pool = MagicMock()
    mock_pool.acquire = _acquire
    return mock_pool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """Test client with one contradiction candidate and one provenance row."""
    app = create_app(use_lifespan=False)
    app.state.knowledge_store = _make_knowledge_store_mock()
    app.state.pg_pool = _make_pg_pool_mock()

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        cookies={"ks_session": make_test_session_cookie()},
    ) as c:
        yield c


@pytest.fixture
async def empty_client():
    """Test client with no contradiction candidates."""
    app = create_app(use_lifespan=False)
    app.state.knowledge_store = _make_knowledge_store_mock(rows=[])
    app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        cookies={"ks_session": make_test_session_cookie()},
    ) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests: basic response structure
# ---------------------------------------------------------------------------


class TestGetContradictionsBasic:
    async def test_returns_200(self, client):
        response = await client.get("/api/knowledge/contradictions")
        assert response.status_code == 200

    async def test_response_is_list(self, client):
        response = await client.get("/api/knowledge/contradictions")
        assert isinstance(response.json(), list)

    async def test_empty_when_no_contradictions(self, empty_client):
        response = await empty_client.get("/api/knowledge/contradictions")
        assert response.status_code == 200
        assert response.json() == []

    async def test_one_contradiction_returned(self, client):
        response = await client.get("/api/knowledge/contradictions")
        data = response.json()
        assert len(data) == 1

    async def test_result_has_required_fields(self, client):
        response = await client.get("/api/knowledge/contradictions")
        item = response.json()[0]
        assert "claim_a" in item
        assert "claim_b" in item
        assert "contradiction_probability" in item
        assert "provenance_a" in item
        assert "provenance_b" in item


# ---------------------------------------------------------------------------
# Tests: claim_a / claim_b contents
# ---------------------------------------------------------------------------


class TestGetContradictionsClaimFields:
    async def test_claim_a_has_subject(self, client):
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        assert item["claim_a"]["subject"] == _SUBJECT

    async def test_claim_a_has_predicate(self, client):
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        assert item["claim_a"]["predicate"] == _PREDICATE

    async def test_claim_a_has_object(self, client):
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        assert item["claim_a"]["object"] == _OBJECT_A

    async def test_claim_a_has_confidence(self, client):
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        assert item["claim_a"]["confidence"] == pytest.approx(0.8, abs=1e-6)

    async def test_claim_b_has_subject(self, client):
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        assert item["claim_b"]["subject"] == _SUBJECT

    async def test_claim_b_has_predicate(self, client):
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        assert item["claim_b"]["predicate"] == _PREDICATE

    async def test_claim_b_has_object(self, client):
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        assert item["claim_b"]["object"] == _OBJECT_B

    async def test_claim_b_has_confidence(self, client):
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        assert item["claim_b"]["confidence"] == pytest.approx(0.6, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: contradiction probability
# ---------------------------------------------------------------------------


class TestGetContradictionsProbability:
    async def test_probability_is_product_of_confidences(self, client):
        """contradiction_probability == conf1 * conf2."""
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        expected = 0.8 * 0.6
        assert item["contradiction_probability"] == pytest.approx(expected, abs=1e-6)

    async def test_probability_is_float(self, client):
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        assert isinstance(item["contradiction_probability"], float)

    async def test_min_confidence_filters_low_probability(self):
        """Pairs below min_confidence threshold are excluded."""
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()  # prob = 0.48
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            # 0.8 * 0.6 = 0.48, request min_confidence=0.5 → nothing returned
            response = await c.get("/api/knowledge/contradictions", params={"min_confidence": 0.5})

        assert response.status_code == 200
        assert response.json() == []

    async def test_min_confidence_passes_matching_probability(self):
        """Pairs at or above min_confidence threshold are included."""
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()  # prob = 0.48
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            # 0.48 >= 0.4 → included
            response = await c.get("/api/knowledge/contradictions", params={"min_confidence": 0.4})

        assert response.status_code == 200
        assert len(response.json()) == 1

    async def test_default_min_confidence_includes_all(self):
        """Without min_confidence all pairs are returned."""
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/knowledge/contradictions")

        assert len(response.json()) == 1


# ---------------------------------------------------------------------------
# Tests: provenance enrichment
# ---------------------------------------------------------------------------


class TestGetContradictionsProvenance:
    async def test_provenance_a_is_list(self, client):
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        assert isinstance(item["provenance_a"], list)

    async def test_provenance_b_is_list(self, client):
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        assert isinstance(item["provenance_b"], list)

    async def test_provenance_a_has_source_url(self, client):
        item = (await client.get("/api/knowledge/contradictions")).json()[0]
        assert item["provenance_a"][0]["source_url"] == "https://example.com/article"

    async def test_empty_provenance_when_none_exists(self):
        """When no provenance rows exist, both provenance lists are empty."""
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/knowledge/contradictions")

        item = response.json()[0]
        assert item["provenance_a"] == []
        assert item["provenance_b"] == []

    async def test_multiple_provenance_rows_included(self):
        prov_rows = [
            {**_SAMPLE_PROVENANCE_ROW, "source_url": "https://example.com/a"},
            {**_SAMPLE_PROVENANCE_ROW, "source_url": "https://example.com/b"},
        ]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=prov_rows)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/knowledge/contradictions")

        item = response.json()[0]
        assert len(item["provenance_a"]) == 2
        assert len(item["provenance_b"]) == 2


# ---------------------------------------------------------------------------
# Tests: SPARQL query construction
# ---------------------------------------------------------------------------


class TestGetContradictionsSparqlQuery:
    async def test_knowledge_store_query_is_called(self):
        app = create_app(use_lifespan=False)
        mock_ks = _make_knowledge_store_mock(rows=[])
        app.state.knowledge_store = mock_ks
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.get("/api/knowledge/contradictions")

        # Two queries: same-predicate contradictions + opposite-predicate contradictions
        assert mock_ks.query.call_count == 2

    async def test_sparql_contains_confidence_predicate(self):
        from knowledge_service.ontology.namespaces import KS_CONFIDENCE

        app = create_app(use_lifespan=False)
        mock_ks = _make_knowledge_store_mock(rows=[])
        app.state.knowledge_store = mock_ks
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.get("/api/knowledge/contradictions")

        sparql_arg = mock_ks.query.call_args_list[0][0][0]
        assert KS_CONFIDENCE.value in sparql_arg

    async def test_sparql_contains_filter_inequality(self):
        app = create_app(use_lifespan=False)
        mock_ks = _make_knowledge_store_mock(rows=[])
        app.state.knowledge_store = mock_ks
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.get("/api/knowledge/contradictions")

        sparql_arg = mock_ks.query.call_args_list[0][0][0]
        # FILTER should exclude duplicates by requiring o1 != o2
        assert "?o1 != ?o2" in sparql_arg


# ---------------------------------------------------------------------------
# Tests: multiple contradictions
# ---------------------------------------------------------------------------


class TestGetContradictionsMultiple:
    async def test_multiple_contradictions_returned(self):
        rows = [
            {
                "s": _FakeNamedNode(_SUBJECT),
                "p": _FakeNamedNode(_PREDICATE),
                "o1": _FakeNamedNode("http://example.com/objA"),
                "o2": _FakeNamedNode("http://example.com/objB"),
                "conf1": _FakeLiteral("0.9"),
                "conf2": _FakeLiteral("0.7"),
            },
            {
                "s": _FakeNamedNode("http://example.com/other-subject"),
                "p": _FakeNamedNode(_PREDICATE),
                "o1": _FakeNamedNode("http://example.com/objC"),
                "o2": _FakeNamedNode("http://example.com/objD"),
                "conf1": _FakeLiteral("0.5"),
                "conf2": _FakeLiteral("0.5"),
            },
        ]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock(rows=rows)
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/knowledge/contradictions")

        data = response.json()
        assert len(data) == 2
        # First pair: 0.9 * 0.7 = 0.63
        assert data[0]["contradiction_probability"] == pytest.approx(0.63, abs=1e-6)
        # Second pair: 0.5 * 0.5 = 0.25
        assert data[1]["contradiction_probability"] == pytest.approx(0.25, abs=1e-6)

    async def test_min_confidence_filters_some_results(self):
        """With multiple contradictions, only those above min_confidence are returned."""
        rows = [
            {
                "s": _FakeNamedNode(_SUBJECT),
                "p": _FakeNamedNode(_PREDICATE),
                "o1": _FakeNamedNode("http://example.com/objA"),
                "o2": _FakeNamedNode("http://example.com/objB"),
                "conf1": _FakeLiteral("0.9"),
                "conf2": _FakeLiteral("0.7"),  # prob = 0.63
            },
            {
                "s": _FakeNamedNode("http://example.com/other-subject"),
                "p": _FakeNamedNode(_PREDICATE),
                "o1": _FakeNamedNode("http://example.com/objC"),
                "o2": _FakeNamedNode("http://example.com/objD"),
                "conf1": _FakeLiteral("0.5"),
                "conf2": _FakeLiteral("0.5"),  # prob = 0.25
            },
        ]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock(rows=rows)
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            # threshold = 0.5: first passes (0.63), second filtered out (0.25)
            response = await c.get("/api/knowledge/contradictions", params={"min_confidence": 0.5})

        data = response.json()
        assert len(data) == 1
        assert data[0]["contradiction_probability"] == pytest.approx(0.63, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: non-numeric confidence values (error handling)
# ---------------------------------------------------------------------------


class TestGetContradictionsNonNumericConfidence:
    async def test_returns_200_with_non_numeric_confidence(self):
        """Endpoint returns 200 (not crash) when SPARQL returns non-numeric confidence."""
        rows = [
            {
                "s": _FakeNamedNode(_SUBJECT),
                "p": _FakeNamedNode(_PREDICATE),
                "o1": _FakeNamedNode(_OBJECT_A),
                "o2": _FakeNamedNode(_OBJECT_B),
                "conf1": _FakeLiteral("invalid_number"),
                "conf2": _FakeLiteral("0.6"),
            },
        ]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock(rows=rows)
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/knowledge/contradictions")

        assert response.status_code == 200
        # Row is skipped due to invalid conf1, so empty list
        assert response.json() == []

    async def test_skips_row_with_non_numeric_conf2(self):
        """Row with non-numeric conf2 is skipped."""
        rows = [
            {
                "s": _FakeNamedNode(_SUBJECT),
                "p": _FakeNamedNode(_PREDICATE),
                "o1": _FakeNamedNode(_OBJECT_A),
                "o2": _FakeNamedNode(_OBJECT_B),
                "conf1": _FakeLiteral("0.8"),
                "conf2": _FakeLiteral("not_a_number"),
            },
        ]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock(rows=rows)
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/knowledge/contradictions")

        assert response.status_code == 200
        assert response.json() == []

    async def test_mixed_valid_and_invalid_confidence_rows(self):
        """Valid rows are returned, invalid rows are skipped."""
        rows = [
            {
                "s": _FakeNamedNode(_SUBJECT),
                "p": _FakeNamedNode(_PREDICATE),
                "o1": _FakeNamedNode(_OBJECT_A),
                "o2": _FakeNamedNode(_OBJECT_B),
                "conf1": _FakeLiteral("invalid"),
                "conf2": _FakeLiteral("0.6"),
            },
            {
                "s": _FakeNamedNode("http://example.com/subject2"),
                "p": _FakeNamedNode(_PREDICATE),
                "o1": _FakeNamedNode("http://example.com/objC"),
                "o2": _FakeNamedNode("http://example.com/objD"),
                "conf1": _FakeLiteral("0.9"),
                "conf2": _FakeLiteral("0.7"),
            },
        ]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock(rows=rows)
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/knowledge/contradictions")

        assert response.status_code == 200
        data = response.json()
        # Only the second (valid) row is returned
        assert len(data) == 1
        assert data[0]["contradiction_probability"] == pytest.approx(0.63, abs=1e-6)
