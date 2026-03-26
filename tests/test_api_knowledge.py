"""Integration tests for GET /api/knowledge/query and POST /api/knowledge/sparql.

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
# Helpers
# ---------------------------------------------------------------------------


# Simulate what KnowledgeStore.query returns for a SELECT with ?s ?p ?o ?conf ?ktype
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


_SAMPLE_SUBJECT = "http://example.com/subject"
_SAMPLE_PREDICATE = "http://example.com/predicate"
_SAMPLE_OBJECT = "hello world"
_SAMPLE_CONFIDENCE = "0.85"
_SAMPLE_KTYPE = "Claim"

_SAMPLE_KS_ROW = {
    "s": _FakeNamedNode(_SAMPLE_SUBJECT),
    "p": _FakeNamedNode(_SAMPLE_PREDICATE),
    "o": _FakeLiteral(_SAMPLE_OBJECT),
    "conf": _FakeLiteral(_SAMPLE_CONFIDENCE),
    "ktype": _FakeLiteral(_SAMPLE_KTYPE),
}

_SAMPLE_PROVENANCE_ROW = {
    "triple_hash": "abc123",
    "subject": _SAMPLE_SUBJECT,
    "predicate": _SAMPLE_PREDICATE,
    "object": _SAMPLE_OBJECT,
    "source_url": "https://example.com/article",
    "source_type": "article",
    "extractor": "manual",
    "confidence": 0.85,
    "ingested_at": "2024-01-01T00:00:00+00:00",
    "valid_from": None,
    "valid_until": None,
    "metadata": "{}",
}


def _make_knowledge_store_mock(rows: list[dict] | None = None):
    """Build a mock KnowledgeStore with configurable query return values."""
    if rows is None:
        rows = [_SAMPLE_KS_ROW]
    mock_ks = MagicMock()
    mock_ks.query.return_value = rows
    return mock_ks


def _make_pg_pool_mock(provenance_rows: list[dict] | None = None):
    """Build a mock asyncpg pool whose .acquire() works as async context manager."""
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
    """Create test client with all external dependencies mocked."""
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
    """Create test client that returns empty results."""
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
# Tests: GET /api/knowledge/query — basic response structure
# ---------------------------------------------------------------------------


class TestGetKnowledgeQueryBasic:
    async def test_returns_200_with_subject(self, client):
        response = await client.get("/api/knowledge/query", params={"subject": _SAMPLE_SUBJECT})
        assert response.status_code == 200

    async def test_response_is_list(self, client):
        response = await client.get("/api/knowledge/query", params={"subject": _SAMPLE_SUBJECT})
        assert isinstance(response.json(), list)

    async def test_result_has_required_fields(self, client):
        response = await client.get("/api/knowledge/query", params={"subject": _SAMPLE_SUBJECT})
        data = response.json()
        assert len(data) == 1
        result = data[0]
        assert "subject" in result
        assert "predicate" in result
        assert "object" in result
        assert "confidence" in result
        assert "knowledge_type" in result
        assert "provenance" in result

    async def test_result_values_match_row(self, client):
        response = await client.get("/api/knowledge/query", params={"subject": _SAMPLE_SUBJECT})
        data = response.json()
        result = data[0]
        assert result["subject"] == _SAMPLE_SUBJECT
        assert result["predicate"] == _SAMPLE_PREDICATE
        assert result["object"] == _SAMPLE_OBJECT
        assert result["confidence"] == pytest.approx(0.85, abs=1e-6)
        assert result["knowledge_type"] == _SAMPLE_KTYPE

    async def test_provenance_list_present(self, client):
        response = await client.get("/api/knowledge/query", params={"subject": _SAMPLE_SUBJECT})
        data = response.json()
        result = data[0]
        assert isinstance(result["provenance"], list)
        assert len(result["provenance"]) == 1

    async def test_provenance_has_source_url(self, client):
        response = await client.get("/api/knowledge/query", params={"subject": _SAMPLE_SUBJECT})
        data = response.json()
        prov = data[0]["provenance"][0]
        assert prov["source_url"] == "https://example.com/article"

    async def test_empty_results_return_empty_list(self, empty_client):
        response = await empty_client.get(
            "/api/knowledge/query", params={"subject": "http://example.com/missing"}
        )
        assert response.status_code == 200
        assert response.json() == []


# ---------------------------------------------------------------------------
# Tests: GET /api/knowledge/query — filter by predicate and object
# ---------------------------------------------------------------------------


class TestGetKnowledgeQueryFilters:
    async def test_filter_by_predicate(self, client):
        response = await client.get("/api/knowledge/query", params={"predicate": _SAMPLE_PREDICATE})
        assert response.status_code == 200

    async def test_filter_by_object_literal(self, client):
        response = await client.get("/api/knowledge/query", params={"object": "hello world"})
        assert response.status_code == 200

    async def test_filter_by_object_uri(self, client):
        response = await client.get(
            "/api/knowledge/query",
            params={"object": "http://example.com/entity"},
        )
        assert response.status_code == 200

    async def test_filter_by_all_three(self, client):
        response = await client.get(
            "/api/knowledge/query",
            params={
                "subject": _SAMPLE_SUBJECT,
                "predicate": _SAMPLE_PREDICATE,
                "object": _SAMPLE_OBJECT,
            },
        )
        assert response.status_code == 200

    async def test_no_params_returns_422(self, client):
        response = await client.get("/api/knowledge/query")
        assert response.status_code == 422

    async def test_sparql_filters_built_for_subject(self):
        """Verify that the SPARQL query passed to KnowledgeStore contains the subject filter."""
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
            await c.get(
                "/api/knowledge/query",
                params={"subject": "http://example.com/thing"},
            )

        mock_ks.query.assert_called_once()
        sparql_arg = mock_ks.query.call_args[0][0]
        assert "http://example.com/thing" in sparql_arg

    async def test_sparql_filters_built_for_predicate(self):
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
            await c.get(
                "/api/knowledge/query",
                params={"predicate": "http://example.com/pred"},
            )

        sparql_arg = mock_ks.query.call_args[0][0]
        assert "http://example.com/pred" in sparql_arg

    async def test_object_uri_uses_angle_brackets_in_sparql(self):
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
            await c.get(
                "/api/knowledge/query",
                params={"object": "http://example.com/obj"},
            )

        sparql_arg = mock_ks.query.call_args[0][0]
        assert "<http://example.com/obj>" in sparql_arg

    async def test_object_literal_uses_quotes_in_sparql(self):
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
            await c.get(
                "/api/knowledge/query",
                params={"object": "plain literal"},
            )

        sparql_arg = mock_ks.query.call_args[0][0]
        assert '"plain literal"' in sparql_arg


# ---------------------------------------------------------------------------
# Tests: GET /api/knowledge/query — provenance enrichment
# ---------------------------------------------------------------------------


class TestGetKnowledgeQueryProvenance:
    async def test_provenance_enriched_per_triple(self, client):
        """Each result is enriched with provenance from ProvenanceStore."""
        response = await client.get("/api/knowledge/query", params={"subject": _SAMPLE_SUBJECT})
        data = response.json()
        assert len(data[0]["provenance"]) == 1

    async def test_no_provenance_when_none_exists(self):
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/knowledge/query", params={"subject": _SAMPLE_SUBJECT})

        data = response.json()
        assert data[0]["provenance"] == []

    async def test_multiple_provenance_rows(self):
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
            response = await c.get("/api/knowledge/query", params={"subject": _SAMPLE_SUBJECT})

        data = response.json()
        assert len(data[0]["provenance"]) == 2


# ---------------------------------------------------------------------------
# Tests: POST /api/knowledge/sparql — basic
# ---------------------------------------------------------------------------


class TestPostKnowledgeSparql:
    async def test_returns_200(self, client):
        response = await client.post(
            "/api/knowledge/sparql",
            json={"query": "SELECT ?s WHERE { ?s ?p ?o }"},
        )
        assert response.status_code == 200

    async def test_response_is_list(self, client):
        response = await client.post(
            "/api/knowledge/sparql",
            json={"query": "SELECT ?s WHERE { ?s ?p ?o }"},
        )
        assert isinstance(response.json(), list)

    async def test_result_values_are_strings(self, client):
        """All values in the result dicts must be plain strings."""
        response = await client.post(
            "/api/knowledge/sparql",
            json={"query": "SELECT ?s ?p ?o ?conf ?ktype WHERE { ?s ?p ?o }"},
        )
        data = response.json()
        assert len(data) == 1
        for v in data[0].values():
            assert isinstance(v, str)

    async def test_result_contains_expected_keys(self, client):
        response = await client.post(
            "/api/knowledge/sparql",
            json={"query": "SELECT ?s ?p ?o ?conf ?ktype WHERE { ?s ?p ?o }"},
        )
        data = response.json()
        row = data[0]
        assert "s" in row
        assert "p" in row
        assert "o" in row

    async def test_empty_results_when_no_matches(self, empty_client):
        response = await empty_client.post(
            "/api/knowledge/sparql",
            json={"query": "SELECT ?s WHERE { ?s ?p ?o }"},
        )
        assert response.status_code == 200
        assert response.json() == []

    async def test_missing_query_field_returns_422(self, client):
        response = await client.post("/api/knowledge/sparql", json={})
        assert response.status_code == 422

    async def test_knowledge_store_query_called_with_body(self):
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
            await c.post(
                "/api/knowledge/sparql",
                json={"query": "SELECT ?x WHERE { ?x ?y ?z }"},
            )

        mock_ks.query.assert_called_once_with("SELECT ?x WHERE { ?x ?y ?z }")

    async def test_multiple_results_returned(self):
        ks_rows = [
            {
                "s": _FakeNamedNode("http://example.com/a"),
                "p": _FakeNamedNode("http://example.com/p"),
                "o": _FakeLiteral("value-a"),
                "conf": _FakeLiteral("0.9"),
                "ktype": _FakeLiteral("Fact"),
            },
            {
                "s": _FakeNamedNode("http://example.com/b"),
                "p": _FakeNamedNode("http://example.com/p"),
                "o": _FakeLiteral("value-b"),
                "conf": _FakeLiteral("0.7"),
                "ktype": _FakeLiteral("Claim"),
            },
        ]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock(rows=ks_rows)
        app.state.pg_pool = _make_pg_pool_mock(provenance_rows=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post(
                "/api/knowledge/sparql",
                json={"query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"},
            )

        data = response.json()
        assert len(data) == 2
        assert data[0]["s"] == "http://example.com/a"
        assert data[1]["s"] == "http://example.com/b"


# ---------------------------------------------------------------------------
# Tests: GET /api/knowledge/query — local-only results
# ---------------------------------------------------------------------------


class TestGetKnowledgeQueryLocal:
    async def test_query_returns_local_results(self):
        """Local results are returned correctly without any federation logic."""
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/knowledge/query", params={"subject": _SAMPLE_SUBJECT})

        assert response.status_code == 200
        assert len(response.json()) == 1  # Just the local result
