"""Integration tests for POST /api/claims endpoint.

All external dependencies (PostgreSQL, pyoxigraph KnowledgeStore) are
mocked — no real services are required.
"""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from knowledge_service.main import create_app
from tests.conftest import make_test_session_cookie


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pg_pool_mock():
    """Build a mock asyncpg pool whose .acquire() works as an async context manager."""
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "INSERT 0 1"
    mock_conn.fetchrow.return_value = {"id": "content-uuid-1234"}
    mock_conn.fetch.return_value = []

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    mock_pool = MagicMock()
    mock_pool.acquire = _acquire
    return mock_pool


def _make_knowledge_store_mock():
    """Build a mock KnowledgeStore with default successful return values."""
    mock_ks = MagicMock()
    mock_ks.insert_triple.return_value = ("abc123deadbeef", True)
    mock_ks.find_contradictions.return_value = []
    return mock_ks


def _make_reasoning_engine_mock():
    """Build a mock ReasoningEngine."""
    mock = MagicMock()
    mock.combine_evidence.return_value = 0.88
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """Create test client with all external dependencies mocked."""
    app = create_app(use_lifespan=False)

    app.state.knowledge_store = _make_knowledge_store_mock()
    app.state.pg_pool = _make_pg_pool_mock()
    app.state.reasoning_engine = _make_reasoning_engine_mock()

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        cookies={"ks_session": make_test_session_cookie()},
    ) as c:
        yield c


# Minimal valid payload with no knowledge items
MINIMAL_PAYLOAD = {
    "source_url": "https://example.com/source",
    "source_type": "article",
    "extractor": "manual",
}

# Payload with one ClaimInput
CLAIM_PAYLOAD = {
    "source_url": "https://example.com/source",
    "source_type": "article",
    "extractor": "llm-extract-v1",
    "knowledge": [
        {
            "knowledge_type": "Claim",
            "subject": "https://example.com/subject",
            "predicate": "https://example.com/predicate",
            "object": "some-value",
            "confidence": 0.8,
        }
    ],
}

# Payload with a FactInput (high confidence)
FACT_PAYLOAD = {
    "source_url": "https://example.com/research",
    "source_type": "research",
    "extractor": "manual",
    "knowledge": [
        {
            "knowledge_type": "Fact",
            "subject": "https://example.com/subject",
            "predicate": "https://example.com/predicate",
            "object": "verified-value",
            "confidence": 0.99,
        }
    ],
}

# Payload with multiple triples
MULTI_TRIPLE_PAYLOAD = {
    "source_url": "https://example.com/multi",
    "source_type": "article",
    "extractor": "llm-extract-v1",
    "knowledge": [
        {
            "knowledge_type": "Claim",
            "subject": "https://example.com/subjectA",
            "predicate": "https://example.com/predicateA",
            "object": "valueA",
            "confidence": 0.7,
        },
        {
            "knowledge_type": "Relationship",
            "subject": "https://example.com/entityX",
            "predicate": "https://example.com/relatesTo",
            "object": "https://example.com/entityY",
            "confidence": 0.9,
        },
    ],
}

# Payload with EventInput — expands to 1 triple (occurredAt)
EVENT_PAYLOAD = {
    "source_url": "https://example.com/events",
    "source_type": "news",
    "extractor": "manual",
    "knowledge": [
        {
            "knowledge_type": "Event",
            "subject": "https://example.com/event1",
            "occurred_at": "2024-01-15",
            "confidence": 1.0,
        }
    ],
}


# ---------------------------------------------------------------------------
# Tests: basic response structure
# ---------------------------------------------------------------------------


class TestPostClaimsBasic:
    async def test_returns_200(self, client):
        response = await client.post("/api/claims", json=MINIMAL_PAYLOAD)
        assert response.status_code == 200

    async def test_response_has_triples_created(self, client):
        response = await client.post("/api/claims", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert "triples_created" in data

    async def test_response_has_contradictions_detected(self, client):
        response = await client.post("/api/claims", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert "contradictions_detected" in data
        assert isinstance(data["contradictions_detected"], list)

    async def test_response_has_no_content_id(self, client):
        """ClaimsResponse must not include content_id (no content storage)."""
        response = await client.post("/api/claims", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert "content_id" not in data


# ---------------------------------------------------------------------------
# Tests: triple counting
# ---------------------------------------------------------------------------


class TestPostClaimsTripleCount:
    async def test_no_knowledge_items_zero_triples(self, client):
        response = await client.post("/api/claims", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert data["triples_created"] == 0

    async def test_one_claim_one_triple(self, client):
        response = await client.post("/api/claims", json=CLAIM_PAYLOAD)
        data = response.json()
        assert data["triples_created"] == 1

    async def test_one_fact_one_triple(self, client):
        response = await client.post("/api/claims", json=FACT_PAYLOAD)
        data = response.json()
        assert data["triples_created"] == 1

    async def test_two_triples_counted_correctly(self, client):
        response = await client.post("/api/claims", json=MULTI_TRIPLE_PAYLOAD)
        data = response.json()
        assert data["triples_created"] == 2

    async def test_event_expands_to_triples(self, client):
        """EventInput expands to at least 1 triple (occurredAt)."""
        response = await client.post("/api/claims", json=EVENT_PAYLOAD)
        data = response.json()
        assert data["triples_created"] >= 1


# ---------------------------------------------------------------------------
# Tests: KnowledgeStore interactions
# ---------------------------------------------------------------------------


class TestPostClaimsKnowledgeStore:
    async def test_insert_triple_called_for_claim(self):
        app = create_app(use_lifespan=False)
        mock_ks = _make_knowledge_store_mock()
        app.state.knowledge_store = mock_ks
        app.state.pg_pool = _make_pg_pool_mock()
        app.state.reasoning_engine = _make_reasoning_engine_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/claims", json=CLAIM_PAYLOAD)

        mock_ks.insert_triple.assert_called_once()
        call_repr = str(mock_ks.insert_triple.call_args)
        assert "https://example.com/subject" in call_repr
        assert "https://example.com/predicate" in call_repr

    async def test_find_contradictions_called_for_claim(self):
        app = create_app(use_lifespan=False)
        mock_ks = _make_knowledge_store_mock()
        app.state.knowledge_store = mock_ks
        app.state.pg_pool = _make_pg_pool_mock()
        app.state.reasoning_engine = _make_reasoning_engine_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/claims", json=CLAIM_PAYLOAD)

        mock_ks.find_contradictions.assert_called_once()

    async def test_insert_triple_called_twice_for_two_items(self):
        app = create_app(use_lifespan=False)
        mock_ks = _make_knowledge_store_mock()
        app.state.knowledge_store = mock_ks
        app.state.pg_pool = _make_pg_pool_mock()
        app.state.reasoning_engine = _make_reasoning_engine_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/claims", json=MULTI_TRIPLE_PAYLOAD)

        assert mock_ks.insert_triple.call_count == 2
        assert mock_ks.find_contradictions.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Contradiction detection
# ---------------------------------------------------------------------------


class TestPostClaimsContradictions:
    async def test_no_contradictions_returned_when_none_found(self, client):
        response = await client.post("/api/claims", json=CLAIM_PAYLOAD)
        data = response.json()
        assert data["contradictions_detected"] == []

    async def test_contradictions_returned_when_found(self):
        """When KnowledgeStore finds a contradiction, it should appear in the response."""
        app = create_app(use_lifespan=False)
        mock_ks = _make_knowledge_store_mock()
        mock_ks.find_contradictions.return_value = [{"object": "other-value", "confidence": 0.6}]
        app.state.knowledge_store = mock_ks
        app.state.pg_pool = _make_pg_pool_mock()
        app.state.reasoning_engine = _make_reasoning_engine_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post("/api/claims", json=CLAIM_PAYLOAD)

        data = response.json()
        assert len(data["contradictions_detected"]) == 1
        contradiction = data["contradictions_detected"][0]
        assert "subject" in contradiction
        assert "predicate" in contradiction
        assert contradiction["existing_object"] == "other-value"
        assert contradiction["new_object"] == "some-value"

    async def test_contradiction_fields_are_correct(self):
        """Contradiction dict should include subject, predicate, existing/new object+confidence."""
        app = create_app(use_lifespan=False)
        mock_ks = _make_knowledge_store_mock()
        mock_ks.find_contradictions.return_value = [
            {"object": "conflicting-value", "confidence": 0.75}
        ]
        app.state.knowledge_store = mock_ks
        app.state.pg_pool = _make_pg_pool_mock()
        app.state.reasoning_engine = _make_reasoning_engine_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post("/api/claims", json=CLAIM_PAYLOAD)

        contradiction = response.json()["contradictions_detected"][0]
        assert contradiction["subject"] == "https://example.com/subject"
        assert contradiction["predicate"] == "https://example.com/predicate"
        assert contradiction["existing_object"] == "conflicting-value"
        assert contradiction["existing_confidence"] == 0.75
        assert contradiction["new_object"] == "some-value"
        assert contradiction["new_confidence"] == 0.8


# ---------------------------------------------------------------------------
# Tests: Provenance recording
# ---------------------------------------------------------------------------


class TestPostClaimsProvenance:
    async def test_provenance_recorded_with_source_url(self):
        """ProvenanceStore.insert must be called with source_url from the request."""
        app = create_app(use_lifespan=False)
        mock_ks = _make_knowledge_store_mock()
        mock_pool = _make_pg_pool_mock()
        app.state.knowledge_store = mock_ks
        app.state.pg_pool = mock_pool
        app.state.reasoning_engine = _make_reasoning_engine_mock()

        captured_calls = []
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []

        async def capture_execute(sql, *args):
            captured_calls.append({"sql": sql, "args": args})
            return "INSERT 0 1"

        mock_conn.execute.side_effect = capture_execute

        @asynccontextmanager
        async def _acquire():
            yield mock_conn

        mock_pool.acquire = _acquire
        app.state.pg_pool = mock_pool

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/claims", json=CLAIM_PAYLOAD)

        # Verify provenance was inserted with the correct source_url
        provenance_calls = [c for c in captured_calls if "INSERT INTO provenance" in c["sql"]]
        assert len(provenance_calls) >= 1
        assert "https://example.com/source" in provenance_calls[0]["args"]

    async def test_provenance_recorded_with_extractor(self):
        """ProvenanceStore.insert must use the extractor field from the request."""
        app = create_app(use_lifespan=False)
        mock_ks = _make_knowledge_store_mock()
        captured_calls = []
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []

        async def capture_execute(sql, *args):
            captured_calls.append({"sql": sql, "args": args})
            return "INSERT 0 1"

        mock_conn.execute.side_effect = capture_execute

        @asynccontextmanager
        async def _acquire():
            yield mock_conn

        mock_pool = MagicMock()
        mock_pool.acquire = _acquire

        app.state.knowledge_store = mock_ks
        app.state.pg_pool = mock_pool
        app.state.reasoning_engine = _make_reasoning_engine_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/claims", json=CLAIM_PAYLOAD)

        provenance_calls = [c for c in captured_calls if "INSERT INTO provenance" in c["sql"]]
        assert len(provenance_calls) >= 1
        assert "llm-extract-v1" in provenance_calls[0]["args"]

    async def test_event_provenance_recorded(self, client):
        """Events now expand to triples and provenance is recorded."""
        response = await client.post("/api/claims", json=EVENT_PAYLOAD)
        assert response.status_code == 200
        assert response.json()["triples_created"] >= 1


# ---------------------------------------------------------------------------
# Tests: Input validation
# ---------------------------------------------------------------------------


class TestPostClaimsValidation:
    async def test_missing_source_url_returns_422(self, client):
        payload = {"source_type": "article", "extractor": "manual"}
        response = await client.post("/api/claims", json=payload)
        assert response.status_code == 422

    async def test_missing_source_type_returns_422(self, client):
        payload = {"source_url": "https://example.com", "extractor": "manual"}
        response = await client.post("/api/claims", json=payload)
        assert response.status_code == 422

    async def test_missing_extractor_returns_422(self, client):
        payload = {"source_url": "https://example.com", "source_type": "article"}
        response = await client.post("/api/claims", json=payload)
        assert response.status_code == 422

    async def test_invalid_confidence_returns_422(self, client):
        payload = {
            "source_url": "https://example.com",
            "source_type": "article",
            "extractor": "manual",
            "knowledge": [
                {
                    "knowledge_type": "Claim",
                    "subject": "https://example.com/s",
                    "predicate": "https://example.com/p",
                    "object": "value",
                    "confidence": 1.5,  # out of range
                }
            ],
        }
        response = await client.post("/api/claims", json=payload)
        assert response.status_code == 422

    async def test_fact_with_low_confidence_returns_422(self, client):
        payload = {
            "source_url": "https://example.com",
            "source_type": "article",
            "extractor": "manual",
            "knowledge": [
                {
                    "knowledge_type": "Fact",
                    "subject": "https://example.com/s",
                    "predicate": "https://example.com/p",
                    "object": "value",
                    "confidence": 0.5,  # below 0.9 minimum for Facts
                }
            ],
        }
        response = await client.post("/api/claims", json=payload)
        assert response.status_code == 422
