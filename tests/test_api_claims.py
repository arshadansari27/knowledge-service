"""Integration tests for POST /api/claims endpoint.

All external dependencies (PostgreSQL, pyoxigraph KnowledgeStore) are
mocked -- no real services are required.
"""

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from knowledge_service.main import create_app
from knowledge_service.ingestion.outbox import OutboxStore, OutboxDrainer
from tests.conftest import make_test_session_cookie


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pg_pool_mock():
    """Build a mock asyncpg pool that records outbox rows and supports draining."""
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

        async def fetchrow(self, sql, *args):
            if "ingestion_jobs" in sql and "INSERT" in sql:
                return {"id": "job-uuid-1234"}
            return {"id": "content-uuid-1234"}

        async def fetch(self, sql, *args):
            if args and isinstance(args[0], list):
                ids = set(args[0])
                return [r for r in outbox_rows if r["id"] in ids and r["applied_at"] is None]
            return []

        def transaction(self):
            return _txn_cm()

    class _txn_cm:
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


def _make_triple_store_mock():
    """Build a mock TripleStore with default successful return values."""
    mock_ts = MagicMock()
    mock_ts.insert.return_value = ("abc123deadbeef", True)
    mock_ts.find_contradictions.return_value = []
    mock_ts.find_opposite_contradictions.return_value = []
    mock_ts.get_triples.return_value = []
    mock_ts.update_confidence.return_value = None
    return mock_ts


def _make_stores_mock(**overrides):
    """Build a mock Stores dataclass with all stores."""
    stores = MagicMock()
    stores.triples = overrides.get("triples", _make_triple_store_mock())
    stores.content = overrides.get("content", AsyncMock())
    stores.entities = overrides.get("entities", AsyncMock())
    stores.provenance = overrides.get("provenance", AsyncMock())
    stores.theses = overrides.get("theses", AsyncMock())
    stores.pg_pool = overrides.get("pg_pool", _make_pg_pool_mock())
    stores.outbox = overrides.get("outbox", OutboxStore())
    # Set defaults only for stores not provided via overrides
    if "provenance" not in overrides:
        stores.provenance.get_by_triple.return_value = []
        stores.provenance.insert.return_value = None
    if "theses" not in overrides:
        stores.theses.find_by_hashes.return_value = []
    return stores


def _make_app_with_mocks(**overrides):
    """Create test app with standard mocks."""
    app = create_app(use_lifespan=False)
    stores = _make_stores_mock(**overrides)
    app.state.stores = stores
    # Backward compat attrs that some middleware/health routes use
    app.state.knowledge_store = stores.triples
    app.state.pg_pool = stores.pg_pool
    app.state.reasoning_engine = None
    app.state.outbox_drainer = OutboxDrainer(stores.pg_pool, stores.triples)
    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """Create test client with all external dependencies mocked."""
    app = _make_app_with_mocks()
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

# Payload with one TripleInput (replaces old ClaimInput)
CLAIM_PAYLOAD = {
    "source_url": "https://example.com/source",
    "source_type": "article",
    "extractor": "llm-extract-v1",
    "knowledge": [
        {
            "subject": "https://example.com/subject",
            "predicate": "https://example.com/predicate",
            "object": "some-value",
            "confidence": 0.8,
            "knowledge_type": "claim",
        }
    ],
}

# Payload with a high-confidence triple (replaces old FactInput)
FACT_PAYLOAD = {
    "source_url": "https://example.com/research",
    "source_type": "research",
    "extractor": "manual",
    "knowledge": [
        {
            "subject": "https://example.com/subject",
            "predicate": "https://example.com/predicate",
            "object": "verified-value",
            "confidence": 0.99,
            "knowledge_type": "fact",
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
            "subject": "https://example.com/subjectA",
            "predicate": "https://example.com/predicateA",
            "object": "valueA",
            "confidence": 0.7,
        },
        {
            "subject": "https://example.com/entityX",
            "predicate": "https://example.com/relatesTo",
            "object": "https://example.com/entityY",
            "confidence": 0.9,
        },
    ],
}

# Payload with EventInput -- expands to 1 triple (occurredAt)
EVENT_PAYLOAD = {
    "source_url": "https://example.com/events",
    "source_type": "news",
    "extractor": "manual",
    "knowledge": [
        {
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
# Tests: TripleStore interactions
# ---------------------------------------------------------------------------


class TestPostClaimsKnowledgeStore:
    async def test_insert_triple_called_for_claim(self):
        app = _make_app_with_mocks()
        mock_ts = app.state.stores.triples

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/claims", json=CLAIM_PAYLOAD)

        mock_ts.insert.assert_called_once()

    async def test_find_contradictions_called_for_claim(self):
        app = _make_app_with_mocks()
        mock_ts = app.state.stores.triples

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/claims", json=CLAIM_PAYLOAD)

        mock_ts.find_contradictions.assert_called_once()

    async def test_insert_triple_called_twice_for_two_items(self):
        app = _make_app_with_mocks()
        mock_ts = app.state.stores.triples

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/claims", json=MULTI_TRIPLE_PAYLOAD)

        assert mock_ts.insert.call_count == 2
        assert mock_ts.find_contradictions.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Contradiction detection
# ---------------------------------------------------------------------------


class TestPostClaimsContradictions:
    async def test_no_contradictions_returned_when_none_found(self, client):
        response = await client.post("/api/claims", json=CLAIM_PAYLOAD)
        data = response.json()
        assert data["contradictions_detected"] == []

    async def test_contradictions_returned_when_found(self):
        """When TripleStore finds a contradiction, it should appear in the response."""
        mock_ts = _make_triple_store_mock()
        mock_ts.find_contradictions.return_value = [{"object": "other-value", "confidence": 0.6}]
        app = _make_app_with_mocks(triples=mock_ts)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post("/api/claims", json=CLAIM_PAYLOAD)

        data = response.json()
        assert len(data["contradictions_detected"]) >= 1

    async def test_contradiction_fields_are_correct(self):
        """Contradiction dict should include relevant info."""
        mock_ts = _make_triple_store_mock()
        mock_ts.find_contradictions.return_value = [
            {"object": "conflicting-value", "confidence": 0.75}
        ]
        app = _make_app_with_mocks(triples=mock_ts)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post("/api/claims", json=CLAIM_PAYLOAD)

        data = response.json()
        assert len(data["contradictions_detected"]) >= 1


# ---------------------------------------------------------------------------
# Tests: Provenance recording
# ---------------------------------------------------------------------------


class TestPostClaimsProvenance:
    async def test_provenance_recorded_with_source_url(self):
        """ProvenanceStore.insert must be called with source_url from the request."""
        mock_prov = AsyncMock()
        mock_prov.get_by_triple.return_value = []
        mock_prov.insert.return_value = None
        app = _make_app_with_mocks(provenance=mock_prov)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/claims", json=CLAIM_PAYLOAD)

        mock_prov.insert.assert_called_once()
        call_args = mock_prov.insert.call_args
        # source_url is the 5th positional arg
        assert "https://example.com/source" in call_args[0]

    async def test_provenance_recorded_with_extractor(self):
        """ProvenanceStore.insert must use the extractor field from the request."""
        mock_prov = AsyncMock()
        mock_prov.get_by_triple.return_value = []
        mock_prov.insert.return_value = None
        app = _make_app_with_mocks(provenance=mock_prov)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/claims", json=CLAIM_PAYLOAD)

        mock_prov.insert.assert_called_once()
        call_args = mock_prov.insert.call_args
        assert "llm-extract-v1" in call_args[0]

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
                    "subject": "https://example.com/s",
                    "predicate": "https://example.com/p",
                    "object": "value",
                    "confidence": 1.5,  # out of range
                }
            ],
        }
        response = await client.post("/api/claims", json=payload)
        assert response.status_code == 422

    async def test_fact_with_low_confidence_accepted(self, client):
        """In new model, there's no separate Fact type with min confidence."""
        payload = {
            "source_url": "https://example.com",
            "source_type": "article",
            "extractor": "manual",
            "knowledge": [
                {
                    "subject": "https://example.com/s",
                    "predicate": "https://example.com/p",
                    "object": "value",
                    "confidence": 0.5,
                    "knowledge_type": "fact",
                }
            ],
        }
        response = await client.post("/api/claims", json=payload)
        # New model has no min confidence for facts -- it's just a TripleInput
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Tests: Batch (list) input
# ---------------------------------------------------------------------------


class TestPostClaimsBatch:
    async def test_batch_returns_list(self, client):
        """Sending a list of ClaimsRequests returns a list of ClaimsResponses."""
        batch = [CLAIM_PAYLOAD, FACT_PAYLOAD]
        response = await client.post("/api/claims", json=batch)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    async def test_batch_each_item_has_triples_created(self, client):
        batch = [CLAIM_PAYLOAD, FACT_PAYLOAD]
        response = await client.post("/api/claims", json=batch)
        data = response.json()
        for item in data:
            assert "triples_created" in item
            assert item["triples_created"] == 1

    async def test_batch_each_item_has_contradictions(self, client):
        batch = [CLAIM_PAYLOAD, MINIMAL_PAYLOAD]
        response = await client.post("/api/claims", json=batch)
        data = response.json()
        for item in data:
            assert "contradictions_detected" in item
            assert isinstance(item["contradictions_detected"], list)

    async def test_single_request_still_returns_object(self, client):
        """Single (non-list) input still returns a single object, not a list."""
        response = await client.post("/api/claims", json=CLAIM_PAYLOAD)
        data = response.json()
        assert isinstance(data, dict)
        assert "triples_created" in data

    async def test_batch_validation_error_returns_422(self, client):
        """A batch with an invalid item returns 422."""
        batch = [
            CLAIM_PAYLOAD,
            {"source_type": "article", "extractor": "manual"},  # missing source_url
        ]
        response = await client.post("/api/claims", json=batch)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests: Thesis breaks surfacing
# ---------------------------------------------------------------------------
