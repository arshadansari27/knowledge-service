"""Integration tests for POST /api/content endpoint (async 202 flow).

All external dependencies (PostgreSQL, Ollama, pyoxigraph KnowledgeStore) are
mocked -- no real services are required.
"""

import pytest
import asyncpg.exceptions
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


def _make_embedding_client_mock():
    """Build a mock EmbeddingClient."""
    mock = AsyncMock()
    mock.embed.return_value = [0.1] * 768
    mock.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]
    return mock


def _make_extraction_client_mock():
    """Build a mock ExtractionClient that returns no extracted items by default."""
    mock = AsyncMock()
    mock.extract.return_value = []
    return mock


def _make_content_store_mock():
    """Build a mock ContentStore with new schema methods."""
    mock = AsyncMock()
    mock.upsert_metadata.return_value = "content-uuid-1234"
    mock.delete_chunks.return_value = None

    async def _insert_chunks(content_id, chunks):
        return [(c["chunk_index"], f"chunk-uuid-{c['chunk_index']}") for c in chunks]

    mock.insert_chunks.side_effect = _insert_chunks
    mock.replace_chunks.side_effect = _insert_chunks
    return mock


def _make_entity_store_mock():
    """Build a mock EntityStore."""
    mock = AsyncMock()

    async def _resolve(label, rdf_type=None):
        slug = label.lower().replace(" ", "_")
        return f"http://knowledge.local/data/{slug}"

    async def _resolve_predicate(label):
        slug = label.lower().replace(" ", "_")
        return f"http://knowledge.local/schema/{slug}"

    mock.resolve_entity.side_effect = _resolve
    mock.resolve_predicate.side_effect = _resolve_predicate
    return mock


def _make_app_with_mocks(**overrides):
    """Create test app with standard mocks. Override any via kwargs."""
    app = create_app(use_lifespan=False)

    mock_ts = overrides.get("triples", _make_triple_store_mock())
    mock_pg = overrides.get("pg_pool", _make_pg_pool_mock())
    mock_content = overrides.get("content", _make_content_store_mock())
    mock_entities = overrides.get("entities", _make_entity_store_mock())
    mock_provenance = overrides.get("provenance", AsyncMock())
    mock_provenance.get_by_triple.return_value = []
    mock_provenance.insert.return_value = None
    mock_theses = overrides.get("theses", AsyncMock())
    mock_theses.find_by_hashes.return_value = []

    stores = MagicMock()
    stores.triples = mock_ts
    stores.content = mock_content
    stores.entities = mock_entities
    stores.provenance = mock_provenance
    stores.theses = mock_theses
    stores.pg_pool = mock_pg
    stores.outbox = overrides.get("outbox", OutboxStore())
    app.state.stores = stores

    app.state.embedding_client = overrides.get("embedding_client", _make_embedding_client_mock())
    app.state.extraction_client = overrides.get("extraction_client", _make_extraction_client_mock())

    # Backward compat
    app.state.knowledge_store = mock_ts
    app.state.pg_pool = mock_pg
    app.state.reasoning_engine = None
    app.state.outbox_drainer = OutboxDrainer(mock_pg, mock_ts)
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
    "url": "https://example.com/article",
    "title": "Test Article",
    "source_type": "article",
}

# Payload with one TripleInput
CLAIM_PAYLOAD = {
    "url": "https://example.com/claim-article",
    "title": "Claim Article",
    "source_type": "article",
    "knowledge": [
        {
            "subject": "https://example.com/subject",
            "predicate": "https://example.com/predicate",
            "object": "some-value",
            "confidence": 0.8,
        }
    ],
}

# Payload with a high-confidence triple
FACT_PAYLOAD = {
    "url": "https://example.com/fact-article",
    "title": "Fact Article",
    "source_type": "research",
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
    "url": "https://example.com/multi",
    "title": "Multi-triple Article",
    "source_type": "article",
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
    "url": "https://example.com/event-article",
    "title": "Event Article",
    "source_type": "news",
    "knowledge": [
        {
            "subject": "https://example.com/event1",
            "occurred_at": "2024-01-15",
            "confidence": 1.0,
        }
    ],
}


# ---------------------------------------------------------------------------
# Tests: basic 202 response structure
# ---------------------------------------------------------------------------


class TestPostContentBasic:
    async def test_returns_202(self, client):
        response = await client.post("/api/content", json=MINIMAL_PAYLOAD)
        assert response.status_code == 202

    async def test_response_has_content_id(self, client):
        response = await client.post("/api/content", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert "content_id" in data
        assert data["content_id"]  # non-empty

    async def test_response_has_job_id(self, client):
        response = await client.post("/api/content", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert "job_id" in data
        assert data["job_id"]

    async def test_response_has_status_accepted(self, client):
        response = await client.post("/api/content", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert data["status"] == "accepted"

    async def test_response_has_chunks_total(self, client):
        response = await client.post("/api/content", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert "chunks_total" in data


# ---------------------------------------------------------------------------
# Tests: TripleStore interactions (via background worker)
# ---------------------------------------------------------------------------


class TestPostContentKnowledgeStore:
    async def test_insert_triple_called_for_claim(self):
        app = _make_app_with_mocks()
        mock_ts = app.state.stores.triples
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=CLAIM_PAYLOAD)

        mock_ts.insert.assert_called_once()

    async def test_insert_triple_called_twice_for_two_items(self):
        app = _make_app_with_mocks()
        mock_ts = app.state.stores.triples
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=MULTI_TRIPLE_PAYLOAD)

        assert mock_ts.insert.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Input validation
# ---------------------------------------------------------------------------


class TestPostContentValidation:
    async def test_missing_url_returns_422(self, client):
        payload = {"title": "No URL", "source_type": "article"}
        response = await client.post("/api/content", json=payload)
        assert response.status_code == 422

    async def test_missing_title_accepted(self, client):
        """Title is optional in the new model."""
        payload = {"url": "https://example.com", "source_type": "article", "raw_text": "Some text"}
        response = await client.post("/api/content", json=payload)
        assert response.status_code == 202

    async def test_missing_source_type_accepted(self, client):
        """Source type is optional in the new model."""
        payload = {"url": "https://example.com", "title": "Test"}
        response = await client.post("/api/content", json=payload)
        assert response.status_code == 202

    async def test_invalid_confidence_returns_422(self, client):
        payload = {
            "url": "https://example.com",
            "title": "Test",
            "source_type": "article",
            "knowledge": [
                {
                    "subject": "https://example.com/s",
                    "predicate": "https://example.com/p",
                    "object": "value",
                    "confidence": 1.5,  # out of range
                }
            ],
        }
        response = await client.post("/api/content", json=payload)
        assert response.status_code == 422

    async def test_fact_with_low_confidence_accepted(self, client):
        """In new model there's no separate Fact type with min confidence."""
        payload = {
            "url": "https://example.com",
            "title": "Test",
            "source_type": "article",
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
        response = await client.post("/api/content", json=payload)
        assert response.status_code == 202


# ---------------------------------------------------------------------------
# Tests: Auto-extraction behaviour
# ---------------------------------------------------------------------------

RAW_TEXT_PAYLOAD = {
    "url": "https://example.com/raw",
    "title": "Raw Article",
    "raw_text": "Cold exposure increases dopamine significantly.",
    "source_type": "article",
}


class TestPostContentExtraction:
    async def test_extract_called_when_no_knowledge_and_raw_text(self):
        mock_xc = _make_extraction_client_mock()
        app = _make_app_with_mocks(extraction_client=mock_xc)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=RAW_TEXT_PAYLOAD)

        mock_xc.extract.assert_called_once()

    async def test_extract_not_called_when_knowledge_provided(self):
        mock_xc = _make_extraction_client_mock()
        app = _make_app_with_mocks(extraction_client=mock_xc)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=CLAIM_PAYLOAD)

        mock_xc.extract.assert_not_called()

    async def test_extract_not_called_when_no_raw_text(self):
        mock_xc = _make_extraction_client_mock()
        app = _make_app_with_mocks(extraction_client=mock_xc)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=MINIMAL_PAYLOAD)

        mock_xc.extract.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Embedding and content storage
# ---------------------------------------------------------------------------


class TestPostContentEmbedding:
    async def test_embedding_client_called(self):
        mock_ec = _make_embedding_client_mock()
        app = _make_app_with_mocks(embedding_client=mock_ec)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=MINIMAL_PAYLOAD)

        # embed_batch called (background worker uses sub-batch loop calling embed_batch)
        mock_ec.embed_batch.assert_called()

    async def test_content_id_from_metadata_insert(self):
        app = _make_app_with_mocks()
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post("/api/content", json=MINIMAL_PAYLOAD)

        data = response.json()
        assert data["content_id"] == "content-uuid-1234"


# ---------------------------------------------------------------------------
# Tests: Batch (list) input
# ---------------------------------------------------------------------------


class TestPostContentBatch:
    async def test_batch_returns_list(self, client):
        batch = [MINIMAL_PAYLOAD, CLAIM_PAYLOAD]
        response = await client.post("/api/content", json=batch)
        assert response.status_code == 202
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    async def test_batch_each_item_has_job_id(self, client):
        batch = [MINIMAL_PAYLOAD, CLAIM_PAYLOAD]
        response = await client.post("/api/content", json=batch)
        data = response.json()
        for item in data:
            assert "job_id" in item

    async def test_single_request_still_returns_object(self, client):
        response = await client.post("/api/content", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert isinstance(data, dict)
        assert "content_id" in data

    async def test_batch_partial_validation_error(self, client):
        """Batch with one invalid item returns 202 with mixed results."""
        batch = [
            MINIMAL_PAYLOAD,
            {"title": "No URL", "source_type": "article"},  # missing url
        ]
        response = await client.post("/api/content", json=batch)
        assert response.status_code == 202
        data = response.json()
        assert len(data) == 2
        assert "job_id" in data[0]  # first item succeeded
        assert "error" in data[1]  # second item failed


# ---------------------------------------------------------------------------
# Tests: Idempotency guard
# ---------------------------------------------------------------------------


class TestIdempotencyGuard:
    async def test_active_job_returns_409(self):
        """Second request for same content_id returns 409 when partial unique index fires."""

        insert_count = 0

        async def _fetchrow(sql, *args):
            nonlocal insert_count
            if "ingestion_jobs" in sql and "INSERT" in sql:
                insert_count += 1
                if insert_count > 1:
                    # Simulate the partial unique index rejecting a duplicate active job
                    raise asyncpg.exceptions.UniqueViolationError(
                        "duplicate key value violates unique constraint "
                        '"idx_ingestion_jobs_active_content"'
                    )
                return {"id": "job-uuid-1234"}
            return {"id": "content-uuid-1234"}

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "INSERT 0 1"
        mock_conn.fetchrow.side_effect = _fetchrow
        mock_conn.fetch.return_value = []

        @asynccontextmanager
        async def _acquire():
            yield mock_conn

        mock_pool = MagicMock()
        mock_pool.acquire = _acquire

        app = _make_app_with_mocks(pg_pool=mock_pool)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            # First request succeeds
            resp1 = await c.post("/api/content", json=MINIMAL_PAYLOAD)
            assert resp1.status_code == 202

            # Second request gets 409 because the unique index blocks the INSERT
            resp2 = await c.post("/api/content", json=MINIMAL_PAYLOAD)
            assert resp2.status_code == 409


# ---------------------------------------------------------------------------
# Tests: Content chunking
# ---------------------------------------------------------------------------

SHORT_TEXT_PAYLOAD = {
    "url": "https://example.com/short",
    "title": "Short Article",
    "raw_text": "This is a short article.",
    "source_type": "article",
}

LONG_TEXT_PAYLOAD = {
    "url": "https://example.com/long",
    "title": "Long Article",
    "raw_text": "A" * 5000,
    "source_type": "article",
}


class TestContentChunking:
    async def test_short_content_creates_one_chunk(self):
        app = _make_app_with_mocks()
        mock_cs = app.state.stores.content
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post("/api/content", json=SHORT_TEXT_PAYLOAD)

        assert response.status_code == 202
        # Background worker replaces chunks atomically
        mock_cs.replace_chunks.assert_called_once()
        chunks = mock_cs.replace_chunks.call_args[0][1]
        assert len(chunks) == 1

    async def test_long_content_creates_multiple_chunks(self):
        mock_ec = _make_embedding_client_mock()
        app = _make_app_with_mocks(embedding_client=mock_ec)
        mock_cs = app.state.stores.content
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post("/api/content", json=LONG_TEXT_PAYLOAD)

        assert response.status_code == 202
        mock_cs.replace_chunks.assert_called_once()
        chunks = mock_cs.replace_chunks.call_args[0][1]
        assert len(chunks) >= 2

    async def test_reingestion_replaces_chunks_atomically(self):
        app = _make_app_with_mocks()
        mock_cs = app.state.stores.content
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=SHORT_TEXT_PAYLOAD)

        mock_cs.replace_chunks.assert_called_once()
        assert mock_cs.replace_chunks.call_args[0][0] == "content-uuid-1234"
        # The old non-atomic pair must not be used from the ingest pipeline
        mock_cs.delete_chunks.assert_not_called()
        mock_cs.insert_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: GET /api/content/{content_id}/chunks
# ---------------------------------------------------------------------------

_SAMPLE_CHUNKS = [
    {
        "chunk_index": 0,
        "chunk_text": "CONSULTANCY AGREEMENT dated 31 July 2025",
        "section_header": "INTERPRETATION",
        "char_start": 0,
        "char_end": 3800,
    },
    {
        "chunk_index": 1,
        "chunk_text": "The Company shall pay a fee of £317.90 per day",
        "section_header": "FEES",
        "char_start": 3800,
        "char_end": 7600,
    },
]


def _make_chunks_app(chunks=None):
    """Create test app with mocked content store that returns chunks."""
    app = create_app(use_lifespan=False)

    content_mock = AsyncMock()
    content_mock.get_chunks.return_value = chunks

    stores = MagicMock()
    stores.content = content_mock
    stores.triples = MagicMock()
    stores.entities = AsyncMock()
    stores.provenance = AsyncMock()
    stores.theses = AsyncMock()
    stores.pg_pool = MagicMock()
    app.state.stores = stores
    app.state.embedding_client = _make_embedding_client_mock()
    app.state.knowledge_store = stores.triples
    app.state.pg_pool = stores.pg_pool
    return app


class TestGetContentChunks:
    async def test_returns_chunks_ordered(self):
        app = _make_chunks_app(chunks=_SAMPLE_CHUNKS)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            resp = await client.get("/api/content/content-uuid-1234/chunks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["chunk_index"] == 0
        assert data[1]["chunk_index"] == 1
        assert "chunk_text" in data[0]
        assert "section_header" in data[0]

    async def test_returns_404_when_not_found(self):
        app = _make_chunks_app(chunks=None)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            resp = await client.get("/api/content/nonexistent-uuid/chunks")
        assert resp.status_code == 404
