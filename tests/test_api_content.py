"""Integration tests for POST /api/content endpoint (async 202 flow).

All external dependencies (PostgreSQL, Ollama, pyoxigraph KnowledgeStore) are
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

    async def _fetchrow(sql, *args):
        if "ingestion_jobs" in sql and "INSERT" in sql:
            return {"id": "job-uuid-1234"}
        if "ingestion_jobs" in sql and "SELECT" in sql and "status NOT IN" in sql:
            return None  # no active job
        return {"id": "content-uuid-1234"}

    mock_conn.fetchrow.side_effect = _fetchrow
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


def _make_embedding_client_mock():
    """Build a mock EmbeddingClient."""
    mock = AsyncMock()
    mock.embed.return_value = [0.1] * 768
    mock.embed_batch.side_effect = lambda texts, **kw: [[0.1] * 768 for _ in texts]
    return mock


def _make_reasoning_engine_mock():
    """Build a mock ReasoningEngine."""
    mock = MagicMock()
    mock.combine_evidence.return_value = 0.88
    return mock


def _make_extraction_client_mock():
    """Build a mock ExtractionClient that returns no extracted items by default."""
    mock = AsyncMock()
    mock.extract.return_value = []
    return mock


def _make_entity_resolver_mock():
    """Build a mock EntityResolver that returns the input as-is (passthrough)."""
    mock = AsyncMock()

    async def _resolve(label, rdf_type=None):
        slug = label.lower().replace(" ", "_")
        return f"http://knowledge.local/data/{slug}"

    async def _resolve_predicate(label):
        slug = label.lower().replace(" ", "_")
        return f"http://knowledge.local/schema/{slug}"

    mock.resolve.side_effect = _resolve
    mock.resolve_predicate.side_effect = _resolve_predicate
    return mock


def _make_embedding_store_mock():
    """Build a mock EmbeddingStore with new schema methods."""
    mock = AsyncMock()
    mock.insert_content_metadata.return_value = "content-uuid-1234"
    mock.delete_chunks.return_value = None

    async def _insert_chunks(content_id, chunks):
        return [(c["chunk_index"], f"chunk-uuid-{c['chunk_index']}") for c in chunks]

    mock.insert_chunks.side_effect = _insert_chunks
    return mock


def _make_app_with_mocks(**overrides):
    """Create test app with standard mocks. Override any via kwargs."""
    app = create_app(use_lifespan=False)
    app.state.knowledge_store = overrides.get("knowledge_store", _make_knowledge_store_mock())
    app.state.pg_pool = overrides.get("pg_pool", _make_pg_pool_mock())
    app.state.embedding_client = overrides.get("embedding_client", _make_embedding_client_mock())
    app.state.extraction_client = overrides.get("extraction_client", _make_extraction_client_mock())
    app.state.reasoning_engine = overrides.get("reasoning_engine", _make_reasoning_engine_mock())
    app.state.embedding_store = overrides.get("embedding_store", _make_embedding_store_mock())
    if "entity_resolver" in overrides:
        app.state.entity_resolver = overrides["entity_resolver"]
    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client():
    """Create test client with all external dependencies mocked."""
    app = _make_app_with_mocks(entity_resolver=_make_entity_resolver_mock())
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

# Payload with one ClaimInput
CLAIM_PAYLOAD = {
    "url": "https://example.com/claim-article",
    "title": "Claim Article",
    "source_type": "article",
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
    "url": "https://example.com/fact-article",
    "title": "Fact Article",
    "source_type": "research",
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
    "url": "https://example.com/multi",
    "title": "Multi-triple Article",
    "source_type": "article",
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
    "url": "https://example.com/event-article",
    "title": "Event Article",
    "source_type": "news",
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
# Tests: KnowledgeStore interactions (via background worker)
# ---------------------------------------------------------------------------


class TestPostContentKnowledgeStore:
    async def test_insert_triple_called_for_claim(self):
        mock_ks = _make_knowledge_store_mock()
        app = _make_app_with_mocks(knowledge_store=mock_ks)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=CLAIM_PAYLOAD)

        mock_ks.insert_triple.assert_called_once()

    async def test_insert_triple_called_twice_for_two_items(self):
        mock_ks = _make_knowledge_store_mock()
        app = _make_app_with_mocks(knowledge_store=mock_ks)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=MULTI_TRIPLE_PAYLOAD)

        assert mock_ks.insert_triple.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Input validation
# ---------------------------------------------------------------------------


class TestPostContentValidation:
    async def test_missing_url_returns_422(self, client):
        payload = {"title": "No URL", "source_type": "article"}
        response = await client.post("/api/content", json=payload)
        assert response.status_code == 422

    async def test_missing_title_returns_422(self, client):
        payload = {"url": "https://example.com", "source_type": "article"}
        response = await client.post("/api/content", json=payload)
        assert response.status_code == 422

    async def test_missing_source_type_returns_422(self, client):
        payload = {"url": "https://example.com", "title": "Test"}
        response = await client.post("/api/content", json=payload)
        assert response.status_code == 422

    async def test_invalid_confidence_returns_422(self, client):
        payload = {
            "url": "https://example.com",
            "title": "Test",
            "source_type": "article",
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
        response = await client.post("/api/content", json=payload)
        assert response.status_code == 422

    async def test_fact_with_low_confidence_returns_422(self, client):
        payload = {
            "url": "https://example.com",
            "title": "Test",
            "source_type": "article",
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
        response = await client.post("/api/content", json=payload)
        assert response.status_code == 422


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
        mock_es = _make_embedding_store_mock()
        app = _make_app_with_mocks(embedding_store=mock_es)
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
        """Second request for same content_id returns 409 when job is active."""

        call_count = 0

        async def _fetchrow(sql, *args):
            nonlocal call_count
            if "ingestion_jobs" in sql and "INSERT" in sql:
                return {"id": "job-uuid-1234"}
            if "ingestion_jobs" in sql and "SELECT" in sql and "status NOT IN" in sql:
                call_count += 1
                if call_count > 1:
                    return {"id": "existing-job"}  # active job exists
                return None  # first call: no active job
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

            # Second request gets 409
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
        mock_es = _make_embedding_store_mock()
        app = _make_app_with_mocks(embedding_store=mock_es)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post("/api/content", json=SHORT_TEXT_PAYLOAD)

        assert response.status_code == 202
        # Background worker inserts chunks
        mock_es.insert_chunks.assert_called_once()
        chunks = mock_es.insert_chunks.call_args[0][1]
        assert len(chunks) == 1

    async def test_long_content_creates_multiple_chunks(self):
        mock_ec = _make_embedding_client_mock()
        mock_es = _make_embedding_store_mock()
        app = _make_app_with_mocks(embedding_client=mock_ec, embedding_store=mock_es)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post("/api/content", json=LONG_TEXT_PAYLOAD)

        assert response.status_code == 202
        mock_es.insert_chunks.assert_called_once()
        chunks = mock_es.insert_chunks.call_args[0][1]
        assert len(chunks) >= 2

    async def test_reingestion_deletes_old_chunks(self):
        mock_es = _make_embedding_store_mock()
        app = _make_app_with_mocks(embedding_store=mock_es)
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=SHORT_TEXT_PAYLOAD)

        mock_es.delete_chunks.assert_called_once_with("content-uuid-1234")


# ---------------------------------------------------------------------------
# Tests: _resolve_labels — literal object guard
# ---------------------------------------------------------------------------


class TestResolveLabelLiteralGuard:
    async def test_resolve_labels_skips_literal_object(self):
        from knowledge_service.api.content import _resolve_labels
        from knowledge_service.models import ClaimInput

        item = ClaimInput(
            subject="cold_exposure",
            predicate="increases_by",
            object="250% dopamine increase",
            object_type="literal",
            confidence=0.7,
        )
        resolver = AsyncMock()
        resolver.resolve = AsyncMock(return_value="http://knowledge.local/data/cold_exposure")
        resolver.resolve_predicate = AsyncMock(
            return_value="http://knowledge.local/schema/increases_by"
        )

        count, result = await _resolve_labels(item, resolver)

        assert resolver.resolve.call_count == 1  # only subject
        assert result.object == "250% dopamine increase"  # unchanged

    async def test_resolve_labels_resolves_entity_object(self):
        from knowledge_service.api.content import _resolve_labels
        from knowledge_service.models import ClaimInput

        item = ClaimInput(
            subject="cold_exposure",
            predicate="increases",
            object="dopamine",
            object_type="entity",
            confidence=0.7,
        )
        resolver = AsyncMock()
        resolver.resolve = AsyncMock(return_value="http://knowledge.local/data/resolved")
        resolver.resolve_predicate = AsyncMock(
            return_value="http://knowledge.local/schema/increases"
        )

        count, result = await _resolve_labels(item, resolver)

        assert resolver.resolve.call_count == 2  # subject + object


# ---------------------------------------------------------------------------
# Tests: _apply_uri_fallback — literal object guard
# ---------------------------------------------------------------------------


class TestApplyUriFallbackLiteralGuard:
    def test_apply_uri_fallback_preserves_literal_object(self):
        from knowledge_service.api._ingest import apply_uri_fallback
        from knowledge_service.models import ClaimInput

        item = ClaimInput(
            subject="http://knowledge.local/data/cold_exposure",
            predicate="http://knowledge.local/schema/increases",
            object="250% dopamine increase",
            object_type="literal",
            confidence=0.7,
        )
        result = apply_uri_fallback(item)
        assert result.object == "250% dopamine increase"

    def test_apply_uri_fallback_converts_entity_object(self):
        from knowledge_service.api._ingest import apply_uri_fallback
        from knowledge_service.models import ClaimInput

        item = ClaimInput(
            subject="http://knowledge.local/data/cold_exposure",
            predicate="http://knowledge.local/schema/increases",
            object="dopamine",
            object_type="entity",
            confidence=0.7,
        )
        result = apply_uri_fallback(item)
        assert result.object.startswith("http://knowledge.local/data/")

    def test_apply_uri_fallback_normalizes_subject_and_predicate(self):
        from knowledge_service.api._ingest import apply_uri_fallback
        from knowledge_service.models import ClaimInput

        item = ClaimInput(
            subject="cold_exposure",
            predicate="increases",
            object="dopamine",
            object_type="entity",
            confidence=0.7,
        )
        result = apply_uri_fallback(item)
        assert result.subject == "http://knowledge.local/data/cold_exposure"
        assert result.predicate == "http://knowledge.local/schema/increases"

    def test_apply_uri_fallback_resolves_predicate_synonym(self):
        from knowledge_service.api._ingest import apply_uri_fallback
        from knowledge_service.models import ClaimInput

        item = ClaimInput(
            subject="http://knowledge.local/data/x",
            predicate="boosts",
            object="y",
            object_type="entity",
            confidence=0.7,
        )
        result = apply_uri_fallback(item)
        assert result.predicate == "http://knowledge.local/schema/increases"
