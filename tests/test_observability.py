"""PR1 observability smoke tests: exception handler and lazy predicate seeding."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from knowledge_service.main import create_app
from knowledge_service.stores.entities import EntityStore


class TestGlobalExceptionHandler:
    def test_unhandled_exception_logged_with_traceback_and_returns_500(self, caplog):
        app = create_app(use_lifespan=False)

        boom = APIRouter()

        @boom.get("/_boom")
        async def _boom():
            raise RuntimeError("boom-marker-xyz")

        app.include_router(boom)

        # The AuthMiddleware was constructed with settings.admin_password; pass
        # that value as X-API-Key to bypass auth on /_boom.
        from knowledge_service.config import settings

        client = TestClient(app, raise_server_exceptions=False)
        with caplog.at_level(logging.ERROR, logger="knowledge_service.main"):
            resp = client.get("/_boom", headers={"X-API-Key": settings.admin_password})

        assert resp.status_code == 500
        body = resp.json()
        assert body["detail"] == "Internal Server Error"
        assert body["type"] == "RuntimeError"

        # Traceback + route context should both appear in the log record.
        log_text = "\n".join(r.getMessage() + "\n" + (r.exc_text or "") for r in caplog.records)
        assert "boom-marker-xyz" in log_text
        assert "/_boom" in log_text


class TestLazyPredicateSeeding:
    async def test_seed_deferred_when_embed_fails_at_startup_then_retries_on_resolve(self):
        pool = MagicMock()
        conn = AsyncMock()
        conn.fetchrow.return_value = None  # no alias
        conn.fetch.return_value = []  # empty search_predicates result
        pool.acquire.return_value.__aenter__.return_value = conn

        client = AsyncMock()
        # First attempt fails (startup); second succeeds (on first resolve).
        fail = RuntimeError("LLM 500")
        client.embed_batch.side_effect = [fail, [[0.1] * 768]]
        client.embed = AsyncMock(return_value=[0.2] * 768)

        store = EntityStore(pool=pool, embedding_client=client)
        store.set_predicate_seed([("http://x/causes", "causes")])

        # Startup-ish best-effort attempt — we expect False, no exception.
        ok = await store.ensure_predicates_seeded()
        assert ok is False
        assert store._predicate_seed_status is False

        # Next resolve triggers a retry, which succeeds.
        await store.resolve_predicate("causes")
        assert store._predicate_seed_status is True
        assert client.embed_batch.call_count == 2

    async def test_seed_is_no_op_when_spec_is_empty(self):
        pool = MagicMock()
        client = AsyncMock()
        store = EntityStore(pool=pool, embedding_client=client)
        # No set_predicate_seed call.
        ok = await store.ensure_predicates_seeded()
        assert ok is True
        client.embed_batch.assert_not_called()


@pytest.mark.asyncio
async def test_items_rejected_flows_into_job_record():
    """End-to-end smoke test: schema-invalid LLM output bumps items_rejected."""
    from knowledge_service.ingestion.phases import ExtractPhase

    bad = {"knowledge_type": "Entity"}  # missing required uri/label
    good = {
        "knowledge_type": "Entity",
        "uri": "x",
        "rdf_type": "schema:Thing",
        "label": "x",
        "properties": {},
        "confidence": 0.9,
    }

    # Build a real ExtractionClient, but intercept the HTTP layer.
    from knowledge_service.clients.llm import ExtractionClient

    xc = ExtractionClient(base_url="http://ignored", model="m", api_key="k")
    xc._call_llm_combined = AsyncMock(return_value=[bad, good])  # type: ignore[assignment]

    phase = ExtractPhase(xc)
    _, _, chunks_failed, chunks_skipped, items_rejected = await phase.run(
        chunk_records=[{"chunk_index": 0, "chunk_text": "t", "section_header": None}],
        chunk_id_map={0: "cid"},
        nlp_hints=None,
    )
    assert chunks_failed == 0
    assert chunks_skipped == 0
    assert items_rejected == 1
