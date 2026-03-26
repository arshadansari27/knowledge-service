# Async Content Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert synchronous content ingestion to async background processing with progress tracking and admin UI.

**Architecture:** `POST /api/content` returns 202 immediately after metadata upsert + chunking. A FastAPI background task handles embedding, extraction, and triple processing. Progress is tracked in a new `ingestion_jobs` PostgreSQL table with per-chunk granularity.

**Tech Stack:** FastAPI BackgroundTasks, asyncpg, Jinja2/htmx/Alpine.js (admin UI)

**Spec:** `docs/superpowers/specs/2026-03-26-async-content-ingestion-design.md`

---

## File Structure

| File | Purpose |
|------|---------|
| `migrations/008_ingestion_jobs.sql` | New table, indexes, trigger |
| `src/knowledge_service/models.py` | Add `ContentAcceptedResponse`, `IngestionJobStatus` (remove `ContentResponse` in Task 10) |
| `src/knowledge_service/clients/llm.py` | Add `batch_size` param to `embed_batch` |
| `src/knowledge_service/api/content.py` | Refactor to async 202 + background worker |
| `src/knowledge_service/admin/jobs.py` | New: `GET /api/admin/jobs` endpoint |
| `src/knowledge_service/admin/routes.py` | Add `/admin/jobs` page route |
| `src/knowledge_service/admin/templates/base.html` | Add "Jobs" nav item |
| `src/knowledge_service/admin/templates/jobs.html` | New: jobs list page |
| `src/knowledge_service/main.py` | Register jobs router, startup recovery |
| `tests/test_embed_subbatch.py` | New: embed sub-batching tests |
| `tests/test_api_content.py` | Update all tests for 202 + async flow |
| `tests/test_ingestion_jobs.py` | New: job status, admin jobs, chunk cap, background worker tests |

---

### Task 1: Migration — `ingestion_jobs` table

**Files:**
- Create: `migrations/008_ingestion_jobs.sql`

- [ ] **Step 1: Write the migration**

```sql
-- migrations/008_ingestion_jobs.sql
CREATE TABLE ingestion_jobs (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id        UUID        NOT NULL REFERENCES content_metadata(id) ON DELETE CASCADE,
    status            TEXT        NOT NULL DEFAULT 'accepted',
    chunks_total      INTEGER     NOT NULL DEFAULT 0,
    chunks_embedded   INTEGER     NOT NULL DEFAULT 0,
    chunks_extracted  INTEGER     NOT NULL DEFAULT 0,
    chunks_failed     INTEGER     NOT NULL DEFAULT 0,
    triples_created   INTEGER     NOT NULL DEFAULT 0,
    entities_resolved INTEGER     NOT NULL DEFAULT 0,
    chunks_capped_from INTEGER,
    error             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_ingestion_jobs_content ON ingestion_jobs (content_id);
CREATE INDEX idx_ingestion_jobs_status ON ingestion_jobs (status);
CREATE INDEX idx_ingestion_jobs_created ON ingestion_jobs (created_at DESC);

CREATE UNIQUE INDEX idx_ingestion_jobs_active ON ingestion_jobs (content_id)
    WHERE status NOT IN ('completed', 'failed');

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = now(); RETURN NEW; END; $$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ingestion_jobs_updated
    BEFORE UPDATE ON ingestion_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
```

- [ ] **Step 2: Commit**

```bash
git add migrations/008_ingestion_jobs.sql
git commit -m "feat: add ingestion_jobs migration"
```

---

### Task 2: Models — new response types

**Files:**
- Modify: `src/knowledge_service/models.py`

- [ ] **Step 1: Write failing test for new models**

Create test in `tests/test_ingestion_jobs.py`:

```python
"""Tests for async ingestion jobs."""

from knowledge_service.models import ContentAcceptedResponse, IngestionJobStatus


class TestContentAcceptedResponse:
    def test_minimal(self):
        r = ContentAcceptedResponse(content_id="abc", job_id="def", chunks_total=5)
        assert r.status == "accepted"
        assert r.chunks_capped_from is None

    def test_with_cap(self):
        r = ContentAcceptedResponse(
            content_id="abc", job_id="def", chunks_total=50, chunks_capped_from=337
        )
        assert r.chunks_capped_from == 337


class TestIngestionJobStatus:
    def test_all_fields(self):
        s = IngestionJobStatus(
            content_id="a", job_id="b", status="extracting",
            chunks_total=10, chunks_embedded=10, chunks_extracted=3,
            chunks_failed=0, triples_created=5, entities_resolved=2,
            error=None, created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:01:00Z",
        )
        assert s.status == "extracting"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingestion_jobs.py::TestContentAcceptedResponse -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Add models to `models.py`**

Add after the existing `ContentResponse` class:

```python
class ContentAcceptedResponse(BaseModel):
    content_id: str
    job_id: str
    status: str = "accepted"
    chunks_total: int
    chunks_capped_from: int | None = None


class IngestionJobStatus(BaseModel):
    content_id: str
    job_id: str
    status: str
    chunks_total: int
    chunks_embedded: int
    chunks_extracted: int
    chunks_failed: int
    triples_created: int
    entities_resolved: int
    error: str | None
    created_at: str
    updated_at: str
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingestion_jobs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/models.py tests/test_ingestion_jobs.py
git commit -m "feat: add ContentAcceptedResponse and IngestionJobStatus models"
```

---

### Task 3: Embedding sub-batching

**Files:**
- Modify: `src/knowledge_service/clients/llm.py:42-44`
- Create: `tests/test_embed_subbatch.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for EmbeddingClient.embed_batch sub-batching."""

import pytest
from unittest.mock import AsyncMock, patch

from knowledge_service.clients.llm import EmbeddingClient


@pytest.fixture
def embedding_client():
    return EmbeddingClient(base_url="http://localhost:11434", model="test", api_key="")


class TestEmbedBatchSubbatching:
    async def test_no_batch_size_sends_all_at_once(self, embedding_client):
        """Default behavior: all texts in one request."""
        embedding_client._request = AsyncMock(
            return_value=[[0.1] * 768 for _ in range(5)]
        )
        result = await embedding_client.embed_batch(["t1", "t2", "t3", "t4", "t5"])
        assert len(result) == 5
        embedding_client._request.assert_called_once()

    async def test_batch_size_splits_requests(self, embedding_client):
        """With batch_size=2, 5 texts → 3 calls (2+2+1)."""
        embedding_client._request = AsyncMock(
            side_effect=[
                [[0.1] * 768, [0.2] * 768],
                [[0.3] * 768, [0.4] * 768],
                [[0.5] * 768],
            ]
        )
        result = await embedding_client.embed_batch(
            ["t1", "t2", "t3", "t4", "t5"], batch_size=2
        )
        assert len(result) == 5
        assert embedding_client._request.call_count == 3

    async def test_batch_size_none_is_default(self, embedding_client):
        """batch_size=None behaves like no batching."""
        embedding_client._request = AsyncMock(
            return_value=[[0.1] * 768 for _ in range(3)]
        )
        result = await embedding_client.embed_batch(["a", "b", "c"], batch_size=None)
        assert len(result) == 3
        embedding_client._request.assert_called_once()

    async def test_batch_size_larger_than_input(self, embedding_client):
        """batch_size > len(texts) → one call."""
        embedding_client._request = AsyncMock(
            return_value=[[0.1] * 768, [0.2] * 768]
        )
        result = await embedding_client.embed_batch(["a", "b"], batch_size=100)
        assert len(result) == 2
        embedding_client._request.assert_called_once()

    async def test_empty_input(self, embedding_client):
        """Empty list returns empty list, no calls."""
        embedding_client._request = AsyncMock(return_value=[])
        result = await embedding_client.embed_batch([], batch_size=20)
        assert result == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embed_subbatch.py -v`
Expected: FAIL (batch_size parameter not accepted)

- [ ] **Step 3: Implement sub-batching in `embed_batch`**

Replace the existing `embed_batch` method in `EmbeddingClient`:

```python
async def embed_batch(
    self, texts: list[str], batch_size: int | None = None
) -> list[list[float]]:
    """Generate embeddings for multiple texts.

    When batch_size is set, splits into sub-batches to avoid overwhelming
    the embedding endpoint. Default (None) sends all texts in one request.
    """
    if not texts:
        return []
    if batch_size is None or batch_size >= len(texts):
        return await self._request(texts)
    results: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        results.extend(await self._request(batch))
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embed_subbatch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/clients/llm.py tests/test_embed_subbatch.py
git commit -m "feat: add batch_size parameter to EmbeddingClient.embed_batch"
```

---

### Task 4: Refactor `content.py` — async 202 + background worker

This is the largest task. The existing `_process_one_content_request` is split into:
- Synchronous part (metadata + chunking + job creation → 202)
- Background worker (embedding + extraction + processing → updates job)

**Files:**
- Modify: `src/knowledge_service/api/content.py`

- [ ] **Step 1: Add constants and imports**

At the top of `content.py`, update imports and constants:

```python
"""POST /api/content endpoint — ingest content with embedded knowledge."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from knowledge_service._utils import _is_uri, is_object_entity
from knowledge_service.api._ingest import apply_uri_fallback, process_triple
from knowledge_service.chunking import chunk_text as split_into_chunks
from knowledge_service.config import settings
from knowledge_service.models import (
    ContentAcceptedResponse,
    ContentRequest,
    expand_to_triples,
)
from knowledge_service.stores.provenance import ProvenanceStore

router = APIRouter()
logger = logging.getLogger(__name__)

_CHUNK_SIZE = 4000
_CHUNK_OVERLAP = 200
_MAX_CHUNKS = 50
_EMBED_BATCH_SIZE = 20
```

- [ ] **Step 2: Keep helper functions unchanged**

`_resolve_labels` and `_dedup_extracted_items` stay exactly as they are.

- [ ] **Step 3: Write the synchronous acceptance function**

Replace `_process_one_content_request` with a new `_accept_content_request` that returns fast:

```python
async def _accept_content_request(
    body: ContentRequest, pg_pool, embedding_store
) -> dict:
    """Synchronous phase: validate, upsert metadata, chunk, create job.

    Returns dict with content_id, job_id, chunks_total, chunks_capped_from,
    and chunk_records for the background worker.
    """
    # Step 1: Upsert content metadata
    content_id = await embedding_store.insert_content_metadata(
        url=body.url,
        title=body.title,
        summary=body.summary or "",
        raw_text=body.raw_text or "",
        source_type=body.source_type,
        tags=body.tags,
        metadata=body.metadata,
    )

    # Step 2: Chunk the text
    text = body.raw_text or body.summary or body.title
    raw_chunks = split_into_chunks(text, chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)

    chunk_records: list[dict] = []
    for i, rc in enumerate(raw_chunks):
        chunk_records.append(
            {
                "chunk_index": i,
                "chunk_text": rc["chunk_text"],
                "char_start": rc["char_start"],
                "char_end": rc["char_end"],
                "section_header": rc.get("section_header"),
            }
        )

    # Step 3: Cap chunks
    chunks_capped_from = None
    if len(chunk_records) > _MAX_CHUNKS:
        chunks_capped_from = len(chunk_records)
        logger.warning(
            "Capping %d chunks to %d for url=%s", len(chunk_records), _MAX_CHUNKS, body.url
        )
        chunk_records = chunk_records[:_MAX_CHUNKS]

    # Step 4: Check for active job (idempotency guard)
    async with pg_pool.acquire() as conn:
        active = await conn.fetchrow(
            """SELECT id FROM ingestion_jobs
               WHERE content_id = $1::uuid AND status NOT IN ('completed', 'failed')""",
            content_id,
        )
    if active:
        return {"conflict": True, "content_id": content_id}

    # Step 5: Create ingestion job
    async with pg_pool.acquire() as conn:
        job_row = await conn.fetchrow(
            """INSERT INTO ingestion_jobs (content_id, chunks_total, chunks_capped_from)
               VALUES ($1::uuid, $2, $3)
               RETURNING id""",
            content_id,
            len(chunk_records),
            chunks_capped_from,
        )
    job_id = str(job_row["id"])

    return {
        "conflict": False,
        "content_id": content_id,
        "job_id": job_id,
        "chunks_total": len(chunk_records),
        "chunks_capped_from": chunks_capped_from,
        "chunk_records": chunk_records,
    }
```

- [ ] **Step 4: Write the background worker function**

```python
async def _run_ingestion_worker(
    job_id: str,
    content_id: str,
    body: ContentRequest,
    chunk_records: list[dict],
    app_state,
) -> None:
    """Background worker: embed, extract, process triples, update job progress."""
    pg_pool = app_state.pg_pool
    embedding_client = app_state.embedding_client
    extraction_client = app_state.extraction_client
    knowledge_store = app_state.knowledge_store
    reasoning_engine = app_state.reasoning_engine
    embedding_store = getattr(app_state, "embedding_store", None)
    entity_resolver = getattr(app_state, "entity_resolver", None)
    provenance_store = ProvenanceStore(pg_pool)

    if embedding_store is None:
        from knowledge_service.stores.embedding import EmbeddingStore
        embedding_store = EmbeddingStore(pg_pool)

    current_phase = "embedding"
    try:
        # --- Phase 1: Embedding ---
        async with pg_pool.acquire() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET status = 'embedding' WHERE id = $1::uuid", job_id
            )

        texts = [c["chunk_text"] for c in chunk_records]
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[i : i + _EMBED_BATCH_SIZE]
            batch_embeddings = await embedding_client.embed_batch(batch)
            embeddings.extend(batch_embeddings)
            async with pg_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE ingestion_jobs SET chunks_embedded = $1 WHERE id = $2::uuid",
                    len(embeddings),
                    job_id,
                )
        for rec, emb in zip(chunk_records, embeddings):
            rec["embedding"] = emb

        # Null out old provenance chunk_ids, delete old chunks, insert new
        async with pg_pool.acquire() as conn:
            await conn.execute(
                """UPDATE provenance SET chunk_id = NULL
                   WHERE chunk_id IN (SELECT id FROM content WHERE content_id = $1::uuid)""",
                content_id,
            )
        await embedding_store.delete_chunks(content_id)
        chunk_id_pairs = await embedding_store.insert_chunks(content_id, chunk_records)
        chunk_id_map = dict(chunk_id_pairs) if chunk_id_pairs else {}

        # --- Phase 2: Extraction ---
        current_phase = "extracting"
        async with pg_pool.acquire() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET status = 'extracting' WHERE id = $1::uuid", job_id
            )

        chunks_failed = 0
        if not body.knowledge and body.raw_text:
            knowledge = []
            chunk_ids_for_items: list[str | None] = []
            for chunk in chunk_records:
                cid = chunk_id_map.get(chunk["chunk_index"])
                items = await extraction_client.extract(
                    chunk["chunk_text"], title=body.title, source_type=body.source_type
                )
                if items is None:
                    chunks_failed += 1
                    async with pg_pool.acquire() as conn:
                        await conn.execute(
                            """UPDATE ingestion_jobs
                               SET chunks_extracted = chunks_extracted + 1,
                                   chunks_failed = chunks_failed + 1
                               WHERE id = $1::uuid""",
                            job_id,
                        )
                    continue
                for item in items:
                    knowledge.append(item)
                    chunk_ids_for_items.append(cid)
                async with pg_pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE ingestion_jobs SET chunks_extracted = chunks_extracted + 1 WHERE id = $1::uuid",
                        job_id,
                    )
            knowledge, chunk_ids_for_items = _dedup_extracted_items(knowledge, chunk_ids_for_items)
            extracted_by_llm = bool(knowledge)
        else:
            knowledge = list(body.knowledge)
            chunk_ids_for_items = [None] * len(knowledge)
            extracted_by_llm = False
            # Mark all chunks as "extracted" (no LLM needed)
            async with pg_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE ingestion_jobs SET chunks_extracted = $1 WHERE id = $2::uuid",
                    len(chunk_records),
                    job_id,
                )

        extractor = f"llm_{settings.llm_chat_model}" if extracted_by_llm else "api"

        # --- Phase 3: Processing ---
        current_phase = "processing"
        async with pg_pool.acquire() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET status = 'processing' WHERE id = $1::uuid", job_id
            )

        # Resolve entity labels
        entities_resolved = 0
        if entity_resolver is not None:
            for i, item in enumerate(knowledge):
                count, knowledge[i] = await _resolve_labels(item, entity_resolver)
                entities_resolved += count

        # URI fallback
        for i, item in enumerate(knowledge):
            knowledge[i] = apply_uri_fallback(item)

        # Expand to triples and process
        triples_created = 0
        for i, item in enumerate(knowledge):
            for t in expand_to_triples(item):
                cid = chunk_ids_for_items[i] if i < len(chunk_ids_for_items) else None
                is_new, _contras = await process_triple(
                    t, knowledge_store, provenance_store, reasoning_engine,
                    body.url, body.source_type, extractor, chunk_id=cid,
                )
                if is_new:
                    triples_created += 1

        # Log ingestion event
        async with pg_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO ingestion_events (event_type, payload, source)
                   VALUES ($1, $2, $3)""",
                "content_ingested",
                json.dumps({"url": body.url, "triples_created": triples_created}),
                body.url,
            )

        # --- Done ---
        async with pg_pool.acquire() as conn:
            await conn.execute(
                """UPDATE ingestion_jobs
                   SET status = 'completed', triples_created = $1,
                       entities_resolved = $2, chunks_failed = $3
                   WHERE id = $4::uuid""",
                triples_created,
                entities_resolved,
                chunks_failed,
                job_id,
            )

    except Exception as exc:
        error_json = json.dumps({
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": current_phase,
        })
        try:
            async with pg_pool.acquire() as conn:
                await conn.execute(
                    "UPDATE ingestion_jobs SET status = 'failed', error = $1 WHERE id = $2::uuid",
                    error_json,
                    job_id,
                )
        except Exception:
            logger.exception("Failed to update job %s to failed status", job_id)
        logger.exception("Ingestion worker failed for job %s in phase %s", job_id, current_phase)
```

- [ ] **Step 5: Rewrite the `post_content` endpoint**

```python
@router.post("/content")
async def post_content(request: Request, background_tasks: BackgroundTasks):
    """Ingest content asynchronously.

    Accepts a single ContentRequest or a list for batch processing.
    Returns 202 Accepted with job info. Processing happens in background.
    """
    raw = await request.json()
    pg_pool = request.app.state.pg_pool
    embedding_store = getattr(request.app.state, "embedding_store", None)
    if embedding_store is None:
        from knowledge_service.stores.embedding import EmbeddingStore
        embedding_store = EmbeddingStore(pg_pool)

    try:
        if isinstance(raw, list):
            results = []
            for item_raw in raw:
                try:
                    body = ContentRequest(**item_raw)
                except ValidationError as exc:
                    results.append({"error": exc.errors(), "status_code": 422})
                    continue
                result = await _accept_content_request(body, pg_pool, embedding_store)
                if result.get("conflict"):
                    results.append({
                        "error": "Active job exists for this content",
                        "content_id": result["content_id"],
                        "status_code": 409,
                    })
                    continue
                background_tasks.add_task(
                    _run_ingestion_worker,
                    result["job_id"], result["content_id"], body,
                    result["chunk_records"], request.app.state,
                )
                results.append(ContentAcceptedResponse(
                    content_id=result["content_id"],
                    job_id=result["job_id"],
                    chunks_total=result["chunks_total"],
                    chunks_capped_from=result["chunks_capped_from"],
                ).model_dump())
            return JSONResponse(results, status_code=202)

        body = ContentRequest(**raw)
        result = await _accept_content_request(body, pg_pool, embedding_store)
        if result.get("conflict"):
            return JSONResponse(
                status_code=409,
                content={"detail": "Active ingestion job exists for this content"},
            )
        background_tasks.add_task(
            _run_ingestion_worker,
            result["job_id"], result["content_id"], body,
            result["chunk_records"], request.app.state,
        )
        return JSONResponse(
            ContentAcceptedResponse(
                content_id=result["content_id"],
                job_id=result["job_id"],
                chunks_total=result["chunks_total"],
                chunks_capped_from=result["chunks_capped_from"],
            ).model_dump(),
            status_code=202,
        )
    except ValidationError as exc:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
```

- [ ] **Step 6: Add the status endpoint**

```python
@router.get("/content/{content_id}/status")
async def get_content_status(content_id: str, request: Request):
    """Return the latest ingestion job status for a content item."""
    pg_pool = request.app.state.pg_pool
    async with pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id, content_id, status, chunks_total, chunks_embedded,
                      chunks_extracted, chunks_failed, triples_created,
                      entities_resolved, error, created_at, updated_at
               FROM ingestion_jobs
               WHERE content_id = $1::uuid
               ORDER BY created_at DESC LIMIT 1""",
            content_id,
        )
    if row is None:
        return JSONResponse(status_code=404, content={"detail": "No job found"})
    return {
        "content_id": str(row["content_id"]),
        "job_id": str(row["id"]),
        "status": row["status"],
        "chunks_total": row["chunks_total"],
        "chunks_embedded": row["chunks_embedded"],
        "chunks_extracted": row["chunks_extracted"],
        "chunks_failed": row["chunks_failed"],
        "triples_created": row["triples_created"],
        "entities_resolved": row["entities_resolved"],
        "error": row["error"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }
```

- [ ] **Step 7: Remove old `_MAX_CONCURRENT_EXTRACTIONS` and `_MAX_CONCURRENT_CONTENT` constants**

These are no longer needed — extraction is sequential, and batch mode no longer uses gather.

- [ ] **Step 8: Run lint**

Run: `uv run ruff check src/knowledge_service/api/content.py`
Expected: Clean (fix any issues)

- [ ] **Step 9: Commit**

```bash
git add src/knowledge_service/api/content.py
git commit -m "feat: refactor content ingestion to async 202 + background worker"
```

---

### Task 5: Update existing content tests

The existing tests in `test_api_content.py` expect 200 and `ContentResponse` fields. They need to be updated for 202 + `ContentAcceptedResponse`. The background worker runs inline during testing (ASGI transport processes background tasks before returning).

**Files:**
- Modify: `tests/test_api_content.py`

- [ ] **Step 1: Update pg_pool mock to support ingestion_jobs queries**

Update `_make_pg_pool_mock` to handle the new INSERT/SELECT queries on `ingestion_jobs`:

```python
def _make_pg_pool_mock():
    """Build a mock asyncpg pool whose .acquire() works as an async context manager."""
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "INSERT 0 1"

    async def _fetchrow(sql, *args):
        if "ingestion_jobs" in sql and "INSERT" in sql:
            return {"id": "job-uuid-1234"}
        if "ingestion_jobs" in sql and "SELECT" in sql and "status NOT IN" in sql:
            return None  # no active job (allow new ones)
        return {"id": "content-uuid-1234"}

    mock_conn.fetchrow.side_effect = _fetchrow
    mock_conn.fetch.return_value = []

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    mock_pool = MagicMock()
    mock_pool.acquire = _acquire
    return mock_pool
```

- [ ] **Step 2: Update status code assertions from 200 to 202**

In `TestPostContentBasic`:
- `test_returns_200` → `test_returns_202`: assert `response.status_code == 202`

In `TestPostContentExtraction`:
- `test_extraction_failure_yields_zero_triples`: change `assert response.status_code == 200` to `202`

In `TestContentChunking`:
- All `assert response.status_code == 200` → `202`

- [ ] **Step 3: Update response field assertions**

The 202 response returns `ContentAcceptedResponse` fields (`content_id`, `job_id`, `status`, `chunks_total`), not `triples_created` or `contradictions_detected`.

Update `TestPostContentBasic`:
```python
class TestPostContentBasic:
    async def test_returns_202(self, client):
        response = await client.post("/api/content", json=MINIMAL_PAYLOAD)
        assert response.status_code == 202

    async def test_response_has_content_id(self, client):
        response = await client.post("/api/content", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert "content_id" in data
        assert data["content_id"]

    async def test_response_has_job_id(self, client):
        response = await client.post("/api/content", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert "job_id" in data

    async def test_response_has_status_accepted(self, client):
        response = await client.post("/api/content", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert data["status"] == "accepted"

    async def test_response_has_chunks_total(self, client):
        response = await client.post("/api/content", json=MINIMAL_PAYLOAD)
        data = response.json()
        assert "chunks_total" in data
```

- [ ] **Step 4: Remove or update tests that depend on synchronous response fields**

`TestPostContentTripleCount` tests checked `triples_created` in the response — these fields are no longer in the 202 response. These tests should be removed or converted to test the background worker directly. Similarly for `TestPostContentContradictions`.

Remove classes: `TestPostContentTripleCount`, `TestPostContentContradictions`, `TestPostContentEntityResolution`.

The functionality they tested is now in the background worker and will be tested via the worker function directly in Task 6.

- [ ] **Step 5: Update batch tests**

```python
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
```

- [ ] **Step 6: Update chunking tests for 202**

Change all `assert response.status_code == 200` to `202` in `TestContentChunking`.

- [ ] **Step 7: Update e2e literal test**

`test_content_ingestion_preserves_literal_objects` needs to check 202 instead of 200 and verify background worker ran via mock assertions rather than response body.

- [ ] **Step 8: Run all tests**

Run: `uv run pytest tests/ -x -q`
Expected: All pass (or identify remaining failures to fix)

- [ ] **Step 9: Run lint**

Run: `uv run ruff check . && uv run ruff format --check .`

- [ ] **Step 10: Commit**

```bash
git add tests/test_api_content.py
git commit -m "test: update content tests for async 202 response"
```

---

### Task 6: Background worker tests + chunk cap test

**Files:**
- Modify: `tests/test_ingestion_jobs.py`

- [ ] **Step 1: Write background worker test**

```python
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from knowledge_service.api.content import _run_ingestion_worker
from knowledge_service.models import ContentRequest


def _make_app_state():
    """Build a mock app.state with all dependencies."""
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "UPDATE 1"
    mock_conn.fetchrow.return_value = None
    mock_conn.fetch.return_value = []

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    state = MagicMock()
    state.pg_pool = MagicMock()
    state.pg_pool.acquire = _acquire

    state.embedding_client = AsyncMock()
    state.embedding_client.embed_batch.return_value = [[0.1] * 768]

    state.extraction_client = AsyncMock()
    state.extraction_client.extract.return_value = []

    state.knowledge_store = MagicMock()
    state.knowledge_store.insert_triple.return_value = ("hash", True)
    state.knowledge_store.find_contradictions.return_value = []

    state.reasoning_engine = MagicMock()
    state.reasoning_engine.combine_evidence.return_value = 0.88

    state.embedding_store = AsyncMock()
    state.embedding_store.delete_chunks.return_value = None
    state.embedding_store.insert_chunks.return_value = [(0, "chunk-uuid-0")]

    state.entity_resolver = None
    return state, mock_conn


class TestIngestionWorker:
    async def test_worker_updates_status_to_completed(self):
        state, conn = _make_app_state()
        body = ContentRequest(
            url="http://test.com", title="Test", source_type="article",
            raw_text="Short text.",
        )
        chunks = [{"chunk_index": 0, "chunk_text": "Short text.",
                    "char_start": 0, "char_end": 11, "section_header": None}]

        await _run_ingestion_worker("job-1", "content-1", body, chunks, state)

        # Verify status transitions happened
        calls = [str(c) for c in conn.execute.call_args_list]
        statuses = [c for c in calls if "ingestion_jobs" in c and "status" in c]
        assert any("embedding" in s for s in statuses)
        assert any("completed" in s for s in statuses)

    async def test_worker_handles_extraction_failure(self):
        state, conn = _make_app_state()
        state.extraction_client.extract.return_value = None  # LLM failure

        body = ContentRequest(
            url="http://test.com", title="Test", source_type="article",
            raw_text="Some text.",
        )
        chunks = [{"chunk_index": 0, "chunk_text": "Some text.",
                    "char_start": 0, "char_end": 10, "section_header": None}]

        await _run_ingestion_worker("job-1", "content-1", body, chunks, state)

        calls = [str(c) for c in conn.execute.call_args_list]
        assert any("completed" in s for s in calls)
        assert any("chunks_failed" in s for s in calls)
```

- [ ] **Step 2: Write chunk cap test**

```python
from knowledge_service.api.content import _MAX_CHUNKS


class TestChunkCap:
    async def test_chunks_capped_at_max(self):
        """Verify _accept_content_request caps chunks at _MAX_CHUNKS."""
        from knowledge_service.api.content import _accept_content_request
        from knowledge_service.models import ContentRequest

        body = ContentRequest(
            url="http://test.com/big",
            title="Big doc",
            source_type="article",
            raw_text="A" * 300_000,  # Will produce many chunks
        )

        mock_conn = AsyncMock()
        mock_conn.fetchrow.side_effect = [
            None,  # no active job
            {"id": "job-uuid"},  # job insert
        ]

        @asynccontextmanager
        async def _acquire():
            yield mock_conn

        mock_pool = MagicMock()
        mock_pool.acquire = _acquire

        mock_es = AsyncMock()
        mock_es.insert_content_metadata.return_value = "content-uuid"

        result = await _accept_content_request(body, mock_pool, mock_es)

        assert not result["conflict"]
        assert result["chunks_total"] == _MAX_CHUNKS
        assert result["chunks_capped_from"] is not None
        assert result["chunks_capped_from"] > _MAX_CHUNKS
        assert len(result["chunk_records"]) == _MAX_CHUNKS
```

- [ ] **Step 3: Write status endpoint test**

```python
class TestContentStatusEndpoint:
    async def test_returns_latest_job(self):
        from datetime import datetime, timezone

        app = create_app(use_lifespan=False)
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            "id": "job-uuid",
            "content_id": "content-uuid",
            "status": "extracting",
            "chunks_total": 10,
            "chunks_embedded": 10,
            "chunks_extracted": 3,
            "chunks_failed": 0,
            "triples_created": 0,
            "entities_resolved": 0,
            "error": None,
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }

        @asynccontextmanager
        async def _acquire():
            yield mock_conn

        app.state.pg_pool = MagicMock()
        app.state.pg_pool.acquire = _acquire

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            resp = await client.get("/api/content/content-uuid/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "extracting"
        assert data["chunks_embedded"] == 10

    async def test_returns_404_when_no_job(self):
        app = create_app(use_lifespan=False)
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None

        @asynccontextmanager
        async def _acquire():
            yield mock_conn

        app.state.pg_pool = MagicMock()
        app.state.pg_pool.acquire = _acquire

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            resp = await client.get("/api/content/nonexistent-uuid/status")

        assert resp.status_code == 404
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ingestion_jobs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_ingestion_jobs.py
git commit -m "test: add background worker, chunk cap, and status endpoint tests"
```

---

### Task 7: Startup recovery + main.py registration

**Files:**
- Modify: `src/knowledge_service/main.py`

- [ ] **Step 1: Add startup recovery in `lifespan()`**

After `await run_migrations(app.state.pg_pool)`, add:

```python
    # Mark orphaned ingestion jobs as failed (lost on restart)
    async with app.state.pg_pool.acquire() as conn:
        updated = await conn.execute(
            """UPDATE ingestion_jobs SET status = 'failed',
                      error = '{"type": "ServiceRestart", "message": "interrupted by service restart", "phase": "unknown"}'
               WHERE status NOT IN ('completed', 'failed')"""
        )
        if updated != "UPDATE 0":
            logger.info("Marked orphaned ingestion jobs as failed: %s", updated)
```

- [ ] **Step 2: Register jobs admin router**

In `create_app()`, add after the existing admin router registrations:

```python
    from knowledge_service.admin.jobs import router as jobs_router
    app.include_router(jobs_router, prefix="/api/admin")
```

- [ ] **Step 3: Run lint**

Run: `uv run ruff check src/knowledge_service/main.py`

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_service/main.py
git commit -m "feat: add startup recovery for orphaned jobs, register jobs router"
```

---

### Task 8: Admin jobs API endpoint

**Files:**
- Create: `src/knowledge_service/admin/jobs.py`

- [ ] **Step 1: Write test**

Add to `tests/test_ingestion_jobs.py`:

```python
from httpx import AsyncClient, ASGITransport
from knowledge_service.main import create_app
from tests.conftest import make_test_session_cookie


def _make_admin_jobs_app(rows=None):
    """Helper: create test app with mocked pg_pool returning given rows."""
    app = create_app(use_lifespan=False)
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = rows or []

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    app.state.pg_pool = MagicMock()
    app.state.pg_pool.acquire = _acquire
    return app, mock_conn


class TestAdminJobsEndpoint:
    async def test_returns_empty_list(self):
        app, _ = _make_admin_jobs_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            resp = await client.get("/api/admin/jobs")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_status_filter_passed_to_query(self):
        app, mock_conn = _make_admin_jobs_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            await client.get("/api/admin/jobs?status=extracting")
        # Verify the status filter was passed to the query
        call_args = mock_conn.fetch.call_args
        assert "extracting" in call_args.args

    async def test_limit_parameter(self):
        app, mock_conn = _make_admin_jobs_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport, base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as client:
            await client.get("/api/admin/jobs?limit=10")
        call_args = mock_conn.fetch.call_args
        assert 10 in call_args.args
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ingestion_jobs.py::TestAdminJobsEndpoint -v`
Expected: FAIL (404, route not found)

- [ ] **Step 3: Implement `admin/jobs.py`**

```python
"""Admin API endpoint for ingestion jobs."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/jobs")
async def list_jobs(
    request: Request,
    limit: int = Query(default=50, le=200),
    status: str | None = Query(default=None),
):
    """List ingestion jobs in descending order."""
    pg_pool = request.app.state.pg_pool

    conditions = []
    params: list = []

    if status:
        params.append(status)
        conditions.append(f"j.status = ${len(params)}")

    params.append(limit)
    limit_placeholder = f"${len(params)}"

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    sql = f"""
        SELECT j.id, j.content_id, j.status, j.chunks_total,
               j.chunks_embedded, j.chunks_extracted, j.chunks_failed,
               j.triples_created, j.entities_resolved, j.chunks_capped_from,
               j.error, j.created_at, j.updated_at,
               m.url, m.title
        FROM ingestion_jobs j
        JOIN content_metadata m ON j.content_id = m.id
        {where}
        ORDER BY j.created_at DESC
        LIMIT {limit_placeholder}
    """

    async with pg_pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    return [
        {
            "id": str(r["id"]),
            "content_id": str(r["content_id"]),
            "status": r["status"],
            "chunks_total": r["chunks_total"],
            "chunks_embedded": r["chunks_embedded"],
            "chunks_extracted": r["chunks_extracted"],
            "chunks_failed": r["chunks_failed"],
            "triples_created": r["triples_created"],
            "entities_resolved": r["entities_resolved"],
            "chunks_capped_from": r["chunks_capped_from"],
            "error": r["error"],
            "url": r["url"],
            "title": r["title"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "updated_at": r["updated_at"].isoformat() if r["updated_at"] else None,
        }
        for r in rows
    ]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ingestion_jobs.py::TestAdminJobsEndpoint -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/admin/jobs.py tests/test_ingestion_jobs.py
git commit -m "feat: add GET /api/admin/jobs endpoint"
```

---

### Task 9: Admin UI — jobs page

**Files:**
- Modify: `src/knowledge_service/admin/templates/base.html`
- Create: `src/knowledge_service/admin/templates/jobs.html`
- Modify: `src/knowledge_service/admin/routes.py`

- [ ] **Step 1: Add "Jobs" to sidebar nav in `base.html`**

Add after the Contradictions nav item:

```html
<li><a href="/admin/jobs" class="block px-3 py-2 rounded hover:bg-gray-700 {% if active == 'jobs' %}bg-gray-700 text-white{% else %}text-gray-300{% endif %}">Jobs</a></li>
```

- [ ] **Step 2: Create `jobs.html` template**

```html
{% extends "base.html" %}
{% block title %}Ingestion Jobs{% endblock %}
{% block content %}
<h2 class="text-2xl font-bold mb-6">Ingestion Jobs</h2>
<div class="bg-gray-800 rounded-lg overflow-hidden"
     x-data="{ jobs: null, loading: true }"
     x-init="fetch('/api/admin/jobs?limit=100').then(r => r.json()).then(d => { jobs = d; loading = false }); setInterval(() => fetch('/api/admin/jobs?limit=100').then(r => r.json()).then(d => jobs = d), 5000)">
    <table class="w-full text-sm">
        <thead>
            <tr class="text-gray-400 border-b border-gray-700">
                <th class="text-left p-3">Status</th>
                <th class="text-left p-3">Title / URL</th>
                <th class="text-left p-3">Embedded</th>
                <th class="text-left p-3">Extracted</th>
                <th class="text-left p-3">Triples</th>
                <th class="text-left p-3">Created</th>
            </tr>
        </thead>
        <tbody>
            <template x-for="job in (jobs || [])" :key="job.id">
                <tr class="border-b border-gray-700/50 hover:bg-gray-700/30">
                    <td class="p-3">
                        <span class="px-2 py-0.5 rounded text-xs font-medium"
                              :class="{
                                  'bg-gray-600 text-gray-200': job.status === 'accepted',
                                  'bg-blue-600 text-white': job.status === 'embedding',
                                  'bg-yellow-600 text-white': job.status === 'extracting',
                                  'bg-purple-600 text-white': job.status === 'processing',
                                  'bg-green-600 text-white': job.status === 'completed',
                                  'bg-red-600 text-white': job.status === 'failed',
                              }"
                              x-text="job.status"></span>
                        <template x-if="job.chunks_capped_from">
                            <span class="ml-1 px-1.5 py-0.5 rounded text-xs bg-orange-700 text-orange-200"
                                  x-text="'capped from ' + job.chunks_capped_from"></span>
                        </template>
                    </td>
                    <td class="p-3">
                        <div class="text-white truncate max-w-xs" x-text="job.title || '(untitled)'"></div>
                        <div class="text-gray-500 text-xs truncate max-w-xs" x-text="job.url"></div>
                    </td>
                    <td class="p-3">
                        <span x-text="job.chunks_embedded + '/' + job.chunks_total"></span>
                    </td>
                    <td class="p-3">
                        <span x-text="job.chunks_extracted + '/' + job.chunks_total"></span>
                        <template x-if="job.chunks_failed > 0">
                            <span class="text-red-400 text-xs ml-1" x-text="'(' + job.chunks_failed + ' failed)'"></span>
                        </template>
                    </td>
                    <td class="p-3" x-text="job.triples_created"></td>
                    <td class="p-3 text-gray-400 text-xs" x-text="new Date(job.created_at).toLocaleString()"></td>
                </tr>
            </template>
            <template x-if="jobs && jobs.length === 0">
                <tr><td colspan="6" class="p-6 text-center text-gray-500">No ingestion jobs yet.</td></tr>
            </template>
        </tbody>
    </table>
</div>
{% endblock %}
```

- [ ] **Step 3: Add route in `routes.py`**

```python
@router.get("/admin/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    return templates.TemplateResponse(request, "jobs.html", {"active": "jobs"})
```

- [ ] **Step 4: Run lint**

Run: `uv run ruff check . && uv run ruff format --check .`

- [ ] **Step 5: Run all tests**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/admin/templates/base.html src/knowledge_service/admin/templates/jobs.html src/knowledge_service/admin/routes.py
git commit -m "feat: add admin jobs page with live progress tracking"
```

---

### Task 10: Final verification + cleanup

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All pass

- [ ] **Step 2: Run lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`

- [ ] **Step 3: Remove `ContentResponse` from `models.py` and all imports**

Delete the `ContentResponse` class from `models.py`. Grep for and remove any remaining imports:

Run: `grep -rn "ContentResponse" src/ tests/`

Fix all found references. Also verify old constants are gone:

Run: `grep -rn "_MAX_CONCURRENT" src/`

- [ ] **Step 4: Final commit if needed**

```bash
git add -A && git commit -m "chore: cleanup dead code and unused imports"
```

- [ ] **Step 5: Create PR**

```bash
gh pr create --title "feat: async content ingestion with background processing" --body "..."
```
