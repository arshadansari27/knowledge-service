# Chunking Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance `/api/content` to chunk long text into individually-embedded segments, and update `/api/search` + `/api/ask` to retrieve chunk-level results for more precise RAG.

**Architecture:** Text arriving via `/api/content` is split into ~4000-char chunks with 200-char overlap using `langchain-text-splitters`. Each chunk gets its own embedding in a new `content_chunks` table (pgvector HNSW + halfvec, same pattern as `content`). Search queries `content_chunks` first, falling back to `content` for un-chunked documents. The RAG retriever uses the same chunk-level search.

**Tech Stack:** Python 3.12, FastAPI, asyncpg, pgvector, langchain-text-splitters, pytest

**Spec:** `docs/superpowers/specs/2026-03-20-chunking-enhancement-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `migrations/002_content_chunks.sql` | DDL for `content_chunks` table + indexes |
| Modify | `pyproject.toml` | Add `langchain-text-splitters` dependency |
| Modify | `src/knowledge_service/stores/embedding.py` | Add `insert_chunks()`, `search_chunks()`, `delete_chunks_by_content_id()` |
| Modify | `src/knowledge_service/api/content.py` | Add chunking + chunk embedding after content upsert |
| Modify | `src/knowledge_service/api/search.py` | Query chunks first, fall back to content |
| Modify | `src/knowledge_service/models.py` | Add `chunk_text` and `chunk_index` to `SearchResult` |
| Modify | `src/knowledge_service/stores/rag.py` | Use `search_chunks()` instead of `search()` |
| Modify | `tests/test_embedding_store.py` | Tests for new EmbeddingStore methods |
| Create | `tests/test_chunking.py` | Tests for chunking logic in content ingestion |
| Modify | `tests/test_api_search.py` | Tests for chunk-level search results |
| Modify | `tests/test_rag_retriever.py` | Tests for chunk-aware RAG retrieval |

---

## Task 1: Migration — `content_chunks` table

**Files:**
- Create: `migrations/002_content_chunks.sql`

- [ ] **Step 1: Write the migration SQL**

```sql
-- 002_content_chunks.sql
-- Chunk-level storage for content documents.
-- Each chunk gets its own embedding for fine-grained RAG retrieval.

CREATE TABLE IF NOT EXISTS content_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL REFERENCES content(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(768),
    char_start INTEGER,
    char_end INTEGER,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(content_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON content_chunks
    USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_content_id ON content_chunks (content_id);
```

- [ ] **Step 2: Verify migration file is valid SQL**

Run: `cat migrations/002_content_chunks.sql`
Expected: The SQL above, no syntax errors.

- [ ] **Step 3: Commit**

```bash
git add migrations/002_content_chunks.sql
git commit -m "feat: add content_chunks migration (002)"
```

---

## Task 2: Add `langchain-text-splitters` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependency to pyproject.toml**

In `pyproject.toml`, add `"langchain-text-splitters>=0.3.0"` to the `dependencies` list (after `"pydantic-settings>=2.7.0"`).

- [ ] **Step 2: Sync the environment**

Run: `uv sync --dev`
Expected: Resolves and installs `langchain-text-splitters` successfully.

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "from langchain_text_splitters import RecursiveCharacterTextSplitter; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add langchain-text-splitters dependency"
```

---

## Task 3: EmbeddingStore — chunk methods (TDD)

**Files:**
- Modify: `src/knowledge_service/stores/embedding.py`
- Modify: `tests/test_embedding_store.py`

### 3a: `delete_chunks_by_content_id()`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_embedding_store.py`:

```python
class TestDeleteChunksByContentId:
    async def test_calls_execute_with_delete(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "DELETE 3"
        await store.delete_chunks_by_content_id("content-uuid-123")
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "DELETE FROM content_chunks" in sql
        assert "content_id" in sql

    async def test_passes_content_id_param(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "DELETE 0"
        await store.delete_chunks_by_content_id("my-uuid")
        args = conn.execute.call_args[0]
        assert "my-uuid" in args
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedding_store.py::TestDeleteChunksByContentId -v`
Expected: FAIL — `AttributeError: 'EmbeddingStore' object has no attribute 'delete_chunks_by_content_id'`

- [ ] **Step 3: Write minimal implementation**

Add to `EmbeddingStore` in `src/knowledge_service/stores/embedding.py`, after the `search()` method (around line 153), in a new section:

```python
    # ------------------------------------------------------------------
    # Content chunks table operations
    # ------------------------------------------------------------------

    async def delete_chunks_by_content_id(self, content_id: str) -> None:
        """Delete all chunks for a given content_id."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM content_chunks WHERE content_id = $1",
                content_id,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embedding_store.py::TestDeleteChunksByContentId -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/stores/embedding.py tests/test_embedding_store.py
git commit -m "feat: add EmbeddingStore.delete_chunks_by_content_id()"
```

### 3b: `insert_chunks()`

- [ ] **Step 6: Write the failing test**

Append to `tests/test_embedding_store.py`:

```python
class TestInsertChunks:
    async def test_inserts_multiple_chunks(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "INSERT 0 1"
        chunks = [
            {"chunk_index": 0, "text": "First chunk", "embedding": [0.1] * 768, "char_start": 0, "char_end": 100},
            {"chunk_index": 1, "text": "Second chunk", "embedding": [0.2] * 768, "char_start": 80, "char_end": 200},
        ]
        await store.insert_chunks("content-uuid-123", chunks)
        assert conn.execute.call_count == 2

    async def test_sql_targets_content_chunks(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "INSERT 0 1"
        chunks = [
            {"chunk_index": 0, "text": "chunk", "embedding": [0.1] * 768, "char_start": 0, "char_end": 50},
        ]
        await store.insert_chunks("uuid-1", chunks)
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO content_chunks" in sql

    async def test_passes_content_id(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "INSERT 0 1"
        chunks = [
            {"chunk_index": 0, "text": "chunk", "embedding": [0.1] * 768, "char_start": 0, "char_end": 50},
        ]
        await store.insert_chunks("my-content-id", chunks)
        args = conn.execute.call_args[0]
        assert "my-content-id" in args

    async def test_no_chunks_no_execute(self, store, mock_pool):
        _, conn = mock_pool
        await store.insert_chunks("uuid-1", [])
        conn.execute.assert_not_called()
```

- [ ] **Step 7: Run test to verify it fails**

Run: `uv run pytest tests/test_embedding_store.py::TestInsertChunks -v`
Expected: FAIL — `AttributeError: 'EmbeddingStore' object has no attribute 'insert_chunks'`

- [ ] **Step 8: Write minimal implementation**

Add to `EmbeddingStore`, after `delete_chunks_by_content_id()`:

```python
    async def insert_chunks(
        self,
        content_id: str,
        chunks: list[dict],
    ) -> None:
        """Insert chunk rows for a content document.

        Each dict in chunks must have: chunk_index, text, embedding, char_start, char_end.
        """
        if not chunks:
            return

        sql = """
            INSERT INTO content_chunks (
                content_id, chunk_index, text, embedding, char_start, char_end
            )
            VALUES ($1, $2, $3, $4::vector(768), $5, $6)
        """

        async with self._pool.acquire() as conn:
            for chunk in chunks:
                embedding_str = self._vector_to_str(chunk["embedding"])
                await conn.execute(
                    sql,
                    content_id,
                    chunk["chunk_index"],
                    chunk["text"],
                    embedding_str,
                    chunk["char_start"],
                    chunk["char_end"],
                )
```

- [ ] **Step 9: Run test to verify it passes**

Run: `uv run pytest tests/test_embedding_store.py::TestInsertChunks -v`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add src/knowledge_service/stores/embedding.py tests/test_embedding_store.py
git commit -m "feat: add EmbeddingStore.insert_chunks()"
```

### 3c: `search_chunks()`

- [ ] **Step 11: Write the failing test**

Append to `tests/test_embedding_store.py`:

```python
class TestSearchChunks:
    async def test_returns_chunk_results(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {
                "chunk_id": "chunk-uuid-1",
                "text": "relevant chunk text",
                "chunk_index": 2,
                "content_id": "content-uuid-1",
                "url": "https://example.com/article",
                "title": "Test Article",
                "source_type": "article",
                "tags": ["test"],
                "ingested_at": "2025-01-01",
                "similarity": 0.93,
            }
        ]
        results = await store.search_chunks(
            query_embedding=[0.1] * 768,
            limit=10,
        )
        assert len(results) == 1
        assert results[0]["chunk_id"] == "chunk-uuid-1"
        assert results[0]["text"] == "relevant chunk text"
        assert results[0]["chunk_index"] == 2
        assert results[0]["content_id"] == "content-uuid-1"

    async def test_sql_joins_content_table(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search_chunks(query_embedding=[0.1] * 768, limit=5)
        sql = conn.fetch.call_args[0][0]
        assert "content_chunks" in sql
        assert "JOIN content" in sql or "join content" in sql.lower()
        assert "<=>" in sql
        assert "halfvec" in sql

    async def test_with_source_type_filter(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search_chunks(
            query_embedding=[0.1] * 768,
            limit=5,
            source_type="article",
        )
        sql = conn.fetch.call_args[0][0]
        assert "source_type" in sql

    async def test_with_tags_filter(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search_chunks(
            query_embedding=[0.1] * 768,
            limit=5,
            tags=["python"],
        )
        sql = conn.fetch.call_args[0][0]
        assert "tags" in sql

    async def test_returns_empty_list(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        results = await store.search_chunks(query_embedding=[0.1] * 768, limit=10)
        assert results == []

    async def test_passes_limit_param(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search_chunks(query_embedding=[0.1] * 768, limit=42)
        args = conn.fetch.call_args[0]
        assert 42 in args
```

- [ ] **Step 12: Run test to verify it fails**

Run: `uv run pytest tests/test_embedding_store.py::TestSearchChunks -v`
Expected: FAIL — `AttributeError: 'EmbeddingStore' object has no attribute 'search_chunks'`

- [ ] **Step 13: Write minimal implementation**

Add to `EmbeddingStore`, after `insert_chunks()`:

```python
    async def search_chunks(
        self,
        query_embedding: list[float],
        limit: int,
        source_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict]:
        """Return chunk rows ranked by cosine similarity, joined with parent content metadata.

        Same filter interface as search() but queries content_chunks table.
        """
        embedding_str = self._vector_to_str(query_embedding)

        conditions: list[str] = []
        params: list[Any] = [embedding_str]

        if source_type is not None:
            params.append(source_type)
            conditions.append(f"p.source_type = ${len(params)}")

        if tags is not None:
            params.append(tags)
            conditions.append(f"p.tags @> ${len(params)}")

        params.append(limit)
        limit_placeholder = f"${len(params)}"

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = f"""
            SELECT
                c.chunk_id, c.text, c.chunk_index,
                p.id AS content_id, p.url, p.title, p.source_type, p.tags, p.ingested_at,
                1 - (c.embedding::halfvec(768) <=> $1::halfvec(768)) AS similarity
            FROM content_chunks c
            JOIN content p ON c.content_id = p.id
            {where_clause}
            ORDER BY c.embedding::halfvec(768) <=> $1::halfvec(768)
            LIMIT {limit_placeholder}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [dict(row) for row in rows]
```

- [ ] **Step 14: Run test to verify it passes**

Run: `uv run pytest tests/test_embedding_store.py::TestSearchChunks -v`
Expected: PASS

- [ ] **Step 15: Commit**

```bash
git add src/knowledge_service/stores/embedding.py tests/test_embedding_store.py
git commit -m "feat: add EmbeddingStore.search_chunks()"
```

---

## Task 4: `SearchResult` model — add chunk fields

**Files:**
- Modify: `src/knowledge_service/models.py:316-324`

- [ ] **Step 1: Add optional chunk fields to SearchResult**

In `src/knowledge_service/models.py`, modify the `SearchResult` class (line 316-324) to add two optional fields:

```python
class SearchResult(BaseModel):
    content_id: str
    url: str
    title: str
    summary: str | None
    similarity: float
    source_type: str
    tags: list[str]
    ingested_at: datetime
    chunk_text: str | None = None
    chunk_index: int | None = None
```

- [ ] **Step 2: Run existing search tests to ensure backward compat**

Run: `uv run pytest tests/test_api_search.py -v`
Expected: All existing tests PASS (new fields have defaults, so no breakage).

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_service/models.py
git commit -m "feat: add chunk_text, chunk_index to SearchResult model"
```

---

## Task 5: Content ingestion — chunking step (TDD)

**Files:**
- Modify: `src/knowledge_service/api/content.py`
- Create: `tests/test_chunking.py`

### 5a: Tests for chunking during content ingestion

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chunking.py`:

```python
"""Tests for content chunking during ingestion."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from knowledge_service.main import create_app
from tests.conftest import make_test_session_cookie


def _make_pg_pool_mock():
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


def _make_embedding_client_mock():
    mock = AsyncMock()
    mock.embed.return_value = [0.1] * 768
    mock.embed_batch.return_value = [[0.1] * 768, [0.2] * 768]
    return mock


def _make_embedding_store_mock():
    mock = AsyncMock()
    mock.insert_content.return_value = "content-uuid-1234"
    mock.delete_chunks_by_content_id.return_value = None
    mock.insert_chunks.return_value = None
    return mock


def _make_knowledge_store_mock():
    mock = MagicMock()
    mock.insert_triple.return_value = ("abc123", True)
    mock.find_contradictions.return_value = []
    return mock


def _make_extraction_client_mock():
    mock = AsyncMock()
    mock.extract.return_value = []
    return mock


def _make_reasoning_engine_mock():
    mock = MagicMock()
    mock.combine_evidence.return_value = 0.88
    return mock


SHORT_TEXT_PAYLOAD = {
    "url": "https://example.com/short",
    "title": "Short Article",
    "raw_text": "This is a short article under 4000 characters.",
    "source_type": "article",
}

LONG_TEXT_PAYLOAD = {
    "url": "https://example.com/long",
    "title": "Long Article",
    "raw_text": "A" * 5000,
    "source_type": "article",
}


class TestChunkingShortContent:
    """Short content (< 4000 chars) should NOT be chunked."""

    async def test_no_chunks_inserted_for_short_text(self):
        app = create_app(use_lifespan=False)
        mock_es = _make_embedding_store_mock()
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock()
        app.state.embedding_client = _make_embedding_client_mock()
        app.state.extraction_client = _make_extraction_client_mock()
        app.state.reasoning_engine = _make_reasoning_engine_mock()
        app.state.embedding_store = mock_es

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post("/api/content", json=SHORT_TEXT_PAYLOAD)

        assert response.status_code == 200
        mock_es.insert_chunks.assert_not_called()

    async def test_no_batch_embed_for_short_text(self):
        app = create_app(use_lifespan=False)
        mock_ec = _make_embedding_client_mock()
        mock_es = _make_embedding_store_mock()
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock()
        app.state.embedding_client = mock_ec
        app.state.extraction_client = _make_extraction_client_mock()
        app.state.reasoning_engine = _make_reasoning_engine_mock()
        app.state.embedding_store = mock_es

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=SHORT_TEXT_PAYLOAD)

        mock_ec.embed_batch.assert_not_called()


class TestChunkingLongContent:
    """Long content (>= 4000 chars) should be chunked."""

    async def test_chunks_inserted_for_long_text(self):
        app = create_app(use_lifespan=False)
        mock_es = _make_embedding_store_mock()
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock()
        app.state.embedding_client = _make_embedding_client_mock()
        app.state.extraction_client = _make_extraction_client_mock()
        app.state.reasoning_engine = _make_reasoning_engine_mock()
        app.state.embedding_store = mock_es

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post("/api/content", json=LONG_TEXT_PAYLOAD)

        assert response.status_code == 200
        mock_es.delete_chunks_by_content_id.assert_called_once_with("content-uuid-1234")
        mock_es.insert_chunks.assert_called_once()
        call_args = mock_es.insert_chunks.call_args
        assert call_args[0][0] == "content-uuid-1234"
        chunks = call_args[0][1]
        assert len(chunks) >= 2
        for chunk in chunks:
            assert "chunk_index" in chunk
            assert "text" in chunk
            assert "embedding" in chunk
            assert "char_start" in chunk
            assert "char_end" in chunk

    async def test_embed_batch_called_for_chunks(self):
        app = create_app(use_lifespan=False)
        mock_ec = _make_embedding_client_mock()
        mock_es = _make_embedding_store_mock()
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock()
        app.state.embedding_client = mock_ec
        app.state.extraction_client = _make_extraction_client_mock()
        app.state.reasoning_engine = _make_reasoning_engine_mock()
        app.state.embedding_store = mock_es

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=LONG_TEXT_PAYLOAD)

        mock_ec.embed_batch.assert_called_once()
        batch_texts = mock_ec.embed_batch.call_args[0][0]
        assert len(batch_texts) >= 2


class TestChunkingNoRawText:
    """Content with no raw_text should not attempt chunking."""

    async def test_no_chunks_when_no_raw_text(self):
        app = create_app(use_lifespan=False)
        mock_es = _make_embedding_store_mock()
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock()
        app.state.embedding_client = _make_embedding_client_mock()
        app.state.extraction_client = _make_extraction_client_mock()
        app.state.reasoning_engine = _make_reasoning_engine_mock()
        app.state.embedding_store = mock_es

        payload = {
            "url": "https://example.com/no-text",
            "title": "No Text",
            "source_type": "article",
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.post("/api/content", json=payload)

        assert response.status_code == 200
        mock_es.insert_chunks.assert_not_called()


class TestChunkingReIngestion:
    """Re-ingesting content (URL conflict) should delete old chunks first."""

    async def test_old_chunks_deleted_on_reingestion(self):
        app = create_app(use_lifespan=False)
        mock_es = _make_embedding_store_mock()
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock()
        app.state.embedding_client = _make_embedding_client_mock()
        app.state.extraction_client = _make_extraction_client_mock()
        app.state.reasoning_engine = _make_reasoning_engine_mock()
        app.state.embedding_store = mock_es

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.post("/api/content", json=LONG_TEXT_PAYLOAD)

        mock_es.delete_chunks_by_content_id.assert_called_once_with("content-uuid-1234")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL — chunking logic doesn't exist yet, so `insert_chunks` is never called.

- [ ] **Step 3: Implement chunking in content.py**

Modify `src/knowledge_service/api/content.py`. Add the import at the top (after existing imports):

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

Add a module-level constant after the `router = APIRouter()` line:

```python
_CHUNK_SIZE = 4000
_CHUNK_OVERLAP = 200
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

In `_process_one_content_request()`, insert the chunking step **after** step 2 (content upsert, line 74) and **before** step 2.5 (auto-extraction, line 77). The new code:

```python
    # Step 2.1: Chunk long text and embed chunks
    if body.raw_text and len(body.raw_text) >= _CHUNK_SIZE:
        chunks_text = _splitter.split_text(body.raw_text)

        # Track char offsets for each chunk
        chunk_records = []
        search_start = 0
        for i, chunk_text in enumerate(chunks_text):
            char_start = body.raw_text.find(chunk_text[:100], search_start)
            if char_start == -1:
                char_start = search_start
            char_end = char_start + len(chunk_text)
            search_start = max(search_start, char_start + 1)
            chunk_records.append({
                "chunk_index": i,
                "text": chunk_text,
                "char_start": char_start,
                "char_end": char_end,
            })

        # Batch embed all chunks
        chunk_embeddings = await embedding_client.embed_batch(
            [c["text"] for c in chunk_records]
        )
        for rec, emb in zip(chunk_records, chunk_embeddings):
            rec["embedding"] = emb

        # Delete old chunks (re-ingestion) and insert new
        await embedding_store.delete_chunks_by_content_id(content_id)
        await embedding_store.insert_chunks(content_id, chunk_records)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: All PASS

- [ ] **Step 5: Run existing content tests to ensure no regression**

Run: `uv run pytest tests/test_api_content.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/api/content.py tests/test_chunking.py
git commit -m "feat: chunk long content and embed chunks during ingestion"
```

---

## Task 6: Search endpoint — chunk-level results (TDD)

**Files:**
- Modify: `src/knowledge_service/api/search.py`
- Modify: `tests/test_api_search.py`

### 6a: Tests for chunk-aware search

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api_search.py`:

```python
# ---------------------------------------------------------------------------
# Tests: chunk-level search results
# ---------------------------------------------------------------------------

_CHUNK_ROW = {
    "chunk_id": "chunk-uuid-1",
    "text": "The relevant chunk text",
    "chunk_index": 2,
    "content_id": "content-uuid-1234",
    "url": "https://example.com/article",
    "title": "Test Article",
    "source_type": "article",
    "tags": ["python"],
    "ingested_at": _NOW,
    "similarity": 0.95,
}

_UNCHUNKED_ROW = {
    "id": "content-uuid-5678",
    "url": "https://example.com/short",
    "title": "Short Article",
    "summary": "A short summary",
    "source_type": "article",
    "tags": ["test"],
    "ingested_at": _NOW,
    "similarity": 0.80,
}


class TestGetSearchChunks:
    async def test_chunk_result_has_chunk_fields(self):
        """When chunks exist, results should include chunk_text and chunk_index."""
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.embedding_client = _make_embedding_client_mock()
        mock_es = AsyncMock()
        mock_es.search_chunks.return_value = [_CHUNK_ROW]
        mock_es.search.return_value = []
        app.state.embedding_store = mock_es

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/search", params={"q": "relevant query"})

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        chunk_result = data[0]
        assert chunk_result["chunk_text"] == "The relevant chunk text"
        assert chunk_result["chunk_index"] == 2
        assert chunk_result["content_id"] == "content-uuid-1234"

    async def test_unchunked_result_has_null_chunk_fields(self):
        """Un-chunked content results should have null chunk_text and chunk_index."""
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.embedding_client = _make_embedding_client_mock()
        mock_es = AsyncMock()
        mock_es.search_chunks.return_value = []
        mock_es.search.return_value = [_UNCHUNKED_ROW]
        app.state.embedding_store = mock_es

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/search", params={"q": "short query"})

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        result = data[0]
        assert result["chunk_text"] is None
        assert result["chunk_index"] is None

    async def test_mixed_chunk_and_content_results(self):
        """Both chunk-level and content-level results can appear together."""
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.embedding_client = _make_embedding_client_mock()
        mock_es = AsyncMock()
        mock_es.search_chunks.return_value = [_CHUNK_ROW]
        mock_es.search.return_value = [_UNCHUNKED_ROW]
        app.state.embedding_store = mock_es

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/search", params={"q": "mixed query", "limit": "10"})

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
```

Note: You'll also need to add `from unittest.mock import AsyncMock` to the imports at the top of the test file (if not already imported).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_search.py::TestGetSearchChunks -v`
Expected: FAIL — search endpoint doesn't call `search_chunks()` yet.

- [ ] **Step 3: Rewrite search endpoint to use chunks**

Replace the body of `get_search()` in `src/knowledge_service/api/search.py`:

```python
"""GET /api/search endpoint — semantic similarity search over ingested content."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from knowledge_service.models import SearchResult
from knowledge_service.stores.embedding import EmbeddingStore

router = APIRouter()


@router.get("/search", response_model=list[SearchResult])
async def get_search(
    request: Request,
    q: str = Query(..., description="Search query text"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of results to return"),
    source_type: str | None = Query(None, description="Filter by source type"),
    tags: list[str] | None = Query(None, description="Filter by tags (all must match)"),
) -> list[SearchResult]:
    """Search ingested content by semantic similarity.

    Queries chunk-level embeddings first, then falls back to content-level
    for documents that were not chunked.
    """
    embedding_client = request.app.state.embedding_client
    embedding_store = getattr(request.app.state, "embedding_store", None)

    if embedding_store is None:
        pg_pool = request.app.state.pg_pool
        embedding_store = EmbeddingStore(pg_pool)

    embedding = await embedding_client.embed(q)

    # Chunk-level search
    chunk_rows = await embedding_store.search_chunks(
        query_embedding=embedding,
        limit=limit,
        source_type=source_type,
        tags=tags,
    )

    results: list[SearchResult] = [
        SearchResult(
            content_id=str(row["content_id"]),
            url=row["url"],
            title=row["title"],
            summary=None,
            similarity=float(row["similarity"]),
            source_type=row["source_type"],
            tags=list(row["tags"]) if row["tags"] else [],
            ingested_at=row["ingested_at"],
            chunk_text=row["text"],
            chunk_index=row["chunk_index"],
        )
        for row in chunk_rows
    ]

    # Fallback: content-level search for un-chunked documents
    remaining = limit - len(results)
    if remaining > 0:
        content_rows = await embedding_store.search(
            query_embedding=embedding,
            limit=remaining,
            source_type=source_type,
            tags=tags,
        )

        # Exclude content IDs already covered by chunks
        seen_content_ids = {r.content_id for r in results}
        for row in content_rows:
            cid = str(row["id"])
            if cid not in seen_content_ids:
                results.append(
                    SearchResult(
                        content_id=cid,
                        url=row["url"],
                        title=row["title"],
                        summary=row.get("summary"),
                        similarity=float(row["similarity"]),
                        source_type=row["source_type"],
                        tags=list(row["tags"]) if row["tags"] else [],
                        ingested_at=row["ingested_at"],
                    )
                )

    # Sort combined results by similarity descending
    results.sort(key=lambda r: r.similarity, reverse=True)

    return results[:limit]
```

- [ ] **Step 4: Run chunk search tests**

Run: `uv run pytest tests/test_api_search.py::TestGetSearchChunks -v`
Expected: All PASS

- [ ] **Step 5: Fix existing search tests for new flow**

The existing tests use `app.state.pg_pool` directly, but the new search endpoint uses `app.state.embedding_store`. The search endpoint now calls `embedding_store.search_chunks()` then `embedding_store.search()` instead of going through the raw pool. Every test that creates its own app instance must set `app.state.embedding_store` with a mock that has both `search_chunks` and `search` methods.

**Add `AsyncMock` to imports** at the top of `tests/test_api_search.py`:
```python
from unittest.mock import AsyncMock, MagicMock
```

**Add a helper** after the existing helpers:
```python
def _make_embedding_store_mock(search_rows=None, chunk_rows=None):
    """Build a mock EmbeddingStore for search tests."""
    mock = AsyncMock()
    mock.search_chunks.return_value = chunk_rows or []
    mock.search.return_value = search_rows or []
    return mock
```

**Replace `client` fixture:**
```python
@pytest.fixture
async def client():
    app = create_app(use_lifespan=False)
    app.state.knowledge_store = _make_knowledge_store_mock()
    app.state.pg_pool = _make_pg_pool_mock()
    app.state.embedding_client = _make_embedding_client_mock()
    app.state.embedding_store = _make_embedding_store_mock(search_rows=[_SAMPLE_ROW])

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        cookies={"ks_session": make_test_session_cookie()},
    ) as c:
        yield c
```

**Replace `empty_client` fixture:**
```python
@pytest.fixture
async def empty_client():
    app = create_app(use_lifespan=False)
    app.state.knowledge_store = _make_knowledge_store_mock()
    app.state.pg_pool = _make_pg_pool_mock(rows=[])
    app.state.embedding_client = _make_embedding_client_mock()
    app.state.embedding_store = _make_embedding_store_mock()

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        cookies={"ks_session": make_test_session_cookie()},
    ) as c:
        yield c
```

**Update `TestGetSearchSimilarity.test_multiple_results_have_similarity`** — replace the test body:
```python
    async def test_multiple_results_have_similarity(self):
        rows = [
            {**_SAMPLE_ROW, "id": "uuid-1", "url": "https://example.com/1", "similarity": 0.95},
            {**_SAMPLE_ROW, "id": "uuid-2", "url": "https://example.com/2", "similarity": 0.80},
        ]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(rows=rows)
        app.state.embedding_client = _make_embedding_client_mock()
        app.state.embedding_store = _make_embedding_store_mock(search_rows=rows)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/search", params={"q": "multiple"})

        data = response.json()
        assert len(data) == 2
        assert data[0]["similarity"] == pytest.approx(0.95, abs=1e-6)
        assert data[1]["similarity"] == pytest.approx(0.80, abs=1e-6)
```

**Update `TestGetSearchValidation.test_default_limit_is_ten`** — add `embedding_store`:
```python
    async def test_default_limit_is_ten(self):
        app = create_app(use_lifespan=False)
        mock_ec = _make_embedding_client_mock()
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(rows=[])
        app.state.embedding_client = mock_ec
        app.state.embedding_store = _make_embedding_store_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/search", params={"q": "test"})

        assert response.status_code == 200
        mock_ec.embed.assert_called_once()
```

**Update `TestGetSearchEmbedding.test_embedding_client_called_with_query`** — add `embedding_store`:
```python
    async def test_embedding_client_called_with_query(self):
        app = create_app(use_lifespan=False)
        mock_ec = _make_embedding_client_mock()
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(rows=[])
        app.state.embedding_client = mock_ec
        app.state.embedding_store = _make_embedding_store_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.get("/api/search", params={"q": "semantic query"})

        mock_ec.embed.assert_called_once_with("semantic query")
```

**Update `TestGetSearchEmbedding.test_embedding_client_called_once_per_request`** — add `embedding_store`:
```python
    async def test_embedding_client_called_once_per_request(self):
        app = create_app(use_lifespan=False)
        mock_ec = _make_embedding_client_mock()
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(rows=[])
        app.state.embedding_client = mock_ec
        app.state.embedding_store = _make_embedding_store_mock()

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            await c.get("/api/search", params={"q": "first"})
            await c.get("/api/search", params={"q": "second"})

        assert mock_ec.embed.call_count == 2
```

**Update `TestGetSearchNullSummary.test_null_summary_allowed`** — add `embedding_store`:
```python
    async def test_null_summary_allowed(self):
        rows = [{**_SAMPLE_ROW, "summary": None}]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(rows=rows)
        app.state.embedding_client = _make_embedding_client_mock()
        app.state.embedding_store = _make_embedding_store_mock(search_rows=rows)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/search", params={"q": "test"})

        assert response.status_code == 200
        data = response.json()
        assert data[0]["summary"] is None
```

**Update `TestGetSearchNullSummary.test_empty_tags_list`** — add `embedding_store`:
```python
    async def test_empty_tags_list(self):
        rows = [{**_SAMPLE_ROW, "tags": []}]
        app = create_app(use_lifespan=False)
        app.state.knowledge_store = _make_knowledge_store_mock()
        app.state.pg_pool = _make_pg_pool_mock(rows=rows)
        app.state.embedding_client = _make_embedding_client_mock()
        app.state.embedding_store = _make_embedding_store_mock(search_rows=rows)

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            cookies={"ks_session": make_test_session_cookie()},
        ) as c:
            response = await c.get("/api/search", params={"q": "test"})

        assert response.status_code == 200
        data = response.json()
        assert data[0]["tags"] == []
```

- [ ] **Step 6: Run all search tests**

Run: `uv run pytest tests/test_api_search.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_service/api/search.py tests/test_api_search.py
git commit -m "feat: search queries chunks first, falls back to content"
```

---

## Task 7: RAG retriever — chunk-aware retrieval (TDD)

**Files:**
- Modify: `src/knowledge_service/stores/rag.py:48`
- Modify: `tests/test_rag_retriever.py`

### 7a: Tests for chunk-aware RAG

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rag_retriever.py`:

```python
class TestRetrieveChunkAware:
    """RAGRetriever should use search_chunks for content retrieval."""

    async def test_calls_search_chunks(self):
        es = _make_embedding_store()
        es.search_chunks = AsyncMock(return_value=[])
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
            knowledge_store=_make_knowledge_store(),
        )
        await retriever.retrieve("question", max_sources=5, min_confidence=0.0)
        es.search_chunks.assert_called_once()

    async def test_search_chunks_limit_matches_max_sources(self):
        es = _make_embedding_store()
        es.search_chunks = AsyncMock(return_value=[])
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
            knowledge_store=_make_knowledge_store(),
        )
        await retriever.retrieve("q", max_sources=7, min_confidence=0.0)
        call_kwargs = es.search_chunks.call_args
        assert call_kwargs.kwargs.get("limit") == 7 or call_kwargs[1].get("limit") == 7

    async def test_also_calls_search_for_unchunked_fallback(self):
        es = _make_embedding_store()
        es.search_chunks = AsyncMock(return_value=[])
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
            knowledge_store=_make_knowledge_store(),
        )
        await retriever.retrieve("q", max_sources=5, min_confidence=0.0)
        es.search.assert_called_once()

    async def test_chunk_results_in_content_results(self):
        chunk_row = {
            "chunk_id": "chunk-1",
            "text": "The chunk text",
            "chunk_index": 0,
            "content_id": "content-1",
            "url": "https://example.com/article",
            "title": "Article",
            "source_type": "article",
            "tags": [],
            "ingested_at": "2026-01-01",
            "similarity": 0.95,
        }
        es = _make_embedding_store()
        es.search_chunks = AsyncMock(return_value=[chunk_row])
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
            knowledge_store=_make_knowledge_store(),
        )
        ctx = await retriever.retrieve("q", max_sources=5, min_confidence=0.0)
        assert len(ctx.content_results) >= 1
        assert ctx.content_results[0]["url"] == "https://example.com/article"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rag_retriever.py::TestRetrieveChunkAware -v`
Expected: FAIL — `search_chunks` is not called.

- [ ] **Step 3: Update RAGRetriever to use chunks**

In `src/knowledge_service/stores/rag.py`, replace Step 2 (content search, line 47-50):

```python
        # Step 2: Chunk-level content search (primary)
        chunk_results = await self._embedding_store.search_chunks(
            query_embedding=embedding, limit=max_sources
        )

        # Step 2b: Fallback content search for un-chunked documents
        remaining = max_sources - len(chunk_results)
        content_results = []
        if remaining > 0:
            content_rows = await self._embedding_store.search(
                query_embedding=embedding, limit=remaining
            )
            # Exclude content IDs already covered by chunks
            seen_ids = {str(r["content_id"]) for r in chunk_results}
            content_results = [r for r in content_rows if str(r["id"]) not in seen_ids]

        # Merge: chunk results + unchunked content results
        all_content = list(chunk_results) + content_results
```

Then update all downstream references from `content_results` to `all_content` (line 98):

```python
        return RetrievalContext(
            content_results=all_content,
            ...
        )
```

- [ ] **Step 4: Run chunk-aware RAG tests**

Run: `uv run pytest tests/test_rag_retriever.py::TestRetrieveChunkAware -v`
Expected: All PASS

- [ ] **Step 5: Run all RAG tests**

Run: `uv run pytest tests/test_rag_retriever.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/stores/rag.py tests/test_rag_retriever.py
git commit -m "feat: RAGRetriever uses chunk-level search with content fallback"
```

---

## Task 8: Full test suite + lint

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Run linter**

Run: `uv run ruff check .`
Expected: No errors

- [ ] **Step 3: Run formatter check**

Run: `uv run ruff format --check .`
Expected: No formatting issues (or run `uv run ruff format .` to fix)

- [ ] **Step 4: Final commit if any lint fixes**

```bash
git add -A
git commit -m "chore: lint fixes"
```

---

## Task 9: Update EmbeddingStore docstring

**Files:**
- Modify: `src/knowledge_service/stores/embedding.py:1-26`

- [ ] **Step 1: Update module docstring to document content_chunks table**

Update the module docstring at the top of `stores/embedding.py` to add the `content_chunks` table:

```python
"""EmbeddingStore: asyncpg-backed store for pgvector semantic similarity search.

Manages three tables (schema from migrations/):

    content:
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid()
        url             TEXT UNIQUE
        title           TEXT
        summary         TEXT
        raw_text        TEXT
        source_type     TEXT NOT NULL
        tags            TEXT[] DEFAULT '{}'
        ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        embedding       vector(768)
        metadata        JSONB DEFAULT '{}'

    content_chunks:
        chunk_id        UUID PRIMARY KEY DEFAULT gen_random_uuid()
        content_id      UUID NOT NULL REFERENCES content(id) ON DELETE CASCADE
        chunk_index     INTEGER NOT NULL
        text            TEXT NOT NULL
        embedding       vector(768)
        char_start      INTEGER
        char_end        INTEGER
        created_at      TIMESTAMPTZ DEFAULT now()

    entity_embeddings:
        uri             TEXT PRIMARY KEY
        label           TEXT NOT NULL
        rdf_type        TEXT DEFAULT ''
        embedding       vector(768)
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()

All tables have HNSW indexes on (embedding::halfvec(768)) using halfvec_cosine_ops.
Queries must cast to halfvec to exploit those indexes.
"""
```

- [ ] **Step 2: Commit**

```bash
git add src/knowledge_service/stores/embedding.py
git commit -m "docs: update EmbeddingStore docstring for content_chunks table"
```
