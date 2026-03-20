# Chunking Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Restructure content storage so every piece of content is stored as chunks (short = 1 chunk, long = N chunks). Search and RAG always operate at chunk level via a single JOIN query.

**Architecture:** `content_metadata` stores document metadata (url, title, tags, etc.). `content` table is repurposed to store chunks — each row has chunk text + embedding. Every content item has >= 1 chunk. Search is a single `content JOIN content_metadata` query. No fallback, no UNION ALL.

**Tech Stack:** Python 3.12, FastAPI, asyncpg, pgvector, langchain-text-splitters, pytest

**Spec:** `docs/superpowers/specs/2026-03-20-chunking-enhancement-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `migrations/002_content_chunks.sql` | Drop old content, create content_metadata + new content |
| Modify | `pyproject.toml` | Add `langchain-text-splitters` dependency |
| Modify | `src/knowledge_service/stores/embedding.py` | Rewrite for new schema: `insert_content_metadata()`, `delete_chunks()`, `insert_chunks()`, updated `search()` |
| Modify | `src/knowledge_service/api/content.py` | Always-chunk ingestion flow |
| Modify | `src/knowledge_service/api/search.py` | Simplified single-query search |
| Modify | `src/knowledge_service/models.py` | `SearchResult` with required `chunk_text`, `chunk_index` |
| Modify | `src/knowledge_service/stores/rag.py` | Minor update — `search()` already returns chunks |
| Modify | `tests/test_embedding_store.py` | Tests for rewritten EmbeddingStore |
| Modify | `tests/test_api_content.py` | Tests for always-chunk ingestion |
| Modify | `tests/test_api_search.py` | Tests for chunk-level search |
| Modify | `tests/test_rag_retriever.py` | Tests for chunk-aware RAG |

---

## Task 1: Migration — `content_metadata` + restructured `content`

**Files:**
- Create: `migrations/002_content_chunks.sql`

- [x] **Step 1: Write the migration SQL**

```sql
-- 002_content_chunks.sql
-- Restructure content storage: metadata in content_metadata, chunks in content.
-- Existing content data is dropped (acceptable at v0.1.x).

DROP TABLE IF EXISTS content;

CREATE TABLE content_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    summary TEXT,
    raw_text TEXT,
    source_type TEXT NOT NULL,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL REFERENCES content_metadata(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(768),
    char_start INTEGER,
    char_end INTEGER,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(content_id, chunk_index)
);

CREATE INDEX idx_content_embedding ON content
    USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops);
CREATE INDEX idx_content_content_id ON content (content_id);
```

- [x] **Step 2: Commit**

```bash
git add migrations/002_content_chunks.sql
git commit -m "feat: migration 002 — content_metadata + content as chunks"
```

---

## Task 2: Add `langchain-text-splitters` dependency

**Files:**
- Modify: `pyproject.toml`

- [x] **Step 1: Add dependency to pyproject.toml**

In `pyproject.toml`, add `"langchain-text-splitters>=0.3.0"` to the `dependencies` list (after `"pydantic-settings>=2.7.0"`).

- [x] **Step 2: Sync the environment**

Run: `uv sync --dev`
Expected: Resolves and installs `langchain-text-splitters` successfully.

- [x] **Step 3: Verify import works**

Run: `uv run python -c "from langchain_text_splitters import RecursiveCharacterTextSplitter; print('OK')"`
Expected: `OK`

- [x] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add langchain-text-splitters dependency"
```

---

## Task 3: EmbeddingStore — rewrite for new schema (TDD)

**Files:**
- Modify: `src/knowledge_service/stores/embedding.py`
- Modify: `tests/test_embedding_store.py`

### 3a: `insert_content_metadata()`

- [x] **Step 1: Write the failing test**

Replace the `TestInsertContent` class in `tests/test_embedding_store.py` with:

```python
class TestInsertContentMetadata:
    async def test_returns_content_id(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = {"id": "metadata-uuid-123"}
        result = await store.insert_content_metadata(
            url="https://example.com/article",
            title="Test Article",
            summary="A test summary",
            raw_text="Full text content",
            source_type="article",
            tags=["test", "example"],
            metadata={},
        )
        conn.fetchrow.assert_called_once()
        assert result == "metadata-uuid-123"

    async def test_sql_targets_content_metadata(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = {"id": "some-uuid"}
        await store.insert_content_metadata(
            url="https://example.com/article",
            title="Test",
            summary="Sum",
            raw_text="Text",
            source_type="article",
            tags=[],
            metadata={},
        )
        sql = conn.fetchrow.call_args[0][0]
        assert "INSERT INTO content_metadata" in sql
        assert "ON CONFLICT" in sql

    async def test_passes_url_param(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetchrow.return_value = {"id": "uuid-abc"}
        await store.insert_content_metadata(
            url="https://example.com/unique",
            title="Title",
            summary="Sum",
            raw_text="Text",
            source_type="article",
            tags=[],
            metadata={},
        )
        args = conn.fetchrow.call_args[0]
        assert "https://example.com/unique" in args
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embedding_store.py::TestInsertContentMetadata -v`
Expected: FAIL — `AttributeError: 'EmbeddingStore' object has no attribute 'insert_content_metadata'`

- [x] **Step 3: Write minimal implementation**

In `src/knowledge_service/stores/embedding.py`, replace the `insert_content()` method with:

```python
    # ------------------------------------------------------------------
    # Content metadata table operations
    # ------------------------------------------------------------------

    async def insert_content_metadata(
        self,
        url: str,
        title: str,
        summary: str,
        raw_text: str,
        source_type: str,
        tags: list[str],
        metadata: dict,
    ) -> str:
        """Upsert a content_metadata row and return its UUID.

        On conflict (url) the existing row is updated with fresh values,
        leaving id and ingested_at unchanged.
        """
        metadata_json = json.dumps(metadata)

        sql = """
            INSERT INTO content_metadata (
                url, title, summary, raw_text, source_type, tags, metadata
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (url) DO UPDATE SET
                title       = EXCLUDED.title,
                summary     = EXCLUDED.summary,
                raw_text    = EXCLUDED.raw_text,
                source_type = EXCLUDED.source_type,
                tags        = EXCLUDED.tags,
                metadata    = EXCLUDED.metadata
            RETURNING id
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql, url, title, summary, raw_text,
                source_type, tags, metadata_json,
            )
        return str(row["id"])
```

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embedding_store.py::TestInsertContentMetadata -v`
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add src/knowledge_service/stores/embedding.py tests/test_embedding_store.py
git commit -m "feat: add EmbeddingStore.insert_content_metadata()"
```

### 3b: `delete_chunks()` and `insert_chunks()`

- [x] **Step 6: Write the failing tests**

Add to `tests/test_embedding_store.py`:

```python
class TestDeleteChunks:
    async def test_calls_execute_with_delete(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "DELETE 3"
        await store.delete_chunks("content-uuid-123")
        conn.execute.assert_called_once()
        sql = conn.execute.call_args[0][0]
        assert "DELETE FROM content" in sql
        assert "content_id" in sql

    async def test_passes_content_id_param(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "DELETE 0"
        await store.delete_chunks("my-uuid")
        args = conn.execute.call_args[0]
        assert "my-uuid" in args


class TestInsertChunks:
    async def test_inserts_multiple_chunks(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "INSERT 0 1"
        chunks = [
            {"chunk_index": 0, "chunk_text": "First chunk", "embedding": [0.1] * 768, "char_start": 0, "char_end": 100},
            {"chunk_index": 1, "chunk_text": "Second chunk", "embedding": [0.2] * 768, "char_start": 80, "char_end": 200},
        ]
        await store.insert_chunks("content-uuid-123", chunks)
        assert conn.execute.call_count == 2

    async def test_sql_targets_content_table(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "INSERT 0 1"
        chunks = [
            {"chunk_index": 0, "chunk_text": "chunk", "embedding": [0.1] * 768, "char_start": 0, "char_end": 50},
        ]
        await store.insert_chunks("uuid-1", chunks)
        sql = conn.execute.call_args[0][0]
        assert "INSERT INTO content" in sql

    async def test_passes_content_id(self, store, mock_pool):
        _, conn = mock_pool
        conn.execute.return_value = "INSERT 0 1"
        chunks = [
            {"chunk_index": 0, "chunk_text": "chunk", "embedding": [0.1] * 768, "char_start": 0, "char_end": 50},
        ]
        await store.insert_chunks("my-content-id", chunks)
        args = conn.execute.call_args[0]
        assert "my-content-id" in args

    async def test_no_chunks_no_execute(self, store, mock_pool):
        _, conn = mock_pool
        await store.insert_chunks("uuid-1", [])
        conn.execute.assert_not_called()
```

- [x] **Step 7: Run test to verify they fail**

Run: `uv run pytest tests/test_embedding_store.py::TestDeleteChunks tests/test_embedding_store.py::TestInsertChunks -v`
Expected: FAIL

- [x] **Step 8: Write minimal implementation**

Add to `EmbeddingStore` after `insert_content_metadata()`:

```python
    # ------------------------------------------------------------------
    # Content (chunks) table operations
    # ------------------------------------------------------------------

    async def delete_chunks(self, content_id: str) -> None:
        """Delete all chunks for a given content_id."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM content WHERE content_id = $1", content_id,
            )

    async def insert_chunks(
        self,
        content_id: str,
        chunks: list[dict],
    ) -> None:
        """Insert chunk rows into the content table.

        Each dict must have: chunk_index, chunk_text, embedding, char_start, char_end.
        """
        if not chunks:
            return

        sql = """
            INSERT INTO content (
                content_id, chunk_index, chunk_text, embedding, char_start, char_end
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
                    chunk["chunk_text"],
                    embedding_str,
                    chunk["char_start"],
                    chunk["char_end"],
                )
```

- [x] **Step 9: Run test to verify they pass**

Run: `uv run pytest tests/test_embedding_store.py::TestDeleteChunks tests/test_embedding_store.py::TestInsertChunks -v`
Expected: PASS

- [x] **Step 10: Commit**

```bash
git add src/knowledge_service/stores/embedding.py tests/test_embedding_store.py
git commit -m "feat: add EmbeddingStore.delete_chunks() and insert_chunks()"
```

### 3c: Rewrite `search()` with JOIN

- [x] **Step 11: Write the failing tests**

Replace the `TestSearch` class in `tests/test_embedding_store.py`:

```python
class TestSearch:
    async def test_search_returns_chunk_results(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {
                "id": "chunk-uuid-1",
                "chunk_text": "relevant chunk text",
                "chunk_index": 0,
                "content_id": "metadata-uuid-1",
                "url": "https://a.com",
                "title": "A",
                "summary": "S",
                "source_type": "article",
                "tags": ["t"],
                "ingested_at": "2025-01-01",
                "similarity": 0.95,
            }
        ]
        results = await store.search(query_embedding=[0.1] * 768, limit=10)
        assert len(results) == 1
        assert results[0]["chunk_text"] == "relevant chunk text"
        assert results[0]["content_id"] == "metadata-uuid-1"
        assert results[0]["similarity"] == 0.95

    async def test_search_calls_fetch(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search(query_embedding=[0.1] * 768, limit=5)
        conn.fetch.assert_called_once()

    async def test_search_sql_joins_content_metadata(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search(query_embedding=[0.1] * 768, limit=5)
        sql = conn.fetch.call_args[0][0]
        assert "content_metadata" in sql
        assert "JOIN" in sql.upper()
        assert "<=>" in sql
        assert "halfvec" in sql

    async def test_search_with_source_type_filter(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search(
            query_embedding=[0.1] * 768, limit=5, source_type="article",
        )
        sql = conn.fetch.call_args[0][0]
        assert "source_type" in sql

    async def test_search_with_tags_filter(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search(
            query_embedding=[0.1] * 768, limit=5, tags=["python", "database"],
        )
        sql = conn.fetch.call_args[0][0]
        assert "tags" in sql

    async def test_search_returns_empty_list(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        results = await store.search(query_embedding=[0.1] * 768, limit=10)
        assert results == []

    async def test_search_passes_limit_param(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search(query_embedding=[0.1] * 768, limit=42)
        args = conn.fetch.call_args[0]
        assert 42 in args
```

- [x] **Step 12: Run test to verify they fail**

Run: `uv run pytest tests/test_embedding_store.py::TestSearch -v`
Expected: FAIL — `search()` still uses old SQL without JOIN.

- [x] **Step 13: Rewrite `search()` implementation**

Replace the existing `search()` method in `EmbeddingStore`:

```python
    async def search(
        self,
        query_embedding: list[float],
        limit: int,
        source_type: str | None = None,
        tags: list[str] | None = None,
        min_date: Any | None = None,
    ) -> list[dict]:
        """Return chunk rows ranked by cosine similarity, joined with content metadata.

        Optional filters:
          source_type — restrict to a single source type
          tags        — restrict to rows that contain ALL given tags
        """
        embedding_str = self._vector_to_str(query_embedding)

        conditions: list[str] = []
        params: list[Any] = [embedding_str]

        if source_type is not None:
            params.append(source_type)
            conditions.append(f"m.source_type = ${len(params)}")

        if tags is not None:
            params.append(tags)
            conditions.append(f"m.tags @> ${len(params)}")

        if min_date is not None:
            params.append(min_date)
            conditions.append(f"m.ingested_at >= ${len(params)}")

        params.append(limit)
        limit_placeholder = f"${len(params)}"

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        sql = f"""
            SELECT
                c.id, c.chunk_text, c.chunk_index,
                m.id AS content_id, m.url, m.title, m.summary,
                m.source_type, m.tags, m.ingested_at,
                1 - (c.embedding::halfvec(768) <=> $1::halfvec(768)) AS similarity
            FROM content c
            JOIN content_metadata m ON c.content_id = m.id
            {where_clause}
            ORDER BY c.embedding::halfvec(768) <=> $1::halfvec(768)
            LIMIT {limit_placeholder}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [dict(row) for row in rows]
```

Also remove the old `insert_content()` method (replaced by `insert_content_metadata()`).

- [x] **Step 14: Run test to verify they pass**

Run: `uv run pytest tests/test_embedding_store.py::TestSearch -v`
Expected: PASS

- [x] **Step 15: Commit**

```bash
git add src/knowledge_service/stores/embedding.py tests/test_embedding_store.py
git commit -m "feat: rewrite EmbeddingStore.search() with content_metadata JOIN"
```

---

## Task 4: `SearchResult` model — add chunk fields

**Files:**
- Modify: `src/knowledge_service/models.py:316-324`

- [x] **Step 1: Update SearchResult**

In `src/knowledge_service/models.py`, replace the `SearchResult` class:

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
    chunk_text: str
    chunk_index: int
```

Both `chunk_text` and `chunk_index` are **required** (not optional) since every search result is a chunk.

- [x] **Step 2: Commit**

```bash
git add src/knowledge_service/models.py
git commit -m "feat: add required chunk_text, chunk_index to SearchResult"
```

---

## Task 5: Content ingestion — always-chunk flow (TDD)

**Files:**
- Modify: `src/knowledge_service/api/content.py`
- Modify: `tests/test_api_content.py`

### 5a: Tests for always-chunk ingestion

- [x] **Step 1: Write the failing tests**

Update `tests/test_api_content.py`. First, update the mock helpers:

**Replace `_make_embedding_client_mock()`** so `embed_batch` scales dynamically:
```python
def _make_embedding_client_mock():
    """Build a mock EmbeddingClient."""
    mock = AsyncMock()
    mock.embed.return_value = [0.1] * 768
    mock.embed_batch.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    return mock
```

**Replace `_make_embedding_store_mock()`:**
```python
def _make_embedding_store_mock():
    """Build a mock EmbeddingStore with new schema methods."""
    mock = AsyncMock()
    mock.insert_content_metadata.return_value = "content-uuid-1234"
    mock.delete_chunks.return_value = None
    mock.insert_chunks.return_value = None
    return mock
```

**Update the `client` fixture** to use the new mock (it currently sets `mock.insert_content`).

**Add new test class at the end:**
```python
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
        mock_es.insert_content_metadata.assert_called_once()
        mock_es.delete_chunks.assert_called_once_with("content-uuid-1234")
        mock_es.insert_chunks.assert_called_once()
        chunks = mock_es.insert_chunks.call_args[0][1]
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert chunks[0]["chunk_text"] == "This is a short article."

    async def test_long_content_creates_multiple_chunks(self):
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
            response = await c.post("/api/content", json=LONG_TEXT_PAYLOAD)

        assert response.status_code == 200
        mock_es.insert_chunks.assert_called_once()
        chunks = mock_es.insert_chunks.call_args[0][1]
        assert len(chunks) >= 2
        mock_ec.embed_batch.assert_called_once()

    async def test_no_raw_text_creates_one_chunk_from_title(self):
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
            response = await c.post("/api/content", json=MINIMAL_PAYLOAD)

        assert response.status_code == 200
        mock_es.insert_chunks.assert_called_once()
        chunks = mock_es.insert_chunks.call_args[0][1]
        assert len(chunks) == 1

    async def test_reingestion_deletes_old_chunks(self):
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
            await c.post("/api/content", json=SHORT_TEXT_PAYLOAD)

        mock_es.delete_chunks.assert_called_once_with("content-uuid-1234")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_content.py::TestContentChunking -v`
Expected: FAIL — content.py still calls `insert_content()` not the new methods.

- [x] **Step 3: Rewrite content ingestion**

In `src/knowledge_service/api/content.py`, add the import at the top:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

Add module-level constants after `router = APIRouter()`:

```python
_CHUNK_SIZE = 4000
_CHUNK_OVERLAP = 200
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

Replace steps 1-2 in `_process_one_content_request()` (lines 60-74) with the new flow:

```python
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

    # Step 2: Always chunk and embed
    text = body.raw_text or body.summary or body.title
    if len(text) >= _CHUNK_SIZE:
        chunks_text = _splitter.split_text(text)
    else:
        chunks_text = [text]

    # Track char offsets
    chunk_records = []
    search_start = 0
    for i, ct in enumerate(chunks_text):
        char_start = text.find(ct[:100], search_start)
        if char_start == -1:
            char_start = search_start
        char_end = char_start + len(ct)
        search_start = max(search_start, char_start + 1)
        chunk_records.append({
            "chunk_index": i,
            "chunk_text": ct,
            "char_start": char_start,
            "char_end": char_end,
        })

    # Embed chunks (batch for multiple, single for one)
    if len(chunk_records) == 1:
        embeddings = [await embedding_client.embed(chunk_records[0]["chunk_text"])]
    else:
        embeddings = await embedding_client.embed_batch(
            [c["chunk_text"] for c in chunk_records]
        )
    for rec, emb in zip(chunk_records, embeddings):
        rec["embedding"] = emb

    # Delete old chunks (re-ingestion) and insert new
    await embedding_store.delete_chunks(content_id)
    await embedding_store.insert_chunks(content_id, chunk_records)
```

Remove the old `embedding = await embedding_client.embed(embed_text)` and `embedding_store.insert_content(...)` calls.

- [x] **Step 4: Run chunking tests**

Run: `uv run pytest tests/test_api_content.py::TestContentChunking -v`
Expected: All PASS

- [x] **Step 5: Update remaining content tests for new mock API**

The other test classes in `test_api_content.py` still reference `mock.insert_content` (old API). Update:

- **`_make_embedding_store_mock()`** already updated in Step 1
- **Tests that assert `content_id`**: `insert_content_metadata` returns `"content-uuid-1234"`, same as before — assertions on `data["content_id"]` still pass
- **`TestPostContentEmbedding.test_pg_pool_used_to_insert_content`**: rename/rewrite to test that `insert_content_metadata` is called
- **`TestPostContentKnowledgeStore` tests**: these don't set `embedding_store` on `app.state` — add `app.state.embedding_store = _make_embedding_store_mock()` to each

Run: `uv run pytest tests/test_api_content.py -v`
Fix any remaining failures.

- [x] **Step 6: Commit**

```bash
git add src/knowledge_service/api/content.py tests/test_api_content.py
git commit -m "feat: always-chunk content ingestion with content_metadata"
```

---

## Task 6: Search endpoint — simplified chunk search (TDD)

**Files:**
- Modify: `src/knowledge_service/api/search.py`
- Modify: `tests/test_api_search.py`

- [x] **Step 1: Update test fixtures and add chunk search tests**

Rewrite `tests/test_api_search.py` fixtures and sample data:

**Update `_SAMPLE_ROW`** to match the new `search()` return shape:
```python
_SAMPLE_ROW = {
    "id": "chunk-uuid-1",
    "chunk_text": "The relevant text from this article",
    "chunk_index": 0,
    "content_id": "content-uuid-1234",
    "url": "https://example.com/article",
    "title": "Test Article",
    "summary": "A test article summary",
    "source_type": "article",
    "tags": ["python", "testing"],
    "ingested_at": _NOW,
    "similarity": 0.92,
}
```

**Add `_make_embedding_store_mock` helper:**
```python
def _make_embedding_store_mock(search_rows=None):
    mock = AsyncMock()
    mock.search.return_value = search_rows or []
    return mock
```

**Update `client` fixture** to use `embedding_store` mock:
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

**Update `empty_client` fixture** similarly with `_make_embedding_store_mock()` (empty).

**Add `AsyncMock` to imports** if not already present.

**Update `TestGetSearchBasic.test_result_has_required_fields`** to also check `chunk_text` and `chunk_index`.

**Update `TestGetSearchBasic.test_result_values_match_row`** to assert:
```python
assert result["chunk_text"] == "The relevant text from this article"
assert result["chunk_index"] == 0
```

**Update ALL tests that create their own app instances** (there are 6 — see list below) to include `app.state.embedding_store = _make_embedding_store_mock(...)`:
- `TestGetSearchSimilarity.test_multiple_results_have_similarity`
- `TestGetSearchValidation.test_default_limit_is_ten`
- `TestGetSearchEmbedding.test_embedding_client_called_with_query`
- `TestGetSearchEmbedding.test_embedding_client_called_once_per_request`
- `TestGetSearchNullSummary.test_null_summary_allowed`
- `TestGetSearchNullSummary.test_empty_tags_list`

Each needs:
```python
app.state.embedding_store = _make_embedding_store_mock(search_rows=<rows_for_this_test>)
```

Where `<rows_for_this_test>` uses `_SAMPLE_ROW` variants matching what the test expects.

- [x] **Step 2: Rewrite search endpoint**

Replace `src/knowledge_service/api/search.py`:

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

    Queries chunk embeddings in the content table, joined with content_metadata
    for filtering and metadata. Every result is a chunk.
    """
    embedding_client = request.app.state.embedding_client
    embedding_store = getattr(request.app.state, "embedding_store", None)

    if embedding_store is None:
        pg_pool = request.app.state.pg_pool
        embedding_store = EmbeddingStore(pg_pool)

    embedding = await embedding_client.embed(q)

    rows = await embedding_store.search(
        query_embedding=embedding,
        limit=limit,
        source_type=source_type,
        tags=tags,
    )

    return [
        SearchResult(
            content_id=str(row["content_id"]),
            url=row["url"],
            title=row["title"],
            summary=row.get("summary"),
            similarity=float(row["similarity"]),
            source_type=row["source_type"],
            tags=list(row["tags"]) if row["tags"] else [],
            ingested_at=row["ingested_at"],
            chunk_text=row["chunk_text"],
            chunk_index=row["chunk_index"],
        )
        for row in rows
    ]
```

No fallback. No deduplication. No sorting. One call.

- [x] **Step 3: Run all search tests**

Run: `uv run pytest tests/test_api_search.py -v`
Expected: All PASS

- [x] **Step 4: Commit**

```bash
git add src/knowledge_service/api/search.py tests/test_api_search.py
git commit -m "feat: simplified chunk-level search endpoint"
```

---

## Task 7: RAG retriever — chunk-aware retrieval (TDD)

**Files:**
- Modify: `src/knowledge_service/stores/rag.py`
- Modify: `tests/test_rag_retriever.py`

Since `EmbeddingStore.search()` now returns chunk-level results automatically, the RAG retriever mostly just works. The only change is that `content_results` now contain `chunk_text` and `chunk_index` keys, and the `id` key is now a chunk UUID rather than content UUID.

- [x] **Step 1: Update RAG retriever test mock**

In `tests/test_rag_retriever.py`, update `_CONTENT_ROW` to match the new `search()` return shape:

```python
_CONTENT_ROW = {
    "id": "chunk-uuid-1",
    "chunk_text": "Relevant text about the topic",
    "chunk_index": 0,
    "content_id": "content-uuid-1",
    "url": "https://example.com/article",
    "title": "Test Article",
    "summary": "A summary",
    "source_type": "article",
    "tags": ["health"],
    "ingested_at": "2026-03-18T10:00:00Z",
    "similarity": 0.92,
}
```

- [x] **Step 2: Run existing RAG tests**

Run: `uv run pytest tests/test_rag_retriever.py -v`
Expected: All PASS (the retriever accesses `row["url"]`, `row.get("title")`, etc. — all still present in the new shape). If any tests reference `row["id"]` as a content_id, update them to use `row["content_id"]`.

- [x] **Step 3: Update RAG prompt builder for chunk text**

In `src/knowledge_service/clients/rag.py`, the `build_rag_prompt()` function currently uses `row.get("summary")` for context. Update to prefer `chunk_text`:

```python
    if context.content_results:
        sections.append("## Relevant Content")
        for row in context.content_results:
            title = row.get("title", "Untitled")
            source_type = row.get("source_type", "unknown")
            similarity = row.get("similarity", 0.0)
            text = row.get("chunk_text") or row.get("summary") or "No content"
            sections.append(f'- "{title}" ({source_type}, similarity: {similarity:.2f}): {text}')
        sections.append("")
```

- [x] **Step 4: Run all RAG tests**

Run: `uv run pytest tests/test_rag_retriever.py tests/test_api_ask.py -v`
Expected: All PASS

- [x] **Step 5: Commit**

```bash
git add src/knowledge_service/stores/rag.py src/knowledge_service/clients/rag.py tests/test_rag_retriever.py
git commit -m "feat: RAG retriever uses chunk-level search results"
```

---

## Task 8: Full test suite + lint

- [x] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All PASS

- [x] **Step 2: Run linter**

Run: `uv run ruff check .`
Expected: No errors

- [x] **Step 3: Run formatter**

Run: `uv run ruff format --check .`
Expected: No issues (or run `uv run ruff format .` to fix)

- [x] **Step 4: Commit if any fixes**

```bash
git add -A
git commit -m "chore: lint fixes"
```

---

## Task 9: Update docstrings

**Files:**
- Modify: `src/knowledge_service/stores/embedding.py:1-26`

- [x] **Step 1: Update EmbeddingStore module docstring**

```python
"""EmbeddingStore: asyncpg-backed store for pgvector semantic similarity search.

Manages three tables (schema from migrations/):

    content_metadata:
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid()
        url             TEXT UNIQUE
        title           TEXT
        summary         TEXT
        raw_text        TEXT
        source_type     TEXT NOT NULL
        tags            TEXT[] DEFAULT '{}'
        metadata        JSONB DEFAULT '{}'
        ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()

    content (chunks):
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid()
        content_id      UUID NOT NULL REFERENCES content_metadata(id) ON DELETE CASCADE
        chunk_index     INTEGER NOT NULL
        chunk_text      TEXT NOT NULL
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

content and entity_embeddings have HNSW indexes on (embedding::halfvec(768))
using halfvec_cosine_ops. Queries must cast to halfvec to exploit those indexes.
"""
```

- [x] **Step 2: Commit**

```bash
git add src/knowledge_service/stores/embedding.py
git commit -m "docs: update EmbeddingStore docstring for new schema"
```
