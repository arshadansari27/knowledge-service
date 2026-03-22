# BM25 Hybrid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PostgreSQL full-text search alongside existing pgvector vector search, fused via Reciprocal Rank Fusion, for both `/api/search` and `RAGRetriever`.

**Architecture:** tsvector column on the `content` table with a trigger for auto-population. New `search_bm25()` method. Existing `search()` gains a `query_text` parameter that enables hybrid mode (vector + BM25 → RRF fusion). Callers pass the raw query text to activate.

**Tech Stack:** PostgreSQL tsvector/tsquery, GIN index, ts_rank, Reciprocal Rank Fusion

**Spec:** `docs/superpowers/specs/2026-03-22-bm25-hybrid-search-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `migrations/005_add_fts.sql` | tsvector column, GIN index, trigger, backfill |
| Modify | `src/knowledge_service/stores/embedding.py:165-216` | Add `search_bm25()`, RRF function, modify `search()` for hybrid mode |
| Modify | `src/knowledge_service/api/search.py:35-40` | Pass `query_text=q` to `search()` |
| Modify | `src/knowledge_service/stores/rag.py:49-51` | Pass `query_text=question` to `search()` |
| Create | `tests/test_hybrid_search.py` | All BM25/RRF/hybrid tests |

---

## Task 1: SQL migration for tsvector

**Files:**
- Create: `migrations/005_add_fts.sql`

- [ ] **Step 1: Create migration file**

```sql
-- Full-text search: tsvector column on content (chunks) table
ALTER TABLE content ADD COLUMN tsv tsvector;

CREATE INDEX idx_content_tsv ON content USING GIN(tsv);

CREATE OR REPLACE FUNCTION content_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', COALESCE(NEW.chunk_text, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_content_tsv
    BEFORE INSERT OR UPDATE OF chunk_text ON content
    FOR EACH ROW EXECUTE FUNCTION content_tsv_trigger();

-- Backfill existing rows
UPDATE content SET tsv = to_tsvector('english', COALESCE(chunk_text, ''));
```

- [ ] **Step 2: Commit**

```bash
git add migrations/005_add_fts.sql
git commit -m "feat: add tsvector column with GIN index and trigger for FTS"
```

---

## Task 2: RRF fusion function

**Files:**
- Modify: `src/knowledge_service/stores/embedding.py` (add module-level function after imports)
- Create: `tests/test_hybrid_search.py`

- [ ] **Step 1: Write failing tests for RRF**

Create `tests/test_hybrid_search.py`:

```python
import pytest
from knowledge_service.stores.embedding import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_single_list(self):
        results = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        fused = reciprocal_rank_fusion(results, key="id", k=60, limit=3)
        assert len(fused) == 3
        assert fused[0]["id"] == "a"  # rank 1 scores highest

    def test_two_lists_overlap_ranks_higher(self):
        """Item in both lists should rank higher than item in one."""
        list1 = [{"id": "shared"}, {"id": "only_vec"}]
        list2 = [{"id": "shared"}, {"id": "only_bm25"}]
        fused = reciprocal_rank_fusion(list1, list2, key="id", k=60, limit=3)
        assert fused[0]["id"] == "shared"

    def test_deduplication(self):
        """Same item from both lists appears once."""
        list1 = [{"id": "x", "source": "vec"}]
        list2 = [{"id": "x", "source": "bm25"}]
        fused = reciprocal_rank_fusion(list1, list2, key="id", k=60, limit=10)
        assert len(fused) == 1

    def test_respects_limit(self):
        list1 = [{"id": str(i)} for i in range(20)]
        fused = reciprocal_rank_fusion(list1, key="id", k=60, limit=5)
        assert len(fused) == 5

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([], [], key="id", k=60, limit=10)
        assert fused == []

    def test_rrf_score_injected(self):
        """Fused results should have rrf_score field."""
        list1 = [{"id": "a", "similarity": 0.9}]
        list2 = [{"id": "a", "similarity": 0.5}]
        fused = reciprocal_rank_fusion(list1, list2, key="id", k=60, limit=10)
        assert "similarity" in fused[0]
        # RRF score should be 2 * 1/(60+1) ≈ 0.0328
        assert fused[0]["similarity"] == pytest.approx(2.0 / 61, rel=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hybrid_search.py -v`
Expected: FAIL — `ImportError: cannot import name 'reciprocal_rank_fusion'`

- [ ] **Step 3: Implement RRF function**

Add to `src/knowledge_service/stores/embedding.py` after the imports (before the class):

```python
def reciprocal_rank_fusion(
    *result_lists: list[dict],
    key: str = "id",
    k: int = 60,
    limit: int = 10,
) -> list[dict]:
    """Fuse multiple ranked result lists via Reciprocal Rank Fusion.

    Each item's score = sum(1 / (k + rank + 1)) across all lists it appears in.
    Items appearing in multiple lists score higher than single-list items.
    The fused RRF score replaces the 'similarity' field in the returned dicts.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            item_key = str(item[key])
            scores[item_key] = scores.get(item_key, 0.0) + 1.0 / (k + rank + 1)
            items[item_key] = item
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    fused = []
    for item_key, score in ranked:
        result = dict(items[item_key])
        result["similarity"] = score
        fused.append(result)
    return fused
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hybrid_search.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/stores/embedding.py tests/test_hybrid_search.py
git commit -m "feat: add reciprocal rank fusion function"
```

---

## Task 3: search_bm25() method

**Files:**
- Modify: `src/knowledge_service/stores/embedding.py` (add method to EmbeddingStore)
- Modify: `tests/test_hybrid_search.py`

- [ ] **Step 1: Write failing tests for search_bm25**

Add to `tests/test_hybrid_search.py`:

```python
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_ctx
    return pool, conn


@pytest.fixture
def store(mock_pool):
    from knowledge_service.stores.embedding import EmbeddingStore
    pool, _ = mock_pool
    return EmbeddingStore(pool)


class TestSearchBM25:
    async def test_returns_matching_rows(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {
                "id": "chunk-1",
                "chunk_text": "cold exposure increases dopamine",
                "chunk_index": 0,
                "content_id": "meta-1",
                "url": "http://example.com",
                "title": "Test",
                "summary": None,
                "source_type": "article",
                "tags": [],
                "ingested_at": "2026-01-01",
                "similarity": 0.5,
            }
        ]
        results = await store.search_bm25("dopamine", limit=10)
        assert len(results) == 1
        assert results[0]["id"] == "chunk-1"
        # Verify SQL uses tsv @@ and ts_rank
        sql = conn.fetch.call_args[0][0]
        assert "tsv @@ plainto_tsquery" in sql
        assert "ts_rank" in sql

    async def test_empty_query_returns_empty(self, store, mock_pool):
        """Stop-word-only or empty queries return [] immediately."""
        results = await store.search_bm25("", limit=10)
        assert results == []

    async def test_stop_word_only_returns_empty(self, store, mock_pool):
        """Stop-word-only queries like 'the' degrade gracefully."""
        _, conn = mock_pool
        conn.fetch.return_value = []
        results = await store.search_bm25("the", limit=10)
        # PostgreSQL returns no matches for stop-word-only tsquery;
        # the DB round-trip is acceptable — no client-side tsquery check needed
        assert isinstance(results, list)

    async def test_special_characters_handled(self, store, mock_pool):
        """Special characters in query don't cause SQL errors."""
        _, conn = mock_pool
        conn.fetch.return_value = []
        results = await store.search_bm25("hello!@#$% world", limit=10)
        assert isinstance(results, list)

    async def test_filters_by_source_type(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        await store.search_bm25("test", limit=10, source_type="article")
        sql = conn.fetch.call_args[0][0]
        assert "source_type" in sql
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hybrid_search.py::TestSearchBM25 -v`
Expected: FAIL — `AttributeError: 'EmbeddingStore' object has no attribute 'search_bm25'`

- [ ] **Step 3: Implement search_bm25**

Add to `EmbeddingStore` class in `embedding.py`, after the `search()` method (after line 216):

```python
async def search_bm25(
    self,
    query_text: str,
    limit: int,
    source_type: str | None = None,
    tags: list[str] | None = None,
    min_date: Any | None = None,
) -> list[dict]:
    """Full-text search using PostgreSQL tsvector/tsquery.

    Returns the same dict shape as search() for RRF compatibility.
    Uses plainto_tsquery for safe natural-language query parsing.
    """
    if not query_text or not query_text.strip():
        return []

    conditions: list[str] = ["c.tsv @@ plainto_tsquery('english', $1)"]
    params: list[Any] = [query_text]

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

    where_clause = f"WHERE {' AND '.join(conditions)}"

    sql = f"""
        SELECT
            c.id, c.chunk_text, c.chunk_index,
            m.id AS content_id, m.url, m.title, m.summary,
            m.source_type, m.tags, m.ingested_at,
            ts_rank(c.tsv, plainto_tsquery('english', $1)) AS similarity
        FROM content c
        JOIN content_metadata m ON c.content_id = m.id
        {where_clause}
        ORDER BY ts_rank(c.tsv, plainto_tsquery('english', $1)) DESC
        LIMIT {limit_placeholder}
    """

    async with self._pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [dict(row) for row in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hybrid_search.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/stores/embedding.py tests/test_hybrid_search.py
git commit -m "feat: add search_bm25 method with tsvector full-text search"
```

---

## Task 4: Hybrid mode in search()

**Files:**
- Modify: `src/knowledge_service/stores/embedding.py:165-216` (modify `search()`)
- Modify: `tests/test_hybrid_search.py`

- [ ] **Step 1: Write failing test for hybrid mode**

Add to `tests/test_hybrid_search.py`:

```python
class TestHybridSearch:
    async def test_hybrid_fuses_vector_and_bm25(self, store, mock_pool):
        """When query_text is provided, search runs both and fuses via RRF."""
        _, conn = mock_pool
        # First call = vector search, second call = BM25 search
        conn.fetch.side_effect = [
            [  # vector results
                {
                    "id": "chunk-vec",
                    "chunk_text": "vector match",
                    "chunk_index": 0,
                    "content_id": "m1",
                    "url": "http://a.com",
                    "title": "A",
                    "summary": None,
                    "source_type": "article",
                    "tags": [],
                    "ingested_at": "2026-01-01",
                    "similarity": 0.8,
                },
            ],
            [  # BM25 results
                {
                    "id": "chunk-bm25",
                    "chunk_text": "keyword match",
                    "chunk_index": 0,
                    "content_id": "m2",
                    "url": "http://b.com",
                    "title": "B",
                    "summary": None,
                    "source_type": "article",
                    "tags": [],
                    "ingested_at": "2026-01-01",
                    "similarity": 0.5,
                },
            ],
        ]
        results = await store.search(
            query_embedding=[0.1] * 768,
            limit=10,
            query_text="keyword match",
        )
        # Both results should be in the fused output
        ids = {r["id"] for r in results}
        assert "chunk-vec" in ids
        assert "chunk-bm25" in ids

    async def test_no_query_text_is_vector_only(self, store, mock_pool):
        """Without query_text, search behaves exactly as before (vector only)."""
        _, conn = mock_pool
        conn.fetch.return_value = [
            {
                "id": "chunk-1",
                "chunk_text": "text",
                "chunk_index": 0,
                "content_id": "m1",
                "url": "http://a.com",
                "title": "A",
                "summary": None,
                "source_type": "article",
                "tags": [],
                "ingested_at": "2026-01-01",
                "similarity": 0.9,
            },
        ]
        results = await store.search(query_embedding=[0.1] * 768, limit=10)
        assert len(results) == 1
        # Only one fetch call (vector only, no BM25)
        assert conn.fetch.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hybrid_search.py::TestHybridSearch -v`
Expected: FAIL — `search() got an unexpected keyword argument 'query_text'`

- [ ] **Step 3: Modify search() for hybrid mode**

In `embedding.py`, update `search()` signature and body. Add `query_text: str | None = None` parameter. At the end of the method, before returning:

```python
async def search(
    self,
    query_embedding: list[float],
    limit: int,
    source_type: str | None = None,
    tags: list[str] | None = None,
    min_date: Any | None = None,
    query_text: str | None = None,
) -> list[dict]:
    """Return chunk rows ranked by similarity, joined with content metadata.

    When query_text is provided, runs hybrid search: vector + BM25 fused via RRF.
    When query_text is None, runs vector-only search (backward compatible).
    """
    overfetch = limit * 3 if query_text else limit

    # --- existing vector search logic, but use overfetch instead of limit ---
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

    params.append(overfetch)
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
        vector_rows = await conn.fetch(sql, *params)
    vector_results = [dict(row) for row in vector_rows]

    if not query_text:
        return vector_results[:limit]

    # Hybrid mode: also run BM25 and fuse
    bm25_results = await self.search_bm25(
        query_text=query_text,
        limit=overfetch,
        source_type=source_type,
        tags=tags,
        min_date=min_date,
    )

    return reciprocal_rank_fusion(
        vector_results, bm25_results, key="id", k=60, limit=limit
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hybrid_search.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite to verify backward compatibility**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS (existing tests don't pass `query_text`, so they get vector-only behavior)

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/stores/embedding.py tests/test_hybrid_search.py
git commit -m "feat: hybrid search mode in search() — vector + BM25 via RRF"
```

---

## Task 5: Wire callers to pass query_text

**Files:**
- Modify: `src/knowledge_service/api/search.py:35-40`
- Modify: `src/knowledge_service/stores/rag.py:49-51`

- [ ] **Step 1: Update /api/search endpoint**

In `src/knowledge_service/api/search.py`, change line 35-40:

```python
rows = await embedding_store.search(
    query_embedding=embedding,
    limit=limit,
    source_type=source_type,
    tags=tags,
    query_text=q,
)
```

- [ ] **Step 2: Update RAGRetriever**

In `src/knowledge_service/stores/rag.py`, change line 49-51:

```python
content_results = await self._embedding_store.search(
    query_embedding=embedding, limit=max_sources, query_text=question
)
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Run lint**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/api/search.py src/knowledge_service/stores/rag.py
git commit -m "feat: wire /api/search and RAGRetriever to use hybrid search"
```

---

## Task 6: Final integration test and lint

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Run lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean. If not: `uv run ruff format .`

- [ ] **Step 3: Fix any issues and commit**

```bash
git add -A && git commit -m "chore: lint fixes for BM25 hybrid search"
```
