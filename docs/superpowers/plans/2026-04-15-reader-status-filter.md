# Reader-side Status Filter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude content whose latest `ingestion_jobs.status` is non-terminal from `/api/search`, `/api/ask`, and `RAGRetriever`, so the retriever never serves chunks whose KG triples have not yet committed.

**Architecture:** SQL-side filter in `ContentStore.search()` and `ContentStore._search_bm25()` via `LEFT JOIN LATERAL` against `ingestion_jobs`. Gated behind `settings.reader_exclude_inflight` (env: `READER_EXCLUDE_INFLIGHT`, default `true`). `RAGRetriever` inherits behavior transparently since it calls `ContentStore.search()`. No write-path changes.

**Tech Stack:** Python 3.12, asyncpg, PostgreSQL, pydantic-settings, pytest / pytest-asyncio, `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-04-15-reader-status-filter-design.md`

---

## File Structure

**Modified files:**
- `src/knowledge_service/config.py` — add `reader_exclude_inflight` setting
- `src/knowledge_service/stores/content.py` — thread flag, add LATERAL clause to two queries
- `src/knowledge_service/main.py:151` — pass flag into `ContentStore(...)`
- `API.md` — add paragraph to `/api/search` and `/api/ask` sections
- `CLAUDE.md` — note the invariant under a new "Reader-side status filtering" section
- `tests/test_content_store.py` — existing mock shape stays working (flag off by construction in current mocks — we update the helper)

**New files:**
- `tests/test_reader_status_filter.py` — unit tests spying on executed SQL
- `tests/e2e/test_e2e_reader_status_filter.py` — end-to-end test against real Postgres

---

## Task 1: Add config flag

**Files:**
- Modify: `src/knowledge_service/config.py`

- [ ] **Step 1: Add the setting field**

Edit `src/knowledge_service/config.py`. After the `nlp_entity_confidence` line (around line 33), before `model_config`, add:

```python
    # Reader-side status filtering
    reader_exclude_inflight: bool = True  # env: READER_EXCLUDE_INFLIGHT
```

- [ ] **Step 2: Verify the setting loads**

Run: `uv run python -c "from knowledge_service.config import settings; print(settings.reader_exclude_inflight)"`
Expected: `True`

- [ ] **Step 3: Verify env override works**

Run: `READER_EXCLUDE_INFLIGHT=false uv run python -c "from knowledge_service.config import Settings; s = Settings(); print(s.reader_exclude_inflight)"`
Expected: `False`

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_service/config.py
git commit -m "feat: add reader_exclude_inflight setting"
```

---

## Task 2: Thread flag into ContentStore

**Files:**
- Modify: `src/knowledge_service/stores/content.py:65-69`
- Modify: `tests/test_content_store.py:8-37` (existing `_make_pool` helper; no logic change, just ensure `ContentStore(pool)` construction still works)

- [ ] **Step 1: Update `ContentStore.__init__`**

Edit `src/knowledge_service/stores/content.py`. Change the class header:

```python
class ContentStore:
    """Wraps an asyncpg connection pool for content metadata and chunk operations."""

    def __init__(self, pool: Any, *, exclude_inflight: bool = False) -> None:
        self._pool = pool
        self._exclude_inflight = exclude_inflight
```

Default is `False` at the store level so bare construction in tests and scripts preserves today's recall behavior. Production wiring (Task 6) passes `True` by reading `settings.reader_exclude_inflight`.

- [ ] **Step 2: Run the existing content-store tests**

Run: `uv run pytest tests/test_content_store.py -v`
Expected: PASS (no behavior change yet — `__init__` gained a kwarg with a default).

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_service/stores/content.py
git commit -m "feat: add exclude_inflight kwarg to ContentStore"
```

---

## Task 3: Add LATERAL filter to ContentStore.search (vector path)

**Files:**
- Create: `tests/test_reader_status_filter.py`
- Modify: `src/knowledge_service/stores/content.py:232-305`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reader_status_filter.py`:

```python
"""Tests for reader-side status filtering on ContentStore search queries.

These tests spy on the SQL strings passed to asyncpg and assert the LATERAL
join + status predicate is added when exclude_inflight=True and omitted when
exclude_inflight=False. Functional coverage (actual row exclusion) lives in
tests/e2e/test_e2e_reader_status_filter.py against a real Postgres.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from knowledge_service.stores.content import ContentStore


def _make_pool_capturing_sql():
    """Return (pool, captured) where captured['sql'] is the last SQL passed to fetch()."""
    captured: dict = {"sql": None, "params": None}

    mock_conn = AsyncMock()

    async def _fetch(sql, *params):
        captured["sql"] = sql
        captured["params"] = params
        return []

    mock_conn.fetch = _fetch

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    pool = MagicMock()
    pool.acquire = _acquire
    return pool, captured


_INFLIGHT_PREDICATE = (
    "j.status IS NULL OR j.status IN ('completed', 'failed')"
)


class TestVectorSearchInflightFilter:
    async def test_lateral_and_predicate_added_when_flag_on(self):
        pool, captured = _make_pool_capturing_sql()
        store = ContentStore(pool, exclude_inflight=True)
        await store.search(query_embedding=[0.1] * 768, limit=5)
        sql = captured["sql"]
        assert "LEFT JOIN LATERAL" in sql
        assert "ingestion_jobs" in sql
        assert "ORDER BY created_at DESC" in sql
        assert _INFLIGHT_PREDICATE in sql

    async def test_no_lateral_when_flag_off(self):
        pool, captured = _make_pool_capturing_sql()
        store = ContentStore(pool, exclude_inflight=False)
        await store.search(query_embedding=[0.1] * 768, limit=5)
        sql = captured["sql"]
        assert "LEFT JOIN LATERAL" not in sql
        assert "ingestion_jobs" not in sql
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/test_reader_status_filter.py::TestVectorSearchInflightFilter -v`
Expected: FAIL with `AssertionError: assert 'LEFT JOIN LATERAL' in sql`.

- [ ] **Step 3: Implement the filter in `search()`**

Edit `src/knowledge_service/stores/content.py`. Replace the SQL block in `search()` (around line 282) so the LATERAL join and predicate are appended when `self._exclude_inflight` is `True`.

The full updated `search()` body, from just after the existing `where_clause` assignment to the return, is:

```python
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        if self._exclude_inflight:
            lateral_join = """
                LEFT JOIN LATERAL (
                    SELECT status
                    FROM ingestion_jobs
                    WHERE content_id = m.id
                    ORDER BY created_at DESC
                    LIMIT 1
                ) j ON TRUE
            """
            inflight_predicate = (
                "(j.status IS NULL OR j.status IN ('completed', 'failed'))"
            )
            where_clause = (
                f"{where_clause} AND {inflight_predicate}"
                if where_clause
                else f"WHERE {inflight_predicate}"
            )
        else:
            lateral_join = ""

        sql = f"""
            SELECT
                c.id, c.chunk_text, c.chunk_index, c.section_header,
                m.id AS content_id, m.url, m.title, m.summary,
                m.source_type, m.tags, m.ingested_at,
                1 - (c.embedding::halfvec(768) <=> $1::halfvec(768)) AS similarity
            FROM content c
            JOIN content_metadata m ON c.content_id = m.id
            {lateral_join}
            {where_clause}
            ORDER BY c.embedding::halfvec(768) <=> $1::halfvec(768)
            LIMIT {limit_placeholder}
        """
```

Everything above this block (embedding_str, conditions construction, params.append calls) stays exactly as it is today. The `if query_text:` branch and `return` at the bottom stay exactly as they are today.

- [ ] **Step 4: Run the test and verify pass**

Run: `uv run pytest tests/test_reader_status_filter.py::TestVectorSearchInflightFilter -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full content-store test file to check nothing regressed**

Run: `uv run pytest tests/test_content_store.py -v`
Expected: PASS (existing tests construct `ContentStore(pool)` without the kwarg, so `exclude_inflight=False`, so the LATERAL branch is skipped and the SQL matches what mocks expect).

- [ ] **Step 6: Commit**

```bash
git add tests/test_reader_status_filter.py src/knowledge_service/stores/content.py
git commit -m "feat: LATERAL in-flight filter in ContentStore.search"
```

---

## Task 4: Add LATERAL filter to _search_bm25 (BM25 path)

**Files:**
- Modify: `tests/test_reader_status_filter.py`
- Modify: `src/knowledge_service/stores/content.py:324-380`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reader_status_filter.py`:

```python
class TestBm25SearchInflightFilter:
    async def test_lateral_and_predicate_added_when_flag_on(self):
        pool, captured = _make_pool_capturing_sql()
        store = ContentStore(pool, exclude_inflight=True)
        await store._search_bm25(query_text="caffeine", limit=5)
        sql = captured["sql"]
        assert "LEFT JOIN LATERAL" in sql
        assert "ingestion_jobs" in sql
        assert _INFLIGHT_PREDICATE in sql

    async def test_no_lateral_when_flag_off(self):
        pool, captured = _make_pool_capturing_sql()
        store = ContentStore(pool, exclude_inflight=False)
        await store._search_bm25(query_text="caffeine", limit=5)
        sql = captured["sql"]
        assert "LEFT JOIN LATERAL" not in sql
        assert "ingestion_jobs" not in sql
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/test_reader_status_filter.py::TestBm25SearchInflightFilter -v`
Expected: FAIL with `AssertionError: assert 'LEFT JOIN LATERAL' in sql`.

- [ ] **Step 3: Implement the filter in `_search_bm25()`**

Edit `src/knowledge_service/stores/content.py`. Replace the SQL assembly block in `_search_bm25()` (around line 363-376) so the LATERAL join is conditionally included. The full updated tail of `_search_bm25()`, starting from just after the existing `where_clause` assignment, is:

```python
        where_clause = f"WHERE {' AND '.join(conditions)}"

        if self._exclude_inflight:
            lateral_join = """
                LEFT JOIN LATERAL (
                    SELECT status
                    FROM ingestion_jobs
                    WHERE content_id = m.id
                    ORDER BY created_at DESC
                    LIMIT 1
                ) j ON TRUE
            """
            inflight_predicate = (
                "(j.status IS NULL OR j.status IN ('completed', 'failed'))"
            )
            where_clause = f"{where_clause} AND {inflight_predicate}"
        else:
            lateral_join = ""

        sql = f"""
            SELECT
                c.id, c.chunk_text, c.chunk_index, c.section_header,
                m.id AS content_id, m.url, m.title, m.summary,
                m.source_type, m.tags, m.ingested_at,
                ts_rank(c.tsv, plainto_tsquery('english', $1)) AS similarity
            FROM content c
            JOIN content_metadata m ON c.content_id = m.id
            {lateral_join}
            {where_clause}
            ORDER BY ts_rank(c.tsv, plainto_tsquery('english', $1)) DESC
            LIMIT {limit_placeholder}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [dict(row) for row in rows]
```

Note: `_search_bm25` always has at least the `tsv @@` condition, so `where_clause` is never empty — the bare-`WHERE` branch from Task 3 is not needed here.

- [ ] **Step 4: Run the test and verify pass**

Run: `uv run pytest tests/test_reader_status_filter.py::TestBm25SearchInflightFilter -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run full test suite smoke check**

Run: `uv run pytest tests/test_content_store.py tests/test_reader_status_filter.py tests/test_api_search.py tests/test_api_ask.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_reader_status_filter.py src/knowledge_service/stores/content.py
git commit -m "feat: LATERAL in-flight filter in ContentStore._search_bm25"
```

---

## Task 5: Wire flag into ContentStore construction in main.py

**Files:**
- Modify: `src/knowledge_service/main.py:151`

- [ ] **Step 1: Edit the lifespan Stores construction**

Edit `src/knowledge_service/main.py`. Change line 151 from:

```python
        content=ContentStore(pg_pool),
```

to:

```python
        content=ContentStore(pg_pool, exclude_inflight=settings.reader_exclude_inflight),
```

- [ ] **Step 2: Verify app starts up**

Run: `uv run python -c "from knowledge_service.main import create_app; app = create_app(use_lifespan=False); print('ok')"`
Expected: `ok` (import + factory succeed).

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/ -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_service/main.py
git commit -m "feat: wire reader_exclude_inflight flag into ContentStore"
```

---

## Task 6: End-to-end test against real Postgres

**Files:**
- Create: `tests/e2e/test_e2e_reader_status_filter.py`

- [ ] **Step 1: Inspect the existing e2e test harness**

Run: `head -60 tests/e2e/conftest.py`
Expected: shows the fixtures used to bring up the service against real Postgres + Ollama. Note the fixture names for `pg_pool` and any HTTP client.

- [ ] **Step 2: Write the e2e test**

Create `tests/e2e/test_e2e_reader_status_filter.py`. The test directly inserts three content rows and three job rows into Postgres, builds a `ContentStore(pool, exclude_inflight=True)`, calls `.search()`, and asserts that only the `completed`, `failed`, and no-job rows come back.

```python
"""End-to-end test: reader-side in-flight filter against real Postgres.

Inserts three content_metadata rows and associated ingestion_jobs rows, then
calls ContentStore.search directly and asserts the LATERAL filter excludes
only the non-terminal job.

Requires: running Postgres with the schema migrated. Run via:
    uv run pytest tests/e2e/test_e2e_reader_status_filter.py -v
"""

from __future__ import annotations

import pytest

from knowledge_service.stores.content import ContentStore

pytestmark = pytest.mark.asyncio


async def _insert_content(pool, url: str, chunk_text: str, embedding):
    async with pool.acquire() as conn:
        async with conn.transaction():
            cid = await conn.fetchval(
                """
                INSERT INTO content_metadata (url, title, summary, raw_text, source_type, tags)
                VALUES ($1, $2, NULL, $2, 'article', ARRAY[]::text[])
                ON CONFLICT (url) DO UPDATE SET title = EXCLUDED.title
                RETURNING id
                """,
                url,
                chunk_text,
            )
            await conn.execute("DELETE FROM content WHERE content_id = $1", cid)
            await conn.execute(
                """
                INSERT INTO content (
                    content_id, chunk_index, chunk_text, embedding,
                    char_start, char_end, section_header
                )
                VALUES ($1, 0, $2, $3::vector(768), 0, $4, NULL)
                """,
                cid,
                chunk_text,
                "[" + ",".join(str(v) for v in embedding) + "]",
                len(chunk_text),
            )
            return cid


async def _set_job_status(pool, content_id, status: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM ingestion_jobs WHERE content_id = $1",
            content_id,
        )
        await conn.execute(
            "INSERT INTO ingestion_jobs (content_id, status) VALUES ($1, $2)",
            content_id,
            status,
        )


async def test_lateral_filter_excludes_only_inflight(pg_pool):
    embedding = [0.1] * 768

    completed_id = await _insert_content(
        pg_pool, "http://ks-test/reader-filter/completed", "alpha payload", embedding
    )
    failed_id = await _insert_content(
        pg_pool, "http://ks-test/reader-filter/failed", "alpha payload", embedding
    )
    inflight_id = await _insert_content(
        pg_pool, "http://ks-test/reader-filter/inflight", "alpha payload", embedding
    )
    no_job_id = await _insert_content(
        pg_pool, "http://ks-test/reader-filter/no-job", "alpha payload", embedding
    )

    await _set_job_status(pg_pool, completed_id, "completed")
    await _set_job_status(pg_pool, failed_id, "failed")
    await _set_job_status(pg_pool, inflight_id, "processing")
    # no_job_id: intentionally leave no row in ingestion_jobs

    store = ContentStore(pg_pool, exclude_inflight=True)
    rows = await store.search(query_embedding=embedding, limit=50)
    returned_ids = {str(r["content_id"]) for r in rows}

    assert str(completed_id) in returned_ids
    assert str(failed_id) in returned_ids
    assert str(no_job_id) in returned_ids
    assert str(inflight_id) not in returned_ids


async def test_latest_job_wins(pg_pool):
    embedding = [0.2] * 768
    content_id = await _insert_content(
        pg_pool, "http://ks-test/reader-filter/reingest", "beta payload", embedding
    )

    async with pg_pool.acquire() as conn:
        await conn.execute("DELETE FROM ingestion_jobs WHERE content_id = $1", content_id)
        await conn.execute(
            "INSERT INTO ingestion_jobs (content_id, status, created_at)"
            " VALUES ($1, 'completed', now() - interval '1 hour')",
            content_id,
        )
        await conn.execute(
            "INSERT INTO ingestion_jobs (content_id, status, created_at)"
            " VALUES ($1, 'processing', now())",
            content_id,
        )

    store = ContentStore(pg_pool, exclude_inflight=True)
    rows = await store.search(query_embedding=embedding, limit=50)
    returned_ids = {str(r["content_id"]) for r in rows}
    assert str(content_id) not in returned_ids


async def test_flag_off_returns_all(pg_pool):
    embedding = [0.3] * 768
    content_id = await _insert_content(
        pg_pool, "http://ks-test/reader-filter/flag-off", "gamma payload", embedding
    )
    await _set_job_status(pg_pool, content_id, "processing")

    store = ContentStore(pg_pool, exclude_inflight=False)
    rows = await store.search(query_embedding=embedding, limit=50)
    returned_ids = {str(r["content_id"]) for r in rows}
    assert str(content_id) in returned_ids
```

- [ ] **Step 3: Run the e2e test**

Run: `uv run pytest tests/e2e/test_e2e_reader_status_filter.py -v`
Expected: PASS (all three tests). If Postgres is not running locally, skip this step and run in CI / local with docker-compose up postgres first.

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/test_e2e_reader_status_filter.py
git commit -m "test: e2e reader-side in-flight filter against real Postgres"
```

---

## Task 7: Update API.md

**Files:**
- Modify: `API.md:264-306` (GET /api/search section)
- Modify: `API.md:446-500` (POST /api/ask section)

- [ ] **Step 1: Add paragraph to /api/search**

Edit `API.md`. After line 276 (the query-parameter table row for `tags`) in the `## GET /api/search` section, insert a new sub-section immediately before the `**Response:**` heading (around line 277):

```markdown
**In-flight content is excluded.** Results only include content whose latest
ingestion job has reached a terminal state (`completed` or `failed`), or
content with no recorded job. Mid-pipeline content is filtered out until
embedding, extraction, and processing finish. This behavior is controlled by
the `READER_EXCLUDE_INFLIGHT` environment variable (default `true`).

```

- [ ] **Step 2: Add paragraph to /api/ask**

Edit `API.md`. In the `## POST /api/ask` section, immediately before the `**Status Codes:**` line (around line 499), insert:

```markdown
**In-flight content is excluded.** The hybrid retriever backing this endpoint
only reads content whose latest ingestion job has reached a terminal state
(`completed` or `failed`), or content with no recorded job. Mid-pipeline
content is filtered out until embedding, extraction, and processing finish.
Controlled by `READER_EXCLUDE_INFLIGHT` (default `true`).

```

- [ ] **Step 3: Commit**

```bash
git add API.md
git commit -m "docs: note reader-side in-flight filter on search and ask"
```

---

## Task 8: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (append a new "Reader-side status filtering" section)

- [ ] **Step 1: Append the invariant section**

Edit `CLAUDE.md`. Append a new section after the existing "ProcessPhase consistency" section:

```markdown
## Reader-side status filtering

`ContentStore.search()` and `ContentStore._search_bm25()` filter out content
whose latest `ingestion_jobs.status` is non-terminal (any of `accepted`,
`embedding`, `analyzing`, `extracting`, `resolving`, `processing`). Content
with `completed`, `failed`, or no job row passes through. This prevents the
hybrid retriever from returning chunks whose KG triples have not yet
committed — the "half-picture" problem.

- The filter is applied in SQL via `LEFT JOIN LATERAL` against
  `ingestion_jobs`, ordered by `created_at DESC LIMIT 1` so the latest job
  wins (re-ingest semantics).
- `RAGRetriever` inherits the filter transparently because it calls
  `ContentStore.search()`.
- `/api/content/{id}/chunks` is deliberately exempt — that endpoint reads
  chunks by ID for operator/debug flows and must see in-flight content.
- Controlled by `settings.reader_exclude_inflight` (env:
  `READER_EXCLUDE_INFLIGHT`, default `true`). The flag exists as a
  rollout escape hatch, not a per-request knob.
- `failed` jobs are intentionally included. The outbox 2PC may have
  committed partial triples before the failure; hiding them would remove
  real evidence. Operators promote failed → completed via re-ingest.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md section for reader-side status filtering"
```

---

## Task 9: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire unit-test suite**

Run: `uv run pytest tests/ --ignore=tests/e2e -q`
Expected: PASS.

- [ ] **Step 2: Run ruff check**

Run: `uv run ruff check .`
Expected: no errors.

- [ ] **Step 3: Run ruff format check**

Run: `uv run ruff format --check .`
Expected: no diffs. If there are diffs, run `uv run ruff format .` and include the result in the next step's diff.

- [ ] **Step 4: Review the diff holistically**

Run: `git log --oneline origin/main..HEAD`
Expected: 8 commits corresponding to Tasks 1–8.

Run: `git diff origin/main -- src/knowledge_service/stores/content.py`
Expected: only the changes from Tasks 2, 3, 4 — `__init__` kwarg + two LATERAL-guarded SQL blocks. No drive-by edits to other methods.

- [ ] **Step 5: Open the PR (draft)**

Run:

```bash
gh pr create --draft --title "Reader-side status filter for in-flight content" --body "$(cat <<'EOF'
## Summary

- Excludes content with a non-terminal latest `ingestion_jobs.status` from `/api/search`, `/api/ask`, and `RAGRetriever`.
- SQL-side `LEFT JOIN LATERAL` on `ingestion_jobs` in `ContentStore.search()` and `_search_bm25()`.
- Gated behind `READER_EXCLUDE_INFLIGHT` (default `true`) for rollout safety.
- No write-path changes. `/api/content/{id}/chunks` behavior unchanged.

Spec: `docs/superpowers/specs/2026-04-15-reader-status-filter-design.md`
Plan: `docs/superpowers/plans/2026-04-15-reader-status-filter.md`

## Test plan

- [ ] `uv run pytest tests/ --ignore=tests/e2e -q` passes
- [ ] `uv run pytest tests/e2e/test_e2e_reader_status_filter.py -v` passes against local Postgres
- [ ] `uv run ruff check . && uv run ruff format --check .` clean
- [ ] Manual: curl `/api/search?q=...` against staging, confirm in-flight content is hidden
- [ ] Manual: set `READER_EXCLUDE_INFLIGHT=false`, restart service, confirm in-flight content is visible again

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
