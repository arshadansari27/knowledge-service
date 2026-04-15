"""End-to-end test: reader-side in-flight filter against real Postgres.

Inserts four content_metadata rows with associated ingestion_jobs, calls
ContentStore.search directly, and asserts the LATERAL filter excludes only
the non-terminal job. Exercises:
    - completed (included)
    - failed (included)
    - processing / in-flight (excluded)
    - no job row at all (included)
    - re-ingest: prior completed then current processing (excluded; latest job wins)
    - flag off: in-flight is visible again

Requires: running Postgres with the schema migrated. Run with:
    uv run pytest tests/e2e/test_e2e_reader_status_filter.py -v
"""

from __future__ import annotations

import pytest

from knowledge_service.stores.content import ContentStore

pytestmark = pytest.mark.asyncio


_TEST_URL_PREFIX = "http://ks-test/reader-filter/"


@pytest.fixture(autouse=True)
async def _cleanup_reader_filter_rows(pg_pool):
    """Remove leftover reader-filter test rows before and after each test.

    Test URLs reuse stable paths (ON CONFLICT UPDATE), so prior runs can leave
    stale ingestion_jobs that would flip assertions — notably the no-job case
    in test_lateral_filter_excludes_only_inflight. DELETE on content_metadata
    cascades to content and ingestion_jobs.
    """

    async def _purge():
        async with pg_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM content_metadata WHERE url LIKE $1",
                _TEST_URL_PREFIX + "%",
            )

    await _purge()
    yield
    await _purge()


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
        await conn.execute("DELETE FROM ingestion_jobs WHERE content_id = $1", content_id)
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
