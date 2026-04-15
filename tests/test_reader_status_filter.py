"""Tests for reader-side status filtering on ContentStore search queries.

These tests spy on the SQL strings passed to asyncpg and assert the LATERAL
join + status predicate is added when exclude_inflight=True and omitted when
exclude_inflight=False. Functional coverage (actual row exclusion) lives in
tests/e2e/test_e2e_reader_status_filter.py against a real Postgres.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

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


_INFLIGHT_PREDICATE = "(j.status IS NULL OR j.status IN ('completed', 'failed'))"


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
