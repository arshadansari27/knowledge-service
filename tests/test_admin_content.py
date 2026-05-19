"""Tests for the admin content-deletion endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from knowledge_service.admin.content import router as content_router


def _make_conn(fetchrow=None, fetch=None, fetchval=None, execute=None) -> AsyncMock:
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=fetchrow)
    conn.fetch = AsyncMock(return_value=fetch or [])
    if isinstance(fetchval, list):
        conn.fetchval = AsyncMock(side_effect=fetchval)
    else:
        conn.fetchval = AsyncMock(return_value=fetchval or 0)
    conn.execute = AsyncMock(return_value=execute or "DELETE 1")
    return conn


def _make_pool(conn) -> MagicMock:
    pool = MagicMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = cm
    return pool


def _make_app(conn, knowledge_store=None) -> FastAPI:
    app = FastAPI()
    app.include_router(content_router, prefix="/api/admin")
    app.state.pg_pool = _make_pool(conn)
    app.state.knowledge_store = knowledge_store or MagicMock()
    return app


@pytest.fixture
def _patch_retraction(monkeypatch):
    """Capture calls to the pyoxigraph-side helpers so we don't need a real store."""
    removed: list[tuple[str, str, str]] = []
    retracted: list[str] = []

    def _fake_remove(store, s, p, o):
        removed.append((s, p, o))

    def _fake_retract(triple_hash, store):
        retracted.append(triple_hash)
        return 0

    monkeypatch.setattr(
        "knowledge_service.admin.content._remove_orphan_triple",
        _fake_remove,
    )
    monkeypatch.setattr(
        "knowledge_service.admin.content.retract_stale_inferences",
        _fake_retract,
    )
    return removed, retracted


async def test_returns_404_when_content_missing(_patch_retraction):
    conn = _make_conn(fetchrow=None)
    app = _make_app(conn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/admin/knowledge/content/does-not-exist")
    assert resp.status_code == 404
    # Must NOT touch provenance / content_metadata if 404
    conn.execute.assert_not_called()


async def test_orphan_triples_removed_and_inferences_retracted(_patch_retraction):
    removed, retracted = _patch_retraction
    # One provenance row, no other sources for the triple → orphan, must be removed.
    conn = _make_conn(
        fetchrow={"url": "https://example.com/test"},
        fetch=[{"triple_hash": "h1", "subject": "s1", "predicate": "p1", "object": "o1"}],
        fetchval=0,  # zero "other" provenance sources
    )
    app = _make_app(conn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/admin/knowledge/content/c1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["content_id"] == "c1"
    assert body["url"] == "https://example.com/test"
    assert body["deleted_provenance_rows"] == 1
    assert body["deleted_triples"] == 1
    assert body["retained_triples_with_other_sources"] == 0
    assert removed == [("s1", "p1", "o1")]
    assert retracted == ["h1"]


async def test_shared_triples_are_retained(_patch_retraction):
    removed, _retracted = _patch_retraction
    # Two provenance rows: one orphan, one with another source still alive.
    conn = _make_conn(
        fetchrow={"url": "https://example.com/test"},
        fetch=[
            {"triple_hash": "orphan", "subject": "s1", "predicate": "p1", "object": "o1"},
            {"triple_hash": "shared", "subject": "s2", "predicate": "p2", "object": "o2"},
        ],
        # fetchval is called per-triple to count other sources;
        # alternate: orphan -> 0, shared -> 1
        fetchval=[0, 1],
    )
    app = _make_app(conn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/admin/knowledge/content/c1")
    body = resp.json()
    assert body["deleted_provenance_rows"] == 2
    assert body["deleted_triples"] == 1
    assert body["retained_triples_with_other_sources"] == 1
    # Only the orphan SPO got removed
    assert removed == [("s1", "p1", "o1")]


async def test_content_with_no_triples(_patch_retraction):
    removed, retracted = _patch_retraction
    conn = _make_conn(
        fetchrow={"url": "https://example.com/empty"},
        fetch=[],  # no provenance rows
    )
    app = _make_app(conn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/admin/knowledge/content/c1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_triples"] == 0
    assert body["deleted_provenance_rows"] == 0
    assert removed == []
    assert retracted == []
    # content_metadata still gets DELETEd (DELETE provenance + DELETE content_metadata = 2 executes)
    assert conn.execute.await_count == 2


async def test_provenance_scoped_by_source_url(_patch_retraction):
    """Regression: both DELETEs must use source_url. Provenance is keyed by
    source_url; content_metadata is deleted by url (not id) so the same
    helper works for both content-id and source-url entry points."""
    conn = _make_conn(
        fetchrow={"url": "https://example.com/scoped"},
        fetch=[],
    )
    app = _make_app(conn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.delete("/api/admin/knowledge/content/c1")
    executed_queries = [call.args for call in conn.execute.await_args_list]
    url = "https://example.com/scoped"
    assert any("DELETE FROM provenance" in q[0] and q[1] == url for q in executed_queries), (
        f"expected DELETE FROM provenance scoped by source_url, got {executed_queries}"
    )
    assert any("DELETE FROM content_metadata" in q[0] and q[1] == url for q in executed_queries), (
        f"expected DELETE FROM content_metadata scoped by url, got {executed_queries}"
    )


# --- DELETE /api/admin/knowledge/source?source_url=... ---


async def test_source_endpoint_404_when_no_provenance(_patch_retraction):
    conn = _make_conn(fetchval=0)  # COUNT(*) = 0
    app = _make_app(conn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete(
            "/api/admin/knowledge/source", params={"source_url": "https://test.local/missing"}
        )
    assert resp.status_code == 404
    conn.execute.assert_not_called()


async def test_source_endpoint_happy_path(_patch_retraction):
    removed, retracted = _patch_retraction
    # First fetchval is the existence COUNT(*); subsequent are per-triple other-source counts.
    conn = _make_conn(
        fetch=[{"triple_hash": "h1", "subject": "s1", "predicate": "p1", "object": "o1"}],
        fetchval=[1, 0],  # exists=1, then other-sources=0
    )
    app = _make_app(conn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete(
            "/api/admin/knowledge/source",
            params={"source_url": "https://test.local/pr74-validation/alias-1"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["url"] == "https://test.local/pr74-validation/alias-1"
    assert body["deleted_provenance_rows"] == 1
    assert body["deleted_triples"] == 1
    assert "content_id" not in body  # source endpoint doesn't echo a content_id
    assert removed == [("s1", "p1", "o1")]
    assert retracted == ["h1"]


async def test_source_endpoint_requires_source_url_query():
    conn = _make_conn()
    app = _make_app(conn)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/admin/knowledge/source")
    assert resp.status_code == 422  # FastAPI: missing required query param
