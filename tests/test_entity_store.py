# tests/test_entity_store.py
from unittest.mock import AsyncMock, MagicMock
from knowledge_service.stores.entities import EntityStore
from knowledge_service.ontology.uri import KS_DATA, KS


def _make_pool():
    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []
    mock_conn.fetchrow.return_value = None
    mock_conn.execute.return_value = "INSERT 0 1"

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    pool = MagicMock()
    pool.acquire = _acquire
    return pool, mock_conn


def _make_embedding_client():
    client = AsyncMock()
    client.embed.return_value = [0.1] * 768
    return client


class TestResolveEntity:
    async def test_creates_new_uri_when_no_match(self):
        pool, conn = _make_pool()
        store = EntityStore(pool, _make_embedding_client())
        uri = await store.resolve_entity("ACME Corp")
        assert uri == f"{KS_DATA}acme_corp"

    async def test_returns_existing_uri_on_high_similarity(self):
        pool, conn = _make_pool()
        conn.fetch.return_value = [
            {"uri": f"{KS_DATA}acme_corporation", "label": "ACME Corporation", "similarity": 0.92}
        ]
        store = EntityStore(pool, _make_embedding_client())
        uri = await store.resolve_entity("ACME Corp")
        assert uri == f"{KS_DATA}acme_corporation"

    async def test_cache_hit(self):
        pool, conn = _make_pool()
        client = _make_embedding_client()
        store = EntityStore(pool, client)
        await store.resolve_entity("ACME Corp")
        await store.resolve_entity("ACME Corp")
        # embed called only once (second is cache hit)
        assert client.embed.call_count == 1


class TestResolvePredicate:
    async def test_creates_new_uri_when_no_match(self):
        pool, conn = _make_pool()
        store = EntityStore(pool, _make_embedding_client())
        uri = await store.resolve_predicate("causes")
        assert uri == f"{KS}causes"
