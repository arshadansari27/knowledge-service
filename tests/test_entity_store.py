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


class TestEntityAliasResolution:
    def _make_pool_with_alias(self, alias: str, canonical: str):
        """Return a pool whose fetchrow returns the alias row for the given alias."""
        from contextlib import asynccontextmanager

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []
        mock_conn.execute.return_value = "INSERT 0 1"

        async def _fetchrow(sql, param, *args):
            # Only the alias lookup passes a single positional param equal to alias
            if param == alias:
                return {"canonical": canonical}
            return None

        mock_conn.fetchrow.side_effect = _fetchrow

        @asynccontextmanager
        async def _acquire():
            yield mock_conn

        pool = MagicMock()
        pool.acquire = _acquire
        return pool, mock_conn

    async def test_resolve_entity_checks_alias_before_embedding(self):
        canonical = f"{KS_DATA}narendra_modi"
        pool, conn = self._make_pool_with_alias("modi", canonical)
        client = _make_embedding_client()
        store = EntityStore(pool, client)

        uri = await store.resolve_entity("Modi")

        assert uri == canonical
        # embed must NOT be called — alias lookup short-circuits
        client.embed.assert_not_called()

    async def test_resolve_entity_caches_alias_result(self):
        canonical = f"{KS_DATA}narendra_modi"
        pool, conn = self._make_pool_with_alias("modi", canonical)
        client = _make_embedding_client()
        store = EntityStore(pool, client)

        uri1 = await store.resolve_entity("Modi")
        uri2 = await store.resolve_entity("Modi")

        assert uri1 == uri2 == canonical
        # fetchrow called only once — second call is LRU cache hit
        assert conn.fetchrow.call_count == 1
        client.embed.assert_not_called()

    async def test_resolve_entity_falls_through_when_no_alias(self):
        pool, conn = _make_pool()
        # fetchrow returns None (default) — no alias exists
        client = _make_embedding_client()
        store = EntityStore(pool, client)

        uri = await store.resolve_entity("ACME Corp")

        # Falls through to embedding path
        client.embed.assert_called_once_with("ACME Corp")
        assert uri == f"{KS_DATA}acme_corp"
