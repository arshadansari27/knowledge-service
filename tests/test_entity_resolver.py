import asyncio

import pytest
from unittest.mock import AsyncMock
from knowledge_service.stores.entity_resolver import EntityResolver


@pytest.fixture
def store():
    """In-memory KnowledgeStore for tests."""
    from knowledge_service.stores.knowledge import KnowledgeStore

    return KnowledgeStore(data_dir=None)


@pytest.fixture
def resolver(store):
    """EntityResolver with real KnowledgeStore and mock EmbeddingStore."""
    mock_embedding_store = AsyncMock()
    mock_embedding_client = AsyncMock()
    # Simulate embedding returning a vector
    mock_embedding_client.embed.return_value = [0.1] * 768
    # Simulate no existing similar entities initially
    mock_embedding_store.search_entities.return_value = []
    return EntityResolver(
        knowledge_store=store,
        embedding_store=mock_embedding_store,
        embedding_client=mock_embedding_client,
    )


class TestResolveEntity:
    async def test_new_entity_creates_uri(self, resolver):
        uri = await resolver.resolve("PostgreSQL 16", "schema:SoftwareApplication")
        assert uri.startswith("http://knowledge.local/data/")
        assert "postgresql" in uri.lower()

    async def test_existing_entity_returns_same_uri(self, resolver):
        # First resolve creates it
        uri1 = await resolver.resolve("PostgreSQL 16", "schema:SoftwareApplication")
        # Mock embedding store to return the entity we just created
        resolver._embedding_store.search_entities.return_value = [
            {"uri": uri1, "label": "PostgreSQL 16", "similarity": 0.95}
        ]
        # Second resolve with slight variation should match
        uri2 = await resolver.resolve("PostgreSQL v16", "schema:SoftwareApplication")
        assert uri1 == uri2

    async def test_low_similarity_creates_new_entity(self, resolver):
        uri1 = await resolver.resolve("PostgreSQL 16", "schema:SoftwareApplication")
        resolver._embedding_store.search_entities.return_value = [
            {"uri": uri1, "label": "PostgreSQL 16", "similarity": 0.4}  # Too low
        ]
        uri2 = await resolver.resolve("MySQL 8", "schema:SoftwareApplication")
        assert uri1 != uri2

    async def test_slugify_produces_valid_uri_segment(self, resolver):
        """URI segment should be lowercase alphanumeric with underscores."""
        uri = await resolver.resolve("My Cool Entity!", "schema:Thing")
        # Segment after the base URI should be a valid slug
        segment = uri.replace("http://knowledge.local/data/", "")
        assert segment == segment.lower()
        assert all(c.isalnum() or c == "_" for c in segment)

    async def test_resolve_without_rdf_type(self, resolver):
        """Resolve should work when rdf_type is None."""
        uri = await resolver.resolve("Some Entity")
        assert uri.startswith("http://knowledge.local/data/")

    async def test_embedding_stored_for_new_entity(self, resolver):
        """insert_entity_embedding should be called when creating a new entity."""
        await resolver.resolve("PostgreSQL 16", "schema:SoftwareApplication")
        resolver._embedding_store.insert_entity_embedding.assert_called_once()
        call_kwargs = resolver._embedding_store.insert_entity_embedding.call_args
        assert call_kwargs.kwargs["label"] == "PostgreSQL 16"
        assert call_kwargs.kwargs["rdf_type"] == "schema:SoftwareApplication"
        assert call_kwargs.kwargs["embedding"] == [0.1] * 768

    async def test_embedding_not_stored_for_existing_entity(self, resolver):
        """insert_entity_embedding should not be called when reusing an existing URI."""
        uri1 = await resolver.resolve("PostgreSQL 16", "schema:SoftwareApplication")
        # Reset the mock call count
        resolver._embedding_store.insert_entity_embedding.reset_mock()
        # Return a high-similarity match
        resolver._embedding_store.search_entities.return_value = [
            {"uri": uri1, "label": "PostgreSQL 16", "similarity": 0.97}
        ]
        await resolver.resolve("PostgreSQL v16", "schema:SoftwareApplication")
        resolver._embedding_store.insert_entity_embedding.assert_not_called()

    async def test_threshold_boundary_exact(self, resolver):
        """A candidate at exactly the threshold should be accepted."""
        uri1 = await resolver.resolve("PostgreSQL 16", "schema:SoftwareApplication")
        resolver._embedding_store.search_entities.return_value = [
            {"uri": uri1, "label": "PostgreSQL 16", "similarity": 0.85}
        ]
        uri2 = await resolver.resolve("Postgres 16", "schema:SoftwareApplication")
        assert uri1 == uri2

    async def test_threshold_boundary_just_below(self, resolver):
        """A candidate just below the threshold should be rejected."""
        uri1 = await resolver.resolve("PostgreSQL 16", "schema:SoftwareApplication")
        resolver._embedding_store.search_entities.return_value = [
            {"uri": uri1, "label": "PostgreSQL 16", "similarity": 0.84}
        ]
        uri2 = await resolver.resolve("Postgres 16", "schema:SoftwareApplication")
        assert uri1 != uri2


@pytest.fixture
def federation_client():
    """Mock FederationClient."""
    mock = AsyncMock()
    mock.lookup_entity.return_value = None
    return mock


@pytest.fixture
def resolver_with_federation(store, federation_client):
    """EntityResolver with federation enabled."""
    mock_embedding_store = AsyncMock()
    mock_embedding_client = AsyncMock()
    mock_embedding_client.embed.return_value = [0.1] * 768
    mock_embedding_store.search_entities.return_value = []
    return EntityResolver(
        knowledge_store=store,
        embedding_store=mock_embedding_store,
        embedding_client=mock_embedding_client,
        federation_client=federation_client,
    )


class TestFederationFallback:
    async def test_federation_hit_returns_local_uri(
        self, resolver_with_federation, federation_client
    ):
        """When local miss + federation hit, returns local URI (not external)."""
        federation_client.lookup_entity.return_value = {
            "uri": "http://dbpedia.org/resource/PostgreSQL",
            "rdf_type": "http://dbpedia.org/ontology/Software",
            "description": "PostgreSQL is a database.",
        }
        uri = await resolver_with_federation.resolve("PostgreSQL")
        assert uri.startswith("http://knowledge.local/data/")
        assert "postgresql" in uri.lower()

    async def test_federation_hit_stores_owl_sameas(
        self, resolver_with_federation, federation_client, store
    ):
        """Federation hit should store an owl:sameAs triple."""
        federation_client.lookup_entity.return_value = {
            "uri": "http://dbpedia.org/resource/PostgreSQL",
            "rdf_type": "http://dbpedia.org/ontology/Software",
            "description": "PostgreSQL is a database.",
        }
        uri = await resolver_with_federation.resolve("PostgreSQL")
        # Check owl:sameAs triple was stored (in a named graph)
        triples = store.query(f"""
            SELECT ?o WHERE {{
                GRAPH ?g {{
                    <{uri}> <http://www.w3.org/2002/07/owl#sameAs> ?o .
                }}
            }}
        """)
        assert len(triples) == 1
        assert triples[0]["o"].value == "http://dbpedia.org/resource/PostgreSQL"

    async def test_federation_hit_stores_rdf_type_in_embedding(
        self, resolver_with_federation, federation_client
    ):
        """Federation hit should pass rdf_type to insert_entity_embedding."""
        federation_client.lookup_entity.return_value = {
            "uri": "http://dbpedia.org/resource/PostgreSQL",
            "rdf_type": "http://dbpedia.org/ontology/Software",
            "description": "PostgreSQL is a database.",
        }
        await resolver_with_federation.resolve("PostgreSQL")
        call_kwargs = (
            resolver_with_federation._embedding_store.insert_entity_embedding.call_args.kwargs
        )
        assert call_kwargs["rdf_type"] == "http://dbpedia.org/ontology/Software"

    async def test_federation_timeout_falls_back_to_local(
        self, resolver_with_federation, federation_client
    ):
        """When federation times out, create local URI (same as no federation)."""
        federation_client.lookup_entity.return_value = None
        uri = await resolver_with_federation.resolve("PostgreSQL")
        assert uri.startswith("http://knowledge.local/data/")
        # Should still store embedding
        resolver_with_federation._embedding_store.insert_entity_embedding.assert_called_once()

    async def test_no_federation_client_skips_federation(self, resolver):
        """When federation_client is None, resolve works as before."""
        uri = await resolver.resolve("PostgreSQL")
        assert uri.startswith("http://knowledge.local/data/")


@pytest.fixture
def resolver_with_federation_enrichment(store, federation_client):
    """EntityResolver with federation enabled and enrich_entity configured."""
    mock_embedding_store = AsyncMock()
    mock_embedding_client = AsyncMock()
    mock_embedding_client.embed.return_value = [0.1] * 768
    mock_embedding_store.search_entities.return_value = []
    federation_client.lookup_entity.return_value = {
        "uri": "http://dbpedia.org/resource/PostgreSQL",
        "rdf_type": "http://dbpedia.org/ontology/Software",
        "description": "PostgreSQL is a database.",
    }
    federation_client.enrich_entity.return_value = [
        {
            "subject": "http://dbpedia.org/resource/PostgreSQL",
            "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "object": "http://dbpedia.org/ontology/Software",
        },
        {
            "subject": "http://dbpedia.org/resource/PostgreSQL",
            "predicate": "http://www.w3.org/2002/07/owl#sameAs",
            "object": "http://www.wikidata.org/entity/Q192490",
        },
    ]
    return EntityResolver(
        knowledge_store=store,
        embedding_store=mock_embedding_store,
        embedding_client=mock_embedding_client,
        federation_client=federation_client,
    )


class TestFederationEnrichment:
    async def test_enrichment_stores_triples_in_federated_graph(
        self, resolver_with_federation_enrichment, federation_client, store
    ):
        """Federation enrichment should store external triples in ks:graph/federated."""
        await resolver_with_federation_enrichment.resolve("PostgreSQL")

        # Wait for fire-and-forget task to complete
        await asyncio.sleep(0.1)

        # Check that enrichment triples are in the federated graph
        triples = store.query("""
            SELECT ?s ?p ?o WHERE {
                GRAPH <http://knowledge.local/schema/graph/federated> {
                    ?s ?p ?o .
                }
            }
        """)
        # Should have at least the 2 enrichment triples (type + sameAs)
        assert len(triples) >= 2
        federation_client.enrich_entity.assert_called_once_with(
            "http://dbpedia.org/resource/PostgreSQL"
        )

    async def test_enrichment_failure_does_not_affect_resolution(
        self, resolver_with_federation_enrichment, federation_client
    ):
        """If enrich_entity raises, resolve() still returns a valid URI."""
        federation_client.enrich_entity.side_effect = Exception("network error")
        uri = await resolver_with_federation_enrichment.resolve("PostgreSQL")
        assert uri.startswith("http://knowledge.local/data/")
        # Wait for task to complete (it should swallow the error)
        await asyncio.sleep(0.1)

    async def test_enrichment_updates_job_counter(
        self, resolver_with_federation_enrichment, federation_client, store
    ):
        """When job_id and pg_pool are provided, federation_enriched counter is updated."""
        from unittest.mock import MagicMock

        mock_conn = AsyncMock()
        mock_acquire_cm = MagicMock()
        mock_acquire_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire_cm.__aexit__ = AsyncMock(return_value=False)
        mock_pg_pool = MagicMock()
        mock_pg_pool.acquire.return_value = mock_acquire_cm

        await resolver_with_federation_enrichment.resolve(
            "PostgreSQL", job_id="test-job-id", pg_pool=mock_pg_pool
        )
        # Wait for fire-and-forget task
        await asyncio.sleep(0.1)

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "federation_enriched" in call_args[0][0]
        assert call_args[0][1] == 2  # 2 triples enriched

    async def test_no_enrichment_without_federation_hit(
        self, resolver_with_federation, federation_client
    ):
        """When federation lookup returns None, no enrichment task is spawned."""
        federation_client.lookup_entity.return_value = None
        await resolver_with_federation.resolve("PostgreSQL")
        federation_client.enrich_entity.assert_not_called()
