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
        # Check owl:sameAs triple was stored
        triples = store.query(f"""
            SELECT ?o WHERE {{
                <{uri}> <http://www.w3.org/2002/07/owl#sameAs> ?o .
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
