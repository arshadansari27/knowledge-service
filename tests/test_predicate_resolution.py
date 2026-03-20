"""Tests for predicate resolution via EntityResolver.resolve_predicate()."""

import pytest
from unittest.mock import AsyncMock

from knowledge_service.stores.entity_resolver import EntityResolver


@pytest.fixture
def store():
    from knowledge_service.stores.knowledge import KnowledgeStore

    return KnowledgeStore(data_dir=None)


@pytest.fixture
def resolver(store):
    mock_embedding_store = AsyncMock()
    mock_embedding_client = AsyncMock()
    mock_embedding_client.embed.return_value = [0.1] * 768
    mock_embedding_store.search_predicates.return_value = []
    mock_embedding_store.search_entities.return_value = []
    return EntityResolver(
        knowledge_store=store,
        embedding_store=mock_embedding_store,
        embedding_client=mock_embedding_client,
    )


class TestResolvePredicateSynonym:
    async def test_known_synonym_returns_canonical_uri(self, resolver):
        """'boosts' should resolve to ks:increases without any embedding call."""
        uri = await resolver.resolve_predicate("boosts")
        assert uri == "http://knowledge.local/schema/increases"
        # No embedding call needed for exact synonym match
        resolver._embedding_client.embed.assert_not_called()

    async def test_canonical_predicate_returns_own_uri(self, resolver):
        """A canonical predicate like 'increases' should resolve to itself."""
        uri = await resolver.resolve_predicate("increases")
        assert uri == "http://knowledge.local/schema/increases"
        resolver._embedding_client.embed.assert_not_called()

    async def test_synonym_leads_to_resolves_to_causes(self, resolver):
        uri = await resolver.resolve_predicate("leads_to")
        assert uri == "http://knowledge.local/schema/causes"


class TestResolvePredicateEmbedding:
    async def test_similar_predicate_resolved_by_embedding(self, resolver):
        """When no synonym match, fall back to embedding similarity search."""
        resolver._embedding_store.search_predicates.return_value = [
            {
                "uri": "http://knowledge.local/schema/increases",
                "label": "increases",
                "similarity": 0.95,
            }
        ]
        uri = await resolver.resolve_predicate("heightens")
        assert uri == "http://knowledge.local/schema/increases"
        resolver._embedding_client.embed.assert_called_once_with("heightens")

    async def test_low_similarity_creates_new_predicate(self, resolver):
        """Below threshold (0.90) should create a new predicate."""
        resolver._embedding_store.search_predicates.return_value = [
            {
                "uri": "http://knowledge.local/schema/increases",
                "label": "increases",
                "similarity": 0.85,
            }
        ]
        uri = await resolver.resolve_predicate("correlates_with")
        assert uri == "http://knowledge.local/schema/correlates_with"
        resolver._embedding_store.insert_predicate_embedding.assert_called_once()

    async def test_unknown_predicate_creates_new_uri(self, resolver):
        """Completely new predicate with no matches should be created."""
        uri = await resolver.resolve_predicate("modulates")
        assert uri == "http://knowledge.local/schema/modulates"
        resolver._embedding_store.insert_predicate_embedding.assert_called_once()
        call_kwargs = resolver._embedding_store.insert_predicate_embedding.call_args.kwargs
        assert call_kwargs["label"] == "modulates"
        assert call_kwargs["uri"] == "http://knowledge.local/schema/modulates"

    async def test_embedding_not_stored_for_matched_predicate(self, resolver):
        """When an existing predicate matches, don't store a duplicate embedding."""
        resolver._embedding_store.search_predicates.return_value = [
            {"uri": "http://knowledge.local/schema/causes", "label": "causes", "similarity": 0.92}
        ]
        await resolver.resolve_predicate("triggers_effect")
        resolver._embedding_store.insert_predicate_embedding.assert_not_called()

    async def test_threshold_boundary_exact(self, resolver):
        """Exactly at threshold (0.90) should match."""
        resolver._embedding_store.search_predicates.return_value = [
            {
                "uri": "http://knowledge.local/schema/inhibits",
                "label": "inhibits",
                "similarity": 0.90,
            }
        ]
        uri = await resolver.resolve_predicate("hinders")
        assert uri == "http://knowledge.local/schema/inhibits"

    async def test_threshold_boundary_just_below(self, resolver):
        """Just below threshold (0.89) should NOT match."""
        resolver._embedding_store.search_predicates.return_value = [
            {
                "uri": "http://knowledge.local/schema/inhibits",
                "label": "inhibits",
                "similarity": 0.89,
            }
        ]
        uri = await resolver.resolve_predicate("hinders")
        assert uri == "http://knowledge.local/schema/hinders"
