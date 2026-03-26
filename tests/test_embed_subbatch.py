"""Tests for EmbeddingClient.embed_batch sub-batching."""

import pytest
from unittest.mock import AsyncMock

from knowledge_service.clients.llm import EmbeddingClient


@pytest.fixture
def embedding_client():
    return EmbeddingClient(base_url="http://localhost:11434", model="test", api_key="")


class TestEmbedBatchSubbatching:
    async def test_no_batch_size_sends_all_at_once(self, embedding_client):
        """Default behavior: all texts in one request."""
        embedding_client._request = AsyncMock(return_value=[[0.1] * 768 for _ in range(5)])
        result = await embedding_client.embed_batch(["t1", "t2", "t3", "t4", "t5"])
        assert len(result) == 5
        embedding_client._request.assert_called_once()

    async def test_batch_size_splits_requests(self, embedding_client):
        """With batch_size=2, 5 texts -> 3 calls (2+2+1)."""
        embedding_client._request = AsyncMock(
            side_effect=[
                [[0.1] * 768, [0.2] * 768],
                [[0.3] * 768, [0.4] * 768],
                [[0.5] * 768],
            ]
        )
        result = await embedding_client.embed_batch(
            ["t1", "t2", "t3", "t4", "t5"], batch_size=2
        )
        assert len(result) == 5
        assert embedding_client._request.call_count == 3

    async def test_batch_size_none_is_default(self, embedding_client):
        """batch_size=None behaves like no batching."""
        embedding_client._request = AsyncMock(return_value=[[0.1] * 768 for _ in range(3)])
        result = await embedding_client.embed_batch(["a", "b", "c"], batch_size=None)
        assert len(result) == 3
        embedding_client._request.assert_called_once()

    async def test_batch_size_larger_than_input(self, embedding_client):
        """batch_size > len(texts) -> one call."""
        embedding_client._request = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])
        result = await embedding_client.embed_batch(["a", "b"], batch_size=100)
        assert len(result) == 2
        embedding_client._request.assert_called_once()

    async def test_empty_input(self, embedding_client):
        """Empty list returns empty list, no calls."""
        result = await embedding_client.embed_batch([], batch_size=20)
        assert result == []
