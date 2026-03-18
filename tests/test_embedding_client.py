import json

import pytest

from knowledge_service.clients.llm import EmbeddingClient, LLMClientError

_BASE = "http://llm-test"
_KEY = "sk-test"


@pytest.fixture
def mock_llm(httpx_mock):
    httpx_mock.add_response(
        url=f"{_BASE}/v1/embeddings",
        json={
            "data": [
                {"embedding": [0.1, 0.2, 0.3] * 256, "index": 0},
                {"embedding": [0.4, 0.5, 0.6] * 256, "index": 1},
            ]
        },
    )
    return httpx_mock


class TestEmbed:
    async def test_embed_returns_vector(self, mock_llm):
        client = EmbeddingClient(base_url=_BASE, model="nomic-embed-text", api_key=_KEY)
        result = await client.embed("test text")
        assert len(result) == 768
        await client.close()

    async def test_embed_batch(self, mock_llm):
        client = EmbeddingClient(base_url=_BASE, model="nomic-embed-text", api_key=_KEY)
        results = await client.embed_batch(["text1", "text2"])
        assert len(results) == 2
        await client.close()

    async def test_embed_returns_floats(self, mock_llm):
        client = EmbeddingClient(base_url=_BASE, model="nomic-embed-text", api_key=_KEY)
        result = await client.embed("test text")
        assert all(isinstance(v, float) for v in result)
        await client.close()

    async def test_embed_batch_each_vector_correct_dim(self, mock_llm):
        client = EmbeddingClient(base_url=_BASE, model="nomic-embed-text", api_key=_KEY)
        results = await client.embed_batch(["text1", "text2"])
        assert all(len(vec) == 768 for vec in results)
        await client.close()

    async def test_custom_model_sent_in_request(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{_BASE}/v1/embeddings",
            json={"data": [{"embedding": [0.1] * 768, "index": 0}]},
        )
        client = EmbeddingClient(base_url=_BASE, model="custom-model", api_key=_KEY)
        await client.embed("test text")
        body = json.loads(httpx_mock.get_requests()[0].content)
        assert body["model"] == "custom-model"
        await client.close()

    async def test_embed_sends_correct_payload(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{_BASE}/v1/embeddings",
            json={"data": [{"embedding": [0.1] * 768, "index": 0}]},
        )
        client = EmbeddingClient(base_url=_BASE, model="nomic-embed-text", api_key=_KEY)
        await client.embed("hello world")
        body = json.loads(httpx_mock.get_requests()[0].content)
        assert body["input"] == ["hello world"]
        await client.close()

    async def test_auth_header_sent(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{_BASE}/v1/embeddings",
            json={"data": [{"embedding": [0.1] * 768, "index": 0}]},
        )
        client = EmbeddingClient(base_url=_BASE, model="nomic-embed-text", api_key="sk-mykey")
        await client.embed("test")
        headers = httpx_mock.get_requests()[0].headers
        assert headers["authorization"] == "Bearer sk-mykey"
        await client.close()

    async def test_close_is_idempotent(self, mock_llm):
        client = EmbeddingClient(base_url=_BASE, model="nomic-embed-text", api_key=_KEY)
        await client.embed("test")
        await client.close()
        await client.close()  # should not raise

    async def test_embed_raises_llm_error_on_server_error(self, httpx_mock):
        httpx_mock.add_response(url=f"{_BASE}/v1/embeddings", status_code=500)
        client = EmbeddingClient(base_url=_BASE, model="nomic-embed-text", api_key=_KEY)
        with pytest.raises(LLMClientError, match="500"):
            await client.embed("test")
        await client.close()


class TestNoAuth:
    async def test_no_auth_header_when_key_empty(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{_BASE}/v1/embeddings",
            json={"data": [{"embedding": [0.1] * 768, "index": 0}]},
        )
        client = EmbeddingClient(base_url=_BASE, model="nomic-embed-text", api_key="")
        await client.embed("test")
        headers = httpx_mock.get_requests()[0].headers
        assert "authorization" not in headers
        await client.close()
