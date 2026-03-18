import json

import pytest

from knowledge_service.clients.llm import (
    ExtractionClient,
    _to_entity_uri,
    _to_predicate_uri,
    _normalize_item_uris,
)

_BASE = "http://llm-test"
_KEY = "sk-test"
_CHAT_URL = f"{_BASE}/v1/chat/completions"


def _make_chat_response(items: list) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": json.dumps({"items": items})}}]
    }


@pytest.fixture
def mock_llm(httpx_mock):
    httpx_mock.add_response(
        url=_CHAT_URL,
        json=_make_chat_response(
            [
                {
                    "knowledge_type": "Claim",
                    "subject": "cold_exposure",
                    "predicate": "increases",
                    "object": "dopamine",
                    "confidence": 0.7,
                }
            ]
        ),
    )
    return httpx_mock


class TestExtract:
    async def test_returns_claim_from_valid_response(self, mock_llm):
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("Cold exposure increases dopamine.")
        assert len(result) == 1
        assert result[0].knowledge_type.value == "Claim"
        await client.close()

    async def test_returns_empty_on_http_error(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, status_code=500)
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("some text")
        assert result == []
        await client.close()

    async def test_returns_empty_on_bad_json(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json={"choices": [{"message": {"content": "not valid json {{"}}]},
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("some text")
        assert result == []
        await client.close()

    async def test_skips_invalid_items_returns_valid(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_chat_response(
                [
                    {
                        "knowledge_type": "Claim",
                        "subject": "a",
                        "predicate": "b",
                        "object": "c",
                        "confidence": 0.7,
                    },
                    {"knowledge_type": "Claim", "missing_required_fields": True},
                ]
            ),
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("text")
        assert len(result) == 1
        await client.close()

    async def test_returns_empty_list_on_empty_items(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_chat_response([]))
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("text")
        assert result == []
        await client.close()

    async def test_model_name_sent_in_request(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_chat_response([]))
        client = ExtractionClient(base_url=_BASE, model="claude-sonnet", api_key=_KEY)
        await client.extract("text")
        body = json.loads(httpx_mock.get_requests()[0].content)
        assert body["model"] == "claude-sonnet"
        await client.close()

    async def test_uses_chat_completions_format(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_chat_response([]))
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        await client.extract("text")
        body = json.loads(httpx_mock.get_requests()[0].content)
        assert "messages" in body
        assert body["messages"][0]["role"] == "user"
        assert body["response_format"] == {"type": "json_object"}
        await client.close()

    async def test_auth_header_sent(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_chat_response([]))
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key="sk-mykey")
        await client.extract("text")
        headers = httpx_mock.get_requests()[0].headers
        assert headers["authorization"] == "Bearer sk-mykey"
        await client.close()

    async def test_close_is_idempotent(self, mock_llm):
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        await client.extract("text")
        await client.close()
        await client.close()  # should not raise


class TestUriNormalisation:
    def test_to_entity_uri_slugifies(self):
        assert _to_entity_uri("cold exposure") == "http://knowledge.local/data/cold_exposure"

    def test_to_entity_uri_preserves_existing_uri(self):
        uri = "http://schema.org/Person"
        assert _to_entity_uri(uri) == uri

    def test_to_predicate_uri_slugifies(self):
        assert _to_predicate_uri("increases") == "http://knowledge.local/schema/increases"

    def test_to_predicate_uri_preserves_existing_uri(self):
        uri = "http://knowledge.local/schema/depends_on"
        assert _to_predicate_uri(uri) == uri

    def test_normalize_claim_subject_and_predicate(self):
        item = {
            "knowledge_type": "Claim",
            "subject": "cold exposure",
            "predicate": "increases",
            "object": "dopamine",
            "confidence": 0.7,
        }
        result = _normalize_item_uris(item)
        assert result["subject"] == "http://knowledge.local/data/cold_exposure"
        assert result["predicate"] == "http://knowledge.local/schema/increases"
        assert result["object"] == "http://knowledge.local/data/dopamine"

    def test_normalize_leaves_literal_objects_unchanged(self):
        item = {
            "knowledge_type": "Claim",
            "subject": "x",
            "predicate": "p",
            "object": "a value with spaces",
            "confidence": 0.7,
        }
        result = _normalize_item_uris(item)
        assert result["object"] == "a value with spaces"


class TestNoAuth:
    async def test_no_auth_header_when_key_empty(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_chat_response([]),
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key="")
        await client.extract("test")
        headers = httpx_mock.get_requests()[0].headers
        assert "authorization" not in headers
        await client.close()
