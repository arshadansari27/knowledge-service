import json

import pytest

from knowledge_service.clients.llm import ExtractionClient
from knowledge_service.models import EntityInput, TripleInput

_BASE = "http://llm-test"
_KEY = "sk-test"
_CHAT_URL = f"{_BASE}/v1/chat/completions"


def _make_chat_response(items: list) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": json.dumps({"items": items})}}]
    }


def _make_combined_response(entities: list, relations: list) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"entities": entities, "relations": relations}),
                }
            }
        ],
    }


@pytest.fixture
def mock_llm(httpx_mock):
    # Single combined response
    httpx_mock.add_response(
        url=_CHAT_URL,
        json=_make_combined_response(
            entities=[
                {
                    "knowledge_type": "Entity",
                    "uri": "cold_exposure",
                    "rdf_type": "schema:Thing",
                    "label": "cold_exposure",
                    "properties": {},
                    "confidence": 0.9,
                },
            ],
            relations=[
                {
                    "knowledge_type": "Claim",
                    "subject": "cold_exposure",
                    "predicate": "increases",
                    "object": "dopamine",
                    "confidence": 0.7,
                }
            ],
        ),
    )
    return httpx_mock


class TestExtract:
    async def test_returns_entity_and_claim_from_valid_response(self, mock_llm):
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("Cold exposure increases dopamine.")
        assert len(result) == 2
        assert isinstance(result[0], EntityInput)
        assert isinstance(result[1], TripleInput)
        await client.close()

    async def test_returns_none_on_http_error(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, status_code=500)
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("some text")
        assert result is None
        await client.close()

    async def test_returns_none_on_bad_json(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json={"choices": [{"message": {"content": "not valid json {{"}}]},
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("some text")
        assert result is None
        await client.close()

    async def test_skips_invalid_items_returns_valid(self, httpx_mock):
        # Combined response: one valid Entity, one invalid item, no relations
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_combined_response(
                entities=[
                    {
                        "knowledge_type": "Entity",
                        "uri": "a",
                        "rdf_type": "schema:Thing",
                        "label": "a",
                        "properties": {},
                        "confidence": 0.9,
                    },
                    {"knowledge_type": "Entity", "missing_required_fields": True},
                ],
                relations=[],
            ),
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("text")
        assert len(result) == 1
        assert isinstance(result[0], EntityInput)
        await client.close()

    async def test_returns_empty_list_on_empty_items(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL, json=_make_combined_response(entities=[], relations=[])
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("text")
        assert result == []
        await client.close()

    async def test_model_name_sent_in_request(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL, json=_make_combined_response(entities=[], relations=[])
        )
        client = ExtractionClient(base_url=_BASE, model="claude-sonnet", api_key=_KEY)
        await client.extract("text")
        body = json.loads(httpx_mock.get_requests()[0].content)
        assert body["model"] == "claude-sonnet"
        await client.close()

    async def test_uses_chat_completions_format(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL, json=_make_combined_response(entities=[], relations=[])
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        await client.extract("text")
        body = json.loads(httpx_mock.get_requests()[0].content)
        assert "messages" in body
        assert body["messages"][0]["role"] == "user"
        assert "response_format" not in body
        await client.close()

    async def test_auth_header_sent(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL, json=_make_combined_response(entities=[], relations=[])
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key="sk-mykey")
        await client.extract("text")
        headers = httpx_mock.get_requests()[0].headers
        assert headers["authorization"] == "Bearer sk-mykey"
        await client.close()

    async def test_extract_returns_raw_labels_not_uris(self, mock_llm):
        """extract() should return items with original labels, not pre-normalized URIs."""
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("Cold exposure increases dopamine.")
        assert len(result) == 2
        # Entity item should have raw label
        entity = result[0]
        assert isinstance(entity, EntityInput)
        assert entity.label == "cold_exposure"
        # Claim subject/predicate/object should be raw labels, NOT http:// URIs
        claim = result[1]
        assert claim.subject == "cold_exposure"
        assert claim.predicate == "increases"
        assert claim.object == "dopamine"
        await client.close()

    async def test_close_is_idempotent(self, mock_llm):
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        await client.extract("text")
        await client.close()
        await client.close()  # should not raise


class TestFallbackPrompts:
    """Tests for the fallback prompt functions used when no DomainRegistry is provided."""

    def test_relation_prompt_includes_relation_types(self):
        from knowledge_service.clients.llm import _build_relation_extraction_prompt_fallback

        prompt = _build_relation_extraction_prompt_fallback("text", None, None, ["a", "b"])
        for t in ("Claim", "Fact", "Relationship", "TemporalState", "Conclusion"):
            assert t in prompt

    def test_relation_prompt_includes_predicates(self):
        from knowledge_service.clients.llm import _build_relation_extraction_prompt_fallback

        prompt = _build_relation_extraction_prompt_fallback("text", None, None, ["a"])
        for pred in ("causes", "increases", "decreases"):
            assert pred in prompt

    def test_relation_prompt_includes_object_type(self):
        from knowledge_service.clients.llm import _build_relation_extraction_prompt_fallback

        prompt = _build_relation_extraction_prompt_fallback("text", None, None, ["a"])
        assert "object_type" in prompt

    def test_entity_prompt_includes_naming_rules(self):
        from knowledge_service.clients.llm import _build_entity_extraction_prompt_fallback

        prompt = _build_entity_extraction_prompt_fallback("text", None, None)
        assert "snake_case" in prompt
        assert "singular" in prompt.lower()

    def test_entity_prompt_includes_example(self):
        from knowledge_service.clients.llm import _build_entity_extraction_prompt_fallback

        prompt = _build_entity_extraction_prompt_fallback("text", None, None)
        assert "Example:" in prompt


class TestEntityExtractionPrompt:
    def test_entity_extraction_prompt_focuses_on_entities(self):
        from knowledge_service.clients.llm import _build_entity_extraction_prompt_fallback

        prompt = _build_entity_extraction_prompt_fallback("Some text", title=None, source_type=None)
        assert "Entity" in prompt
        assert "Event" in prompt
        assert "Claim:" not in prompt
        assert "Relationship:" not in prompt
        assert "snake_case" in prompt

    def test_entity_extraction_prompt_includes_text(self):
        from knowledge_service.clients.llm import _build_entity_extraction_prompt_fallback

        prompt = _build_entity_extraction_prompt_fallback(
            "Cold exposure boosts dopamine.", title="Test", source_type="article"
        )
        assert "Cold exposure boosts dopamine" in prompt
        assert "Title: Test" in prompt


class TestRelationExtractionPrompt:
    def test_relation_extraction_prompt_includes_entity_list(self):
        from knowledge_service.clients.llm import _build_relation_extraction_prompt_fallback

        prompt = _build_relation_extraction_prompt_fallback(
            "text", None, None, entities=["cold_exposure", "dopamine"]
        )
        assert "cold_exposure" in prompt
        assert "dopamine" in prompt
        assert "Claim" in prompt
        assert "causes" in prompt

    def test_relation_extraction_prompt_constrains_to_entities(self):
        from knowledge_service.clients.llm import _build_relation_extraction_prompt_fallback

        prompt = _build_relation_extraction_prompt_fallback(
            "text", None, None, entities=["entity_a", "entity_b"]
        )
        assert "entity_a" in prompt
        assert "Prefer these entities" in prompt


class TestSinglePassExtract:
    async def test_makes_one_llm_call(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_combined_response(
                entities=[
                    {
                        "knowledge_type": "Entity",
                        "uri": "dopamine",
                        "rdf_type": "schema:Thing",
                        "label": "dopamine",
                        "properties": {},
                        "confidence": 0.9,
                    },
                ],
                relations=[
                    {
                        "knowledge_type": "Claim",
                        "subject": "cold_exposure",
                        "predicate": "increases",
                        "object": "dopamine",
                        "confidence": 0.7,
                    },
                ],
            ),
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("Cold exposure increases dopamine.")
        assert len(httpx_mock.get_requests()) == 1
        assert len(result) == 2
        await client.close()

    async def test_returns_entities_only_when_no_relations(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_combined_response(
                entities=[
                    {
                        "knowledge_type": "Entity",
                        "uri": "x",
                        "rdf_type": "schema:Thing",
                        "label": "x",
                        "properties": {},
                        "confidence": 0.9,
                    },
                ],
                relations=[],
            ),
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("text")
        assert len(result) == 1
        assert isinstance(result[0], EntityInput)
        await client.close()

    async def test_returns_none_when_call_fails(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, status_code=500)
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("text")
        assert result is None
        assert len(httpx_mock.get_requests()) == 1
        await client.close()

    async def test_legacy_items_format_still_works(self, httpx_mock):
        """extract() should also accept legacy {"items": [...]} format."""
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_chat_response(
                [
                    {
                        "knowledge_type": "Entity",
                        "uri": "x",
                        "rdf_type": "schema:Thing",
                        "label": "x",
                        "properties": {},
                        "confidence": 0.9,
                    },
                ]
            ),
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("text")
        assert len(result) == 1
        assert isinstance(result[0], EntityInput)
        await client.close()


class TestNoAuth:
    async def test_no_auth_header_when_key_empty(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_combined_response(entities=[], relations=[]),
        )
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key="")
        await client.extract("test")
        headers = httpx_mock.get_requests()[0].headers
        assert "authorization" not in headers
        await client.close()
