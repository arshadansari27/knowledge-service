import json

import pytest

from knowledge_service.clients.llm import (
    ExtractionClient,
    normalize_item_uris,
    to_entity_uri,
    to_predicate_uri,
    resolve_predicate_synonym,
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

    async def test_extract_returns_raw_labels_not_uris(self, mock_llm):
        """extract() should return items with original labels, not pre-normalized URIs."""
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.extract("Cold exposure increases dopamine.")
        assert len(result) == 1
        # Subject/predicate/object should be raw labels, NOT http:// URIs
        assert result[0].subject == "cold_exposure"
        assert result[0].predicate == "increases"
        assert result[0].object == "dopamine"
        await client.close()

    async def test_close_is_idempotent(self, mock_llm):
        client = ExtractionClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        await client.extract("text")
        await client.close()
        await client.close()  # should not raise


class TestUriNormalisation:
    def testto_entity_uri_slugifies(self):
        assert to_entity_uri("cold exposure") == "http://knowledge.local/data/cold_exposure"

    def testto_entity_uri_preserves_existing_uri(self):
        uri = "http://schema.org/Person"
        assert to_entity_uri(uri) == uri

    def testto_predicate_uri_slugifies(self):
        assert to_predicate_uri("increases") == "http://knowledge.local/schema/increases"

    def testto_predicate_uri_preserves_existing_uri(self):
        uri = "http://knowledge.local/schema/depends_on"
        assert to_predicate_uri(uri) == uri

    def test_normalize_claim_subject_and_predicate(self):
        item = {
            "knowledge_type": "Claim",
            "subject": "cold exposure",
            "predicate": "increases",
            "object": "dopamine",
            "confidence": 0.7,
        }
        result = normalize_item_uris(item)
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
        result = normalize_item_uris(item)
        assert result["object"] == "a value with spaces"

    def test_normalize_resolves_predicate_synonym(self):
        item = {
            "knowledge_type": "Claim",
            "subject": "x",
            "predicate": "boosts",
            "object": "y",
            "confidence": 0.7,
        }
        result = normalize_item_uris(item)
        assert result["predicate"] == "http://knowledge.local/schema/increases"

    def test_object_type_entity_converts_to_uri(self):
        item = {
            "knowledge_type": "Claim",
            "subject": "x",
            "predicate": "p",
            "object": "dopamine levels in the brain",
            "object_type": "entity",
            "confidence": 0.7,
        }
        result = normalize_item_uris(item)
        assert result["object"].startswith("http://knowledge.local/data/")
        assert "object_type" not in result

    def test_object_type_literal_preserved(self):
        item = {
            "knowledge_type": "Claim",
            "subject": "x",
            "predicate": "p",
            "object": "dopamine",
            "object_type": "literal",
            "confidence": 0.7,
        }
        result = normalize_item_uris(item)
        # Even though "dopamine" has no spaces and is short, object_type=literal wins
        assert result["object"] == "dopamine"
        assert "object_type" not in result

    def test_missing_object_type_falls_back_to_heuristic(self):
        item = {
            "knowledge_type": "Claim",
            "subject": "x",
            "predicate": "p",
            "object": "dopamine",
            "confidence": 0.7,
        }
        result = normalize_item_uris(item)
        # No object_type → heuristic: short, no spaces → entity URI
        assert result["object"] == "http://knowledge.local/data/dopamine"


class TestPredicateSynonymResolution:
    def test_known_synonym_resolves(self):
        assert resolve_predicate_synonym("boosts") == "increases"

    def test_canonical_predicate_unchanged(self):
        assert resolve_predicate_synonym("increases") == "increases"

    def test_unknown_predicate_unchanged(self):
        assert resolve_predicate_synonym("correlates_with") == "correlates_with"

    def test_synonym_case_insensitive(self):
        assert resolve_predicate_synonym("Boosts") == "increases"

    def test_synonym_with_spaces(self):
        assert resolve_predicate_synonym("leads to") == "causes"

    def test_synonym_with_hyphens(self):
        assert resolve_predicate_synonym("results-in") == "causes"


def test_extraction_prompt_includes_all_seven_types():
    from knowledge_service.clients.llm import _build_extraction_prompt

    prompt = _build_extraction_prompt("Some text", title=None, source_type=None)
    for type_name in (
        "Claim",
        "Fact",
        "Relationship",
        "Event",
        "Entity",
        "TemporalState",
        "Conclusion",
    ):
        assert type_name in prompt, f"Missing knowledge type '{type_name}' in extraction prompt"


def test_extraction_prompt_includes_predicate_vocabulary():
    from knowledge_service.clients.llm import _build_extraction_prompt

    prompt = _build_extraction_prompt("Some text", title=None, source_type=None)
    for pred in ("causes", "increases", "decreases", "inhibits", "activates"):
        assert pred in prompt, f"Missing predicate '{pred}' in extraction prompt"


def test_extraction_prompt_includes_naming_rules():
    from knowledge_service.clients.llm import _build_extraction_prompt

    prompt = _build_extraction_prompt("Some text", title=None, source_type=None)
    assert "snake_case" in prompt
    assert "singular" in prompt.lower()
    assert "canonical" in prompt.lower()


def test_extraction_prompt_includes_example():
    from knowledge_service.clients.llm import _build_extraction_prompt

    prompt = _build_extraction_prompt("Some text", title=None, source_type=None)
    assert "Example:" in prompt
    assert "cold_water_immersion" in prompt


def test_extraction_prompt_includes_object_type():
    from knowledge_service.clients.llm import _build_extraction_prompt

    prompt = _build_extraction_prompt("Some text", title=None, source_type=None)
    assert "object_type" in prompt
    assert '"entity"' in prompt or "'entity'" in prompt


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
