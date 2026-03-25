import json

from knowledge_service._utils import _extract_json
from knowledge_service.clients.classifier import QueryClassifier

_BASE = "http://llm-test"
_KEY = "sk-test"
_CHAT_URL = f"{_BASE}/v1/chat/completions"


def _make_response(intent: str, entities: list[str]) -> dict:
    return {
        "choices": [{"message": {"content": json.dumps({"intent": intent, "entities": entities})}}]
    }


class TestClassify:
    async def test_returns_semantic_intent(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_response("semantic", []))
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("find articles about stress")
        assert result.intent == "semantic"
        assert result.entities == []
        await c.close()

    async def test_returns_entity_intent_with_entities(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_response("entity", ["dopamine"]))
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("what is dopamine?")
        assert result.intent == "entity"
        assert "dopamine" in result.entities
        await c.close()

    async def test_returns_graph_intent(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL, json=_make_response("graph", ["cortisol", "inflammation"])
        )
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("how is cortisol connected to inflammation?")
        assert result.intent == "graph"
        assert len(result.entities) == 2
        await c.close()

    async def test_falls_back_to_semantic_on_http_error(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, status_code=500)
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("some question")
        assert result.intent == "semantic"
        assert result.entities == []
        await c.close()

    async def test_falls_back_to_semantic_on_bad_json(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json={"choices": [{"message": {"content": "not json {{"}}]},
        )
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("some question")
        assert result.intent == "semantic"
        await c.close()

    async def test_returns_global_intent(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_response("global", []))
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("what are the main themes in my knowledge base?")
        assert result.intent == "global"
        await c.close()

    async def test_falls_back_on_invalid_intent(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_response("unknown_type", []))
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("question")
        assert result.intent == "semantic"
        await c.close()

    async def test_handles_freeform_response_with_trailing_text(self, httpx_mock):
        """LLM returns JSON embedded in conversational text."""
        freeform = (
            "Here is the classification:\n\n```json\n"
            '{"intent": "entity", "entities": ["dopamine"]}\n'
            "```\n\nLet me know if you need anything else!"
        )
        httpx_mock.add_response(
            url=_CHAT_URL,
            json={"choices": [{"message": {"content": freeform}}]},
        )
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("what is dopamine?")
        assert result.intent == "entity"
        assert "dopamine" in result.entities
        await c.close()

    async def test_handles_thinking_tags(self, httpx_mock):
        """qwen3 thinking mode wraps output in <think> tags."""
        thinking = (
            "<think>The user is asking about a specific entity.</think>\n"
            '{"intent": "graph", "entities": ["cortisol", "inflammation"]}'
        )
        httpx_mock.add_response(
            url=_CHAT_URL,
            json={"choices": [{"message": {"content": thinking}}]},
        )
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("how is cortisol connected to inflammation?")
        assert result.intent == "graph"
        assert len(result.entities) == 2
        await c.close()


class TestExtractJson:
    def test_clean_json(self):
        assert _extract_json('{"intent": "entity"}') == {"intent": "entity"}

    def test_markdown_fenced(self):
        text = '```json\n{"intent": "graph"}\n```'
        assert _extract_json(text)["intent"] == "graph"

    def test_thinking_tags(self):
        text = '<think>reasoning here</think>\n{"intent": "global"}'
        assert _extract_json(text)["intent"] == "global"

    def test_trailing_text(self):
        text = '{"intent": "entity", "entities": ["x"]}\n\nHope this helps!'
        result = _extract_json(text)
        assert result["intent"] == "entity"

    def test_no_json_returns_none(self):
        assert _extract_json("no json here at all") is None

    def test_empty_json_object(self):
        assert _extract_json("{}") == {}

    def test_nested_json_in_prose(self):
        text = 'Here is the result:\n{"items": [{"name": "a"}, {"name": "b"}]}\nDone!'
        result = _extract_json(text)
        assert result == {"items": [{"name": "a"}, {"name": "b"}]}

    def test_nested_objects(self):
        text = 'Output: {"outer": {"inner": {"deep": true}}}'
        result = _extract_json(text)
        assert result == {"outer": {"inner": {"deep": True}}}

    def test_braces_inside_string_values(self):
        text = 'Result: {"msg": "use {braces} here"}'
        result = _extract_json(text)
        assert result == {"msg": "use {braces} here"}
