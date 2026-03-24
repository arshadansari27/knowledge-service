import json
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
