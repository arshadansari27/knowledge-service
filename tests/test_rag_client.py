"""Unit tests for RAGClient — prompt construction and LLM response parsing."""

from __future__ import annotations

import json

import pytest

from knowledge_service.clients.rag import RAGClient, RAGAnswer, build_rag_prompt
from knowledge_service.stores.rag import RetrievalContext

_BASE = "http://llm-test"
_KEY = "sk-test"
_CHAT_URL = f"{_BASE}/v1/chat/completions"


def _make_chat_response(answer: str, source_urls: list[str] | None = None) -> dict:
    content = json.dumps(
        {
            "answer": answer,
            "source_urls_cited": source_urls or [],
        }
    )
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _sample_context() -> RetrievalContext:
    return RetrievalContext(
        content_results=[
            {
                "id": "uuid-1",
                "url": "https://example.com/article",
                "title": "Cold Exposure and Dopamine",
                "summary": "A review of cold exposure effects on dopamine.",
                "source_type": "article",
                "tags": ["health"],
                "ingested_at": "2026-03-18T10:00:00Z",
                "similarity": 0.94,
            }
        ],
        knowledge_triples=[
            {
                "subject": "http://ks/cold_shock",
                "predicate": "http://ks/increases",
                "object": "http://ks/dopamine",
                "confidence": 0.88,
                "knowledge_type": "Claim",
                "valid_from": None,
                "valid_until": None,
            }
        ],
        contradictions=[
            {
                "subject": "http://ks/cold_shock",
                "predicate": "http://ks/decreases",
                "object": "http://ks/dopamine",
                "confidence": 0.3,
            }
        ],
        entities_found=["http://ks/cold_shock"],
    )


class TestBuildPromptLengthCap:
    def test_prompt_capped_with_many_large_sources(self):
        """RAG prompt should not exceed ~50K chars even with 100 large chunks."""
        content_results = [
            {
                "title": f"Doc {i}",
                "source_type": "article",
                "similarity": 0.9,
                "chunk_text": "x" * 4000,
            }
            for i in range(100)
        ]
        context = RetrievalContext(content_results=content_results)
        prompt = build_rag_prompt("test question", context)
        assert len(prompt) <= 55_000  # Allow some overhead for headers

    def test_prompt_includes_truncation_notice(self):
        """When truncated, prompt should indicate more sources exist."""
        content_results = [
            {
                "title": f"Doc {i}",
                "source_type": "article",
                "similarity": 0.9,
                "chunk_text": "x" * 4000,
            }
            for i in range(100)
        ]
        context = RetrievalContext(content_results=content_results)
        prompt = build_rag_prompt("test question", context)
        assert "truncated" in prompt.lower()

    def test_small_prompt_not_truncated(self):
        """Small prompts should not be truncated."""
        prompt = build_rag_prompt("q", _sample_context())
        assert "truncated" not in prompt.lower()


class TestBuildPrompt:
    def test_includes_question(self):
        prompt = build_rag_prompt("Does cold exposure increase dopamine?", _sample_context())
        assert "Does cold exposure increase dopamine?" in prompt

    def test_includes_content_title(self):
        prompt = build_rag_prompt("q", _sample_context())
        assert "Cold Exposure and Dopamine" in prompt

    def test_includes_knowledge_triple(self):
        prompt = build_rag_prompt("q", _sample_context())
        assert "increases" in prompt
        assert "0.88" in prompt

    def test_includes_contradictions(self):
        prompt = build_rag_prompt("q", _sample_context())
        assert "Contradictions" in prompt
        assert "decreases" in prompt

    def test_empty_context_still_valid(self):
        prompt = build_rag_prompt("q", RetrievalContext())
        assert "q" in prompt

    def test_includes_section_header_when_present(self):
        ctx = RetrievalContext(
            content_results=[
                {
                    "id": "uuid-2",
                    "url": "https://example.com/article",
                    "title": "My Article",
                    "source_type": "article",
                    "similarity": 0.9,
                    "chunk_text": "Some text",
                    "section_header": "Title > Methods",
                }
            ]
        )
        prompt = build_rag_prompt("q", ctx)
        assert "[Section: Title > Methods]" in prompt

    def test_no_section_tag_when_header_absent(self):
        prompt = build_rag_prompt("q", _sample_context())
        assert "[Section:" not in prompt


class TestRAGClientAnswer:
    async def test_returns_rag_answer(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json=_make_chat_response(
                "Cold exposure increases dopamine.", ["https://example.com/article"]
            ),
        )
        client = RAGClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.answer("Does cold exposure increase dopamine?", _sample_context())
        assert isinstance(result, RAGAnswer)
        assert "dopamine" in result.answer.lower()
        assert result.source_urls_cited == ["https://example.com/article"]
        await client.close()

    async def test_fallback_on_bad_json(self, httpx_mock):
        httpx_mock.add_response(
            url=_CHAT_URL,
            json={"choices": [{"message": {"content": "not json at all {{"}}]},
        )
        client = RAGClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.answer("q", _sample_context())
        assert result.answer == "not json at all {{"
        assert result.source_urls_cited == []
        await client.close()

    async def test_fallback_on_markdown_fenced_json(self, httpx_mock):
        fenced = (
            "```json\n" + json.dumps({"answer": "fenced answer", "source_urls_cited": []}) + "\n```"
        )
        httpx_mock.add_response(
            url=_CHAT_URL,
            json={"choices": [{"message": {"content": fenced}}]},
        )
        client = RAGClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await client.answer("q", _sample_context())
        assert result.answer == "fenced answer"
        await client.close()

    async def test_http_error_raises(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, status_code=500)
        client = RAGClient(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        with pytest.raises(Exception):
            await client.answer("q", _sample_context())
        await client.close()

    async def test_model_name_in_request(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_chat_response("a"))
        client = RAGClient(base_url=_BASE, model="my-model", api_key=_KEY)
        await client.answer("q", _sample_context())
        body = json.loads(httpx_mock.get_requests()[0].content)
        assert body["model"] == "my-model"
        await client.close()

    async def test_no_response_format_constraint(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_chat_response("a"))
        client = RAGClient(base_url=_BASE, model="m", api_key=_KEY)
        await client.answer("q", _sample_context())
        body = json.loads(httpx_mock.get_requests()[0].content)
        assert "response_format" not in body
        await client.close()
