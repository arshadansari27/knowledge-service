"""Tests for the graph-as-verifier answer path (lever E):
RAGClient.answer_verified / answer_auto and build_verify_prompt.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from knowledge_service.clients.rag import RAGClient, build_verify_prompt
from knowledge_service.stores.rag import RetrievalContext

_BASE = "http://llm-test"


def _client():
    return RAGClient(base_url=_BASE, model="m", api_key="k")


def _ctx_with_triples():
    return RetrievalContext(
        content_results=[{"title": "Doc", "source_type": "x", "chunk_text": "body"}],
        knowledge_triples=[{"subject": "a", "predicate": "p", "object": "b", "confidence": 0.8}],
    )


def _ctx_empty():
    return RetrievalContext(
        content_results=[{"title": "Doc", "source_type": "x", "chunk_text": "body"}],
        knowledge_triples=[],
        contradictions=[],
    )


class TestBuildVerifyPrompt:
    def test_contains_draft_facts_and_question(self):
        ctx = _ctx_with_triples()
        p = build_verify_prompt("the question", "the draft answer", ctx)
        assert "the draft answer" in p
        assert "Knowledge Graph Facts" in p
        assert "the question" in p
        assert "fact-checker" in p.lower()


class TestAnswerVerified:
    async def test_two_calls_returns_verified(self):
        client = _client()
        client._complete = AsyncMock(side_effect=["DRAFT", "VERIFIED"])
        out = await client.answer_verified("q", _ctx_with_triples())
        assert out.answer == "VERIFIED"
        assert client._complete.await_count == 2

    async def test_verify_failure_returns_draft(self):
        client = _client()
        client._complete = AsyncMock(side_effect=["DRAFT", RuntimeError("verify boom")])
        out = await client.answer_verified("q", _ctx_with_triples())
        assert out.answer == "DRAFT"
        assert client._complete.await_count == 2


class TestAnswerAuto:
    async def test_verify_with_graph_runs_two_calls(self):
        client = _client()
        client._complete = AsyncMock(side_effect=["DRAFT", "VERIFIED"])
        out = await client.answer_auto("q", _ctx_with_triples(), "verify")
        assert out.answer == "VERIFIED"
        assert client._complete.await_count == 2

    async def test_verify_with_empty_graph_degrades_to_one_call(self):
        client = _client()
        client._complete = AsyncMock(side_effect=["DIRECT"])
        out = await client.answer_auto("q", _ctx_empty(), "verify")
        assert out.answer == "DIRECT"
        assert client._complete.await_count == 1  # no pointless verify pass

    async def test_direct_runs_one_call(self):
        client = _client()
        client._complete = AsyncMock(side_effect=["DIRECT"])
        out = await client.answer_auto("q", _ctx_with_triples(), "direct")
        assert out.answer == "DIRECT"
        assert client._complete.await_count == 1


@pytest.mark.parametrize(
    "mode,has_graph,expected_calls",
    [("verify", True, 2), ("verify", False, 1), ("direct", True, 1), ("direct", False, 1)],
)
async def test_answer_auto_dispatch_matrix(mode, has_graph, expected_calls):
    client = _client()
    client._complete = AsyncMock(side_effect=["A", "B"])
    ctx = _ctx_with_triples() if has_graph else _ctx_empty()
    await client.answer_auto("q", ctx, mode)
    assert client._complete.await_count == expected_calls
