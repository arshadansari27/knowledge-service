"""Tests that /api/ask resolves answer_mode and dispatches via answer_auto."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from knowledge_service.api.ask import AskRequest, post_ask
from knowledge_service.config import settings
from knowledge_service.stores.rag import QueryIntent, RetrievalContext


def _request(triples=None):
    state = MagicMock()
    retriever = AsyncMock()
    retriever.classify.return_value = QueryIntent(intent="entity", entities=["x"])
    retriever.retrieve.return_value = RetrievalContext(
        content_results=[], knowledge_triples=triples or [], contradictions=[]
    )
    state.rag_retriever = retriever
    rag_client = AsyncMock()
    rag_client.answer_auto.return_value = MagicMock(answer="generated")
    state.rag_client = rag_client
    state.stores = None
    req = MagicMock()
    req.app.state = state
    return req, rag_client


class TestAnswerModeWiring:
    async def test_explicit_verify_passed_to_answer_auto(self):
        req, rag_client = _request(
            triples=[{"subject": "a", "predicate": "p", "object": "b", "confidence": 0.5}]
        )
        resp = await post_ask(AskRequest(question="q", answer_mode="verify"), req)
        assert rag_client.answer_auto.await_args.args[2] == "verify"
        assert resp.answer_mode == "verify"

    async def test_default_uses_settings(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_default_answer_mode", "verify")
        req, rag_client = _request()
        resp = await post_ask(AskRequest(question="q"), req)
        assert rag_client.answer_auto.await_args.args[2] == "verify"
        assert resp.answer_mode == "verify"

    async def test_direct_default(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_default_answer_mode", "direct")
        req, rag_client = _request()
        resp = await post_ask(AskRequest(question="q"), req)
        assert rag_client.answer_auto.await_args.args[2] == "direct"
        assert resp.answer_mode == "direct"
