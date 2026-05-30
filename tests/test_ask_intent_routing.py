"""Tests for /api/ask intent-based retrieval-mode routing.

The 2026-05-31 relevance-ranking eval showed the graph helps on entity/relationship
questions but mildly hurts plain semantic search. The "auto" default routes each
query to the mode that wins for its intent.
"""

from __future__ import annotations

from knowledge_service.api.ask import _needs_classification, _resolve_mode


class TestNeedsClassification:
    def test_explicit_chunks_only_skips(self):
        assert _needs_classification("chunks_only", "auto") is False

    def test_explicit_full_needs(self):
        assert _needs_classification("full", "auto") is True

    def test_default_auto_needs(self):
        assert _needs_classification(None, "auto") is True

    def test_default_full_needs(self):
        assert _needs_classification(None, "full") is True

    def test_default_chunks_only_skips(self):
        assert _needs_classification(None, "chunks_only") is False


class TestResolveMode:
    def test_explicit_wins_over_default(self):
        assert _resolve_mode("chunks_only", "auto", "entity") == "chunks_only"
        assert _resolve_mode("full", "chunks_only", None) == "full"

    def test_auto_routes_entity_to_full(self):
        assert _resolve_mode(None, "auto", "entity") == "full"

    def test_auto_routes_graph_to_full(self):
        assert _resolve_mode(None, "auto", "graph") == "full"

    def test_auto_routes_semantic_to_chunks_only(self):
        assert _resolve_mode(None, "auto", "semantic") == "chunks_only"

    def test_auto_routes_unknown_intent_to_chunks_only(self):
        assert _resolve_mode(None, "auto", None) == "chunks_only"

    def test_static_default_full(self):
        assert _resolve_mode(None, "full", "semantic") == "full"

    def test_static_default_chunks_only(self):
        assert _resolve_mode(None, "chunks_only", "entity") == "chunks_only"
