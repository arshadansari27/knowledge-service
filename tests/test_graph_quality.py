"""Tests for the 2026-05-31 graph-quality levers.

A — relevance floor; D — predicate-lookup gate; B/C — triple finalize
(verbalize + novelty filter); plus the verbalized prompt rendering and the
``include_graph`` switch. See
docs/superpowers/specs/2026-05-31-graph-quality-improvements-design.md.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from knowledge_service.clients.rag import _render_triple, build_rag_prompt
from knowledge_service.config import settings
from knowledge_service.stores.rag import RAGRetriever, RetrievalContext


def _triple(obj, conf=0.5, subject="http://knowledge.local/data/dopamine"):
    return {
        "subject": subject,
        "predicate": "http://knowledge.local/schema/affects",
        "object": obj,
        "confidence": conf,
        "graph": "http://knowledge.local/schema/graph/extracted",
    }


def _retriever(embed_batch=None, provenance_store=None):
    ec = AsyncMock()
    ec.embed.return_value = [1.0, 0.0, 0.0]
    if embed_batch is not None:
        ec.embed_batch.side_effect = embed_batch
    return RAGRetriever(
        embedding_client=ec,
        embedding_store=AsyncMock(),
        knowledge_store=MagicMock(),
        provenance_store=provenance_store,
    )


# --- Lever A: relevance floor -------------------------------------------------


class TestRelevanceFloor:
    async def test_drops_triples_below_floor(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_triple_relevance_floor", 0.5)
        query = [1.0, 0.0, 0.0]
        good = _triple("alertness")  # cosine 1.0 with query
        bad = _triple("shoelaces")  # cosine 0.0

        def vec(texts):
            return [[1.0, 0.0, 0.0] if "alertness" in t else [0.0, 1.0, 0.0] for t in texts]

        r = _retriever(embed_batch=vec)
        ranked = await r._rank_triples_by_relevance([good, bad], query, limit=10)
        assert [t["object"] for t in ranked] == ["alertness"]

    async def test_returns_empty_when_none_clear_floor(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_triple_relevance_floor", 0.9)
        query = [1.0, 0.0, 0.0]
        triples = [_triple("shoelaces"), _triple("weather")]
        r = _retriever(embed_batch=lambda texts: [[0.0, 1.0, 0.0] for _ in texts])
        assert await r._rank_triples_by_relevance(triples, query, limit=10) == []

    async def test_floor_zero_keeps_top_n(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_triple_relevance_floor", 0.0)
        query = [1.0, 0.0, 0.0]
        triples = [_triple(f"obj{i}") for i in range(5)]
        r = _retriever(embed_batch=lambda texts: [[0.0, 1.0, 0.0] for _ in texts])
        ranked = await r._rank_triples_by_relevance(triples, query, limit=3)
        assert len(ranked) == 3  # nothing dropped by floor; only the cap applies

    async def test_confidence_fallback_ignores_floor(self, monkeypatch):
        # Embedding backend down -> confidence fallback must NOT apply the floor
        # (it has no comparable score; dropping everything would be worse).
        monkeypatch.setattr(settings, "rag_triple_relevance_floor", 0.9)
        query = [1.0, 0.0, 0.0]
        triples = [_triple("a", 0.8), _triple("b", 0.2)]

        async def _boom(texts):
            raise RuntimeError("embed down")

        ec = AsyncMock()
        ec.embed_batch.side_effect = _boom
        r = RAGRetriever(
            embedding_client=ec, embedding_store=AsyncMock(), knowledge_store=MagicMock()
        )
        ranked = await r._rank_triples_by_relevance(triples, query, limit=10)
        assert len(ranked) == 2  # fallback kept both, floor not applied


# --- Lever D: predicate-lookup gate -------------------------------------------


class TestPredicateLookupGate:
    async def test_off_skips_store(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_predicate_lookup_enabled", False)
        es = AsyncMock()
        r = RAGRetriever(
            embedding_client=AsyncMock(),
            embedding_store=AsyncMock(),
            knowledge_store=MagicMock(),
            entity_store=es,
        )
        out = await r._lookup_triples_by_predicate([0.1, 0.2])
        assert out == []
        es.search_predicates.assert_not_called()

    async def test_on_queries_store(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_predicate_lookup_enabled", True)
        es = AsyncMock()
        es.search_predicates.return_value = []
        r = RAGRetriever(
            embedding_client=AsyncMock(),
            embedding_store=AsyncMock(),
            knowledge_store=MagicMock(),
            entity_store=es,
        )
        await r._lookup_triples_by_predicate([0.1, 0.2])
        es.search_predicates.assert_called_once()


# --- Levers B + C: finalize (verbalize + novelty) -----------------------------


def _prov_store(mapping):
    store = AsyncMock()
    store.get_by_triples.return_value = mapping
    return store


class TestFinalizeTriples:
    async def test_both_flags_off_is_noop(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_verbalize_triples", False)
        monkeypatch.setattr(settings, "rag_triple_novelty_filter", False)
        prov = _prov_store({})
        r = _retriever(provenance_store=prov)
        triples = [_triple("alertness")]
        out = await r._finalize_triples(triples, [])
        assert out == triples
        prov.get_by_triples.assert_not_called()

    async def test_verbalize_attaches_labels_and_source(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_verbalize_triples", True)
        monkeypatch.setattr(settings, "rag_triple_novelty_filter", False)
        from knowledge_service._utils import compute_triple_hash

        t = _triple("http://knowledge.local/data/serotonin")
        h = compute_triple_hash(t["subject"], t["predicate"], t["object"])
        prov = _prov_store({h: [{"source_url": "https://src.example/doc"}]})
        r = _retriever(provenance_store=prov)
        [out] = await r._finalize_triples([t], [])
        assert out["subject_label"] == "dopamine"
        assert out["predicate_label"] == "affects"
        assert out["object_label"] == "serotonin"
        assert out["source_url"] == "https://src.example/doc"

    async def test_novelty_drops_already_retrieved_keeps_cross_doc(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_verbalize_triples", False)
        monkeypatch.setattr(settings, "rag_triple_novelty_filter", True)
        from knowledge_service._utils import compute_triple_hash

        retrieved = _triple("a")
        cross = _triple("b")
        hr = compute_triple_hash(retrieved["subject"], retrieved["predicate"], retrieved["object"])
        hc = compute_triple_hash(cross["subject"], cross["predicate"], cross["object"])
        prov = _prov_store(
            {
                hr: [{"source_url": "https://seen.example/doc"}],
                hc: [{"source_url": "https://other.example/doc"}],
            }
        )
        r = _retriever(provenance_store=prov)
        content_results = [{"url": "https://seen.example/doc"}]
        out = await r._finalize_triples([retrieved, cross], content_results)
        assert [t["object"] for t in out] == ["b"]  # retrieved-source triple dropped

    async def test_novelty_keeps_triple_without_provenance(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_verbalize_triples", False)
        monkeypatch.setattr(settings, "rag_triple_novelty_filter", True)
        prov = _prov_store({})  # no provenance for anything
        r = _retriever(provenance_store=prov)
        content_results = [{"url": "https://seen.example/doc"}]
        out = await r._finalize_triples([_triple("a")], content_results)
        assert len(out) == 1  # can't prove redundancy -> kept

    async def test_provenance_failure_degrades(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_verbalize_triples", True)
        monkeypatch.setattr(settings, "rag_triple_novelty_filter", True)
        prov = AsyncMock()
        prov.get_by_triples.side_effect = RuntimeError("db down")
        r = _retriever(provenance_store=prov)
        content_results = [{"url": "https://seen.example/doc"}]
        out = await r._finalize_triples([_triple("a")], content_results)
        assert len(out) == 1  # no crash; novelty no-op
        assert out[0]["subject_label"] == "dopamine"  # labels still attached

    async def test_provenance_none_attaches_labels_only(self, monkeypatch):
        monkeypatch.setattr(settings, "rag_verbalize_triples", True)
        monkeypatch.setattr(settings, "rag_triple_novelty_filter", False)
        r = _retriever(provenance_store=None)
        [out] = await r._finalize_triples([_triple("alertness")], [])
        assert out["subject_label"] == "dopamine"
        assert "source_url" not in out


# --- Rendering: verbalized vs legacy + include_graph --------------------------


class TestRendering:
    def test_render_verbalized_when_labelled(self):
        t = {
            "subject_label": "cold exposure",
            "predicate_label": "increases",
            "object_label": "dopamine",
            "confidence": 0.7,
            "source_url": "https://s/d",
        }
        line = _render_triple(t)
        assert line == "- cold exposure increases dopamine (confidence: 0.7) · source: https://s/d"
        assert "->" not in line
        assert "http://knowledge.local" not in line

    def test_render_legacy_when_unlabelled(self):
        t = {
            "subject": "dopamine",
            "predicate": "affects",
            "object": "mood",
            "knowledge_type": "Fact",
            "confidence": 0.9,
            "trust_tier": "extracted",
        }
        line = _render_triple(t)
        assert line == "- [extracted] dopamine -> affects -> mood (Fact, confidence: 0.9)"

    def test_include_graph_false_omits_sections(self):
        ctx = RetrievalContext(
            content_results=[{"title": "Doc", "source_type": "x", "chunk_text": "body"}],
            knowledge_triples=[_triple("alertness")],
            contradictions=[{"subject": "a", "predicate": "p", "object": "o", "confidence": 0.5}],
        )
        prompt = build_rag_prompt("q", ctx, include_graph=False)
        assert "Knowledge Graph Facts" not in prompt
        assert "Contradictions Found" not in prompt
        assert "## Relevant Content" in prompt  # chunks still present

    def test_include_graph_true_keeps_sections(self):
        ctx = RetrievalContext(
            content_results=[],
            knowledge_triples=[_triple("alertness")],
            contradictions=[{"subject": "a", "predicate": "p", "object": "o", "confidence": 0.5}],
        )
        prompt = build_rag_prompt("q", ctx, include_graph=True)
        assert "Knowledge Graph Facts" in prompt
        assert "Contradictions Found" in prompt
