"""Tests for CoreferencePhase: Wikidata-QID-driven merging + alias persistence."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock


from knowledge_service.ingestion.coreference import (
    CoreferencePhase,
    CoreferenceResult,
    EntityGroup,
)
from knowledge_service.nlp import NlpEntity, NlpResult
from knowledge_service.ontology.uri import to_entity_uri


def _make_pool(conn: AsyncMock | None = None) -> MagicMock:
    if conn is None:
        conn = AsyncMock()
    pool = MagicMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool


def _make_nlp_result(
    chunk_index: int,
    entities: list[tuple[str, str | None]],
) -> NlpResult:
    nlp_entities = [
        NlpEntity(text=text, label="MISC", start_char=0, end_char=len(text), wikidata_id=wid)
        for text, wid in entities
    ]
    return NlpResult(chunk_index=chunk_index, entities=nlp_entities)


class TestEntityGroup:
    def test_entity_group_defaults(self):
        group = EntityGroup(
            canonical_label="narendra_modi",
            canonical_uri=to_entity_uri("narendra_modi"),
        )
        assert group.aliases == []
        assert group.wikidata_id is None

    def test_entity_group_with_aliases_and_wikidata(self):
        group = EntityGroup(
            canonical_label="narendra_modi",
            canonical_uri=to_entity_uri("narendra_modi"),
            aliases=["modi", "pm_modi"],
            wikidata_id="Q1058",
        )
        assert len(group.aliases) == 2
        assert group.wikidata_id == "Q1058"


class TestCoreferenceResultCanonicalize:
    def test_canonicalize_rewrites_subject(self):
        group = EntityGroup(
            canonical_label="narendra_modi",
            canonical_uri=to_entity_uri("narendra_modi"),
            aliases=["modi"],
        )
        result = CoreferenceResult(groups=[group])
        items = [{"subject": "modi", "predicate": "is_pm_of", "object": "india"}]
        rewritten = result.canonicalize(items)
        assert rewritten[0].subject == "narendra_modi"

    def test_canonicalize_rewrites_object(self):
        group = EntityGroup(
            canonical_label="india",
            canonical_uri=to_entity_uri("india"),
            aliases=["bharat"],
        )
        result = CoreferenceResult(groups=[group])
        items = [{"subject": "modi", "predicate": "leads", "object": "bharat"}]
        rewritten = result.canonicalize(items)
        assert rewritten[0].object == "india"

    def test_canonicalize_case_insensitive(self):
        group = EntityGroup(
            canonical_label="narendra_modi",
            canonical_uri=to_entity_uri("narendra_modi"),
            aliases=["modi"],
        )
        result = CoreferenceResult(groups=[group])
        items = [{"subject": "Modi", "predicate": "p", "object": "o"}]
        rewritten = result.canonicalize(items)
        assert rewritten[0].subject == "narendra_modi"

    def test_canonicalize_leaves_unmatched_unchanged(self):
        result = CoreferenceResult(groups=[])
        items = [{"subject": "x", "predicate": "p", "object": "y"}]
        rewritten = result.canonicalize(items)
        assert rewritten[0].subject == "x"

    def test_canonicalize_preserves_entity_input(self):
        """Entity items (uri/rdf_type/label) survive coref intact."""
        group = EntityGroup(
            canonical_label="dopamine",
            canonical_uri=to_entity_uri("dopamine"),
            aliases=["da"],
        )
        result = CoreferenceResult(groups=[group])
        items = [
            {
                "knowledge_type": "Entity",
                "uri": "da",
                "rdf_type": "schema:Thing",
                "label": "da",
                "properties": {},
                "confidence": 0.9,
            }
        ]
        rewritten = result.canonicalize(items)
        assert hasattr(rewritten[0], "to_triples")
        assert rewritten[0].uri == "dopamine"
        assert rewritten[0].label == "dopamine"

    def test_canonicalize_preserves_event_input(self):
        """Event items (subject/occurred_at) survive coref intact."""
        from datetime import date

        result = CoreferenceResult(groups=[])
        items = [
            {
                "knowledge_type": "Event",
                "subject": "launch",
                "occurred_at": date(2026, 4, 30),
                "properties": {},
                "confidence": 0.9,
            }
        ]
        rewritten = result.canonicalize(items)
        assert hasattr(rewritten[0], "to_triples")
        assert rewritten[0].subject == "launch"


class TestCoreferencePhaseWikidataTier:
    async def test_merges_entities_with_same_qid(self):
        pool = _make_pool()
        nlp_results = [
            _make_nlp_result(0, [("Modi", "Q1058"), ("Narendra Modi", "Q1058")]),
        ]
        phase = CoreferencePhase(pool)
        result = await phase.run([], nlp_results)

        assert len(result.groups) == 1
        group = result.groups[0]
        assert group.wikidata_id == "Q1058"
        assert group.canonical_label == "modi"
        assert "narendra_modi" in group.aliases

    async def test_different_qids_produce_separate_groups(self):
        pool = _make_pool()
        nlp_results = [
            _make_nlp_result(0, [("Modi", "Q1058"), ("London", "Q84")]),
        ]
        phase = CoreferencePhase(pool)
        result = await phase.run([], nlp_results)

        qids = sorted(g.wikidata_id for g in result.groups)
        assert qids == ["Q1058", "Q84"]

    async def test_entities_without_qid_ignored(self):
        pool = _make_pool()
        nlp_results = [
            _make_nlp_result(0, [("Obscure Thing", None), ("Modi", "Q1058")]),
        ]
        phase = CoreferencePhase(pool)
        result = await phase.run([], nlp_results)

        assert len(result.groups) == 1
        assert result.groups[0].wikidata_id == "Q1058"

    async def test_empty_nlp_results_produces_no_groups(self):
        pool = _make_pool()
        phase = CoreferencePhase(pool)
        result = await phase.run([], [])
        assert result.groups == []


class TestCoreferencePhaseStoreAliases:
    async def test_store_aliases_calls_executemany(self):
        conn = AsyncMock()
        pool = _make_pool(conn)

        groups = [
            EntityGroup(
                canonical_label="narendra_modi",
                canonical_uri=to_entity_uri("narendra_modi"),
                aliases=["modi", "pm_modi"],
                wikidata_id="Q1058",
            )
        ]

        phase = CoreferencePhase(pool)
        await phase._store_aliases(groups)

        conn.executemany.assert_called_once()
        sql = conn.executemany.call_args[0][0]
        assert "INSERT INTO entity_aliases" in sql
        assert "ON CONFLICT" in sql

    async def test_store_aliases_empty_groups_skips_db(self):
        conn = AsyncMock()
        pool = _make_pool(conn)

        phase = CoreferencePhase(pool)
        await phase._store_aliases([])

        conn.executemany.assert_not_called()

    async def test_store_aliases_group_with_no_aliases_still_stores_canonical(self):
        conn = AsyncMock()
        pool = _make_pool(conn)

        groups = [
            EntityGroup(
                canonical_label="dopamine",
                canonical_uri=to_entity_uri("dopamine"),
                aliases=[],
            )
        ]

        phase = CoreferencePhase(pool)
        await phase._store_aliases(groups)

        conn.executemany.assert_called_once()
        rows = conn.executemany.call_args[0][1]
        assert len(rows) == 1
        assert rows[0][0] == "dopamine"

    async def test_store_aliases_tags_source_as_spacy(self):
        conn = AsyncMock()
        pool = _make_pool(conn)

        groups = [
            EntityGroup(
                canonical_label="london",
                canonical_uri=to_entity_uri("london"),
                aliases=["city_of_london"],
                wikidata_id="Q84",
            )
        ]

        phase = CoreferencePhase(pool)
        await phase._store_aliases(groups)

        rows = conn.executemany.call_args[0][1]
        for _, _, source_tag in rows:
            assert source_tag == "spacy_linking"

    async def test_store_aliases_swallows_db_errors(self):
        conn = AsyncMock()
        conn.executemany.side_effect = RuntimeError("db down")
        pool = _make_pool(conn)

        groups = [
            EntityGroup(
                canonical_label="x",
                canonical_uri=to_entity_uri("x"),
                aliases=["y"],
                wikidata_id="Q1",
            )
        ]

        phase = CoreferencePhase(pool)
        await phase._store_aliases(groups)  # must not raise


class TestCoreferencePhaseRunIntegration:
    async def test_run_returns_coreference_result(self):
        pool = _make_pool()
        phase = CoreferencePhase(pool)
        result = await phase.run([], [])

        assert isinstance(result, CoreferenceResult)
        assert result.groups == []

    async def test_run_canonicalize_after_run(self):
        """Full pipeline: run then canonicalize rewrites knowledge items correctly."""
        pool = _make_pool()
        nlp_results = [
            _make_nlp_result(0, [("Modi", "Q1058"), ("Narendra Modi", "Q1058")]),
        ]
        knowledge_items = [
            {
                "subject": "narendra_modi",
                "predicate": "is",
                "object": "pm",
                "object_type": "entity",
            },
        ]

        phase = CoreferencePhase(pool)
        result = await phase.run(knowledge_items, nlp_results)
        rewritten = result.canonicalize(knowledge_items)

        # "narendra_modi" is an alias of canonical "modi" (first QID-linked label)
        assert rewritten[0].subject == "modi"
