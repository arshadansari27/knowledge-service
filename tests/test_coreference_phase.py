"""Tests for CoreferencePhase: tier-1 Wikidata merging + tier-2 LLM grouping."""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(conn: AsyncMock | None = None) -> MagicMock:
    """Return a pg_pool mock using the asynccontextmanager acquire pattern."""
    if conn is None:
        conn = AsyncMock()
    pool = MagicMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire
    return pool


def _make_extraction_client(llm_response: list[dict] | None = None) -> MagicMock:
    """Return an extraction_client mock with _call_llm as AsyncMock."""
    client = MagicMock()
    client._call_llm = AsyncMock(return_value=llm_response)
    return client


def _make_nlp_result(
    chunk_index: int,
    entities: list[tuple[str, str | None]],
) -> NlpResult:
    """Build an NlpResult with (text, wikidata_id) pairs."""
    nlp_entities = [
        NlpEntity(text=text, label="MISC", start_char=0, end_char=len(text), wikidata_id=wid)
        for text, wid in entities
    ]
    return NlpResult(chunk_index=chunk_index, entities=nlp_entities)


# ---------------------------------------------------------------------------
# Tests: EntityGroup and CoreferenceResult dataclasses
# ---------------------------------------------------------------------------


class TestEntityGroup:
    def test_entity_group_defaults(self):
        group = EntityGroup(
            canonical_label="narendra_modi",
            canonical_uri=to_entity_uri("narendra_modi"),
        )
        assert group.aliases == []
        assert group.wikidata_id is None
        assert group.rdf_type is None

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
        assert rewritten[0]["subject"] == "narendra_modi"

    def test_canonicalize_rewrites_object(self):
        group = EntityGroup(
            canonical_label="narendra_modi",
            canonical_uri=to_entity_uri("narendra_modi"),
            aliases=["pm modi"],
        )
        result = CoreferenceResult(groups=[group])
        items = [{"subject": "india", "predicate": "has_leader", "object": "pm modi"}]
        rewritten = result.canonicalize(items)
        assert rewritten[0]["object"] == "narendra_modi"

    def test_canonicalize_case_insensitive(self):
        group = EntityGroup(
            canonical_label="narendra_modi",
            canonical_uri=to_entity_uri("narendra_modi"),
            aliases=["Modi"],
        )
        result = CoreferenceResult(groups=[group])
        items = [{"subject": "MODI", "predicate": "speaks", "object": "something"}]
        rewritten = result.canonicalize(items)
        assert rewritten[0]["subject"] == "narendra_modi"

    def test_canonicalize_no_match_unchanged(self):
        group = EntityGroup(
            canonical_label="narendra_modi",
            canonical_uri=to_entity_uri("narendra_modi"),
            aliases=["pm"],
        )
        result = CoreferenceResult(groups=[group])
        items = [{"subject": "london", "predicate": "is_city_of", "object": "uk"}]
        rewritten = result.canonicalize(items)
        assert rewritten[0]["subject"] == "london"
        assert rewritten[0]["object"] == "uk"

    def test_canonicalize_does_not_mutate_original(self):
        group = EntityGroup(
            canonical_label="a",
            canonical_uri=to_entity_uri("a"),
            aliases=["b"],
        )
        result = CoreferenceResult(groups=[group])
        original = {"subject": "b", "predicate": "rel", "object": "c"}
        result.canonicalize([original])
        # original dict should be unchanged
        assert original["subject"] == "b"

    def test_canonicalize_empty_groups(self):
        result = CoreferenceResult()
        items = [{"subject": "x", "predicate": "y", "object": "z"}]
        rewritten = result.canonicalize(items)
        assert rewritten[0]["subject"] == "x"


# ---------------------------------------------------------------------------
# Tests: Tier 1 — Wikidata QID merging (no LLM call)
# ---------------------------------------------------------------------------


class TestCoreferencePhaseWikidataTier:
    async def test_tier1_same_qid_merges_entities(self):
        """Entities from different chunks sharing a QID should form one group."""
        client = _make_extraction_client(llm_response=None)
        pool = _make_pool()

        nlp_results = [
            _make_nlp_result(0, [("Modi", "Q1058")]),
            _make_nlp_result(1, [("Narendra Modi", "Q1058")]),
        ]

        phase = CoreferencePhase(client, pool)
        result = await phase.run([], nlp_results)

        # Should produce exactly one group with QID Q1058
        qid_groups = [g for g in result.groups if g.wikidata_id == "Q1058"]
        assert len(qid_groups) == 1
        group = qid_groups[0]
        # Canonical is first label seen (slugified)
        assert group.canonical_label == "modi"
        assert "narendra_modi" in group.aliases

    async def test_tier1_no_llm_call_when_all_linked(self):
        """No LLM call should be made when all knowledge item entities are Wikidata-linked."""
        client = _make_extraction_client(llm_response=None)
        pool = _make_pool()

        # Both entities in the knowledge item are Wikidata-linked
        nlp_results = [
            _make_nlp_result(0, [("modi", "Q1058"), ("india", "Q668")]),
        ]
        # knowledge_items reference slugified labels that match linked entities
        knowledge_items = [
            {
                "subject": "modi",
                "predicate": "is_leader_of",
                "object": "india",
                "object_type": "entity",
            },
        ]

        phase = CoreferencePhase(client, pool)
        await phase.run(knowledge_items, nlp_results)

        client._call_llm.assert_not_called()

    async def test_tier1_different_qids_produce_separate_groups(self):
        client = _make_extraction_client(llm_response=[])
        pool = _make_pool()

        nlp_results = [
            _make_nlp_result(0, [("Modi", "Q1058"), ("London", "Q84")]),
        ]

        phase = CoreferencePhase(client, pool)
        result = await phase.run([], nlp_results)

        qids = {g.wikidata_id for g in result.groups if g.wikidata_id}
        assert "Q1058" in qids
        assert "Q84" in qids

    async def test_tier1_entity_without_wikidata_is_unlinked(self):
        """Entities with no wikidata_id are not grouped in tier 1."""
        client = _make_extraction_client(llm_response=[])
        pool = _make_pool()

        nlp_results = [
            _make_nlp_result(0, [("SomeUnknownEntity", None)]),
        ]
        knowledge_items = [
            {
                "subject": "someunknownentity",
                "predicate": "does",
                "object": "something",
                "object_type": "entity",
            }
        ]

        phase = CoreferencePhase(client, pool)
        result = await phase.run(knowledge_items, nlp_results)

        # Tier 1 produces no groups for this entity
        qid_groups = [g for g in result.groups if g.wikidata_id]
        assert len(qid_groups) == 0


# ---------------------------------------------------------------------------
# Tests: Tier 2 — LLM coreference for unlinked entities
# ---------------------------------------------------------------------------


class TestCoreferencePhaseUnlinkedLLMTier:
    async def test_tier2_llm_called_for_unlinked_entities(self):
        """Unlinked entities in knowledge items trigger the LLM call."""
        llm_response = [{"canonical": "cold_exposure", "aliases": ["cold", "cold_water_immersion"]}]
        client = _make_extraction_client(llm_response=llm_response)
        pool = _make_pool()

        knowledge_items = [
            {
                "subject": "cold",
                "predicate": "increases",
                "object": "dopamine",
                "object_type": "entity",
            },
            {
                "subject": "cold_water_immersion",
                "predicate": "causes",
                "object": "shivering",
                "object_type": "entity",
            },
        ]

        phase = CoreferencePhase(client, pool)
        await phase.run(knowledge_items, [])

        client._call_llm.assert_called_once()
        prompt_arg = client._call_llm.call_args[0][0]
        assert "cold" in prompt_arg
        assert "cold_water_immersion" in prompt_arg

    async def test_tier2_llm_groups_create_entity_groups(self):
        """LLM-returned groups produce EntityGroup objects with correct canonical/aliases."""
        llm_response = [{"canonical": "cold_exposure", "aliases": ["cold", "cold_water_immersion"]}]
        client = _make_extraction_client(llm_response=llm_response)
        pool = _make_pool()

        knowledge_items = [
            {
                "subject": "cold",
                "predicate": "increases",
                "object": "dopamine",
                "object_type": "entity",
            },
        ]

        phase = CoreferencePhase(client, pool)
        result = await phase.run(knowledge_items, [])

        llm_groups = [g for g in result.groups if g.wikidata_id is None]
        assert len(llm_groups) == 1
        group = llm_groups[0]
        assert group.canonical_label == "cold_exposure"
        assert group.canonical_uri == to_entity_uri("cold_exposure")
        assert "cold" in group.aliases
        assert "cold_water_immersion" in group.aliases

    async def test_tier2_llm_returns_none_produces_unmapped(self):
        """When LLM call fails (returns None), entities go to unmapped."""
        client = _make_extraction_client(llm_response=None)
        pool = _make_pool()

        knowledge_items = [
            {
                "subject": "unknown_entity",
                "predicate": "does",
                "object": "thing",
                "object_type": "entity",
            },
        ]

        phase = CoreferencePhase(client, pool)
        result = await phase.run(knowledge_items, [])

        assert "unknown_entity" in result.unmapped

    async def test_tier2_llm_returns_empty_list_produces_unmapped(self):
        """When LLM returns empty list, unlinked entities become unmapped."""
        client = _make_extraction_client(llm_response=[])
        pool = _make_pool()

        knowledge_items = [
            {
                "subject": "entity_a",
                "predicate": "rel",
                "object": "entity_b",
                "object_type": "entity",
            },
        ]

        phase = CoreferencePhase(client, pool)
        result = await phase.run(knowledge_items, [])

        assert "entity_a" in result.unmapped
        assert "entity_b" in result.unmapped

    async def test_tier2_canonical_slugified(self):
        """LLM-returned canonical labels are slugified."""
        llm_response = [{"canonical": "Narendra Modi", "aliases": ["Modi PM"]}]
        client = _make_extraction_client(llm_response=llm_response)
        pool = _make_pool()

        knowledge_items = [
            {
                "subject": "narendra_modi",
                "predicate": "is",
                "object": "pm",
                "object_type": "entity",
            },
        ]

        phase = CoreferencePhase(client, pool)
        result = await phase.run(knowledge_items, [])

        llm_groups = [g for g in result.groups if g.wikidata_id is None]
        assert len(llm_groups) == 1
        assert llm_groups[0].canonical_label == "narendra_modi"
        assert "modi_pm" in llm_groups[0].aliases


# ---------------------------------------------------------------------------
# Tests: _store_aliases
# ---------------------------------------------------------------------------


class TestCoreferencePhaseStoreAliases:
    async def test_store_aliases_calls_executemany(self):
        """_store_aliases should call executemany on the connection."""
        conn = AsyncMock()
        pool = _make_pool(conn)
        client = _make_extraction_client(llm_response=[])

        groups = [
            EntityGroup(
                canonical_label="narendra_modi",
                canonical_uri=to_entity_uri("narendra_modi"),
                aliases=["modi", "pm_modi"],
                wikidata_id="Q1058",
            )
        ]

        phase = CoreferencePhase(client, pool)
        await phase._store_aliases(groups)

        conn.executemany.assert_called_once()
        # Check it was called with the INSERT statement
        call_args = conn.executemany.call_args
        sql = call_args[0][0]
        assert "INSERT INTO entity_aliases" in sql
        assert "ON CONFLICT" in sql

    async def test_store_aliases_empty_groups_skips_db(self):
        """Empty groups list should not touch the database."""
        conn = AsyncMock()
        pool = _make_pool(conn)
        client = _make_extraction_client(llm_response=[])

        phase = CoreferencePhase(client, pool)
        await phase._store_aliases([])

        conn.executemany.assert_not_called()

    async def test_store_aliases_group_with_no_aliases_still_stores_canonical(self):
        """A group with no aliases should still store the canonical label."""
        conn = AsyncMock()
        pool = _make_pool(conn)
        client = _make_extraction_client(llm_response=[])

        groups = [
            EntityGroup(
                canonical_label="dopamine",
                canonical_uri=to_entity_uri("dopamine"),
                aliases=[],
            )
        ]

        phase = CoreferencePhase(client, pool)
        await phase._store_aliases(groups)

        conn.executemany.assert_called_once()
        rows = conn.executemany.call_args[0][1]
        assert len(rows) == 1
        assert rows[0][0] == "dopamine"

    async def test_store_aliases_source_tag_spacy_for_wikidata_group(self):
        """Groups with wikidata_id should be tagged 'spacy_linking'."""
        conn = AsyncMock()
        pool = _make_pool(conn)
        client = _make_extraction_client(llm_response=[])

        groups = [
            EntityGroup(
                canonical_label="london",
                canonical_uri=to_entity_uri("london"),
                aliases=["city_of_london"],
                wikidata_id="Q84",
            )
        ]

        phase = CoreferencePhase(client, pool)
        await phase._store_aliases(groups)

        rows = conn.executemany.call_args[0][1]
        for _, _, source_tag in rows:
            assert source_tag == "spacy_linking"

    async def test_store_aliases_source_tag_llm_for_unlinked_group(self):
        """Groups without wikidata_id should be tagged 'llm_coreference'."""
        conn = AsyncMock()
        pool = _make_pool(conn)
        client = _make_extraction_client(llm_response=[])

        groups = [
            EntityGroup(
                canonical_label="cold_exposure",
                canonical_uri=to_entity_uri("cold_exposure"),
                aliases=["cold_immersion"],
                wikidata_id=None,
            )
        ]

        phase = CoreferencePhase(client, pool)
        await phase._store_aliases(groups)

        rows = conn.executemany.call_args[0][1]
        for _, _, source_tag in rows:
            assert source_tag == "llm_coreference"


# ---------------------------------------------------------------------------
# Tests: Integration — full run() flow
# ---------------------------------------------------------------------------


class TestCoreferencePhaseRunIntegration:
    async def test_run_returns_coreference_result(self):
        client = _make_extraction_client(llm_response=[])
        pool = _make_pool()

        phase = CoreferencePhase(client, pool)
        result = await phase.run([], [])

        assert isinstance(result, CoreferenceResult)
        assert result.groups == []
        assert result.unmapped == []

    async def test_run_combined_tier1_and_tier2(self):
        """Tier 1 merges Wikidata entities; Tier 2 groups unlinked leftovers."""
        llm_response = [{"canonical": "cold_exposure", "aliases": ["cold", "cold_water"]}]
        client = _make_extraction_client(llm_response=llm_response)
        pool = _make_pool()

        nlp_results = [
            _make_nlp_result(0, [("Modi", "Q1058"), ("Narendra Modi", "Q1058")]),
        ]
        knowledge_items = [
            {
                "subject": "cold",
                "predicate": "increases",
                "object": "dopamine",
                "object_type": "entity",
            },
            {
                "subject": "cold_water",
                "predicate": "causes",
                "object": "shock",
                "object_type": "entity",
            },
        ]

        phase = CoreferencePhase(client, pool)
        result = await phase.run(knowledge_items, nlp_results)

        # Tier 1 group
        wikidata_groups = [g for g in result.groups if g.wikidata_id == "Q1058"]
        assert len(wikidata_groups) == 1

        # Tier 2 group from LLM
        llm_groups = [g for g in result.groups if g.canonical_label == "cold_exposure"]
        assert len(llm_groups) == 1

    async def test_run_canonicalize_after_run(self):
        """Full pipeline: run then canonicalize rewrites knowledge items correctly."""
        llm_response = [{"canonical": "cold_exposure", "aliases": ["cold", "cold_water_immersion"]}]
        client = _make_extraction_client(llm_response=llm_response)
        pool = _make_pool()

        knowledge_items = [
            {
                "subject": "cold",
                "predicate": "increases",
                "object": "dopamine",
                "object_type": "entity",
            },
            {
                "subject": "cold_water_immersion",
                "predicate": "causes",
                "object": "shivering",
                "object_type": "literal",
            },
        ]

        phase = CoreferencePhase(client, pool)
        result = await phase.run(knowledge_items, [])
        rewritten = result.canonicalize(knowledge_items)

        assert rewritten[0]["subject"] == "cold_exposure"
        assert rewritten[1]["subject"] == "cold_exposure"
