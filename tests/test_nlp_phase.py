"""Tests for NlpPhase and NLP dataclasses."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from knowledge_service.nlp import NlpEntity, NlpPhase, NlpResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linked_entity(qid: str) -> MagicMock:
    linked = MagicMock()
    linked.get_id.return_value = qid
    return linked


def _make_spacy_ent(
    text: str,
    label: str,
    start_char: int,
    end_char: int,
    linked_entities: list | None = None,
) -> MagicMock:
    ent = MagicMock()
    ent.text = text
    ent.label_ = label
    ent.start_char = start_char
    ent.end_char = end_char
    # spacy-entity-linker stores results on ent._.linkedEntities
    ent._.linkedEntities = linked_entities if linked_entities is not None else []
    return ent


def _make_sentence() -> MagicMock:
    return MagicMock()


def _make_nlp(*docs: MagicMock) -> MagicMock:
    """Return a callable mock that yields docs in order for successive calls."""
    nlp = MagicMock(side_effect=list(docs))
    return nlp


def _make_doc(
    ents: list[MagicMock],
    sentence_count: int = 1,
) -> MagicMock:
    doc = MagicMock()
    doc.ents = ents
    doc.sents = [_make_sentence() for _ in range(sentence_count)]
    return doc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNlpEntityCreation:
    def test_nlp_entity_creation(self):
        entity = NlpEntity(
            text="Narendra Modi",
            label="PERSON",
            start_char=0,
            end_char=13,
            wikidata_id="Q1156",
        )
        assert entity.text == "Narendra Modi"
        assert entity.label == "PERSON"
        assert entity.start_char == 0
        assert entity.end_char == 13
        assert entity.wikidata_id == "Q1156"

    def test_nlp_entity_defaults_to_no_wikidata(self):
        entity = NlpEntity(text="London", label="GPE", start_char=5, end_char=11)
        assert entity.wikidata_id is None


class TestNlpResultCreation:
    def test_nlp_result_creation(self):
        result = NlpResult(chunk_index=2, sentence_count=3)
        assert result.chunk_index == 2
        assert result.entities == []
        assert result.sentence_count == 3

    def test_nlp_result_defaults(self):
        result = NlpResult(chunk_index=0)
        assert result.entities == []
        assert result.sentence_count == 0


class TestNlpPhaseRunExtractsEntities:
    async def test_run_extracts_entities(self):
        linked = _make_linked_entity("Q1156")
        ent = _make_spacy_ent("Narendra Modi", "PERSON", 0, 13, linked_entities=[linked])
        doc = _make_doc(ents=[ent], sentence_count=2)
        nlp = _make_nlp(doc)

        phase = NlpPhase(nlp)
        results = await phase.run([{"chunk_index": 0, "chunk_text": "Narendra Modi spoke."}])

        assert len(results) == 1
        result = results[0]
        assert result.chunk_index == 0
        assert result.sentence_count == 2
        assert len(result.entities) == 1
        entity = result.entities[0]
        assert entity.text == "Narendra Modi"
        assert entity.label == "PERSON"
        assert entity.start_char == 0
        assert entity.end_char == 13
        assert entity.wikidata_id == "Q1156"


class TestNlpPhaseHandlesNoEntities:
    async def test_run_handles_no_entities(self):
        doc = _make_doc(ents=[], sentence_count=1)
        nlp = _make_nlp(doc)

        phase = NlpPhase(nlp)
        results = await phase.run([{"chunk_index": 0, "chunk_text": "No named entities here."}])

        assert len(results) == 1
        result = results[0]
        assert result.entities == []
        assert result.sentence_count == 1


class TestNlpPhaseMultipleChunks:
    async def test_run_multiple_chunks(self):
        linked_a = _make_linked_entity("Q1156")
        ent_a = _make_spacy_ent("Modi", "PERSON", 0, 4, linked_entities=[linked_a])
        doc_a = _make_doc(ents=[ent_a], sentence_count=1)

        linked_b = _make_linked_entity("Q84")
        ent_b = _make_spacy_ent("London", "GPE", 10, 16, linked_entities=[linked_b])
        doc_b = _make_doc(ents=[ent_b], sentence_count=3)

        nlp = _make_nlp(doc_a, doc_b)

        chunk_records = [
            {"chunk_index": 0, "chunk_text": "Modi spoke."},
            {"chunk_index": 1, "chunk_text": "The event was in London yesterday."},
        ]

        phase = NlpPhase(nlp)
        results = await phase.run(chunk_records)

        assert len(results) == 2

        assert results[0].chunk_index == 0
        assert len(results[0].entities) == 1
        assert results[0].entities[0].wikidata_id == "Q1156"

        assert results[1].chunk_index == 1
        assert len(results[1].entities) == 1
        assert results[1].entities[0].text == "London"
        assert results[1].entities[0].wikidata_id == "Q84"
        assert results[1].sentence_count == 3

    async def test_run_handles_empty_chunk_list(self):
        nlp = MagicMock()
        phase = NlpPhase(nlp)
        results = await phase.run([])
        assert results == []
        nlp.assert_not_called()
