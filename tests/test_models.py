import pytest
from datetime import date
from pydantic import ValidationError

from knowledge_service.models import TripleInput, EventInput, EntityInput
from knowledge_service.ontology.uri import KS, KS_DATA, RDF_TYPE, RDFS_LABEL


class TestTripleInput:
    def test_basic_triple(self):
        t = TripleInput(subject="dopamine", predicate="causes", object="alertness")
        triples = t.to_triples()
        assert len(triples) == 1
        assert triples[0]["subject"] == f"{KS_DATA}dopamine"
        assert triples[0]["predicate"] == f"{KS}causes"
        assert triples[0]["object"] == "alertness"
        assert triples[0]["confidence"] == 0.8
        assert triples[0]["knowledge_type"] == "claim"

    def test_uri_passthrough(self):
        t = TripleInput(
            subject="http://example.com/s",
            predicate="http://example.com/p",
            object="http://example.com/o",
        )
        triples = t.to_triples()
        assert triples[0]["subject"] == "http://example.com/s"
        assert triples[0]["predicate"] == "http://example.com/p"

    def test_fact_type(self):
        t = TripleInput(
            subject="earth",
            predicate="is_a",
            object="planet",
            knowledge_type="fact",
            confidence=0.99,
        )
        assert t.to_triples()[0]["knowledge_type"] == "fact"

    def test_temporal_state(self):
        t = TripleInput(
            subject="acme",
            predicate="revenue",
            object="50M",
            knowledge_type="temporal_state",
            valid_from=date(2025, 1, 1),
            valid_until=date(2025, 12, 31),
        )
        triple = t.to_triples()[0]
        assert triple["valid_from"] == date(2025, 1, 1)
        assert triple["valid_until"] == date(2025, 12, 31)

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            TripleInput(subject="a", predicate="b", object="c", confidence=1.5)
        with pytest.raises(ValidationError):
            TripleInput(subject="a", predicate="b", object="c", confidence=-0.1)


class TestEventInput:
    def test_basic_event(self):
        e = EventInput(subject="ipo_acme", occurred_at=date(2025, 6, 1))
        triples = e.to_triples()
        assert len(triples) == 1
        assert triples[0]["predicate"] == f"{KS}occurredAt"
        assert triples[0]["object"] == "2025-06-01"
        assert triples[0]["knowledge_type"] == "event"

    def test_with_properties(self):
        e = EventInput(
            subject="ipo_acme",
            occurred_at=date(2025, 6, 1),
            properties={"amount": "1B", "currency": "USD"},
        )
        triples = e.to_triples()
        assert len(triples) == 3
        predicates = {t["predicate"] for t in triples}
        assert f"{KS}occurredAt" in predicates
        assert f"{KS}amount" in predicates
        assert f"{KS}currency" in predicates


class TestEntityInput:
    def test_basic_entity(self):
        e = EntityInput(uri="acme_corp", rdf_type="schema:Corporation", label="ACME Corp")
        triples = e.to_triples()
        assert len(triples) == 2
        subjects = {t["subject"] for t in triples}
        assert len(subjects) == 1
        assert f"{KS_DATA}acme_corp" in subjects

        type_triple = [t for t in triples if t["predicate"] == RDF_TYPE][0]
        assert type_triple["object"] == "schema:Corporation"

        label_triple = [t for t in triples if t["predicate"] == RDFS_LABEL][0]
        assert label_triple["object"] == "ACME Corp"

    def test_with_properties(self):
        e = EntityInput(
            uri="acme_corp",
            rdf_type="schema:Corporation",
            label="ACME Corp",
            properties={"ticker": "ACME"},
        )
        triples = e.to_triples()
        assert len(triples) == 3
