import pytest
from datetime import date
from pydantic import TypeAdapter, ValidationError

from knowledge_service.models import KnowledgeInput, TripleInput, EventInput, EntityInput
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


# --- Regression tests for production-observed LLM output shapes ---
# Production logs over a 30-day window showed ~9% of LLM-extracted items being
# silently rejected. Sample dicts are pasted verbatim from prod logs in the
# failing-items investigation. See PR description for the root-cause analysis.


_ADAPTER: TypeAdapter[KnowledgeInput] = TypeAdapter(KnowledgeInput)


class TestUnionDiscriminator:
    """Pattern A — KnowledgeInput must route by knowledge_type, not by shape."""

    def test_entity_routes_to_entity_input(self):
        payload = {
            "knowledge_type": "Entity",
            "uri": "amy",
            "rdf_type": "schema:Person",
            "label": "amy",
            "properties": {},
            "confidence": 0.95,
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, EntityInput)

    def test_claim_routes_to_triple_input(self):
        payload = {
            "knowledge_type": "Claim",
            "subject": "cold_exposure",
            "predicate": "increases",
            "object": "dopamine",
            "confidence": 0.7,
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, TripleInput)

    def test_event_routes_to_event_input(self):
        payload = {
            "knowledge_type": "Event",
            "subject": "ipo_acme",
            "occurred_at": "2025-06-01",
            "confidence": 0.9,
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, EventInput)

    def test_routes_by_shape_when_knowledge_type_missing(self):
        # The current API contract (api/claims.py) accepts payloads without a
        # knowledge_type tag — pre-discriminator validation used shape. Keep
        # that path working: occurred_at -> Event, rdf_type -> Entity, else
        # Triple.
        event_payload = {"subject": "x", "occurred_at": "2024-01-15", "confidence": 1.0}
        assert isinstance(_ADAPTER.validate_python(event_payload), EventInput)

        entity_payload = {"uri": "x", "rdf_type": "schema:Thing", "label": "x"}
        assert isinstance(_ADAPTER.validate_python(entity_payload), EntityInput)

        triple_payload = {"subject": "x", "predicate": "is_a", "object": "y"}
        assert isinstance(_ADAPTER.validate_python(triple_payload), TripleInput)

    def test_knowledge_type_is_case_insensitive(self):
        # LLM emits CapitalCase ("Entity", "Claim"); to_triples writes lowercase
        # ("entity", "event"). Both must route to the same member.
        for label in ("entity", "Entity", "ENTITY", "EnTiTy"):
            payload = {
                "knowledge_type": label,
                "uri": "x",
                "rdf_type": "schema:Thing",
                "label": "x",
                "confidence": 0.9,
            }
            result = _ADAPTER.validate_python(payload)
            assert isinstance(result, EntityInput), f"failed for label={label!r}"


class TestErrorMessagesAfterDiscriminator:
    """Pattern B — validation error must point at the correct member, not bury
    the real reason behind 'TripleInput: subject/predicate/object missing'."""

    def test_bad_entity_does_not_complain_about_triple_fields(self):
        # Missing required `label` — error should mention entity, not triple.
        payload = {
            "knowledge_type": "Entity",
            "uri": "x",
            "rdf_type": "schema:Thing",
            "confidence": 0.9,
        }
        with pytest.raises(ValidationError) as exc_info:
            _ADAPTER.validate_python(payload)
        msg = str(exc_info.value).lower()
        assert "subject" not in msg, f"error still mentions triple fields: {msg}"
        assert "predicate" not in msg, f"error still mentions triple fields: {msg}"


class TestPropertyListValues:
    """Pattern C — qwen3:14b regularly emits list values inside properties."""

    def test_entity_accepts_list_property_values(self):
        # Verbatim sample from prod logs.
        payload = {
            "knowledge_type": "Entity",
            "uri": "amy",
            "rdf_type": "schema:Person",
            "label": "amy",
            "properties": {"occupation": ["lawyer", "executive"]},
            "confidence": 0.95,
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, EntityInput)

    def test_entity_list_values_expand_to_one_triple_per_item(self):
        e = EntityInput(
            uri="amy",
            rdf_type="schema:Person",
            label="amy",
            properties={"occupation": ["lawyer", "executive"]},
            confidence=0.95,
        )
        triples = e.to_triples()
        # 2 base (rdf:type + rdfs:label) + 2 occupation entries = 4
        assert len(triples) == 4
        occupation_objects = sorted(
            t["object"] for t in triples if t["predicate"] == f"{KS}occupation"
        )
        assert occupation_objects == ["executive", "lawyer"]

    def test_entity_mixed_str_and_list_property_values(self):
        e = EntityInput(
            uri="amy",
            rdf_type="schema:Person",
            label="amy",
            properties={"ticker": "AMY", "occupation": ["lawyer", "executive"]},
        )
        triples = e.to_triples()
        # 2 base + 1 ticker + 2 occupation = 5
        assert len(triples) == 5

    def test_event_accepts_list_property_values(self):
        # Verbatim sample from prod logs (also has unparseable date — see Pattern E).
        payload = {
            "knowledge_type": "Event",
            "subject": "paper_submission",
            "occurred_at": "2024-06-01",
            "confidence": 0.9,
            "properties": {"authors": ["kyle_higgins", "other_authors"]},
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, EventInput)

    def test_event_list_values_expand_to_one_triple_per_item(self):
        e = EventInput(
            subject="paper_submission",
            occurred_at=date(2024, 6, 1),
            properties={"authors": ["kyle_higgins", "other_authors"]},
            confidence=0.9,
        )
        triples = e.to_triples()
        # 1 occurredAt + 2 author entries = 3
        assert len(triples) == 3
        author_objects = sorted(t["object"] for t in triples if t["predicate"] == f"{KS}authors")
        assert author_objects == ["kyle_higgins", "other_authors"]


class TestPropertyShapeDrift:
    """Pattern D — LLM sometimes nests rdf_type or label inside properties."""

    def test_nested_rdf_type_is_lifted(self):
        # Verbatim sample from prod logs.
        payload = {
            "knowledge_type": "Entity",
            "uri": "sree_bhattacharyya",
            "label": "sree_bhattacharyya",
            "properties": {"rdf_type": "schema:Person", "name": "Sree Bhattacharyya"},
            "confidence": 0.95,
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, EntityInput)
        assert result.rdf_type == "schema:Person"
        # The nested rdf_type should not leak through as a triple — properties
        # should no longer contain it.
        assert "rdf_type" not in result.properties

    def test_nested_label_is_lifted_when_missing_top_level(self):
        payload = {
            "knowledge_type": "Entity",
            "uri": "x",
            "rdf_type": "schema:Thing",
            "properties": {"label": "X Name"},
            "confidence": 0.9,
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, EntityInput)
        assert result.label == "X Name"

    def test_top_level_label_wins_over_nested(self):
        payload = {
            "knowledge_type": "Entity",
            "uri": "x",
            "rdf_type": "schema:Thing",
            "label": "top_level",
            "properties": {"label": "nested"},
            "confidence": 0.9,
        }
        result = _ADAPTER.validate_python(payload)
        assert result.label == "top_level"


class TestFuzzyDateCoercion:
    """Pattern E — EventInput.occurred_at should not crash on fuzzy LLM dates."""

    def test_unparseable_occurred_at_is_coerced_to_none(self):
        # Verbatim sample from prod logs.
        payload = {
            "knowledge_type": "Event",
            "subject": "paper_submission",
            "occurred_at": "some_25_years_ago",
            "confidence": 0.9,
            "properties": {"authors": ["kyle_higgins", "other_authors"]},
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, EventInput)
        assert result.occurred_at is None

    def test_event_with_none_occurred_at_drops_event_triples(self):
        # If we can't pin the event in time, drop it entirely rather than
        # surface a meaningless event without a timestamp.
        e = EventInput(
            subject="paper_submission",
            occurred_at=None,
            properties={"authors": ["kyle_higgins"]},
            confidence=0.9,
        )
        assert e.to_triples() == []

    def test_event_iso_date_string_still_parses(self):
        payload = {
            "knowledge_type": "Event",
            "subject": "ipo_acme",
            "occurred_at": "2025-06-01",
            "confidence": 0.9,
        }
        result = _ADAPTER.validate_python(payload)
        assert result.occurred_at == date(2025, 6, 1)
