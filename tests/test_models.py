import pytest
from pydantic import ValidationError
from knowledge_service.models import (
    ClaimInput,
    FactInput,
    EventInput,
    ConclusionInput,
    TemporalStateInput,
    EntityInput,
    expand_to_triples,
)
from datetime import date


def test_claim_accepts_any_confidence():
    c = ClaimInput(subject="s", predicate="p", object="o", confidence=0.3)
    assert c.knowledge_type.value == "Claim"


def test_fact_rejects_low_confidence():
    with pytest.raises(ValidationError):
        FactInput(subject="s", predicate="p", object="o", confidence=0.5)


def test_event_requires_occurred_at():
    with pytest.raises(ValidationError):
        EventInput(subject="s")  # missing occurred_at


def test_temporal_state_requires_valid_until():
    with pytest.raises(ValidationError):
        TemporalStateInput(
            subject="s",
            property="p",
            value="v",
            valid_from=date(2025, 1, 1),
            # missing valid_until
        )


def test_temporal_state_validates_date_order():
    with pytest.raises(ValidationError):
        TemporalStateInput(
            subject="s",
            property="p",
            value="v",
            valid_from=date(2025, 6, 1),
            valid_until=date(2025, 1, 1),  # before valid_from
        )


def test_conclusion_requires_derived_from():
    with pytest.raises(ValidationError):
        ConclusionInput(concludes="x", inference_method="manual", confidence=0.8)
        # missing derived_from


# --- expand_to_triples tests ---


def test_expand_claim_returns_one_triple():
    c = ClaimInput(subject="http://s", predicate="http://p", object="val", confidence=0.7)
    triples = expand_to_triples(c)
    assert len(triples) == 1
    assert triples[0]["subject"] == "http://s"
    assert triples[0]["knowledge_type"] == "Claim"


def test_expand_event_creates_occurred_at_triple():
    e = EventInput(subject="http://ev", occurred_at=date(2024, 1, 15), confidence=1.0)
    triples = expand_to_triples(e)
    assert len(triples) >= 1
    assert any("occurredAt" in t["predicate"] for t in triples)


def test_expand_event_with_properties():
    e = EventInput(
        subject="http://ev",
        occurred_at=date(2024, 1, 15),
        confidence=1.0,
        properties={"amount": "500"},
    )
    triples = expand_to_triples(e)
    assert len(triples) == 2  # occurredAt + amount


def test_expand_entity_creates_type_and_label():
    e = EntityInput(
        uri="http://ent",
        rdf_type="http://schema.org/Thing",
        label="My Thing",
        confidence=0.95,
    )
    triples = expand_to_triples(e)
    assert len(triples) == 2  # rdf:type + rdfs:label
    predicates = [t["predicate"] for t in triples]
    assert any("type" in p for p in predicates)
    assert any("label" in p for p in predicates)


def test_expand_conclusion_creates_concludes_and_derived_from():
    c = ConclusionInput(
        concludes="Cold exposure increases dopamine",
        derived_from=["http://src1", "http://src2"],
        inference_method="bayesian",
        confidence=0.88,
    )
    triples = expand_to_triples(c)
    # 1 concludes + 2 derivedFrom + 1 inferenceMethod = 4
    assert len(triples) == 4


def test_expand_temporal_state_preserves_dates():
    ts = TemporalStateInput(
        subject="http://s",
        property="http://p",
        value="100",
        valid_from=date(2025, 1, 1),
        valid_until=date(2025, 6, 1),
        confidence=1.0,
    )
    triples = expand_to_triples(ts)
    assert len(triples) == 1
    assert triples[0]["valid_from"] == date(2025, 1, 1)
    assert triples[0]["valid_until"] == date(2025, 6, 1)
