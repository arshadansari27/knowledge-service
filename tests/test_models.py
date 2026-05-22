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

    def test_event_iso_datetime_string_strips_time(self):
        # qwen3 occasionally emits ISO datetime with time/timezone — strip
        # the time portion rather than drop the event.
        for ts in ("2025-06-01T10:00:00Z", "2025-06-01T10:00:00+05:30", "2025-06-01 10:00:00"):
            payload = {
                "knowledge_type": "Event",
                "subject": "x",
                "occurred_at": ts,
                "confidence": 0.9,
            }
            result = _ADAPTER.validate_python(payload)
            assert result.occurred_at == date(2025, 6, 1), f"failed for ts={ts!r}"


class TestNullKnowledgeType:
    """Pattern A follow-up — qwen3 sometimes emits ``knowledge_type: null``.
    Without coercion every such item is silently rejected by the str field."""

    def test_null_knowledge_type_routes_by_shape(self):
        payload = {
            "knowledge_type": None,
            "uri": "x",
            "rdf_type": "schema:Thing",
            "label": "x",
            "confidence": 0.9,
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, EntityInput)


class TestPropertyListCoercion:
    """Pattern C follow-up — list values may contain non-string scalars."""

    def test_list_values_with_numbers_are_stringified(self):
        e = EntityInput(
            uri="acme",
            rdf_type="schema:Corporation",
            label="acme",
            properties={"founded": [2020, 2021]},
        )
        triples = e.to_triples()
        founded_objects = sorted(t["object"] for t in triples if "founded" in t["predicate"])
        assert founded_objects == ["2020", "2021"]


# --- Follow-up regression tests (post-PR #73) ---
# Categorising all 1,875 rejections in the 8-day prod log showed PR #73 only
# recovered ~16.6% of them. The remaining patterns are below — each verbatim
# from production. Group naming matches the follow-up-PR description.


class TestTripleAliases:
    """~1,060 rejections — qwen3 emits triples in non-canonical shapes. We
    alias the common keys back to ``subject`` / ``predicate`` / ``object``."""

    def test_from_relation_to_aliases(self):
        # Verbatim sample from prod logs.
        payload = {"from": "dec_vaxstation_2000", "to": "xwm", "relation": "uses"}
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, TripleInput)
        assert result.subject == "dec_vaxstation_2000"
        assert result.object == "xwm"
        assert result.predicate == "uses"

    def test_source_target_relation_aliases(self):
        payload = {
            "source": "transmittance_lut",
            "target": "frame_buffer_object",
            "relation": "uses",
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, TripleInput)
        assert result.subject == "transmittance_lut"
        assert result.object == "frame_buffer_object"
        assert result.predicate == "uses"

    def test_source_target_type_aliases(self):
        # "type" is the alias when the LLM uses {source, target, type}.
        payload = {"source": "gerry_wheeler", "target": "byte_magazine", "type": "published_in"}
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, TripleInput)
        assert result.predicate == "published_in"

    def test_head_relation_tail_aliases(self):
        payload = {"head": "snowflake_postgres", "relation": "developed_by", "tail": "crunchy_data"}
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, TripleInput)
        assert result.subject == "snowflake_postgres"
        assert result.object == "crunchy_data"

    def test_subject_predicate_aliases_dont_override_canonical(self):
        # If both alias and canonical are present, canonical wins.
        payload = {
            "subject": "real_s",
            "from": "alias_s",
            "predicate": "real_p",
            "object": "real_o",
        }
        result = _ADAPTER.validate_python(payload)
        assert result.subject == "real_s"

    def test_alias_with_confidence(self):
        payload = {"from": "ibm", "to": "lenovo", "relation": "acquired_by", "confidence": 0.95}
        result = _ADAPTER.validate_python(payload)
        assert result.subject == "ibm"
        assert result.confidence == 0.95


class TestObjectFailsLoudOnUnknownDict:
    """Symmetric to test_property_dict_without_value_key_still_rejected —
    pin the contract that genuinely-unknown nested-dict shapes still fail
    loudly so future schema drift surfaces in logs rather than dropping
    silently."""

    def test_object_dict_without_value_key_still_rejected(self):
        payload = {
            "subject": "x",
            "predicate": "p",
            "object": {"foo": "bar", "baz": "qux"},
        }
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python(payload)

    def test_object_value_type_dict_with_null_value_still_rejects(self):
        # Unwrap yields None; TripleInput.object: str rejects loudly.
        payload = {
            "subject": "x",
            "predicate": "p",
            "object": {"value": None, "type": "literal"},
        }
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python(payload)


class TestObjectUnwrap:
    """~55 rejections — qwen3 emits the object as a {value, type} dict like
    ``{"object": {"type": "literal", "value": "65.2"}}``."""

    def test_object_value_type_dict_is_unwrapped(self):
        payload = {
            "subject": "matryoshka_embeddings",
            "predicate": "scores",
            "object": {"type": "literal", "value": "65.2"},
            "confidence": 0.9,
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, TripleInput)
        assert result.object == "65.2"

    def test_object_value_type_entity_is_unwrapped(self):
        payload = {
            "knowledge_type": "Fact",
            "subject": "meta",
            "predicate": "receives",
            "object": {"type": "entity", "value": "tax_break"},
            "confidence": 0.95,
        }
        result = _ADAPTER.validate_python(payload)
        assert result.object == "tax_break"


class TestPropertiesNestedDictUnwrap:
    """~30 rejections — qwen3 nests property values as {value, type} dicts."""

    def test_entity_property_with_value_type_dict_is_unwrapped(self):
        payload = {
            "knowledge_type": "Entity",
            "uri": "pywikibot",
            "rdf_type": "schema:SoftwareApplication",
            "label": "pywikibot",
            "properties": {
                "description": {
                    "value": "A Python library for Wikipedia bot development",
                    "type": "literal",
                }
            },
            "confidence": 0.95,
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, EntityInput)
        assert result.properties["description"] == "A Python library for Wikipedia bot development"

    def test_event_property_with_value_type_dict_is_unwrapped(self):
        payload = {
            "knowledge_type": "Event",
            "subject": "x",
            "occurred_at": "2024-01-01",
            "properties": {"title": {"value": "Hello World", "type": "literal"}},
            "confidence": 0.9,
        }
        result = _ADAPTER.validate_python(payload)
        assert result.properties["title"] == "Hello World"

    def test_property_dict_without_value_key_still_rejected(self):
        # Only the canonical {value, type} shape is unwrapped — random dicts
        # should still fail loudly so we don't hide real schema drift.
        payload = {
            "knowledge_type": "Entity",
            "uri": "x",
            "rdf_type": "schema:Thing",
            "label": "x",
            "properties": {"weird": {"foo": "bar", "baz": "qux"}},
        }
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python(payload)


class TestObjectScalarStringify:
    """7 prod rejections — object emitted as bool/int directly."""

    def test_bool_object_stringified(self):
        payload = {
            "knowledge_type": "Fact",
            "subject": "mullvad_vpn",
            "predicate": "offers_multiple_exit_ips",
            "object": True,
            "confidence": 0.9,
        }
        result = _ADAPTER.validate_python(payload)
        assert result.object == "True"

    def test_int_object_stringified(self):
        payload = {
            "subject": "x",
            "predicate": "count",
            "object": 42,
        }
        result = _ADAPTER.validate_python(payload)
        assert result.object == "42"


class TestEntityRdfTypeDefault:
    """518 rejections — entities arrive without ``rdf_type``. Default to
    ``schema:Thing`` rather than dropping them; this matches the existing
    spaCy-fallback default in phases.py."""

    def test_entity_without_rdf_type_defaults_to_thing(self):
        # Verbatim sample from prod logs.
        payload = {
            "knowledge_type": "Entity",
            "uri": "exim",
            "label": "exim",
            "properties": {"description": "An email server software."},
            "confidence": 0.95,
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, EntityInput)
        assert result.rdf_type == "schema:Thing"

    def test_entity_with_rdf_type_keeps_it(self):
        payload = {
            "knowledge_type": "Entity",
            "uri": "x",
            "label": "x",
            "rdf_type": "schema:Person",
        }
        result = _ADAPTER.validate_python(payload)
        assert result.rdf_type == "schema:Person"

    def test_entity_with_just_uri_label_kt_validates(self):
        # Shape: {knowledge_type, label, uri} — 50 prod hits. No rdf_type,
        # no properties — should default and pass.
        payload = {"uri": "use_nix_atc", "label": "use_nix_atc", "knowledge_type": "entity"}
        result = _ADAPTER.validate_python(payload)
        assert result.rdf_type == "schema:Thing"

    def test_rdf_type_default_emits_debug_log(self, caplog):
        # Operators need a way to chart how often the LLM omits rdf_type.
        import logging

        with caplog.at_level(logging.DEBUG, logger="knowledge_service.models"):
            _ADAPTER.validate_python({"knowledge_type": "Entity", "uri": "x", "label": "x"})
        debug_lines = [r.getMessage() for r in caplog.records if "defaulted" in r.getMessage()]
        assert debug_lines, "expected DEBUG line when rdf_type defaulted"


class TestEntityMisplacedConfidence:
    """26 rejections — qwen3 nests ``confidence`` inside properties when the
    top-level field is missing. Lift it (and drop from properties)."""

    def test_misplaced_confidence_is_lifted(self):
        payload = {
            "knowledge_type": "Entity",
            "uri": "caltech",
            "rdf_type": "schema:Organization",
            "label": "caltech",
            "properties": {"confidence": 0.95},
        }
        result = _ADAPTER.validate_python(payload)
        assert isinstance(result, EntityInput)
        assert result.confidence == 0.95
        assert "confidence" not in result.properties

    def test_top_level_confidence_wins_over_misplaced(self):
        payload = {
            "knowledge_type": "Entity",
            "uri": "x",
            "rdf_type": "schema:Thing",
            "label": "x",
            "confidence": 0.7,
            "properties": {"confidence": 0.95},
        }
        result = _ADAPTER.validate_python(payload)
        assert result.confidence == 0.7


class TestSchemaPlaceholderRejection:
    """The LLM occasionally emits the literal placeholder strings ``"schema"``,
    ``"schema_person"``, ``"schema:Country"`` etc. as entity URIs / triple
    subjects — likely pattern-matched off the ``rdf_type`` example like
    ``"schema:Person"`` in the prompt. Items with these placeholder URIs
    must be rejected at validation so they don't survive to ``to_triples()``
    and collect unrelated entities under one junk URI."""

    @pytest.mark.parametrize(
        "bad",
        [
            "schema",
            "Schema",
            "  schema  ",
            "schema_person",
            "Schema_Country",
            "schema:Person",
            "schema:Thing",
        ],
    )
    def test_entity_with_schema_uri_is_rejected(self, bad):
        payload = {
            "knowledge_type": "Entity",
            "uri": bad,
            "rdf_type": "schema:Person",
            "label": "anything",
        }
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python(payload)

    @pytest.mark.parametrize("bad", ["schema", "schema_country", "schema:Country"])
    def test_triple_with_schema_subject_is_rejected(self, bad):
        payload = {
            "knowledge_type": "Claim",
            "subject": bad,
            "predicate": "located_in",
            "object": "australia",
        }
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python(payload)

    @pytest.mark.parametrize("bad", ["schema", "schema_country"])
    def test_triple_with_schema_object_is_rejected(self, bad):
        payload = {
            "knowledge_type": "Claim",
            "subject": "australia",
            "predicate": "is_a",
            "object": bad,
        }
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python(payload)

    def test_event_with_schema_subject_is_rejected(self):
        payload = {
            "knowledge_type": "Event",
            "subject": "schema_event",
            "occurred_at": "2026-05-22",
        }
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python(payload)

    def test_legitimate_words_containing_schema_pass(self):
        """Real entities whose name *contains* "schema" but isn't the placeholder
        — e.g. "schemata", "json_schema_validator" — must still validate.
        Only the exact placeholders are rejected."""
        for ok in ("schemata", "json_schema_validator", "x_schema_y"):
            entity = EntityInput(uri=ok, rdf_type="schema:Thing", label=ok)
            assert entity.uri == ok
