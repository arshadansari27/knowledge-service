"""Tests for centralized URI normalization."""

from knowledge_service.ontology.namespaces import ensure_entity_uri, ensure_predicate_uri


def test_ensure_entity_uri_passthrough():
    assert ensure_entity_uri("http://example.com/foo") == "http://example.com/foo"
    assert ensure_entity_uri("https://example.com/foo") == "https://example.com/foo"
    assert ensure_entity_uri("urn:example:foo") == "urn:example:foo"


def test_ensure_entity_uri_bare_label():
    result = ensure_entity_uri("cold_exposure")
    assert result == "http://knowledge.local/data/cold_exposure"


def test_ensure_entity_uri_slugifies():
    result = ensure_entity_uri("Cold Exposure!")
    assert result.startswith("http://knowledge.local/data/")
    assert " " not in result


def test_ensure_predicate_uri_passthrough():
    assert (
        ensure_predicate_uri("http://knowledge.local/schema/causes")
        == "http://knowledge.local/schema/causes"
    )


def test_ensure_predicate_uri_bare_label():
    result = ensure_predicate_uri("causes")
    assert result == "http://knowledge.local/schema/causes"
