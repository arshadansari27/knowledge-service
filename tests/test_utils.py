from knowledge_service._utils import _is_uri, _triple_hash, _rdf_value_to_str
from pyoxigraph import NamedNode, Literal


def test_is_uri_http():
    assert _is_uri("http://example.com") is True


def test_is_uri_https():
    assert _is_uri("https://example.com") is True


def test_is_uri_urn():
    assert _is_uri("urn:isbn:123") is True


def test_is_uri_plain_string():
    assert _is_uri("some_value") is False


def test_triple_hash_is_deterministic():
    assert _triple_hash("http://s", "http://p", "o") == _triple_hash("http://s", "http://p", "o")


def test_triple_hash_differs_for_different_inputs():
    assert _triple_hash("http://s", "http://p", "a") != _triple_hash("http://s", "http://p", "b")


def test_rdf_value_to_str_named_node():
    assert _rdf_value_to_str(NamedNode("http://example.com")) == "http://example.com"


def test_rdf_value_to_str_literal():
    assert _rdf_value_to_str(Literal("hello")) == "hello"


def test_rdf_value_to_str_none():
    assert _rdf_value_to_str(None) == ""


def test_rdf_value_to_str_plain_string():
    assert _rdf_value_to_str("plain") == "plain"


from knowledge_service._utils import is_object_entity


class TestIsObjectEntity:
    def test_explicit_entity(self):
        assert is_object_entity({"object": "dopamine", "object_type": "entity"}) is True

    def test_explicit_literal(self):
        assert is_object_entity({"object": "dopamine", "object_type": "literal"}) is False

    def test_none_object_type_short_no_spaces_is_entity(self):
        assert is_object_entity({"object": "dopamine"}) is True

    def test_none_object_type_spaces_is_literal(self):
        assert is_object_entity({"object": "250% dopamine increase"}) is False

    def test_none_object_type_long_string_is_literal(self):
        assert is_object_entity({"object": "a" * 61}) is False

    def test_empty_object_is_not_entity(self):
        assert is_object_entity({"object": ""}) is False

    def test_pydantic_model_with_object_type(self):
        """Works with Pydantic models (attribute access, not dict)."""
        from knowledge_service.models import ClaimInput

        item = ClaimInput(
            subject="x", predicate="p", object="250%",
            object_type="literal", confidence=0.7,
        )
        assert is_object_entity(item) is False

    def test_pydantic_model_entity_heuristic(self):
        from knowledge_service.models import ClaimInput

        item = ClaimInput(
            subject="x", predicate="p", object="dopamine", confidence=0.7,
        )
        assert is_object_entity(item) is True


def test_object_type_survives_discriminated_union():
    """object_type flows through TypeAdapter(KnowledgeInput) validation."""
    from pydantic import TypeAdapter
    from knowledge_service.models import KnowledgeInput

    adapter = TypeAdapter(KnowledgeInput)
    item = adapter.validate_python({
        "knowledge_type": "Claim",
        "subject": "x",
        "predicate": "p",
        "object": "250%",
        "object_type": "literal",
        "confidence": 0.7,
    })
    assert item.object_type == "literal"
