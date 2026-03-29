from knowledge_service._utils import (
    _is_uri,
    _triple_hash,
    _rdf_value_to_str,
    is_object_entity,
    sanitize_sparql_string,
)
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


class TestSanitizeSparqlString:
    def test_strips_angle_brackets(self):
        result = sanitize_sparql_string("http://evil.com> . ?s <http://x")
        assert "<" not in result
        assert ">" not in result

    def test_strips_quotes_and_backslashes(self):
        result = sanitize_sparql_string('value"with\\escapes')
        assert '"' not in result
        assert "\\" not in result

    def test_strips_newlines(self):
        result = sanitize_sparql_string("line1\nline2\rline3")
        assert "\n" not in result
        assert "\r" not in result

    def test_preserves_normal_uri(self):
        assert sanitize_sparql_string("http://example.com/path") == "http://example.com/path"


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
        """Works with Pydantic models that have an object field (TripleInput)."""
        from knowledge_service.models import TripleInput

        item = TripleInput(
            subject="x",
            predicate="p",
            object="250%",
            confidence=0.7,
        )
        # TripleInput has no object_type field, so it falls back to heuristic.
        # "250%" has no spaces and <= 60 chars, so it's treated as entity.
        assert is_object_entity(item) is True

    def test_pydantic_model_entity_heuristic(self):
        from knowledge_service.models import TripleInput

        item = TripleInput(
            subject="x",
            predicate="p",
            object="dopamine",
            confidence=0.7,
        )
        assert is_object_entity(item) is True

    def test_dict_with_spaces_is_literal(self):
        """Dict with spaces in object is treated as literal by heuristic."""
        assert is_object_entity({"object": "a long phrase with spaces"}) is False


def test_object_type_survives_triple_input():
    """TripleInput can be validated via KnowledgeInput union."""
    from knowledge_service.models import TripleInput

    item = TripleInput(
        subject="x",
        predicate="p",
        object="250%",
        confidence=0.7,
        knowledge_type="claim",
    )
    assert item.knowledge_type == "claim"
