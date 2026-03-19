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
