"""Shared RDF utility helpers reused across API and store modules."""

from __future__ import annotations
import hashlib
from pyoxigraph import Literal, NamedNode, Triple


def _is_uri(value: str) -> bool:
    return value.startswith(("http://", "https://", "urn:"))


def _to_rdf_term(value: str) -> NamedNode | Literal:
    if _is_uri(value):
        return NamedNode(value)
    return Literal(value)


def _triple_hash(subject: str, predicate: str, object_: str) -> str:
    """SHA-256 hash of a triple — must match KnowledgeStore.insert_triple logic."""
    s = NamedNode(subject)
    p = NamedNode(predicate)
    o = _to_rdf_term(object_)
    triple = Triple(s, p, o)
    return hashlib.sha256(str(triple).encode()).hexdigest()


def _rdf_value_to_str(value: object) -> str:
    """Convert a pyoxigraph RDF term or None to a plain Python string."""
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)
