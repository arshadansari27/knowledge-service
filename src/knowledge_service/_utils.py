"""Shared utility helpers reused across API and store modules."""

from __future__ import annotations

import hashlib
import json
import re

from pyoxigraph import Literal, NamedNode, Triple


def _is_uri(value: str) -> bool:
    return value.startswith(("http://", "https://", "urn:"))


def sanitize_sparql_string(value: str) -> str:
    """Sanitize a string for safe inclusion in SPARQL queries."""
    return re.sub(r'["\\\n\r<>]', "", value)


def is_object_entity(item) -> bool:
    """Decide whether an item's object field is an entity reference (vs a literal).

    Checks the object_type hint first (from LLM or user), falls back to
    heuristic: no spaces and <= 60 chars suggests an entity.

    Works with both dicts and Pydantic models.
    """
    obj_type = (
        item.get("object_type") if isinstance(item, dict) else getattr(item, "object_type", None)
    )
    if obj_type == "entity":
        return True
    if obj_type == "literal":
        return False
    obj = item.get("object", "") if isinstance(item, dict) else getattr(item, "object", "")
    return bool(obj) and " " not in obj and len(obj) <= 60


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


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from freeform LLM output.

    Handles markdown code fences, qwen3 <think> tags, and trailing text.
    Returns None if no valid JSON object is found.
    """
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped)
    stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL).strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    start = stripped.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(stripped, start)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
        start = stripped.find("{", start + 1)
    return None
