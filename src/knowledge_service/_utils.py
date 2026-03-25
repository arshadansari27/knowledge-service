"""Shared utility helpers reused across API and store modules."""

from __future__ import annotations

import hashlib
import json
import re

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
    start = stripped.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(stripped)):
            if stripped[i] == "{":
                depth += 1
            elif stripped[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
        start = stripped.find("{", start + 1)
    return None
