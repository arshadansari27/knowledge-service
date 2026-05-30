"""Golden query set: dataclasses + a validating JSON loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_VALID_QUERY_TYPES = {"semantic", "entity", "graph", "gtd"}


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    query_type: str
    relevant_source_ids: list[str]
    reference_answer: str
    notes: str = ""


def _coerce_item(raw: dict) -> GoldenItem:
    required = ("id", "question", "query_type", "relevant_source_ids", "reference_answer")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"golden item missing required fields: {missing} in {raw!r}")
    if raw["query_type"] not in _VALID_QUERY_TYPES:
        raise ValueError(
            f"invalid query_type {raw['query_type']!r}; must be one of {_VALID_QUERY_TYPES}"
        )
    if not isinstance(raw["relevant_source_ids"], list):
        raise ValueError(f"relevant_source_ids must be a list in {raw['id']!r}")
    return GoldenItem(
        id=str(raw["id"]),
        question=str(raw["question"]),
        query_type=str(raw["query_type"]),
        relevant_source_ids=[str(x) for x in raw["relevant_source_ids"]],
        reference_answer=str(raw["reference_answer"]),
        notes=str(raw.get("notes", "")),
    )


def load_golden(path: str | Path) -> list[GoldenItem]:
    """Load + validate the golden set. Raises ValueError on any malformed item."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("golden file must be a JSON array of items")
    items = [_coerce_item(raw) for raw in data]
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise ValueError(f"duplicate golden id: {it.id!r}")
        seen.add(it.id)
    return items
