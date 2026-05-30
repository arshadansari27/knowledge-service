"""Tests for the golden-set loader and validation."""

from __future__ import annotations

import json

import pytest

from knowledge_service.eval.golden import GoldenItem, load_golden


def _write(tmp_path, items):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(items))
    return path


def test_loads_valid_items(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "id": "q1",
                "question": "What is X?",
                "query_type": "entity",
                "relevant_source_ids": ["c1", "c2"],
                "reference_answer": "X is a thing.",
                "notes": "",
            }
        ],
    )
    items = load_golden(path)
    assert len(items) == 1
    assert isinstance(items[0], GoldenItem)
    assert items[0].id == "q1"
    assert items[0].query_type == "entity"
    assert items[0].relevant_source_ids == ["c1", "c2"]


def test_rejects_unknown_query_type(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "id": "q1",
                "question": "Q",
                "query_type": "nonsense",
                "relevant_source_ids": [],
                "reference_answer": "A",
            }
        ],
    )
    with pytest.raises(ValueError, match="query_type"):
        load_golden(path)


def test_rejects_duplicate_ids(tmp_path):
    item = {
        "id": "dup",
        "question": "Q",
        "query_type": "semantic",
        "relevant_source_ids": [],
        "reference_answer": "A",
    }
    path = _write(tmp_path, [item, dict(item)])
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(path)


def test_missing_required_field_raises(tmp_path):
    path = _write(tmp_path, [{"id": "q1", "question": "Q", "query_type": "semantic"}])
    with pytest.raises(ValueError):
        load_golden(path)


def test_empty_array_is_valid(tmp_path):
    path = _write(tmp_path, [])
    assert load_golden(path) == []
