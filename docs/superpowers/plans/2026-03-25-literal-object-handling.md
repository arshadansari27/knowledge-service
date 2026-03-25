# Literal Object Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop literal objects (measurements, dates, descriptions) from being incorrectly treated as entities during knowledge ingestion.

**Architecture:** Add an optional `object_type` field to `TripleInput`, move the entity/literal decision function to `_utils.py`, and guard `_resolve_labels` and `_apply_uri_fallback` so they skip entity resolution for literals.

**Tech Stack:** Python 3.12, Pydantic v2, FastAPI, pyoxigraph, pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-03-25-literal-object-handling-design.md`

---

### Task 1: Add `is_object_entity` to `_utils.py`

**Files:**
- Modify: `src/knowledge_service/_utils.py:12` (add after `_is_uri`)
- Test: `tests/test_utils.py` (create if doesn't exist, or append)

- [ ] **Step 1: Write failing tests for `is_object_entity` (dict-only tests)**

Add to `tests/test_utils.py`:

```python
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
```

Note: Pydantic model tests are deferred to Task 2 so this commit stays green.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_utils.py::TestIsObjectEntity -v`
Expected: FAIL — `ImportError: cannot import name 'is_object_entity'`

- [ ] **Step 3: Implement `is_object_entity` in `_utils.py`**

Add after the `_is_uri` function in `src/knowledge_service/_utils.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_utils.py::TestIsObjectEntity -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/_utils.py tests/test_utils.py
git commit -m "feat: add is_object_entity utility for entity/literal detection"
```

---

### Task 2: Add `object_type` field to `TripleInput`

**Files:**
- Modify: `src/knowledge_service/models.py:23-31`
- Test: `tests/test_utils.py` (existing Pydantic tests from Task 1)

- [ ] **Step 1: Write failing tests for Pydantic model + `object_type`**

Add to `TestIsObjectEntity` in `tests/test_utils.py`:

```python
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
```

Run: `uv run pytest tests/test_utils.py -v -k "pydantic"`
Expected: FAIL — `TypeError: unexpected keyword argument 'object_type'`

- [ ] **Step 2: Add `object_type` to `TripleInput`**

In `src/knowledge_service/models.py`, change `TripleInput`:

```python
class TripleInput(BaseModel):
    """Base for knowledge that maps to a single S-P-O triple."""

    subject: str
    predicate: str
    object: str
    object_type: Literal["entity", "literal"] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: date | None = None
    valid_until: date | None = None
```

- [ ] **Step 3: Write test that `object_type` survives discriminated union validation**

Add to `tests/test_utils.py`:

```python
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
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/test_utils.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/models.py tests/test_utils.py
git commit -m "feat: add object_type field to TripleInput model"
```

---

### Task 3: Guard `_resolve_labels` for literal objects

**Files:**
- Modify: `src/knowledge_service/api/content.py:28-58`
- Test: `tests/test_api_content.py` (or similar existing content test file)

- [ ] **Step 1: Write failing test for `_resolve_labels` skipping literals**

Find the existing content test file and add:

```python
import pytest
from unittest.mock import AsyncMock

from knowledge_service.api.content import _resolve_labels
from knowledge_service.models import ClaimInput


async def test_resolve_labels_skips_literal_object():
    """Literal objects should NOT be resolved through entity_resolver."""
    item = ClaimInput(
        subject="cold_exposure",
        predicate="increases_by",
        object="250% dopamine increase",
        object_type="literal",
        confidence=0.7,
    )
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(return_value="http://knowledge.local/data/cold_exposure")
    resolver.resolve_predicate = AsyncMock(return_value="http://knowledge.local/schema/increases_by")

    count, result = await _resolve_labels(item, resolver)

    # subject and predicate resolved, but object was NOT
    assert resolver.resolve.call_count == 1  # only subject
    assert result.object == "250% dopamine increase"  # unchanged


async def test_resolve_labels_resolves_entity_object():
    """Entity objects should still be resolved."""
    item = ClaimInput(
        subject="cold_exposure",
        predicate="increases",
        object="dopamine",
        object_type="entity",
        confidence=0.7,
    )
    resolver = AsyncMock()
    resolver.resolve = AsyncMock(return_value="http://knowledge.local/data/resolved")
    resolver.resolve_predicate = AsyncMock(return_value="http://knowledge.local/schema/increases")

    count, result = await _resolve_labels(item, resolver)

    assert resolver.resolve.call_count == 2  # subject + object
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_content.py -v -k "resolve_labels_skips_literal or resolve_labels_resolves_entity"`
Expected: FAIL — resolver.resolve.call_count assertion fails (currently always resolves object)

- [ ] **Step 3: Fix `_resolve_labels` in `content.py`**

Add import at top of `src/knowledge_service/api/content.py`:

```python
from knowledge_service._utils import _is_uri, is_object_entity
```

(Remove the standalone `from knowledge_service._utils import _is_uri` import.)

Change lines 43-45 from:

```python
        if not _is_uri(item.object):
            item.object = await entity_resolver.resolve(item.object)
            resolved += 1
```

To:

```python
        if not _is_uri(item.object) and is_object_entity(item):
            item.object = await entity_resolver.resolve(item.object)
            resolved += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api_content.py -v -k "resolve_labels_skips_literal or resolve_labels_resolves_entity"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/api/content.py tests/test_api_content.py
git commit -m "fix: skip entity resolution for literal objects in _resolve_labels"
```

---

### Task 4: Guard `_apply_uri_fallback` for literal objects

**Files:**
- Modify: `src/knowledge_service/api/content.py:61-93`
- Test: `tests/test_api_content.py`

- [ ] **Step 1: Write failing test for `_apply_uri_fallback` preserving literals**

```python
from knowledge_service.api.content import _apply_uri_fallback
from knowledge_service.models import ClaimInput


def test_apply_uri_fallback_preserves_literal_object():
    """Literal objects should NOT be converted to entity URIs."""
    item = ClaimInput(
        subject="http://knowledge.local/data/cold_exposure",
        predicate="http://knowledge.local/schema/increases",
        object="250% dopamine increase",
        object_type="literal",
        confidence=0.7,
    )
    result = _apply_uri_fallback(item)
    assert result.object == "250% dopamine increase"


def test_apply_uri_fallback_converts_entity_object():
    """Entity objects without URIs should still be converted."""
    item = ClaimInput(
        subject="http://knowledge.local/data/cold_exposure",
        predicate="http://knowledge.local/schema/increases",
        object="dopamine",
        object_type="entity",
        confidence=0.7,
    )
    result = _apply_uri_fallback(item)
    assert result.object.startswith("http://knowledge.local/data/")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_content.py -v -k "apply_uri_fallback"`
Expected: FAIL — literal object gets converted to URI

- [ ] **Step 3: Fix `_apply_uri_fallback` in `content.py`**

Change lines 75-77 from:

```python
        obj = item.object
        if obj and not _is_uri(obj):
            item.object = to_entity_uri(obj)
```

To:

```python
        obj = item.object
        if obj and not _is_uri(obj) and is_object_entity(item):
            item.object = to_entity_uri(obj)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api_content.py -v -k "apply_uri_fallback"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/api/content.py tests/test_api_content.py
git commit -m "fix: skip URI conversion for literal objects in _apply_uri_fallback"
```

---

### Task 5: Delete dead code from `llm.py` and migrate tests

**Files:**
- Modify: `src/knowledge_service/clients/llm.py:193-235` (delete)
- Modify: `tests/test_extraction_client.py:5-11,186-255` (remove/migrate)

- [ ] **Step 1: Delete `_is_object_entity` and `normalize_item_uris` from `llm.py`**

Remove lines 193-235 from `src/knowledge_service/clients/llm.py` (the `_is_object_entity` function and the `normalize_item_uris` function).

- [ ] **Step 2: Remove the import from `tests/test_extraction_client.py`**

Change the import block at line 5-11 from:

```python
from knowledge_service.clients.llm import (
    ExtractionClient,
    normalize_item_uris,
    to_entity_uri,
    to_predicate_uri,
    resolve_predicate_synonym,
)
```

To:

```python
from knowledge_service.clients.llm import (
    ExtractionClient,
    to_entity_uri,
    to_predicate_uri,
    resolve_predicate_synonym,
)
```

- [ ] **Step 3: Delete only the `normalize_item_uris` test methods (lines 183-255)**

In `tests/test_extraction_client.py`, remove these methods from `class TestUriNormalisation`:
- `test_normalize_claim_subject_and_predicate` (line 183)
- `test_normalize_leaves_literal_objects_unchanged` (line 196)
- `test_normalize_resolves_predicate_synonym` (line 207)
- `test_object_type_entity_converts_to_uri` (line 218)
- `test_object_type_literal_preserved` (line 231)
- `test_missing_object_type_falls_back_to_heuristic` (line 245)

**Keep** the class and its first 4 methods (lines 168-181): `testto_entity_uri_slugifies`, `testto_entity_uri_preserves_existing_uri`, `testto_predicate_uri_slugifies`, `testto_predicate_uri_preserves_existing_uri`. These test `to_entity_uri` and `to_predicate_uri` which are NOT being deleted.

- [ ] **Step 3b: Add replacement tests for subject/predicate normalization in `tests/test_api_content.py`**

The deleted tests covered subject/predicate URI normalization and synonym resolution, which now live in `_apply_uri_fallback`. Add to `tests/test_api_content.py`:

```python
def test_apply_uri_fallback_normalizes_subject_and_predicate():
    """Subject and predicate are converted to URIs by _apply_uri_fallback."""
    from knowledge_service.api.content import _apply_uri_fallback
    from knowledge_service.models import ClaimInput

    item = ClaimInput(
        subject="cold_exposure",
        predicate="increases",
        object="dopamine",
        object_type="entity",
        confidence=0.7,
    )
    result = _apply_uri_fallback(item)
    assert result.subject == "http://knowledge.local/data/cold_exposure"
    assert result.predicate == "http://knowledge.local/schema/increases"


def test_apply_uri_fallback_resolves_predicate_synonym():
    """Predicate synonyms are resolved before URI conversion."""
    from knowledge_service.api.content import _apply_uri_fallback
    from knowledge_service.models import ClaimInput

    item = ClaimInput(
        subject="http://knowledge.local/data/x",
        predicate="boosts",
        object="y",
        object_type="entity",
        confidence=0.7,
    )
    result = _apply_uri_fallback(item)
    assert result.predicate == "http://knowledge.local/schema/increases"
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS — no remaining references to deleted functions

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/clients/llm.py tests/test_extraction_client.py
git commit -m "chore: remove dead normalize_item_uris and _is_object_entity from llm.py"
```

---

### Task 6: End-to-end integration test

**Files:**
- Test: `tests/test_api_content.py`

- [ ] **Step 1: Write end-to-end test for mixed entity/literal ingestion**

This test verifies the full pipeline using the existing `client` fixture from `tests/test_api_content.py`. It asserts against mock `insert_triple` call args to verify literal objects are passed as plain strings (not URIs).

```python
async def test_content_ingestion_preserves_literal_objects(client):
    """End-to-end: literal objects stay as plain strings through the pipeline."""
    resp = await client.post(
        "/api/content",
        json={
            "url": "http://example.com/test",
            "title": "Test",
            "source_type": "article",
            "knowledge": [
                {
                    "knowledge_type": "Claim",
                    "subject": "cold_exposure",
                    "predicate": "increases",
                    "object": "dopamine",
                    "object_type": "entity",
                    "confidence": 0.7,
                },
                {
                    "knowledge_type": "Claim",
                    "subject": "cold_exposure",
                    "predicate": "has_effect",
                    "object": "250% dopamine increase lasting 2 hours",
                    "object_type": "literal",
                    "confidence": 0.7,
                },
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["triples_created"] == 2

    # Get insert_triple call args from the mock knowledge_store
    # The client fixture sets up app.state.knowledge_store as a MagicMock
    ks = client._transport.app.state.knowledge_store
    calls = ks.insert_triple.call_args_list
    assert len(calls) == 2

    # First call: entity object should be a URI
    entity_call_kwargs = calls[0].kwargs if calls[0].kwargs else {}
    entity_call_args = calls[0].args if calls[0].args else ()
    # insert_triple(subject, predicate, object_, confidence, ...)
    # object_ is the 3rd positional arg
    entity_obj = entity_call_args[2] if len(entity_call_args) > 2 else entity_call_kwargs.get("object_")
    assert entity_obj.startswith("http://")

    # Second call: literal object should NOT be a URI
    literal_call_args = calls[1].args if calls[1].args else ()
    literal_call_kwargs = calls[1].kwargs if calls[1].kwargs else {}
    literal_obj = literal_call_args[2] if len(literal_call_args) > 2 else literal_call_kwargs.get("object_")
    assert literal_obj == "250% dopamine increase lasting 2 hours"
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_api_content.py -v -k "preserves_literal"`
Expected: PASS

- [ ] **Step 3: Run full test suite to confirm no regressions**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_api_content.py
git commit -m "test: add end-to-end test for literal object handling"
```

---

### Task 7: Final lint and format check

**Files:** All modified files

- [ ] **Step 1: Run linter**

Run: `uv run ruff check .`
Expected: No errors

- [ ] **Step 2: Run formatter check**

Run: `uv run ruff format --check .`
If fails: `uv run ruff format .`

- [ ] **Step 3: Run full test suite one more time**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Final commit if formatting changed**

```bash
git add -u
git commit -m "style: format changes for literal object handling"
```
