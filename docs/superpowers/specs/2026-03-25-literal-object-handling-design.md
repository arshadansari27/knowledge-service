# Literal Object Handling Design

## Problem

Literal objects (e.g., "250% dopamine increase", "2024-01-15") are incorrectly treated as entities throughout the ingestion pipeline. This causes:

1. Literals embedded and stored in `entity_embeddings` as if they were entities
2. Literals converted to entity URIs (e.g., `http://knowledge.local/data/250__dopamine_increase`)
3. Corrupted query results where measurements and dates appear as entity nodes

### Root Cause

The LLM extraction prompt correctly asks for `object_type` ("entity" or "literal"), but:

- `TripleInput` Pydantic model has no `object_type` field, so the hint is silently dropped
- `_resolve_labels()` in `content.py` blindly resolves all objects through `entity_resolver.resolve()`
- `_apply_uri_fallback()` in `content.py` blindly converts all non-URI objects to entity URIs
- `normalize_item_uris()` in `llm.py` was designed to handle this correctly but is dead code (never called in the pipeline, only referenced in tests)

## Solution: Thread `object_type` Through the Model Layer

### 1. Add `object_type` to `TripleInput`

**File:** `src/knowledge_service/models.py`

Add an optional field to `TripleInput`:

```python
class TripleInput(BaseModel):
    subject: str
    predicate: str
    object: str
    object_type: Literal["entity", "literal"] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    valid_from: date | None = None
    valid_until: date | None = None
```

This flows to `ClaimInput`, `FactInput`, and `RelationshipInput` via inheritance. The `/api/claims` endpoint accepts it optionally from users; the LLM extraction pipeline sets it explicitly.

### 2. Move `_is_object_entity` to shared utility

**From:** `src/knowledge_service/clients/llm.py`
**To:** `src/knowledge_service/_utils.py`

Adapt the function to work with both dicts (LLM output) and Pydantic models:

```python
def is_object_entity(item) -> bool:
    """Decide whether an item's object field is an entity reference (vs a literal).

    Checks the object_type hint first (from LLM or user), falls back to
    heuristic: no spaces and <= 60 chars suggests an entity.
    """
    obj_type = item.get("object_type", "") if isinstance(item, dict) else getattr(item, "object_type", "")
    if obj_type == "entity":
        return True
    if obj_type == "literal":
        return False
    obj = item.get("object", "") if isinstance(item, dict) else getattr(item, "object", "")
    return bool(obj) and " " not in obj and len(obj) <= 60
```

### 3. Fix `_resolve_labels` in `content.py`

**File:** `src/knowledge_service/api/content.py`

Guard entity resolution on object type:

```python
if not _is_uri(item.object) and is_object_entity(item):
    item.object = await entity_resolver.resolve(item.object)
    resolved += 1
```

When the object is a literal, skip `entity_resolver.resolve()` entirely. This prevents:
- Embedding literal values in `entity_embeddings`
- Returning a URI for a measurement or date string

### 4. Fix `_apply_uri_fallback` in `content.py`

**File:** `src/knowledge_service/api/content.py`

Same guard for URI conversion:

```python
obj = item.object
if obj and not _is_uri(obj) and is_object_entity(item):
    item.object = to_entity_uri(obj)
# Literals are left as plain strings; _to_rdf_term() in knowledge.py
# creates RDF Literal nodes from non-URI strings.
```

### 5. Wire `object_type` into extraction pipeline

**File:** `src/knowledge_service/clients/llm.py`

In `ExtractionClient.extract()` phase 2, the LLM already produces `object_type` in its output dicts. Currently `normalize_item_uris()` would strip it before Pydantic validation, but since `normalize_item_uris()` is never called, the field is simply dropped by Pydantic's strict validation.

With the new `object_type` field on `TripleInput`, the LLM's hint flows through naturally. No extraction code changes needed.

### 6. Delete dead code: `normalize_item_uris` and `_is_object_entity` from `llm.py`

**File:** `src/knowledge_service/clients/llm.py`

Remove `normalize_item_uris()` (lines ~209-235) and the private `_is_object_entity()` (lines ~193-206). They are dead code:
- `normalize_item_uris` is only referenced in `test_extraction_client.py` tests
- `_is_object_entity` is only called by `normalize_item_uris`

The logic is replaced by `is_object_entity()` in `_utils.py` and the guards in `_resolve_labels`/`_apply_uri_fallback`.

Update `test_extraction_client.py` to remove tests for the deleted function, or migrate them to test `is_object_entity` in `_utils.py`.

### 7. Strip `object_type` before storage

**File:** `src/knowledge_service/models.py`

In `expand_to_triples()`, exclude `object_type` from the triple dicts it produces. It is routing metadata, not RDF data. The existing `_to_rdf_term()` in `knowledge.py` already converts plain strings to RDF Literal nodes.

### 8. Add entity resolution to `/api/claims`

**File:** `src/knowledge_service/api/claims.py`

Currently `/api/claims` skips `_resolve_labels` and `_apply_uri_fallback`, so manually submitted claims bypass entity deduplication entirely. Add the same resolution + fallback pipeline:

```python
entity_resolver = getattr(request.app.state, "entity_resolver", None)

for item in body.knowledge:
    if entity_resolver is not None:
        _, item = await _resolve_labels(item, entity_resolver)
    item = _apply_uri_fallback(item)
    for t in expand_to_triples(item):
        # ... existing process_triple logic
```

Import `_resolve_labels` and `_apply_uri_fallback` from `content.py`, or move them to a shared module if preferred.

## Data Flow After Fix

```
LLM output: {subject: "cold_exposure", predicate: "increases_by", object: "250%", object_type: "literal"}
    |
    v
Pydantic validation: ClaimInput(object_type="literal")  -- field preserved
    |
    v
_resolve_labels: object_type="literal" → skip entity_resolver.resolve()
    |
    v
_apply_uri_fallback: object_type="literal" → leave "250%" as plain string
    |
    v
expand_to_triples: strips object_type, passes "250%" as object
    |
    v
insert_triple → _to_rdf_term("250%") → RDF Literal("250%")
```

For entities, the flow is unchanged — they still get resolved and URI-ified.

## Testing

- Unit test `is_object_entity()` with explicit entity/literal hints and heuristic fallback
- Test `_resolve_labels` skips resolver for literal objects
- Test `_apply_uri_fallback` leaves literal objects as plain strings
- Test end-to-end: content ingestion with mixed entity/literal objects produces correct RDF terms
- Test `/api/claims` with `object_type` specified and omitted (heuristic fallback)
- Verify `expand_to_triples` excludes `object_type` from output dicts

## Files Changed

| File | Change |
|------|--------|
| `src/knowledge_service/models.py` | Add `object_type` to `TripleInput`; strip in `expand_to_triples` |
| `src/knowledge_service/_utils.py` | Add `is_object_entity()` |
| `src/knowledge_service/api/content.py` | Guard object resolution/fallback on `is_object_entity()` |
| `src/knowledge_service/api/claims.py` | Add entity resolution + fallback pipeline |
| `src/knowledge_service/clients/llm.py` | Delete `normalize_item_uris()` and `_is_object_entity()` |
| `tests/test_extraction_client.py` | Remove/migrate `normalize_item_uris` tests |
| `tests/` | New tests for literal handling |
