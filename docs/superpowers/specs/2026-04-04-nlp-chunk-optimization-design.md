# NLP Chunk Optimization: Reduce LLM Usage During Ingestion

**Date:** 2026-04-04
**Status:** Draft
**Problem:** Ingesting large documents (PDFs, books) monopolizes the GPU because every chunk gets 2 LLM calls. A 25-chunk PDF requires ~50 LLM calls, blocking all other work.
**Goal:** Reduce LLM calls by 70-80% using cheap NLP-based filtering and single-pass extraction, with sensible baked-in defaults.

---

## Approach

Two complementary optimizations:

1. **NLP-scored chunk gating** — score chunks using cheap NLP signals, skip low-value chunks from LLM extraction entirely, fall back to spaCy NER entities for skipped chunks.
2. **Single-pass LLM extraction** — merge the current two-phase extraction (entities → relations) into a single LLM call per chunk.

Combined effect: a 25-chunk document goes from ~50 LLM calls to ~10-15.

---

## 1. Chunk Scoring & Gating

### New module: `ingestion/chunk_filter.py`

`score_chunk(chunk: dict, nlp_result: NlpResult | None) -> float` returns 0.0–1.0 representing extraction value.

`filter_chunks(chunk_records, nlp_results) -> tuple[list[int], list[int]]` returns `(extract_indices, skip_indices)`.

### Scoring signals

| Signal | Weight | Logic |
|--------|--------|-------|
| **Section skip-list** | instant 0.0 | Regex on `section_header` or first line: "References", "Bibliography", "Acknowledgements", "Table of Contents", "Appendix", "Index", "About the Author" → 0.0 |
| **NER entity density** | 0.4 | `entity_count / (len(chunk_text) / 100)`. More named entities = more extractable knowledge |
| **Sentence count** | 0.2 | Chunks with < 3 sentences score low (tables, number lists, fragment chunks) |
| **Boilerplate ratio** | 0.2 | Fraction matching citation patterns (`[1]`, `(Author, YYYY)`), bare URLs, numeric-heavy lines. High ratio → low score |
| **Lexical diversity** | 0.2 | Unique words / total words. Very low diversity = repetitive content (data tables) |

### Threshold

Chunks scoring below 0.3 are skipped from LLM extraction. No config knob — this is a baked-in default tuned to be aggressive but safe.

### Integration point

`ExtractPhase.run()` — after building `hint_map`, before the per-chunk loop. Calls `filter_chunks()` to partition chunks into extract vs skip sets.

---

## 2. Single-Pass LLM Extraction

### Change: Merge two-phase extraction in `ExtractionClient.extract()`

Currently `extract()` makes 2 sequential LLM calls per chunk (entities, then relations constrained to those entities). The new behavior is a single prompt that asks the LLM to:

1. First identify all entities and events
2. Then extract relations using those entities

In one JSON response:

```json
{
  "entities": [
    {"knowledge_type": "Entity", "uri": "dopamine", "label": "dopamine", "rdf_type": "schema:Chemical", "confidence": 0.9}
  ],
  "relations": [
    {"knowledge_type": "Claim", "subject": "cold_water_immersion", "predicate": "increases", "object": "dopamine", "object_type": "entity", "confidence": 0.8}
  ]
}
```

### What changes

- **`ExtractionClient.extract()`** — single LLM call, returns combined items list
- **`PromptBuilder`** — new merged template `base_combined.txt` with both entity and relation instructions, domain predicates, and NLP hints
- **Old templates** (`base_entities.txt`, `base_relations.txt`) — kept but unused, can be removed later

### What stays the same

- Output shape: `extract()` still returns `list[dict]` of knowledge items
- `ExtractPhase` — no changes, it already just calls `extract()` and collects items
- NLP hint injection — same mechanism, passed to the combined prompt
- `_extract_json()` parsing — same util, expects the new response shape

### Risk mitigation

The prompt explicitly instructs "first list entities, then list relations referencing those entities" — gives the LLM the same sequential reasoning as two separate calls within one generation.

---

## 3. NER Fallback for Skipped Chunks

### What happens to filtered-out chunks

- No LLM call
- All spaCy entities from the chunk's `NlpResult` are ingested as `EntityInput` at `nlp_entity_confidence` (0.5)
- Counted as `chunks_skipped` (new counter) rather than `chunks_extracted` or `chunks_failed`
- If NLP phase is disabled (no spaCy), skipped chunks produce nothing

### Job tracking

- New `chunks_skipped` column in `ingestion_jobs` table (migration required)
- Added to `_ALLOWED_JOB_COLUMNS` in `worker.py`
- Logged: "25 chunks: 12 extracted, 10 skipped, 3 failed"

### ExtractPhase loop change

```
for chunk in chunk_records:
    if chunk_index in skip_set:
        → emit NER fallback entities
        → chunks_skipped += 1
        → continue
    → single-pass LLM extract
    → NER fallback for missed entities (existing logic)
```

---

## 4. Testing

### New: `tests/test_chunk_filter.py`

- `score_chunk()` with references section → returns 0.0
- `score_chunk()` with entity-rich text → high score
- `score_chunk()` with boilerplate-heavy text → low score
- `score_chunk()` with low lexical diversity → low score
- `filter_chunks()` correctly partitions into extract vs skip sets

### Updated: extraction tests

- Single-pass extraction returns combined entities + relations from one call
- Merged prompt includes both entity and relation instructions
- NLP hints included in combined prompt

### Updated: phase tests

- `ExtractPhase` skips filtered chunks, emits NER fallback entities
- `chunks_skipped` counter tracked correctly
- Skipped chunks with no NLP results produce no items

### Unchanged: E2E tests

No changes — they hit the full pipeline, which will just be faster.

---

## 5. Files Changed

| File | Change |
|------|--------|
| `ingestion/chunk_filter.py` | **New** — `score_chunk()`, `filter_chunks()` |
| `ingestion/phases.py` | Integrate chunk filtering in `ExtractPhase.run()`, handle skip path |
| `clients/llm.py` | Merge two-phase extraction into single `extract()` call |
| `clients/prompt_builder.py` | New `base_combined.txt` template |
| `clients/templates/base_combined.txt` | **New** — merged entity + relation prompt |
| `ingestion/worker.py` | Add `chunks_skipped` to `_ALLOWED_JOB_COLUMNS` |
| `migrations/` | **New** — add `chunks_skipped` column to `ingestion_jobs` |
| `tests/test_chunk_filter.py` | **New** — chunk scoring tests |
| `tests/test_extraction.py` | Update for single-pass extraction |
| `tests/test_phases.py` | Update for skip path + counter |

---

## 6. Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| LLM calls per chunk | 2 | 0 (skipped) or 1 (extracted) |
| Chunks sent to LLM (25-chunk doc) | 25 | ~10-15 |
| Total LLM calls (25-chunk doc) | 50 | ~10-15 |
| **Overall reduction** | | **~70-80%** |

Coreference phase is unchanged (still 1 LLM call total). Community summarization is unchanged.
