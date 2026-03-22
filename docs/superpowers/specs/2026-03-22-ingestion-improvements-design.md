# Ingestion Improvements: Two-Phase Extraction + Structure-Aware Chunking

**Date:** 2026-03-22
**Phase:** 5+6 of 9 (KG-RAG improvement roadmap)
**Scope:** Improve extraction quality via entity-first two-phase LLM extraction, and improve chunking quality via markdown-aware splitting with section metadata

---

## Context

Current extraction uses a single LLM call that produces all 7 knowledge types at once. This causes hallucinated relations — the LLM invents entity names during relation extraction that don't match earlier entity mentions. KGGen research shows two-phase extraction (entities first, relations second) reduces hallucinated triples by ~35%.

Current chunking uses `RecursiveCharacterTextSplitter` with no awareness of document structure. Chunks can split mid-section, losing heading context.

---

## Phase 5: Two-Phase Extraction

### Problem

Single-pass extraction asks the LLM to simultaneously identify entities, classify knowledge types, extract relations, assign confidence, and follow naming conventions. This cognitive overload leads to:
- Inconsistent entity naming across items (same entity named differently)
- Hallucinated entities in relations that weren't mentioned in the text
- Poor type discrimination (Claims vs Facts)

### Design

Split `ExtractionClient.extract()` into two sequential LLM calls.

#### Phase 1 — Entity extraction

```
Input: chunk text + title + source_type
Output: Entity + Event items only
```

Phase 1 prompt focuses on:
- Entity items: uri, rdf_type (Schema.org), label, properties, confidence
- Event items: subject, occurred_at, properties, confidence
- Entity naming rules (canonical names, snake_case, singular, specific)
- "Extract 2-6 entities/events. If nothing found, return empty."

#### Phase 2 — Relation extraction

```
Input: chunk text + entity list from phase 1
Output: Claim, Fact, Relationship, TemporalState, Conclusion items
```

Phase 2 prompt provides the entity names from phase 1 as a constraint:
- "Only use these entities as subjects/objects: [entity_1, entity_2, ...]"
- Uses canonical predicates list (18 predicates)
- Confidence scoring (Claim 0.0-0.89, Fact 0.9-1.0)
- object_type discrimination (entity vs literal)
- "Extract 3-8 relationships. Only use entities from the list above."

#### API contract unchanged

`ExtractionClient.extract()` still returns `list[KnowledgeInput]`. Internally:
1. Call LLM with phase 1 prompt → parse Entity/Event items
2. Extract entity names from phase 1 results
3. Call LLM with phase 2 prompt (including entity list) → parse relation items
4. Return concatenated list

Callers see no change.

#### Implementation in clients/llm.py

Replace `_build_extraction_prompt()` with:
- `_build_entity_extraction_prompt(text, title, source_type)` → phase 1
- `_build_relation_extraction_prompt(text, title, source_type, entities)` → phase 2

Replace `ExtractionClient.extract()` body:
1. Build phase 1 prompt, call LLM, parse response
2. Collect entity names: `[item.label for item in phase1_items if hasattr(item, 'label')]` + event subjects
3. Build phase 2 prompt with entity list, call LLM, parse response
4. Return `phase1_items + phase2_items`

If phase 1 returns no entities, skip phase 2 and return just phase 1 results (which may be empty).

#### Prompt design notes

- Phase 1 uses the same JSON response format (`{"items": [...]}`)
- Phase 2 uses the same JSON response format
- Both prompts include the same entity naming rules for consistency
- Phase 2 explicitly lists the extracted entity names as allowed subjects/objects
- Phase 2 can still produce literal objects (measurements, dates) — only entity references are constrained

### Constraints

- Always two-phase (no config toggle)
- 2x LLM calls per chunk (acceptable for batch ingestion)
- If phase 1 fails, return empty list (same as current behavior on failure)
- If phase 2 fails, return phase 1 results only (entities without relations is still useful)

### Tests

- Test phase 1 prompt contains entity-focused instructions
- Test phase 2 prompt contains entity list from phase 1
- Test extract() makes 2 LLM calls and returns combined results
- Test extract() returns only entities when phase 2 fails
- Test extract() returns empty when phase 1 fails
- Test entity names from phase 1 appear in phase 2 prompt

---

## Phase 6: Structure-Aware Chunking

### Problem

`RecursiveCharacterTextSplitter` treats all text as flat. Chunks can split mid-section, and heading context is lost. A chunk starting with "This approach works well for..." loses the context of what "this approach" refers to.

### Design

#### Markdown detection

Simple heuristic: if the text contains `# ` or `## ` in the first 2000 characters, treat as markdown.

```python
def _is_markdown(text: str) -> bool:
    sample = text[:2000]
    return bool(re.search(r'^#{1,3}\s', sample, re.MULTILINE))
```

#### Markdown path

Use `MarkdownHeaderTextSplitter` from `langchain_text_splitters` to split on headings first (`#`, `##`, `###`). Each resulting section gets header metadata (the heading hierarchy).

If any section exceeds `_CHUNK_SIZE` (4000 chars), sub-split with `RecursiveCharacterTextSplitter`.

Each chunk carries a `section_header` string like `"Key Findings > Methodology"` derived from the heading hierarchy.

#### Plain text path

Use `RecursiveCharacterTextSplitter` with improved separators:
```python
["# ", "## ", "### ", "\n\n", "\n", ". ", " ", ""]
```

This adds heading awareness as a bonus even for text that doesn't pass the markdown heuristic. `section_header` is `None` for these chunks.

#### Schema change

New migration `006_add_section_header.sql`:
```sql
ALTER TABLE content ADD COLUMN section_header TEXT;
```

Nullable — existing rows keep `NULL`.

#### Chunk record changes

The `chunk_records` dict gains `section_header: str | None`. Passed through `insert_chunks()` to the DB.

`insert_chunks()` SQL updated to include `section_header` in INSERT.

#### Search and RAG integration

`SearchResult` model gains `section_header: str | None = None`.

RAG prompt (`clients/rag.py`) shows section header when present:
```
- "Article Title" (article, similarity: 0.85) [Section: Key Findings > Methodology]: chunk text here
```

### Implementation in content.py

Replace the current chunking block with:

```python
from knowledge_service.chunking import chunk_text

chunks_with_headers = chunk_text(text, chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)
# Returns: list[dict] with keys: chunk_text, section_header
```

Extract the chunking logic into a dedicated module `src/knowledge_service/chunking.py` to keep `content.py` focused on the ingestion pipeline. The module exposes:

```python
def chunk_text(
    text: str,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """Split text into chunks with optional section headers.

    Returns list of {"chunk_text": str, "section_header": str | None}.
    Uses MarkdownHeaderTextSplitter for markdown, RecursiveCharacterTextSplitter for plain text.
    """
```

### Constraints

- No new dependencies — `MarkdownHeaderTextSplitter` is already in `langchain_text_splitters`
- Existing data unaffected — `section_header` is nullable, old chunks keep `NULL`
- No retroactive re-chunking — only new ingestions get structure-aware chunks
- Markdown detection is a simple heuristic — false positives are harmless (heading separators just add more split points)

### Tests

- Test markdown detection: text with `## Heading` is markdown, plain text is not
- Test markdown chunking: sections split on headings, section_header populated
- Test large markdown section gets sub-split by RecursiveCharacterTextSplitter
- Test plain text uses improved separators, section_header is None
- Test section_header stored in DB and returned in search results
- Test RAG prompt shows section header when present

---

## Cross-cutting

### File changes summary

| File | Phase | Change |
|---|---|---|
| `src/knowledge_service/clients/llm.py` | 5 | Two-phase prompts, updated extract() |
| `src/knowledge_service/chunking.py` | 6 | New module: markdown-aware chunking |
| `src/knowledge_service/api/content.py` | 6 | Use new chunking module |
| `src/knowledge_service/stores/embedding.py` | 6 | insert_chunks includes section_header |
| `src/knowledge_service/clients/rag.py` | 6 | Show section_header in prompt |
| `src/knowledge_service/models.py` | 6 | SearchResult gains section_header |
| `migrations/006_add_section_header.sql` | 6 | New column on content table |
| `tests/test_extraction_client.py` | 5 | Two-phase extraction tests |
| `tests/test_chunking.py` | 6 | Chunking module tests |

### Backward compatibility

- `ExtractionClient.extract()` API unchanged — callers unaffected
- `insert_chunks()` accepts optional `section_header` in chunk dicts — existing callers work
- `SearchResult.section_header` defaults to `None` — API consumers see no breaking change
- Old chunks keep `section_header = NULL`
