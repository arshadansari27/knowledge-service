# Knowledge Service Chunking Enhancement Design

**Date**: 2026-03-20
**Status**: Draft
**Author**: Arshad + Claude
**Related**: `aegis/docs/superpowers/specs/2026-03-20-content-intelligence-pipeline-design.md`

## Problem

The knowledge service stores content as single full-text units — one embedding per document. This works poorly for RAG when documents are long (articles, research papers, transcripts). Search returns entire documents ranked by similarity, but the relevant information may be in a small section. LLM context gets filled with irrelevant text surrounding the useful parts.

AEGIS is building a content extraction pipeline that will send full article text (not just URLs/titles) to the knowledge service. With richer, longer content coming in, chunk-level storage and retrieval becomes essential.

## Goal

Restructure content storage so every piece of content is stored as one or more chunks (short content = 1 chunk, long content = N chunks). Search and RAG always operate at chunk level via a single query — no fallback logic.

## Design Principles

- **Always chunk** — every content item has >= 1 chunk, eliminating dual-path search
- **Normalized** — metadata lives in `content_metadata`, chunks in `content`
- **Single search path** — one JOIN query, no UNION ALL or fallback
- **Same patterns** — HNSW + halfvec index, same embedding model

## Schema Changes

### New Table: `content_metadata`

Stores document-level metadata. One row per ingested document. URL is the dedup key.

```sql
CREATE TABLE content_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    summary TEXT,
    raw_text TEXT,
    source_type TEXT NOT NULL,
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### Restructured Table: `content`

Repurposed to store chunks. Each row is a searchable chunk with its own embedding. Every `content_metadata` row has >= 1 `content` row.

```sql
DROP TABLE IF EXISTS content;  -- existing data can be dropped

CREATE TABLE content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL REFERENCES content_metadata(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(768),
    char_start INTEGER,
    char_end INTEGER,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(content_id, chunk_index)
);

CREATE INDEX idx_content_embedding ON content
    USING hnsw ((embedding::halfvec(768)) halfvec_cosine_ops);
CREATE INDEX idx_content_content_id ON content (content_id);
```

### `/api/content` Enhancement

Current flow: embed full text → upsert content → extract triples → store.

New flow:

1. **Upsert metadata** into `content_metadata` (url, title, source_type, etc.) → get `content_id`
2. **Chunk text** using `RecursiveCharacterTextSplitter`
   - Chunk size: ~4000 chars (~1000 tokens) with 200-char overlap
   - Split hierarchy: paragraph breaks → sentences → words
   - **Short content (< 4000 chars)**: 1 chunk = the full text (no splitting needed)
3. **Embed chunks** via batch embedding call
4. **Delete old chunks** for this `content_id` (re-ingestion) and **insert new chunks**
5. Extract triples from full text → store (unchanged)

### `/api/search` Enhancement

Current: queries `content` table, returns document-level results.

New: single query against `content JOIN content_metadata`. Every result is a chunk.

```sql
SELECT
    c.id, c.chunk_text, c.chunk_index,
    m.id AS content_id, m.url, m.title, m.summary,
    m.source_type, m.tags, m.ingested_at,
    1 - (c.embedding::halfvec(768) <=> $1::halfvec(768)) AS similarity
FROM content c
JOIN content_metadata m ON c.content_id = m.id
WHERE ...filters on m.source_type, m.tags...
ORDER BY c.embedding::halfvec(768) <=> $1::halfvec(768)
LIMIT $N
```

No UNION ALL. No fallback. One query.

**Response shape change**: `SearchResult` gains `chunk_text` and `chunk_index` fields (always present, not optional).

```python
class SearchResult(BaseModel):
    content_id: str
    url: str
    title: str
    summary: str | None
    similarity: float
    source_type: str
    tags: list[str]
    ingested_at: datetime
    chunk_text: str        # NEW — the matching chunk text (always present)
    chunk_index: int       # NEW — position in parent document (always present)
```

### `/api/ask` Enhancement

The RAG retriever calls `EmbeddingStore.search()`, which now returns chunk-level results automatically. No code change needed beyond what `search()` already returns.

- More precise context for the LLM (relevant section, not whole document)
- Can include more sources in context window (chunks are smaller)
- Source attribution becomes more specific (document + chunk position)

No change to the `AskRequest`/`AskResponse` API shape.

### What Doesn't Change

- `/api/claims` — claims ingestion is independent of content chunking
- `/api/knowledge/query` — SPARQL/triple queries are unaffected
- Triple extraction — still runs on full text, not per-chunk
- Contradiction detection — unchanged
- Entity resolution — unchanged

## Dependencies

### New Python Package

- `langchain-text-splitters` — provides `RecursiveCharacterTextSplitter`

Lightweight package (~50KB) with no heavy transitive deps.

### Migration

Migration 002: drop old `content` table, create `content_metadata` and new `content` (chunks) table. Existing data is dropped (acceptable at v0.1.x).

## Implementation Notes

- Chunking runs inside `_process_one_content_request()` after metadata upsert, before triple extraction
- Embedding calls are batched (embed all chunks in one call via `embed_batch()`)
- `EmbeddingStore` methods change: `insert_content()` → `insert_content_metadata()`, `search()` rewritten with JOIN, new `insert_chunks()` and `delete_chunks()`
- The 60s timeout on `/api/content` may need to increase for long documents with many chunks
