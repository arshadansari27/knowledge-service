# BM25 Hybrid Search

**Date:** 2026-03-22
**Phase:** 4 of 9 (KG-RAG improvement roadmap)
**Scope:** Add PostgreSQL full-text search alongside existing vector search, fused via Reciprocal Rank Fusion

---

## Context

The system currently uses vector-only search (pgvector cosine similarity) for both `/api/search` and `RAGRetriever.retrieve()`. Vector search excels at semantic similarity but fails on exact-match lookups (names, IDs, dates, codes, technical terms). Every modern KG-RAG system (LightRAG, Graphiti, HippoRAG) uses hybrid retrieval as a baseline.

**What this enables:**
- Exact-match recall for entity names, technical terms, dates
- Better retrieval for queries like "PostgreSQL 16 configuration" where the exact term matters
- Foundation for Phase 7 (query intent routing can direct keyword-heavy queries to BM25)

---

## Design

### Schema: tsvector column on content table

New migration `005_add_fts.sql`:

```sql
ALTER TABLE content ADD COLUMN tsv tsvector;

CREATE INDEX idx_content_tsv ON content USING GIN(tsv);

CREATE OR REPLACE FUNCTION content_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('english', COALESCE(NEW.chunk_text, ''));
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_content_tsv
    BEFORE INSERT OR UPDATE OF chunk_text ON content
    FOR EACH ROW EXECUTE FUNCTION content_tsv_trigger();

UPDATE content SET tsv = to_tsvector('english', COALESCE(chunk_text, ''));
```

**Language config:** `english` — stemming and stop word removal. Content is predominantly natural language text. Technical terms that matter (entity names, predicates) are already in the knowledge graph; BM25 is for text chunks.

**Trigger-based:** The tsvector is populated automatically on INSERT/UPDATE. No changes needed to `insert_chunks()`.

**Backfill:** The `UPDATE` statement populates existing rows on migration.

### EmbeddingStore changes

#### New method: `search_bm25()`

```python
async def search_bm25(
    self,
    query_text: str,
    limit: int,
    source_type: str | None = None,
    tags: list[str] | None = None,
    min_date: Any | None = None,
) -> list[dict]:
```

Converts query via `plainto_tsquery('english', query_text)`. The SQL **must** include `WHERE c.tsv @@ query` to use the GIN index — without it, `ts_rank` would compute over all rows (O(N) sequential scan). Ranks matching rows with `ts_rank(c.tsv, query)`. Joins with `content_metadata` for filtering (same as vector search).

**Return shape:** Must return the same dict keys as `search()`, critically including `id` as the `content.id` chunk UUID (not `content_id`). This key is used by RRF for deduplication across result lists.

**Empty query guard:** If `plainto_tsquery('english', query_text)` produces an empty tsquery (e.g., stop-word-only input like "the"), return `[]` immediately rather than running a pointless query.

Uses `plainto_tsquery` (not `to_tsquery`) because user queries are natural language, not structured boolean expressions. `plainto_tsquery` handles spaces and punctuation safely.

#### Modified method: `search()` gains `query_text` parameter

```python
async def search(
    self,
    query_embedding: list[float],
    limit: int,
    source_type: str | None = None,
    tags: list[str] | None = None,
    min_date: Any | None = None,
    query_text: str | None = None,  # NEW
) -> list[dict]:
```

When `query_text` is provided (hybrid mode):
1. Run vector search (existing logic, top `limit * 3` to over-fetch for fusion)
2. Run BM25 search (`search_bm25`, top `limit * 3`)
3. Fuse via RRF with `k=60`
4. Return top `limit` results

When `query_text` is None: existing vector-only behavior (fully backward compatible).

### Reciprocal Rank Fusion

Module-level function in `embedding.py` (pure function, no dependencies):

```python
def reciprocal_rank_fusion(
    *result_lists: list[dict],
    key: str = "id",
    k: int = 60,
    limit: int = 10,
) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            item_key = str(item[key])
            scores[item_key] = scores.get(item_key, 0.0) + 1.0 / (k + rank + 1)
            items[item_key] = item
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [items[item_key] for item_key, _ in ranked]
```

`k=60` is the standard constant used in the literature. It dampens rank differences so that being rank 1 vs rank 5 matters more than being rank 50 vs rank 55.

**Fused result `similarity` field:** After RRF fusion, the `similarity` field in each result dict is **replaced with the RRF score**. This is a normalized score where higher = more relevant across both retrieval methods. The raw cosine similarity and ts_rank scores are discarded. This avoids the problem of BM25-only results having ts_rank values in a field consumers expect to be cosine similarity. The RRF score is always in a comparable range regardless of which search found the item.

### Caller changes

#### `/api/search` endpoint

Pass raw query text alongside embedding:

```python
rows = await embedding_store.search(
    query_embedding=embedding,
    limit=limit,
    source_type=source_type,
    tags=tags,
    query_text=q,
)
```

No API contract change. `SearchResult` model unchanged.

#### `RAGRetriever.retrieve()`

Pass question text alongside embedding:

```python
content_results = await self._embedding_store.search(
    query_embedding=embedding,
    limit=max_sources,
    query_text=question,
)
```

### What does NOT change

- `insert_chunks()` — trigger handles tsvector population automatically
- `search_entities()` / `search_predicates()` — embedding-only, not text search
- Knowledge graph queries — SPARQL, not text search
- No new API endpoints or response models
- No new infrastructure or extensions

---

## Constraints

- No `to_tsquery` (requires boolean syntax) — use `plainto_tsquery` for natural language safety
- No separate search endpoint for BM25-only — always fused with vector
- RRF constant `k=60` is not configurable (standard value, tuning is premature)
- No reranking step after fusion (future: Phase 7 could add this)

## Tests

- Test `search_bm25()` returns results matching keyword query
- Test `search_bm25()` returns empty for non-matching query
- Test `search()` with `query_text` produces hybrid results (items found by BM25 that vector missed)
- Test RRF fusion: item found by both searches ranks higher than item found by one
- Test RRF deduplication: same chunk from both searches appears once
- Test backward compatibility: `search()` without `query_text` behaves identically to before
- Test trigger: inserting a chunk auto-populates `tsv` column
- Test `plainto_tsquery` handles edge cases: empty string, special characters, very long queries
- Test empty query guard: stop-word-only query ("the") returns empty BM25 results, fusion degrades to vector-only
- Test fused results have RRF score as `similarity` (not raw cosine or ts_rank)
- Note: trigger test requires real PostgreSQL — verify migration SQL correctness by inspection, not unit test (project uses mocked DB in tests)
