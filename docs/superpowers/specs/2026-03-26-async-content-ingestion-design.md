# Async Content Ingestion Design

## Problem

`POST /api/content` processes everything synchronously: chunking, embedding, LLM extraction, entity resolution, and triple processing. For large documents (e.g. 1.2MB producing 337 chunks), this blocks the HTTP response for tens of minutes. The LLM extraction phase generates hundreds of concurrent requests to Ollama (which processes sequentially on a single GPU), causing massive queue buildup where individual requests take 40+ minutes due to wait time.

## Solution

Split content ingestion into a fast synchronous acceptance phase and a background processing phase. Cap chunk count at 50. Batch embeddings in groups of 20. Track progress in a new `ingestion_jobs` table with per-chunk granularity. Add an admin UI to monitor jobs.

## API Contract Changes

### `POST /api/content` (modified)

Request body: `ContentRequest` (unchanged).

Response changes from `200` to `202 Accepted`:

```json
{
  "content_id": "uuid",
  "job_id": "uuid",
  "status": "accepted",
  "chunks_total": 12,
  "chunks_capped_from": null
}
```

`chunks_capped_from` is non-null only when the text was truncated (e.g. `337` if 337 chunks were produced but capped to 50).

**Synchronous phase** (what happens before the 202 is returned):
1. Validate and parse `ContentRequest`
2. Upsert `content_metadata` row
3. Chunk the text (CPU-only, fast)
4. If chunks > 50, cap to first 50 and log a warning
5. Check for active job on this `content_id` — if one exists, return 409 Conflict
6. Create `ingestion_jobs` row with `status=accepted`, `chunks_total=N`
7. Kick off background task
8. Return 202

**Batch mode** (list input): Returns a list of 202 responses. Each item gets its own job. If one item fails validation, only that item returns an error — others proceed (partial success). Items are processed sequentially in the synchronous phase.

### `GET /api/content/{content_id}/status` (new)

Returns current ingestion status:

```json
{
  "content_id": "uuid",
  "job_id": "uuid",
  "status": "accepted|embedding|extracting|processing|completed|failed",
  "chunks_total": 12,
  "chunks_embedded": 12,
  "chunks_extracted": 8,
  "chunks_failed": 1,
  "triples_created": 24,
  "entities_resolved": 5,
  "error": null,
  "created_at": "2026-03-26T00:00:00Z",
  "updated_at": "2026-03-26T00:01:30Z"
}
```

### `GET /api/admin/jobs` (new)

Returns ingestion jobs in descending `created_at` order. Supports `?limit=` and `?status=` query params. Protected by admin auth.

## Database

### New migration: `008_ingestion_jobs.sql`

```sql
CREATE TABLE ingestion_jobs (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id        UUID        NOT NULL REFERENCES content_metadata(id) ON DELETE CASCADE,
    status            TEXT        NOT NULL DEFAULT 'accepted',
    chunks_total      INTEGER     NOT NULL DEFAULT 0,
    chunks_embedded   INTEGER     NOT NULL DEFAULT 0,
    chunks_extracted  INTEGER     NOT NULL DEFAULT 0,
    chunks_failed     INTEGER     NOT NULL DEFAULT 0,
    triples_created   INTEGER     NOT NULL DEFAULT 0,
    entities_resolved INTEGER     NOT NULL DEFAULT 0,
    chunks_capped_from INTEGER,
    error             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_ingestion_jobs_content ON ingestion_jobs (content_id);
CREATE INDEX idx_ingestion_jobs_status ON ingestion_jobs (status);
CREATE INDEX idx_ingestion_jobs_created ON ingestion_jobs (created_at DESC);

-- Only one active (non-terminal) job per content_id at a time
CREATE UNIQUE INDEX idx_ingestion_jobs_active ON ingestion_jobs (content_id)
    WHERE status NOT IN ('completed', 'failed');

-- Auto-update updated_at on every row change
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$ BEGIN NEW.updated_at = now(); RETURN NEW; END; $$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ingestion_jobs_updated
    BEFORE UPDATE ON ingestion_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
```

Status transitions: `accepted → embedding → extracting → processing → completed` (or `→ failed` from any state).

The partial unique index on `content_id` prevents concurrent jobs for the same content. The `updated_at` trigger removes the need to manually set it in every UPDATE.

## Background Worker

Runs as a FastAPI `BackgroundTask`. No external queue infrastructure needed.

### Processing pipeline:

1. **Set status = `embedding`**
2. Embed chunks in sub-batches of 20
   - After each sub-batch completes, `UPDATE ingestion_jobs SET chunks_embedded = $1, updated_at = now()`
3. Insert all embedded chunks to `content` table
4. Null out old provenance chunk_ids, delete old chunks (re-ingestion support)
5. **Set status = `extracting`**
6. Extract knowledge per chunk **sequentially** (concurrency = 1, no `asyncio.gather`)
   - After each chunk, update `chunks_extracted` or `chunks_failed`
7. Deduplicate extracted items
8. **Set status = `processing`**
9. Resolve entity labels (EntityResolver)
10. Apply URI fallback
11. Expand to triples and process — increment `triples_created` per triple
12. Log ingestion event
13. **Set status = `completed`**

On any unhandled exception: set `status = failed`, `error` = JSON with `{"type": "...", "message": "...", "phase": "..."}` for actionable diagnostics in the admin UI.

### Startup recovery

In `lifespan()`, after migrations run, mark any orphaned jobs (status not in `completed`, `failed`) as `failed` with `error = "interrupted by service restart"`. This handles jobs lost when the service restarts mid-processing.

### Embedding sub-batching

Modify existing `EmbeddingClient.embed_batch` to accept an optional `batch_size` parameter (default `None` = send all at once, current behavior). When `batch_size` is set:
- Splits texts into groups of `batch_size`
- Calls `_request` for each group sequentially
- Returns concatenated results

The background worker calls `embed_batch(texts, batch_size=20)`. Existing callers are unaffected.

The EmbeddingClient read timeout stays at 30s (adequate for 20 chunks of nomic-embed-text).

### Chunk cap

New constant `_MAX_CHUNKS = 50` in `content.py`. After chunking, if `len(chunk_records) > _MAX_CHUNKS`, truncate to first 50 `chunk_records` (post-chunking cap, not raw text truncation — more precise). Log a warning with the original count. The original count is stored in `ingestion_jobs.chunks_capped_from` and returned in the 202 response.

### Extraction concurrency

Change from `asyncio.gather` with semaphore to a simple sequential `for` loop. One chunk at a time = one LLM call at a time on Ollama. This means ~30s per extraction phase × 2 phases × 50 chunks = ~50 minutes worst case, but it runs in the background so it doesn't block anything.

## Admin UI

### New page: `/admin/jobs`

- Added to sidebar nav in `base.html`
- Table with columns: Status (color badge), Content URL (truncated, linked), Chunks progress (embedded/total, extracted/total), Triples, Created, Updated
- Status badges: blue=`embedding`, yellow=`extracting`, purple=`processing`, green=`completed`, red=`failed`, gray=`accepted`
- Auto-refresh via htmx polling every 5s on rows with in-progress status
- Route in `admin/routes.py`, template `admin/templates/jobs.html`
- Data from `GET /api/admin/jobs`

### API endpoint: `GET /api/admin/jobs`

New file `admin/jobs.py` (follows pattern of `admin/stats.py` and `admin/communities.py`):
- Queries `ingestion_jobs` joined with `content_metadata` for URL/title
- Returns JSON array, descending by `created_at`
- Supports `?limit=50` (default 50) and `?status=` filter

## Response Model Changes

### New models in `models.py`:

```python
class ContentAcceptedResponse(BaseModel):
    content_id: str
    job_id: str
    status: str = "accepted"
    chunks_total: int
    chunks_capped_from: int | None = None

class IngestionJobStatus(BaseModel):
    content_id: str
    job_id: str
    status: str
    chunks_total: int
    chunks_embedded: int
    chunks_extracted: int
    chunks_failed: int
    triples_created: int
    entities_resolved: int
    error: str | None
    created_at: str
    updated_at: str
```

`ContentResponse` can be removed — it is no longer returned from the API and nothing consumes it internally.

## Files Changed

| File | Change |
|------|--------|
| `migrations/008_ingestion_jobs.sql` | New: ingestion_jobs table |
| `src/knowledge_service/models.py` | Add `ContentAcceptedResponse`, `IngestionJobStatus` |
| `src/knowledge_service/api/content.py` | Refactor to async: fast 202 + background worker, chunk cap, sequential extraction |
| `src/knowledge_service/clients/llm.py` | Add `batch_size` param to `embed_batch()` |
| `src/knowledge_service/admin/routes.py` | Add `/admin/jobs` route |
| `src/knowledge_service/admin/jobs.py` | New: `GET /api/admin/jobs` endpoint |
| `src/knowledge_service/admin/templates/jobs.html` | New: jobs list template |
| `src/knowledge_service/admin/templates/base.html` | Add "Jobs" nav item |
| `src/knowledge_service/main.py` | Register jobs router, add startup recovery for orphaned jobs |
| `tests/` | Update existing content tests for 202 response, add job status tests |

## Testing

- Existing content ingestion tests updated to expect 202 + `ContentAcceptedResponse` shape
- Background worker tested by calling it directly (not via BackgroundTask) so assertions can be synchronous
- Job status endpoint tested with mock DB rows
- Chunk cap tested: verify >50 chunks get truncated
- Embed sub-batching tested: verify correct splitting and reassembly
- Admin jobs endpoint tested: ordering, filtering, join with content_metadata

## Not In Scope

- Retry logic for failed jobs (triage pipeline will resend)
- WebSocket/SSE for live progress (htmx polling is sufficient)
- Job cancellation endpoint (can add later if needed)
- Storing contradiction results on the job row (available via existing `/api/contradictions` endpoint after processing)
