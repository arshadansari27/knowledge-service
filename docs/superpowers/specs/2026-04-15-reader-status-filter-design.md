# Reader-side status filtering for in-flight / stuck content

**Status:** Design approved, implementation plan pending
**Date:** 2026-04-15

## Problem

`/api/search`, `/api/ask`, and the hybrid retriever in
`src/knowledge_service/stores/content.py` (`ContentStore.search()` at line 232,
`_search_bm25()` at line 324) join `content` chunks to `content_metadata` with
no awareness of `ingestion_jobs.status`.

Content whose ingestion has chunks written but no triples yet (in-flight), or
whose latest job is stuck in a non-terminal status, still shows up in search
results. Chunks without their corresponding KG triples give the retriever half
the picture — similarity hits on chunks whose KG evidence is missing degrade
RAG quality in a way that is hard to diagnose from the response.

### Pipeline states in play

`ingestion_jobs.status` values, from `src/knowledge_service/ingestion/worker.py`:

- **Non-terminal:** `accepted`, `embedding`, `analyzing`, `extracting`,
  `resolving`, `processing`
- **Terminal:** `completed`, `failed`

Chunks land in the `content` table at the end of `embedding` (via
`ContentStore.replace_chunks()`). Triples + provenance land at the end of
`processing` via the `triple_outbox` 2PC commit. "Chunks-ready-but-triples-
pending" therefore covers every status from `analyzing` onward until
`completed`.

## Decision: Option B — include terminal, exclude non-terminal

Readers exclude content whose latest job is in a **non-terminal** state.
Content whose latest job is `completed` or `failed` is included. Content with
no job record (legacy rows or test fixtures that bypass the worker) is also
included.

### Rationale

- `completed` is the invariant the outbox 2PC work codifies: chunks + triples +
  provenance are all durable.
- `failed` may carry partially-committed triples via the outbox. Hiding them
  entirely (Option A) removes real evidence from retrieval. Operators can
  re-ingest to promote a failed job to completed; until then the evidence that
  did commit is visible.
- Non-terminal jobs produce the "half-picture" problem the spec is trying to
  solve. These are excluded.

Option A (strict `completed`-only) was considered and rejected — recall loss
on legitimately-indexable `failed`-with-outbox-evidence content outweighs the
consistency win. Option C (per-request flag) was rejected because per-call
knobs proliferate and get misused; the rollout flag below is an
operational-level switch, not a per-query one.

## Architecture

### Where the filter lives

In SQL, not in application code. `ContentStore.search()` and `_search_bm25()`
each add a `LEFT JOIN LATERAL` against `ingestion_jobs` that selects the
latest job per content:

```sql
LEFT JOIN LATERAL (
    SELECT status
    FROM ingestion_jobs
    WHERE content_id = m.id
    ORDER BY created_at DESC
    LIMIT 1
) j ON TRUE
WHERE ...
  AND (j.status IS NULL OR j.status IN ('completed', 'failed'))
```

Two design choices baked in:

1. **LATERAL with `ORDER BY created_at DESC LIMIT 1`.** A content can have
   multiple historical jobs (completed → re-ingest → completed again). A plain
   `LEFT JOIN ingestion_jobs` would fan out to N rows per content. LATERAL
   gives us exactly one row per content, and "latest job wins" matches the
   `idx_ingestion_jobs_active` invariant.
2. **Explicit `j.status IS NULL OR j.status IN (...)`.** The `NULL` branch
   covers legacy content ingested before migration 008 existed, and test
   fixtures that insert `content_metadata` directly. `COALESCE` would express
   the same logic but hide one of the two pass-through cases from a reviewer.

A post-filter in Python was considered and rejected: it breaks `LIMIT`
semantics. We would either overfetch-then-filter (approximate) or issue chase
queries (latency-heavy).

### Existing index coverage

`idx_ingestion_jobs_content` (from migration 008) already covers the LATERAL
subquery join on `content_id`. No new index required. `ORDER BY created_at
DESC LIMIT 1` on the small per-content result set is cheap.

### Config flag

Single boolean, set at startup, plumbed into `ContentStore.__init__`:

```python
# knowledge_service/config.py (env var: KS_READER_EXCLUDE_INFLIGHT)
KS_READER_EXCLUDE_INFLIGHT: bool = True  # default on
```

Behavior:

- `True` (default): LATERAL join and status predicate are added to both
  search queries.
- `False`: LATERAL join and status predicate are omitted entirely — no perf
  cost and no behavior change from today.

Not threaded as a per-call argument. Per-call means every caller has to
remember to pass it, which in practice means it gets missed. Operational
decisions (flip the flag, bounce the service) belong in config, not in the
request shape.

`RAGRetriever` in `src/knowledge_service/stores/rag.py` inherits the behavior
transparently because it calls `ContentStore.search()` — no changes to
retriever code.

## API contract

No response-shape change on `/api/search` or `/api/ask`. The only visible
change is **recall**: mid-pipeline content temporarily will not appear in
search results until its pipeline finishes.

`docs/API.md` gets one paragraph added under each of the two endpoints:

> Results only include content whose latest ingestion job has reached a
> terminal state (`completed` or `failed`), or content with no recorded job.
> Mid-pipeline content is excluded until embedding, extraction, and
> processing finish. This behavior is controlled by the
> `KS_READER_EXCLUDE_INFLIGHT` environment variable (default `true`).

`/api/content/{id}/chunks` is explicitly untouched. It reads chunks by ID
directly via `ContentStore.get_chunks()`, which is an operator/debug path
rather than a retrieval path, and needs to see in-flight content.

## Testing

A new file `tests/test_reader_status_filter.py` covers both `search()` and
`_search_bm25()` paths:

1. **In-flight excluded:** content whose latest job is `processing` → not
   returned.
2. **Completed included:** content whose latest job is `completed` →
   returned.
3. **Failed included:** content whose latest job is `failed` → returned.
   (Captures Option B semantic explicitly.)
4. **No-job included:** content with no `ingestion_jobs` row → returned.
   (Captures legacy/fixture case.)
5. **Latest-job-wins:** content with one `completed` job followed by one
   `processing` job → excluded.
6. **Flag off bypasses filter:** with `KS_READER_EXCLUDE_INFLIGHT=False`, all
   of the above rows are returned regardless of status.

Each test runs against both `search()` (vector path) and `_search_bm25()`
(BM25 path) so the parallel code paths don't diverge. Tests use the existing
asyncpg-backed test lane where available.

One end-to-end test in `tests/e2e/` verifies the LATERAL SQL is valid under
real PostgreSQL.

## Boundaries

Explicitly out of scope:

- Any change to the write path (`EmbedPhase`, `ProcessPhase`, outbox drainer).
- Changes to `/api/content/{id}/chunks`.
- Changes to the job janitor in `main.py` lifespan.
- Any new ingestion state or status value.

## Rollout

1. Ship with `KS_READER_EXCLUDE_INFLIGHT=true` as the compile-time default.
2. Deploy. Monitor `/api/search` and `/api/ask` for recall drop reports from
   AEGIS and other consumers.
3. If any downstream consumer breaks, flip the env var to `false` on the
   production stack via `docker --context swarm-baa service update
   --env-add KS_READER_EXCLUDE_INFLIGHT=false aegis_knowledge` and
   investigate.
4. Once stable for one week, the flag can be removed in a follow-up; not
   part of this spec.
