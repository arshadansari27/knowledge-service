# KS Eval Harness — design

**Date:** 2026-05-30
**Status:** Approved (brainstorming) — pending spec review before planning
**Author:** Arshad + Claude

## Problem

knowledge-service is actively used by aegis as a GTD reference tool: aegis writes a
corpus into it (`reference`, `intelligence`, `clarification`, `calendar`, `chat`
content) and reads it back via `/api/search`, `/api/ask`, and
`/api/knowledge/query`. But there is **no way to measure whether retrieval or
answers are any good**. Tests cover plumbing, not quality. As a result we cannot
answer the question that gates all further investment: *does the knowledge-graph
layer earn its keep over plain hybrid (vector + BM25) RAG, or is it decorative?*

Today the KG layer is largely cosmetic in the read path: contradictions and
trust-tiers are surfaced but never affect ranking or the answer; chunks dominate
`/api/ask`. We need an evaluation harness to make quality observable and to settle
the graph-on/graph-off question with data.

## Goals

1. A repeatable, layered eval harness that scores **retrieval quality** and
   **answer quality** over a fixed, realistic corpus and golden query set.
2. A first-class **`retrieval_mode`** toggle (`full` vs `chunks_only`) so the same
   golden set can be run graph-on and graph-off and compared.
3. A scored report broken down by query type, from which the keep/cut decision
   reads directly.

## Non-goals (YAGNI)

- No dashboard / web UI.
- No CI integration of the full eval run (it is corpus- and LLM-dependent; it runs
  locally like the existing `tests/e2e/`). Only the harness's own unit tests run in
  CI.
- No automatic regeneration of the golden set on each run (it is curated and
  checked in).
- No HTTP/black-box arm. The harness drives `RAGRetriever`/`RAGClient` in-process.
  The FastAPI layer over them is thin; if contract drift ever bites we add a small
  HTTP smoke layer later.
- No aegis-side changes. Wiring the aegis GTD→KS read-loop is a separate,
  deferred plan.

## Decisions (locked during brainstorming)

| Question | Decision |
|---|---|
| Eval target | Layered: retrieval metrics + answer quality (+ GTD-surfacing as a query-type flavor of retrieval) |
| Corpus scope | Everything ingested (reference + intelligence + clarification + calendar + chat) |
| Work scope now | KS-only. aegis read-loop is a separate follow-up plan. |
| Corpus source | Export a prod snapshot |
| Golden set | Hybrid: ~40–60 auto-then-curated + ~10–15 hand-authored GTD-style |
| Keep/cut bar | **Do no harm** — keep the KG layer unless it measurably worsens answers; note where it clearly helps for routing |
| Harness architecture | In-process harness + real `retrieval_mode` flag (Approach 1) |
| Judge | Claude (so qwen3, the system-under-test, does not grade itself) |

## Architecture

New package `src/knowledge_service/eval/`:

```
eval/
  __init__.py
  runner.py      # orchestrates: load golden -> run each mode -> score -> report
  metrics.py     # recall@k, precision@k, MRR, nDCG (pure functions)
  judge.py       # Claude-as-judge: faithfulness + correctness
  golden.json    # the golden query set (checked in)
  reports/       # timestamped JSON outputs (gitignored)
```

Each unit has one purpose and a clear interface:
- `metrics.py` — pure functions over `(retrieved_ids, relevant_ids)`; no I/O;
  fully unit-tested.
- `judge.py` — given `(question, reference_answer, generated_answer,
  retrieved_context)` returns `{faithfulness, correctness, rationale}` via a Claude
  client. Isolated so the judge model/prompt can change without touching the runner.
- `runner.py` — wires corpus + golden + retriever/client + metrics + judge into a
  report. Knows nothing about metric internals or judge internals.

### The `retrieval_mode` flag (only production change)

Add `retrieval_mode: str = "full"` to `RAGRetriever.retrieve()`:
- `"full"` — current behaviour (classification + entity/predicate/triple lookup +
  contradiction detection + content search).
- `"chunks_only"` — skip all graph work; return a `RetrievalContext` populated only
  with `content_results` (pure hybrid vector + BM25). No classification call.

Threaded through `/api/ask` as an optional request field defaulting to `"full"`, so
aegis and all existing callers are unaffected. This flag is the graph-on/off switch
**and** a permanent capability consistent with the "do no harm" stance (we can
route or disable the graph in prod without a redeploy).

## Corpus snapshot

A fixed, reproducible local corpus exported read-only from prod:

1. `pg_dump` the `knowledge` DB (online, safe) and restore into a local Postgres.
   Covers `content_metadata`, `content` (chunks + embeddings), `provenance`,
   `ingestion_jobs`, `entity_embeddings`, `predicate_embeddings`, `entity_aliases`,
   `triple_outbox`, `schema_migrations`.
2. Dump the prod oxigraph store to N-Quads and load into a local oxigraph data dir.
   This holds the RDF triples the `full` arm depends on.

Both halves are required: Postgres holds chunks/embeddings/provenance; oxigraph
holds triples. The snapshot lives outside git (size); a small `manifest.json`
records content counts per `source_type` and a snapshot date for reproducibility.

**Open operational point (to resolve in the plan):** cleanest oxigraph dump method.
Options: (a) `pyoxigraph.Store` opened read-only against a copy of the prod data
dir, `.dump()` to N-Quads; (b) run a dump inside the prod container; (c) briefly
quiesce `aegis_knowledge` for a consistent file copy. Preference order: (a) → (b) →
(c). pg_dump is online and needs no quiescing.

## Golden set

`eval/golden.json`. Each entry:

```json
{
  "id": "gtd-references-llm-eval-001",
  "question": "What references do I have about evaluating LLM applications?",
  "query_type": "gtd",
  "relevant_source_ids": ["<content_id-or-chunk_id>", "..."],
  "reference_answer": "...",
  "notes": "hand-authored; cross-document reference lookup"
}
```

- `query_type` ∈ `{semantic, entity, graph, gtd}` to mirror the retriever's intents
  plus the GTD reference-tool flavor.
- `relevant_source_ids` are ids present in the snapshot (content/chunk ids).

Construction (hybrid):
- **~40–60 auto-then-curated:** sample chunks across all `source_type`s; an LLM
  generates a question answerable from each sampled chunk; the source chunk(s)
  become the relevance label and seed the reference answer. Curate out leaky
  (answer-verbatim-in-question), trivial, or ambiguous items.
- **~10–15 hand-authored GTD-style:** cross-document questions ("what references do
  I have on X", "what did I decide about Y", "how does X relate to Y") written
  against known snapshot contents — the real reference-tool use case auto-gen won't
  produce well.

## Metrics & judge

**Retrieval layer** (`metrics.py`, pure, unit-tested): for each query compare the
top-k `content_results` ids to `relevant_source_ids`:
- `recall@k`, `precision@k`, `MRR`, `nDCG@k`.
- A **triple-contribution** stat: count of triples surfaced and whether any involve
  golden-relevant entities — diagnostic for whether the graph arm adds signal.

**Answer layer** (`judge.py`): Claude scores each generated answer on:
- **faithfulness** — is the answer grounded in the retrieved context; no
  fabrication.
- **correctness** — does it match the reference answer (semantically).
Returns numeric scores + a short rationale per query. SUT (qwen3) calls run at
temperature 0 for determinism. The judge is Claude via an Anthropic-compatible
client; the API key comes from an env var (never hardcoded), and the harness fails
with a clear message if it is missing.

## Run flow & output

```
uv run python -m knowledge_service.eval --modes full,chunks_only --k 5
```

- Loads `golden.json`, runs every query through each requested mode (retrieval +
  `/api/ask`-equivalent answer generation in-process), scores retrieval + answer.
- Writes `eval/reports/<timestamp>.json` (per-query detail + aggregates) and prints
  a summary table:

```
mode         query_type   recall@5  prec@5  MRR   nDCG  faithful  correct
full         semantic     ...
chunks_only  semantic     ...
full         gtd          ...
chunks_only  gtd          ...
...
```

**Reading the decision:** graph-on "does no harm" if `full ≥ chunks_only` (within
run-to-run noise) on aggregate answer quality. The per-query-type breakdown shows
where `full` clearly wins — those become candidates for intent-based routing rather
than cutting.

## Testing the harness

- Unit tests (run in CI): metric functions against synthetic
  `(retrieved, relevant)` fixtures with known recall/precision/MRR/nDCG; judge
  response parsing; `retrieval_mode="chunks_only"` dispatch returns content-only
  context and makes no classification/triple calls (mock-based, mirrors existing
  `tests/test_rag_retriever.py` style).
- The full corpus-dependent run is manual/local, like `tests/e2e/`.

## Risks & mitigations

- **qwen3 capacity / timeout cascade** (known prod issue): the eval batches many
  LLM calls. Mitigate by bounding concurrency in the runner (small semaphore) and
  temperature 0; this is an eval-local concern, not the prod extraction path.
- **Auto-gen golden leakage** (questions trivially answerable from their own text):
  mitigated by the curation pass.
- **Judge variance:** fixed judge prompt + temperature 0; report rationales so
  scores are auditable.
- **Snapshot staleness:** manifest records snapshot date + counts; re-export when
  the corpus has meaningfully grown.

## Follow-up (separate plan, not this work)

Wire the aegis GTD→KS read-loop so KS references actually surface during GTD
engage / weekly review — the end-to-end "reference tool" experience. Tracked as its
own brainstorm → spec → plan.
