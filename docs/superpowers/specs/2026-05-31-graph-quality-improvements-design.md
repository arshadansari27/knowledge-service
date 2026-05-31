# Graph Quality Improvements — Triple Denoising + Graph-as-Verifier — Design Spec

**Date:** 2026-05-31
**Status:** Approved (design) — proceeding to implementation plan.
**Branch:** `graph-quality-verifier` (worktree off `main` @ v0.1.118)

## Goal

Push the knowledge-graph answer path from *break-even* (where the
2026-05-31 relevance-ranking + intent-routing work left it) to a **measured
win** on faithfulness and correctness, via four surgical retrieval-denoising
levers plus a two-call "graph-as-verifier" answer strategy. Every change is
independently togglable by a config flag and independently A/B'd on the frozen
79-question golden set, then the winners are collapsed into defaults and the
losing flags + dead code removed.

## Background — why these levers

Reading `stores/rag.py` + `clients/rag.py` against the residual eval wounds
(semantic correctness −0.117, persistent faithfulness −0.036, graph-intent
recall −0.083 — all `full` − `chunks_only`) points at four concrete
mechanisms:

1. **No relevance *floor*, only a *cap*.** `_rank_triples_by_relevance`
   (`stores/rag.py`) keeps the top-15 triples by cosine *unconditionally*. A
   semantic question with no relevant triples still gets 15 marginally-related
   facts injected — the semantic-question harm.
2. **The prompt dumps raw URIs and decontextualized assertions.**
   `clients/rag.py` `build_rag_prompt` emits
   `- [extracted] http://knowledge.local/data/cold_exposure -> increases -> dopamine (Fact, confidence: 0.7)`.
   `_localize()` exists but is used only for *embedding*, never for *display*;
   each triple is an unverifiable claim with no link to its source sentence.
   This is the faithfulness leak: the judge sees an answer claim "supported"
   only by a bare extracted triple.
3. **`_lookup_triples_by_predicate` is a relevance-blind noise pump.** It pulls
   up to 10 triples that merely share a *predicate* with the query, regardless
   of subject relevance, diluting the candidate pool.
4. **Contradictions are inert.** `build_rag_prompt` lists a "Contradictions
   Found" block that never re-ranks, filters, or otherwise shapes the answer.

The verifier (E) addresses (4) and the faithfulness leak structurally: it
stops using the graph as *context to stuff* and uses it as a *fact-checker*.

## Architecture

Two workstreams sharing the existing `src/knowledge_service/eval/` harness:

1. **Retrieval denoising (levers A/B/C/D)** — all in `stores/rag.py` and
   `clients/rag.py`, gated by config flags. Decides *which* triples reach the
   prompt and *how* they are rendered.
2. **Graph-as-verifier (lever E)** — a new answer-generation strategy in the
   answer layer (`clients/rag.py` + `api/ask.py`), selected by a new
   `answer_mode` request parameter. Orthogonal to `retrieval_mode`: one decides
   *what* to retrieve, the other *how* to answer.

These are deliberately decoupled — `answer_mode` works with any
`retrieval_mode`, and each denoising lever is independent of the others.

## Component 1 — Denoising levers

All four default to **current behavior** so a fresh deploy is a no-op until the
eval picks winners.

### A. Relevance floor

- `_rank_triples_by_relevance(triples, query_embedding, limit)` gains an
  internal `min_score` read from `settings.rag_triple_relevance_floor`.
- After scoring by cosine, drop every triple whose score `< min_score`. If none
  clear the floor, return `[]` (inject *zero* triples rather than the
  least-irrelevant N).
- The confidence-ranking **fallback** path (`_rank_and_cap_triples`, used when
  the embedding backend is down or returns a wrong-length batch) does **not**
  apply the floor — there is no comparable relevance score there, and silently
  dropping all triples on an embedding outage would be a worse failure than
  keeping the confidence top-N. Documented in the docstring.
- Config: `rag_triple_relevance_floor: float = 0.0` (0.0 = disabled = current
  behavior). Eval sweeps {0.0, 0.3, 0.4, 0.5}.

### B. Localize + verbalize + source in the prompt

- The **retriever** enriches each capped triple dict (before it lands in
  `RetrievalContext.knowledge_triples`) with `subject_label`,
  `predicate_label`, `object_label` (via the existing `_localize`) and
  `source_title`. The prompt builder stays dumb — it only formats.
- `build_rag_prompt` renders verbalized prose when the labels are present:
  `- cold exposure increases dopamine (confidence 0.70 · source: "<title>")`.
  When labels are absent (flag off), it keeps the legacy
  `[trust] s -> p -> o (ktype, confidence: c)` format.
- Config: `rag_verbalize_triples: bool = True`. Gated for clean A/B even though
  localization is expected to be a strict improvement.

### C. Novelty filter

- After `content_results` and the capped triples are known, drop any triple
  whose source content is already present in `content_results` — it restates
  prose the LLM already has. Keep triples from *other* documents (the graph's
  unique cross-document contribution).
- A triple with **no resolvable provenance / content_id is kept** (cannot prove
  redundancy).
- Config: `rag_triple_novelty_filter: bool = False` (default off = current).

### Shared provenance enrichment (B + C)

B (needs `source_title`) and C (needs `source_content_id`) share **one batched
provenance lookup** per request: `provenance_store.get_by_triples(hashes)` over
the ≤15 capped triples, mapping each triple to its source title + content_id.

- Runs only when B or C is enabled (skip the DB round-trip otherwise).
- The retriever needs access to `ProvenanceStore` + `ContentStore` (for title).
  It already receives `embedding_store` (= `ContentStore`); provenance is
  passed in at construction (`main.py` wiring) — `None`-safe so unit tests and
  the eval harness that build a retriever without provenance still work.
- **Failure handling:** any enrichment exception is logged and skipped — triples
  fall back to localized labels with no `source_title`, and the novelty filter
  becomes a no-op for that request. Retrieval never crashes on provenance.

### D. Drop predicate-similarity lookup

- `rag_predicate_lookup_enabled: bool = True`. When `False`,
  `_lookup_triples_by_predicate` is not called (the semantic and entity
  strategies skip it entirely).
- If the eval shows off ≥ on (neutral-or-better), the **collapse step deletes**
  `_lookup_triples_by_predicate`, its call sites, and the flag.

## Component 2 — Graph-as-verifier

- New `answer_mode: Literal["direct", "verify"] | None = None` on `AskRequest`.
  `None` → server default `settings.rag_default_answer_mode` (`"direct"`).
- `RAGClient.answer_verified(question, context)`:
  - **Call 1 (answer):** answer from `content_results` **only** — a prompt with
    the chunk context but no triples/contradictions. Implemented by a
    `include_graph: bool` parameter on `build_rag_prompt` (when `False`, the
    Knowledge-Graph-Facts and Contradictions sections are omitted).
  - **Call 2 (verify):** a prompt containing the Call-1 draft answer + the graph
    facts + contradictions, instructing the model to verify each claim against
    the facts, correct any claim a fact contradicts, flag unsupported claims,
    and return the revised answer. The revised answer is the response.
- **Empty-context degradation (confirmed judgment call):** when `answer_mode`
  resolves to `verify` but retrieval produced **zero triples and zero
  contradictions**, `post_ask` makes a single direct call instead of the
  two-call sequence. A verify pass with nothing to check against can only make
  the model second-guess a grounded answer; this skips a provably empty LLM
  call. It does **not** gate retrieval (the declined option) — verify always
  fires whenever the graph has something to say.
- **Call-2 failure handling:** if the verify call raises, fall back to the
  Call-1 draft answer and log a warning — do not surface a 502 because
  *verification* failed when a valid draft already exists.
- Config: `rag_default_answer_mode: str = "direct"`.

### `post_ask` control flow (`api/ask.py`)

1. Resolve `retrieval_mode` (existing `auto` intent-routing — unchanged).
2. Resolve `answer_mode` (explicit request value wins, else
   `settings.rag_default_answer_mode`).
3. `context = await retriever.retrieve(...)` (unchanged signature).
4. If `answer_mode == "verify"` **and** (`context.knowledge_triples` or
   `context.contradictions`): `raw = await rag_client.answer_verified(...)`.
   Else: `raw = await rag_client.answer(...)` (existing direct path).
5. Response assembly (sources, confidence, evidence, contradictions) unchanged.

## Component 3 — Eval campaign

- **Frozen 79-Q golden set**, reused as-is for clean attributable deltas vs the
  prior runs (graph-intent stays n=12; comparability prioritized over noise
  reduction this round).
- **Baseline** = current `main` (v0.1.118) behavior (all new flags at default,
  `answer_mode=direct`).
- **Runs:** each lever flipped *independently* from baseline in `full` mode →
  measure delta; then one combined "best-of" run; then `answer_mode=verify` vs
  `direct` (best-of denoising held fixed).
- **Harness extension:** `eval/runner.py` (+ `__main__.py`) gains an
  `--answer-mode` option threaded into `score_query`; the denoising flags are
  read from `settings`, so each run sets them via env. No golden-set change.
- **Decision rule (unchanged):** do-no-harm on aggregate faithfulness /
  correctness. Keep a lever only if it improves or holds; otherwise revert it.
- **Collapse:** set winning defaults, delete losing flags + dead code, write
  **Update 3** in `docs/kg-vs-rag-eval-findings.md`.

## Config additions (`config.py`)

```python
rag_triple_relevance_floor: float = 0.0      # A; env RAG_TRIPLE_RELEVANCE_FLOOR
rag_verbalize_triples: bool = True           # B; env RAG_VERBALIZE_TRIPLES
rag_triple_novelty_filter: bool = False      # C; env RAG_TRIPLE_NOVELTY_FILTER
rag_predicate_lookup_enabled: bool = True    # D; env RAG_PREDICATE_LOOKUP_ENABLED
rag_default_answer_mode: str = "direct"      # E; env RAG_DEFAULT_ANSWER_MODE
```

## Files touched

| File | Change |
|---|---|
| `src/knowledge_service/config.py` | 5 new settings (above) |
| `src/knowledge_service/stores/rag.py` | relevance floor; predicate-lookup gate; provenance enrichment (labels + source_title + content_id); novelty filter; constructor takes optional `provenance_store` |
| `src/knowledge_service/clients/rag.py` | verbalized triple rendering; `build_rag_prompt(..., include_graph=True)`; `RAGClient.answer_verified()` |
| `src/knowledge_service/api/ask.py` | `answer_mode` field + resolution + dispatch + empty-context degradation |
| `src/knowledge_service/main.py` | pass `provenance_store` into `RAGRetriever` |
| `src/knowledge_service/eval/runner.py`, `eval/__main__.py` | `--answer-mode` threaded into `score_query` |
| `docs/kg-vs-rag-eval-findings.md` | Update 3 (after campaign) |
| `tests/...` | unit tests per lever + verifier (all mocked) |

## Testing

All mocked, CI-safe (no PostgreSQL / Ollama / network):

- **A:** floor drops sub-threshold triples; returns `[]` when none clear it;
  floor not applied on the confidence fallback path.
- **B:** prompt renders localized labels + source when present; legacy
  SPO-with-URI format when labels absent (flag off).
- **C:** drops triples whose content_id ∈ retrieved set; keeps cross-document
  triples; keeps triples with no provenance.
- **D:** predicate lookup skipped when flag off; called when on.
- **Provenance enrichment:** failure logs + degrades (no crash, novelty becomes
  no-op).
- **E:** two-call sequence fires when triples/contradictions present (mock both
  calls); degrades to single direct call on empty context; Call-2 failure falls
  back to Call-1 draft.
- **Wiring:** `RAGRetriever` with `provenance_store=None` behaves as today.

## Out of scope (this round)

- Golden-set expansion / more graph-intent questions (deferred — reuse as-is).
- Lever F (traversal depth tuning / path presentation) and Tier-3 extraction
  precision (materiality-weighted predicates). Tracked as next levers in the
  findings doc; not part of this plan.
- Re-ingestion of the corpus (no extraction-side changes here).
```
