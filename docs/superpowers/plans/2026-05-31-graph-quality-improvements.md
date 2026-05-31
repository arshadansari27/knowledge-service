# Graph Quality Improvements Implementation Plan

> **For agentic workers:** execute task-by-task. Steps use `- [ ]` checkboxes.

**Goal:** Ship four flag-gated triple-denoising levers (A relevance floor, B verbalize+source, C novelty filter, D drop predicate-noise) plus a two-call graph-as-verifier (`answer_mode=verify`), then A/B them on the eval and collapse winners into defaults.

**Architecture:** Denoising lives in `stores/rag.py` (+ rendering in `clients/rag.py`); the verifier is a new answer strategy in `clients/rag.py` dispatched from `api/ask.py` + the eval runner. All new flags read from `settings` so a per-run env var controls each A/B; defaults = current behavior so a deploy is a no-op until the eval picks winners.

**Tech stack:** Python 3.12, pyoxigraph, pgvector/asyncpg, pytest (mocked), ruff.

**Spec:** `docs/superpowers/specs/2026-05-31-graph-quality-improvements-design.md`

**Simplification taken:** lever B/C use the triple's `source_url` (already on the provenance row) as both the novelty key and the prompt source label — one `get_by_triples` call, no title lookup.

---

## Task 1: Config flags

**Files:** Modify `src/knowledge_service/config.py`; Test `tests/test_config_graph_quality.py`

- [ ] Add 5 settings after `rag_default_retrieval_mode`:
  - `rag_triple_relevance_floor: float = 0.0`
  - `rag_verbalize_triples: bool = True`
  - `rag_triple_novelty_filter: bool = False`
  - `rag_predicate_lookup_enabled: bool = True`
  - `rag_default_answer_mode: str = "direct"`
- [ ] Test asserts the five defaults load and that env overrides parse (e.g. `RAG_TRIPLE_RELEVANCE_FLOOR=0.4`).
- [ ] `grep -c` each new field name in config.py == 1 (guard against the silent-Edit-drop gotcha).

## Task 2: Lever A — relevance floor

**Files:** Modify `stores/rag.py` (`_rank_triples_by_relevance`); Test `tests/test_triple_relevance_floor.py`

- [ ] After cosine scoring, keep only triples with `score >= settings.rag_triple_relevance_floor`, then take top-`limit`. If none clear it, return `[]`.
- [ ] Floor is applied ONLY on the relevance path — the `_rank_and_cap_triples` confidence fallback is unchanged (documented in docstring).
- [ ] Tests: drops sub-floor triples; returns `[]` when none clear floor; floor=0.0 == current behavior; fallback path ignores floor.

## Task 3: Lever D — gate predicate lookup

**Files:** Modify `stores/rag.py` (`_lookup_triples_by_predicate`); Test `tests/test_predicate_lookup_flag.py`

- [ ] Early-return `[]` at the top of `_lookup_triples_by_predicate` when `not settings.rag_predicate_lookup_enabled`.
- [ ] Tests: flag off → method returns `[]` without touching the store (mock asserts no call); flag on → existing behavior.

## Task 4: Levers B + C — provenance enrichment, verbalize, novelty

**Files:** Modify `stores/rag.py` (`__init__`, new `_finalize_triples`, call it in all 3 strategies); Test `tests/test_triple_finalize.py`

- [ ] `__init__` gains `provenance_store=None` → `self._provenance_store`.
- [ ] New `async def _finalize_triples(self, triples, content_results)`:
  - If `triples` and (`settings.rag_triple_novelty_filter` or `settings.rag_verbalize_triples`) and `self._provenance_store`: one `get_by_triples([compute_triple_hash(s,p,o) ...])` call (wrapped in try/except → `{}` on failure, logged).
  - Novelty (if flag): drop triples whose first provenance `source_url` is in `{r["url"] for r in content_results if r.get("url")}`. Triples with no provenance/source_url are kept.
  - Verbalize (if flag): attach `subject_label/predicate_label/object_label` via `_localize` (no DB needed) and `source_url` (first provenance row, else absent).
  - Return the surviving/enriched list.
- [ ] Call `capped = await self._finalize_triples(capped, content_results)` in `_retrieve_semantic`, `_retrieve_entity`, `_retrieve_graph`, BEFORE `_detect_contradictions(capped)` and the `RetrievalContext`.
- [ ] Tests (mock provenance_store): novelty drops same-`source_url` triple, keeps cross-doc, keeps no-provenance; verbalize attaches labels + source_url; both flags off → triples unchanged & no provenance call; provenance exception → triples unchanged, no crash; `provenance_store=None` → no-op.

## Task 5: Verbalized rendering + include_graph in prompt builder

**Files:** Modify `clients/rag.py` (`build_rag_prompt`); Test `tests/test_rag_prompt_render.py`

- [ ] `build_rag_prompt(question, context, include_graph=True)`. When `include_graph` is False, omit the Knowledge-Graph-Facts and Contradictions sections.
- [ ] Triple line: if `t.get("subject_label")` present → `- {subject_label} {predicate_label} {object_label} (confidence: {conf})` and append ` · source: {source_url}` when `source_url` present. Else → existing legacy `[trust] s -> p -> o (ktype, confidence: c)` line.
- [ ] Tests: labelled triple → prose (no raw URI, no `->`); unlabelled → legacy; `include_graph=False` → neither section present; existing callers (default True) unaffected.

## Task 6: Verifier — answer_verified + answer_auto

**Files:** Modify `clients/rag.py` (`_complete`, `answer`, `answer_verified`, `answer_auto`, `build_verify_prompt`); Test `tests/test_rag_verifier.py`

- [ ] Extract the POST+`_extract_json` body into `async def _complete(self, prompt) -> str` returning the answer string; `answer()` uses it with `build_rag_prompt(..., include_graph=True)`.
- [ ] `build_verify_prompt(question, draft_answer, context)`: instruction ("verify each claim in the draft against the facts below; correct anything they contradict; flag unsupported claims; return JSON {\"answer\": ...}"), the draft, then graph facts + contradictions (reuse the rendering).
- [ ] `async def answer_verified(question, context)`: draft = `_complete(build_rag_prompt(..., include_graph=False))`; then try `final = _complete(build_verify_prompt(...))` → `RAGAnswer(final)`; on exception log + return `RAGAnswer(draft)`.
- [ ] `async def answer_auto(question, context, answer_mode)`: if `answer_mode == "verify"` and (`context.knowledge_triples` or `context.contradictions`) → `answer_verified`; else `answer`.
- [ ] Tests (mock `_complete`/httpx): verify fires two completions when triples present; degrades to one when context empty; call-2 raises → returns draft; `answer_auto` dispatch matrix.

## Task 7: Wire verifier into /api/ask

**Files:** Modify `api/ask.py`; Test `tests/test_ask_answer_mode.py`

- [ ] `AskRequest`: add `answer_mode: Literal["direct", "verify"] | None = None`.
- [ ] `AskResponse`: add `answer_mode: str | None = None` (observability).
- [ ] In `post_ask`: `answer_mode = body.answer_mode or settings.rag_default_answer_mode`; replace `raw_answer = await rag_client.answer(...)` with `raw_answer = await rag_client.answer_auto(body.question, context, answer_mode)`; set `answer_mode=answer_mode` on the response.
- [ ] Tests: explicit `verify` calls `answer_auto` with verify; default uses `settings.rag_default_answer_mode`; response echoes the mode.

## Task 8: Wire provenance into retriever construction

**Files:** Modify `src/knowledge_service/main.py`; Modify `eval/runner.py` (`_build_components`)

- [ ] main.py: add `provenance_store=stores.provenance,` to the `RAGRetriever(...)` kwargs.
- [ ] eval `_build_components`: `from knowledge_service.stores.provenance import ProvenanceStore`; add `provenance_store=ProvenanceStore(pool)` and `max_triples=settings.rag_max_triples` to the retriever (so eval matches prod wiring).
- [ ] `grep -c "provenance_store=" main.py` == 1.

## Task 9: Eval runner — answer_mode

**Files:** Modify `eval/runner.py`; Test `tests/eval/test_runner_answer_mode.py`

- [ ] `run_eval(modes, k, golden_path, answer_mode="direct")`; in `_one`, replace `rag_client.answer(...)` with `rag_client.answer_auto(item.question, context, answer_mode)`.
- [ ] CLI: `--answer-mode` default `"direct"`, threaded through `_amain` → `run_eval`.
- [ ] Test: `run_eval` passes `answer_mode` to `answer_auto` (mock components).

## Task 10: Full verification

- [ ] `uv run ruff check . && uv run ruff format --check .`
- [ ] `uv run pytest tests/ -q` — all green; fix any fallout from signature/behaviour changes.
- [ ] `grep` audit: every new flag referenced ≥1 place; no `answer` direct-call left in ask.py/runner that should be `answer_auto`; no orphaned helper.
- [ ] Commit.

## Task 11: Eval campaign (corpus required)

- [ ] Re-export prod snapshot (`scripts/export_prod_snapshot.py` via swarm) → restore local pgvector (port 5434) + oxigraph dir.
- [ ] Regenerate golden candidates (`scripts/gen_golden_candidates.py`) and auto-curate to ~70-80 items (semantic/entity/graph); save to a scratch golden.json (NOT committed).
- [ ] Runs (SUT gpt-oss:20b, judge claude-haiku, k=5): baseline (defaults) full vs chunks_only; then flip each lever; then best-of denoising; then `--answer-mode verify` vs direct. NEVER redeploy litellm mid-run.
- [ ] Decide winners under do-no-harm (faithfulness/correctness). Set winning defaults in config.py. If D wins, delete `_lookup_triples_by_predicate` + its flag + call sites.
- [ ] Write Update 3 in `docs/kg-vs-rag-eval-findings.md`. Commit.

## Task 12: Merge + deploy + verify

- [ ] Final `ruff` + `pytest` green; tests for any collapsed dead code removed.
- [ ] Merge worktree branch → `main`, push (CI auto-bumps version).
- [ ] Deploy: `docker --context swarm-baa service update --image arshadansari27/knowledge-service:latest --force aegis_knowledge` (after CI builds the image).
- [ ] Verify live: `/health` ok; `/api/ask` with entity question (intent routing + verbalized triples), semantic question (floor → few/zero triples), and `answer_mode=verify`.
- [ ] Save cmemory lesson + update auto-memory.
