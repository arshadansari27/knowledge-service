# KS Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an in-process evaluation harness that scores knowledge-service retrieval and answer quality over a fixed prod snapshot and golden query set, with a real `retrieval_mode` toggle so graph-on (`full`) and graph-off (`chunks_only`) can be compared head-to-head.

**Architecture:** A new `src/knowledge_service/eval/` package drives `RAGRetriever`/`RAGClient` in-process (no HTTP). Pure metric functions (`metrics.py`) score retrieval against golden relevance labels; a Claude-as-judge module (`judge.py`) scores answers for faithfulness/correctness (qwen3 stays the system-under-test). A `runner.py` CLI runs the golden set through each mode and writes a JSON report + summary table. The only production change is adding `retrieval_mode` to `RAGRetriever.retrieve()` and threading it through `/api/ask`.

**Tech Stack:** Python 3.12, FastAPI, asyncpg, pyoxigraph, httpx, pytest + pytest-asyncio (`asyncio_mode = "auto"`), pytest-httpx, uv. Spec: `docs/superpowers/specs/2026-05-30-ks-eval-harness-design.md`.

---

## Pre-flight: worktree

Per the user's worktree policy, do NOT work on `main`. Before Task 1, create an isolated worktree (via `superpowers:using-git-worktrees`). The spec file `docs/superpowers/specs/2026-05-30-ks-eval-harness-design.md` and this plan currently exist only as untracked files on `main`; copy both into the worktree (or `git add` them there) so spec, plan, and code land in the same PR. All commits below happen in the worktree.

---

## File Structure

**Production change (minimal):**
- Modify `src/knowledge_service/stores/rag.py` — add `retrieval_mode` param to `retrieve()` + a `chunks_only` branch.
- Modify `src/knowledge_service/api/ask.py` — add `retrieval_mode` field to `AskRequest`; skip classification when `chunks_only`; pass mode through.
- Modify `src/knowledge_service/config.py` — add judge + eval-concurrency settings.

**New eval package:**
- `src/knowledge_service/eval/__init__.py` — package marker.
- `src/knowledge_service/eval/metrics.py` — pure retrieval metrics.
- `src/knowledge_service/eval/judge.py` — Claude-as-judge client + response parsing.
- `src/knowledge_service/eval/golden.py` — golden-set dataclasses + JSON loader/validator.
- `src/knowledge_service/eval/report.py` — pure aggregation + summary-table formatting.
- `src/knowledge_service/eval/runner.py` — orchestration + CLI.
- `src/knowledge_service/eval/__main__.py` — `python -m knowledge_service.eval` entrypoint.
- `src/knowledge_service/eval/golden.json` — the curated golden query set (checked in).
- `src/knowledge_service/eval/README.md` — run instructions.

**New scripts (operational):**
- `scripts/export_prod_snapshot.py` — dumps prod Postgres + oxigraph to a local snapshot dir with a manifest.
- `scripts/gen_golden_candidates.py` — auto-generates golden-set candidates from snapshot chunks.

**Tests (run in CI):**
- `tests/eval/__init__.py`
- `tests/eval/test_retrieval_mode.py` (covers `rag.py` + `ask.py` changes)
- `tests/eval/test_metrics.py`
- `tests/eval/test_golden.py`
- `tests/eval/test_judge.py`
- `tests/eval/test_report.py`
- `tests/eval/test_runner.py`

**Gitignored:** `src/knowledge_service/eval/reports/`, `data/snapshot/`, `golden_candidates.json`.

---

## Reference: exact existing signatures (verified against source, do not guess)

From `src/knowledge_service/stores/rag.py`:
- `RAGRetriever.__init__(self, embedding_client, embedding_store, knowledge_store, entity_store=None, classify_client=None)`
- `async def retrieve(self, question, max_sources=5, min_confidence=0.0, intent=None) -> RetrievalContext`
- `async def classify(self, question) -> QueryIntent` → `QueryIntent(intent, entities)`
- `RetrievalContext(content_results=[], knowledge_triples=[], contradictions=[], traversal_depth=None)` — all fields default to empty/None.
- Embedding via `self._embedding_client.embed(question)`.

From `src/knowledge_service/clients/rag.py`:
- `RAGClient.answer(self, question: str, context: RetrievalContext) -> RAGAnswer`; `RAGAnswer(answer: str)`.

From `src/knowledge_service/api/ask.py`:
- `AskRequest(question, max_sources=5, min_confidence=0.0)`; `post_ask(body, request)` calls `retriever.classify()` then `retriever.retrieve(...)`. Final `AskResponse` already guards `intent.intent if intent else None`.

Standalone wiring (from `main.py` lifespan) — **signatures verified**:
- `EmbeddingClient(base_url, model, api_key)` and `ExtractionClient(base_url, model, api_key)` from `knowledge_service.clients.llm`.
- `RAGClient(base_url, model, api_key)` from `knowledge_service.clients.rag` (**not** `clients.llm`).
- `BaseLLMClient(base_url, model, api_key)` from `knowledge_service.clients.base` — used as `classify_client` (main.py builds a dedicated one; it does not reuse the extraction client).
- `TripleStore(data_dir=...)` — exposes `flush()` (no `close()`).
- `ContentStore(pool, *, exclude_inflight=False)`.
- `EntityStore(pool, embedding_client, cache_size=None)`.
- `bootstrap_ontology(store, ontology_dir)` with `ontology_dir = Path(knowledge_service.__file__).parent / "ontology"`.
- `pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)`.

From `src/knowledge_service/_utils.py`: `_extract_json(text) -> dict | None` (tolerates markdown fences / `<think>` tags / trailing text).

From `src/knowledge_service/config.py`: `Settings` (pydantic-settings, `extra="ignore"`, env_file `.env`). `admin_password`, `secret_key` required (no default).

Test conventions (from `tests/test_rag_retriever.py`): `AsyncMock` for clients/stores; `mock.embed.return_value = [0.1] * 768`; `asyncio_mode = "auto"` so async tests need no decorator.

---

## Task 1: Add `retrieval_mode` to RAGRetriever.retrieve()

**Files:**
- Modify: `src/knowledge_service/stores/rag.py` (the `retrieve` method, ~line 165)
- Create: `tests/eval/__init__.py`
- Test: `tests/eval/test_retrieval_mode.py`

- [ ] **Step 1: Create the test package marker**

Create `tests/eval/__init__.py` (empty file):

```python
```

- [ ] **Step 2: Write the failing test**

Create `tests/eval/test_retrieval_mode.py`:

```python
"""Tests for the retrieval_mode toggle on RAGRetriever and /api/ask."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from knowledge_service.stores.rag import RAGRetriever, RetrievalContext


def _make_embedding_client():
    mock = AsyncMock()
    mock.embed.return_value = [0.1] * 768
    return mock


def _make_embedding_store(content_rows=None, entity_rows=None, predicate_rows=None):
    mock = AsyncMock()
    mock.search.return_value = content_rows or []
    mock.search_entities.return_value = entity_rows or []
    mock.search_predicates.return_value = predicate_rows or []
    return mock


def _make_knowledge_store():
    mock = MagicMock()
    mock.get_triples.return_value = []
    mock.find_contradictions.return_value = []
    return mock


_CONTENT_ROW = {
    "id": "chunk-1",
    "content_id": "content-1",
    "chunk_text": "text",
    "url": "https://example.com/a",
    "title": "A",
    "source_type": "article",
    "similarity": 0.9,
}


class TestChunksOnlyMode:
    async def test_chunks_only_returns_content_only(self):
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
            knowledge_store=_make_knowledge_store(),
        )
        ctx = await retriever.retrieve(
            "q", max_sources=5, min_confidence=0.0, retrieval_mode="chunks_only"
        )
        assert isinstance(ctx, RetrievalContext)
        assert len(ctx.content_results) == 1
        assert ctx.knowledge_triples == []
        assert ctx.contradictions == []

    async def test_chunks_only_skips_graph_calls(self):
        ks = _make_knowledge_store()
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
            knowledge_store=ks,
        )
        await retriever.retrieve(
            "q", max_sources=5, min_confidence=0.0, retrieval_mode="chunks_only"
        )
        es.search_entities.assert_not_called()
        es.search_predicates.assert_not_called()
        ks.get_triples.assert_not_called()
        ks.find_contradictions.assert_not_called()

    async def test_full_mode_is_default(self):
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        retriever = RAGRetriever(
            embedding_client=_make_embedding_client(),
            embedding_store=es,
            knowledge_store=_make_knowledge_store(),
        )
        await retriever.retrieve("q", max_sources=5, min_confidence=0.0)
        es.search_entities.assert_called()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `uv run pytest tests/eval/test_retrieval_mode.py -v`
Expected: FAIL — `retrieve()` got an unexpected keyword argument `retrieval_mode`.

- [ ] **Step 4: Implement the `retrieval_mode` branch**

In `src/knowledge_service/stores/rag.py`, replace the `retrieve` method head (signature + dispatch). Only the signature line and the `chunks_only` block are new; the rest is unchanged:

```python
    async def retrieve(
        self,
        question: str,
        max_sources: int = 5,
        min_confidence: float = 0.0,
        intent: QueryIntent | None = None,
        retrieval_mode: str = "full",
    ) -> RetrievalContext:
        embedding = await self._embedding_client.embed(question)

        if retrieval_mode == "chunks_only":
            content_results = await self._embedding_store.search(
                query_embedding=embedding, limit=max_sources, query_text=question
            )
            return RetrievalContext(content_results=content_results)

        if intent is None or intent.intent == "semantic":
            return await self._retrieve_semantic(question, embedding, max_sources, min_confidence)
        elif intent.intent == "entity":
            return await self._retrieve_entity(
                question, embedding, intent.entities, max_sources, min_confidence
            )
        elif intent.intent == "graph":
            return await self._retrieve_graph(
                question,
                embedding,
                intent.entities,
                max_sources,
                min_confidence,
            )
        else:
            return await self._retrieve_semantic(question, embedding, max_sources, min_confidence)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_retrieval_mode.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Run existing retriever tests for regressions**

Run: `uv run pytest tests/test_rag_retriever.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_service/stores/rag.py tests/eval/__init__.py tests/eval/test_retrieval_mode.py
git commit -m "feat(rag): add retrieval_mode toggle (full|chunks_only) to RAGRetriever"
```

---

## Task 2: Thread `retrieval_mode` through /api/ask

**Files:**
- Modify: `src/knowledge_service/api/ask.py` (`AskRequest` ~line 15, `post_ask` ~line 53)
- Test: `tests/eval/test_retrieval_mode.py` (append)

- [ ] **Step 1: Write the failing test (append)**

Append to `tests/eval/test_retrieval_mode.py`:

```python
from knowledge_service.api.ask import AskRequest


class TestAskRequestMode:
    def test_default_mode_is_full(self):
        req = AskRequest(question="hello")
        assert req.retrieval_mode == "full"

    def test_chunks_only_is_accepted(self):
        req = AskRequest(question="hello", retrieval_mode="chunks_only")
        assert req.retrieval_mode == "chunks_only"

    def test_invalid_mode_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AskRequest(question="hello", retrieval_mode="bogus")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/eval/test_retrieval_mode.py::TestAskRequestMode -v`
Expected: FAIL — `AskRequest` has no field `retrieval_mode`.

- [ ] **Step 3: Add the field to `AskRequest`**

In `src/knowledge_service/api/ask.py`, add the import at the top (with the other imports):

```python
from typing import Literal
```

Then add the field to `AskRequest`:

```python
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=_MAX_QUESTION_LEN)
    max_sources: int = Field(5, ge=1, le=20)
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)
    retrieval_mode: Literal["full", "chunks_only"] = "full"
```

- [ ] **Step 4: Use the mode in `post_ask`**

In `post_ask`, replace the classify + retrieve block (currently lines ~59-67):

```python
    # Classify only when the graph path will use the intent.
    if body.retrieval_mode == "chunks_only":
        intent = None
    else:
        intent = await retriever.classify(body.question)

    context = await retriever.retrieve(
        body.question,
        max_sources=body.max_sources,
        min_confidence=body.min_confidence,
        intent=intent,
        retrieval_mode=body.retrieval_mode,
    )
```

The final `AskResponse(...)` already handles a `None` intent via `intent.intent if intent else None`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_retrieval_mode.py::TestAskRequestMode -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Run the ask API tests for regressions**

Run: `uv run pytest tests/test_api_ask.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_service/api/ask.py tests/eval/test_retrieval_mode.py
git commit -m "feat(api): thread retrieval_mode through /api/ask (default full)"
```

---

## Task 3: Pure retrieval metrics

**Files:**
- Create: `src/knowledge_service/eval/__init__.py`
- Create: `src/knowledge_service/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

- [ ] **Step 1: Create the package marker**

Create `src/knowledge_service/eval/__init__.py`:

```python
"""Evaluation harness for knowledge-service retrieval and answer quality."""
```

- [ ] **Step 2: Write the failing tests**

Create `tests/eval/test_metrics.py`:

```python
"""Unit tests for pure retrieval metrics."""

from __future__ import annotations

import math

from knowledge_service.eval.metrics import (
    dcg,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0

    def test_half_relevant_retrieved(self):
        assert recall_at_k(["a", "x"], {"a", "b"}, k=5) == 0.5

    def test_k_truncates(self):
        assert recall_at_k(["x", "b"], {"a", "b"}, k=1) == 0.0

    def test_no_relevant_set_is_zero(self):
        assert recall_at_k(["a"], set(), k=5) == 0.0


class TestPrecisionAtK:
    def test_all_retrieved_relevant(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, k=5) == 1.0

    def test_half_precision(self):
        assert precision_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5

    def test_empty_retrieved_is_zero(self):
        assert precision_at_k([], {"a"}, k=5) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mrr(["a", "x"], {"a"}) == 1.0

    def test_second_position(self):
        assert mrr(["x", "a"], {"a"}) == 0.5

    def test_no_hit_is_zero(self):
        assert mrr(["x", "y"], {"a"}) == 0.0


class TestNDCG:
    def test_dcg_single_hit_first(self):
        assert dcg(["a"], {"a"}) == 1.0

    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_reversed_is_less_than_one(self):
        val = ndcg_at_k(["x", "a"], {"a"}, k=2)
        assert 0.0 < val < 1.0
        assert math.isclose(val, (1 / math.log2(3)) / 1.0)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/eval/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: knowledge_service.eval.metrics`.

- [ ] **Step 4: Implement `metrics.py`**

Create `src/knowledge_service/eval/metrics.py`:

```python
"""Pure retrieval metrics. No I/O. Inputs: ranked retrieved ids + a relevant set."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant ids found in the top-k retrieved ids."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant_set) / len(relevant_set)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top-k retrieved ids that are relevant."""
    relevant_set = set(relevant)
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant_set) / len(top)


def mrr(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant id (0.0 if none)."""
    relevant_set = set(relevant)
    for idx, r in enumerate(retrieved, start=1):
        if r in relevant_set:
            return 1.0 / idx
    return 0.0


def dcg(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Discounted cumulative gain with binary gains (1 if relevant else 0)."""
    relevant_set = set(relevant)
    total = 0.0
    for idx, r in enumerate(retrieved, start=1):
        if r in relevant_set:
            total += 1.0 / math.log2(idx + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalized DCG at k with binary relevance."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    actual = dcg(retrieved[:k], relevant_set)
    ideal_hits = min(len(relevant_set), k)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return actual / ideal if ideal else 0.0
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/eval/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/eval/__init__.py src/knowledge_service/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat(eval): pure retrieval metrics (recall/precision/MRR/nDCG)"
```

---

## Task 4: Golden-set dataclasses + loader

**Files:**
- Create: `src/knowledge_service/eval/golden.py`
- Create: `src/knowledge_service/eval/golden.json` (empty array placeholder)
- Test: `tests/eval/test_golden.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_golden.py`:

```python
"""Tests for the golden-set loader and validation."""

from __future__ import annotations

import json

import pytest

from knowledge_service.eval.golden import GoldenItem, load_golden


def _write(tmp_path, items):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(items))
    return path


def test_loads_valid_items(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "id": "q1",
                "question": "What is X?",
                "query_type": "entity",
                "relevant_source_ids": ["c1", "c2"],
                "reference_answer": "X is a thing.",
                "notes": "",
            }
        ],
    )
    items = load_golden(path)
    assert len(items) == 1
    assert isinstance(items[0], GoldenItem)
    assert items[0].id == "q1"
    assert items[0].query_type == "entity"
    assert items[0].relevant_source_ids == ["c1", "c2"]


def test_rejects_unknown_query_type(tmp_path):
    path = _write(
        tmp_path,
        [
            {
                "id": "q1",
                "question": "Q",
                "query_type": "nonsense",
                "relevant_source_ids": [],
                "reference_answer": "A",
            }
        ],
    )
    with pytest.raises(ValueError, match="query_type"):
        load_golden(path)


def test_rejects_duplicate_ids(tmp_path):
    item = {
        "id": "dup",
        "question": "Q",
        "query_type": "semantic",
        "relevant_source_ids": [],
        "reference_answer": "A",
    }
    path = _write(tmp_path, [item, dict(item)])
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(path)


def test_missing_required_field_raises(tmp_path):
    path = _write(tmp_path, [{"id": "q1", "question": "Q", "query_type": "semantic"}])
    with pytest.raises(ValueError):
        load_golden(path)


def test_empty_array_is_valid(tmp_path):
    path = _write(tmp_path, [])
    assert load_golden(path) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/eval/test_golden.py -v`
Expected: FAIL — `ModuleNotFoundError: knowledge_service.eval.golden`.

- [ ] **Step 3: Implement `golden.py`**

Create `src/knowledge_service/eval/golden.py`:

```python
"""Golden query set: dataclasses + a validating JSON loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_VALID_QUERY_TYPES = {"semantic", "entity", "graph", "gtd"}


@dataclass(frozen=True)
class GoldenItem:
    id: str
    question: str
    query_type: str
    relevant_source_ids: list[str]
    reference_answer: str
    notes: str = ""


def _coerce_item(raw: dict) -> GoldenItem:
    required = ("id", "question", "query_type", "relevant_source_ids", "reference_answer")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"golden item missing required fields: {missing} in {raw!r}")
    if raw["query_type"] not in _VALID_QUERY_TYPES:
        raise ValueError(
            f"invalid query_type {raw['query_type']!r}; must be one of {_VALID_QUERY_TYPES}"
        )
    if not isinstance(raw["relevant_source_ids"], list):
        raise ValueError(f"relevant_source_ids must be a list in {raw['id']!r}")
    return GoldenItem(
        id=str(raw["id"]),
        question=str(raw["question"]),
        query_type=str(raw["query_type"]),
        relevant_source_ids=[str(x) for x in raw["relevant_source_ids"]],
        reference_answer=str(raw["reference_answer"]),
        notes=str(raw.get("notes", "")),
    )


def load_golden(path: str | Path) -> list[GoldenItem]:
    """Load + validate the golden set. Raises ValueError on any malformed item."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("golden file must be a JSON array of items")
    items = [_coerce_item(raw) for raw in data]
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise ValueError(f"duplicate golden id: {it.id!r}")
        seen.add(it.id)
    return items
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/eval/test_golden.py -v`
Expected: PASS.

- [ ] **Step 5: Create the golden.json placeholder**

Create `src/knowledge_service/eval/golden.json` (populated for real in Task 9):

```json
[]
```

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/eval/golden.py src/knowledge_service/eval/golden.json tests/eval/test_golden.py
git commit -m "feat(eval): golden-set dataclasses and validating loader"
```

---

## Task 5: Claude-as-judge module

**Files:**
- Modify: `src/knowledge_service/config.py` (add judge + concurrency settings)
- Create: `src/knowledge_service/eval/judge.py`
- Test: `tests/eval/test_judge.py`

- [ ] **Step 1: Add settings**

In `src/knowledge_service/config.py`, add these fields before `model_config` (after the `reader_exclude_inflight` line):

```python
    # Eval harness
    eval_judge_base_url: str = "https://api.anthropic.com"
    eval_judge_model: str = "claude-opus-4-8"
    eval_judge_api_key: str = ""  # Anthropic key; required only when running the eval judge
    eval_concurrency: int = 4
```

- [ ] **Step 2: Write the failing tests**

Create `tests/eval/test_judge.py`:

```python
"""Tests for the Claude-as-judge module (HTTP mocked)."""

from __future__ import annotations

import json

import pytest

from knowledge_service.eval.judge import Judge, JudgeScore, parse_judge_response


class TestParseJudgeResponse:
    def test_parses_scores_and_rationale(self):
        raw = json.dumps({"faithfulness": 0.9, "correctness": 0.8, "rationale": "well grounded"})
        score = parse_judge_response(raw)
        assert score == JudgeScore(faithfulness=0.9, correctness=0.8, rationale="well grounded")

    def test_parses_with_markdown_fence(self):
        raw = '```json\n{"faithfulness": 1, "correctness": 0, "rationale": "x"}\n```'
        score = parse_judge_response(raw)
        assert score.faithfulness == 1.0
        assert score.correctness == 0.0

    def test_clamps_out_of_range(self):
        raw = json.dumps({"faithfulness": 2.5, "correctness": -1, "rationale": ""})
        score = parse_judge_response(raw)
        assert score.faithfulness == 1.0
        assert score.correctness == 0.0

    def test_unparseable_returns_zero_with_rationale(self):
        score = parse_judge_response("the model rambled with no json")
        assert score.faithfulness == 0.0
        assert score.correctness == 0.0
        assert "unparseable" in score.rationale.lower()


class TestJudge:
    async def test_calls_anthropic_and_returns_score(self, httpx_mock):
        httpx_mock.add_response(
            url="https://api.anthropic.com/v1/messages",
            json={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"faithfulness": 0.7, "correctness": 0.6, "rationale": "ok"}
                        ),
                    }
                ]
            },
        )
        judge = Judge(base_url="https://api.anthropic.com", model="claude-opus-4-8", api_key="k")
        score = await judge.score_one(
            question="What is X?",
            reference_answer="X is a thing.",
            generated_answer="X is a thing.",
            retrieved_context="some context",
        )
        await judge.close()
        assert isinstance(score, JudgeScore)
        assert score.faithfulness == 0.7
        assert score.correctness == 0.6

    async def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="api key"):
            Judge(base_url="https://api.anthropic.com", model="claude-opus-4-8", api_key="")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/eval/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: knowledge_service.eval.judge`.

- [ ] **Step 4: Implement `judge.py`**

Create `src/knowledge_service/eval/judge.py`:

```python
"""Claude-as-judge: scores generated answers for faithfulness + correctness.

The system-under-test is qwen3; the judge is Claude so the SUT does not grade
itself. Uses the Anthropic Messages API directly (httpx), not the OpenAI shim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from knowledge_service._utils import _extract_json

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """You are a strict evaluator of a question-answering system.

Score the GENERATED ANSWER on two axes from 0.0 to 1.0:
- "faithfulness": is the answer grounded in the RETRIEVED CONTEXT, with no
  fabricated claims? 1.0 = fully grounded, 0.0 = fabricated/unsupported.
- "correctness": does the answer match the REFERENCE ANSWER semantically?
  1.0 = equivalent, 0.0 = wrong or missing.

Return ONLY a JSON object:
{{"faithfulness": <float>, "correctness": <float>, "rationale": "<one sentence>"}}

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RETRIEVED CONTEXT:
{retrieved_context}

GENERATED ANSWER:
{generated_answer}
"""


@dataclass(frozen=True)
class JudgeScore:
    faithfulness: float
    correctness: float
    rationale: str


def _clamp(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def parse_judge_response(raw: str) -> JudgeScore:
    """Parse the judge's JSON reply, tolerating fences/think-tags via _extract_json."""
    parsed = _extract_json(raw)
    if not isinstance(parsed, dict):
        return JudgeScore(0.0, 0.0, "unparseable judge response")
    return JudgeScore(
        faithfulness=_clamp(parsed.get("faithfulness")),
        correctness=_clamp(parsed.get("correctness")),
        rationale=str(parsed.get("rationale", "")),
    )


class Judge:
    """Anthropic Messages API client for answer scoring."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        if not api_key:
            raise ValueError("Judge requires an Anthropic api key (eval_judge_api_key)")
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )

    async def score_one(
        self,
        question: str,
        reference_answer: str,
        generated_answer: str,
        retrieved_context: str,
    ) -> JudgeScore:
        prompt = _JUDGE_PROMPT.format(
            question=question,
            reference_answer=reference_answer,
            retrieved_context=retrieved_context,
            generated_answer=generated_answer,
        )
        resp = await self._client.post(
            "/v1/messages",
            json={
                "model": self._model,
                "max_tokens": 512,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        blocks = resp.json().get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return parse_judge_response(text)

    async def close(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/eval/test_judge.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/knowledge_service/config.py src/knowledge_service/eval/judge.py tests/eval/test_judge.py
git commit -m "feat(eval): Claude-as-judge scoring (faithfulness/correctness)"
```

---

## Task 6: Report aggregation + summary table

**Files:**
- Create: `src/knowledge_service/eval/report.py`
- Test: `tests/eval/test_report.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/eval/test_report.py`:

```python
"""Tests for report aggregation + summary-table formatting (pure)."""

from __future__ import annotations

from knowledge_service.eval.report import QueryResult, aggregate, format_summary_table


def _qr(mode, qtype, **metrics):
    base = {
        "recall_at_k": 0.0,
        "precision_at_k": 0.0,
        "mrr": 0.0,
        "ndcg_at_k": 0.0,
        "faithfulness": 0.0,
        "correctness": 0.0,
    }
    base.update(metrics)
    return QueryResult(query_id=f"{mode}-{qtype}", mode=mode, query_type=qtype, metrics=base)


class TestAggregate:
    def test_averages_within_mode_and_type(self):
        results = [
            _qr("full", "semantic", recall_at_k=1.0, correctness=0.8),
            _qr("full", "semantic", recall_at_k=0.0, correctness=0.4),
        ]
        cell = aggregate(results)[("full", "semantic")]
        assert cell["recall_at_k"] == 0.5
        assert cell["correctness"] == 0.6
        assert cell["count"] == 2

    def test_separates_modes(self):
        results = [
            _qr("full", "gtd", recall_at_k=1.0),
            _qr("chunks_only", "gtd", recall_at_k=0.0),
        ]
        agg = aggregate(results)
        assert agg[("full", "gtd")]["recall_at_k"] == 1.0
        assert agg[("chunks_only", "gtd")]["recall_at_k"] == 0.0


class TestFormatSummaryTable:
    def test_table_has_header_and_rows(self):
        agg = aggregate([_qr("full", "semantic", recall_at_k=1.0)])
        table = format_summary_table(agg)
        assert "mode" in table
        assert "query_type" in table
        assert "full" in table
        assert "semantic" in table
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/eval/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: knowledge_service.eval.report`.

- [ ] **Step 3: Implement `report.py`**

Create `src/knowledge_service/eval/report.py`:

```python
"""Report types + pure aggregation and summary-table formatting."""

from __future__ import annotations

from dataclasses import dataclass, field

_METRIC_KEYS = (
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "faithfulness",
    "correctness",
)


@dataclass
class QueryResult:
    query_id: str
    mode: str
    query_type: str
    metrics: dict[str, float] = field(default_factory=dict)
    answer: str = ""
    rationale: str = ""


def aggregate(results: list[QueryResult]) -> dict[tuple[str, str], dict[str, float]]:
    """Average each metric within every (mode, query_type) bucket.

    Returns a mapping keyed by (mode, query_type) to averaged metrics plus a
    "count" of contributing queries.
    """
    buckets: dict[tuple[str, str], list[QueryResult]] = {}
    for r in results:
        buckets.setdefault((r.mode, r.query_type), []).append(r)

    agg: dict[tuple[str, str], dict[str, float]] = {}
    for key, rows in buckets.items():
        cell: dict[str, float] = {}
        for mk in _METRIC_KEYS:
            vals = [row.metrics.get(mk, 0.0) for row in rows]
            cell[mk] = sum(vals) / len(vals) if vals else 0.0
        cell["count"] = len(rows)
        agg[key] = cell
    return agg


def format_summary_table(agg: dict[tuple[str, str], dict[str, float]]) -> str:
    """Render the aggregated metrics as a fixed-width text table."""
    header = (
        f"{'mode':<12} {'query_type':<10} "
        f"{'recall@k':>9} {'prec@k':>8} {'mrr':>6} {'ndcg':>6} "
        f"{'faithful':>9} {'correct':>8} {'n':>4}"
    )
    lines = [header, "-" * len(header)]
    for mode, qtype in sorted(agg.keys()):
        cell = agg[(mode, qtype)]
        lines.append(
            f"{mode:<12} {qtype:<10} "
            f"{cell['recall_at_k']:>9.3f} {cell['precision_at_k']:>8.3f} "
            f"{cell['mrr']:>6.3f} {cell['ndcg_at_k']:>6.3f} "
            f"{cell['faithfulness']:>9.3f} {cell['correctness']:>8.3f} "
            f"{int(cell['count']):>4}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/eval/test_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/eval/report.py tests/eval/test_report.py
git commit -m "feat(eval): report aggregation + summary-table formatting"
```

---

## Task 7: Runner + CLI

**Files:**
- Create: `src/knowledge_service/eval/runner.py`
- Create: `src/knowledge_service/eval/__main__.py`
- Test: `tests/eval/test_runner.py`

The corpus-dependent run (real Postgres + oxigraph + LLMs) is exercised manually against the snapshot, like `tests/e2e/`. The unit test covers the pure scoring seam `score_query`.

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_runner.py`:

```python
"""Unit test for the runner's pure scoring seam."""

from __future__ import annotations

from knowledge_service.eval.golden import GoldenItem
from knowledge_service.eval.judge import JudgeScore
from knowledge_service.eval.runner import score_query
from knowledge_service.stores.rag import RetrievalContext


def test_score_query_combines_retrieval_and_judge():
    item = GoldenItem(
        id="q1",
        question="What is X?",
        query_type="entity",
        relevant_source_ids=["content-1"],
        reference_answer="X is a thing.",
    )
    ctx = RetrievalContext(
        content_results=[{"content_id": "content-1", "chunk_text": "X is a thing."}]
    )
    result = score_query(
        item=item,
        mode="full",
        context=ctx,
        generated_answer="X is a thing.",
        judge_score=JudgeScore(faithfulness=0.9, correctness=0.8, rationale="ok"),
        k=5,
    )
    assert result.mode == "full"
    assert result.query_type == "entity"
    assert result.metrics["recall_at_k"] == 1.0
    assert result.metrics["precision_at_k"] == 1.0
    assert result.metrics["faithfulness"] == 0.9
    assert result.metrics["correctness"] == 0.8


def test_score_query_falls_back_to_id_when_no_content_id():
    item = GoldenItem(
        id="q2",
        question="Q",
        query_type="semantic",
        relevant_source_ids=["chunk-9"],
        reference_answer="A",
    )
    ctx = RetrievalContext(content_results=[{"id": "chunk-9", "chunk_text": "A"}])
    result = score_query(
        item=item,
        mode="chunks_only",
        context=ctx,
        generated_answer="A",
        judge_score=JudgeScore(0.0, 0.0, ""),
        k=5,
    )
    assert result.metrics["recall_at_k"] == 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/eval/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: knowledge_service.eval.runner`.

- [ ] **Step 3: Implement `runner.py`**

Create `src/knowledge_service/eval/runner.py`:

```python
"""Eval runner: load golden -> run each mode -> score -> report.

The corpus-dependent run (real Postgres, oxigraph, LLMs) is invoked via the CLI.
`score_query` is the pure scoring seam, unit-tested in isolation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from knowledge_service.config import settings
from knowledge_service.eval.golden import GoldenItem, load_golden
from knowledge_service.eval.judge import Judge, JudgeScore
from knowledge_service.eval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from knowledge_service.eval.report import QueryResult, aggregate, format_summary_table
from knowledge_service.stores.rag import RetrievalContext

logger = logging.getLogger(__name__)

_DEFAULT_GOLDEN = Path(__file__).with_name("golden.json")
_REPORTS_DIR = Path(__file__).with_name("reports")


def _retrieved_ids(context: RetrievalContext) -> list[str]:
    """Ordered content/chunk ids from a retrieval context (content_id, else id)."""
    ids: list[str] = []
    for row in context.content_results:
        cid = row.get("content_id") or row.get("id")
        if cid is not None:
            ids.append(str(cid))
    return ids


def _format_context_for_judge(context: RetrievalContext) -> str:
    parts: list[str] = []
    for row in context.content_results:
        parts.append(str(row.get("chunk_text") or row.get("summary") or ""))
    for t in context.knowledge_triples:
        parts.append(f"{t.get('subject')} -> {t.get('predicate')} -> {t.get('object')}")
    return "\n".join(parts)


def score_query(
    item: GoldenItem,
    mode: str,
    context: RetrievalContext,
    generated_answer: str,
    judge_score: JudgeScore,
    k: int,
) -> QueryResult:
    """Pure: combine retrieval metrics + judge scores into a QueryResult."""
    retrieved = _retrieved_ids(context)
    relevant = item.relevant_source_ids
    metrics = {
        "recall_at_k": recall_at_k(retrieved, relevant, k),
        "precision_at_k": precision_at_k(retrieved, relevant, k),
        "mrr": mrr(retrieved, relevant),
        "ndcg_at_k": ndcg_at_k(retrieved, relevant, k),
        "faithfulness": judge_score.faithfulness,
        "correctness": judge_score.correctness,
        "triples_surfaced": float(len(context.knowledge_triples)),
    }
    return QueryResult(
        query_id=item.id,
        mode=mode,
        query_type=item.query_type,
        metrics=metrics,
        answer=generated_answer,
        rationale=judge_score.rationale,
    )


async def _build_components():
    """Construct stores + clients + retriever against the configured snapshot.

    Mirrors main.py lifespan wiring but without the FastAPI app / worker loop.
    Signatures verified against the codebase:
      - RAGClient lives in knowledge_service.clients.rag (NOT clients.llm).
      - EntityStore(pool, embedding_client).
      - ContentStore(pool, exclude_inflight=...).
      - bootstrap_ontology(store, ontology_dir).
      - classify_client is a plain BaseLLMClient (matches main.py).
      - TripleStore has flush(), not close().
    """
    import asyncpg

    import knowledge_service
    from knowledge_service.clients.base import BaseLLMClient
    from knowledge_service.clients.llm import EmbeddingClient, ExtractionClient
    from knowledge_service.clients.rag import RAGClient
    from knowledge_service.ontology.bootstrap import bootstrap_ontology
    from knowledge_service.stores.content import ContentStore
    from knowledge_service.stores.entities import EntityStore
    from knowledge_service.stores.rag import RAGRetriever
    from knowledge_service.stores.triples import TripleStore

    pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=10)
    knowledge_store = TripleStore(data_dir=settings.oxigraph_data_dir)
    ontology_dir = Path(knowledge_service.__file__).resolve().parent / "ontology"
    bootstrap_ontology(knowledge_store, ontology_dir)

    embedding_client = EmbeddingClient(
        settings.llm_base_url, settings.llm_embed_model, settings.llm_api_key
    )
    content_store = ContentStore(pool, exclude_inflight=settings.reader_exclude_inflight)
    entity_store = EntityStore(pool, embedding_client)

    extraction_client = ExtractionClient(
        settings.llm_base_url, settings.llm_chat_model, settings.llm_api_key
    )
    rag_model = settings.llm_rag_model or settings.llm_chat_model
    rag_client = RAGClient(settings.llm_base_url, rag_model, settings.llm_api_key)
    classify_client = BaseLLMClient(
        settings.llm_base_url, settings.llm_chat_model, settings.llm_api_key
    )

    retriever = RAGRetriever(
        embedding_client=embedding_client,
        embedding_store=content_store,
        knowledge_store=knowledge_store,
        entity_store=entity_store,
        classify_client=classify_client,
    )
    clients = (embedding_client, extraction_client, rag_client, classify_client)
    return pool, knowledge_store, retriever, rag_client, clients


async def run_eval(modes: list[str], k: int, golden_path: Path) -> list[QueryResult]:
    items = load_golden(golden_path)
    judge = Judge(
        base_url=settings.eval_judge_base_url,
        model=settings.eval_judge_model,
        api_key=settings.eval_judge_api_key,
    )
    pool, knowledge_store, retriever, rag_client, clients = await _build_components()
    sem = asyncio.Semaphore(settings.eval_concurrency)

    async def _one(item: GoldenItem, mode: str) -> QueryResult:
        async with sem:
            intent = None if mode == "chunks_only" else await retriever.classify(item.question)
            context = await retriever.retrieve(
                item.question,
                max_sources=k,
                min_confidence=0.0,
                intent=intent,
                retrieval_mode=mode,
            )
            answer_obj = await rag_client.answer(item.question, context)
            judge_score = await judge.score_one(
                question=item.question,
                reference_answer=item.reference_answer,
                generated_answer=answer_obj.answer,
                retrieved_context=_format_context_for_judge(context),
            )
            return score_query(item, mode, context, answer_obj.answer, judge_score, k)

    try:
        tasks = [_one(item, mode) for mode in modes for item in items]
        results = await asyncio.gather(*tasks)
    finally:
        await judge.close()
        for c in clients:
            await c.close()
        await pool.close()
        knowledge_store.flush()  # TripleStore exposes flush(), not close()
    return list(results)


def _write_report(results: list[QueryResult], k: int, timestamp: str) -> Path:
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    agg = aggregate(results)
    payload = {
        "k": k,
        "timestamp": timestamp,
        "summary": {f"{m}::{t}": cell for (m, t), cell in agg.items()},
        "queries": [
            {
                "query_id": r.query_id,
                "mode": r.mode,
                "query_type": r.query_type,
                "metrics": r.metrics,
                "answer": r.answer,
                "rationale": r.rationale,
            }
            for r in results
        ],
    }
    path = _REPORTS_DIR / f"{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


async def _amain(args: argparse.Namespace) -> None:
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    golden_path = Path(args.golden) if args.golden else _DEFAULT_GOLDEN
    results = await run_eval(modes=modes, k=args.k, golden_path=golden_path)
    print(format_summary_table(aggregate(results)))
    path = _write_report(results, args.k, args.timestamp)
    print(f"\nReport written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the KS eval harness.")
    parser.add_argument("--modes", default="full,chunks_only")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--golden", default="")
    parser.add_argument(
        "--timestamp",
        required=True,
        help="Report filename stamp, e.g. 2026-05-30T1200 (passed in; the harness "
        "does not call the clock so runs stay reproducible).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `__main__.py`**

Create `src/knowledge_service/eval/__main__.py`:

```python
"""Entrypoint: python -m knowledge_service.eval"""

from knowledge_service.eval.runner import main

main()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_runner.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the whole eval test suite**

Run: `uv run pytest tests/eval/ -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_service/eval/runner.py src/knowledge_service/eval/__main__.py tests/eval/test_runner.py
git commit -m "feat(eval): runner + CLI (python -m knowledge_service.eval)"
```

---

## Task 8: Snapshot export script

**Files:**
- Create: `scripts/export_prod_snapshot.py`
- Test: none (operational wrapper; validated by running against prod read-only)

Reads prod connection details from env vars only (NEVER hardcoded, per the credentials policy). The oxigraph half resolves the spec's open question by preferring an offline dump from a copied data dir.

- [ ] **Step 1: Implement the script**

Create `scripts/export_prod_snapshot.py`:

```python
#!/usr/bin/env python3
"""Export a prod knowledge-service snapshot (Postgres + oxigraph) for the eval harness.

Reads connection details from env vars only (no hardcoded credentials):
    KS_PROD_DATABASE_URL   postgres URL to dump (pg_dump is online/read-only safe)
    KS_PROD_OXIGRAPH_DIR   path to a COPY of the prod oxigraph data dir (offline dump)
    KS_SNAPSHOT_DIR        output directory (default ./data/snapshot)

Usage:
    KS_PROD_DATABASE_URL=postgresql://... \\
    KS_PROD_OXIGRAPH_DIR=/path/to/oxigraph-copy \\
    python scripts/export_prod_snapshot.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"error: required environment variable {name} is not set")
    return value


def _dump_postgres(database_url: str, out_dir: Path) -> Path:
    dump_path = out_dir / "knowledge.dump"
    subprocess.run(
        ["pg_dump", "--format=custom", "--no-owner", "--file", str(dump_path), database_url],
        check=True,
    )
    return dump_path


def _dump_oxigraph(oxigraph_dir: str, out_dir: Path) -> Path:
    """Dump a COPY of the prod oxigraph store to N-Quads (offline, read-only open)."""
    import pyoxigraph

    nq_path = out_dir / "oxigraph.nq"
    store = pyoxigraph.Store(oxigraph_dir)
    with open(nq_path, "wb") as fh:
        store.dump(fh, "application/n-quads")
    return nq_path


def main() -> None:
    database_url = _require_env("KS_PROD_DATABASE_URL")
    oxigraph_dir = _require_env("KS_PROD_OXIGRAPH_DIR")
    snapshot_dir = Path(os.environ.get("KS_SNAPSHOT_DIR", "./data/snapshot"))
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    pg_path = _dump_postgres(database_url, snapshot_dir)
    nq_path = _dump_oxigraph(oxigraph_dir, snapshot_dir)

    manifest = {
        "postgres_dump": pg_path.name,
        "oxigraph_nquads": nq_path.name,
        "postgres_dump_bytes": pg_path.stat().st_size,
        "oxigraph_nquads_bytes": nq_path.stat().st_size,
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"snapshot written to {snapshot_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check missing-env failure**

Run: `uv run python scripts/export_prod_snapshot.py`
Expected: exits with `error: required environment variable KS_PROD_DATABASE_URL is not set` (confirms no hardcoded creds + clean failure).

- [ ] **Step 3: Commit**

```bash
git add scripts/export_prod_snapshot.py
git commit -m "feat(eval): prod snapshot export script (pg_dump + oxigraph n-quads)"
```

---

## Task 9: Build the golden set

**Files:**
- Create: `scripts/gen_golden_candidates.py`
- Modify: `src/knowledge_service/eval/golden.json` (final curated set)

Data work. Requires a restored local snapshot (Task 8) and the configured LLM.

- [ ] **Step 1: Implement the candidate generator**

Create `scripts/gen_golden_candidates.py`:

```python
#!/usr/bin/env python3
"""Generate golden-set CANDIDATES from snapshot chunks (auto-then-curate workflow).

Samples chunks across source_types, asks the LLM to produce a question answerable
from each chunk, and emits candidate golden items (relevance label = that chunk's
content_id). Output is a CANDIDATE file you curate by hand before saving to
src/knowledge_service/eval/golden.json.

Usage:
    python scripts/gen_golden_candidates.py --per-type 10 --out golden_candidates.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from knowledge_service._utils import _extract_json
from knowledge_service.clients.llm import ExtractionClient
from knowledge_service.config import settings

_Q_PROMPT = """Given this document chunk, write ONE natural question a user would ask
that is answerable ONLY from this chunk, plus a one-sentence reference answer.
Do not quote the chunk verbatim in the question.

Return JSON: {{"question": "...", "reference_answer": "..."}}

CHUNK:
{chunk}
"""


async def _sample_chunks(pool, per_type: int) -> list[dict]:
    rows = await pool.fetch(
        """
        SELECT c.content_id::text AS content_id, c.chunk_text, cm.source_type
        FROM content c
        JOIN content_metadata cm ON cm.id = c.content_id
        WHERE c.chunk_text <> ''
        ORDER BY cm.source_type, random()
        """
    )
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        bucket = by_type.setdefault(r["source_type"], [])
        if len(bucket) < per_type:
            bucket.append(dict(r))
    sampled: list[dict] = []
    for items in by_type.values():
        sampled.extend(items)
    return sampled


async def _amain(args) -> None:
    import asyncpg

    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=4)
    client = ExtractionClient(settings.llm_base_url, settings.llm_chat_model, settings.llm_api_key)
    chunks = await _sample_chunks(pool, args.per_type)
    candidates = []
    for i, ch in enumerate(chunks):
        resp = await client.client.post(
            "/v1/chat/completions",
            json={
                "model": client.model,
                "messages": [
                    {"role": "user", "content": _Q_PROMPT.format(chunk=ch["chunk_text"][:3000])}
                ],
            },
        )
        resp.raise_for_status()
        parsed = _extract_json(resp.json()["choices"][0]["message"]["content"])
        if not isinstance(parsed, dict) or "question" not in parsed:
            continue
        candidates.append(
            {
                "id": f"auto-{ch['source_type']}-{i:03d}",
                "question": parsed["question"],
                "query_type": "semantic",
                "relevant_source_ids": [ch["content_id"]],
                "reference_answer": parsed.get("reference_answer", ""),
                "notes": f"auto-generated from source_type={ch['source_type']}; CURATE ME",
            }
        )
    Path(args.out).write_text(json.dumps(candidates, indent=2))
    print(f"wrote {len(candidates)} candidates to {args.out}")
    await client.close()
    await pool.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-type", type=int, default=10)
    p.add_argument("--out", default="golden_candidates.json")
    asyncio.run(_amain(p.parse_args()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate candidates against the restored snapshot**

Run: `uv run python scripts/gen_golden_candidates.py --per-type 10 --out golden_candidates.json`
Expected: writes `golden_candidates.json` with ~40–60 candidates across source types.

- [ ] **Step 3: Curate candidates by hand**

Open `golden_candidates.json`. For each item: delete leaky/trivial ones; set `query_type` to the best of `semantic|entity|graph|gtd`; correct the reference answer; verify the `relevant_source_ids` chunk actually answers it. Keep ~40–60.

- [ ] **Step 4: Hand-author the GTD cross-document entries**

Add ~10–15 entries with `"query_type": "gtd"` covering real reference-tool questions ("what references do I have about X", "what did I decide about Y", "how does X relate to Y"). Look up the content ids that should surface via a quick `psql` query against the restored DB. Example shape:

```json
{
  "id": "gtd-001",
  "question": "What references do I have about evaluating LLM applications?",
  "query_type": "gtd",
  "relevant_source_ids": ["<content_id_1>", "<content_id_2>"],
  "reference_answer": "References on LLM eval include <...>.",
  "notes": "hand-authored cross-document reference lookup"
}
```

- [ ] **Step 5: Save the curated set and validate it loads**

Save the final array to `src/knowledge_service/eval/golden.json`, then:

Run: `uv run python -c "from pathlib import Path; from knowledge_service.eval.golden import load_golden; print(len(load_golden(Path('src/knowledge_service/eval/golden.json'))))"`
Expected: prints the item count (50–75) with no validation error.

- [ ] **Step 6: Commit**

```bash
git add scripts/gen_golden_candidates.py src/knowledge_service/eval/golden.json
git commit -m "feat(eval): golden query set (auto-curated + hand-authored GTD)"
```

---

## Task 10: Gitignore + run docs

**Files:**
- Modify: `.gitignore`
- Create: `src/knowledge_service/eval/README.md`

- [ ] **Step 1: Ignore reports + snapshot + candidate files**

Append to `.gitignore`:

```
# Eval harness outputs + local snapshot
src/knowledge_service/eval/reports/
data/snapshot/
golden_candidates.json
```

- [ ] **Step 2: Write the run doc**

Create `src/knowledge_service/eval/README.md`:

````markdown
# KS Eval Harness

Measures retrieval + answer quality and compares graph-on (`full`) vs graph-off
(`chunks_only`). Spec: `docs/superpowers/specs/2026-05-30-ks-eval-harness-design.md`.

## One-time: build the snapshot

```bash
# 1. Copy the prod oxigraph data dir locally (offline dump needs a copy, not the live store).
# 2. Export Postgres + oxigraph to ./data/snapshot:
KS_PROD_DATABASE_URL=postgresql://... \
KS_PROD_OXIGRAPH_DIR=/path/to/oxigraph-copy \
python scripts/export_prod_snapshot.py

# 3. Restore Postgres locally and load oxigraph N-Quads into a local store dir:
createdb knowledge_eval
pg_restore --no-owner --dbname "postgresql://localhost/knowledge_eval" data/snapshot/knowledge.dump
python -c "import pyoxigraph as o; s=o.Store('./data/oxigraph-eval'); s.bulk_load(open('data/snapshot/oxigraph.nq','rb'), 'application/n-quads')"
```

## Build the golden set

```bash
python scripts/gen_golden_candidates.py --per-type 10 --out golden_candidates.json
# curate by hand -> src/knowledge_service/eval/golden.json (see plan Task 9)
```

## Run

```bash
export DATABASE_URL=postgresql://localhost/knowledge_eval
export OXIGRAPH_DATA_DIR=./data/oxigraph-eval
export EVAL_JUDGE_API_KEY=...   # Anthropic key for the judge
uv run python -m knowledge_service.eval --modes full,chunks_only --k 5 --timestamp 2026-05-30T1200
```

Prints a `mode × query_type` summary table and writes
`src/knowledge_service/eval/reports/<timestamp>.json`.

## Reading the result

Graph-on "does no harm" if `full ≥ chunks_only` (within run-to-run noise) on
aggregate faithfulness/correctness. Per-query-type rows show where `full` clearly
wins — candidates for intent-based routing rather than cutting.
````

- [ ] **Step 3: Commit**

```bash
git add .gitignore src/knowledge_service/eval/README.md
git commit -m "docs(eval): gitignore reports/snapshot + run instructions"
```

---

## Task 11: Full suite + lint gate

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: PASS (all tests, including new `tests/eval/`; e2e remains ignored per pyproject).

- [ ] **Step 2: Lint + format check**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: no errors. If format check fails, run `uv run ruff format .` and re-commit.

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -A
git commit -m "chore(eval): lint + format"
```

---

## Self-Review (completed against the spec)

**Spec coverage:**
- `retrieval_mode` flag (full|chunks_only), threaded through `/api/ask` → Tasks 1–2. ✓
- Layered metrics (recall@k, precision@k, MRR, nDCG) + triple-contribution stat → Task 3 + `triples_surfaced` in Task 7's `score_query`. ✓
- Claude-as-judge faithfulness + correctness, key from env, fail-clear on missing key → Task 5. ✓
- Golden set: hybrid auto-curated + hand-authored GTD, validated schema → Tasks 4 + 9. ✓
- Corpus snapshot: pg_dump + oxigraph N-Quads + manifest, env-only creds, oxigraph open-question resolved (offline dump from a copy) → Task 8 + README restore steps. ✓
- Run flow + report + summary table broken down by mode × query_type → Tasks 6–7. ✓
- Harness unit tests in CI; corpus run manual like e2e → Tasks 3–7 tests + Task 11. ✓
- YAGNI non-goals (no dashboard, no full-run CI, no HTTP arm, no aegis changes) → respected. ✓
- "Do no harm" decision rule documented for the reader → Task 10 README + spec. ✓

**Placeholder scan:** No TBD/TODO work items. The `"CURATE ME"` string in Task 9 is intentional data-workflow guidance written into generated candidate files, not a code placeholder.

**Type consistency:** `RetrievalContext(content_results=...)` content-only construction (Task 1) matches its dataclass defaults. `QueryResult(query_id, mode, query_type, metrics, answer, rationale)` defined in Task 6, constructed identically in Task 7. `JudgeScore(faithfulness, correctness, rationale)` consistent across Tasks 5 and 7. `GoldenItem` fields consistent across Tasks 4, 7, 9. `score_query(...)` signature matches its Task 7 test. Standalone wiring in `_build_components` uses verified signatures: `RAGClient` from `clients.rag`, `EntityStore(pool, embedding_client)`, `ContentStore(pool, exclude_inflight=...)`, `bootstrap_ontology(store, ontology_dir)`, `TripleStore.flush()`. CLI `--timestamp` required (no clock call) for reproducibility.
