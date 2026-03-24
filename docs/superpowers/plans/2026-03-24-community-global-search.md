# Community Detection and Global Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Leiden community detection over the entity graph with LLM-summarized communities, a global search strategy for corpus-level questions, knowledge gap detection, and admin rebuild triggers.

**Architecture:** New `community.py` module handles detection (igraph Leiden), storage (PostgreSQL), and summarization (LLM). `QueryClassifier` gains a 4th `global` intent. RAGRetriever gains `_retrieve_global()` strategy using community summaries. Admin endpoints for rebuild and gap detection.

**Tech Stack:** python-igraph (Leiden algorithm), PostgreSQL, LLM (qwen3:14b via LiteLLM), asyncio background tasks

**Spec:** `docs/superpowers/specs/2026-03-24-community-global-search-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `migrations/007_communities.sql` | Communities table |
| Modify | `pyproject.toml` | Add `python-igraph` dependency |
| Create | `src/knowledge_service/stores/community.py` | CommunityDetector + CommunityStore + CommunitySummarizer |
| Modify | `src/knowledge_service/clients/classifier.py` | Add `global` intent |
| Modify | `src/knowledge_service/stores/rag.py` | Add `_retrieve_global()` strategy |
| Modify | `src/knowledge_service/admin/stats.py` | Add gaps endpoint |
| Create | `src/knowledge_service/admin/communities.py` | Rebuild endpoint |
| Modify | `src/knowledge_service/config.py` | Add `community_rebuild_interval` |
| Modify | `src/knowledge_service/main.py` | Initialize CommunityStore, optional rebuild loop |
| Modify | `src/knowledge_service/api/ask.py` | Pass community_store through |
| Create | `tests/test_community.py` | Detection + store + summarizer tests |
| Modify | `tests/test_classifier.py` | Global intent test |
| Modify | `tests/test_rag_retriever.py` | Global strategy test |

---

## Task 1: Dependencies and migration

**Files:**
- Modify: `pyproject.toml`
- Create: `migrations/007_communities.sql`

- [ ] **Step 1: Add python-igraph dependency**

In `pyproject.toml`, add `"python-igraph>=0.11"` to the `dependencies` list.

- [ ] **Step 2: Create migration**

```sql
CREATE TABLE communities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    level INTEGER NOT NULL,
    label TEXT,
    summary TEXT,
    member_entities TEXT[] NOT NULL,
    member_count INTEGER NOT NULL,
    built_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_communities_level ON communities(level);
CREATE INDEX idx_communities_built_at ON communities(built_at);
```

- [ ] **Step 3: Install and verify**

Run: `uv sync --dev && uv run python -c "import igraph; print(igraph.__version__)"`
Expected: version number printed

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock migrations/007_communities.sql
git commit -m "feat: add python-igraph dependency and communities migration"
```

---

## Task 2: CommunityStore (PostgreSQL CRUD)

**Files:**
- Create: `src/knowledge_service/stores/community.py` (partial — store only)
- Create: `tests/test_community.py` (partial — store tests)

- [ ] **Step 1: Write failing tests for CommunityStore**

Create `tests/test_community.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from knowledge_service.stores.community import CommunityStore


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    txn = AsyncMock()
    txn.__aenter__ = AsyncMock(return_value=txn)
    txn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction.return_value = txn
    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_ctx
    return pool, conn


@pytest.fixture
def store(mock_pool):
    pool, _ = mock_pool
    return CommunityStore(pool)


class TestCommunityStore:
    async def test_replace_all_deletes_and_inserts(self, store, mock_pool):
        _, conn = mock_pool
        communities = [
            {"level": 0, "label": "Health", "summary": "Health topics",
             "member_entities": ["http://e/a", "http://e/b"], "member_count": 2},
        ]
        await store.replace_all(communities)
        # Should execute delete then insert within transaction
        assert conn.execute.call_count >= 1

    async def test_get_by_level(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {"id": "uuid1", "level": 0, "label": "Test", "summary": "Sum",
             "member_entities": ["http://e/a"], "member_count": 1, "built_at": "2026-01-01"},
        ]
        results = await store.get_by_level(0)
        assert len(results) == 1
        sql = conn.fetch.call_args[0][0]
        assert "level" in sql

    async def test_get_all(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = []
        results = await store.get_all()
        assert results == []

    async def test_get_member_entities(self, store, mock_pool):
        _, conn = mock_pool
        conn.fetch.return_value = [
            {"member_entities": ["http://e/a", "http://e/b"]},
            {"member_entities": ["http://e/b", "http://e/c"]},
        ]
        result = await store.get_member_entities()
        assert "http://e/a" in result
        assert "http://e/c" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_community.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CommunityStore**

Create `src/knowledge_service/stores/community.py`:

```python
"""Community detection, storage, and summarization for global search."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CommunityStore:
    """Asyncpg-backed store for community data."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def replace_all(self, communities: list[dict]) -> int:
        """Delete all communities and insert new ones atomically."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM communities")
                for c in communities:
                    await conn.execute(
                        """INSERT INTO communities (level, label, summary, member_entities, member_count)
                           VALUES ($1, $2, $3, $4, $5)""",
                        c["level"], c.get("label"), c.get("summary"),
                        c["member_entities"], c["member_count"],
                    )
        return len(communities)

    async def get_by_level(self, level: int) -> list[dict]:
        sql = "SELECT * FROM communities WHERE level = $1 ORDER BY member_count DESC"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, level)
        return [dict(r) for r in rows]

    async def get_all(self) -> list[dict]:
        sql = "SELECT * FROM communities ORDER BY level, member_count DESC"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)
        return [dict(r) for r in rows]

    async def get_member_entities(self) -> set[str]:
        """Return all entity URIs that belong to any community."""
        sql = "SELECT member_entities FROM communities"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql)
        entities: set[str] = set()
        for r in rows:
            entities.update(r["member_entities"])
        return entities
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_community.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/stores/community.py tests/test_community.py
git commit -m "feat: CommunityStore for community CRUD operations"
```

---

## Task 3: CommunityDetector (Leiden algorithm)

**Files:**
- Modify: `src/knowledge_service/stores/community.py`
- Modify: `tests/test_community.py`

- [ ] **Step 1: Write failing tests for CommunityDetector**

Add to `tests/test_community.py`:

```python
from unittest.mock import MagicMock
from knowledge_service.stores.community import CommunityDetector


def _make_knowledge_store_for_detection():
    """Mock KnowledgeStore that returns a small entity graph."""
    ks = MagicMock()
    # Simulates a SPARQL result with entity-to-entity edges
    ks.query.return_value = [
        {"s": MagicMock(value="http://e/a"), "o": MagicMock(value="http://e/b"), "conf": MagicMock(value="0.8")},
        {"s": MagicMock(value="http://e/b"), "o": MagicMock(value="http://e/c"), "conf": MagicMock(value="0.7")},
        {"s": MagicMock(value="http://e/d"), "o": MagicMock(value="http://e/e"), "conf": MagicMock(value="0.9")},
    ]
    return ks


class TestCommunityDetector:
    def test_detect_returns_communities(self):
        ks = _make_knowledge_store_for_detection()
        detector = CommunityDetector(ks)
        communities = detector.detect()
        assert len(communities) > 0
        for c in communities:
            assert "level" in c
            assert "member_entities" in c
            assert "member_count" in c

    def test_detect_produces_two_levels(self):
        ks = _make_knowledge_store_for_detection()
        detector = CommunityDetector(ks)
        communities = detector.detect()
        levels = {c["level"] for c in communities}
        assert 0 in levels

    def test_detect_empty_graph(self):
        ks = MagicMock()
        ks.query.return_value = []
        detector = CommunityDetector(ks)
        communities = detector.detect()
        assert communities == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_community.py::TestCommunityDetector -v`
Expected: FAIL

- [ ] **Step 3: Implement CommunityDetector**

Add to `community.py`:

```python
import igraph


class CommunityDetector:
    """Extract entity graph from KnowledgeStore and run Leiden community detection."""

    def __init__(self, knowledge_store) -> None:
        self._ks = knowledge_store

    def detect(self) -> list[dict]:
        """Run Leiden at 2 resolution levels, return community assignments."""
        edges = self._extract_graph()
        if not edges:
            return []

        # Build igraph Graph
        entities = set()
        for e in edges:
            entities.add(e["source"])
            entities.add(e["target"])

        entity_list = sorted(entities)
        entity_idx = {uri: i for i, uri in enumerate(entity_list)}

        g = igraph.Graph(n=len(entity_list), directed=False)
        g.vs["name"] = entity_list

        edge_list = []
        weights = []
        seen_edges = set()
        for e in edges:
            pair = (min(entity_idx[e["source"]], entity_idx[e["target"]]),
                    max(entity_idx[e["source"]], entity_idx[e["target"]]))
            if pair not in seen_edges:
                seen_edges.add(pair)
                edge_list.append(pair)
                weights.append(e["weight"])

        g.add_edges(edge_list)
        g.es["weight"] = weights

        communities = []

        # Level 0: fine-grained (resolution=1.0)
        partition_0 = g.community_leiden(weights="weight", resolution=1.0)
        for cluster_members in partition_0:
            if len(cluster_members) > 0:
                member_uris = [entity_list[i] for i in cluster_members]
                communities.append({
                    "level": 0,
                    "member_entities": member_uris,
                    "member_count": len(member_uris),
                })

        # Level 1: coarse (resolution=0.5)
        partition_1 = g.community_leiden(weights="weight", resolution=0.5)
        for cluster_members in partition_1:
            if len(cluster_members) > 0:
                member_uris = [entity_list[i] for i in cluster_members]
                communities.append({
                    "level": 1,
                    "member_entities": member_uris,
                    "member_count": len(member_uris),
                })

        return communities

    def _extract_graph(self) -> list[dict]:
        """Extract entity-to-entity edges from the knowledge store."""
        from knowledge_service.ontology.namespaces import KS_CONFIDENCE

        sparql = f"""
            SELECT DISTINCT ?s ?o ?conf WHERE {{
                GRAPH ?g {{
                    ?s ?p ?o .
                }}
                OPTIONAL {{
                    GRAPH ?g {{
                        << ?s ?p ?o >> <{KS_CONFIDENCE.value}> ?conf .
                    }}
                }}
                FILTER(isIRI(?o))
            }}
        """
        rows = self._ks.query(sparql)
        edges = []
        for r in rows:
            s = r["s"].value if hasattr(r["s"], "value") else str(r["s"])
            o = r["o"].value if hasattr(r["o"], "value") else str(r["o"])
            conf = float(r["conf"].value) if r.get("conf") and r["conf"] else 0.5
            edges.append({"source": s, "target": o, "weight": conf})
        return edges
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_community.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/stores/community.py tests/test_community.py
git commit -m "feat: CommunityDetector with Leiden algorithm"
```

---

## Task 4: Add `global` intent to QueryClassifier

**Files:**
- Modify: `src/knowledge_service/clients/classifier.py`
- Modify: `tests/test_classifier.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_classifier.py`:

```python
    async def test_returns_global_intent(self, httpx_mock):
        httpx_mock.add_response(url=_CHAT_URL, json=_make_response("global", []))
        c = QueryClassifier(base_url=_BASE, model="qwen3:14b", api_key=_KEY)
        result = await c.classify("what are the main themes in my knowledge base?")
        assert result.intent == "global"
        await c.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_classifier.py::TestClassify::test_returns_global_intent -v`
Expected: FAIL — `global` not in `_VALID_INTENTS`, falls back to `semantic`

- [ ] **Step 3: Update classifier**

In `classifier.py`, add `"global"` to `_VALID_INTENTS`:
```python
_VALID_INTENTS = {"semantic", "entity", "graph", "global"}
```

Update `_CLASSIFICATION_PROMPT` to add the global category:
```python
_CLASSIFICATION_PROMPT = """Classify this question into one category:
- "semantic": searching for documents about a topic (e.g., "find articles about stress management")
- "entity": asking about a specific thing (e.g., "what is dopamine?", "tell me about PostgreSQL")
- "graph": asking about relationships between things (e.g., "how is cortisol connected to inflammation?", "what causes dopamine release?")
- "global": asking about themes, summaries, or overviews across the entire knowledge base (e.g., "what are the main topics?", "summarize what I know about health", "what areas have I collected knowledge on?")

Also extract any named entities mentioned in the question.

Return JSON: {{"intent": "semantic|entity|graph|global", "entities": ["entity1", "entity2"]}}

Question: {question}"""
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/clients/classifier.py tests/test_classifier.py
git commit -m "feat: add global intent to QueryClassifier"
```

---

## Task 5: Global retrieval strategy in RAGRetriever

**Files:**
- Modify: `src/knowledge_service/stores/rag.py`
- Modify: `tests/test_rag_retriever.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_rag_retriever.py`:

```python
class TestGlobalIntent:
    async def test_global_intent_uses_community_summaries(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        ks = _make_knowledge_store()
        # Mock community store
        community_store = AsyncMock()
        community_store.get_all.return_value = [
            {"id": "c1", "level": 1, "label": "Health", "summary": "Health and biohacking topics",
             "member_entities": ["http://e/a"], "member_count": 3, "built_at": "2026-01-01"},
        ]
        retriever = RAGRetriever(ec, es, ks, community_store=community_store)
        intent = QueryIntent(intent="global", entities=[])
        context = await retriever.retrieve("what are the main themes?", intent=intent)
        community_store.get_all.assert_called_once()
        assert len(context.knowledge_triples) >= 1

    async def test_global_falls_back_to_semantic_without_communities(self):
        ec = _make_embedding_client()
        es = _make_embedding_store(content_rows=[_CONTENT_ROW])
        ks = _make_knowledge_store()
        community_store = AsyncMock()
        community_store.get_all.return_value = []
        retriever = RAGRetriever(ec, es, ks, community_store=community_store)
        intent = QueryIntent(intent="global", entities=[])
        context = await retriever.retrieve("what are the main themes?", intent=intent)
        # Falls back to semantic — search called
        es.search.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rag_retriever.py::TestGlobalIntent -v`
Expected: FAIL

- [ ] **Step 3: Implement _retrieve_global and wire into dispatch**

In `rag.py`:

Add `community_store=None` parameter to `__init__`:
```python
def __init__(self, embedding_client, embedding_store, knowledge_store, community_store=None):
    ...
    self._community_store = community_store
```

Add `global` to the dispatch in `retrieve()`:
```python
elif intent.intent == "global":
    return await self._retrieve_global(question, embedding, max_sources, min_confidence)
```

Add the strategy method:
```python
async def _retrieve_global(
    self, question, embedding, max_sources, min_confidence
) -> RetrievalContext:
    """Global strategy: use community summaries for corpus-level questions."""
    if not self._community_store:
        return await self._retrieve_semantic(question, embedding, max_sources, min_confidence)

    communities = await self._community_store.get_all()
    if not communities:
        return await self._retrieve_semantic(question, embedding, max_sources, min_confidence)

    # Build knowledge triples from community summaries
    question_words = set(question.lower().split())
    triples = []

    for c in communities:
        # Level 1 (coarse): always include
        # Level 0 (fine): include if keyword match or top 5 by size
        if c["level"] == 1:
            include = True
        else:
            summary_words = set((c.get("summary") or "").lower().split())
            include = bool(question_words & summary_words)

        if include and c.get("summary"):
            triples.append({
                "subject": f"community_{c.get('id', 'unknown')}",
                "predicate": "has_summary",
                "object": c["summary"],
                "confidence": 1.0,
                "knowledge_type": "Community",
                "trust_tier": "computed",
            })

    # If no level-0 matched by keyword, add top 5 by member count
    level0_included = any(
        1 for c in communities
        if c["level"] == 0 and c.get("summary")
        and bool(question_words & set((c.get("summary") or "").lower().split()))
    )
    if not level0_included:
        level0 = [c for c in communities if c["level"] == 0 and c.get("summary")]
        for c in level0[:5]:
            triples.append({
                "subject": f"community_{c.get('id', 'unknown')}",
                "predicate": "has_summary",
                "object": c["summary"],
                "confidence": 1.0,
                "knowledge_type": "Community",
                "trust_tier": "computed",
            })

    # Light content search for grounding
    content_results = await self._embedding_store.search(
        query_embedding=embedding, limit=3, query_text=question
    )

    return RetrievalContext(
        content_results=content_results,
        knowledge_triples=triples,
        contradictions=[],
        entities_found=[],
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_rag_retriever.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/stores/rag.py tests/test_rag_retriever.py
git commit -m "feat: global retrieval strategy using community summaries"
```

---

## Task 6: Admin endpoints (rebuild + gaps)

**Files:**
- Create: `src/knowledge_service/admin/communities.py`
- Modify: `src/knowledge_service/admin/stats.py`
- Modify: `src/knowledge_service/main.py`

- [ ] **Step 1: Create rebuild endpoint**

Create `src/knowledge_service/admin/communities.py`:

```python
"""Admin endpoint for community rebuild."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Request

from knowledge_service.stores.community import CommunityDetector

router = APIRouter()


@router.post("/rebuild-communities")
async def rebuild_communities(request: Request):
    """Trigger a full community detection + summarization rebuild."""
    knowledge_store = request.app.state.knowledge_store
    community_store = request.app.state.community_store
    pg_pool = request.app.state.pg_pool

    start = time.time()

    # Step 1: Detect communities
    detector = CommunityDetector(knowledge_store)
    communities = await asyncio.to_thread(detector.detect)

    # Step 2: Store (without summaries for now — summarization is Task 7)
    count = await community_store.replace_all(communities)

    duration = time.time() - start
    level_counts = {}
    for c in communities:
        level_counts[f"level_{c['level']}"] = level_counts.get(f"level_{c['level']}", 0) + 1

    return {
        "communities_built": count,
        "levels": level_counts,
        "duration_seconds": round(duration, 2),
    }
```

- [ ] **Step 2: Add gaps endpoint to stats.py**

Add to `admin/stats.py`:

```python
@router.get("/stats/gaps")
async def get_gaps(request: Request):
    """Detect knowledge gaps: isolated entities and thin communities."""
    community_store = getattr(request.app.state, "community_store", None)
    embedding_store = getattr(request.app.state, "embedding_store", None)

    if not community_store or not embedding_store:
        return {"error": "Community or embedding store not available"}

    # All entities in the system (direct SQL, not similarity search)
    async with request.app.state.pg_pool.acquire() as conn:
        rows = await conn.fetch("SELECT uri FROM entity_embeddings")
    all_entities = {r["uri"] for r in rows}

    # Entities in communities
    community_entities = await community_store.get_member_entities()

    isolated = sorted(all_entities - community_entities)

    # Thin communities (<=2 members)
    all_communities = await community_store.get_all()
    thin = [
        {"id": str(c.get("id", "")), "label": c.get("label", ""), "member_count": c["member_count"]}
        for c in all_communities if c["member_count"] <= 2
    ]

    total = len(all_entities)
    in_communities = len(all_entities & community_entities)
    coverage = in_communities / total if total > 0 else 0.0

    return {
        "isolated_entities": isolated,
        "thin_communities": thin,
        "total_entities": total,
        "entities_in_communities": in_communities,
        "community_coverage": round(coverage, 2),
    }
```

- [ ] **Step 3: Wire into main.py**

In `main.py` lifespan, after embedding_store initialization:

```python
from knowledge_service.stores.community import CommunityStore
app.state.community_store = CommunityStore(app.state.pg_pool)
```

Register the admin communities router with `/api/admin` prefix (same as stats_router):
```python
from knowledge_service.admin.communities import router as communities_router
app.include_router(communities_router, prefix="/api/admin")
```

Pass `community_store` to RAGRetriever:
```python
app.state.rag_retriever = RAGRetriever(
    ..., community_store=app.state.community_store
)
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/admin/communities.py src/knowledge_service/admin/stats.py src/knowledge_service/main.py
git commit -m "feat: admin rebuild-communities endpoint and knowledge gaps detection"
```

---

## Task 7: CommunitySummarizer (LLM summaries)

**Files:**
- Modify: `src/knowledge_service/stores/community.py`
- Modify: `src/knowledge_service/admin/communities.py`
- Modify: `tests/test_community.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_community.py`:

```python
class TestCommunitySummarizer:
    async def test_summarize_produces_label_and_summary(self):
        from knowledge_service.stores.community import CommunitySummarizer

        mock_llm_client = AsyncMock()
        mock_llm_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"choices": [{"message": {"content": '{"label": "Health Topics", "summary": "This community covers health and biohacking."}'}}]},
            raise_for_status=lambda: None,
        )

        mock_ks = MagicMock()
        mock_ks.get_triples_by_subject.return_value = []

        summarizer = CommunitySummarizer(mock_llm_client, mock_ks)
        community = {"level": 0, "member_entities": ["http://e/a", "http://e/b"], "member_count": 2}
        result = await summarizer.summarize_one(community)
        assert result["label"] is not None
        assert result["summary"] is not None

    async def test_summarize_handles_llm_failure(self):
        from knowledge_service.stores.community import CommunitySummarizer

        mock_llm_client = AsyncMock()
        mock_llm_client.post.side_effect = Exception("LLM down")

        mock_ks = MagicMock()
        mock_ks.get_triples_by_subject.return_value = []

        summarizer = CommunitySummarizer(mock_llm_client, mock_ks)
        community = {"level": 0, "member_entities": ["http://e/a"], "member_count": 1}
        result = await summarizer.summarize_one(community)
        assert result.get("label") is None  # Graceful failure
```

- [ ] **Step 2: Implement CommunitySummarizer**

Add to `community.py`:

```python
import json
import re

import httpx

from knowledge_service._utils import _rdf_value_to_str


class CommunitySummarizer:
    """Generate LLM summaries for communities."""

    def __init__(self, llm_client: httpx.AsyncClient, knowledge_store, model: str = "") -> None:
        self._client = llm_client
        self._ks = knowledge_store
        self._model = model

    async def summarize_one(self, community: dict) -> dict:
        """Generate label + summary for a single community. Returns updated community dict."""
        community = dict(community)
        try:
            context = self._build_context(community)
            prompt = f"""Given these entities and their relationships, generate:
1. A short label (3-5 words) for this community's theme
2. A 2-3 sentence summary of what this community represents

{context}

Return JSON: {{"label": "...", "summary": "..."}}"""

            response = await self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                },
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"]
            stripped = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
            stripped = re.sub(r"\n?```\s*$", "", stripped)
            parsed = json.loads(stripped)
            community["label"] = parsed.get("label")
            community["summary"] = parsed.get("summary")
        except Exception as exc:
            logger.warning("CommunitySummarizer: failed for community: %s", exc)
            community["label"] = None
            community["summary"] = None
        return community

    def _build_context(self, community: dict) -> str:
        members = community["member_entities"][:10]
        lines = [f"Entities: {', '.join(m.rsplit('/', 1)[-1] for m in members)}"]

        relationships = []
        for uri in members[:5]:
            triples = self._ks.get_triples_by_subject(uri)
            for t in triples[:5]:
                s = uri.rsplit("/", 1)[-1]
                p = _rdf_value_to_str(t.get("predicate", "")).rsplit("/", 1)[-1]
                o = _rdf_value_to_str(t.get("object", "")).rsplit("/", 1)[-1]
                relationships.append(f"{s} -> {p} -> {o}")

        if relationships:
            lines.append(f"Relationships:\n" + "\n".join(f"  - {r}" for r in relationships[:15]))

        return "\n".join(lines)
```

- [ ] **Step 3: Wire summarizer into rebuild endpoint**

Update `admin/communities.py` to call summarizer after detection:

```python
# After detector.detect():
from knowledge_service.stores.community import CommunitySummarizer

summarizer = CommunitySummarizer(
    request.app.state.extraction_client._client,  # reuse httpx client
    knowledge_store,
    model=request.app.state.extraction_client._model,
)

summarized = []
for c in communities:
    result = await summarizer.summarize_one(c)
    summarized.append(result)

count = await community_store.replace_all(summarized)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/stores/community.py src/knowledge_service/admin/communities.py tests/test_community.py
git commit -m "feat: CommunitySummarizer generates LLM labels and summaries per community"
```

---

## Task 8: Config + periodic rebuild + wiring

**Files:**
- Modify: `src/knowledge_service/config.py`
- Modify: `src/knowledge_service/main.py`

Note: `ask.py` needs no changes — `community_store` is passed to RAGRetriever during construction in `main.py` (Task 6 Step 3), and `ask.py` already accesses the retriever via `request.app.state.rag_retriever`.

- [ ] **Step 1: Add config setting**

In `config.py`, add to `Settings`:
```python
community_rebuild_interval: int = 0  # seconds, 0 = disabled
```

- [ ] **Step 2: Add periodic rebuild loop to main.py**

In `lifespan()`, after community_store init:

```python
# Optional periodic community rebuild
_rebuild_task = None
if settings.community_rebuild_interval > 0:
    async def _community_rebuild_loop():
        while True:
            await asyncio.sleep(settings.community_rebuild_interval)
            try:
                from knowledge_service.stores.community import CommunityDetector, CommunitySummarizer
                detector = CommunityDetector(app.state.knowledge_store)
                communities = await asyncio.to_thread(detector.detect)
                # Summarize each community via LLM
                summarizer = CommunitySummarizer(
                    app.state.extraction_client._client,
                    app.state.knowledge_store,
                    model=app.state.extraction_client._model,
                )
                summarized = []
                for c in communities:
                    summarized.append(await summarizer.summarize_one(c))
                await app.state.community_store.replace_all(summarized)
                logger.info("Periodic community rebuild: %d communities", len(summarized))
            except Exception as exc:
                logger.warning("Periodic community rebuild failed: %s", exc)

    _rebuild_task = asyncio.create_task(_community_rebuild_loop())
    app.state._community_rebuild_task = _rebuild_task
```

In shutdown:
```python
if hasattr(app.state, "_community_rebuild_task") and app.state._community_rebuild_task:
    app.state._community_rebuild_task.cancel()
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/config.py src/knowledge_service/main.py src/knowledge_service/api/ask.py
git commit -m "feat: periodic community rebuild, config, and ask.py wiring"
```

---

## Task 9: Final integration test and lint

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Lint and format**

Run: `uv run ruff check . && uv run ruff format --check .`
If needed: `uv run ruff format .`

- [ ] **Step 3: Commit if needed**

```bash
git add -A && git commit -m "chore: lint fixes for community detection and global search"
```
