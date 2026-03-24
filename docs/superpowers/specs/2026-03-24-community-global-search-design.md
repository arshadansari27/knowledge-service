# Community Detection and Global Search

**Date:** 2026-03-24
**Phase:** 9 of 9 (KG-RAG improvement roadmap — final phase)
**Scope:** Leiden community detection over the entity graph, LLM-summarized communities, global search strategy, knowledge gap detection, and rebuild triggers

---

## Context

The system currently handles semantic, entity, and graph queries well, but cannot answer corpus-level questions like "what are the main themes in my knowledge base?" or "summarize what I know about health topics." These require understanding the structure of the entire knowledge graph — which entities cluster together and what those clusters represent.

Microsoft GraphRAG's key insight: pre-compute entity communities with summaries, then use those summaries to answer global questions in a map-reduce style.

**Dependencies:** Phase 8 (multi-hop retrieval) provides the graph traversal infrastructure. Phase 7 (query routing) provides the intent classification that routes global queries.

---

## Design

### Community Detection

New module `src/knowledge_service/stores/community.py`.

#### CommunityDetector

Synchronous class — called via `asyncio.to_thread`.

**Algorithm:**
1. Extract entity graph from pyoxigraph: SPARQL query for all entity-to-entity triples across named graphs (skip literal-valued objects)
2. Build `igraph.Graph` from edges, weighted by triple confidence
3. Run Leiden algorithm (`igraph.Graph.community_leiden()`) at 2 resolution levels:
   - Level 0 (resolution=1.0): fine-grained clusters — tight topic groups
   - Level 1 (resolution=0.5): coarse clusters — broad themes
4. Return community assignments as `list[Community]`

```python
@dataclass
class Community:
    level: int                    # 0 = fine, 1 = coarse
    member_entities: list[str]    # entity URIs
    member_count: int
```

**New dependency:** `igraph` (add to `pyproject.toml` dependencies)

**Graph extraction SPARQL:**
```sparql
SELECT DISTINCT ?s ?o ?conf WHERE {
    GRAPH ?g {
        ?s ?p ?o .
    }
    OPTIONAL {
        GRAPH ?g {
            << ?s ?p ?o >> <ks:confidence> ?conf .
        }
    }
    FILTER(isIRI(?o))
    FILTER(BOUND(?conf))
}
```

This gives all entity-to-entity edges with confidence weights.

#### CommunitySummarizer

For each community, gather context and call the LLM to generate a label and summary:

1. Collect member entity URIs
2. For each member (up to 10), get top 5 triples via `get_triples_by_subject`
3. For each member (up to 5), get top 2 content chunks via entity embedding similarity
4. Build a summarization prompt:
   ```
   Given these entities and their relationships, generate:
   1. A short label (3-5 words) for this community's theme
   2. A 2-3 sentence summary of what this community represents

   Entities: [entity1, entity2, ...]
   Relationships: [entity1 -> causes -> entity2, ...]
   Supporting text: [chunk excerpts...]

   Return JSON: {"label": "...", "summary": "..."}
   ```
5. Store label + summary in `communities` table

Uses the existing LLM endpoint (qwen3:14b via LiteLLM). Reuses the HTTP client pattern from `ExtractionClient`.

### PostgreSQL Storage

New migration `007_communities.sql`:

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

Communities are **fully replaced on each rebuild**: delete all rows, insert new ones. No incremental updates — full rebuild is simple, correct, and fast at the expected scale.

#### CommunityStore

New class in `community.py`. Asyncpg-backed, same pattern as `ProvenanceStore`:

```python
class CommunityStore:
    async def replace_all(self, communities: list[dict]) -> int
    async def get_by_level(self, level: int) -> list[dict]
    async def get_all(self) -> list[dict]
    async def get_member_entities(self) -> set[str]
```

### Global Search Strategy

#### New query intent: `global`

Extend `QueryClassifier` to recognize a 4th intent:

| Intent | Example questions |
|---|---|
| `semantic` | "find articles about stress management" |
| `entity` | "what is dopamine?" |
| `graph` | "how is cortisol connected to inflammation?" |
| `global` | "what are the main themes?", "summarize what I know about health", "what topics have I collected?" |

Update `_VALID_INTENTS` to include `"global"`. Update the classification prompt with the new category and examples.

#### `_retrieve_global()` strategy in RAGRetriever

New strategy method triggered when intent is `global`:

1. Query `communities` table via `CommunityStore.get_all()`
2. For level 1 (coarse) communities: include full summaries in context
3. For level 0 (fine) communities matching the question: include summaries
   - Match by checking if any community member entity names appear in the question
   - Or if none match, include top 5 level-0 communities by member count
4. Light hybrid search (top 3 chunks) for grounding
5. Return `RetrievalContext` with community summaries as knowledge triples:
   ```python
   {
       "subject": f"community_{community_id}",
       "predicate": "has_summary",
       "object": community_summary,
       "confidence": 1.0,
       "knowledge_type": "Community",
       "trust_tier": "computed",
   }
   ```

Community summaries flow through the existing RAG prompt as knowledge triples with `trust_tier: "computed"`. The LLM sees them alongside any content chunks. No prompt changes needed.

#### RAGRetriever changes

- Constructor gains `community_store` parameter (optional, for tests without communities)
- `retrieve()` dispatch gains `global` case → `_retrieve_global()`
- Fallback: if no communities exist (table empty), fall back to `semantic` strategy

### Rebuild Triggers

#### Admin endpoint: `POST /api/admin/rebuild-communities`

New route in admin module. Triggers full community rebuild:

1. `CommunityDetector.detect(knowledge_store)` → community assignments
2. `CommunitySummarizer.summarize(communities, knowledge_store, embedding_store, llm_client)` → labels + summaries
3. `CommunityStore.replace_all(communities_with_summaries)`
4. Return `{"communities_built": N, "levels": {"level_0": X, "level_1": Y}, "duration_seconds": Z}`

Requires admin auth (existing session/API-key middleware).

#### Periodic rebuild (optional)

Config setting: `COMMUNITY_REBUILD_INTERVAL` (integer seconds, default `0` = disabled).

When > 0, a background `asyncio.Task` starts in `lifespan()`:

```python
async def _community_rebuild_loop(app, interval):
    while True:
        await asyncio.sleep(interval)
        try:
            await rebuild_communities(app)
        except Exception as exc:
            logger.warning("Community rebuild failed: %s", exc)
```

Started in `lifespan()` if interval > 0. Cancelled on shutdown. Errors logged, never crash the app.

### Knowledge Gap Detection

New admin endpoint: `GET /api/admin/stats/gaps`

Returns:
```json
{
    "isolated_entities": ["http://...uri1", "http://...uri2"],
    "thin_communities": [{"id": "...", "label": "...", "member_count": 1}],
    "total_entities": 50,
    "entities_in_communities": 42,
    "community_coverage": 0.84
}
```

Computed by comparing all entity URIs (from `entity_embeddings` table) against community member lists. Entities not in any community are "isolated." Communities with <=2 members are "thin."

---

## File changes summary

| File | Change |
|------|--------|
| `src/knowledge_service/stores/community.py` | NEW: CommunityDetector + CommunitySummarizer + CommunityStore |
| `src/knowledge_service/clients/classifier.py` | Add `global` to valid intents and classification prompt |
| `src/knowledge_service/stores/rag.py` | Add `_retrieve_global()` strategy, `community_store` param |
| `src/knowledge_service/admin/routes.py` | Add rebuild-communities endpoint |
| `src/knowledge_service/admin/stats.py` | Add gaps endpoint |
| `src/knowledge_service/api/ask.py` | Pass community_store to retriever |
| `src/knowledge_service/main.py` | Initialize CommunityStore, optional rebuild loop |
| `src/knowledge_service/config.py` | Add `community_rebuild_interval` setting |
| `migrations/007_communities.sql` | Communities table |
| `pyproject.toml` | Add `igraph` dependency |
| `tests/test_community.py` | NEW: detection + store + summarizer tests |
| `tests/test_classifier.py` | Add global intent test |
| `tests/test_rag_retriever.py` | Global strategy tests |

## Constraints

- Leiden runs synchronously via `asyncio.to_thread` (igraph is not async)
- Community summaries require 1 LLM call per community — for N communities, N calls. At small scale this is fast; at 100+ communities, rebuild takes minutes
- Communities are ephemeral — fully replaced on rebuild, no history
- `global` intent fallback: if no communities exist, uses `semantic` strategy
- Periodic rebuild is opt-in (disabled by default)
- Max 2 hierarchy levels (fine + coarse) — more levels add complexity without clear value at this scale

## Tests

- Test Leiden produces communities from a known graph
- Test community store: replace_all, get_by_level, get_all
- Test summarizer generates label + summary from entity context
- Test classifier recognizes global intent for thematic questions
- Test `_retrieve_global` uses community summaries as knowledge triples
- Test `_retrieve_global` falls back to semantic when no communities exist
- Test admin rebuild endpoint triggers full rebuild
- Test gaps endpoint returns isolated entities and thin communities
- Test periodic rebuild loop starts when interval > 0
- Test periodic rebuild loop does not start when interval = 0
- Backward compat: existing intents (semantic, entity, graph) unchanged
