# Foundation Improvements: ProbLog Rules, Named Graphs, Chunk Provenance

**Date:** 2026-03-21
**Phase:** 1-3 of 9 (low-effort foundation work)
**Scope:** Three independent improvements to reasoning, storage trust boundaries, and provenance granularity

---

## Context

Assessment of the knowledge-service against modern KG-RAG architectures identified 9 gaps. This spec covers the 3 lowest-effort, highest-foundation-value improvements that later phases build on.

**What this enables:**
- Phase 1 (ProbLog rules) → used by Phase 8 (multi-hop retrieval) and Phase 7 (query routing for rule questions)
- Phase 2 (Named graphs) → used by Phase 8 (trust-weighted traversal) and Phase 7 (routing by graph tier)
- Phase 3 (Chunk provenance) → enables "show me the evidence" in `/api/ask` responses

---

## Phase 1: Expanded ProbLog Rules

### Problem

The reasoning engine has 3 substantive rules (`contradicts`, `value_conflict`, `supported`) and a placeholder `expired` rule. The ProbLog infrastructure exists but is severely underused — no transitive reasoning, no temporal validity, no confidence-based resolution.

### Design

Add 12 domain-agnostic structural rules across 4 files. Rules reason about the *shape* of knowledge (chains, conflicts, temporal bounds), not domain content.

#### base.pl — add to existing rules

```prolog
% Inverse predicate inference: if A contains B, then B part_of A
inverse_holds(S, P2, O) :-
    claims(O, P1, S, _),
    inverse(P1, P2).

% Multi-source corroboration: claim from 2+ independent sources
corroborated(S, P, O) :-
    claims(S, P, O, Src1),
    claims(S, P, O, Src2),
    Src1 \= Src2.
```

#### inference_chains.pl — new file

```prolog
% Transitive link (bounded to 2-hop to prevent runaway)
indirect_link(A, P, C) :-
    claims(A, P, B, _),
    claims(B, P, C, _),
    A \= C.

% Cross-predicate causal chains
causal_propagation(A, C) :-
    claims(A, causes, B, _),
    claims(B, increases, C, _).

causal_propagation(A, C) :-
    claims(A, causes, B, _),
    claims(B, decreases, C, _).
```

#### confidence.pl — new file

```prolog
% High-confidence: supported and no value conflicts
high_confidence(S, P, O) :-
    supported(S, P, O),
    \+ value_conflict(S, P, _, _).

% Contested: supported but has conflicting values
contested(S, P, O) :-
    supported(S, P, O),
    value_conflict(S, P, O, _).

% Fact overrides claim when both exist
authoritative(S, P, O) :-
    claims(S, P, O, _),
    claim_type(S, P, O, fact).
```

#### temporal.pl — replace placeholder

```prolog
% Expired: valid_until has passed
expired(S, P, O) :-
    claims(S, P, O, _),
    valid_until(S, P, O, Until),
    current_date(Now),
    Now > Until.

% Currently valid: has temporal bounds and not expired
currently_valid(S, P, O) :-
    claims(S, P, O, _),
    valid_from(S, P, O, From),
    current_date(Now),
    Now >= From,
    \+ expired(S, P, O).

% Temporal supersedes: newer temporal state replaces older for same S-P
supersedes(S, P, O_new, O_old) :-
    claims(S, P, O_new, _),
    claims(S, P, O_old, _),
    valid_from(S, P, O_new, F1),
    valid_from(S, P, O_old, F2),
    F1 > F2,
    O_new \= O_old.
```

### Integration changes

#### ReasoningEngine.__init__ must load new rule files

The current `__init__` explicitly loads 3 files by name (`base.pl`, `knowledge_types.pl`, `temporal.pl`). Change to glob-load all `*.pl` files from `rules_dir`:

```python
def __init__(self, rules_dir: str | Path) -> None:
    self._rules_dir = Path(rules_dir)
    parts = []
    for pl_file in sorted(self._rules_dir.glob("*.pl")):
        parts.append(pl_file.read_text(encoding="utf-8"))
    self._all_rules: str = "\n".join(parts)
    self._base_rules: str = self._load_rules("base.pl")  # kept for _fallback_infer
```

#### New ontology property: `ks:inversePredicate`

`inverse/2` pairs are semantically different from `opposite/2` pairs. `opposite` means contradictory (increases vs decreases); `inverse` means mutually implied (contains vs part_of). Add `ks:inversePredicate` to `schema.ttl`:

```turtle
ks:inversePredicate rdf:type rdf:Property ; rdfs:label "inverse predicate" .
ks:contains    ks:inversePredicate ks:part_of .
ks:created_by  ks:inversePredicate ks:produces .  # future expansion
```

**schema.ttl correction:** The existing `ks:contains ks:oppositePredicate ks:part_of` triple is semantically wrong — contains/part_of are inverses, not opposites. Remove it and replace with `ks:contains ks:inversePredicate ks:part_of`.

The `inverse/2` ProbLog facts are emitted from `ks:inversePredicate` triples in `schema.ttl`, NOT from `ks:oppositePredicate`.

#### Extended `infer()` and `check_contradiction()` fact emission

To support `claim_type/4`, `valid_from/4`, `valid_until/4`, and `current_date/1`:

- Both `infer()` and `check_contradiction()` change to accept 5-tuples: `list[tuple[str, str, str, float, dict]]` where the dict contains optional `knowledge_type`, `valid_from`, `valid_until` fields. Existing callers pass empty dict `{}` for backward compat. `check_contradiction()` needs the metadata so that rules like `authoritative/3` (which checks `claim_type/4`) can fire during contradiction checks.
- Program assembly emits additional facts when metadata is present:
  ```python
  if meta.get("knowledge_type"):
      program_parts.append(f"claim_type({s}, {p}, {o}, {meta['knowledge_type']}).")
  if meta.get("valid_from"):
      program_parts.append(f"valid_from({s}, {p}, {o}, '{meta['valid_from']}').")
  if meta.get("valid_until"):
      program_parts.append(f"valid_until({s}, {p}, {o}, '{meta['valid_until']}').")
  ```
- `current_date/1` is injected as a dynamic fact at program assembly time:
  ```python
  from datetime import date
  program_parts.append(f"current_date('{date.today().isoformat()}').")
  ```
- Temporal date comparison uses ISO 8601 strings (`'2026-03-21'`). Prolog term ordering on strings is lexicographic, which works correctly for ISO dates as long as format is consistent. This is safe because all dates enter the system via Pydantic models that validate ISO format.

### Constraints

- No domain-specific rules (finance, health, etc.)
- No unbounded recursion — chains capped at 2-hop
- No auto-triggering — rules execute only when `infer()` or `check_contradiction()` is called

### Tests

- Unit tests per rule file: assert expected derivations from known fact sets
- Test bounded chains don't recurse beyond 2-hop
- Test temporal rules with mock `current_date/1`
- Test `inverse_holds` with `contains`/`part_of` pair

---

## Phase 2: Named Graphs for Trust Boundaries

### Problem

All triples go into pyoxigraph's default graph. No distinction between ontology definitions, LLM-extracted facts, human-submitted facts, or inferred conclusions. This means retrieval cannot weight triples by trust level, and inferred facts are indistinguishable from source facts.

### Design

#### Graph scheme

| Graph URI | Contents | Trust tier | Populated by |
|---|---|---|---|
| `ks:graph/ontology` | schema.ttl classes, properties, opposite pairs, canonical predicates | Definitional (highest) | `bootstrap.py` at startup |
| `ks:graph/asserted` | Triples from `/api/claims` and `/api/content` with explicit knowledge items | Human-vetted | `process_triple()` when `extractor == "api"` |
| `ks:graph/extracted` | Triples auto-extracted by LLM | Derived (lower trust) | `process_triple()` when `extractor.startswith("llm_")` |
| `ks:graph/inferred` | ProbLog conclusions, Noisy-OR combined results | Computed (lowest) | Future: when reasoning produces new triples |

#### Constants

Add to `ontology/namespaces.py`:

```python
KS_GRAPH_ONTOLOGY = f"{KS}graph/ontology"
KS_GRAPH_ASSERTED = f"{KS}graph/asserted"
KS_GRAPH_EXTRACTED = f"{KS}graph/extracted"
KS_GRAPH_INFERRED = f"{KS}graph/inferred"
```

#### KnowledgeStore changes

**Critical: all existing SPARQL queries must be rewritten.** pyoxigraph's SPARQL queries without a `GRAPH` clause only search the default graph. After moving triples to named graphs, every bare `SELECT ?o WHERE { <s> <p> ?o . }` will return 0 results. Every method that uses SPARQL or `quads_for_pattern()` must be updated:

**Affected methods and their changes:**

1. **`insert_triple()`** — gains `graph: str | None` parameter (defaults to `KS_GRAPH_EXTRACTED`). Uses `store.add(Quad(s, p, o, NamedNode(graph)))`. RDF-star annotation `INSERT DATA` statements use `GRAPH <uri> { ... }` syntax.

2. **`get_triples_by_subject()`** — rewrite SPARQL to use `GRAPH ?g { ?s ?p ?o }` pattern. Returns `graph` field in results. Gains optional `graphs: list[str] | None` filter — if provided, uses `VALUES ?g { <g1> <g2> }` to restrict; if None, queries all named graphs.

3. **`find_contradictions()`** — rewrite SPARQL to use `GRAPH ?g { ... }` pattern. Contradictions span all graphs intentionally.

4. **`find_opposite_predicate_contradictions()`** — same `GRAPH ?g` rewrite.

5. **`update_confidence()`** — uses `quads_for_pattern()` with `graph_name` parameter. Currently passes `None` which matches all graphs. This continues to work but should pass the specific graph when known.

6. **Idempotency check in `insert_triple()`** — uses `quads_for_pattern(s, p, o, None)` to check for existing triples. After migration, must check across all named graphs (passing `None` for graph_name already does this in pyoxigraph's API).

7. **`query()`** — no change to the method itself, but callers writing raw SPARQL must use `GRAPH` patterns. Document this in the method docstring.

8. **`get_triples_by_predicate()`** — same `GRAPH ?g { ... }` rewrite as `get_triples_by_subject()`. Returns `graph` field in results.

#### bootstrap.py changes

- `store.load()` gains `to_graph=NamedNode(KS_GRAPH_ONTOLOGY)` parameter to load into the named graph
- **Idempotency:** `bootstrap_ontology()` is called on every startup. After migration, it must load into `ks:graph/ontology` exclusively. Add a guard: check if `ks:graph/ontology` already has triples via `quads_for_pattern(None, None, None, NamedNode(KS_GRAPH_ONTOLOGY))`; if non-empty, skip the load. Also ensure no triples leak to the default graph — remove the current ungated `store.load()` call

#### process_triple() changes

- Map `extractor` to graph: `"api"` → `KS_GRAPH_ASSERTED`, `"llm_*"` → `KS_GRAPH_EXTRACTED`
- **Fallback:** any unrecognized extractor value (e.g., `"manual"`, `"import_script"`) maps to `KS_GRAPH_ASSERTED` (treat as human-provided)
- Pass graph URI to `knowledge_store.insert_triple()`

#### RAGRetriever changes

- `get_triples_by_subject()` results include their source graph
- Context building presents `asserted` triples first (marked "verified"), `extracted` second (marked "extracted")
- RAG prompt updated to tell the LLM which facts are verified vs. extracted

#### Migration

Python migration script (not SQL — this is pyoxigraph). Runs during `lifespan()` startup, after asyncpg pool is created (needs DB access for provenance lookups).

**Two-phase approach to prevent data loss on partial failure:**

Phase A — Copy:
1. Read all existing triples from default graph via `quads_for_pattern(None, None, None, store.default_graph)`
2. Identify ontology triples (subjects matching `schema.ttl` namespace patterns) → copy to `ks:graph/ontology`
3. For remaining triples, batch-lookup `provenance.extractor` by triple hash (async, uses the pg_pool from lifespan)
4. Copy each triple to appropriate named graph (`api` → asserted, `llm_*` → extracted, unknown/missing → asserted)

Phase B — Delete from default:
5. Only after all copies succeed, remove triples from default graph
6. Write a completion marker triple: `ks:migration/named_graphs ks:completedAt "2026-03-21T..."` in `ks:graph/ontology`

**Guard:** On startup, check for the completion marker. If present, skip migration. This handles both "already migrated" and "partial failure" (re-run copies everything, then deletes).

### Constraints

- No access control per graph — all graphs readable by all queries
- No automatic promotion from `extracted` → `asserted` (future: human review workflow)
- `ks:graph/inferred` created but empty until reasoning integration in later phases

### Tests

- Test `insert_triple()` with explicit graph parameter → verify triple in correct named graph
- Test `get_triples_by_subject()` with graph filter → only returns triples from specified graphs
- Test `bootstrap.py` loads into `ks:graph/ontology`
- Test migration script correctly redistributes existing triples
- Test RAG context labels triples with trust tier

---

## Phase 3: Chunk-to-Triple Provenance

### Problem

The `provenance` table keys on `(triple_hash, source_url)`. When a user asks "why do you believe X?", the system can say "from this URL" but not "from this paragraph." The extraction pipeline processes full raw_text (truncated to 4000 chars), losing the connection between specific chunks and the triples they produce.

### Design

#### Schema change

New migration `004_add_chunk_provenance.sql`:

```sql
ALTER TABLE provenance ADD COLUMN chunk_id UUID REFERENCES content(id) ON DELETE SET NULL;
CREATE INDEX idx_provenance_chunk_id ON provenance(chunk_id);
```

`ON DELETE SET NULL` rationale: when content is re-ingested, old chunks are deleted and new ones created. The provenance row survives — it just loses its chunk link until the triple is re-extracted. The triple itself and its source_url provenance remain valid.

#### Extraction pipeline change

Current flow in `content.py`:
1. Chunk text → embed → store chunks
2. If no knowledge provided, extract from full `raw_text` (truncated to 4000 chars)
3. Expand to triples → `process_triple()`

New flow:

1. Chunk text → embed → store chunks **with RETURNING id**
2. If no knowledge provided, extract **per chunk** using returned IDs
3. When knowledge IS provided (explicit items), `chunk_id` stays `NULL` (items aren't chunk-derived)

**`insert_chunks()` must return chunk IDs.** Currently uses `conn.execute()` and chunk UUIDs are generated server-side by `gen_random_uuid()`, never returned. Change to use `RETURNING id` and return a list of `(chunk_index, chunk_id)` tuples:

```python
async def insert_chunks(self, content_id, chunks) -> list[tuple[int, str]]:
    sql = """
        INSERT INTO content (content_id, chunk_index, chunk_text, embedding, char_start, char_end)
        VALUES ($1, $2, $3, $4::vector(768), $5, $6)
        RETURNING id
    """
    results = []
    async with self._pool.acquire() as conn:
        for chunk in chunks:
            row = await conn.fetchrow(sql, ...)
            results.append((chunk["chunk_index"], str(row["id"])))
    return results
```

Per-chunk extraction then uses the returned IDs:

```python
chunk_ids = await embedding_store.insert_chunks(content_id, chunk_records)
chunk_id_map = dict(chunk_ids)  # {chunk_index: chunk_id}

for chunk in chunk_records:
    items = await extraction_client.extract(chunk["chunk_text"], ...)
    cid = chunk_id_map[chunk["chunk_index"]]
    for item in items:
        for triple in expand_to_triples(item):
            process_triple(triple, ..., chunk_id=cid)
```

This naturally resolves the 4000-char truncation limitation — each chunk is already within the extraction prompt's size budget.

**Duplicate triples from overlapping chunks:** With `_CHUNK_OVERLAP = 200`, overlapping text may produce the same triple twice. The existing `insert_triple` idempotency (SHA-256 hash) handles identical triples — the second insert is a no-op. Near-duplicate triples with slightly different phrasing will create separate entries; this is acceptable and preferable to losing information.

#### process_triple / provenance_store changes

- `process_triple()` accepts optional `chunk_id: UUID | None` parameter, passes to provenance
- `provenance_store.insert()` includes `chunk_id` in INSERT/UPSERT
- `provenance_store.get_by_triple()` returns `chunk_id` in results

#### /api/ask response changes

New response model additions:

```python
class EvidenceSnippet(BaseModel):
    triple_subject: str
    triple_predicate: str
    triple_object: str
    chunk_text: str
    source_url: str

class AskResponse(BaseModel):
    # ... existing fields ...
    evidence: list[EvidenceSnippet]
```

After retrieval, for each knowledge triple:

1. Look up provenance by triple hash (via `ProvenanceStore`)
2. If `chunk_id` is present, fetch `chunk_text` from `content` table
3. Include as evidence snippet in response

**Cross-store join:** `ProvenanceStore` returns `chunk_id`, but `chunk_text` lives in the `content` table managed by `EmbeddingStore`. Add a new method to `EmbeddingStore`:

```python
async def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, str]:
    """Return {chunk_id: chunk_text} for the given IDs."""
    sql = "SELECT id, chunk_text FROM content WHERE id = ANY($1::uuid[])"
    async with self._pool.acquire() as conn:
        rows = await conn.fetch(sql, chunk_ids)
    return {str(r["id"]): r["chunk_text"] for r in rows}
```

The evidence assembly in `ask.py` calls `provenance_store.get_by_triple()` then `embedding_store.get_chunks_by_ids()` for any non-null chunk_ids. This keeps the two stores decoupled — no SQL join across store boundaries.

#### /api/claims path

No changes. `/api/claims` has no chunks — `chunk_id` stays `NULL`.

### Trade-offs

Per-chunk extraction means:
- **More LLM calls** (N chunks per document vs. 1 call for truncated text)
- **Better provenance** (each triple traces to its exact chunk)
- **Better extraction quality** (focused context per call, no truncation)
- **Handles long documents** (no more 4000-char limit on extractable content)

Ingestion latency increases but ingestion is not latency-sensitive.

### Constraints

- No retroactive chunk linking for existing provenance rows (they keep `chunk_id = NULL`)
- No chunk-level confidence (confidence stays per-triple)
- No admin panel UI changes (future: highlight source text)

### Tests

- Test provenance insert with chunk_id → verify stored and returned
- Test ON DELETE SET NULL → delete chunk, provenance row survives with NULL chunk_id
- Test per-chunk extraction → each triple's provenance has correct chunk_id
- Test `/api/ask` returns evidence snippets with chunk_text
- Test `/api/claims` path → chunk_id is NULL in provenance

---

## Cross-cutting Concerns

### Migration ordering

1. SQL migration `004_add_chunk_provenance.sql` runs via existing migration runner (advisory-lock protected)
2. Named graph migration runs as a one-time Python script at startup (idempotent, guarded)
3. ProbLog rules are file-based — no migration needed, but `ReasoningEngine.__init__` must be changed to glob-load `*.pl` (see Phase 1 integration changes)

### Backward compatibility

- All changes are additive (new columns, new files, new parameters with defaults)
- Existing API contracts unchanged — new fields are optional additions
- Existing provenance rows get `chunk_id = NULL` (valid state)
- Existing triples in default graph get migrated to named graphs (one-time)

### What comes next

These foundations enable:
- **Phase 4 (BM25):** hybrid search adds keyword retrieval alongside existing vector search
- **Phase 7 (Query routing):** can route rule-questions to ProbLog, weight results by named graph trust tier
- **Phase 8 (Multi-hop):** ProbLog inference chains + named graph trust weighting for graph traversal
