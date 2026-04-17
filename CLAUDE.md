# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install dependencies (uses uv)
uv sync --dev

# Run the service (requires PostgreSQL + Ollama/LiteLLM)
uv run uvicorn knowledge_service.main:app --reload

# Run all tests (no external deps needed — everything is mocked)
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_triple_store.py -v

# Run a single test by name
uv run pytest tests/test_api_claims.py -v -k "test_ingest_claim"

# Lint and format
uv run ruff check .
uv run ruff format --check .
uv run ruff format .          # auto-fix formatting

# Docker (full stack)
docker compose up -d
```

## Architecture

Single-process FastAPI service. No microservices.

The user-visible feature is **hybrid BM25 + vector RAG** over ingested documents: chunks with pgvector embeddings in `content`, fused with a Postgres full-text tsvector via Reciprocal Rank Fusion. A small **RDF knowledge graph layer** sits behind it: an LLM extracts entity/relation triples per chunk (pyoxigraph named graphs, Postgres provenance), multi-source confidences combine via **Noisy-OR** (a 4-line function), and a **3-rule forward-chaining inference engine** (inverse / transitive / type-inheritance) derives extra triples at ingestion time.

Contradictions are detected at retrieval and reported in the `/api/ask` response; per-graph trust labels (`verified` / `federated` / `extracted`) are tagged onto retrieved triples for the LLM prompt. Neither contradictions nor trust tiers filter or re-rank results today — they are surfaced, not enforced.

### Data Flow

Content arrives via `/api/content`, `/api/content/upload` (file upload), or `/api/claims` → **documents are parsed** (PDF via PyMuPDF, HTML via readability+BeautifulSoup, CSV/JSON via stdlib, or passthrough for plain text) → text is chunked (short = 1 chunk, long ≥4000 chars = N overlapping chunks) → metadata upserted to `content_metadata`, chunks with embeddings stored in `content` → **NLP pre-pass** (spaCy NER + Wikidata entity linking, when available) produces entity hints → knowledge items are extracted via two-phase LLM extraction (entities then relations, enriched by NLP hints) → **coreference resolution** merges entities sharing a Wikidata QID from the NLP pre-pass → items expanded to RDF triples and ingested via the pipeline (`ingestion/pipeline.py`) → triples stored in pyoxigraph with RDF-star annotations (confidence, type, temporal bounds) → provenance rows go to PostgreSQL → Noisy-OR combines multi-source confidence → contradictions detected against existing triples → **inference engine derives new triples** (inverse, transitive, type inheritance) into `ks:graph/inferred` → thesis impact checked.

**Full pipeline:** Parse → Chunk → Embed → NLP Pre-pass → Extract → Coreference → Process

### Key Components

- **TripleStore** (`stores/triples.py`): pyoxigraph wrapper with **named graph support**. Triples are stored in 5 named graphs by provenance class: `ks:graph/ontology` (schema), `ks:graph/asserted` (human-provided), `ks:graph/extracted` (LLM-derived), `ks:graph/inferred` (computed), `ks:graph/federated` (external sources). The graph a triple lives in is surfaced to readers as a `trust_tier` label but retrieval does not filter by tier — it's informational. All triples are content-addressed via SHA-256 hash. RDF-star annotations attach confidence, knowledge type, and temporal validity. Single `get_triples(subject, predicate, object_, graphs)` method replaces 3 separate query methods. Confidence updates use the Python API to find reification blank nodes.

- **ContentStore** (`stores/content.py`): PostgreSQL + pgvector. Manages `content_metadata` (document metadata) and `content` (chunks with embeddings). Uses `halfvec(768)` for nomic-embed-text embeddings. Hybrid search via vector + BM25 with Reciprocal Rank Fusion.

- **EntityStore** (`stores/entities.py`): PostgreSQL + pgvector. Manages `entity_embeddings`, `predicate_embeddings`, and `entity_aliases`. Embedding-based entity deduplication (threshold 0.85) and predicate resolution (threshold 0.90) with LRU caching. Alias table lookup (from coreference) is checked before falling back to embedding similarity.

- **ProvenanceStore** (`stores/provenance.py`): PostgreSQL. One row per source per triple, keyed by triple SHA-256 hash. Tracks source_url, source_type, extractor, confidence, timestamps, temporal validity, and **chunk_id** (FK to content table) for chunk-level evidence tracing.

- **ThesisStore** (`stores/theses.py`): PostgreSQL. Named collections of claims with break detection. Theses have status lifecycle (draft → active → archived). Claims are linked by triple hash.

- **Stores dataclass** (`stores/__init__.py`): Single `Stores` dataclass holds all stores + pg_pool. Set as `app.state.stores` in lifespan.

- **Ingestion Pipeline** (`ingestion/pipeline.py`): Discrete per-triple steps: delta detection → retract stale inferences → insert → contradiction detection → penalty → provenance → evidence combination → **inference**.

- **Ingestion Worker** (`ingestion/worker.py`): Five-phase worker: Embed → NLP Pre-pass → Extract → Coreference → Process. Job tracking via `JobTracker` with status labels: `embedding`, `analyzing`, `extracting`, `resolving`, `processing`.

- **Parser Registry** (`parsing/__init__.py`): Pluggable document parsing layer. `ParserRegistry` with format detection (content-type > URL extension > magic bytes > fallback). Built-in parsers: `PdfParser` (PyMuPDF), `HtmlParser` (readability-lxml + BeautifulSoup), `StructuredParser` (JSON/CSV), `TextParser` (passthrough). Image formats are detected but no image parser is registered; `/api/content/upload` returns 422 for image uploads. Each parser produces a `ParsedDocument` (text, title, metadata, source_format, images).

- **NLP Phase** (`nlp/__init__.py`): spaCy-based NER + Wikidata entity linking pre-pass. Runs on each chunk, produces `NlpResult` with `NlpEntity` objects (text, label, start/end char, wikidata_id). Entity hints are forwarded to LLM extraction to improve recognition. Fallback entities (spaCy-only, not confirmed by LLM) are included at `nlp_entity_confidence` (default 0.5). Graceful degradation when spaCy is unavailable.

- **Coreference Phase** (`ingestion/coreference.py`): Deterministic entity deduplication. Entities sharing a Wikidata QID (from NLP pre-pass) are merged into a single `EntityGroup`. Results stored in `entity_aliases` table. `canonicalize()` rewrites knowledge item labels before processing.

- **Noisy-OR** (`reasoning/noisy_or.py`): 4-line evidence combination: `P = 1 - product(1 - ci)`. Replaces the earlier 332-line ProbLog-based ReasoningEngine — this is the entirety of the "probabilistic" part, don't read more into it.

- **Inference Engine** (`reasoning/engine.py`): Forward-chaining inference at ingestion time. Three ontology-declared rule types: `InverseRule` (materializes `ks:inversePredicate` pairs), `TransitiveRule` (closes `ks:transitivePredicate` chains), `TypeInheritanceRule` (propagates `has_property` through `is_a`). All three rules guard against literal objects via `is_uri()` checks — literals cannot be used as SPARQL subjects. BFS execution with depth cap of 3 and cycle detection via hash dedup. Confidence = product of source confidences. Derived triples go to `ks:graph/inferred` with `ks:derivedFrom` and `ks:inferenceMethod` RDF-star annotations. Retraction cascades when source triples change. Initialized once in app lifespan, stored on `app.state.inference_engine`.

- **DomainRegistry** (`ontology/registry.py`): Reads predicate metadata (labels, synonyms, materiality weights, domains) from the ontology graph via SPARQL. Enables additive domain vocabulary — new domains = new `.ttl` files in `ontology/domains/`.

- **PromptBuilder** (`clients/prompt_builder.py`): Builds domain-aware extraction prompts from templates + DomainRegistry. Supports file-based overrides with inline fallbacks.

- **RAGRetriever** (`stores/rag.py`): Hybrid retrieval — combines chunk-level hybrid search (vector + BM25 via RRF) with knowledge graph triples for RAG context. Retrieved triples are tagged with a `trust_tier` label (verified/federated/extracted) derived from their source graph — the prompt discloses this to the LLM, but ranking and filtering are tier-agnostic. Contradictions found during retrieval are passed through to the `/api/ask` response as-is; they don't penalise or exclude triples at read time. `/api/ask` returns **evidence snippets** with exact source chunk text.

### Ontology

- **URI normalization**: Single source of truth in `ontology/uri.py` — `to_entity_uri()`, `to_predicate_uri()`, `slugify()`, `is_uri()`.
- **Namespaces**: `ontology/namespaces.py` — NamedNode factories, graph constants, `ks:` prefix helpers.
- **Schema**: `ontology/schema.ttl` — knowledge type classes, properties, domain predicates, opposite/inverse pairs.
- **Domains**: `ontology/domains/` — domain-specific `.ttl` files (`base.ttl`, `health.ttl`, `technology.ttl`, `research.ttl`). `base.ttl` defines 18 canonical predicates with synonyms, materiality weights, opposite predicates, transitive annotations (`part_of`, `is_a`, `located_in`, `depends_on`), and inverse pairs (`contains ↔ part_of`).
- **Bootstrap**: `ontology/bootstrap.py` — loads `schema.ttl` + all `domains/*.ttl` into `ks:graph/ontology`. Accepts `TripleStore` wrapper.

### Models

3 knowledge input types in `models.py` using union type (no Pydantic discriminator): `TripleInput`, `EventInput`, `EntityInput`. Each has a `to_triples()` method that expands to `(subject, predicate, object)` tuples for ingestion.

### App Lifecycle

`main.py:create_app()` creates the FastAPI app. `lifespan()` handles startup (pyoxigraph init, asyncpg pool, migrations, DomainRegistry, LLM clients, ParserRegistry, spaCy NLP pipeline, Stores dataclass) and shutdown (flush, close). Tests use `create_app(use_lifespan=False)` and set `app.state` manually.

### Migrations

SQL migrations live in `migrations/` and are applied automatically at startup via advisory-lock-protected runner in `stores/migrations.py`. Tracked in `schema_migrations` table.

## Content pipeline invariants

Ingestion splits across a sync acceptance phase (`_accept_content_request`) and an async background worker (`run_ingestion`). Because triples live in pyoxigraph and chunks/provenance live in PostgreSQL, no single transaction can cover the whole pipeline. The state machine instead relies on these invariants:

- **`ingestion_jobs` is the source of truth for completion.** Readers that care about finished state should join on `status='completed'`. A `content_metadata` row can exist without a completed job (accepted, in-flight, or failed).
- **Only one active job per `content_id`.** Enforced by the partial unique index `idx_ingestion_jobs_active` (`WHERE status NOT IN ('completed','failed')`). Re-ingest is blocked until the current job reaches a terminal state.
- **Startup janitor recovers from process crashes.** The lifespan in `main.py` marks every non-terminal job as `failed` on startup so a SIGTERM/OOM mid-worker doesn't leave the content permanently un-reingestable.
- **EmbedPhase uses `ContentStore.replace_chunks()`, not separate `delete_chunks` + `insert_chunks`.** DELETE and INSERT must land in a single PG transaction. Losing this atomicity is a silent data-quality bug: `provenance.chunk_id` is `ON DELETE SET NULL`, so a delete without a follow-up insert permanently wipes chunk-level evidence for every previously-ingested triple linked to that content.
- **ProcessPhase is not transactional across stores.** A crash during triple ingestion can leave partial pyoxigraph writes. This is a known 2PC-shaped gap; current mitigation is the job status + startup janitor + operator-driven retry.

## Testing Patterns

- **CI tests** (`tests/`): All tests mock external dependencies — no PostgreSQL, Ollama, or network needed. `pytest-asyncio` with `asyncio_mode = "auto"`. `pytest-httpx` for mocking HTTP calls. Tests create `TripleStore(data_dir=None)` for in-memory pyoxigraph. API tests use `create_app(use_lifespan=False)` and inject mocks via `app.state.stores`.
- **E2E tests** (`tests/e2e/`): Run against real PostgreSQL + Ollama. Excluded from CI via `--ignore=tests/e2e` in pyproject.toml. Run locally with `uv run pytest tests/e2e/ -v`. Service starts as a subprocess, tests hit it via HTTP. Requires: running PostgreSQL, Ollama with embedding + chat models.

## Ruff Configuration

- Target: Python 3.12
- Line length: 100

## CI/CD

GitHub Actions: lint → test → auto-patch-bump (main only) → Docker build+push. Version bump commits include `[skip ci]`. Version is the single source of truth in `pyproject.toml`.

## Production Deployment

Deployed as part of the **AEGIS Docker Swarm stack** on a homelab cluster. Full details in `docs/deployment.md`.

- **Infra repo:** `Workspace/infrastructure/homelab-gitops` (Ansible role: `roles/aegis/`)
- **Service name:** `aegis_knowledge` on node `meem` (10.20.0.20)
- **URL:** `https://knowledge.hikmahtech.in` (Traefik reverse proxy, IP-whitelisted)
- **Database:** `knowledge` DB on shared `pgvector/pgvector:pg16` instance (`aegis_postgres`)
- **LLM access:** Via LiteLLM proxy (`https://litellm.hikmahtech.in`) → Ollama instances (nomic-embed-text on meem, qwen3:14b on asif)
- **RDF storage:** Docker volume `aegis_knowledge_oxigraph` → `/app/data/oxigraph`
- **spaCy KB:** Docker volume `aegis_knowledge_spacy` → `/app/data/spacy` (~1GB Wikidata KB, downloaded on first start)
- **Resources:** 3G memory limit, 512M reservation
- **Deploy:** `source .env && ansible-playbook -i inventory/hosts.yml playbooks/deploy-aegis.yml`
- **Quick update:** `docker --context swarm-baa service update --image arshadansari27/knowledge-service:latest --force aegis_knowledge`

## ProcessPhase consistency

`ProcessPhase` writes to both pyoxigraph (triples, RDF-star annotations) and PostgreSQL (provenance, outbox). These stores cannot share a transaction, so coordination is handled via an **outbox pattern**:

- **Commit boundary is PostgreSQL.** Every pyoxigraph write is first staged as a row in `triple_outbox` inside the same PG transaction as its matching `provenance` row. After commit, an `OutboxDrainer` replays the staged rows to pyoxigraph and marks them `applied_at`.
- **Invariant:** Every `provenance` row references a triple that is either already durable in pyoxigraph or present as an unapplied `triple_outbox` row. The inverse (pyoxigraph triple without provenance) is never produced by this layer.
- **Drain happens twice:** synchronously after each PG commit (fast path) and at application startup via `app.state.outbox_drainer.drain_pending()` (recovery path for crashes between commit and drain).
- **All outbox operations are idempotent.** Re-applying an `insert` is a no-op (pyoxigraph deduplicates by content hash). `update_confidence` is idempotent when writing the target value. `insert_inferred` guards RDF-star annotations with SPARQL ASK (per `lesson_pyoxigraph_rdfstar`). `retract_inference` re-runs against a hash whose inferences have already been removed and finds nothing to do.
- **Derived work is skippable.** Contradictions penalty and inference-engine runs happen *after* the base triple is durable in both stores. A crash during derived work leaves the base triple intact; re-ingestion re-runs derived work deterministically because the engine is pure and content-addressed inserts are idempotent.
- **Not the same as the stuck-job janitor.** The janitor marks `ingestion_jobs` as failed on process restart; the outbox drainer recovers per-triple store drift. They are independent mechanisms.

## Reader-side status filtering

`ContentStore.search()` and `ContentStore._search_bm25()` filter out content
whose latest `ingestion_jobs.status` is non-terminal (any of `accepted`,
`embedding`, `analyzing`, `extracting`, `resolving`, `processing`). Content
with `completed`, `failed`, or no job row passes through. This prevents the
hybrid retriever from returning chunks whose KG triples have not yet
committed — the "half-picture" problem.

- The filter is applied in SQL via `LEFT JOIN LATERAL` against
  `ingestion_jobs`, ordered by `created_at DESC LIMIT 1` so the latest job
  wins (re-ingest semantics).
- `RAGRetriever` inherits the filter transparently because it calls
  `ContentStore.search()`.
- `/api/content/{id}/chunks` is deliberately exempt — that endpoint reads
  chunks by ID for operator/debug flows and must see in-flight content.
- Controlled by `settings.reader_exclude_inflight` (env:
  `READER_EXCLUDE_INFLIGHT`, default `true`). The flag exists as a rollout
  escape hatch, not a per-request knob.
- `failed` jobs are intentionally included. The outbox 2PC may have
  committed partial triples before the failure; hiding them would remove
  real evidence. Operators promote failed → completed via re-ingest.

## LLM Integration Gotchas

- **Do NOT use `response_format: {"type": "json_object"}`** with qwen3 via Ollama/LiteLLM. It returns empty `{}` silently, breaking extraction. The `_extract_json()` utility in `_utils.py` already handles freeform LLM output (markdown fences, `<think>` tags, trailing text).
- All LLM clients (`EmbeddingClient`, `ExtractionClient`, `RAGClient`, health check) strip a trailing `/v1` from `LLM_BASE_URL` at init time, then use `/v1/...` relative paths in requests. This prevents `/v1/v1/...` double-pathing.
- `TripleStore.insert()` normalizes bare entity labels (e.g. `"cold_exposure"`) to `http://knowledge.local/data/cold_exposure` URIs via `ontology/uri.py` before creating `NamedNode` objects. Without this, pyoxigraph raises `ValueError: No scheme found in an absolute IRI`.
- **Inference engine URI normalization**: The pipeline normalizes subject/predicate URIs before passing triples to `InferenceEngine.run()`, since the engine's `DerivedTriple.compute_hash()` creates `NamedNode` objects that require full URIs. All three inference rules (`InverseRule`, `TransitiveRule`, `TypeInheritanceRule`) skip when the object is a literal (non-URI), since literals cannot become RDF subjects or be used in SPARQL `<brackets>`.
- **URL auto-fetch body propagation**: `_accept_content_request()` returns the potentially-updated `body` (with fetched `raw_text` and `title`) in its result dict. Both single and batch callers in `post_content()` use `result["body"]` — not the original `body` — when dispatching the background worker. Without this, the worker receives empty `raw_text` and skips LLM extraction for all auto-fetched URLs. The function also updates `content_metadata` after a successful fetch so the database reflects the fetched content.

## Project Lessons (auto-managed by cmemory)
- cmemory's readStdin() in shared.ts only resolved on stdin 'end' event, but Claude Code hooks may not immediately close the stdin pipe. Fix: try JSON.parse() eagerly on each 'data' chunk so the promise resolves as soon as complete JSON arrives, without waiting for 'end'.
