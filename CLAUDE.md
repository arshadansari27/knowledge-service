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

Single-process FastAPI service combining an RDF knowledge graph with Noisy-OR probabilistic reasoning. No microservices.

### Data Flow

Content arrives via `/api/content` or `/api/claims` → text is chunked (short = 1 chunk, long ≥4000 chars = N overlapping chunks) → metadata upserted to `content_metadata`, chunks with embeddings stored in `content` → knowledge items are extracted via two-phase LLM extraction (entities then relations) → items expanded to RDF triples and ingested via the pipeline (`ingestion/pipeline.py`) → triples stored in pyoxigraph with RDF-star annotations (confidence, type, temporal bounds) → provenance rows go to PostgreSQL → Noisy-OR combines multi-source confidence → contradictions detected against existing triples → thesis impact checked.

### Key Components

- **TripleStore** (`stores/triples.py`): pyoxigraph wrapper with **named graph support**. Triples are stored in 4 trust-tiered named graphs: `ks:graph/ontology` (schema), `ks:graph/asserted` (human-provided), `ks:graph/extracted` (LLM-derived), `ks:graph/inferred` (computed). All triples are content-addressed via SHA-256 hash. RDF-star annotations attach confidence, knowledge type, and temporal validity. Single `get_triples(subject, predicate, object_, graphs)` method replaces 3 separate query methods. Confidence updates use the Python API to find reification blank nodes.

- **ContentStore** (`stores/content.py`): PostgreSQL + pgvector. Manages `content_metadata` (document metadata) and `content` (chunks with embeddings). Uses `halfvec(768)` for nomic-embed-text embeddings. Hybrid search via vector + BM25 with Reciprocal Rank Fusion.

- **EntityStore** (`stores/entities.py`): PostgreSQL + pgvector. Manages `entity_embeddings` and `predicate_embeddings`. Embedding-based entity deduplication (threshold 0.85) and predicate resolution (threshold 0.90) with LRU caching.

- **ProvenanceStore** (`stores/provenance.py`): PostgreSQL. One row per source per triple, keyed by triple SHA-256 hash. Tracks source_url, source_type, extractor, confidence, timestamps, temporal validity, and **chunk_id** (FK to content table) for chunk-level evidence tracing.

- **ThesisStore** (`stores/theses.py`): PostgreSQL. Named collections of claims with break detection. Theses have status lifecycle (draft → active → archived). Claims are linked by triple hash.

- **Stores dataclass** (`stores/__init__.py`): Single `Stores` dataclass holds all stores + pg_pool. Set as `app.state.stores` in lifespan.

- **Ingestion Pipeline** (`ingestion/pipeline.py`): Replaces the old `process_triple()` god function. Discrete steps: delta detection → insert → contradiction detection → penalty → provenance → evidence combination → thesis impact check.

- **Ingestion Worker** (`ingestion/worker.py`): Three-phase worker: Embed → Extract → Process. Job tracking via `JobTracker`.

- **Noisy-OR** (`reasoning/noisy_or.py`): 6-line evidence combination: `P = 1 - product(1 - ci)`. Replaces the 332-line ProbLog-based ReasoningEngine.

- **DomainRegistry** (`ontology/registry.py`): Reads predicate metadata (labels, synonyms, materiality weights, domains) from the ontology graph via SPARQL. Enables additive domain vocabulary — new domains = new `.ttl` files in `ontology/domains/`.

- **PromptBuilder** (`clients/prompt_builder.py`): Builds domain-aware extraction prompts from templates + DomainRegistry. Supports file-based overrides with inline fallbacks.

- **RAGRetriever** (`stores/rag.py`): Hybrid retrieval — combines chunk-level semantic search (pgvector) with knowledge graph triples for RAG context. Triples include **trust tier** labels (verified/extracted). `/api/ask` returns **evidence snippets** with exact source chunk text.

### Ontology

- **URI normalization**: Single source of truth in `ontology/uri.py` — `to_entity_uri()`, `to_predicate_uri()`, `slugify()`, `is_uri()`.
- **Namespaces**: `ontology/namespaces.py` — NamedNode factories, graph constants, `ks:` prefix helpers.
- **Schema**: `ontology/schema.ttl` — knowledge type classes, properties, domain predicates, opposite/inverse pairs.
- **Domains**: `ontology/domains/base.ttl` — 18 canonical predicates with synonyms, materiality weights, opposite predicates.
- **Bootstrap**: `ontology/bootstrap.py` — loads `schema.ttl` + all `domains/*.ttl` into `ks:graph/ontology`. Accepts `TripleStore` wrapper.

### Models

3 knowledge input types in `models.py` using discriminated union: `TripleInput`, `EventInput`, `EntityInput`. Each has a `to_triples()` method that expands to `(subject, predicate, object)` tuples for ingestion.

### App Lifecycle

`main.py:create_app()` creates the FastAPI app. `lifespan()` handles startup (pyoxigraph init, asyncpg pool, migrations, DomainRegistry, LLM clients, Stores dataclass) and shutdown (flush, close). Tests use `create_app(use_lifespan=False)` and set `app.state` manually.

### Migrations

SQL migrations live in `migrations/` and are applied automatically at startup via advisory-lock-protected runner in `stores/migrations.py`. Tracked in `schema_migrations` table.

## Testing Patterns

- All tests mock external dependencies — no PostgreSQL, Ollama, or network needed
- `pytest-asyncio` with `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed)
- `pytest-httpx` for mocking HTTP calls to LLM APIs
- Tests create `TripleStore(data_dir=None)` for in-memory pyoxigraph
- API tests use `create_app(use_lifespan=False)` and inject mocks via `app.state.stores` (Stores dataclass)

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
- **Resources:** 1G memory limit, 256M reservation
- **Deploy:** `source .env && ansible-playbook -i inventory/hosts.yml playbooks/deploy-aegis.yml`
- **Quick update:** `docker --context swarm-baa service update --image arshadansari27/knowledge-service:latest --force aegis_knowledge`

## LLM Integration Gotchas

- **Do NOT use `response_format: {"type": "json_object"}`** with qwen3 via Ollama/LiteLLM. It returns empty `{}` silently, breaking extraction. The `_extract_json()` utility in `_utils.py` already handles freeform LLM output (markdown fences, `<think>` tags, trailing text).
- All LLM clients (`EmbeddingClient`, `ExtractionClient`, `RAGClient`, `QueryClassifier`, health check) strip a trailing `/v1` from `LLM_BASE_URL` at init time, then use `/v1/...` relative paths in requests. This prevents `/v1/v1/...` double-pathing.
- `TripleStore.insert()` normalizes bare entity labels (e.g. `"cold_exposure"`) to `http://knowledge.local/data/cold_exposure` URIs via `ontology/uri.py` before creating `NamedNode` objects. Without this, pyoxigraph raises `ValueError: No scheme found in an absolute IRI`.

## Project Lessons (auto-managed by cmemory)
- cmemory's readStdin() in shared.ts only resolved on stdin 'end' event, but Claude Code hooks may not immediately close the stdin pipe. Fix: try JSON.parse() eagerly on each 'data' chunk so the promise resolves as soon as complete JSON arrives, without waiting for 'end'.
