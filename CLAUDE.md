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
uv run pytest tests/test_knowledge_store.py -v

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

Single-process FastAPI service combining an RDF knowledge graph with Bayesian probabilistic reasoning. No microservices.

### Data Flow

Content arrives via `/api/content` or `/api/claims` → text is chunked (short = 1 chunk, long ≥4000 chars = N overlapping chunks) → metadata upserted to `content_metadata`, chunks with embeddings stored in `content` → knowledge items are written as RDF triples to pyoxigraph with RDF-star annotations (confidence, type, temporal bounds) → provenance rows go to PostgreSQL → Noisy-OR combines multi-source confidence → ProbLog propagates probabilities through inference chains.

### Key Components

- **KnowledgeStore** (`stores/knowledge.py`): pyoxigraph wrapper with **named graph support**. Triples are stored in 4 trust-tiered named graphs: `ks:graph/ontology` (schema), `ks:graph/asserted` (human-provided), `ks:graph/extracted` (LLM-derived), `ks:graph/inferred` (computed). All triples are content-addressed via SHA-256 hash. RDF-star annotations attach confidence, knowledge type, and temporal validity. All SPARQL queries use `GRAPH ?g { ... }` patterns. Confidence updates use the Python API to find reification blank nodes and preserve the named graph.

- **ReasoningEngine** (`reasoning/engine.py`): ProbLog wrapper. Noisy-OR evidence combination: `P = 1 - product(1 - ci)`. Glob-loads all `.pl` rule files from `reasoning/rules/`. Supports 5-tuple claims with metadata dict (knowledge_type, valid_from, valid_until). 12 domain-agnostic rules across 5 files: base (contradicts, value_conflict, supported, inverse_holds, corroborated), inference_chains (indirect_link, causal_propagation), confidence (high_confidence, contested, authoritative), temporal (expired, currently_valid, supersedes), knowledge_types.

- **EmbeddingStore** (`stores/embedding.py`): PostgreSQL + pgvector. Manages `content_metadata` (document metadata), `content` (chunks with embeddings), `entity_embeddings`, and `predicate_embeddings`. Uses `halfvec(768)` for nomic-embed-text embeddings. `insert_chunks()` returns `(chunk_index, chunk_id)` pairs for provenance linking. `get_chunks_by_ids()` supports batch chunk text lookup.

- **ProvenanceStore** (`stores/provenance.py`): PostgreSQL. One row per source per triple, keyed by triple SHA-256 hash. Tracks source_url, source_type, extractor, confidence, timestamps, temporal validity, and **chunk_id** (FK to content table) for chunk-level evidence tracing.

- **EntityResolver** (`stores/entity_resolver.py`): Embedding-based entity deduplication + predicate resolution + optional DBpedia/Wikidata federation. Entity threshold 0.85, predicate threshold 0.90.

- **RAGRetriever** (`stores/rag.py`): Hybrid retrieval — combines chunk-level semantic search (pgvector) with knowledge graph triples for RAG context. Triples include **trust tier** labels (verified/extracted). `/api/ask` returns **evidence snippets** with exact source chunk text.

### App Lifecycle

`main.py:create_app()` creates the FastAPI app. `lifespan()` handles startup (pyoxigraph init, asyncpg pool, migrations, LLM clients) and shutdown (flush, close). Tests use `create_app(use_lifespan=False)` and set `app.state` manually.

### Migrations

SQL migrations live in `migrations/` and are applied automatically at startup via advisory-lock-protected runner in `main.py`. Tracked in `schema_migrations` table.

### Models

All 7 knowledge types are discriminated union types in `models.py` using `Annotated[Union[...], Field(discriminator="knowledge_type")]`. Types: Claim, Fact, Event, Entity, Relationship, Conclusion, TemporalState.

### Namespaces

Custom `ks:` namespace is minimal. Reuses `schema:`, `dc:`, `skos:`, `foaf:`, `prov:`. Defined in `ontology/namespaces.py`. Base ontology (opposite predicates, etc.) bootstrapped into pyoxigraph on startup via `ontology/bootstrap.py`.

## Testing Patterns

- All tests mock external dependencies — no PostgreSQL, Ollama, or network needed
- `pytest-asyncio` with `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed)
- `pytest-httpx` for mocking HTTP calls to LLM APIs
- Tests create `KnowledgeStore(data_dir=None)` for in-memory pyoxigraph
- API tests use `create_app(use_lifespan=False)` and inject mocks via `app.state`

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
- `KnowledgeStore.insert_triple` normalizes bare entity labels (e.g. `"cold_exposure"`) to `http://knowledge.local/data/cold_exposure` URIs before creating `NamedNode` objects. Without this, pyoxigraph raises `ValueError: No scheme found in an absolute IRI`.

## Project Lessons (auto-managed by cmemory)
- cmemory's readStdin() in shared.ts only resolved on stdin 'end' event, but Claude Code hooks may not immediately close the stdin pipe. Fix: try JSON.parse() eagerly on each 'data' chunk so the promise resolves as soon as complete JSON arrives, without waiting for 'end'.
