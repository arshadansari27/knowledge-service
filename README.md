# Knowledge Service

[![CI](https://github.com/arshadansari27/knowledge-service/actions/workflows/ci.yml/badge.svg)](https://github.com/arshadansari27/knowledge-service/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/docker/v/arshadansari27/knowledge-service?label=docker&sort=semver)](https://hub.docker.com/r/arshadansari27/knowledge-service)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A personal knowledge service that ingests content you encounter and exposes it through **hybrid BM25 + vector RAG** with a small **RDF knowledge-graph layer** behind it. Documents are parsed, chunked, and embedded; an LLM extracts entity/relation triples per chunk; triples carry per-source confidence and are combined via Noisy-OR when multiple sources agree; a lightweight 3-rule inference pass (inverse / transitive / type-inheritance) materialises extra conclusions at ingestion time. Queryable via SPARQL and semantic search, with source chunks and provenance returned as evidence.

Built by [Hikmah Technologies](https://hikmahtechnologies.com) | [@hikmahtech](https://x.com/hikmahtech) | [@arshadansari27](https://x.com/arshadansari27)

The primary consumer is AEGIS, where AI agents gain awareness of your accumulated knowledge. The service also stands alone as a reusable knowledge API.

> **The ontology is the product. Sources are just input channels.**

---

## What Makes This Different

Every "second brain" tool treats all content as equal — a bookmark, a note, a highlight are all flat objects with tags. No confidence. No provenance. No temporal validity. No contradiction detection. No inference.

This system separates **content** (what you consumed) from **knowledge** (what you derived from it), and models knowledge with:

- **Uncertainty** — triples carry a confidence score; when multiple sources assert the same fact, their confidences combine via Noisy-OR (`1 - Π(1 - cᵢ)`)
- **Provenance** — every triple traces back to its source, extraction method, timestamp, and the specific chunk it was derived from
- **Temporality** — knowledge has `valid_from` / `valid_until`, not just `created_at`
- **Ontological structure** — concepts link to established vocabularies (Schema.org, Dublin Core, SKOS) so "PostgreSQL" in your codebase and "PostgreSQL" in an article are the same entity
- **Inference** — inverse/transitive/type-inheritance rules derive extra triples at ingestion time, with source triples preserved for retraction

---

## Architecture

Single FastAPI process with embedded components. No microservices.

```
FastAPI Process
├── ParserRegistry     Pluggable document parsing (PDF, HTML, CSV, JSON, images)
├── NlpPhase           spaCy NER + Wikidata entity linking pre-pass
├── CoreferencePhase   Entity dedup by shared Wikidata QID
├── TripleStore        pyoxigraph — RDF 1.2, RDF-star, 5 named graphs by provenance
├── InferenceEngine    3 forward-chaining rules (inverse, transitive, type inheritance)
├── QueryClassifier    Intent routing (semantic / entity / graph), LLM-classified
├── RAGRetriever       Hybrid chunk retrieval (BM25 + vector RRF) + KG triple context
├── ContentStore       PostgreSQL + pgvector — BM25 + vector hybrid search (RRF)
├── ExtractionClient   LLM extraction with retry-on-5xx/timeout
└── ProvenanceStore    Per-source evidence rows with chunk_id FK

Pipeline: Parse → Chunk → Embed → NLP Pre-pass → Extract → Coreference → Process

PostgreSQL
├── content_metadata   Document metadata (url, title, source_type, tags, raw_text)
├── content            Chunks with embeddings, section headers, full-text search
├── provenance         Per-source evidence rows with chunk_id FK
├── entity_embeddings  Entity URIs with embeddings for resolution
├── entity_aliases     Coreference alias → canonical URI mappings
├── ingestion_jobs     Async job tracking with per-phase progress
└── triple_outbox      Staged pyoxigraph writes — drained after PG commit
```

**ProcessPhase consistency.** Triples live in pyoxigraph, provenance lives in PostgreSQL, and a single transaction cannot cover both. Each per-triple write is staged as a row in `triple_outbox` inside the same PG transaction as its `provenance` row; an `OutboxDrainer` applies staged rows to pyoxigraph after commit, and re-runs on application startup to recover from crashes between commit and drain. Every outbox operation is idempotent (content-addressed inserts, SPARQL ASK-guarded RDF-star annotations). See `docs/superpowers/specs/2026-04-15-processphase-2pc-outbox-design.md`.

**Reader-side status filtering.** `/api/search`, `/api/ask`, and `RAGRetriever` only return content whose latest `ingestion_jobs.status` is terminal (`completed` or `failed`), or has no job row. In-flight content is hidden in SQL via a `LEFT JOIN LATERAL` against `ingestion_jobs`, so chunks without their KG triples never reach the retriever. Controlled by `READER_EXCLUDE_INFLIGHT` (default `true`). `/api/content/{id}/chunks` is deliberately exempt. See `docs/superpowers/specs/2026-04-15-reader-status-filter-design.md`.

For deployment details, see [docs/deployment.md](docs/deployment.md).

---

## The 7 Knowledge Types

Every piece of knowledge is classified as one of these seven types:

| Type | Truth Model | Example |
|------|-------------|---------|
| **Claim** | Probabilistic (0.0–1.0) | "Intermittent fasting reduces inflammation" — 0.7 from a YouTube video |
| **Fact** | High-confidence (≥0.9) | "Project AEGIS uses PostgreSQL 16" — from codebase scan |
| **Event** | Timestamped, deterministic | Salary payment received 2026-03-01 |
| **Entity** | Typed, ontology-linked | "AEGIS is a schema:SoftwareApplication" |
| **Relationship** | Typed link between entities | "AEGIS depends-on PostgreSQL" |
| **Conclusion** | Derived, reasoning chain preserved | "Cold exposure likely increases dopamine" — Noisy-OR combination of 3 sources |
| **TemporalState** | Time-bounded property (valid_until required) | Bitcoin price $X between date A and date B |

---

## Confidence Model

**Two-layer design:**

1. **RDF-star annotation on each triple** — the combined confidence after Noisy-OR over all sources:
   ```turtle
   <<:cold_exposure :increases :dopamine>>
       ks:confidence "0.88"^^xsd:float .
   ```

2. **PostgreSQL provenance table** — one row per source per triple:
   ```
   (triple_hash, source_url, source_type, extractor, confidence, ingested_at, valid_from, valid_until)
   ```

When the same claim arrives from multiple sources, **Noisy-OR** combines their individual confidences:
```
combined = 1 - product(1 - ci)

# Example: source A at 0.7, source B at 0.6
combined = 1 - (0.3 × 0.4) = 0.88
```

The combined value is written back to the RDF-star annotation. The inference engine propagates confidence through forward-chaining rules (inverse, transitive, type inheritance) for derived conclusions.

---

## API Reference

**Base URL:** `http://localhost:8000`
**Interactive docs:** `http://localhost:8000/docs`

### Health

```http
GET /health
```

Returns status of all components (knowledge store, PostgreSQL, LLM API).

---

### Ingest Content

```http
POST /api/content
Content-Type: application/json
```

Ingest content with knowledge items. Parses documents (auto-detects format), chunks text, runs NLP pre-pass + LLM extraction, resolves entities via coreference, and writes triples. If `url` is provided without `raw_text` and starts with `http`, the URL is fetched and parsed automatically.

Accepts a **single object** or a **JSON array** for batch processing.

**Single request:**

```json
{
  "url": "https://example.com/article",
  "title": "Cold Exposure and Dopamine",
  "summary": "A review of studies on cold exposure effects.",
  "raw_text": "...",
  "source_type": "article",
  "tags": ["health", "neuroscience"],
  "metadata": {},
  "knowledge": [
    {
      "knowledge_type": "Claim",
      "subject": "http://dbpedia.org/resource/Cold_shock_response",
      "predicate": "http://knowledge.local/schema/increases",
      "object": "http://dbpedia.org/resource/Dopamine",
      "confidence": 0.75
    },
    {
      "knowledge_type": "Entity",
      "uri": "http://dbpedia.org/resource/Dopamine",
      "rdf_type": "schema:ChemicalSubstance",
      "label": "Dopamine",
      "properties": {}
    }
  ]
}
```

**Response (202 Accepted):**
```json
{
  "content_id": "uuid",
  "job_id": "uuid",
  "status": "accepted",
  "chunks_total": 5,
  "chunks_capped_from": null
}
```

Processing happens asynchronously. Poll `/api/content/{content_id}/status` for progress.

**Batch request** — send an array, get an array:

```json
[
  { "url": "https://a.com", "title": "Article A", "source_type": "article" },
  { "url": "https://b.com", "title": "Article B", "source_type": "article" }
]
```

---

### Upload File

```http
POST /api/content/upload
Content-Type: multipart/form-data
```

Upload a file (PDF, HTML, CSV, JSON, plain text) for ingestion. Format is auto-detected from filename, content-type, or magic bytes. Returns 202 with a job ID — poll `/api/content/{id}/status` for progress.

```bash
curl -X POST http://localhost:8000/api/content/upload \
  -H "X-API-Key: your-password" \
  -F "file=@paper.pdf;type=application/pdf" \
  -F "title=Research Paper" \
  -F "source_type=paper"
```

**Response (202):** `{"content_id": "uuid", "job_id": "uuid", "chunks_total": 3}`

---

### Check Ingestion Status

```http
GET /api/content/{content_id}/status
```

Returns job status with progress counters. Status values: `embedding` → `analyzing` → `extracting` → `resolving` → `processing` → `completed` / `failed`.

---

### Ingest Claims Directly

```http
POST /api/claims
Content-Type: application/json
```

Ingest knowledge items without storing raw content. Useful for programmatic ingestion where content storage is not needed.

Accepts a **single object** or a **JSON array** for batch processing. When an array is sent, a matching array of responses is returned.

**Single request:**

```json
{
  "source_url": "https://example.com/paper",
  "source_type": "paper",
  "extractor": "llm_qwen3:14b",
  "knowledge": [
    {
      "knowledge_type": "Fact",
      "subject": "http://knowledge.local/data/aegis",
      "predicate": "http://schema.org/softwareRequirements",
      "object": "http://dbpedia.org/resource/PostgreSQL",
      "confidence": 0.99
    }
  ]
}
```

**Batch request** — send an array, get an array:

```json
[
  {
    "source_url": "https://example.com/a",
    "source_type": "bookmark",
    "extractor": "n8n",
    "knowledge": [{ "knowledge_type": "Claim", "subject": "...", "predicate": "...", "object": "...", "confidence": 0.85 }]
  },
  {
    "source_url": "https://example.com/b",
    "source_type": "bookmark",
    "extractor": "n8n",
    "knowledge": [{ "knowledge_type": "Claim", "subject": "...", "predicate": "...", "object": "...", "confidence": 0.9 }]
  }
]
```

All 7 knowledge types are accepted. Examples for each:

```json
// Event
{
  "knowledge_type": "Event",
  "subject": "http://knowledge.local/data/payment/2026-03-01",
  "occurred_at": "2026-03-01",
  "properties": { "amount": "4500", "currency": "GBP" }
}

// TemporalState (valid_until is mandatory)
{
  "knowledge_type": "TemporalState",
  "subject": "http://dbpedia.org/resource/Bitcoin",
  "property": "http://schema.org/price",
  "value": "65000",
  "valid_from": "2024-03-01",
  "valid_until": "2024-03-31"
}

// Conclusion
{
  "knowledge_type": "Conclusion",
  "concludes": "Cold exposure likely increases dopamine based on 3 independent sources",
  "derived_from": ["<triple_hash_1>", "<triple_hash_2>", "<triple_hash_3>"],
  "inference_method": "bayesian_combination",
  "confidence": 0.88
}

// Relationship
{
  "knowledge_type": "Relationship",
  "subject": "http://knowledge.local/data/aegis",
  "predicate": "http://schema.org/hasPart",
  "object": "http://knowledge.local/data/knowledge-service",
  "confidence": 0.99
}
```

---

### Semantic Search

```http
GET /api/search?q=cold+exposure+dopamine&limit=10&source_type=article
```

Searches ingested content by semantic similarity using pgvector cosine distance. Returns chunk-level results — each result is the most relevant chunk of a document, not the full document. Short documents have a single chunk; long documents (≥4000 chars) are split into overlapping chunks.

**Parameters:**
- `q` (required) — query text
- `limit` — max results (1–100, default 10)
- `source_type` — filter by source type (`article`, `video`, etc.)
- `tags` — filter by tags (repeat for multiple: `?tags=health&tags=neuroscience`)

**Response:**
```json
[
  {
    "content_id": "uuid",
    "url": "https://...",
    "title": "Cold Exposure and Dopamine",
    "summary": "...",
    "similarity": 0.94,
    "source_type": "article",
    "tags": ["health"],
    "ingested_at": "2026-03-18T10:00:00Z",
    "chunk_text": "The relevant section matching the query...",
    "chunk_index": 0
  }
]
```

---

### Query the Knowledge Graph

```http
GET /api/knowledge/query?subject=http://dbpedia.org/resource/Dopamine
```

Structured query with optional `subject`, `predicate`, `object` filters. Returns triples with confidence, knowledge type, temporal bounds, and provenance.

**Parameters:** `subject`, `predicate`, `object` (at least one required, all are URIs or literals)

**Response:**
```json
[
  {
    "subject": "http://dbpedia.org/resource/Cold_shock_response",
    "predicate": "http://knowledge.local/schema/increases",
    "object": "http://dbpedia.org/resource/Dopamine",
    "confidence": 0.88,
    "knowledge_type": "Claim",
    "valid_from": null,
    "valid_until": null,
    "provenance": [
      {
        "source_url": "https://example.com/article",
        "source_type": "article",
        "confidence": "0.75",
        "ingested_at": "2026-03-18T10:00:00+00:00"
      }
    ]
  }
]
```

---

### Raw SPARQL Query

```http
POST /api/knowledge/sparql
```

Execute a SPARQL 1.2 SELECT or ASK query against the knowledge graph. Supports RDF-star syntax for querying annotations. Only SELECT and ASK queries are allowed (no INSERT, DELETE, or UPDATE).

Accepts two content types:

**JSON body:**

```json
{
  "query": "SELECT ?s ?p ?o ?conf WHERE { ?s ?p ?o . << ?s ?p ?o >> <http://knowledge.local/schema/confidence> ?conf . FILTER(?conf > 0.8) }"
}
```

**Raw SPARQL body** (`Content-Type: application/sparql-query`):

```sparql
SELECT ?s ?p ?o ?conf WHERE {
  ?s ?p ?o .
  << ?s ?p ?o >> <http://knowledge.local/schema/confidence> ?conf .
  FILTER(?conf > 0.8)
}
```

---

### Contradictions

```http
GET /api/knowledge/contradictions?min_confidence=0.5
```

Surfaces contradictions in the knowledge graph. Detects two patterns:

- **Same predicate, different objects** — e.g., "born in London" vs "born in Paris"
- **Opposite predicates** — e.g., "increases dopamine" vs "decreases dopamine" (via `ks:oppositePredicate` declarations)

Contradiction probability is the product of both claims' confidence scores.

**Parameters:**
- `min_confidence` — filter to pairs where `conf_a × conf_b ≥ threshold` (default 0.0)

**Response:**
```json
[
  {
    "claim_a": {
      "subject": "...",
      "predicate": "...",
      "object": "beneficial",
      "confidence": 0.75
    },
    "claim_b": {
      "subject": "...",
      "predicate": "...",
      "object": "harmful",
      "confidence": 0.6
    },
    "contradiction_probability": 0.45,
    "provenance_a": [...],
    "provenance_b": [...]
  }
]
```

---

### Ask a Question (RAG)

```http
POST /api/ask
Content-Type: application/json
```

Ask a natural language question against the knowledge base. Retrieves relevant content (semantic search) and knowledge graph triples, checks for contradictions, and generates an LLM-powered answer grounded in your data.

```json
{
  "question": "Does cold exposure increase dopamine?",
  "max_sources": 5,
  "min_confidence": 0.3
}
```

**Parameters:**
- `question` (required) — natural language question (max 4000 chars)
- `max_sources` — max content items to retrieve (1–20, default 5)
- `min_confidence` — filter out knowledge triples below this confidence (0.0–1.0, default 0.0)

**Response:**
```json
{
  "answer": "Based on your knowledge base, cold exposure likely increases dopamine...",
  "confidence": 0.88,
  "sources": [
    {
      "url": "https://example.com/article",
      "title": "Cold Exposure and Dopamine",
      "source_type": "article"
    }
  ],
  "knowledge_types_used": ["Claim"],
  "contradictions": [],
  "evidence": [{"triple_subject": "...", "triple_predicate": "...", "triple_object": "...", "chunk_text": "...", "source_url": "..."}],
  "intent": "graph",
  "traversal_depth": 3,
  "inferred_triples": 0
}
```

---

## Admin Panel

A built-in web UI for monitoring and querying your knowledge base. Accessible at `/admin` after logging in.

### Features

- **Dashboard** — stats cards (triples, entities, content, events), confidence distribution chart, knowledge type breakdown, recent ingestion activity
- **Knowledge Explorer** — searchable, filterable, paginated triple browser with entity detail views and content inspection
- **Chat** — ask natural language questions against your knowledge base (uses the RAG pipeline), with source citations and confidence scores
- **Contradictions** — visual side-by-side comparison of conflicting claims with confidence bars

### Authentication

All routes (UI and API) are protected behind a password. Set `ADMIN_PASSWORD` in your `.env` file or environment:

```bash
ADMIN_PASSWORD=your-password-here
```

The service will not start without this variable. Visit `/login` to sign in — no username needed, just the password.

Sessions last 24 hours (signed cookie). Set `SECRET_KEY` for persistent sessions across restarts; if omitted, a random key is generated at startup (sessions lost on restart).

### Tech

Server-rendered Jinja2 templates with Alpine.js and TailwindCSS (CDN). No JS build pipeline — everything ships inside the Python package.

---

## Running Locally

### Prerequisites

- Python 3.12+
- PostgreSQL 16 with [pgvector](https://github.com/pgvector/pgvector) extension
- An LLM provider (Ollama or LiteLLM)

### Option A: Ollama (simplest)

[Ollama](https://ollama.ai) runs models locally with zero configuration.

1. Install Ollama and pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:14b
```

2. Start PostgreSQL (via docker-compose):

```bash
docker compose up -d postgres
```

3. Install and run:

```bash
pip install -e ".[dev]"
cp .env.example .env   # defaults work for Ollama — no changes needed
uvicorn knowledge_service.main:app --reload
```

### Option B: LiteLLM Proxy

[LiteLLM](https://docs.litellm.ai/) provides a unified OpenAI-compatible gateway to 100+ LLM providers.

1. Deploy LiteLLM with the required models in your `litellm_config.yaml`:

```yaml
model_list:
  - model_name: nomic-embed-text
    litellm_params:
      model: ollama/nomic-embed-text
      api_base: http://localhost:11434
  - model_name: qwen3:14b
    litellm_params:
      model: ollama/qwen3:14b
      api_base: http://localhost:11434
```

2. Start LiteLLM:

```bash
litellm --config litellm_config.yaml
```

3. Configure and run:

```bash
pip install -e ".[dev]"
cp .env.example .env
# Edit .env:
#   LLM_BASE_URL=http://localhost:4000
#   LLM_API_KEY=sk-your-litellm-key
uvicorn knowledge_service.main:app --reload
```

### With Docker (full stack)

A pre-built image is available on Docker Hub: [`arshadansari27/knowledge-service`](https://hub.docker.com/r/arshadansari27/knowledge-service)

```bash
# Set admin password (required)
export ADMIN_PASSWORD=changeme

# Using docker-compose (builds locally)
docker compose up -d

# Or pull the pre-built image directly
docker pull arshadansari27/knowledge-service:latest
```

Service available at `http://localhost:8000`. Admin panel at `http://localhost:8000/login`. Ollama must be running on the host machine.

### Configuration

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://knowledge:knowledge@localhost:5433/knowledge` | PostgreSQL connection |
| `LLM_BASE_URL` | `http://localhost:11434` | LLM API endpoint (Ollama or LiteLLM) |
| `LLM_API_KEY` | *(empty)* | API key (leave empty for Ollama) |
| `LLM_EMBED_MODEL` | `nomic-embed-text` | Embedding model (768-dim vectors) |
| `LLM_CHAT_MODEL` | `qwen3:14b` | Chat model for knowledge extraction |
| `LLM_RAG_MODEL` | *(empty)* | RAG answer model (defaults to `LLM_CHAT_MODEL` if empty) |
| `OXIGRAPH_DATA_DIR` | `./data/oxigraph` | RDF store data directory |
| `FEDERATION_ENABLED` | `true` | Enable DBpedia/Wikidata federation |
| `FEDERATION_TIMEOUT` | `3.0` | Federation query timeout (seconds) |
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Port |
| `ADMIN_PASSWORD` | *(required)* | Password for admin panel login and API key auth |
| `SECRET_KEY` | *(required)* | Session signing key |
| `SPACY_DATA_DIR` | `/app/data/spacy` | spaCy Wikidata KB storage directory |
| `MAX_UPLOAD_SIZE` | `52428800` | Maximum file upload size in bytes (default 50MB) |
| `URL_FETCH_TIMEOUT` | `30` | Timeout for URL auto-fetch (seconds) |
| `NLP_ENTITY_CONFIDENCE` | `0.5` | Confidence for spaCy-only fallback entities |
| `READER_EXCLUDE_INFLIGHT` | `true` | Exclude in-flight content from hybrid retrieval |

---

## Bulk Ingest CLI

A standalone script to ingest a directory of files and/or a list of URLs in one go.

### Usage

```bash
# Ingest all supported files in a directory (recursive)
uv run python scripts/bulk_ingest.py ./documents/

# Ingest URLs from a file (one per line, # comments and blank lines ignored)
uv run python scripts/bulk_ingest.py --urls urls.txt

# Both at once, with tags and domain hints
uv run python scripts/bulk_ingest.py ./documents/ --urls urls.txt --tags health,research --domains health

# Preview what would be ingested without doing it
uv run python scripts/bulk_ingest.py ./documents/ --dry-run
```

### Supported file types

`.pdf`, `.html`, `.htm`, `.txt`, `.md`, `.json`, `.csv`

Files are uploaded via `POST /api/content/upload`. URLs are submitted via `POST /api/content` (server auto-fetches).

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `path` (positional) | — | Directory to scan recursively |
| `--urls FILE` | — | Text file with one URL per line |
| `--server URL` | `KNOWLEDGE_URL` env or `http://localhost:8000` | Target server |
| `--api-key KEY` | `KNOWLEDGE_API_KEY` env | API key for authentication |
| `--tags t1,t2` | — | Comma-separated tags for all items |
| `--domains d1,d2` | — | Comma-separated domain hints for extraction |
| `--dry-run` | — | List items without ingesting |
| `--poll-timeout N` | `300` | Seconds to wait per item before giving up |

At least one of `path` or `--urls` is required.

### How it works

Items are processed sequentially. For each item, the script POSTs to the server, then polls `GET /api/content/{id}/status` every 5 seconds until the job completes, fails, or times out. Progress is printed as it goes:

```
Bulk ingest: 6 items (5 files, 1 URLs)
Target: http://localhost:8000

  [1/6] report.pdf .................. OK  triples=14  23s
  [2/6] notes.txt ................... OK  triples=6   8s
  [3/6] https://example.com/article . FAIL  extraction failed  45s
  ...

Results: 4 passed, 1 failed, 1 skipped (total: 4m 32s)
```

Exit code is 0 if all items passed, 1 if any failed.

### Against production

```bash
KNOWLEDGE_URL=https://knowledge.hikmahtech.in \
KNOWLEDGE_API_KEY=your-key \
  uv run python scripts/bulk_ingest.py ./documents/ --poll-timeout 600
```

### Against local docker-compose

```bash
docker compose up -d
uv run python scripts/bulk_ingest.py ./documents/ --api-key changeme
```

---

## Running Tests

```bash
pytest
```

All tests mock external dependencies — no PostgreSQL or LLM provider required.

---

## CI/CD

GitHub Actions pipeline on every push/merge to `main`:

1. **Lint** — `ruff check` + `ruff format --check`
2. **Test** — `pytest tests/ -v` (660+ tests)
3. **Version bump** — auto-increments patch version in `pyproject.toml`, commits back to `main`, creates `vX.Y.Z` git tag
4. **Docker build** — builds and pushes to Docker Hub as `arshadansari27/knowledge-service:X.Y.Z` and `:latest`

Version is read from `pyproject.toml` and used as the Docker image tag and git tag. Bump commits include `[skip ci]` to prevent infinite loops.

Pull requests run lint + test only (no version bump or Docker push).

---

## Project Structure

```
src/knowledge_service/
├── main.py                  # FastAPI app factory + lifespan
├── config.py                # Settings (pydantic-settings, .env)
├── models.py                # Pydantic models for all 7 knowledge types + API contracts
├── _utils.py                # Shared RDF helpers + JSON extraction from LLM output
├── chunking.py              # Markdown-aware text splitting with section headers
├── admin/
│   ├── auth.py              # AuthMiddleware, login/logout, rate limiter, session cookies
│   ├── routes.py            # Admin page routes (dashboard, knowledge, chat, contradictions)
│   ├── stats.py             # /api/admin/stats/* and /api/admin/knowledge/triples endpoints
│   ├── jobs.py              # /api/admin/jobs
│   └── templates/           # Jinja2 templates (base, dashboard, knowledge, chat, etc.)
├── api/
│   ├── content.py           # POST /api/content (JSON + URL auto-fetch)
│   ├── upload.py            # POST /api/content/upload (multipart file upload)
│   ├── claims.py            # POST /api/claims
│   ├── search.py            # GET /api/search
│   ├── knowledge.py         # GET /api/knowledge/query, POST /api/knowledge/sparql
│   ├── contradictions.py    # GET /api/knowledge/contradictions
│   ├── ask.py               # POST /api/ask (RAG question answering)
│   ├── changes.py           # GET /api/entity/{id}/changes
│   └── health.py            # GET /health
├── parsing/
│   ├── __init__.py          # ParserRegistry, ParsedDocument, Parser protocol
│   ├── pdf.py               # PdfParser (PyMuPDF)
│   ├── html.py              # HtmlParser (readability-lxml + BeautifulSoup)
│   ├── structured.py        # StructuredParser (JSON/CSV)
│   ├── image.py             # ImageParser (stub for future OCR)
│   └── text.py              # TextParser (passthrough)
├── nlp/
│   ├── __init__.py          # NlpPhase, NlpResult, NlpEntity
│   └── bootstrap.py         # spaCy model + Wikidata KB loading
├── ingestion/
│   ├── pipeline.py          # Per-triple processing (delta, insert, contradiction, provenance, inference)
│   ├── worker.py            # 5-phase orchestrator (Embed → NLP → Extract → Coref → Process)
│   ├── phases.py            # EmbedPhase, ExtractPhase, ProcessPhase
│   ├── coreference.py       # CoreferencePhase (Wikidata QID merging)
│   └── federation.py        # FederationPhase (DBpedia/Wikidata entity enrichment)
├── stores/
│   ├── __init__.py          # Stores dataclass
│   ├── triples.py           # pyoxigraph wrapper — RDF-star, named graphs
│   ├── content.py           # ContentStore — metadata + chunks + embeddings
│   ├── entities.py          # EntityStore — entity/predicate resolution + aliases
│   ├── provenance.py        # ProvenanceStore — per-source evidence rows
│   ├── rag.py               # RAGRetriever — hybrid chunk + KG retrieval, intent-routed
│   └── graph_migration.py   # One-time migration to named graphs
├── reasoning/
│   ├── engine.py            # InferenceEngine — forward-chaining rules
│   └── noisy_or.py          # Noisy-OR evidence combination
├── ontology/
│   ├── uri.py               # URI normalization (to_entity_uri, to_predicate_uri, slugify)
│   ├── namespaces.py        # ks:, schema:, dc:, skos:, prov: namespace constants
│   ├── registry.py          # DomainRegistry — predicate metadata from ontology
│   ├── bootstrap.py         # Loads schema.ttl + domains/*.ttl into ks:graph/ontology
│   ├── schema.ttl            # Knowledge type classes, properties
│   ├── domains/             # Domain TTL files (base, health, technology, research)
│   └── prompts/             # LLM extraction prompt templates (entities, relations)
├── clients/
│   ├── llm.py               # EmbeddingClient + ExtractionClient (two-phase)
│   ├── prompt_builder.py    # Domain-aware extraction prompts from templates
│   ├── rag.py               # RAGClient — LLM-powered answer generation
│   └── federation.py        # FederationClient — DBpedia/Wikidata SPARQL federation
└── migrations/              # SQL migrations (auto-applied at startup)
```

---

## Ontology

The system reuses established vocabularies and keeps the custom `ks:` namespace minimal:

| Domain | Namespace | Purpose |
|--------|-----------|---------|
| Content metadata | `dc:` / `dcterms:` | Title, creator, date, format, source |
| General entities | `schema:` (Schema.org) | People, organisations, software, events |
| Topic hierarchies | `skos:` | Broader/narrower relationships, labels |
| People and social | `foaf:` | Personal information, social connections |
| Provenance | `prov:` (PROV-O) | Activity, agent, entity chains |
| Service-specific | `ks:` | Confidence, knowledge type, temporal validity, extraction metadata |

---

## Status

Deployed to production (~640 tests).

| Capability | What |
|------------|------|
| Knowledge model | 7 knowledge types (Claim / Fact / Event / Entity / Relationship / Conclusion / TemporalState), RDF-star confidence, temporal validity |
| RDF store | pyoxigraph, 5 named graphs by provenance class (ontology / asserted / extracted / inferred / federated) |
| Ingestion | Parse (PDF / HTML / CSV / JSON / text) → chunk → embed → spaCy NER + Wikidata linking → LLM extraction → QID-based coreference → ingest |
| Hybrid retrieval | BM25 (OR-tokenised `to_tsquery`) + pgvector, fused via Reciprocal Rank Fusion |
| RAG endpoint | `/api/ask` with intent classification (semantic / entity / graph), returns answer + source chunks + triples + contradictions |
| Evidence combination | Noisy-OR across multi-source confidences: `1 - Π(1 − cᵢ)` |
| Inference | 3 forward-chaining rules (inverse / transitive / type-inheritance), retraction cascade when source triples change |
| Consistency | Outbox 2PC between pyoxigraph and Postgres, startup drainer recovers from crash-between-commit-and-drain |
| Reader-side filter | In-flight content excluded from retrieval until ingestion reaches a terminal status |

**Not currently enforced** (declared but not filtered on the read path): graph-level trust tiers, contradictions. Both are surfaced as labels / response fields, not used to re-rank or exclude results.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Deployment](docs/deployment.md) | Production AEGIS stack deployment |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | Python 3.12, FastAPI, uvicorn |
| Knowledge store | pyoxigraph (embedded, RDF 1.2, SPARQL 1.2, RocksDB) |
| Confidence combination | Noisy-OR (4-line function) |
| Inference | 3 forward-chaining rules at ingestion time (inverse / transitive / type-inheritance) |
| Operational store | PostgreSQL 16 |
| Vector search | pgvector (HNSW index, halfvec) |
| Document parsing | PyMuPDF (PDF), readability-lxml + BeautifulSoup (HTML), stdlib (CSV/JSON) |
| NLP pre-pass | spaCy (en_core_web_sm) + spacy-entity-linker (Wikidata KB) |
| LLM gateway | Ollama (local) or LiteLLM (proxy) — any OpenAI-compatible API |
| Embeddings | nomic-embed-text (768-dim) |
| Knowledge extraction | qwen3:14b (auto-extracts from raw_text) |
| Infrastructure | Docker Compose, GitHub Actions CI/CD, Docker Swarm (production) |
