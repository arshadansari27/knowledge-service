# Knowledge Service

[![CI](https://github.com/arshadansari27/knowledge-service/actions/workflows/ci.yml/badge.svg)](https://github.com/arshadansari27/knowledge-service/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/docker/v/arshadansari27/knowledge-service?label=docker&sort=semver)](https://hub.docker.com/r/arshadansari27/knowledge-service)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A personal knowledge graph with Bayesian epistemics. Ingests content you encounter, structures it using established ontologies, reasons over it with probabilistic logic, and makes the resulting knowledge queryable via SPARQL and semantic search — without requiring deliberate organisation from the user.

Built by [Hikmah Technologies](https://hikmahtechnologies.com) | [@hikmahtech](https://x.com/hikmahtech) | [@arshadansari27](https://x.com/arshadansari27)

The primary consumer is AEGIS, where AI agents gain awareness of your accumulated knowledge. The service also stands alone as a reusable knowledge API.

> **The ontology is the product. Sources are just input channels.**

---

## What Makes This Different

Every "second brain" tool treats all content as equal — a bookmark, a note, a highlight are all flat objects with tags. No confidence. No provenance. No temporal validity. No contradiction detection. No inference.

This system separates **content** (what you consumed) from **knowledge** (what you derived from it), and models knowledge with:

- **Uncertainty** — claims carry Bayesian probability, not boolean truth
- **Provenance** — every triple traces back to its source, extraction method, and timestamp
- **Temporality** — knowledge has `valid_from` / `valid_until`, not just `created_at`
- **Ontological structure** — concepts link to established vocabularies (Schema.org, Dublin Core, SKOS) so "PostgreSQL" in your codebase and "PostgreSQL" in an article are the same entity
- **Inference** — derived conclusions preserve their reasoning chain

---

## Architecture

Single FastAPI process with embedded components. No microservices.

```
FastAPI Process
├── KnowledgeStore     pyoxigraph — RDF 1.2 triplestore, RDF-star, SPARQL 1.2 (disk-backed via RocksDB)
├── ReasoningEngine    ProbLog — Bayesian inference, Noisy-OR evidence combination
├── EmbeddingClient    LLM API (Ollama or LiteLLM) — embeddings (nomic-embed-text, 768-dim)
├── ExtractionClient   LLM API (Ollama or LiteLLM) — knowledge extraction (qwen3:14b)
├── EmbeddingStore     PostgreSQL + pgvector — semantic similarity search
└── ProvenanceStore    PostgreSQL — source tracking and evidence trail

PostgreSQL
├── content            Raw content rows with embeddings (pgvector halfvec)
├── provenance         Per-source evidence rows linked to triples by SHA-256 hash
└── ingestion_events   Append-only audit log
```

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
| **Conclusion** | Derived, reasoning chain preserved | "Cold exposure likely increases dopamine" — Bayesian combination of 3 sources |
| **TemporalState** | Time-bounded property (valid_until required) | Bitcoin price $X between date A and date B |

---

## Confidence Model

**Two-layer design:**

1. **RDF-star annotation on each triple** — the system's current Bayesian belief:
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

The combined value is written back to the RDF-star annotation. ProbLog propagates these probabilities through inference chains for derived conclusions.

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

Ingest a piece of content with its associated knowledge. Generates an embedding for semantic search, stores the content in PostgreSQL, and writes all knowledge items to the RDF graph with provenance.

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

**Response:**
```json
{
  "content_id": "uuid",
  "triples_created": 3,
  "contradictions_detected": [],
  "entities_resolved": 0
}
```

---

### Ingest Claims Directly

```http
POST /api/claims
Content-Type: application/json
```

Ingest knowledge items without storing raw content. Useful for programmatic ingestion where content storage is not needed.

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

Searches ingested content by semantic similarity using pgvector cosine distance. Returns content rows ranked by similarity to the query.

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
    "ingested_at": "2026-03-18T10:00:00Z"
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

Execute any SPARQL 1.2 SELECT query directly against the knowledge graph. Supports RDF-star syntax for querying annotations.

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
# Using docker-compose (builds locally)
docker compose up -d

# Or pull the pre-built image directly
docker pull arshadansari27/knowledge-service:latest
```

Service available at `http://localhost:8000`. Ollama must be running on the host machine.

### Configuration

All settings via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://knowledge:knowledge@localhost:5433/knowledge` | PostgreSQL connection |
| `LLM_BASE_URL` | `http://localhost:11434` | LLM API endpoint (Ollama or LiteLLM) |
| `LLM_API_KEY` | *(empty)* | API key (leave empty for Ollama) |
| `LLM_EMBED_MODEL` | `nomic-embed-text` | Embedding model (768-dim vectors) |
| `LLM_CHAT_MODEL` | `qwen3:14b` | Chat model for knowledge extraction |
| `OXIGRAPH_DATA_DIR` | `./data/oxigraph` | RDF store data directory |
| `PROBLOG_RULES_DIR` | `./src/knowledge_service/reasoning/rules` | ProbLog rules |
| `FEDERATION_ENABLED` | `true` | Enable DBpedia/Wikidata federation |
| `FEDERATION_TIMEOUT` | `3.0` | Federation query timeout (seconds) |
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Port |

---

## Running Tests

```bash
pytest
```

All tests mock external dependencies — no PostgreSQL or LLM provider required.

---

## Project Structure

```
src/knowledge_service/
├── main.py                  # FastAPI app factory + lifespan
├── config.py                # Settings (pydantic-settings, .env)
├── models.py                # Pydantic models for all 7 knowledge types + API contracts
├── api/
│   ├── content.py           # POST /api/content
│   ├── claims.py            # POST /api/claims
│   ├── search.py            # GET /api/search
│   ├── knowledge.py         # GET /api/knowledge/query, POST /api/knowledge/sparql
│   ├── contradictions.py    # GET /api/knowledge/contradictions
│   └── health.py            # GET /health
├── stores/
│   ├── knowledge.py         # pyoxigraph wrapper — RDF-star inserts, SPARQL queries
│   ├── provenance.py        # PostgreSQL provenance table (asyncpg)
│   ├── embedding.py         # pgvector content store + cosine similarity search
│   └── entity_resolver.py   # Embedding-based entity deduplication
├── reasoning/
│   ├── engine.py            # ProbLog wrapper — Noisy-OR, contradiction detection
│   └── rules/base.pl        # Core ProbLog rules
├── ontology/
│   ├── namespaces.py        # ks:, schema:, dc:, skos:, prov: namespace constants
│   └── bootstrap.py        # Loads base ontology into pyoxigraph on startup
└── clients/
    ├── llm.py               # EmbeddingClient + ExtractionClient (httpx → LLM API)
    └── federation.py        # FederationClient — DBpedia/Wikidata SPARQL federation
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

| Phase | Status | What |
|-------|--------|------|
| **Phase 1** | ✅ Complete | Knowledge model, RDF store, probabilistic reasoning, all 7 types, Docker |
| **Phase 2** | ✅ Complete | DBpedia/Wikidata federation (ingestion-time + query-time) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | Python 3.12, FastAPI, uvicorn |
| Knowledge store | pyoxigraph (embedded, RDF 1.2, SPARQL 1.2, RocksDB) |
| Reasoning | ProbLog 2.2 (probabilistic logic programming) |
| Operational store | PostgreSQL 16 |
| Vector search | pgvector (HNSW index, halfvec) |
| LLM gateway | Ollama (local) or LiteLLM (proxy) — any OpenAI-compatible API |
| Embeddings | nomic-embed-text (768-dim) |
| Knowledge extraction | qwen3:14b (auto-extracts from raw_text) |
| Infrastructure | Docker Compose |
