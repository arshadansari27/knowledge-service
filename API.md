# Knowledge Service API Reference

**Base URL:** `http://localhost:8000`
**Interactive Docs:** `http://localhost:8000/docs` (auto-generated OpenAPI/Swagger)
**Authentication:** None (all endpoints are public)
**Response Format:** JSON (`application/json`)
**Versioning:** Patch version auto-incremented on every push to `main` (see `pyproject.toml`)

---

## Endpoints Overview

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check for all dependencies |
| POST | `/api/content` | Ingest content with knowledge items |
| POST | `/api/claims` | Ingest knowledge items directly |
| GET | `/api/search` | Semantic similarity search |
| GET | `/api/knowledge/query` | Structured knowledge graph query |
| POST | `/api/knowledge/sparql` | Raw SPARQL query |
| GET | `/api/knowledge/contradictions` | Detect contradictions |
| POST | `/api/ask` | RAG-powered question answering |

---

## GET /health

Check the health of all service dependencies.

**Response:**

```json
{
  "status": "ok | degraded",
  "components": {
    "oxigraph": "ok | error: ...",
    "postgresql": "ok | error: ...",
    "llm": "ok | error: ..."
  }
}
```

---

## POST /api/content

Ingest a piece of content (webpage, article, video) with associated knowledge items. Generates an embedding for semantic search, stores content in PostgreSQL, and writes knowledge items to the RDF graph with provenance.

If `knowledge` is empty but `raw_text` is provided, knowledge is auto-extracted via LLM.

**Request Body:**

```json
{
  "url": "string (required)",
  "title": "string (required)",
  "summary": "string (optional)",
  "raw_text": "string (optional)",
  "source_type": "string (required) — e.g. 'article', 'video', 'paper'",
  "tags": ["string"] (optional, default: []),
  "metadata": {} (optional, default: {}),
  "knowledge": [KnowledgeInput] (optional, default: [])
}
```

**Response:**

```json
{
  "content_id": "UUID string",
  "triples_created": 5,
  "contradictions_detected": [
    {
      "subject": "ks:caffeine",
      "predicate": "ks:affects",
      "existing_object": "improved alertness",
      "existing_confidence": 0.85,
      "new_object": "no effect on alertness",
      "new_confidence": 0.6
    }
  ],
  "entities_resolved": 2
}
```

**Status Codes:** `200` OK, `422` Validation Error

### Example

```bash
curl -X POST http://localhost:8000/api/content \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "title": "Effects of Caffeine on Sleep",
    "source_type": "article",
    "tags": ["health", "sleep"],
    "knowledge": [
      {
        "knowledge_type": "Claim",
        "subject": "ks:caffeine",
        "predicate": "ks:affects",
        "object": "sleep quality",
        "confidence": 0.9
      },
      {
        "knowledge_type": "Entity",
        "uri": "ks:caffeine",
        "rdf_type": "schema:ChemicalSubstance",
        "label": "Caffeine"
      }
    ]
  }'
```

---

## POST /api/claims

Ingest knowledge items directly without storing raw content. Useful for programmatic ingestion from structured sources.

**Request Body:**

```json
{
  "source_url": "string (required)",
  "source_type": "string (required) — e.g. 'paper', 'database'",
  "extractor": "string (required) — e.g. 'llm_qwen3:14b', 'api'",
  "knowledge": [KnowledgeInput] (required)
}
```

**Response:**

```json
{
  "triples_created": 3,
  "contradictions_detected": []
}
```

**Status Codes:** `200` OK, `422` Validation Error

### Example

```bash
curl -X POST http://localhost:8000/api/claims \
  -H "Content-Type: application/json" \
  -d '{
    "source_url": "https://pubmed.ncbi.nlm.nih.gov/12345",
    "source_type": "paper",
    "extractor": "manual",
    "knowledge": [
      {
        "knowledge_type": "Fact",
        "subject": "ks:vitamin_d",
        "predicate": "ks:supports",
        "object": "bone health",
        "confidence": 0.95
      }
    ]
  }'
```

---

## GET /api/search

Search ingested content by semantic similarity using pgvector cosine distance. Returns chunk-level results — each result is a chunk of a document, not the full document.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `q` | string | yes | — | Search query text |
| `limit` | int | no | 10 | Max results (1–100) |
| `source_type` | string | no | — | Filter by source type |
| `tags` | string[] | no | — | Filter by tags (repeat for multiple) |

**Response:**

```json
[
  {
    "content_id": "UUID string",
    "url": "https://example.com/article",
    "title": "Effects of Caffeine on Sleep",
    "summary": "A study on caffeine...",
    "similarity": 0.87,
    "source_type": "article",
    "tags": ["health", "sleep"],
    "ingested_at": "2025-01-15T10:30:00Z",
    "chunk_text": "The relevant section of the document matching the query...",
    "chunk_index": 0
  }
]
```

| Field | Description |
|-------|-------------|
| `chunk_text` | The text of the matching chunk (always present) |
| `chunk_index` | Position of this chunk within its parent document (0-indexed) |

### Example

```bash
curl "http://localhost:8000/api/search?q=caffeine+sleep&limit=5&source_type=article"
```

---

## GET /api/knowledge/query

Structured query of the RDF knowledge graph. Returns triples with confidence, knowledge type, temporal bounds, and provenance.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `subject` | string | at least one required | Filter by subject URI |
| `predicate` | string | at least one required | Filter by predicate URI |
| `object` | string | at least one required | Filter by object URI or literal |

At least one of `subject`, `predicate`, or `object` must be provided.

**Response:**

```json
[
  {
    "subject": "ks:caffeine",
    "predicate": "ks:affects",
    "object": "sleep quality",
    "confidence": 0.9,
    "knowledge_type": "Claim",
    "valid_from": "2025-01-01",
    "valid_until": null,
    "provenance": [
      {
        "source_url": "https://example.com/article",
        "source_type": "article",
        "extractor": "llm_qwen3:14b",
        "confidence": "0.9",
        "ingested_at": "2025-01-15T10:30:00Z",
        "metadata": {},
        "valid_from": null,
        "valid_until": null
      }
    ],
    "source": null
  }
]
```

The `source` field is populated only for federated results (e.g. `"dbpedia"` or `"wikidata"`).

**Status Codes:** `200` OK, `422` Validation Error (no parameters provided)

### Example

```bash
curl "http://localhost:8000/api/knowledge/query?subject=ks:caffeine"
```

---

## POST /api/knowledge/sparql

Execute arbitrary SPARQL SELECT queries against the knowledge graph. Supports SPARQL 1.2 and RDF-star syntax.

**Request Body (JSON):**

```json
{
  "query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
}
```

**Alternative:** Send raw SPARQL with `Content-Type: application/sparql-query`.

**Response:**

```json
[
  {
    "s": "http://knowledge.service/ks/caffeine",
    "p": "http://knowledge.service/ks/affects",
    "o": "sleep quality"
  }
]
```

**Status Codes:** `200` OK, `422` Validation Error (empty query)

### Example

```bash
curl -X POST http://localhost:8000/api/knowledge/sparql \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"}'
```

---

## GET /api/knowledge/contradictions

Detect contradictions in the knowledge graph. Finds two patterns:
- **Same predicate, different objects** (e.g. "born in London" vs "born in Paris")
- **Opposite predicates** (e.g. "increases" vs "decreases" via `ks:oppositePredicate`)

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_confidence` | float | no | 0.0 | Minimum confidence product threshold |

**Response:**

```json
[
  {
    "claim_a": {
      "subject": "ks:caffeine",
      "predicate": "ks:affects",
      "object": "improved alertness",
      "confidence": 0.85
    },
    "claim_b": {
      "subject": "ks:caffeine",
      "predicate": "ks:affects",
      "object": "no effect on alertness",
      "confidence": 0.6
    },
    "contradiction_probability": 0.51,
    "provenance_a": [...],
    "provenance_b": [...]
  }
]
```

### Example

```bash
curl "http://localhost:8000/api/knowledge/contradictions?min_confidence=0.5"
```

---

## POST /api/ask

Ask a natural language question against the knowledge base. Retrieves relevant content (semantic search) and knowledge graph triples, checks for contradictions, and generates an LLM-powered answer grounded in your data.

**Request Body:**

```json
{
  "question": "string (required, max 4000 chars)",
  "max_sources": 5,
  "min_confidence": 0.0
}
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `question` | string | yes | — | Natural language question (max 4000 chars) |
| `max_sources` | int | no | 5 | Max content items to retrieve (1–100) |
| `min_confidence` | float | no | 0.0 | Filter out knowledge triples below this confidence (0.0–1.0) |

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
  "contradictions": [
    {
      "subject": "http://dbpedia.org/resource/Cold_shock_response",
      "predicate": "http://knowledge.local/schema/decreases",
      "object": "http://dbpedia.org/resource/Dopamine",
      "confidence": 0.3
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `answer` | string | LLM-generated natural language response |
| `confidence` | float \| null | Highest confidence among supporting knowledge triples. `null` if no triples found. |
| `sources` | array | Deduplicated content sources used in retrieval |
| `knowledge_types_used` | string[] | Which of the 7 knowledge types contributed to the answer |
| `contradictions` | array | Conflicting claims with subject, predicate, object, and confidence |

**Status Codes:** `200` OK, `422` Validation Error, `502` LLM Service Error

### Example

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Does cold exposure increase dopamine?",
    "max_sources": 5,
    "min_confidence": 0.3
  }'
```

---

## Knowledge Types Reference

Knowledge items are sent in the `knowledge` array of `/api/content` and `/api/claims`. Each item has a `knowledge_type` discriminator field.

### Claim

A statement with a confidence score. General-purpose triple.

```json
{
  "knowledge_type": "Claim",
  "subject": "ks:caffeine",
  "predicate": "ks:affects",
  "object": "sleep quality",
  "confidence": 0.85,
  "valid_from": "2025-01-01",
  "valid_until": null
}
```

### Fact

A high-confidence claim. Confidence must be >= 0.9.

```json
{
  "knowledge_type": "Fact",
  "subject": "ks:earth",
  "predicate": "ks:orbits",
  "object": "ks:sun",
  "confidence": 1.0
}
```

### Relationship

A directional link between two entities.

```json
{
  "knowledge_type": "Relationship",
  "subject": "ks:python",
  "predicate": "ks:influencedBy",
  "object": "ks:abc_language",
  "confidence": 0.95
}
```

### Event

A timestamped occurrence.

```json
{
  "knowledge_type": "Event",
  "subject": "ks:moon_landing",
  "occurred_at": "1969-07-20",
  "confidence": 1.0,
  "properties": {
    "location": "Sea of Tranquility",
    "crew": "Apollo 11"
  }
}
```

### Entity

A typed entity with ontology class and label.

```json
{
  "knowledge_type": "Entity",
  "uri": "ks:python",
  "rdf_type": "schema:ProgrammingLanguage",
  "label": "Python",
  "properties": {
    "creator": "Guido van Rossum"
  },
  "confidence": 0.95
}
```

### Conclusion

A derived statement with a reasoning chain.

```json
{
  "knowledge_type": "Conclusion",
  "concludes": "Caffeine disrupts deep sleep phases",
  "derived_from": ["hash1", "hash2"],
  "inference_method": "bayesian_combination",
  "confidence": 0.78
}
```

### TemporalState

A time-bounded property value. Both `valid_from` and `valid_until` are required, and `valid_until` must be >= `valid_from`.

```json
{
  "knowledge_type": "TemporalState",
  "subject": "ks:tesla",
  "property": "ks:ceo",
  "value": "Elon Musk",
  "valid_from": "2008-10-01",
  "valid_until": "2025-12-31",
  "confidence": 1.0
}
```

---

## Configuration

The service is configured via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://knowledge:knowledge@localhost:5433/knowledge` | PostgreSQL connection string |
| `LLM_BASE_URL` | `http://localhost:11434` | LLM API endpoint (Ollama or LiteLLM) |
| `LLM_API_KEY` | `""` | API key for LLM (empty for Ollama) |
| `LLM_EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `LLM_CHAT_MODEL` | `qwen3:14b` | Chat/extraction model name |
| `LLM_RAG_MODEL` | `""` | RAG answer model (defaults to `LLM_CHAT_MODEL` if empty) |
| `OXIGRAPH_DATA_DIR` | `./data/oxigraph` | RDF store data directory |
| `PROBLOG_RULES_DIR` | `./src/knowledge_service/reasoning/rules` | ProbLog rules directory |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8000` | API port |
| `FEDERATION_ENABLED` | `true` | Enable DBpedia/Wikidata federation |
| `FEDERATION_TIMEOUT` | `3.0` | Federation query timeout (seconds) |
