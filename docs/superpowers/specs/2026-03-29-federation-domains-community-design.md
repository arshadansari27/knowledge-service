# Federation, Domain Ontology, Community Detection & Smoke Test Design

**Date:** 2026-03-29
**Status:** Approved

## Motivation

The knowledge graph is feature-complete but data-starved (16 triples) and has unused capabilities. This design activates three dormant features (federation, domain ontology, community detection) and adds production verification tooling.

## 1. Federation Enrichment (Background Post-processing)

### Approach

After each successful ingestion job, a background task enriches new entities with external knowledge from DBpedia/Wikidata via the existing `FederationClient`.

### How it works

1. At the end of `run_ingestion()`, after the job completes successfully, queue federation enrichment
2. Query `entity_embeddings` for entities without a corresponding `owl:sameAs` triple in the graph
3. For each unlinked entity (max 10 per job), call `FederationClient.lookup_entity(label)`
4. If found: insert `owl:sameAs` triple + external type/description into `ks:graph/asserted` at confidence 0.95
5. Rate-limited: 1s delay between calls to respect public SPARQL endpoints
6. Controlled by `federation_enabled` config (default: `true`)
7. Errors are logged but don't affect the ingestion job status (already completed)

### Changes

- **`main.py`**: Instantiate `FederationClient` in lifespan, store on `app.state.federation_client`
- **`ingestion/worker.py`**: After `tracker.complete()`, call `_enrich_entities()` if federation enabled
- **New `ingestion/federation.py`**: `FederationPhase` class with `run(knowledge_items, triple_store, entity_store, federation_client)` method
- **`config.py`**: `federation_enabled` and `federation_timeout` already exist — no changes needed

### Named graph

Federation triples go into `ks:graph/asserted` (authoritative external sources, not LLM-derived).

### Rate limiting

- Max 10 entity lookups per ingestion job
- 1s delay between lookups
- 3s timeout per SPARQL request (existing `federation_timeout` config)
- Skip entities that already have `owl:sameAs` triples

## 2. Domain Ontology Files

Three new `.ttl` files in `ontology/domains/`. No code changes — auto-loaded by bootstrap at startup, auto-registered by DomainRegistry.

### `health.ttl` (~15 predicates)

Domain: `"health"`. Covers supplements, compounds, conditions, biomarkers.

| Predicate | Opposite | Transitive | Materiality | Key Synonyms |
|-----------|----------|------------|-------------|-------------|
| `treats` | — | — | 0.8 | remedies, alleviates, addresses |
| `has_side_effect` | — | — | 0.7 | adverse_effect, causes_side_effect |
| `recommended_dose` | — | — | 0.6 | dosage, suggested_intake |
| `contraindicated_with` | — | — | 0.8 | interacts_negatively_with, unsafe_with |
| `affects_biomarker` | — | — | 0.7 | modulates, influences_level_of |
| `improves_condition` | `worsens_condition` | — | 0.7 | helps_with, beneficial_for, alleviates |
| `worsens_condition` | `improves_condition` | — | 0.7 | aggravates, exacerbates |
| `has_mechanism` | — | — | 0.5 | works_by, mechanism_of_action |
| `metabolized_by` | — | — | 0.5 | processed_by, broken_down_by |
| `absorbs_with` | — | — | 0.5 | enhanced_by, bioavailable_with |
| `depletes` | — | — | 0.6 | reduces_levels_of, lowers |
| `synergizes_with` | — | — | 0.6 | works_well_with, potentiates |
| `has_half_life` | — | — | 0.4 | duration, active_for |
| `evidence_level` | — | — | 0.5 | evidence_quality, study_quality |
| `bioavailable_as` | — | — | 0.5 | absorbed_as, active_form |

### `technology.ttl` (~12 predicates)

Domain: `"technology"`. Covers software, frameworks, services, infrastructure.

| Predicate | Opposite | Transitive | Materiality | Key Synonyms |
|-----------|----------|------------|-------------|-------------|
| `implements` | — | — | 0.6 | provides, supports_feature |
| `integrates_with` | — | — | 0.6 | connects_to, works_with |
| `replaces` | — | — | 0.7 | supersedes, deprecates |
| `requires` | — | yes | 0.6 | depends_on, needs |
| `compatible_with` | `incompatible_with` | — | 0.5 | supports, works_on |
| `incompatible_with` | `compatible_with` | — | 0.7 | breaks_with, conflicts_with |
| `performs_better_than` | — | — | 0.5 | faster_than, outperforms |
| `maintained_by` | — | — | 0.4 | developed_by, owned_by |
| `licensed_as` | — | — | 0.4 | license, open_source |
| `written_in` | — | — | 0.5 | built_with, language |
| `deployed_on` | — | — | 0.5 | runs_on, hosted_on |
| `alternative_to` | — | — | 0.5 | similar_to, competitor_of |

### `research.ttl` (~10 predicates)

Domain: `"research"`. Covers papers, studies, findings, methodology.

| Predicate | Opposite | Transitive | Materiality | Key Synonyms |
|-----------|----------|------------|-------------|-------------|
| `cites` | — | — | 0.5 | references, builds_on |
| `contradicts_finding` | — | — | 0.8 | disagrees_with, refutes |
| `replicates` | — | — | 0.7 | confirms, validates |
| `extends` | — | — | 0.6 | builds_upon, expands |
| `funded_by` | — | — | 0.4 | supported_by, sponsored_by |
| `authored_by` | — | — | 0.5 | written_by, lead_author |
| `published_in` | — | — | 0.4 | appears_in, journal |
| `has_sample_size` | — | — | 0.5 | participants, n_equals |
| `uses_methodology` | — | — | 0.5 | method, study_design |
| `finds` | — | — | 0.7 | concludes, shows, demonstrates |

## 3. Auto-enable Community Detection

### Approach

Replace the blunt timer (`community_rebuild_interval`) with a data-driven trigger. After each ingestion job that creates triples, check if conditions are met for a community rebuild.

### Trigger conditions (all must be true)

1. `triples_created > 0` in the just-completed job
2. Total triple count in graph >= `community_min_triples` (default: 50)
3. Last community rebuild was more than `community_cooldown` seconds ago (default: 3600 = 1 hour)

### Changes

- **`config.py`**: Add `community_min_triples: int = 50` and `community_cooldown: int = 3600`
- **`ingestion/worker.py`**: After `tracker.complete()`, check conditions and trigger `_maybe_rebuild_communities()` in background
- **`main.py`**: Store last rebuild timestamp on `app.state._last_community_rebuild` (initialized to 0)
- The existing periodic loop (`community_rebuild_interval`) stays as-is for users who prefer timer-based. The new threshold trigger is additive.
- The existing manual admin endpoint (`POST /api/admin/rebuild-communities`) stays unchanged.

### Community rebuild flow (unchanged)

1. `CommunityDetector.detect()` — Leiden algorithm on entity graph
2. `CommunitySummarizer.summarize_one()` — LLM labels each community
3. `CommunityStore.replace_all()` — atomic replace in PostgreSQL

## 4. Production Smoke Test Script

### `scripts/smoke_test.py`

Standalone Python script (not pytest) that verifies the full pipeline works in production.

### What it does

1. **Health check** — `GET /health`, verify all components OK
2. **Ingest raw text** — `POST /api/content` with a health-domain paragraph, poll to completion
3. **Upload HTML** — `POST /api/content/upload` with sample HTML, poll to completion
4. **Upload PDF** — `POST /api/content/upload` with sample PDF, poll to completion
5. **Check stats** — `GET /api/admin/stats/counts`, verify triples increased
6. **Ask a question** — `POST /api/ask`, verify non-empty answer
7. **Query knowledge graph** — `GET /api/knowledge/query`, verify triples exist
8. Print pass/fail summary with timing

### Configuration

```bash
# Defaults to production
KNOWLEDGE_URL=https://knowledge.hikmahtech.in
KNOWLEDGE_API_KEY=<admin_password>

# Run
uv run python scripts/smoke_test.py
```

### Test content

Hardcoded paragraphs covering all 3 domains (health, technology, research) so the new ontology files get exercised. Plus the existing fixture files (sample.html, sample.pdf) for upload testing.

## What doesn't change

- Existing ingestion pipeline flow (Parse → Chunk → Embed → NLP → Extract → Coreference → Process)
- API contracts
- Database schema (no new migrations)
- Docker image dependencies
