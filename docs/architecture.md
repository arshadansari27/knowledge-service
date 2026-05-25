# Architecture notes

This document is the design-rationale companion to the README. The README tells you *what* the service does; this tells you *why* the interesting pieces are shaped the way they are. Read it if you want to understand the non-obvious choices — and the ones I regret.

The headline stories are [Noisy-OR replacing 332 lines of ProbLog](#noisy-or-vs-problog-332-lines-to-4), the [pyoxigraph ↔ Postgres outbox](#the-outbox-two-stores-one-truth), and [named graphs as trust labels rather than filters](#named-graphs-as-trust-labels-not-filters). The rest fills in the supporting decisions.

---

## Named graphs as trust labels, not filters

Most knowledge-graph systems that care about provenance face the same fork: when a fact is "less trustworthy", do you (a) drop it from the graph entirely, (b) tag it and let it surface with the tag, or (c) bury it behind a query-time confidence threshold?

The system stores every triple in one of five named graphs:

| Graph URI | Origin |
|-----------|--------|
| `ks:graph/ontology` | Schema + domain vocabularies (loaded from `.ttl` files at startup) |
| `ks:graph/asserted` | Human-provided via `POST /api/claims` with a stated extractor |
| `ks:graph/extracted` | LLM-derived from ingested content |
| `ks:graph/inferred` | Forward-chaining derivations from existing triples |

The choice that matters: **the graph a triple lives in is surfaced to readers as a `trust_tier` label, but retrieval does not filter by tier.** When `RAGRetriever` builds the LLM prompt, every retrieved triple comes annotated with which graph it came from, and the prompt instructs the model to weight `verified` evidence above `extracted` when they conflict. Ranking, however, is tier-agnostic.

I argued myself out of doing tier-based filtering twice. The reason: as soon as you make the tier load-bearing for retrieval, you create two-tiered behaviour that's hard to debug — a question succeeds with `extracted` triples in `/api/ask` but fails the same logic in `/api/knowledge/query` because one filters and the other doesn't. The current design forces tier-awareness into a single place (the LLM prompt and the response JSON) and keeps the SPARQL layer pure. If you want to filter, you can — by adding `GRAPH <ks:graph/verified>` to your query. The default is "show everything, label everything."

The same principle applies to contradictions. `/api/ask` returns conflicting claims in the response payload rather than suppressing them. Telling the caller "these two sources disagree" is more useful than silently picking one.

This works because the graph is small, individually curated, and queried by humans. It would not work at compliance-vendor scale where filtering is the product. Different decision for a different shape.

---

## Noisy-OR vs ProbLog: 332 lines to 4

The most-edited file in the early history of this project was a `ReasoningEngine` built around [ProbLog](https://dtai.cs.kuleuven.be/problog/), a probabilistic logic programming language. The plan was to express knowledge-graph rules as ProbLog clauses, throw evidence at the engine, and read out posterior probabilities. It worked. It was also 332 lines of glue code, a hard dependency, slow at startup, and producing answers I could not always defend.

Then I noticed that the actual semantics I needed boiled down to one operation: *when N sources independently claim the same triple, what's the combined probability that the triple is true?* That's not a logic-programming problem. It's textbook noisy-OR.

The replacement, in its entirety:

```python
# src/knowledge_service/reasoning/noisy_or.py
"""Evidence combination via Noisy-OR. Replaces the 332-line ReasoningEngine."""

from math import prod


def noisy_or(confidences: list[float]) -> float:
    if not confidences:
        return 0.0
    clamped = [max(0.0, min(1.0, c)) for c in confidences]
    return 1.0 - prod(1.0 - c for c in clamped)
```

Four lines of logic. One stdlib import. No new dependencies.

The model: each source is treated as an independent noisy channel that *might* assert the truth. Source `i` with confidence `cᵢ` has a `1 − cᵢ` chance of failing to assert a true claim. For the claim to remain unsupported, *every* source must fail independently, so the combined probability of failure is `∏(1 − cᵢ)`, and the combined probability the claim is supported is `1 − ∏(1 − cᵢ)`.

This gives the right shape: more sources at the same individual confidence raise combined confidence, but with diminishing returns; one very strong source dominates many weak ones; confidence is bounded in `[0, 1]` by construction.

What I gave up: ProbLog could express joint distributions, mutually-exclusive evidence, and conditional dependencies between facts. Noisy-OR cannot. For this system that's fine — sources are treated as independent, and contradictions are surfaced rather than resolved probabilistically. If I ever need joint reasoning, it'll be obvious because Noisy-OR will give answers that don't fit the case. Until then, four lines.

The clamping (`max(0.0, min(1.0, c))`) was added later — see commit `ba1851b` — after upstream extractors started occasionally emitting confidences outside `[0, 1]` (qwen3 has rare overshoot). Without the clamp, a `c > 1` produces a negative `1 − c` and the product flips sign. The right place to fix is at the boundary; the validator at the model layer would do too, but the inner function should also not be a sharp edge.

The wider lesson — and the one worth carrying out of this project — is that *probabilistic* doesn't mean *complicated*. Noisy-OR is what I would have wanted ten years ago, when I last had this conversation with myself about Bayesian networks.

---

## The outbox: two stores, one truth

Triples live in pyoxigraph (file-backed RocksDB, in-process). Provenance and content live in PostgreSQL. The two cannot share a transaction. That makes "ingest a triple plus its provenance" a distributed-transaction problem in miniature, and the standard answer is the [transactional outbox pattern](https://microservices.io/patterns/data/transactional-outbox.html).

The implementation:

1. Per-triple writes (insert, confidence update, inference, retraction) are appended as rows to `triple_outbox` in the *same* PostgreSQL transaction as the matching `provenance` row.
2. After the PG transaction commits, an `OutboxDrainer` reads pending rows and applies them to pyoxigraph.
3. On startup, the drainer runs again, catching any rows whose process died between the PG commit and the pyoxigraph apply.

The schema (`migrations/014_triple_outbox.sql`):

```sql
CREATE TABLE triple_outbox (
  id              BIGSERIAL PRIMARY KEY,
  triple_hash     TEXT NOT NULL,
  operation       TEXT NOT NULL,        -- insert | update_confidence | insert_inferred | retract_inference
  subject         TEXT NOT NULL,
  predicate       TEXT NOT NULL,
  object          TEXT NOT NULL,
  ...
  graph           TEXT NOT NULL,
  payload         JSONB,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  applied_at      TIMESTAMPTZ           -- NULL until drained
);
CREATE INDEX idx_outbox_pending ON triple_outbox (id) WHERE applied_at IS NULL;
```

The invariant that pays for the complexity: **every `provenance` row references a triple that is either already durable in pyoxigraph, or staged in `triple_outbox` for replay.** The inverse — a pyoxigraph triple without provenance — is never produced. Which means if you ever want to audit a triple, the provenance is always there.

Idempotency is structural, not added later:

- `insert` re-runs are no-ops because pyoxigraph deduplicates by content hash, and the triple hash is the same.
- `update_confidence` is idempotent because writing the target value twice yields the target value.
- `insert_inferred` guards its RDF-star annotations with `ASK` before writing, because pyoxigraph's RDF-star INSERT will happily duplicate reification blank nodes if you ask it to. (Pyoxigraph at this layer is the right tool, but the RDF-star API has sharp edges — `DELETE WHERE { << ?s ?p ?o >> ?annotation_key ?value }` does not behave as the spec suggests it should, which is why annotation writes are read-then-write rather than DELETE-then-INSERT.)
- `retract_inference` against a hash whose inferences have already been removed finds nothing to do.

The drainer runs in two places: synchronously after each ingestion-pipeline PG commit (fast path), and at lifespan startup via `app.state.outbox_drainer.drain_pending()` (recovery path). Both paths share the same code.

What's still 2PC-shaped and unsolved: the inference engine and the contradiction-penalty step run *after* the base triple is durable in both stores. A crash in derived work leaves the base triple intact, but the derived triples are incomplete. The current mitigation is that derived work is deterministic over the base graph — re-ingesting the same source re-runs derived work to the same result, and inferred triples are content-addressed so duplicates are idempotent. The right shape, if this ever needed to be stronger, is to put the derived steps inside the same outbox protocol.

The other thing this is *not*: it is not the same as the stuck-job janitor. The janitor (in `main.py`'s lifespan) marks `ingestion_jobs` rows as `failed` on process restart, which is a per-document recovery mechanism. The outbox is a per-triple recovery mechanism. They handle different failure granularities, and both are needed.

---

## Reader-side status filter: the half-picture problem

`ContentStore.search()` and `RAGRetriever` both filter their result sets against `ingestion_jobs`, returning only content whose latest job has reached a terminal status (`completed` or `failed`), or has no job row at all.

The bug this prevents: in-flight content has its chunks written first (`EmbedPhase`) and its triples written last (`ProcessPhase`). Without the filter, a query that lands during ingestion can match the chunks of a half-ingested document and return them to the LLM *without* the KG triples that would contextualise them. The user sees an answer grounded in evidence whose structured representation the system hasn't actually finished computing — the half-picture.

The filter is applied in SQL via `LEFT JOIN LATERAL` against `ingestion_jobs`, ordered by `created_at DESC LIMIT 1` so re-ingestion (which creates a new job for the same `content_id`) is the one that counts:

```sql
LEFT JOIN LATERAL (
  SELECT status FROM ingestion_jobs
  WHERE content_id = c.content_id
  ORDER BY created_at DESC
  LIMIT 1
) latest_job ON true
WHERE latest_job.status IS NULL
   OR latest_job.status IN ('completed', 'failed')
```

`failed` jobs are deliberately included. The outbox 2PC may have committed partial triples before the failure, and those triples remain in the graph; hiding them would remove real evidence. Operators promote `failed` → `completed` by re-ingesting, which creates a new job row and supersedes the failed one.

`/api/content/{id}/chunks` is deliberately exempt from the filter — that endpoint is for operator/debug flows and must see in-flight content.

Controlled by `READER_EXCLUDE_INFLIGHT` (default `true`). The flag exists as a rollout escape hatch, not a per-request knob. If I were doing this again I would not have shipped it as a flag at all — it's the kind of correctness fix that doesn't warrant a way to turn it off — but it's there now and removing it is more disruptive than keeping it.

---

## Forward-chaining inference: three rules, BFS, retraction cascade

The inference engine in `reasoning/engine.py` runs three forward-chaining rules at ingestion time:

- **InverseRule** materialises `ks:inversePredicate` pairs. Asserting `Alice :knows Bob` triggers `Bob :knows Alice` if `knows` is declared inverse to itself, or `Alice :hasChild Bob` triggers `Bob :hasParent Alice` for a directed inverse pair.
- **TransitiveRule** closes chains over `ks:transitivePredicate` declarations. `part_of`, `is_a`, `located_in`, and `depends_on` are transitive in `domains/base.ttl`. Asserting `London :part_of England` plus `England :part_of UK` derives `London :part_of UK`.
- **TypeInheritanceRule** propagates `has_property` triples through `is_a` chains, so saying "dogs have_property fur" + "Rex is_a dog" derives "Rex has_property fur".

Execution is BFS with depth cap 3 and cycle detection via hash dedup. The depth cap exists because transitive closures over real-world data can explode (e.g. `is_a` chains in well-typed ontologies are deep, and a naive closure pulls in the whole class hierarchy).

Derived triples live in `ks:graph/inferred` with two RDF-star annotations: `ks:derivedFrom` pointing at the source triples' hashes, and `ks:inferenceMethod` naming the rule. This is what makes retraction cascading possible:

> If the confidence of a source triple drops below the inference threshold, or the triple is removed, every triple whose `ks:derivedFrom` references its hash is also retracted. The trigger hashes are computed up-front per inference run, so retraction is a single SPARQL pattern match.

The two non-obvious gotchas, both encoded as guards in every rule:

1. **Literal-object guard.** All three rules check `is_uri(obj)` before deriving. Literals (strings, numbers, dates) cannot become RDF subjects — pyoxigraph rejects them — so any rule that would derive `<literal> :something :something_else` must skip. This came up as a production bug after a `TypeInheritanceRule` tried to inherit `has_property` claims onto literal values (commits `c7ec66d`, `f436af7`).
2. **URI normalisation before triple hashing.** The pipeline normalises subject/predicate URIs via `ontology/uri.py` *before* passing triples to `InferenceEngine.run()`. The engine's `DerivedTriple.compute_hash()` creates `NamedNode` objects, which require absolute IRIs. A bare label like `"sam_altman"` would crash the hasher; normalising it to `http://knowledge.local/data/sam_altman` first is cheap and unambiguous.

Confidence on derived triples is `prod(source_confidences)` rather than Noisy-OR. The intuition: derivation is a logical *and* (all sources must hold for the inference to hold), not an *or* (any source suffices). When source confidences are 0.8 and 0.9, the inference has confidence 0.72, not 0.98. Get this wrong and the inferred graph becomes more confident than the data that produced it — the kind of error that compounds quietly.

---

## Two-phase LLM extraction: entities first, then relations

Knowledge extraction runs as two separate LLM calls per chunk:

1. **Entity pass.** "What named things are in this text? Return URIs, types, and labels."
2. **Relation pass.** "Given these entities, what relations exist between them or between them and literal values?"

Why split this rather than ask for entities-and-relations in one go:

- **Entity URIs become available to the relation prompt.** The model sees its own canonical URI choices from the first pass and uses them as subject/object in the second. This is a cheap way to get URI consistency within a document.
- **The relation prompt can include the NLP pre-pass hints** (spaCy NER + Wikidata QIDs) for each detected entity. The model gets "you already identified Sam Altman as the entity at `http://www.wikidata.org/entity/Q1346942`" and tends to use that URI rather than inventing a new one.
- **Token budget.** The relation prompt is bigger because it must explain the predicate vocabulary, and stuffing both passes into one would either overflow context on long chunks or force a smaller predicate list. Splitting keeps each prompt focused and roomy.
- **Failure isolation.** When entity extraction returns garbage (rare), relation extraction sees garbage too — but the failure mode is contained and visible. A single-pass JSON failure could silently drop relations while keeping entities, which is worse.

The known cost: two round-trips per chunk. For a long document that's `2 × chunks` LLM calls, not `chunks`. In production this is the throughput bottleneck — qwen3:14b on a single homelab GPU is overloaded under sustained ingestion — and it's why the Phase 2 wedge of an extraction-precision pass (see `CLAUDE.md` followups) is more valuable than chasing throughput. Better data per call beats more calls per second when the model is the constraint.

---

## Coreference via Wikidata QIDs

The NLP pre-pass (`nlp/__init__.py`) runs spaCy NER + `spacy-entity-linker` against each chunk before extraction. When an entity links to a Wikidata QID (e.g. "Sam Altman" → `Q1346942`), that QID is forwarded to the coreference phase.

Coreference is deterministic: any two entities that share a Wikidata QID are merged into a single `EntityGroup`. The merge is recorded in the `entity_aliases` table (PostgreSQL) and applied to all triples emitted in this ingestion job before they reach `ProcessPhase`.

What this gives you:

- "Sam Altman", "Mr Altman", and "Altman" across multiple documents resolve to one URI when the linker fires.
- Cross-document deduplication without a fuzzy-match heuristic. The Wikidata QID is the equivalence relation.
- Predictable behaviour. There is no "did the system merge these or not" debugging cycle — the alias table is the answer.

What this gives up:

- Coverage. Entities not in Wikidata (private individuals, niche topics, future-dated events) don't get QIDs and therefore don't get cross-document merging. They get embedding-based dedup as a fallback (`EntityStore` with cosine threshold 0.85), which is fuzzier and harder to audit.
- Disambiguation. spaCy-entity-linker picks the most likely QID and runs with it. Wrong picks propagate. Mitigation: confidence on the link is recorded; future work could surface ambiguous links to a reviewer.

The reason this is in the system at all rather than relying purely on embedding similarity: I wanted *one* of the dedup paths to be exact and auditable, not all of them probabilistic. Embedding similarity is a great fallback, but it's a bad foundation.

---

## What's deliberately *not* in here

- **Tier-based retrieval filtering.** Discussed above. Tiers are labels, not filters. If the credibility story for retrieval ever needs to be enforced, this is where it lives.
- **Numerical contradiction detection.** Today the contradiction endpoint fires on (a) same predicate, different *string-equal* objects, and (b) opposite-predicate pairs. It does *not* fire on numerically-different objects with the same predicate (e.g. "guidance £200M → £180M"). This is the most-requested gap and a leading Phase 2 wedge candidate (see `CLAUDE.md` followups).
- **Temporal contradictions and aging.** Triples have `valid_from` / `valid_until` fields, but the contradiction detector and the retriever ignore them. Two contradictory claims with non-overlapping validity intervals should not be a contradiction; today they are. This is honest tech debt.
- **Multi-writer pyoxigraph.** The store is single-writer-safe only. Running >1 ingestion replica against the same pyoxigraph volume would corrupt it. Production deploys to a single `aegis_knowledge` replica with the volume mounted only there.
- **Streaming ingestion.** Everything is batch-per-document. There is no event-stream input. For a feed-shaped use case this would need a Kafka/queue front-end; not in scope.
- **Cost telemetry.** LLM calls are made, results are stored, but per-document cost is not tracked. For a deployment paying real API bills, this would matter.

---

## Reading order if you're new to the codebase

1. `src/knowledge_service/main.py` — `create_app()` and `lifespan()` are the entry points; everything else is reachable from here.
2. `src/knowledge_service/ingestion/worker.py` — the five-phase pipeline orchestrator.
3. `src/knowledge_service/reasoning/noisy_or.py` — the four lines above.
4. `src/knowledge_service/reasoning/engine.py` — the inference engine.
5. `src/knowledge_service/ingestion/outbox.py` — the 2PC.
6. `src/knowledge_service/stores/triples.py` — pyoxigraph wrapper and the named-graph contract.
7. `src/knowledge_service/stores/rag.py` — hybrid retrieval with tier labels.

The tests in `tests/` are the runnable spec. `tests/test_inference_engine.py` and `tests/test_outbox.py` are the most pedagogical.
