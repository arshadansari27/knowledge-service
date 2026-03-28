# Federation, Domains, Community Detection & Smoke Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate federation enrichment, add 3 domain ontology files, auto-trigger community detection, and add a production smoke test script.

**Architecture:** Four independent additions: (1) FederationPhase runs as background post-processing after ingestion, (2) domain .ttl files auto-load via existing bootstrap, (3) community rebuild triggers when triple count crosses threshold, (4) standalone smoke test script hits production API.

**Tech Stack:** Python, httpx, pyoxigraph, Turtle RDF, asyncio

---

### Task 1: Domain Ontology Files (health.ttl, technology.ttl, research.ttl)

**Files:**
- Create: `src/knowledge_service/ontology/domains/health.ttl`
- Create: `src/knowledge_service/ontology/domains/technology.ttl`
- Create: `src/knowledge_service/ontology/domains/research.ttl`
- Test: `tests/test_domain_ontology.py`

- [ ] **Step 1: Write test that new domain files load and register predicates**

```python
# tests/test_domain_ontology.py
"""Test that domain ontology files load correctly and register predicates."""

from pathlib import Path

from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.ontology.registry import DomainRegistry
from knowledge_service.stores.triples import TripleStore


class TestDomainOntologyFiles:
    def _load_registry(self) -> DomainRegistry:
        ts = TripleStore(data_dir=None)
        ontology_dir = Path(__file__).resolve().parent.parent / "src" / "knowledge_service" / "ontology"
        bootstrap_ontology(ts, ontology_dir)
        prompts_dir = ontology_dir / "prompts"
        registry = DomainRegistry(ts, prompts_dir)
        registry.load()
        return registry

    def test_health_domain_predicates_loaded(self):
        registry = self._load_registry()
        predicates = registry.get_predicates(["health"])
        labels = {p.label for p in predicates}
        assert "treats" in labels
        assert "improves condition" in labels
        assert "contraindicated with" in labels
        assert len(predicates) >= 15

    def test_technology_domain_predicates_loaded(self):
        registry = self._load_registry()
        predicates = registry.get_predicates(["technology"])
        labels = {p.label for p in predicates}
        assert "implements" in labels
        assert "integrates with" in labels
        assert "replaces" in labels
        assert len(predicates) >= 12

    def test_research_domain_predicates_loaded(self):
        registry = self._load_registry()
        predicates = registry.get_predicates(["research"])
        labels = {p.label for p in predicates}
        assert "cites" in labels
        assert "contradicts finding" in labels
        assert "finds" in labels
        assert len(predicates) >= 10

    def test_health_synonyms_resolve(self):
        registry = self._load_registry()
        resolved = registry.resolve_synonym("remedies")
        assert "treats" in resolved

    def test_health_opposite_predicates(self):
        registry = self._load_registry()
        predicates = registry.get_predicates(["health"])
        improves = next((p for p in predicates if p.label == "improves condition"), None)
        assert improves is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_domain_ontology.py -v`
Expected: FAIL — domain files don't exist yet

- [ ] **Step 3: Create health.ttl**

```turtle
# src/knowledge_service/ontology/domains/health.ttl
@prefix ks:   <http://knowledge.local/schema/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

# --- Health & Biohacking domain predicates ---

ks:treats a ks:Predicate ;
    rdfs:label "treats" ;
    ks:domain "health" ;
    ks:materialityWeight "0.8"^^xsd:float ;
    ks:synonym "remedies", "alleviates", "addresses", "helps_treat" .

ks:has_side_effect a ks:Predicate ;
    rdfs:label "has side effect" ;
    ks:domain "health" ;
    ks:materialityWeight "0.7"^^xsd:float ;
    ks:synonym "adverse_effect", "causes_side_effect", "side_effect_of" .

ks:recommended_dose a ks:Predicate ;
    rdfs:label "recommended dose" ;
    ks:domain "health" ;
    ks:materialityWeight "0.6"^^xsd:float ;
    ks:synonym "dosage", "suggested_intake", "optimal_dose", "daily_dose" .

ks:contraindicated_with a ks:Predicate ;
    rdfs:label "contraindicated with" ;
    ks:domain "health" ;
    ks:materialityWeight "0.8"^^xsd:float ;
    ks:synonym "interacts_negatively_with", "unsafe_with", "should_not_combine" .

ks:affects_biomarker a ks:Predicate ;
    rdfs:label "affects biomarker" ;
    ks:domain "health" ;
    ks:materialityWeight "0.7"^^xsd:float ;
    ks:synonym "modulates", "influences_level_of", "changes_biomarker" .

ks:improves_condition a ks:Predicate ;
    rdfs:label "improves condition" ;
    ks:domain "health" ;
    ks:materialityWeight "0.7"^^xsd:float ;
    ks:oppositePredicate ks:worsens_condition ;
    ks:synonym "helps_with", "beneficial_for", "alleviates_symptoms" .

ks:worsens_condition a ks:Predicate ;
    rdfs:label "worsens condition" ;
    ks:domain "health" ;
    ks:materialityWeight "0.7"^^xsd:float ;
    ks:oppositePredicate ks:improves_condition ;
    ks:synonym "aggravates", "exacerbates", "makes_worse" .

ks:has_mechanism a ks:Predicate ;
    rdfs:label "has mechanism" ;
    ks:domain "health" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "works_by", "mechanism_of_action", "acts_via" .

ks:metabolized_by a ks:Predicate ;
    rdfs:label "metabolized by" ;
    ks:domain "health" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "processed_by", "broken_down_by", "cleared_by" .

ks:absorbs_with a ks:Predicate ;
    rdfs:label "absorbs with" ;
    ks:domain "health" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "enhanced_by", "bioavailable_with", "better_absorbed_with" .

ks:depletes a ks:Predicate ;
    rdfs:label "depletes" ;
    ks:domain "health" ;
    ks:materialityWeight "0.6"^^xsd:float ;
    ks:synonym "reduces_levels_of", "lowers_stores_of", "drains" .

ks:synergizes_with a ks:Predicate ;
    rdfs:label "synergizes with" ;
    ks:domain "health" ;
    ks:materialityWeight "0.6"^^xsd:float ;
    ks:synonym "works_well_with", "potentiates", "amplifies_effect_of" .

ks:has_half_life a ks:Predicate ;
    rdfs:label "has half life" ;
    ks:domain "health" ;
    ks:materialityWeight "0.4"^^xsd:float ;
    ks:synonym "duration", "active_for", "elimination_half_life" .

ks:evidence_level a ks:Predicate ;
    rdfs:label "evidence level" ;
    ks:domain "health" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "evidence_quality", "study_quality", "strength_of_evidence" .

ks:bioavailable_as a ks:Predicate ;
    rdfs:label "bioavailable as" ;
    ks:domain "health" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "absorbed_as", "active_form", "converted_to" .
```

- [ ] **Step 4: Create technology.ttl**

```turtle
# src/knowledge_service/ontology/domains/technology.ttl
@prefix ks:   <http://knowledge.local/schema/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

# --- Technology & Software domain predicates ---

ks:implements a ks:Predicate ;
    rdfs:label "implements" ;
    ks:domain "technology" ;
    ks:materialityWeight "0.6"^^xsd:float ;
    ks:synonym "provides", "supports_feature", "offers" .

ks:integrates_with a ks:Predicate ;
    rdfs:label "integrates with" ;
    ks:domain "technology" ;
    ks:materialityWeight "0.6"^^xsd:float ;
    ks:synonym "connects_to", "works_with", "interfaces_with" .

ks:replaces a ks:Predicate ;
    rdfs:label "replaces" ;
    ks:domain "technology" ;
    ks:materialityWeight "0.7"^^xsd:float ;
    ks:synonym "supersedes", "deprecates", "succeeds" .

ks:requires a ks:Predicate ;
    rdfs:label "requires" ;
    ks:domain "technology" ;
    ks:materialityWeight "0.6"^^xsd:float ;
    ks:transitivePredicate "true"^^xsd:boolean ;
    ks:synonym "needs", "prerequisite" .

ks:compatible_with a ks:Predicate ;
    rdfs:label "compatible with" ;
    ks:domain "technology" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:oppositePredicate ks:incompatible_with ;
    ks:synonym "supports", "works_on", "runs_with" .

ks:incompatible_with a ks:Predicate ;
    rdfs:label "incompatible with" ;
    ks:domain "technology" ;
    ks:materialityWeight "0.7"^^xsd:float ;
    ks:oppositePredicate ks:compatible_with ;
    ks:synonym "breaks_with", "conflicts_with", "not_supported_on" .

ks:performs_better_than a ks:Predicate ;
    rdfs:label "performs better than" ;
    ks:domain "technology" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "faster_than", "outperforms", "more_efficient_than" .

ks:maintained_by a ks:Predicate ;
    rdfs:label "maintained by" ;
    ks:domain "technology" ;
    ks:materialityWeight "0.4"^^xsd:float ;
    ks:synonym "developed_by", "owned_by", "created_by_org" .

ks:licensed_as a ks:Predicate ;
    rdfs:label "licensed as" ;
    ks:domain "technology" ;
    ks:materialityWeight "0.4"^^xsd:float ;
    ks:synonym "license", "open_source", "license_type" .

ks:written_in a ks:Predicate ;
    rdfs:label "written in" ;
    ks:domain "technology" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "built_with", "language", "programming_language" .

ks:deployed_on a ks:Predicate ;
    rdfs:label "deployed on" ;
    ks:domain "technology" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "runs_on", "hosted_on", "infrastructure" .

ks:alternative_to a ks:Predicate ;
    rdfs:label "alternative to" ;
    ks:domain "technology" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "similar_to", "competitor_of", "comparable_to" .
```

- [ ] **Step 5: Create research.ttl**

```turtle
# src/knowledge_service/ontology/domains/research.ttl
@prefix ks:   <http://knowledge.local/schema/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

# --- Research & Science domain predicates ---

ks:cites a ks:Predicate ;
    rdfs:label "cites" ;
    ks:domain "research" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "references", "builds_on", "refers_to" .

ks:contradicts_finding a ks:Predicate ;
    rdfs:label "contradicts finding" ;
    ks:domain "research" ;
    ks:materialityWeight "0.8"^^xsd:float ;
    ks:synonym "disagrees_with", "refutes", "challenges" .

ks:replicates a ks:Predicate ;
    rdfs:label "replicates" ;
    ks:domain "research" ;
    ks:materialityWeight "0.7"^^xsd:float ;
    ks:synonym "confirms", "validates", "reproduces" .

ks:extends a ks:Predicate ;
    rdfs:label "extends" ;
    ks:domain "research" ;
    ks:materialityWeight "0.6"^^xsd:float ;
    ks:synonym "builds_upon", "expands", "generalizes" .

ks:funded_by a ks:Predicate ;
    rdfs:label "funded by" ;
    ks:domain "research" ;
    ks:materialityWeight "0.4"^^xsd:float ;
    ks:synonym "supported_by", "sponsored_by", "grant_from" .

ks:authored_by a ks:Predicate ;
    rdfs:label "authored by" ;
    ks:domain "research" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "written_by", "lead_author", "first_author" .

ks:published_in a ks:Predicate ;
    rdfs:label "published in" ;
    ks:domain "research" ;
    ks:materialityWeight "0.4"^^xsd:float ;
    ks:synonym "appears_in", "journal", "venue" .

ks:has_sample_size a ks:Predicate ;
    rdfs:label "has sample size" ;
    ks:domain "research" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "participants", "n_equals", "study_size" .

ks:uses_methodology a ks:Predicate ;
    rdfs:label "uses methodology" ;
    ks:domain "research" ;
    ks:materialityWeight "0.5"^^xsd:float ;
    ks:synonym "method", "study_design", "approach" .

ks:finds a ks:Predicate ;
    rdfs:label "finds" ;
    ks:domain "research" ;
    ks:materialityWeight "0.7"^^xsd:float ;
    ks:synonym "concludes", "shows", "demonstrates", "reports" .
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_domain_ontology.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_service/ontology/domains/health.ttl src/knowledge_service/ontology/domains/technology.ttl src/knowledge_service/ontology/domains/research.ttl tests/test_domain_ontology.py
git commit -m "feat: add health, technology, research domain ontology files"
```

---

### Task 2: Federation Enrichment Phase

**Files:**
- Create: `src/knowledge_service/ingestion/federation.py`
- Modify: `src/knowledge_service/ingestion/worker.py`
- Modify: `src/knowledge_service/main.py`
- Test: `tests/test_federation_phase.py`

- [ ] **Step 1: Write test for FederationPhase**

```python
# tests/test_federation_phase.py
"""Test federation enrichment phase."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_service.ingestion.federation import FederationPhase
from knowledge_service.stores.triples import TripleStore


class TestFederationPhase:
    def _make_phase(self, lookup_results=None):
        federation_client = AsyncMock()
        if lookup_results is None:
            lookup_results = {}
        federation_client.lookup_entity = AsyncMock(
            side_effect=lambda label: lookup_results.get(label)
        )
        triple_store = TripleStore(data_dir=None)
        return FederationPhase(
            federation_client=federation_client,
            triple_store=triple_store,
            max_lookups=10,
            delay=0,  # no delay in tests
        ), federation_client, triple_store

    async def test_enriches_entity_with_dbpedia_match(self):
        phase, client, ts = self._make_phase(
            lookup_results={
                "dopamine": {
                    "uri": "http://dbpedia.org/resource/Dopamine",
                    "rdf_type": "http://dbpedia.org/ontology/ChemicalSubstance",
                    "description": "A neurotransmitter",
                }
            }
        )
        entities = [{"label": "dopamine", "uri": "http://knowledge.local/data/dopamine"}]
        result = await phase.run(entities)
        assert result.entities_enriched == 1
        client.lookup_entity.assert_called_once_with("dopamine")

    async def test_skips_when_no_match(self):
        phase, client, ts = self._make_phase(lookup_results={})
        entities = [{"label": "obscure_thing", "uri": "http://knowledge.local/data/obscure_thing"}]
        result = await phase.run(entities)
        assert result.entities_enriched == 0

    async def test_respects_max_lookups(self):
        phase, client, ts = self._make_phase()
        phase._max_lookups = 2
        entities = [
            {"label": f"entity_{i}", "uri": f"http://knowledge.local/data/entity_{i}"}
            for i in range(5)
        ]
        await phase.run(entities)
        assert client.lookup_entity.call_count == 2

    async def test_handles_lookup_error_gracefully(self):
        phase, client, ts = self._make_phase()
        client.lookup_entity = AsyncMock(side_effect=Exception("network error"))
        entities = [{"label": "test", "uri": "http://knowledge.local/data/test"}]
        result = await phase.run(entities)
        assert result.entities_enriched == 0  # no crash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_federation_phase.py -v`
Expected: FAIL — `FederationPhase` doesn't exist

- [ ] **Step 3: Implement FederationPhase**

```python
# src/knowledge_service/ingestion/federation.py
"""Background federation enrichment — looks up entities on DBpedia/Wikidata after ingestion."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from knowledge_service.ontology.namespaces import KS_GRAPH_ASSERTED

logger = logging.getLogger(__name__)

_OWL_SAME_AS = "http://www.w3.org/2002/07/owl#sameAs"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDFS_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"


@dataclass
class FederationResult:
    entities_enriched: int = 0
    entities_skipped: int = 0
    errors: int = 0


class FederationPhase:
    """Enrich extracted entities with external knowledge from DBpedia/Wikidata."""

    def __init__(
        self,
        federation_client: Any,
        triple_store: Any,
        max_lookups: int = 10,
        delay: float = 1.0,
    ) -> None:
        self._client = federation_client
        self._store = triple_store
        self._max_lookups = max_lookups
        self._delay = delay

    async def run(self, entities: list[dict]) -> FederationResult:
        """Enrich entities with external URIs and metadata.

        Args:
            entities: List of dicts with 'label' and 'uri' keys.

        Returns:
            FederationResult with counts.
        """
        result = FederationResult()
        lookups = 0

        for entity in entities:
            if lookups >= self._max_lookups:
                break

            label = entity.get("label", "")
            local_uri = entity.get("uri", "")
            if not label or not local_uri:
                continue

            # Skip if already has owl:sameAs
            existing = self._store.get_triples(subject=local_uri, predicate=_OWL_SAME_AS)
            if existing:
                result.entities_skipped += 1
                continue

            try:
                match = await self._client.lookup_entity(label)
                lookups += 1
            except Exception:
                logger.warning("Federation lookup failed for %s", label, exc_info=True)
                result.errors += 1
                continue

            if match is None:
                continue

            # Insert owl:sameAs triple
            external_uri = match["uri"]
            self._store.insert(
                subject=local_uri,
                predicate=_OWL_SAME_AS,
                object_=external_uri,
                graph=KS_GRAPH_ASSERTED,
                confidence=0.95,
                knowledge_type="Fact",
            )

            # Insert rdf:type if available
            if match.get("rdf_type"):
                self._store.insert(
                    subject=local_uri,
                    predicate=_RDF_TYPE,
                    object_=match["rdf_type"],
                    graph=KS_GRAPH_ASSERTED,
                    confidence=0.9,
                    knowledge_type="Fact",
                )

            # Insert description if available
            if match.get("description"):
                self._store.insert(
                    subject=local_uri,
                    predicate=_RDFS_COMMENT,
                    object_=match["description"],
                    graph=KS_GRAPH_ASSERTED,
                    confidence=0.9,
                    knowledge_type="Fact",
                )

            result.entities_enriched += 1
            logger.info("Federation: %s → %s", label, external_uri)

            if self._delay > 0 and lookups < self._max_lookups:
                await asyncio.sleep(self._delay)

        return result
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_federation_phase.py -v`
Expected: PASS

- [ ] **Step 5: Wire up in worker.py and main.py**

Add to `worker.py` after `tracker.complete()` (line ~165):

```python
# After tracker.complete() — background federation enrichment
if federation_client is not None and triples_created > 0:
    try:
        from knowledge_service.ingestion.federation import FederationPhase
        from knowledge_service.config import settings

        fed_phase = FederationPhase(
            federation_client=federation_client,
            triple_store=stores.triples,
            max_lookups=10,
            delay=1.0,
        )
        # Collect entity labels from knowledge items
        fed_entities = []
        for item in knowledge_items:
            if hasattr(item, "label") and hasattr(item, "uri"):
                fed_entities.append({"label": item.label, "uri": item.uri})
            elif isinstance(item, dict):
                label = item.get("label") or item.get("subject", "")
                uri = item.get("uri") or item.get("subject", "")
                if label and uri:
                    fed_entities.append({"label": label, "uri": uri})
        if fed_entities:
            fed_result = await fed_phase.run(fed_entities)
            logger.info(
                "Federation enrichment for job %s: %d enriched, %d skipped",
                job_id, fed_result.entities_enriched, fed_result.entities_skipped,
            )
    except Exception:
        logger.warning("Federation enrichment failed for job %s", job_id, exc_info=True)
```

Add `federation_client` parameter to `run_ingestion()` signature.

In `main.py`, after extraction_client creation (~line 184):

```python
# Federation client (optional enrichment)
federation_client = None
if settings.federation_enabled:
    from knowledge_service.clients.federation import FederationClient
    federation_client = FederationClient(timeout=settings.federation_timeout)
    app.state.federation_client = federation_client
```

Pass `federation_client` to `_run_ingestion_worker()` calls in `content.py` and `upload.py`.

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add src/knowledge_service/ingestion/federation.py tests/test_federation_phase.py src/knowledge_service/ingestion/worker.py src/knowledge_service/main.py src/knowledge_service/api/content.py src/knowledge_service/api/upload.py
git commit -m "feat: wire up federation enrichment as background post-processing"
```

---

### Task 3: Auto-trigger Community Detection

**Files:**
- Modify: `src/knowledge_service/config.py`
- Modify: `src/knowledge_service/ingestion/worker.py`
- Modify: `src/knowledge_service/main.py`
- Test: `tests/test_community_trigger.py`

- [ ] **Step 1: Write test for community trigger logic**

```python
# tests/test_community_trigger.py
"""Test community detection auto-trigger conditions."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestCommunityTrigger:
    def test_triggers_when_conditions_met(self):
        """Should trigger when triples > min and cooldown elapsed."""
        from knowledge_service.ingestion.worker import _should_rebuild_communities

        assert _should_rebuild_communities(
            triples_created=5,
            total_triples=100,
            min_triples=50,
            last_rebuild=0,
            cooldown=3600,
        )

    def test_skips_when_no_triples_created(self):
        from knowledge_service.ingestion.worker import _should_rebuild_communities

        assert not _should_rebuild_communities(
            triples_created=0,
            total_triples=100,
            min_triples=50,
            last_rebuild=0,
            cooldown=3600,
        )

    def test_skips_when_below_threshold(self):
        from knowledge_service.ingestion.worker import _should_rebuild_communities

        assert not _should_rebuild_communities(
            triples_created=5,
            total_triples=30,
            min_triples=50,
            last_rebuild=0,
            cooldown=3600,
        )

    def test_skips_when_cooldown_not_elapsed(self):
        from knowledge_service.ingestion.worker import _should_rebuild_communities

        assert not _should_rebuild_communities(
            triples_created=5,
            total_triples=100,
            min_triples=50,
            last_rebuild=time.time(),  # just now
            cooldown=3600,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_community_trigger.py -v`
Expected: FAIL — `_should_rebuild_communities` doesn't exist

- [ ] **Step 3: Add config and trigger function**

Add to `config.py` after `community_rebuild_interval`:

```python
community_min_triples: int = 50
community_cooldown: int = 3600  # seconds between auto-rebuilds
```

Add to `worker.py`:

```python
import time

def _should_rebuild_communities(
    triples_created: int,
    total_triples: int,
    min_triples: int,
    last_rebuild: float,
    cooldown: int,
) -> bool:
    """Check if community detection should be triggered."""
    if triples_created <= 0:
        return False
    if total_triples < min_triples:
        return False
    if time.time() - last_rebuild < cooldown:
        return False
    return True
```

In `run_ingestion()`, after federation enrichment, add:

```python
# Auto-trigger community detection if conditions met
if hasattr(stores, 'triples') and triples_created > 0:
    try:
        total = stores.triples.count_triples()
        from knowledge_service.config import settings
        app_state = getattr(stores, '_app_state', None)
        last_rebuild = getattr(app_state, '_last_community_rebuild', 0) if app_state else 0

        if _should_rebuild_communities(
            triples_created=triples_created,
            total_triples=total,
            min_triples=settings.community_min_triples,
            last_rebuild=last_rebuild,
            cooldown=settings.community_cooldown,
        ):
            # Import and run rebuild
            import asyncio
            from knowledge_service.stores.community import CommunityDetector, CommunitySummarizer

            detector = CommunityDetector(stores.triples)
            communities = await asyncio.to_thread(detector.detect)
            if communities and extraction_client:
                summarizer = CommunitySummarizer(
                    extraction_client._client, stores.triples, model=extraction_client._model
                )
                summarized = []
                for c in communities:
                    summarized.append(await summarizer.summarize_one(c))
                community_store = getattr(app_state, 'community_store', None)
                if community_store:
                    await community_store.replace_all(summarized)
                    if app_state:
                        app_state._last_community_rebuild = time.time()
                    logger.info("Auto community rebuild: %d communities", len(summarized))
    except Exception:
        logger.warning("Auto community rebuild failed", exc_info=True)
```

Initialize `_last_community_rebuild` in `main.py` lifespan:

```python
app.state._last_community_rebuild = 0.0
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_community_trigger.py tests/ -v --tb=short`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/knowledge_service/config.py src/knowledge_service/ingestion/worker.py src/knowledge_service/main.py tests/test_community_trigger.py
git commit -m "feat: auto-trigger community detection when triple count crosses threshold"
```

---

### Task 4: Production Smoke Test Script

**Files:**
- Create: `scripts/smoke_test.py`

- [ ] **Step 1: Create smoke test script**

```python
#!/usr/bin/env python3
"""Production smoke test for knowledge-service.

Verifies the full pipeline works: health check, content ingestion (raw text,
HTML upload, PDF upload), knowledge graph query, and RAG question answering.

Usage:
    KNOWLEDGE_URL=https://knowledge.hikmahtech.in KNOWLEDGE_API_KEY=your-key \
        uv run python scripts/smoke_test.py
"""

import os
import sys
import time
from pathlib import Path

import httpx

BASE_URL = os.getenv("KNOWLEDGE_URL", "https://knowledge.hikmahtech.in")
API_KEY = os.getenv("KNOWLEDGE_API_KEY", "")
FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"

client = httpx.Client(base_url=BASE_URL, timeout=60, headers={"X-API-Key": API_KEY})

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def poll_job(content_id: str, timeout: int = 180) -> dict:
    elapsed = 0
    while elapsed < timeout:
        resp = client.get(f"/api/content/{content_id}/status")
        if resp.status_code == 200:
            status = resp.json()
            if status["status"] in ("completed", "failed"):
                return status
        time.sleep(5)
        elapsed += 5
    return {"status": "timeout"}


def main():
    print(f"\nSmoke test against {BASE_URL}\n")

    # 1. Health check
    print("1. Health check")
    resp = client.get("/health")
    check("GET /health", resp.status_code == 200 and resp.json().get("status") == "ok",
          resp.text[:100])

    # 2. Get baseline stats
    print("\n2. Baseline stats")
    resp = client.get("/api/admin/stats/counts")
    baseline = resp.json() if resp.status_code == 200 else {}
    baseline_triples = baseline.get("triples", 0)
    check("GET /api/admin/stats/counts", resp.status_code == 200,
          f"triples={baseline_triples}")

    # 3. Ingest raw text (health domain)
    print("\n3. Ingest raw text (health domain)")
    resp = client.post("/api/content", json={
        "url": f"smoke-test://health-{int(time.time())}",
        "title": "Vitamin D3 and Immune Function",
        "raw_text": (
            "Vitamin D3 (cholecalciferol) at a dose of 4000 IU daily improves immune "
            "function and reduces the risk of respiratory infections. A meta-analysis of "
            "25 randomized controlled trials with a combined sample size of 11,321 "
            "participants found that vitamin D supplementation reduces acute respiratory "
            "tract infections by 12%. Vitamin D3 is metabolized by the liver into "
            "25-hydroxyvitamin D, which is the biomarker measured in blood tests. "
            "Magnesium is required for vitamin D activation — vitamin D depletes magnesium. "
            "Vitamin D3 absorbs better with dietary fat."
        ),
        "source_type": "article",
    })
    check("POST /api/content (health)", resp.status_code == 202, resp.text[:100])
    if resp.status_code == 202:
        cid = resp.json()["content_id"]
        status = poll_job(cid)
        check("Ingestion completed", status["status"] == "completed",
              f"triples={status.get('triples_created', 0)}")

    # 4. Ingest raw text (technology domain)
    print("\n4. Ingest raw text (technology domain)")
    resp = client.post("/api/content", json={
        "url": f"smoke-test://tech-{int(time.time())}",
        "title": "PostgreSQL vs MySQL for Vector Search",
        "raw_text": (
            "PostgreSQL 16 with the pgvector extension provides native vector similarity "
            "search using HNSW indexes. PostgreSQL integrates with Python via asyncpg and "
            "psycopg2. pgvector is compatible with PostgreSQL 12 through 17. MySQL does "
            "not natively support vector search — it requires external tools like Milvus. "
            "PostgreSQL is licensed as PostgreSQL License (permissive open source). "
            "For AI applications, PostgreSQL with pgvector performs better than MySQL "
            "for hybrid search combining BM25 full-text and vector similarity."
        ),
        "source_type": "article",
    })
    check("POST /api/content (tech)", resp.status_code == 202, resp.text[:100])
    if resp.status_code == 202:
        cid = resp.json()["content_id"]
        status = poll_job(cid)
        check("Ingestion completed", status["status"] == "completed",
              f"triples={status.get('triples_created', 0)}")

    # 5. Upload HTML file
    print("\n5. Upload HTML file")
    html_path = FIXTURES_DIR / "sample.html"
    if html_path.exists():
        with open(html_path, "rb") as f:
            resp = client.post(
                "/api/content/upload",
                files={"file": ("sample.html", f, "text/html")},
                data={"title": "Smoke Test HTML", "source_type": "article"},
            )
        check("POST /api/content/upload (HTML)", resp.status_code == 202, resp.text[:100])
        if resp.status_code == 202:
            cid = resp.json()["content_id"]
            status = poll_job(cid)
            check("HTML ingestion completed", status["status"] == "completed",
                  f"triples={status.get('triples_created', 0)}")
    else:
        check("HTML fixture exists", False, f"Not found: {html_path}")

    # 6. Upload PDF file
    print("\n6. Upload PDF file")
    pdf_path = FIXTURES_DIR / "sample.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            resp = client.post(
                "/api/content/upload",
                files={"file": ("sample.pdf", f, "application/pdf")},
                data={"title": "Smoke Test PDF", "source_type": "paper"},
            )
        check("POST /api/content/upload (PDF)", resp.status_code == 202, resp.text[:100])
        if resp.status_code == 202:
            cid = resp.json()["content_id"]
            status = poll_job(cid)
            check("PDF ingestion completed", status["status"] == "completed",
                  f"triples={status.get('triples_created', 0)}")
    else:
        check("PDF fixture exists", False, f"Not found: {pdf_path}")

    # 7. Check stats increased
    print("\n7. Verify stats increased")
    resp = client.get("/api/admin/stats/counts")
    if resp.status_code == 200:
        final = resp.json()
        final_triples = final.get("triples", 0)
        check("Triples increased", final_triples > baseline_triples,
              f"{baseline_triples} → {final_triples}")
    else:
        check("Stats endpoint", False, resp.text[:100])

    # 8. Ask a question
    print("\n8. Ask a question (RAG)")
    resp = client.post("/api/ask", json={
        "question": "What is the recommended dose of Vitamin D3?",
        "max_sources": 3,
    })
    if resp.status_code == 200:
        answer = resp.json()
        check("POST /api/ask", True, f"answer length={len(answer.get('answer', ''))}")
    else:
        check("POST /api/ask", False, f"status={resp.status_code}")

    # Summary
    passed = sum(1 for _, p, _ in results if p)
    failed = sum(1 for _, p, _ in results if not p)
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")
    print(f"{'='*50}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: Set KNOWLEDGE_API_KEY environment variable")
        sys.exit(1)
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/smoke_test.py
git commit -m "feat: add production smoke test script"
```

---

### Task 5: Run all tests, lint, and final commit

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: 617+ pass

- [ ] **Step 2: Run lint**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: Clean

- [ ] **Step 3: Push and create PR**

```bash
git push -u origin worktree-federation-domains-community
gh pr create --title "feat: federation enrichment, domain ontology, auto community detection, smoke test" --body "..."
```
