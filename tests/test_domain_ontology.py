"""Test that domain ontology files load correctly and register predicates."""

from pathlib import Path

from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.ontology.registry import DomainRegistry
from knowledge_service.stores.triples import TripleStore


class TestDomainOntologyFiles:
    def _load_registry(self) -> DomainRegistry:
        ts = TripleStore(data_dir=None)
        ontology_dir = (
            Path(__file__).resolve().parent.parent / "src" / "knowledge_service" / "ontology"
        )
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
