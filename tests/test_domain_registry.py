import pytest
from pathlib import Path
from knowledge_service.stores.triples import TripleStore
from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.ontology.registry import DomainRegistry
from knowledge_service.ontology.uri import KS


@pytest.fixture
def registry(tmp_path):
    store = TripleStore(data_dir=None)
    ontology_dir = Path("src/knowledge_service/ontology")
    bootstrap_ontology(store, ontology_dir)
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    reg = DomainRegistry(store, prompts_dir)
    reg.load()
    return reg


class TestGetPredicates:
    def test_all_predicates(self, registry):
        preds = registry.get_predicates()
        labels = [p.label for p in preds]
        assert "causes" in labels
        assert "increases" in labels
        assert len(preds) >= 18

    def test_filter_by_domain(self, registry):
        preds = registry.get_predicates(domains=["base"])
        assert len(preds) >= 18

    def test_unknown_domain_empty(self, registry):
        preds = registry.get_predicates(domains=["nonexistent"])
        assert preds == []


class TestResolveSynonym:
    def test_known_synonym(self, registry):
        assert registry.resolve_synonym("boosts") == f"{KS}increases"

    def test_canonical_passthrough(self, registry):
        assert registry.resolve_synonym("causes") == f"{KS}causes"

    def test_unknown_returns_original(self, registry):
        assert registry.resolve_synonym("unknown_pred") == "unknown_pred"


class TestGetMateriality:
    def test_known_predicate(self, registry):
        weight = registry.get_materiality(f"{KS}causes")
        assert isinstance(weight, float)
        assert 0.0 <= weight <= 1.0

    def test_unknown_returns_default(self, registry):
        assert registry.get_materiality("http://unknown/pred") == 0.5


class TestPromptLoading:
    def test_loads_prompt_files(self, tmp_path):
        store = TripleStore(data_dir=None)
        ontology_dir = Path("src/knowledge_service/ontology")
        bootstrap_ontology(store, ontology_dir)
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "test_prompt.txt").write_text("Hello {title}")
        reg = DomainRegistry(store, prompts_dir)
        reg.load()
        assert reg.get_prompt("test_prompt") == "Hello {title}"

    def test_missing_prompt_returns_none(self, registry):
        assert registry.get_prompt("nonexistent") is None
