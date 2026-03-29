# tests/test_inference_engine.py
import pytest
from pathlib import Path

from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.ontology.namespaces import KS, KS_DATA, KS_GRAPH_EXTRACTED
from knowledge_service.reasoning.engine import (
    DerivedTriple,
    InferenceEngine,
    InverseRule,
    TransitiveRule,
    TypeInheritanceRule,
)
from knowledge_service.stores.triples import TripleStore


@pytest.fixture
def store_with_ontology():
    ts = TripleStore(data_dir=None)
    ontology_dir = Path(__file__).resolve().parent.parent / "src" / "knowledge_service" / "ontology"
    bootstrap_ontology(ts, ontology_dir)
    return ts


class TestDerivedTriple:
    def test_to_dict(self):
        dt = DerivedTriple(
            subject="http://knowledge.local/data/a",
            predicate="http://knowledge.local/schema/part_of",
            object_="http://knowledge.local/data/b",
            confidence=0.56,
            derived_from=["hash1", "hash2"],
            inference_method="transitive",
            depth=1,
        )
        d = dt.to_dict()
        assert d["subject"] == "http://knowledge.local/data/a"
        assert d["predicate"] == "http://knowledge.local/schema/part_of"
        assert d["object"] == "http://knowledge.local/data/b"
        assert d["confidence"] == pytest.approx(0.56)
        assert d["knowledge_type"] == "inferred"
        assert d["derived_from"] == ["hash1", "hash2"]
        assert d["inference_method"] == "transitive"

    def test_compute_hash_is_deterministic(self):
        dt = DerivedTriple(
            subject="http://knowledge.local/data/a",
            predicate="http://knowledge.local/schema/part_of",
            object_="http://knowledge.local/data/b",
            confidence=0.9,
            derived_from=[],
            inference_method="transitive",
            depth=1,
        )
        assert dt.compute_hash() == dt.compute_hash()

    def test_compute_hash_differs_for_different_triples(self):
        dt1 = DerivedTriple(
            subject="http://knowledge.local/data/a",
            predicate="http://knowledge.local/schema/part_of",
            object_="http://knowledge.local/data/b",
            confidence=0.9,
            derived_from=[],
            inference_method="transitive",
            depth=1,
        )
        dt2 = DerivedTriple(
            subject="http://knowledge.local/data/c",
            predicate="http://knowledge.local/schema/part_of",
            object_="http://knowledge.local/data/d",
            confidence=0.9,
            derived_from=[],
            inference_method="transitive",
            depth=1,
        )
        assert dt1.compute_hash() != dt2.compute_hash()


class TestInverseRule:
    def test_produces_inverse(self, store_with_ontology):
        ts = store_with_ontology
        ts.insert(
            f"{KS_DATA}a",
            f"{KS}contains",
            f"{KS_DATA}b",
            confidence=0.8,
            knowledge_type="claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        rule = InverseRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = rule.discover(trigger, ts, depth=1)
        assert len(results) == 1
        derived = results[0]
        assert derived.subject == f"{KS_DATA}b"
        assert derived.predicate == f"{KS}part_of"
        assert derived.object_ == f"{KS_DATA}a"
        assert derived.confidence == pytest.approx(0.8)
        assert derived.inference_method == "inverse"

    def test_inverse_is_bidirectional(self, store_with_ontology):
        """part_of should also resolve its inverse (contains)."""
        ts = store_with_ontology
        rule = InverseRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}b",
            "predicate": f"{KS}part_of",
            "object": f"{KS_DATA}a",
            "confidence": 0.7,
        }
        results = rule.discover(trigger, ts, depth=1)
        assert len(results) == 1
        derived = results[0]
        assert derived.subject == f"{KS_DATA}a"
        assert derived.predicate == f"{KS}contains"
        assert derived.object_ == f"{KS_DATA}b"

    def test_no_inverse_for_unrelated_predicate(self, store_with_ontology):
        ts = store_with_ontology
        rule = InverseRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}causes",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = rule.discover(trigger, ts, depth=1)
        assert len(results) == 0

    def test_derived_triple_has_source_hash(self, store_with_ontology):
        ts = store_with_ontology
        rule = InverseRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = rule.discover(trigger, ts, depth=1)
        assert len(results) == 1
        assert len(results[0].derived_from) == 1
        assert isinstance(results[0].derived_from[0], str)
        assert len(results[0].derived_from[0]) == 64  # SHA-256 hex digest

    def test_depth_is_propagated(self, store_with_ontology):
        ts = store_with_ontology
        rule = InverseRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = rule.discover(trigger, ts, depth=3)
        assert results[0].depth == 3


class TestTransitiveRule:
    def test_forward_transitive(self, store_with_ontology):
        ts = store_with_ontology
        ts.insert(
            f"{KS_DATA}b",
            f"{KS}part_of",
            f"{KS_DATA}c",
            confidence=0.7,
            knowledge_type="claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        rule = TransitiveRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}part_of",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = rule.discover(trigger, ts, depth=1)
        forward = [r for r in results if r.subject == f"{KS_DATA}a" and r.object_ == f"{KS_DATA}c"]
        assert len(forward) == 1
        assert forward[0].confidence == pytest.approx(0.56)
        assert forward[0].inference_method == "transitive"

    def test_backward_transitive(self, store_with_ontology):
        ts = store_with_ontology
        ts.insert(
            f"{KS_DATA}z",
            f"{KS}part_of",
            f"{KS_DATA}a",
            confidence=0.9,
            knowledge_type="claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        rule = TransitiveRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}part_of",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = rule.discover(trigger, ts, depth=1)
        backward = [r for r in results if r.subject == f"{KS_DATA}z" and r.object_ == f"{KS_DATA}b"]
        assert len(backward) == 1
        assert backward[0].confidence == pytest.approx(0.72)

    def test_literal_object_skipped(self, store_with_ontology):
        """Transitive rule should skip when object is a literal (non-URI)."""
        ts = store_with_ontology
        rule = TransitiveRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}dagster_pipeline",
            "predicate": f"{KS}part_of",
            "object": "some literal value",
            "confidence": 0.9,
        }
        results = rule.discover(trigger, ts, depth=1)
        assert len(results) == 0

    def test_non_transitive_predicate_skipped(self, store_with_ontology):
        ts = store_with_ontology
        rule = TransitiveRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}causes",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = rule.discover(trigger, ts, depth=1)
        assert len(results) == 0

    def test_confidence_product(self, store_with_ontology):
        ts = store_with_ontology
        ts.insert(
            f"{KS_DATA}b",
            f"{KS}part_of",
            f"{KS_DATA}c",
            confidence=0.7,
            knowledge_type="claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        rule = TransitiveRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}part_of",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = rule.discover(trigger, ts, depth=1)
        forward = [r for r in results if r.object_ == f"{KS_DATA}c"]
        assert len(forward) == 1
        assert forward[0].confidence == pytest.approx(0.8 * 0.7)


class TestTypeInheritanceRule:
    def test_inherit_property_from_type(self, store_with_ontology):
        ts = store_with_ontology
        ts.insert(
            f"{KS_DATA}mammal",
            f"{KS}has_property",
            f"{KS_DATA}warm_blooded",
            confidence=0.9,
            knowledge_type="fact",
            graph=KS_GRAPH_EXTRACTED,
        )
        rule = TypeInheritanceRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}dog",
            "predicate": f"{KS}is_a",
            "object": f"{KS_DATA}mammal",
            "confidence": 0.95,
        }
        results = rule.discover(trigger, ts, depth=1)
        assert len(results) == 1
        assert results[0].subject == f"{KS_DATA}dog"
        assert results[0].predicate == f"{KS}has_property"
        assert results[0].object_ == f"{KS_DATA}warm_blooded"
        assert results[0].confidence == pytest.approx(0.95 * 0.9)

    def test_propagate_property_to_instances(self, store_with_ontology):
        ts = store_with_ontology
        ts.insert(
            f"{KS_DATA}dog",
            f"{KS}is_a",
            f"{KS_DATA}mammal",
            confidence=0.95,
            knowledge_type="fact",
            graph=KS_GRAPH_EXTRACTED,
        )
        rule = TypeInheritanceRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}mammal",
            "predicate": f"{KS}has_property",
            "object": f"{KS_DATA}warm_blooded",
            "confidence": 0.9,
        }
        results = rule.discover(trigger, ts, depth=1)
        assert len(results) == 1
        assert results[0].subject == f"{KS_DATA}dog"
        assert results[0].object_ == f"{KS_DATA}warm_blooded"
        assert results[0].confidence == pytest.approx(0.95 * 0.9)

    def test_literal_object_skipped_is_a(self, store_with_ontology):
        """TypeInheritanceRule should skip when is_a object is a literal."""
        ts = store_with_ontology
        rule = TypeInheritanceRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}pipeline",
            "predicate": f"{KS}is_a",
            "object": "some literal type",
            "confidence": 0.9,
        }
        results = rule.discover(trigger, ts, depth=1)
        assert len(results) == 0

    def test_unrelated_predicate_skipped(self, store_with_ontology):
        ts = store_with_ontology
        rule = TypeInheritanceRule()
        rule.configure(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}causes",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = rule.discover(trigger, ts, depth=1)
        assert len(results) == 0


class TestInferenceEngine:
    def _make_engine(self, ts, max_depth=3):
        rules = [InverseRule(), TransitiveRule(), TypeInheritanceRule()]
        engine = InferenceEngine(ts, rules, max_depth=max_depth)
        engine.configure()
        return engine

    def test_single_inverse(self, store_with_ontology):
        ts = store_with_ontology
        ts.insert(
            f"{KS_DATA}a",
            f"{KS}contains",
            f"{KS_DATA}b",
            confidence=0.8,
            knowledge_type="claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        engine = self._make_engine(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = engine.run(trigger)
        inverses = [r for r in results if r.inference_method == "inverse"]
        assert len(inverses) >= 1
        assert inverses[0].subject == f"{KS_DATA}b"
        assert inverses[0].predicate == f"{KS}part_of"

    def test_chaining_inverse_then_transitive(self, store_with_ontology):
        """A contains B → B part_of A (inverse). C part_of B exists → C part_of A (transitive)."""
        ts = store_with_ontology
        ts.insert(
            f"{KS_DATA}c",
            f"{KS}part_of",
            f"{KS_DATA}b",
            confidence=0.7,
            knowledge_type="claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        ts.insert(
            f"{KS_DATA}a",
            f"{KS}contains",
            f"{KS_DATA}b",
            confidence=0.8,
            knowledge_type="claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        engine = self._make_engine(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = engine.run(trigger)
        transitive = [r for r in results if r.inference_method == "transitive"]
        assert any(r.subject == f"{KS_DATA}c" and r.object_ == f"{KS_DATA}a" for r in transitive)

    def test_max_depth_cap(self, store_with_ontology):
        """With max_depth=1, chaining should NOT happen."""
        ts = store_with_ontology
        ts.insert(
            f"{KS_DATA}c",
            f"{KS}part_of",
            f"{KS_DATA}b",
            confidence=0.7,
            knowledge_type="claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        ts.insert(
            f"{KS_DATA}a",
            f"{KS}contains",
            f"{KS_DATA}b",
            confidence=0.8,
            knowledge_type="claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        engine = self._make_engine(ts, max_depth=1)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}contains",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = engine.run(trigger)
        assert all(r.depth <= 1 for r in results)
        transitive = [r for r in results if r.inference_method == "transitive"]
        assert len(transitive) == 0

    def test_cycle_detection(self, store_with_ontology):
        """A part_of B, B part_of A should not loop infinitely."""
        ts = store_with_ontology
        ts.insert(
            f"{KS_DATA}a",
            f"{KS}part_of",
            f"{KS_DATA}b",
            confidence=0.8,
            knowledge_type="claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        ts.insert(
            f"{KS_DATA}b",
            f"{KS}part_of",
            f"{KS_DATA}a",
            confidence=0.7,
            knowledge_type="claim",
            graph=KS_GRAPH_EXTRACTED,
        )
        engine = self._make_engine(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}part_of",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = engine.run(trigger)
        assert len(results) < 20

    def test_no_rules_match(self, store_with_ontology):
        ts = store_with_ontology
        engine = self._make_engine(ts)
        trigger = {
            "subject": f"{KS_DATA}a",
            "predicate": f"{KS}causes",
            "object": f"{KS_DATA}b",
            "confidence": 0.8,
        }
        results = engine.run(trigger)
        assert len(results) == 0
