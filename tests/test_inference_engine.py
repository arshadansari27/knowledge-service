# tests/test_inference_engine.py
import pytest
from pathlib import Path

from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.ontology.namespaces import KS, KS_DATA, KS_GRAPH_EXTRACTED
from knowledge_service.reasoning.engine import DerivedTriple, InverseRule
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
