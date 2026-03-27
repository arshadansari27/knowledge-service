# tests/test_bootstrap.py
import pytest
from pathlib import Path

from knowledge_service.stores.triples import TripleStore
from knowledge_service.ontology.bootstrap import bootstrap_ontology
from knowledge_service.ontology.uri import KS


@pytest.fixture
def store():
    return TripleStore(data_dir=None)


ONTOLOGY_DIR = Path(__file__).resolve().parent.parent / "src" / "knowledge_service" / "ontology"


class TestBootstrap:
    def test_loads_base_predicates(self, store):
        bootstrap_ontology(store, ONTOLOGY_DIR)
        triples = store.query(f"""
            SELECT ?p WHERE {{
                GRAPH <{KS}graph/ontology> {{
                    ?p a <{KS}Predicate> .
                }}
            }}
        """)
        predicate_uris = [t["p"].value for t in triples]
        assert f"{KS}causes" in predicate_uris
        assert f"{KS}increases" in predicate_uris
        assert f"{KS}decreases" in predicate_uris
        assert f"{KS}is_a" in predicate_uris
        assert f"{KS}associated_with" in predicate_uris
        # All 18 canonical predicates
        assert len(predicate_uris) >= 18

    def test_loads_synonyms(self, store):
        bootstrap_ontology(store, ONTOLOGY_DIR)
        triples = store.query(f"""
            SELECT ?syn WHERE {{
                GRAPH <{KS}graph/ontology> {{
                    <{KS}increases> <{KS}synonym> ?syn .
                }}
            }}
        """)
        synonyms = [t["syn"].value for t in triples]
        assert "boosts" in synonyms
        assert "enhances" in synonyms
        assert "elevates" in synonyms

    def test_loads_materiality_weight(self, store):
        bootstrap_ontology(store, ONTOLOGY_DIR)
        triples = store.query(f"""
            SELECT ?w WHERE {{
                GRAPH <{KS}graph/ontology> {{
                    <{KS}causes> <{KS}materialityWeight> ?w .
                }}
            }}
        """)
        assert len(triples) >= 1
        weight = float(triples[0]["w"].value)
        assert 0.0 <= weight <= 1.0

    def test_loads_opposite_predicates(self, store):
        bootstrap_ontology(store, ONTOLOGY_DIR)
        triples = store.query(f"""
            SELECT ?opp WHERE {{
                GRAPH <{KS}graph/ontology> {{
                    <{KS}increases> <{KS}oppositePredicate> ?opp .
                }}
            }}
        """)
        opposites = [t["opp"].value for t in triples]
        assert f"{KS}decreases" in opposites

    def test_loads_domain_tag(self, store):
        bootstrap_ontology(store, ONTOLOGY_DIR)
        triples = store.query(f"""
            SELECT ?d WHERE {{
                GRAPH <{KS}graph/ontology> {{
                    <{KS}causes> <{KS}domain> ?d .
                }}
            }}
        """)
        domains = [t["d"].value for t in triples]
        assert "base" in domains

    def test_idempotent(self, store):
        bootstrap_ontology(store, ONTOLOGY_DIR)
        bootstrap_ontology(store, ONTOLOGY_DIR)  # second call
        triples = store.query(f"""
            SELECT ?p WHERE {{
                GRAPH <{KS}graph/ontology> {{
                    ?p a <{KS}Predicate> .
                }}
            }}
        """)
        uris = [t["p"].value for t in triples]
        assert len(uris) == len(set(uris))

    def test_schema_triples_still_loaded(self, store):
        """Verify that schema.ttl content is still loaded alongside domain TTLs."""
        bootstrap_ontology(store, ONTOLOGY_DIR)
        triples = store.query(f"""
            SELECT ?cls WHERE {{
                GRAPH <{KS}graph/ontology> {{
                    ?cls a <http://www.w3.org/2000/01/rdf-schema#Class> .
                }}
            }}
        """)
        class_uris = [t["cls"].value for t in triples]
        assert f"{KS}Claim" in class_uris
        assert f"{KS}Fact" in class_uris
