"""Tests for the maintenance sweep that cleans up production drift.

Uses a real in-memory pyoxigraph Store so RDF-star semantics are exercised
end-to-end (the operations rely on subtle quoted-triple matching that's
easy to get wrong against a mock).
"""

from __future__ import annotations

from pyoxigraph import Literal, NamedNode, Quad, Store

from knowledge_service.maintenance.normalizer import (
    normalize_knowledge_types,
    normalize_spacy_rdf_types,
)
from knowledge_service.ontology.namespaces import KS_KNOWLEDGE_TYPE
from knowledge_service.ontology.uri import RDF_TYPE


def _annotation_quad(store: Store, graph_uri: str, value: str) -> Quad:
    """Insert a base triple + its ks:knowledgeType reification into
    ``graph_uri`` and return the annotation Quad."""
    g = NamedNode(graph_uri)
    s = NamedNode("http://example.com/s")
    p = NamedNode("http://example.com/p")
    o = NamedNode("http://example.com/o")
    store.add(Quad(s, p, o, g))

    bnode = NamedNode("http://example.com/bnode")
    reifies = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies")
    # Skip the quoted-triple complication for the test: store ks:knowledgeType
    # directly on a bnode as if it's the reification node. The normalizer
    # finds rows by SPARQL on the predicate alone, which works on either
    # bnode-with-reifies or bnode-with-direct-anchor.
    store.add(Quad(bnode, reifies, NamedNode("http://example.com/anchor"), g))
    annotation = Quad(bnode, KS_KNOWLEDGE_TYPE, Literal(value), g)
    store.add(annotation)
    return annotation


class TestNormalizeKnowledgeTypes:
    def test_uppercase_lowered(self):
        store = Store()
        _annotation_quad(store, "http://knowledge.local/schema/graph/extracted", "Fact")

        stats = normalize_knowledge_types(store)

        assert stats["scanned"] == 1
        assert stats["changed"] == 1
        # Verify the new value is the lowercase form
        rows = list(
            store.query(
                f"SELECT ?val WHERE {{ GRAPH ?g {{ ?b <{KS_KNOWLEDGE_TYPE.value}> ?val }} }}"
            )
        )
        assert {str(r["val"].value) for r in rows} == {"fact"}

    def test_already_lowercase_is_no_op(self):
        store = Store()
        _annotation_quad(store, "http://knowledge.local/schema/graph/extracted", "claim")

        stats = normalize_knowledge_types(store)

        assert stats["scanned"] == 1
        assert stats["changed"] == 0

    def test_relation_alias_to_relationship(self):
        store = Store()
        _annotation_quad(store, "http://knowledge.local/schema/graph/extracted", "Relation")

        stats = normalize_knowledge_types(store)

        assert stats["changed"] == 1
        rows = list(
            store.query(
                f"SELECT ?val WHERE {{ GRAPH ?g {{ ?b <{KS_KNOWLEDGE_TYPE.value}> ?val }} }}"
            )
        )
        assert {str(r["val"].value) for r in rows} == {"relationship"}

    def test_idempotent(self):
        store = Store()
        _annotation_quad(store, "http://knowledge.local/schema/graph/extracted", "Fact")
        normalize_knowledge_types(store)
        stats = normalize_knowledge_types(store)
        assert stats["changed"] == 0


class TestNormalizeSpacyRdfTypes:
    def _insert_type(self, store: Store, subject: str, type_value: str):
        g = NamedNode("http://knowledge.local/schema/graph/extracted")
        store.add(Quad(NamedNode(subject), NamedNode(RDF_TYPE), Literal(type_value), g))

    def test_org_remapped_to_organization(self):
        store = Store()
        self._insert_type(store, "http://example.com/apple", "schema:ORG")

        stats = normalize_spacy_rdf_types(store)

        assert stats["remapped"] == 1
        assert stats["dropped"] == 0
        rows = list(store.query(f"SELECT ?o WHERE {{ GRAPH ?g {{ ?s <{RDF_TYPE}> ?o }} }}"))
        assert {str(r["o"].value) for r in rows} == {"schema:Organization"}

    def test_money_label_dropped(self):
        store = Store()
        self._insert_type(store, "http://example.com/cash", "schema:MONEY")

        stats = normalize_spacy_rdf_types(store)

        assert stats["dropped"] == 1
        rows = list(store.query(f"SELECT ?o WHERE {{ GRAPH ?g {{ ?s <{RDF_TYPE}> ?o }} }}"))
        assert rows == []

    def test_idempotent(self):
        store = Store()
        self._insert_type(store, "http://example.com/apple", "schema:PERSON")
        normalize_spacy_rdf_types(store)
        stats = normalize_spacy_rdf_types(store)
        assert stats["remapped"] == 0
        assert stats["dropped"] == 0

    def test_unknown_label_passes_through(self):
        """Labels not in the remap table (e.g. legitimate schema.org types
        the pipeline already emits canonically) must be left alone."""
        store = Store()
        self._insert_type(store, "http://example.com/foo", "schema:SoftwareApplication")

        stats = normalize_spacy_rdf_types(store)

        assert stats["remapped"] == 0
        assert stats["dropped"] == 0
        rows = list(store.query(f"SELECT ?o WHERE {{ GRAPH ?g {{ ?s <{RDF_TYPE}> ?o }} }}"))
        assert {str(r["o"].value) for r in rows} == {"schema:SoftwareApplication"}
