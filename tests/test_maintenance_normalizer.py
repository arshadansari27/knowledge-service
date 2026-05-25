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
from knowledge_service.ontology.namespaces import KS, KS_KNOWLEDGE_TYPE
from knowledge_service.ontology.uri import RDF_TYPE


def _annotation_quad(store: Store, graph_uri: str, value, bnode_iri: str | None = None):
    """Insert a base triple + its ks:knowledgeType reification into
    ``graph_uri``. ``value`` may be a NamedNode (the canonical ingestion
    shape — see ``stores/triples.py:149``) or a Literal (the broken shape
    the first maintenance run accidentally created)."""
    g = NamedNode(graph_uri)
    s = NamedNode("http://example.com/s")
    p = NamedNode("http://example.com/p")
    o = NamedNode("http://example.com/o")
    store.add(Quad(s, p, o, g))

    bnode = NamedNode(bnode_iri or "http://example.com/bnode")
    reifies = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies")
    store.add(Quad(bnode, reifies, NamedNode("http://example.com/anchor"), g))
    annotation = Quad(bnode, KS_KNOWLEDGE_TYPE, value, g)
    store.add(annotation)
    return annotation


class TestNormalizeKnowledgeTypes:
    """Production stores knowledge_type as a NamedNode URI like
    ``<http://knowledge.local/schema/fact>`` (see
    ``stores/triples.py:149``). The normalizer must preserve that shape;
    any drift to Literals or mixed-case URIs gets rewritten."""

    GRAPH = "http://knowledge.local/schema/graph/extracted"

    def _all_vals(self, store: Store) -> set[str]:
        rows = list(
            store.query(
                f"SELECT ?val WHERE {{ GRAPH ?g {{ ?b <{KS_KNOWLEDGE_TYPE.value}> ?val }} }}"
            )
        )
        return {str(r["val"].value) for r in rows}

    def test_uppercase_uri_lowered(self):
        store = Store()
        _annotation_quad(store, self.GRAPH, NamedNode(f"{KS}Fact"))

        stats = normalize_knowledge_types(store)

        assert stats == {"scanned": 1, "changed": 1}
        assert self._all_vals(store) == {f"{KS}fact"}

    def test_already_lowercase_uri_is_no_op(self):
        store = Store()
        _annotation_quad(store, self.GRAPH, NamedNode(f"{KS}claim"))

        stats = normalize_knowledge_types(store)

        assert stats == {"scanned": 1, "changed": 0}

    def test_relation_alias_to_relationship(self):
        store = Store()
        _annotation_quad(store, self.GRAPH, NamedNode(f"{KS}Relation"))

        stats = normalize_knowledge_types(store)

        assert stats["changed"] == 1
        assert self._all_vals(store) == {f"{KS}relationship"}

    def test_literal_with_full_uri_is_repaired(self):
        """Corrective pass for the shape bug introduced 2026-05-25: a
        previous maintenance run wrote knowledge_type as
        ``Literal("http://knowledge.local/schema/fact")``. Subsequent runs
        must rewrite those to the proper NamedNode URI form so analytics
        stop bifurcating along literal-vs-URI lines."""
        store = Store()
        _annotation_quad(store, self.GRAPH, Literal(f"{KS}fact"))

        stats = normalize_knowledge_types(store)

        assert stats == {"scanned": 1, "changed": 1}
        # Verify the literal was replaced with a NamedNode in the same slot
        rows = list(
            store.query(
                f"SELECT ?val WHERE {{ GRAPH ?g {{ ?b <{KS_KNOWLEDGE_TYPE.value}> ?val }} FILTER(isIRI(?val)) }}"
            )
        )
        assert len(rows) == 1
        assert str(rows[0]["val"].value) == f"{KS}fact"

    def test_literal_bare_name_normalised_to_uri(self):
        """If a literal slipped in carrying just ``"fact"``, rebuild it as
        a proper ``<ks:fact>`` NamedNode."""
        store = Store()
        _annotation_quad(store, self.GRAPH, Literal("Fact"))

        stats = normalize_knowledge_types(store)

        assert stats["changed"] == 1
        assert self._all_vals(store) == {f"{KS}fact"}

    def test_idempotent(self):
        store = Store()
        _annotation_quad(store, self.GRAPH, NamedNode(f"{KS}Fact"))
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
