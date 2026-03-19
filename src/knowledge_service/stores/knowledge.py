"""KnowledgeStore — Core RDF knowledge graph wrapper around pyoxigraph.

Provides triple storage with RDF-star annotations for confidence scores,
knowledge types, and temporal validity ranges. Uses pyoxigraph (Rust-based
RDF triplestore) for fast in-memory or disk-backed storage with full
SPARQL 1.2 support.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime

from pyoxigraph import (
    Literal,
    NamedNode,
    Quad,
    RdfFormat,
    Store,
    Triple,
)

from knowledge_service.ontology.namespaces import (
    KS,
    KS_CONFIDENCE,
    KS_KNOWLEDGE_TYPE,
    KS_OPPOSITE_PREDICATE,
    KS_VALID_FROM,
    KS_VALID_UNTIL,
    XSD,
)

RDF_REIFIES = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies")


def _strip_ks_prefix(value: str) -> str:
    """Strip the ks: namespace prefix from a URI, returning the local name."""
    if value.startswith(KS):
        return value[len(KS) :]
    return value


def _is_uri(value: str) -> bool:
    """Check if a string looks like a URI."""
    return value.startswith("http://") or value.startswith("https://") or value.startswith("urn:")


def _to_rdf_term(value: str) -> NamedNode | Literal:
    """Convert a string to a NamedNode (if URI) or Literal."""
    if _is_uri(value):
        return NamedNode(value)
    return Literal(value)


def _sparql_object(object_: str) -> str:
    """Format an object for use in SPARQL queries.

    URIs go in <brackets>, literals go in "quotes".
    """
    if _is_uri(object_):
        return f"<{object_}>"
    return f'"{object_}"'


class KnowledgeStore:
    """Core RDF knowledge graph with RDF-star annotation support.

    Wraps pyoxigraph Store, adding:
    - Confidence scores as RDF-star annotations on triples
    - Knowledge type classification (Claim, Fact, Event, etc.)
    - Temporal validity ranges (validFrom / validUntil)
    - Contradiction detection
    - Content-addressed triple hashing (SHA-256)
    """

    def __init__(self, data_dir: str | None = None):
        """Initialize the knowledge store.

        Args:
            data_dir: Path for disk-backed storage. None = in-memory (for tests).
        """
        if data_dir is not None:
            self._store = Store(data_dir)
        else:
            self._store = Store()

    @property
    def store(self) -> Store:
        """Access the underlying pyoxigraph Store."""
        return self._store

    def insert_triple(
        self,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float,
        knowledge_type: str,
        valid_from: date | datetime | None = None,
        valid_until: date | datetime | None = None,
    ) -> tuple[str, bool]:
        """Insert a triple with RDF-star annotations.

        The base triple (subject, predicate, object) is stored in the default
        graph. Metadata (confidence, knowledge_type, temporal bounds) is attached
        via RDF-star quoted triple annotations using pyoxigraph's SPARQL 1.2
        reification support.

        If the triple already exists, its annotations are left unchanged and
        the same hash is returned (idempotent).

        Args:
            subject: URI of the subject.
            predicate: URI of the predicate.
            object_: URI or literal value for the object.
            confidence: Confidence score between 0.0 and 1.0.
            knowledge_type: One of Claim, Fact, Event, Entity, etc.
            valid_from: Optional start of temporal validity.
            valid_until: Optional end of temporal validity.

        Returns:
            Tuple of (SHA-256 hex digest of the canonical triple form, is_new).
            is_new is True if the triple did not previously exist, False otherwise.
        """
        s = NamedNode(subject)
        p = NamedNode(predicate)
        o = _to_rdf_term(object_)

        triple = Triple(s, p, o)
        triple_hash = hashlib.sha256(str(triple).encode()).hexdigest()

        # Insert base triple (idempotent — pyoxigraph deduplicates quads)
        self._store.add(Quad(s, p, o))

        # Check if annotations already exist for this triple
        existing_reifications = list(self._store.quads_for_pattern(None, RDF_REIFIES, triple, None))
        if existing_reifications:
            # Annotations already exist; skip to preserve idempotency
            return triple_hash, False

        # Insert RDF-star annotations via SPARQL UPDATE
        obj_sparql = _sparql_object(object_)
        conf_literal = f'"{confidence}"^^<{XSD}float>'
        type_uri = f"<{KS}{knowledge_type}>"

        sparql = f"""
            INSERT DATA {{
                << <{subject}> <{predicate}> {obj_sparql} >>
                    <{KS_CONFIDENCE.value}> {conf_literal} .
                << <{subject}> <{predicate}> {obj_sparql} >>
                    <{KS_KNOWLEDGE_TYPE.value}> {type_uri} .
            }}
        """
        self._store.update(sparql)

        # Optional temporal annotations
        if valid_from is not None:
            if isinstance(valid_from, date) and not isinstance(valid_from, datetime):
                vf = f'"{valid_from.isoformat()}"^^<{XSD}date>'
            else:
                vf = f'"{valid_from.isoformat()}"^^<{XSD}dateTime>'
            self._store.update(f"""
                INSERT DATA {{
                    << <{subject}> <{predicate}> {obj_sparql} >>
                        <{KS_VALID_FROM.value}> {vf} .
                }}
            """)

        if valid_until is not None:
            if isinstance(valid_until, date) and not isinstance(valid_until, datetime):
                vu = f'"{valid_until.isoformat()}"^^<{XSD}date>'
            else:
                vu = f'"{valid_until.isoformat()}"^^<{XSD}dateTime>'
            self._store.update(f"""
                INSERT DATA {{
                    << <{subject}> <{predicate}> {obj_sparql} >>
                        <{KS_VALID_UNTIL.value}> {vu} .
                }}
            """)

        return triple_hash, True

    def query(self, sparql: str) -> list[dict]:
        """Execute a SPARQL SELECT query.

        Args:
            sparql: SPARQL SELECT query string.

        Returns:
            List of dicts mapping variable names (without '?') to values.
        """
        query_result = self._store.query(sparql)
        variables = [str(v).lstrip("?") for v in query_result.variables]
        results = []
        for solution in query_result:
            row = {}
            for var_name in variables:
                value = solution[var_name]
                row[var_name] = value
            results.append(row)
        return results

    def get_triples_by_subject(self, subject: str) -> list[dict]:
        """Get all annotated triples for a given subject.

        Returns only triples that have RDF-star annotations (filters out
        bare schema/ontology triples). Each result contains:
        predicate, object, confidence, knowledge_type, valid_from, valid_until.

        Args:
            subject: URI of the subject to look up.

        Returns:
            List of dicts with triple data and annotations.
        """
        sparql = f"""
            SELECT ?p ?o ?conf ?ktype ?vfrom ?vuntil WHERE {{
                <{subject}> ?p ?o .
                OPTIONAL {{
                    << <{subject}> ?p ?o >>
                        <{KS_CONFIDENCE.value}> ?conf .
                }}
                OPTIONAL {{
                    << <{subject}> ?p ?o >>
                        <{KS_KNOWLEDGE_TYPE.value}> ?ktype .
                }}
                OPTIONAL {{
                    << <{subject}> ?p ?o >>
                        <{KS_VALID_FROM.value}> ?vfrom .
                }}
                OPTIONAL {{
                    << <{subject}> ?p ?o >>
                        <{KS_VALID_UNTIL.value}> ?vuntil .
                }}
                FILTER(BOUND(?conf))
            }}
        """
        query_result = self._store.query(sparql)
        results = []
        for solution in query_result:
            row = {
                "predicate": solution["p"],
                "object": solution["o"],
                "confidence": float(solution["conf"].value) if solution["conf"] else None,
                "knowledge_type": _strip_ks_prefix(solution["ktype"].value)
                if solution["ktype"]
                else None,
                "valid_from": solution["vfrom"].value if solution["vfrom"] else None,
                "valid_until": solution["vuntil"].value if solution["vuntil"] else None,
            }
            results.append(row)
        return results

    def get_triples_by_predicate(self, predicate: str) -> list[dict]:
        """Get all annotated triples for a given predicate.

        Args:
            predicate: URI of the predicate to look up.

        Returns:
            List of dicts with triple data and annotations.
        """
        sparql = f"""
            SELECT ?s ?o ?conf ?ktype ?vfrom ?vuntil WHERE {{
                ?s <{predicate}> ?o .
                OPTIONAL {{
                    << ?s <{predicate}> ?o >>
                        <{KS_CONFIDENCE.value}> ?conf .
                }}
                OPTIONAL {{
                    << ?s <{predicate}> ?o >>
                        <{KS_KNOWLEDGE_TYPE.value}> ?ktype .
                }}
                OPTIONAL {{
                    << ?s <{predicate}> ?o >>
                        <{KS_VALID_FROM.value}> ?vfrom .
                }}
                OPTIONAL {{
                    << ?s <{predicate}> ?o >>
                        <{KS_VALID_UNTIL.value}> ?vuntil .
                }}
                FILTER(BOUND(?conf))
            }}
        """
        query_result = self._store.query(sparql)
        results = []
        for solution in query_result:
            row = {
                "subject": solution["s"],
                "predicate": NamedNode(predicate),
                "object": solution["o"],
                "confidence": float(solution["conf"].value) if solution["conf"] else None,
                "knowledge_type": _strip_ks_prefix(solution["ktype"].value)
                if solution["ktype"]
                else None,
                "valid_from": solution["vfrom"].value if solution["vfrom"] else None,
                "valid_until": solution["vuntil"].value if solution["vuntil"] else None,
            }
            results.append(row)
        return results

    def update_confidence(
        self,
        subject: str,
        predicate: str,
        object_: str,
        new_confidence: float,
    ) -> None:
        """Update the confidence annotation for a triple.

        Uses the Python API to find the reification blank node and replace
        the old confidence literal with the new one. This avoids SPARQL
        DELETE/INSERT limitations with RDF-star syntax.

        Args:
            subject: URI of the subject.
            predicate: URI of the predicate.
            object_: URI or literal value for the object.
            new_confidence: New confidence score between 0.0 and 1.0.
        """
        s = NamedNode(subject)
        p = NamedNode(predicate)
        o = _to_rdf_term(object_)
        target_triple = Triple(s, p, o)

        # Find blank nodes that reify this triple
        reification_quads = list(
            self._store.quads_for_pattern(None, RDF_REIFIES, target_triple, None)
        )

        for rq in reification_quads:
            bnode = rq.subject
            # Find existing confidence quads for this blank node
            conf_quads = list(self._store.quads_for_pattern(bnode, KS_CONFIDENCE, None, None))
            # Remove old confidence values
            for cq in conf_quads:
                self._store.remove(cq)
            # Add new confidence
            new_val = Literal(
                str(new_confidence),
                datatype=NamedNode(f"{XSD}float"),
            )
            self._store.add(Quad(bnode, KS_CONFIDENCE, new_val))

    def find_contradictions(
        self,
        subject: str,
        predicate: str,
        object_: str,
    ) -> list[dict]:
        """Find existing triples that may contradict the given triple.

        Searches for triples with the same subject and predicate but a
        different object. Returns them with their confidence annotations.

        Args:
            subject: URI of the subject.
            predicate: URI of the predicate.
            object_: URI or literal of the proposed object.

        Returns:
            List of dicts with object and confidence for contradicting triples.
        """
        obj_sparql = _sparql_object(object_)

        sparql = f"""
            SELECT ?o ?conf WHERE {{
                <{subject}> <{predicate}> ?o .
                FILTER(?o != {obj_sparql})
                OPTIONAL {{
                    << <{subject}> <{predicate}> ?o >>
                        <{KS_CONFIDENCE.value}> ?conf .
                }}
            }}
        """
        query_result = self._store.query(sparql)
        results = []
        for solution in query_result:
            row = {
                "object": solution["o"],
                "confidence": float(solution["conf"].value) if solution["conf"] else None,
            }
            results.append(row)
        return results

    def find_opposite_predicate_contradictions(
        self,
        subject: str,
        predicate: str,
        object_: str,
    ) -> list[dict]:
        """Find existing triples whose predicate is declared opposite to ``predicate``.

        Returns triples with the same subject and object but an opposite predicate.
        Each result dict has: predicate_in_store, confidence.

        Uses UNION to cover both directions of oppositePredicate (A→B and B→A).
        """
        obj_sparql = _sparql_object(object_)

        sparql = f"""
            SELECT DISTINCT ?p_stored ?conf WHERE {{
                {{
                    <{predicate}> <{KS_OPPOSITE_PREDICATE.value}> ?p_stored .
                }} UNION {{
                    ?p_stored <{KS_OPPOSITE_PREDICATE.value}> <{predicate}> .
                }}
                <{subject}> ?p_stored {obj_sparql} .
                OPTIONAL {{
                    << <{subject}> ?p_stored {obj_sparql} >>
                        <{KS_CONFIDENCE.value}> ?conf .
                }}
            }}
        """
        query_result = self._store.query(sparql)
        results = []
        for solution in query_result:
            p_term = solution["p_stored"]
            try:
                conf_term = solution["conf"]
            except Exception:
                conf_term = None
            results.append({
                "predicate_in_store": p_term.value if hasattr(p_term, "value") else str(p_term),
                "confidence": float(conf_term.value) if conf_term else None,
            })
        return results

    def backup(self, path: str) -> None:
        """Dump the store contents to an N-Quads file.

        Args:
            path: File path to write the N-Quads dump to.
        """
        with open(path, "wb") as f:
            self._store.dump(f, RdfFormat.N_QUADS)

    def flush(self) -> None:
        """Flush any pending writes to disk.

        No-op for in-memory stores.
        """
        self._store.flush()
