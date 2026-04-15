"""TripleStore — Core RDF knowledge graph wrapper around pyoxigraph.

Provides triple storage with RDF-star annotations for confidence scores,
knowledge types, and temporal validity ranges. Uses pyoxigraph (Rust-based
RDF triplestore) for fast in-memory or disk-backed storage with full
SPARQL 1.2 support.

All data triples are stored in named graphs for trust-tier separation:
- ks:graph/ontology  — schema, class hierarchy, predicate metadata
- ks:graph/asserted  — user-provided / high-trust triples
- ks:graph/extracted — LLM-extracted triples (default)
- ks:graph/inferred  — reasoning-engine derived triples
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

from knowledge_service._utils import _rdf_value_to_str, _to_rdf_term
from knowledge_service.ontology.namespaces import (
    KS,
    KS_CONFIDENCE,
    KS_GRAPH_EXTRACTED,
    KS_KNOWLEDGE_TYPE,
    KS_OPPOSITE_PREDICATE,
    KS_VALID_FROM,
    KS_VALID_UNTIL,
    XSD,
)
from knowledge_service.ontology.uri import is_uri, to_entity_uri, to_predicate_uri


def _strip_ks_prefix(value: str) -> str:
    """Strip the ks: namespace prefix from a URI, returning the local name."""
    if value.startswith(KS):
        return value[len(KS) :]
    return value


def _sparql_escape(value: str) -> str:
    """Escape a string for use as a SPARQL literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")


def _sparql_object(object_: str) -> str:
    """Format an object for use in SPARQL queries.

    URIs go in <brackets>, literals go in "quotes" with special chars escaped.
    """
    if is_uri(object_):
        return f"<{object_}>"
    return f'"{_sparql_escape(object_)}"'


class TripleStore:
    """Core RDF knowledge graph with RDF-star annotation support.

    Wraps pyoxigraph Store, adding:
    - Confidence scores as RDF-star annotations on triples
    - Knowledge type classification (Claim, Fact, Event, etc.)
    - Temporal validity ranges (validFrom / validUntil)
    - Contradiction detection
    - Content-addressed triple hashing (SHA-256)
    - Named graph separation for trust tiers
    """

    def __init__(self, data_dir: str | None = None):
        """Initialize the triple store.

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

    def insert(
        self,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float,
        knowledge_type: str,
        valid_from: date | datetime | None = None,
        valid_until: date | datetime | None = None,
        graph: str | None = None,
    ) -> tuple[str, bool]:
        """Insert a triple with RDF-star annotations into a named graph.

        The base triple (subject, predicate, object) is stored in the specified
        named graph (defaults to ks:graph/extracted). Metadata (confidence,
        knowledge_type, temporal bounds) is attached via RDF-star quoted triple
        annotations using pyoxigraph's SPARQL 1.2 reification support.

        If the triple already exists (in any graph), its annotations are left
        unchanged and the same hash is returned (idempotent).

        Returns:
            Tuple of (SHA-256 hex digest of the canonical triple form, is_new).
        """
        subject = to_entity_uri(subject)
        predicate = to_predicate_uri(predicate)

        s = NamedNode(subject)
        p = NamedNode(predicate)
        o = _to_rdf_term(object_)

        triple = Triple(s, p, o)
        triple_hash = hashlib.sha256(str(triple).encode()).hexdigest()

        graph_uri = graph or KS_GRAPH_EXTRACTED
        graph_node = NamedNode(graph_uri)

        # Insert base triple into named graph (idempotent — pyoxigraph deduplicates quads)
        self._store.add(Quad(s, p, o, graph_node))

        # Check if annotations already exist using SPARQL
        obj_sparql = _sparql_object(object_)
        ask_sparql = f"""
            ASK {{
                GRAPH ?g {{
                    << <{subject}> <{predicate}> {obj_sparql} >>
                        <{KS_CONFIDENCE.value}> ?conf .
                }}
            }}
        """
        if self._store.query(ask_sparql):
            return triple_hash, False

        # Build all RDF-star annotations in a single SPARQL INSERT DATA
        conf_literal = f'"{confidence}"^^<{XSD}float>'
        type_uri = f"<{KS}{knowledge_type}>"
        quoted = f"<< <{subject}> <{predicate}> {obj_sparql} >>"

        annotation_lines = [
            f"{quoted} <{KS_CONFIDENCE.value}> {conf_literal} .",
            f"{quoted} <{KS_KNOWLEDGE_TYPE.value}> {type_uri} .",
        ]

        if valid_from is not None:
            if isinstance(valid_from, date) and not isinstance(valid_from, datetime):
                vf = f'"{valid_from.isoformat()}"^^<{XSD}date>'
            else:
                vf = f'"{valid_from.isoformat()}"^^<{XSD}dateTime>'
            annotation_lines.append(f"{quoted} <{KS_VALID_FROM.value}> {vf} .")

        if valid_until is not None:
            if isinstance(valid_until, date) and not isinstance(valid_until, datetime):
                vu = f'"{valid_until.isoformat()}"^^<{XSD}date>'
            else:
                vu = f'"{valid_until.isoformat()}"^^<{XSD}dateTime>'
            annotation_lines.append(f"{quoted} <{KS_VALID_UNTIL.value}> {vu} .")

        body = "\n                    ".join(annotation_lines)
        self._store.update(f"""
            INSERT DATA {{
                GRAPH <{graph_uri}> {{
                    {body}
                }}
            }}
        """)

        return triple_hash, True

    def _query_annotated(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
        graphs: list[str] | None = None,
    ) -> list[dict]:
        """Build and execute a single SPARQL query for annotated triples.

        Dynamically binds subject/predicate/object as constants or variables,
        and includes all 4 OPTIONAL annotation blocks (confidence, knowledge_type,
        valid_from, valid_until). Filters to only return triples with annotations.
        """
        # Build the triple pattern with variables or constants
        s_term = f"<{subject}>" if subject else "?s"
        p_term = f"<{predicate}>" if predicate else "?p"
        if object_ is not None:
            o_term = _sparql_object(object_)
        else:
            o_term = "?o"

        # Build the quoted triple pattern for annotations
        quoted = f"<< {s_term} {p_term} {o_term} >>"

        # Graph filter
        graph_filter = ""
        if graphs:
            values = " ".join(f"<{g}>" for g in graphs)
            graph_filter = f"VALUES ?g {{ {values} }}"

        # Select variables — only include unbound ones
        select_vars = ["?g", "?conf", "?ktype", "?vfrom", "?vuntil"]
        if not subject:
            select_vars.append("?s")
        if not predicate:
            select_vars.append("?p")
        if object_ is None:
            select_vars.append("?o")

        select_clause = " ".join(select_vars)

        sparql = f"""
            SELECT {select_clause} WHERE {{
                {graph_filter}
                GRAPH ?g {{
                    {s_term} {p_term} {o_term} .
                }}
                OPTIONAL {{
                    GRAPH ?g {{
                        {quoted} <{KS_CONFIDENCE.value}> ?conf .
                    }}
                }}
                OPTIONAL {{
                    GRAPH ?g {{
                        {quoted} <{KS_KNOWLEDGE_TYPE.value}> ?ktype .
                    }}
                }}
                OPTIONAL {{
                    GRAPH ?g {{
                        {quoted} <{KS_VALID_FROM.value}> ?vfrom .
                    }}
                }}
                OPTIONAL {{
                    GRAPH ?g {{
                        {quoted} <{KS_VALID_UNTIL.value}> ?vuntil .
                    }}
                }}
                FILTER(BOUND(?conf))
            }}
        """
        query_result = self._store.query(sparql)
        results = []
        for solution in query_result:
            results.append(self._parse_annotated_row(solution, subject, predicate, object_))
        return results

    def _parse_annotated_row(
        self,
        solution: object,
        subject: str | None,
        predicate: str | None,
        object_: str | None,
    ) -> dict:
        """Parse a single SPARQL solution into a triple dict with annotations."""
        row: dict = {
            "graph": solution["g"].value,
            "subject": subject if subject else _rdf_value_to_str(solution["s"]),
            "predicate": predicate if predicate else _rdf_value_to_str(solution["p"]),
            "object": object_ if object_ is not None else _rdf_value_to_str(solution["o"]),
            "confidence": float(solution["conf"].value) if solution["conf"] else None,
            "knowledge_type": _strip_ks_prefix(solution["ktype"].value)
            if solution["ktype"]
            else None,
            "valid_from": solution["vfrom"].value if solution["vfrom"] else None,
            "valid_until": solution["vuntil"].value if solution["vuntil"] else None,
        }
        return row

    def get_triples(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        object_: str | None = None,
        graphs: list[str] | None = None,
    ) -> list[dict]:
        """Get annotated triples matching optional subject/predicate/object filters.

        Replaces get_triples_by_subject, get_triples_by_predicate, get_triples_by_object.
        Returns only triples that have RDF-star annotations (confidence etc.).

        Returns:
            List of dicts with subject, predicate, object, graph, confidence,
            knowledge_type, valid_from, valid_until.
        """
        return self._query_annotated(
            subject=subject, predicate=predicate, object_=object_, graphs=graphs
        )

    def update_confidence(self, triple_dict: dict, new_confidence: float) -> None:
        """Update the confidence annotation on a triple.

        Args:
            triple_dict: Dict with 'subject', 'predicate', 'object' keys.
            new_confidence: New confidence score between 0.0 and 1.0.
        """
        subject = triple_dict["subject"]
        predicate = triple_dict["predicate"]
        object_ = triple_dict["object"]
        obj_sparql = _sparql_object(object_)

        # SPARQL SELECT to find the reification bnode with confidence annotation
        select_sparql = f"""
            SELECT ?g ?bnode ?old_conf WHERE {{
                GRAPH ?g {{
                    ?bnode <http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies>
                        <<( <{subject}> <{predicate}> {obj_sparql} )>> .
                    ?bnode <{KS_CONFIDENCE.value}> ?old_conf .
                }}
            }}
        """
        results = self._store.query(select_sparql)

        new_val = Literal(
            str(new_confidence),
            datatype=NamedNode(f"{XSD}float"),
        )
        for solution in results:
            bnode = solution["bnode"]
            graph_node = solution["g"]
            old_conf = solution["old_conf"]

            self._store.remove(Quad(bnode, KS_CONFIDENCE, old_conf, graph_node))
            self._store.add(Quad(bnode, KS_CONFIDENCE, new_val, graph_node))

    def find_contradictions(
        self,
        subject: str,
        predicate: str,
        object_: str,
    ) -> list[dict]:
        """Find existing triples that may contradict the given triple.

        Searches for triples with the same subject and predicate but a
        different object. Returns them with their confidence annotations.

        Returns:
            List of dicts with object and confidence for contradicting triples.
        """
        obj_sparql = _sparql_object(object_)

        sparql = f"""
            SELECT ?o ?conf WHERE {{
                GRAPH ?g {{
                    <{subject}> <{predicate}> ?o .
                }}
                FILTER(?o != {obj_sparql})
                OPTIONAL {{
                    GRAPH ?g {{
                        << <{subject}> <{predicate}> ?o >>
                            <{KS_CONFIDENCE.value}> ?conf .
                    }}
                }}
            }}
        """
        query_result = self._store.query(sparql)
        results = []
        for solution in query_result:
            results.append(
                {
                    "object": _rdf_value_to_str(solution["o"]),
                    "confidence": float(solution["conf"].value) if solution["conf"] else None,
                }
            )
        return results

    def find_opposite_contradictions(
        self,
        subject: str,
        predicate: str,
        object_: str,
    ) -> list[dict]:
        """Find triples whose predicate is declared opposite to ``predicate``.

        Returns triples with the same subject and object but an opposite predicate.
        Each result dict has: predicate_in_store, confidence.
        """
        obj_sparql = _sparql_object(object_)

        sparql = f"""
            SELECT DISTINCT ?p_stored ?conf WHERE {{
                GRAPH ?gont {{
                    {{
                        <{predicate}> <{KS_OPPOSITE_PREDICATE.value}> ?p_stored .
                    }} UNION {{
                        ?p_stored <{KS_OPPOSITE_PREDICATE.value}> <{predicate}> .
                    }}
                }}
                GRAPH ?g {{
                    <{subject}> ?p_stored {obj_sparql} .
                }}
                OPTIONAL {{
                    GRAPH ?g {{
                        << <{subject}> ?p_stored {obj_sparql} >>
                            <{KS_CONFIDENCE.value}> ?conf .
                    }}
                }}
            }}
        """
        query_result = self._store.query(sparql)
        results = []
        for solution in query_result:
            p_term = solution["p_stored"]
            conf_term = solution["conf"]
            results.append(
                {
                    "predicate_in_store": p_term.value if hasattr(p_term, "value") else str(p_term),
                    "confidence": float(conf_term.value) if conf_term else None,
                }
            )
        return results

    def count_triples(self) -> int:
        """Return the number of annotated triples in the store."""
        result = self._store.query(
            f"""SELECT (COUNT(*) AS ?cnt) WHERE {{
                GRAPH ?g {{ ?s ?p ?o . }}
                GRAPH ?g {{ << ?s ?p ?o >> <{KS_CONFIDENCE.value}> ?conf . }}
            }}"""
        )
        for row in result:
            return int(row["cnt"].value)
        return 0

    def query(self, sparql: str):
        """Execute a SPARQL query.

        For SELECT queries returns a list of dicts mapping variable names
        (without '?') to values.  For ASK queries returns a bool.
        """
        from pyoxigraph import QueryBoolean  # noqa: PLC0415

        query_result = self._store.query(sparql)
        if isinstance(query_result, QueryBoolean):
            return bool(query_result)
        # SELECT (QuerySolutions)
        variables = [str(v).lstrip("?") for v in query_result.variables]
        results = []
        for solution in query_result:
            row = {}
            for var_name in variables:
                value = solution[var_name]
                row[var_name] = value
            results.append(row)
        return results

    def backup(self, path: str) -> None:
        """Dump the store contents to an N-Quads file."""
        with open(path, "wb") as f:
            self._store.dump(f, RdfFormat.N_QUADS)

    def flush(self) -> None:
        """Flush any pending writes to disk. No-op for in-memory stores."""
        self._store.flush()
