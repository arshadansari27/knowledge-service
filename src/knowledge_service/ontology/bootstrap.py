"""Bootstrap the knowledge store with the base ontology schema.

Loads the ks: schema (knowledge types, properties) from schema.ttl
into the ks:graph/ontology named graph in pyoxigraph on startup.
"""

import logging
from pathlib import Path

from pyoxigraph import NamedNode, Quad, RdfFormat, Store

from knowledge_service.ontology.namespaces import KS, KS_GRAPH_ONTOLOGY, OWL, RDF

logger = logging.getLogger(__name__)

# Predicates that are single-valued (owl:FunctionalProperty).
# Value-conflict contradiction detection only applies to these.
_FUNCTIONAL_PREDICATES = ["amount", "currency"]


def _ensure_functional_properties(store: Store) -> int:
    """Ensure owl:FunctionalProperty triples exist for single-valued predicates.

    Runs on every startup so existing stores pick up schema changes.
    """
    graph_node = NamedNode(KS_GRAPH_ONTOLOGY)
    rdf_type = NamedNode(f"{RDF}type")
    owl_functional = NamedNode(f"{OWL}FunctionalProperty")
    added = 0

    for pred_name in _FUNCTIONAL_PREDICATES:
        pred_node = NamedNode(f"{KS}{pred_name}")
        existing = list(store.quads_for_pattern(pred_node, rdf_type, owl_functional, graph_node))
        if not existing:
            store.add(Quad(pred_node, rdf_type, owl_functional, graph_node))
            added += 1

    if added:
        logger.info("Added owl:FunctionalProperty to %d predicates", added)
    return added


def bootstrap_ontology(store: Store) -> int:
    """Load ks: schema into the ontology named graph.

    Idempotent: skips loading if the ontology graph already has triples.
    Always ensures functional property annotations are present.

    Args:
        store: The pyoxigraph Store to load the schema into.

    Returns:
        Number of triples loaded from the schema.
    """
    schema_path = Path(__file__).parent / "schema.ttl"
    graph_node = NamedNode(KS_GRAPH_ONTOLOGY)

    # Idempotency: skip if ontology graph already has triples
    existing = list(store.quads_for_pattern(None, None, None, graph_node))
    if existing:
        _ensure_functional_properties(store)
        return 0

    initial_count = len(list(store.quads_for_pattern(None, None, None, None)))
    with open(schema_path, "rb") as f:
        store.load(f, RdfFormat.TURTLE, to_graph=graph_node)
    final_count = len(list(store.quads_for_pattern(None, None, None, None)))
    return final_count - initial_count
