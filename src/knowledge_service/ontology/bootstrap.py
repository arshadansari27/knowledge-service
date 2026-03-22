"""Bootstrap the knowledge store with the base ontology schema.

Loads the ks: schema (knowledge types, properties) from schema.ttl
into the ks:graph/ontology named graph in pyoxigraph on startup.
"""

from pathlib import Path

from pyoxigraph import NamedNode, RdfFormat, Store

from knowledge_service.ontology.namespaces import KS_GRAPH_ONTOLOGY


def bootstrap_ontology(store: Store) -> int:
    """Load ks: schema into the ontology named graph.

    Idempotent: skips loading if the ontology graph already has triples.

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
        return 0

    initial_count = len(list(store.quads_for_pattern(None, None, None, None)))
    with open(schema_path, "rb") as f:
        store.load(f, RdfFormat.TURTLE, to_graph=graph_node)
    final_count = len(list(store.quads_for_pattern(None, None, None, None)))
    return final_count - initial_count
