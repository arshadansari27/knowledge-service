"""Bootstrap the knowledge store with the base ontology schema.

Loads the ks: schema (knowledge types, properties) from schema.ttl
into the pyoxigraph Store on startup.
"""

from pathlib import Path

from pyoxigraph import Store


def bootstrap_ontology(store: Store) -> int:
    """Load ks: schema into the store.

    Args:
        store: The pyoxigraph Store to load the schema into.

    Returns:
        Number of triples loaded from the schema.
    """
    schema_path = Path(__file__).parent / "schema.ttl"
    initial_count = len(list(store.quads_for_pattern(None, None, None, None)))
    with open(schema_path, "rb") as f:
        store.load(f, "text/turtle")
    final_count = len(list(store.quads_for_pattern(None, None, None, None)))
    return final_count - initial_count
