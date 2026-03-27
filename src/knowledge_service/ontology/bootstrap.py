"""Bootstrap the knowledge store with the base ontology schema.

Loads the ks: schema (knowledge types, properties) from schema.ttl
into the ks:graph/ontology named graph in pyoxigraph on startup,
then additively loads all domain .ttl files from ontology/domains/.
"""

import logging
from pathlib import Path

from pyoxigraph import NamedNode, Quad, RdfFormat, Store

from knowledge_service.ontology.namespaces import KS, KS_GRAPH_ONTOLOGY, OWL, RDF
from knowledge_service.stores.triples import TripleStore

logger = logging.getLogger(__name__)

# Predicates that are single-valued (owl:FunctionalProperty).
# Value-conflict contradiction detection only applies to these.
_FUNCTIONAL_PREDICATES = ["amount", "currency"]


def _ensure_functional_properties(raw_store: Store) -> int:
    """Ensure owl:FunctionalProperty triples exist for single-valued predicates.

    Runs on every startup so existing stores pick up schema changes.
    """
    graph_node = NamedNode(KS_GRAPH_ONTOLOGY)
    rdf_type = NamedNode(f"{RDF}type")
    owl_functional = NamedNode(f"{OWL}FunctionalProperty")
    added = 0

    for pred_name in _FUNCTIONAL_PREDICATES:
        pred_node = NamedNode(f"{KS}{pred_name}")
        existing = list(
            raw_store.quads_for_pattern(pred_node, rdf_type, owl_functional, graph_node)
        )
        if not existing:
            raw_store.add(Quad(pred_node, rdf_type, owl_functional, graph_node))
            added += 1

    if added:
        logger.info("Added owl:FunctionalProperty to %d predicates", added)
    return added


def bootstrap_ontology(store: TripleStore, ontology_dir: Path) -> int:
    """Load ks: schema and domain TTLs into the ontology named graph.

    Idempotent: skips loading if the ontology graph already has triples.
    Always ensures functional property annotations are present.

    Args:
        store: The TripleStore wrapper (access raw pyoxigraph Store via store.store).
        ontology_dir: Path to the ontology directory containing schema.ttl and domains/.

    Returns:
        Number of triples loaded from all TTL files.
    """
    raw_store = store.store
    graph_node = NamedNode(KS_GRAPH_ONTOLOGY)

    # Idempotency: skip if ontology graph already has triples
    existing = list(raw_store.quads_for_pattern(None, None, None, graph_node))
    if existing:
        _ensure_functional_properties(raw_store)
        return 0

    initial_count = len(list(raw_store.quads_for_pattern(None, None, None, None)))

    # Load schema.ttl first
    schema_path = ontology_dir / "schema.ttl"
    if schema_path.exists():
        with open(schema_path, "rb") as f:
            raw_store.load(f, RdfFormat.TURTLE, to_graph=graph_node)
        logger.info("Loaded schema.ttl into ontology graph")

    # Additively load all .ttl files from domains/
    domains_dir = ontology_dir / "domains"
    if domains_dir.is_dir():
        for ttl_path in sorted(domains_dir.glob("*.ttl")):
            with open(ttl_path, "rb") as f:
                raw_store.load(f, RdfFormat.TURTLE, to_graph=graph_node)
            logger.info("Loaded domain TTL: %s", ttl_path.name)

    _ensure_functional_properties(raw_store)

    final_count = len(list(raw_store.quads_for_pattern(None, None, None, None)))
    loaded = final_count - initial_count
    logger.info("Bootstrap loaded %d triples into ontology graph", loaded)
    return loaded
