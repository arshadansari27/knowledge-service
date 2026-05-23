from pyoxigraph import NamedNode

# External ontology prefixes
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XSD = "http://www.w3.org/2001/XMLSchema#"
OWL = "http://www.w3.org/2002/07/owl#"

# Custom namespace
KS = "http://knowledge.local/schema/"
KS_DATA = "http://knowledge.local/data/"


def ks(term: str) -> NamedNode:
    """Create a NamedNode in the ks: namespace."""
    return NamedNode(f"{KS}{term}")


# Common ks: terms as constants
KS_CONFIDENCE = ks("confidence")
KS_KNOWLEDGE_TYPE = ks("knowledgeType")
KS_VALID_FROM = ks("validFrom")
KS_VALID_UNTIL = ks("validUntil")
KS_OPPOSITE_PREDICATE = ks("oppositePredicate")
KS_INVERSE_PREDICATE = ks("inversePredicate")
KS_TRANSITIVE_PREDICATE = ks("transitivePredicate")

# Named graphs for trust-tier separation
KS_GRAPH_ONTOLOGY = f"{KS}graph/ontology"
KS_GRAPH_ASSERTED = f"{KS}graph/asserted"
KS_GRAPH_EXTRACTED = f"{KS}graph/extracted"
KS_GRAPH_INFERRED = f"{KS}graph/inferred"
KS_GRAPH_FEDERATED = f"{KS}graph/federated"
