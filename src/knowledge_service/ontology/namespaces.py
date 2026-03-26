from pyoxigraph import NamedNode

# External ontology prefixes
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
DC = "http://purl.org/dc/elements/1.1/"
DCTERMS = "http://purl.org/dc/terms/"
SCHEMA = "http://schema.org/"
SKOS = "http://www.w3.org/2004/02/skos/core#"
OWL = "http://www.w3.org/2002/07/owl#"

# Custom namespace
KS = "http://knowledge.local/schema/"
KS_DATA = "http://knowledge.local/data/"


def ks(term: str) -> NamedNode:
    """Create a NamedNode in the ks: namespace."""
    return NamedNode(f"{KS}{term}")


def ks_data(term: str) -> NamedNode:
    """Create a NamedNode in the ks-data: namespace."""
    return NamedNode(f"{KS_DATA}{term}")


def rdf(term: str) -> NamedNode:
    return NamedNode(f"{RDF}{term}")


def rdfs(term: str) -> NamedNode:
    return NamedNode(f"{RDFS}{term}")


def xsd(term: str) -> NamedNode:
    return NamedNode(f"{XSD}{term}")


def schema(term: str) -> NamedNode:
    return NamedNode(f"{SCHEMA}{term}")


def skos(term: str) -> NamedNode:
    return NamedNode(f"{SKOS}{term}")


def owl(term: str) -> NamedNode:
    return NamedNode(f"{OWL}{term}")


# Common ks: terms as constants
KS_CONFIDENCE = ks("confidence")
KS_KNOWLEDGE_TYPE = ks("knowledgeType")
KS_VALID_FROM = ks("validFrom")
KS_VALID_UNTIL = ks("validUntil")
KS_CLAIM = ks("Claim")
KS_FACT = ks("Fact")
KS_EVENT = ks("Event")
KS_ENTITY = ks("Entity")
KS_RELATIONSHIP = ks("Relationship")
KS_CONCLUSION = ks("Conclusion")
KS_TEMPORAL_STATE = ks("TemporalState")
KS_OPPOSITE_PREDICATE = ks("oppositePredicate")
KS_INVERSE_PREDICATE = ks("inversePredicate")

# Named graphs for trust-tier separation
KS_GRAPH_ONTOLOGY = f"{KS}graph/ontology"
KS_GRAPH_ASSERTED = f"{KS}graph/asserted"
KS_GRAPH_EXTRACTED = f"{KS}graph/extracted"
KS_GRAPH_INFERRED = f"{KS}graph/inferred"
KS_GRAPH_FEDERATED = f"{KS}graph/federated"


def ensure_entity_uri(value: str) -> str:
    """Ensure a value is a valid entity URI. Bare labels are slugified under KS_DATA."""
    if value.startswith(("http://", "https://", "urn:")):
        return value
    from urllib.parse import quote
    slug = quote(value, safe="/_-:.~")
    return f"{KS_DATA}{slug}"


def ensure_predicate_uri(value: str) -> str:
    """Ensure a value is a valid predicate URI. Bare labels are slugified under KS."""
    if value.startswith(("http://", "https://", "urn:")):
        return value
    from urllib.parse import quote
    slug = quote(value, safe="/_-:.~")
    return f"{KS}{slug}"
