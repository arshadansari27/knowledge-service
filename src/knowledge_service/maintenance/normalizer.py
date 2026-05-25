"""Idempotent data-quality cleanup operations.

Each operation:
    - Returns a stat dict ``{changed: int, scanned: int, ...}``
    - Is safe to run repeatedly (no-op after convergence)
    - Targets a specific drift surfaced in the 2026-05-26 production audit
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pyoxigraph import Literal, NamedNode, Quad

from knowledge_service.ontology.namespaces import KS, KS_KNOWLEDGE_TYPE
from knowledge_service.ontology.uri import RDF_TYPE

logger = logging.getLogger(__name__)


# spaCy NER labels emitted under ``schema:`` need remapping to schema.org
# canonical names. Labels that map to ``None`` describe values, not entities,
# and should be dropped from rdf:type entirely (they were never valid types).
# Mirrors the live filter in ``ingestion/phases.py:_SPACY_LABEL_TO_SCHEMA``.
_SCHEMA_TYPE_REMAP: dict[str, str | None] = {
    "schema:ORG": "schema:Organization",
    "schema:PERSON": "schema:Person",
    "schema:GPE": "schema:Place",
    "schema:LOC": "schema:Place",
    "schema:FAC": "schema:Place",
    "schema:NORP": "schema:Organization",
    "schema:PRODUCT": "schema:Product",
    "schema:WORK_OF_ART": "schema:CreativeWork",
    "schema:EVENT": "schema:Event",
    "schema:LAW": "schema:Legislation",
    "schema:LANGUAGE": "schema:Language",
    "schema:CARDINAL": None,
    "schema:ORDINAL": None,
    "schema:QUANTITY": None,
    "schema:PERCENT": None,
    "schema:MONEY": None,
    "schema:DATE": None,
    "schema:TIME": None,
    # Without the ``schema:`` prefix — same fix.
    "Thing": "schema:Thing",
    "Organization": "schema:Organization",
    "Person": "schema:Person",
    "Place": "schema:Place",
    "Product": "schema:Product",
    "CreativeWork": "schema:CreativeWork",
    "Event": "schema:Event",
}


_KNOWLEDGE_TYPE_ALIASES = {"relation": "relationship"}


def _canonical_knowledge_type_uri(raw_value: str) -> str:
    """Strip a ``http://knowledge.local/schema/`` prefix if present, lowercase
    the suffix, and apply alias collapse — returns the canonical ``ks:<name>``
    URI string. Handles both already-URI inputs and bare type names that
    might have slipped in as literals."""
    stripped = raw_value.strip()
    if stripped.startswith(KS):
        stripped = stripped[len(KS) :]
    suffix = _KNOWLEDGE_TYPE_ALIASES.get(stripped.lower(), stripped.lower())
    return f"{KS}{suffix}"


def normalize_knowledge_types(triple_store: Any) -> dict[str, int]:
    """Canonicalise every ``ks:knowledgeType`` RDF-star annotation to a
    lowercase ``<ks:type>`` NamedNode. The ingestion path stores these as
    ``<{KS}{knowledge_type}>`` URIs (see ``stores/triples.py:149``); we
    rewrite any that drifted to mixed-case URIs (``ks:Fact``), to literal
    strings, or to the wrong shape entirely. Also collapses the
    ``Relation`` alias to ``relationship``."""
    select = f"""
        SELECT ?g ?bnode ?val WHERE {{
            GRAPH ?g {{
                ?bnode <{KS_KNOWLEDGE_TYPE.value}> ?val .
            }}
        }}
    """
    rows = list(triple_store.query(select))

    changed = 0
    for row in rows:
        old_term = row["val"]
        # Both NamedNode.value and Literal.value expose ``.value`` as a str.
        raw = str(old_term.value)
        canon_uri = _canonical_knowledge_type_uri(raw)
        new_term = NamedNode(canon_uri)
        if old_term == new_term:
            continue
        bnode = row["bnode"]
        graph_node = row["g"]
        triple_store.remove(Quad(bnode, KS_KNOWLEDGE_TYPE, old_term, graph_node))
        triple_store.add(Quad(bnode, KS_KNOWLEDGE_TYPE, new_term, graph_node))
        changed += 1

    return {"scanned": len(rows), "changed": changed}


def normalize_spacy_rdf_types(triple_store: Any) -> dict[str, int]:
    """Remap ``schema:PERSON`` → ``schema:Person`` (and friends), and drop
    ``rdf:type`` triples whose value is a numeric/quantity label
    (``CARDINAL``, ``MONEY``, ``PERCENT``, …) — those describe values, not
    entities, and were never valid as rdf:type."""
    rdf_type = NamedNode(RDF_TYPE)
    select = f"""
        SELECT ?g ?s ?o WHERE {{
            GRAPH ?g {{
                ?s <{RDF_TYPE}> ?o .
            }}
            FILTER(isLiteral(?o))
        }}
    """
    rows = list(triple_store.query(select))

    remapped = 0
    dropped = 0
    for row in rows:
        value = str(row["o"].value)
        if value not in _SCHEMA_TYPE_REMAP:
            continue
        target = _SCHEMA_TYPE_REMAP[value]
        subject = row["s"]
        graph_node = row["g"]
        old_obj = row["o"]
        triple_store.remove(Quad(subject, rdf_type, old_obj, graph_node))
        if target is None:
            dropped += 1
        else:
            triple_store.add(Quad(subject, rdf_type, Literal(target), graph_node))
            remapped += 1

    return {"scanned": len(rows), "remapped": remapped, "dropped": dropped}


async def run_all(stores: Any) -> dict[str, dict[str, int]]:
    """Run every cleanup operation. Returns a dict keyed by operation name.

    Operations run on the pyoxigraph store, so they're CPU-bound — dispatch
    them to a thread to avoid blocking the event loop.
    """
    triple_store = stores.triples
    raw_store = triple_store.store  # pyoxigraph Store

    kt_stats = await asyncio.to_thread(normalize_knowledge_types, raw_store)
    logger.info(
        "maintenance: knowledge_type normalization scanned=%d changed=%d",
        kt_stats["scanned"],
        kt_stats["changed"],
    )

    rdf_stats = await asyncio.to_thread(normalize_spacy_rdf_types, raw_store)
    logger.info(
        "maintenance: rdf:type normalization scanned=%d remapped=%d dropped=%d",
        rdf_stats["scanned"],
        rdf_stats["remapped"],
        rdf_stats["dropped"],
    )

    return {
        "knowledge_type": kt_stats,
        "rdf_type": rdf_stats,
    }
