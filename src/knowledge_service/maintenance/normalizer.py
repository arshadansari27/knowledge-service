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

from knowledge_service.ontology.namespaces import KS_KNOWLEDGE_TYPE
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


def normalize_knowledge_types(triple_store: Any) -> dict[str, int]:
    """Lowercase every ``ks:knowledgeType`` RDF-star annotation and collapse
    the ``Relation`` alias to ``relationship``. Production was bifurcated
    across ``Fact``/``fact``, ``Claim``/``claim``, etc. — same logical type,
    different storage value, broken analytics."""
    aliases = {"relation": "relationship"}
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
        raw = str(row["val"].value)
        canon = aliases.get(raw.strip().lower(), raw.strip().lower())
        if raw == canon:
            continue
        bnode = row["bnode"]
        graph_node = row["g"]
        triple_store.remove(Quad(bnode, KS_KNOWLEDGE_TYPE, row["val"], graph_node))
        triple_store.add(Quad(bnode, KS_KNOWLEDGE_TYPE, Literal(canon), graph_node))
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
