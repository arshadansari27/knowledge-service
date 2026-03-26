"""One-time migration: move triples from default graph to named graphs."""

from __future__ import annotations

import hashlib
import logging

from pyoxigraph import DefaultGraph, Literal, NamedNode, Quad, Store, Triple

from knowledge_service.ontology.namespaces import (
    KS,
    KS_GRAPH_ASSERTED,
    KS_GRAPH_EXTRACTED,
    KS_GRAPH_FEDERATED,  # noqa: F401
    KS_GRAPH_ONTOLOGY,
)

logger = logging.getLogger(__name__)

MIGRATION_MARKER_S = NamedNode(f"{KS}migration/named_graphs")
MIGRATION_MARKER_P = NamedNode(f"{KS}completedAt")


def _is_migration_done(store: Store) -> bool:
    quads = list(
        store.quads_for_pattern(
            MIGRATION_MARKER_S, MIGRATION_MARKER_P, None, NamedNode(KS_GRAPH_ONTOLOGY)
        )
    )
    return len(quads) > 0


def _triple_hash(triple: Triple) -> str:
    return hashlib.sha256(str(triple).encode()).hexdigest()


async def migrate_to_named_graphs(store: Store, pg_pool) -> int:
    """Migrate triples from default graph to named graphs.
    Returns number of triples migrated, or 0 if already done.
    """
    if _is_migration_done(store):
        logger.info("Named graph migration already completed, skipping")
        return 0

    # Phase A: Read all triples from default graph
    default_quads = list(store.quads_for_pattern(None, None, None, DefaultGraph()))
    if not default_quads:
        logger.info("No triples in default graph, nothing to migrate")
        return 0

    logger.info("Migrating %d quads from default graph to named graphs", len(default_quads))

    # Identify ontology triples (KS namespace subjects)
    ontology_subjects = set()
    for q in default_quads:
        s_val = q.subject.value if hasattr(q.subject, "value") else str(q.subject)
        if s_val.startswith(KS):
            ontology_subjects.add(s_val)

    # Batch lookup provenance extractors
    triple_hashes = {}
    for q in default_quads:
        t = Triple(q.subject, q.predicate, q.object)
        triple_hashes[_triple_hash(t)] = q

    extractor_map = {}
    if pg_pool is not None:
        async with pg_pool.acquire() as conn:
            hashes = list(triple_hashes.keys())
            for i in range(0, len(hashes), 500):
                batch = hashes[i : i + 500]
                rows = await conn.fetch(
                    "SELECT DISTINCT triple_hash, extractor FROM provenance WHERE triple_hash = ANY($1)",
                    batch,
                )
                for row in rows:
                    extractor_map[row["triple_hash"]] = row["extractor"]

    # Phase A: Copy to named graphs
    migrated = 0
    for q in default_quads:
        s_val = q.subject.value if hasattr(q.subject, "value") else str(q.subject)
        t = Triple(q.subject, q.predicate, q.object)
        th = _triple_hash(t)

        if s_val in ontology_subjects:
            target = KS_GRAPH_ONTOLOGY
        elif extractor_map.get(th, "").startswith("llm_"):
            target = KS_GRAPH_EXTRACTED
        else:
            target = KS_GRAPH_ASSERTED

        store.add(Quad(q.subject, q.predicate, q.object, NamedNode(target)))
        migrated += 1

    # Phase B: Delete from default graph
    for q in default_quads:
        store.remove(q)

    # Write completion marker
    from datetime import datetime

    store.add(
        Quad(
            MIGRATION_MARKER_S,
            MIGRATION_MARKER_P,
            Literal(datetime.now().isoformat()),
            NamedNode(KS_GRAPH_ONTOLOGY),
        )
    )

    store.flush()
    logger.info("Named graph migration complete: %d triples migrated", migrated)
    return migrated
