"""Admin API endpoint for deleting a content item and its contributed triples.

Removes the content's contribution across both stores:
- PostgreSQL: provenance rows for the source URL, then the content_metadata
  row (which cascades to chunks + ingestion_jobs via FK ON DELETE CASCADE).
- pyoxigraph: any triple this content solely supported is removed from all
  content-bearing named graphs; downstream inferences derived from those
  triples are retracted via the same code path used by the ingestion
  pipeline when source triples change.

Triples co-cited from another source are left intact. Noisy-OR confidence
is NOT recomputed for retained triples — operators can re-ingest the
remaining sources if they want the confidence refreshed.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from knowledge_service.ingestion.pipeline import (
    _remove_inferred_triple_with_annotations,
    retract_stale_inferences,
)
from knowledge_service.ontology.namespaces import (
    KS_GRAPH_ASSERTED,
    KS_GRAPH_EXTRACTED,
    KS_GRAPH_FEDERATED,
    KS_GRAPH_INFERRED,
)
from knowledge_service.ontology.uri import is_uri

logger = logging.getLogger(__name__)
router = APIRouter()


_CONTENT_GRAPHS: tuple[str, ...] = (
    KS_GRAPH_ASSERTED,
    KS_GRAPH_EXTRACTED,
    KS_GRAPH_FEDERATED,
    KS_GRAPH_INFERRED,
)


def _remove_orphan_triple(triple_store, subject: str, predicate: str, obj: str) -> None:
    """Remove (s, p, o) from every content-bearing named graph, along with
    its RDF-star annotations. Each graph is independent; pyoxigraph's
    ``remove`` is a no-op when the quad isn't present, so sweeping all
    graphs is safe and avoids needing to ask which graph the triple lives
    in."""
    from pyoxigraph import Literal as Lit
    from pyoxigraph import NamedNode as NN

    s = NN(subject)
    p = NN(predicate)
    o = NN(obj) if is_uri(obj) else Lit(obj)

    for graph_uri in _CONTENT_GRAPHS:
        g = NN(graph_uri)
        try:
            _remove_inferred_triple_with_annotations(triple_store.store, s, p, o, g)
        except Exception:
            logger.exception(
                "delete_content: failed to remove triple from graph %s (s=%s, p=%s)",
                graph_uri,
                subject,
                predicate,
            )


@router.delete("/knowledge/content/{content_id}")
async def delete_content(content_id: str, request: Request) -> dict:
    """Delete a content item and the triples it solely supports.

    Returns a summary: ``{content_id, url, deleted_provenance_rows,
    deleted_triples, retained_triples_with_other_sources,
    retracted_inferences}``.

    404 if ``content_id`` is unknown.
    """
    pg_pool = request.app.state.pg_pool
    knowledge_store = request.app.state.knowledge_store

    async with pg_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT url FROM content_metadata WHERE id = $1",
            content_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail=f"content {content_id} not found")
        url = row["url"]

        prov_rows = await conn.fetch(
            "SELECT triple_hash, subject, predicate, object FROM provenance WHERE source_url = $1",
            url,
        )

        orphans: list[tuple[str, str, str, str]] = []
        retained = 0
        for r in prov_rows:
            other_count = await conn.fetchval(
                "SELECT COUNT(*) FROM provenance WHERE triple_hash = $1 AND source_url != $2",
                r["triple_hash"],
                url,
            )
            if other_count and other_count > 0:
                retained += 1
            else:
                orphans.append((r["triple_hash"], r["subject"], r["predicate"], r["object"]))

        await conn.execute("DELETE FROM provenance WHERE source_url = $1", url)
        await conn.execute("DELETE FROM content_metadata WHERE id = $1", content_id)

    retracted_inferences = 0
    for triple_hash, subj, pred, obj in orphans:
        await asyncio.to_thread(_remove_orphan_triple, knowledge_store, subj, pred, obj)
        retracted_inferences += await asyncio.to_thread(
            retract_stale_inferences, triple_hash, knowledge_store
        )

    logger.info(
        "delete_content: removed content_id=%s url=%s prov=%d orphans=%d retained=%d inferences=%d",
        content_id,
        url,
        len(prov_rows),
        len(orphans),
        retained,
        retracted_inferences,
    )

    return {
        "content_id": content_id,
        "url": url,
        "deleted_provenance_rows": len(prov_rows),
        "deleted_triples": len(orphans),
        "retained_triples_with_other_sources": retained,
        "retracted_inferences": retracted_inferences,
    }
