"""Admin API endpoints for deleting an ingested source's contribution.

Two surfaces are exposed because ingestion has two write paths:

- ``DELETE /api/admin/knowledge/content/{content_id}`` — for content
  ingested via ``/api/content`` (has a row in ``content_metadata``).
- ``DELETE /api/admin/knowledge/source?source_url=…`` — for sources
  ingested via ``/api/claims`` (provenance only, no ``content_metadata``).

Both share the same cleanup helper and the same semantics:

- PostgreSQL: provenance rows for the source URL are deleted; ``content_metadata``
  (when present) is deleted last so its FK ``ON DELETE CASCADE`` takes
  chunks and ingestion_jobs with it.
- pyoxigraph: any triple this source solely supported is removed from all
  content-bearing named graphs along with its RDF-star annotations;
  downstream inferences derived from those triples are retracted via the
  same code path the ingestion pipeline uses when source triples change.

Triples co-cited from another source are left intact. Noisy-OR confidence
is NOT recomputed for retained triples — operators can re-ingest the
remaining sources if they want the confidence refreshed.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, Request

from knowledge_service.ingestion.pipeline import (
    _remove_inferred_triple_with_annotations,
    retract_stale_inferences,
)
from knowledge_service.ontology.namespaces import (
    KS_GRAPH_ASSERTED,
    KS_GRAPH_EXTRACTED,
    KS_GRAPH_INFERRED,
)
from knowledge_service.ontology.uri import is_uri

logger = logging.getLogger(__name__)
router = APIRouter()


_CONTENT_GRAPHS: tuple[str, ...] = (
    KS_GRAPH_ASSERTED,
    KS_GRAPH_EXTRACTED,
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


async def _purge_source(conn, knowledge_store, url: str) -> dict:
    """Delete provenance + content_metadata for ``url``, then orphan-purge
    triples from pyoxigraph. Returns the operation summary (without
    ``content_id`` — caller adds that when relevant)."""
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
    # No-op when there's no content_metadata row (e.g. /api/claims path).
    await conn.execute("DELETE FROM content_metadata WHERE url = $1", url)

    retracted_inferences = 0
    for triple_hash, subj, pred, obj in orphans:
        await asyncio.to_thread(_remove_orphan_triple, knowledge_store, subj, pred, obj)
        retracted_inferences += await asyncio.to_thread(
            retract_stale_inferences, triple_hash, knowledge_store
        )

    return {
        "url": url,
        "deleted_provenance_rows": len(prov_rows),
        "deleted_triples": len(orphans),
        "retained_triples_with_other_sources": retained,
        "retracted_inferences": retracted_inferences,
    }


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
        summary = await _purge_source(conn, knowledge_store, row["url"])

    summary["content_id"] = content_id
    logger.info(
        "delete_content: content_id=%s url=%s prov=%d orphans=%d retained=%d inferences=%d",
        content_id,
        summary["url"],
        summary["deleted_provenance_rows"],
        summary["deleted_triples"],
        summary["retained_triples_with_other_sources"],
        summary["retracted_inferences"],
    )
    return summary


@router.delete("/knowledge/source")
async def delete_source(
    request: Request,
    source_url: str = Query(..., description="Exact source URL whose ingestion to purge"),
) -> dict:
    """Delete every triple solely supported by ``source_url``.

    Counterpart to ``DELETE /knowledge/content/{content_id}`` for ingestions
    that came in via ``/api/claims`` (no ``content_metadata`` row to delete
    by id). Uses the same cleanup semantics — co-cited triples are retained,
    inferences derived from orphan triples are cascade-retracted.

    404 if no provenance rows exist for ``source_url``.
    """
    pg_pool = request.app.state.pg_pool
    knowledge_store = request.app.state.knowledge_store

    async with pg_pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM provenance WHERE source_url = $1",
            source_url,
        )
        if not count:
            raise HTTPException(
                status_code=404,
                detail=f"no provenance rows for source_url={source_url!r}",
            )
        summary = await _purge_source(conn, knowledge_store, source_url)

    logger.info(
        "delete_source: url=%s prov=%d orphans=%d retained=%d inferences=%d",
        summary["url"],
        summary["deleted_provenance_rows"],
        summary["deleted_triples"],
        summary["retained_triples_with_other_sources"],
        summary["retracted_inferences"],
    )
    return summary
