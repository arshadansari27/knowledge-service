"""Ingestion worker: orchestrates phases and tracks job progress."""

import json
import logging
import time
from typing import Any

from knowledge_service.ingestion.phases import EmbedPhase, ExtractPhase, ProcessPhase
from knowledge_service.ontology.namespaces import KS_GRAPH_ASSERTED, KS_GRAPH_EXTRACTED

logger = logging.getLogger(__name__)


_ALLOWED_JOB_COLUMNS = frozenset(
    {
        "chunks_embedded",
        "chunks_extracted",
        "chunks_failed",
        "chunks_skipped",
        "triples_created",
        "entities_resolved",
        "entities_linked",
        "entities_coref",
        "error",
    }
)


class JobTracker:
    """Tracks ingestion job progress in the database."""

    def __init__(self, job_id: str, pool: Any):
        self._job_id = job_id
        self._pool = pool

    async def update_status(self, status: str, **kwargs) -> None:
        invalid = set(kwargs) - _ALLOWED_JOB_COLUMNS
        if invalid:
            raise ValueError(f"Invalid job columns: {invalid}")
        sets = ["status = $2"]
        params: list = [self._job_id, status]
        for key, value in kwargs.items():
            params.append(value)
            sets.append(f"{key} = ${len(params)}")
        sql = f"UPDATE ingestion_jobs SET {', '.join(sets)} WHERE id = $1::uuid"
        async with self._pool.acquire() as conn:
            await conn.execute(sql, *params)

    async def complete(
        self,
        triples_created: int,
        entities_resolved: int,
        chunks_failed: int,
        chunks_skipped: int = 0,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE ingestion_jobs
                   SET status = 'completed', triples_created = $1,
                       entities_resolved = $2, chunks_failed = $3,
                       chunks_skipped = $4
                   WHERE id = $5::uuid""",
                triples_created,
                entities_resolved,
                chunks_failed,
                chunks_skipped,
                self._job_id,
            )

    async def fail(self, exc: Exception, phase: str = "unknown") -> None:
        error_json = json.dumps(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "phase": phase,
            }
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE ingestion_jobs SET status = 'failed', error = $1 WHERE id = $2::uuid",
                error_json,
                self._job_id,
            )


def _should_rebuild_communities(
    triples_created: int,
    total_triples: int,
    min_triples: int,
    last_rebuild: float,
    cooldown: int,
) -> bool:
    """Check if community detection should be triggered."""
    if triples_created <= 0:
        return False
    if total_triples < min_triples:
        return False
    if time.time() - last_rebuild < cooldown:
        return False
    return True


async def run_ingestion(
    job_id: str,
    content_id: str,
    chunk_records: list[dict],
    raw_text: str | None,
    knowledge: list | None,
    title: str | None,
    source_url: str,
    source_type: str,
    stores: Any,
    embedding_client: Any,
    extraction_client: Any | None,
    entity_store: Any | None = None,
    engine: Any | None = None,
    nlp: Any | None = None,
    federation_client: Any | None = None,
    app_state: Any | None = None,
) -> None:
    """Orchestrate the multi-phase ingestion pipeline.

    Args:
        job_id: UUID of the ingestion job.
        content_id: UUID of the content being ingested.
        chunk_records: List of chunk dicts with chunk_text, chunk_index, etc.
        raw_text: Original raw text (if extraction needed).
        knowledge: Pre-supplied knowledge items (if any).
        title: Content title for extraction context.
        source_url: URL of the content source.
        source_type: Type of the source (article, paper, etc.).
        stores: Stores dataclass with triples, content, entities, provenance.
        embedding_client: Client for generating embeddings.
        extraction_client: Client for LLM extraction (optional if knowledge pre-supplied).
        entity_store: Optional entity store for resolution.
        engine: Optional reasoning engine.
        nlp: Optional spaCy nlp pipeline. When provided, enables NLP pre-pass
             and coreference resolution phases.
    """
    tracker = JobTracker(job_id, stores.pg_pool)
    current_phase = "embedding"

    try:
        # Phase 1: Embed
        await tracker.update_status("embedding")
        embed = EmbedPhase(embedding_client, stores.content)
        chunk_id_map = await embed.run(content_id, chunk_records)
        await tracker.update_status("embedding", chunks_embedded=len(chunk_id_map))

        # Phase 2: NLP Pre-pass (optional)
        nlp_results = None
        if nlp is not None:
            current_phase = "analyzing"
            await tracker.update_status("analyzing")
            from knowledge_service.nlp import NlpPhase  # noqa: PLC0415

            nlp_phase = NlpPhase(nlp)
            nlp_results = await nlp_phase.run(chunk_records)
            entities_linked = sum(1 for r in nlp_results for e in r.entities if e.wikidata_id)
            await tracker.update_status("analyzing", entities_linked=entities_linked)

        # Phase 3: Extract
        current_phase = "extracting"
        await tracker.update_status("extracting")

        chunks_failed = 0
        chunks_skipped = 0
        if not knowledge and raw_text and extraction_client:
            extract = ExtractPhase(extraction_client)
            knowledge_items, chunk_ids_for_items, chunks_failed, chunks_skipped = await extract.run(
                chunk_records,
                chunk_id_map,
                title=title,
                source_type=source_type,
                nlp_hints=nlp_results,
            )
            extractor = "llm"
            chunks_extracted = len(chunk_records) - chunks_failed - chunks_skipped
        else:
            knowledge_items = list(knowledge or [])
            chunk_ids_for_items = [None] * len(knowledge_items)
            extractor = "api"
            chunks_extracted = 0

        await tracker.update_status(
            "extracting",
            chunks_extracted=chunks_extracted,
            chunks_failed=chunks_failed,
            chunks_skipped=chunks_skipped,
        )
        graph = KS_GRAPH_ASSERTED if extractor == "api" else KS_GRAPH_EXTRACTED

        # Phase 4: Coreference (optional — requires NLP results + extracted items)
        if nlp_results and knowledge_items and extraction_client:
            current_phase = "resolving"
            await tracker.update_status("resolving")
            from knowledge_service.ingestion.coreference import (  # noqa: PLC0415
                CoreferencePhase,
            )

            coref = CoreferencePhase(extraction_client, stores.pg_pool)
            coref_result = await coref.run(knowledge_items, nlp_results)
            knowledge_items = coref_result.canonicalize(knowledge_items)
            await tracker.update_status("resolving", entities_coref=len(coref_result.groups))

        # Phase 5: Process
        current_phase = "processing"
        await tracker.update_status("processing")
        drainer = getattr(app_state, "outbox_drainer", None) if app_state is not None else None
        process = ProcessPhase(stores, entity_store, engine=engine, drainer=drainer)
        triples_created, entities_resolved = await process.run(
            knowledge_items,
            chunk_ids_for_items,
            source_url,
            source_type,
            extractor,
            graph,
        )

        await tracker.complete(triples_created, entities_resolved, chunks_failed, chunks_skipped)

        if chunks_failed > 0 and triples_created == 0:
            total_chunks = len(chunk_records)
            logger.warning(
                "Ingestion job %s: all %d/%d chunks failed extraction, 0 triples created "
                "(title=%s, url=%s)",
                job_id,
                chunks_failed,
                total_chunks,
                title,
                source_url,
            )

        # Background federation enrichment (best-effort, after job marked complete)
        if federation_client is not None and triples_created > 0:
            try:
                from knowledge_service.ingestion.federation import (  # noqa: PLC0415
                    FederationPhase,
                )

                fed_phase = FederationPhase(
                    federation_client=federation_client,
                    triple_store=stores.triples,
                    max_lookups=10,
                    delay=1.0,
                )
                # Collect entity labels from knowledge items
                fed_entities = []
                for item in knowledge_items:
                    if hasattr(item, "label") and hasattr(item, "uri"):
                        fed_entities.append({"label": item.label, "uri": item.uri})
                    elif isinstance(item, dict):
                        label = item.get("label") or item.get("subject", "")
                        uri = item.get("uri") or item.get("subject", "")
                        if label and uri:
                            fed_entities.append({"label": label, "uri": uri})
                if fed_entities:
                    fed_result = await fed_phase.run(fed_entities)
                    logger.info(
                        "Federation enrichment for job %s: %d enriched, %d skipped",
                        job_id,
                        fed_result.entities_enriched,
                        fed_result.entities_skipped,
                    )
            except Exception:
                logger.warning("Federation enrichment failed for job %s", job_id, exc_info=True)

        # Auto-trigger community detection if conditions met
        if triples_created > 0 and app_state is not None:
            try:
                from knowledge_service.config import settings as _settings  # noqa: PLC0415

                total = stores.triples.count_triples()
                last_rebuild = getattr(app_state, "_last_community_rebuild", 0.0)

                if _should_rebuild_communities(
                    triples_created=triples_created,
                    total_triples=total,
                    min_triples=_settings.community_min_triples,
                    last_rebuild=last_rebuild,
                    cooldown=_settings.community_cooldown,
                ):
                    import asyncio  # noqa: PLC0415

                    from knowledge_service.stores.community import (  # noqa: PLC0415
                        CommunityDetector,
                        CommunitySummarizer,
                    )

                    detector = CommunityDetector(stores.triples)
                    communities = await asyncio.to_thread(detector.detect)
                    if communities and extraction_client:
                        summarizer = CommunitySummarizer(
                            extraction_client._client,
                            stores.triples,
                            model=extraction_client._model,
                        )
                        summarized = []
                        for c in communities:
                            summarized.append(await summarizer.summarize_one(c))
                        community_store = getattr(app_state, "community_store", None)
                        if community_store:
                            await community_store.replace_all(summarized)
                            app_state._last_community_rebuild = time.time()
                            logger.info("Auto community rebuild: %d communities", len(summarized))
            except Exception:
                logger.warning("Auto community rebuild failed", exc_info=True)

    except Exception as exc:
        logger.exception("Ingestion failed for job %s in phase %s", job_id, current_phase)
        await tracker.fail(exc, phase=current_phase)
