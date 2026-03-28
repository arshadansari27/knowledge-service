"""Ingestion worker: orchestrates phases and tracks job progress."""

import json
import logging
from typing import Any

from knowledge_service.ingestion.phases import EmbedPhase, ExtractPhase, ProcessPhase
from knowledge_service.ontology.namespaces import KS_GRAPH_ASSERTED, KS_GRAPH_EXTRACTED

logger = logging.getLogger(__name__)


class JobTracker:
    """Tracks ingestion job progress in the database."""

    def __init__(self, job_id: str, pool: Any):
        self._job_id = job_id
        self._pool = pool

    async def update_status(self, status: str, **kwargs) -> None:
        sets = ["status = $2"]
        params: list = [self._job_id, status]
        for key, value in kwargs.items():
            params.append(value)
            sets.append(f"{key} = ${len(params)}")
        sql = f"UPDATE ingestion_jobs SET {', '.join(sets)} WHERE id = $1::uuid"
        async with self._pool.acquire() as conn:
            await conn.execute(sql, *params)

    async def complete(
        self, triples_created: int, entities_resolved: int, chunks_failed: int
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE ingestion_jobs
                   SET status = 'completed', triples_created = $1,
                       entities_resolved = $2, chunks_failed = $3
                   WHERE id = $4::uuid""",
                triples_created,
                entities_resolved,
                chunks_failed,
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

        # Phase 2: NLP Pre-pass (optional)
        nlp_results = None
        if nlp is not None:
            current_phase = "analyzing"
            await tracker.update_status("analyzing")
            from knowledge_service.nlp import NlpPhase  # noqa: PLC0415

            nlp_phase = NlpPhase(nlp)
            nlp_results = await nlp_phase.run(chunk_records)

        # Phase 3: Extract
        current_phase = "extracting"
        await tracker.update_status("extracting")

        chunks_failed = 0
        if not knowledge and raw_text and extraction_client:
            extract = ExtractPhase(extraction_client)
            knowledge_items, chunk_ids_for_items, chunks_failed = await extract.run(
                chunk_records,
                chunk_id_map,
                title=title,
                source_type=source_type,
                nlp_hints=nlp_results,
            )
            extractor = "llm"
        else:
            knowledge_items = list(knowledge or [])
            chunk_ids_for_items = [None] * len(knowledge_items)
            extractor = "api"

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

        # Phase 5: Process
        current_phase = "processing"
        await tracker.update_status("processing")
        process = ProcessPhase(stores, entity_store, engine=engine)
        triples_created, entities_resolved = await process.run(
            knowledge_items,
            chunk_ids_for_items,
            source_url,
            source_type,
            extractor,
            graph,
        )

        await tracker.complete(triples_created, entities_resolved, chunks_failed)

    except Exception as exc:
        logger.exception("Ingestion failed for job %s in phase %s", job_id, current_phase)
        await tracker.fail(exc, phase=current_phase)
