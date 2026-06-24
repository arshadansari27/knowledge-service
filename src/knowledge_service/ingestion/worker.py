"""Ingestion worker: orchestrates phases and tracks job progress."""

import json
import logging
from typing import Any

from knowledge_service.ingestion.phases import EmbedPhase

logger = logging.getLogger(__name__)


_ALLOWED_JOB_COLUMNS = frozenset(
    {
        "chunks_embedded",
        "chunks_extracted",
        "chunks_failed",
        "items_rejected",
        "triples_created",
        "entities_resolved",
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
        items_rejected: int = 0,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """UPDATE ingestion_jobs
                   SET status = 'completed', triples_created = $1,
                       entities_resolved = $2, chunks_failed = $3,
                       items_rejected = $4
                   WHERE id = $5::uuid""",
                triples_created,
                entities_resolved,
                chunks_failed,
                items_rejected,
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
    stores: Any,
    embedding_client: Any,
) -> None:
    """Orchestrate the multi-phase ingestion pipeline (embed-only).

    Phases 1 (embed) runs; phases 2-5 (NLP, extract, coreference, process/graph)
    are skipped. ponytail: extraction/triple/entity/inference code is retained
    for phase 3 deletion; it's just not called from the ingest path anymore.

    Args:
        job_id: UUID of the ingestion job.
        content_id: UUID of the content being ingested.
        chunk_records: List of chunk dicts with chunk_text, chunk_index, etc.
        (other args unused after embed, retained for backward compat)
    """
    tracker = JobTracker(job_id, stores.pg_pool)
    current_phase = "embedding"

    try:
        # Phase 1: Embed
        await tracker.update_status("embedding")
        embed = EmbedPhase(embedding_client, stores.content)
        chunk_id_map = await embed.run(content_id, chunk_records)
        await tracker.update_status("embedding", chunks_embedded=len(chunk_id_map))

        # ponytail: phases 2-5 (NLP pre-pass, extract, coreference, process/graph)
        # skipped. Ingest is now chunk + embed only. These phases remain in the
        # codebase for phase 3 complete deletion; they're just unreachable here.

        await tracker.complete(
            triples_created=0,
            entities_resolved=0,
            chunks_failed=0,
            items_rejected=0,
        )

    except Exception as exc:
        logger.exception("Ingestion failed for job %s in phase %s", job_id, current_phase)
        await tracker.fail(exc, phase=current_phase)
