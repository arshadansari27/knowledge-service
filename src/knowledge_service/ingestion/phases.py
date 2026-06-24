"""Ingestion phase: embed chunks only (extract/process removed)."""

import logging
from typing import Any

from knowledge_service.config import settings

logger = logging.getLogger(__name__)


class EmbedPhase:
    """Phase 1: Embed chunks and store in content table.

    ``batch_size`` controls the size of each embed_batch() call. Defaults to
    ``settings.embed_batch_size`` (``EMBED_BATCH_SIZE`` env var, default 20)
    so operators tuning the env var get the change without a redeploy of
    code-level constants.
    """

    def __init__(
        self,
        embedding_client: Any,
        content_store: Any,
        batch_size: int | None = None,
    ):
        self._embedding_client = embedding_client
        self._content_store = content_store
        self._batch_size = batch_size if batch_size is not None else settings.embed_batch_size

    async def run(
        self,
        content_id: str,
        chunk_records: list[dict],
    ) -> dict[int, str]:
        """Embed all chunks and insert into content table.

        Returns chunk_id_map: {chunk_index: chunk_uuid}.
        """
        texts = [c["chunk_text"] for c in chunk_records]
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            batch_embeddings = await self._embedding_client.embed_batch(batch)
            embeddings.extend(batch_embeddings)

        for rec, emb in zip(chunk_records, embeddings):
            rec["embedding"] = emb

        chunk_id_pairs = await self._content_store.replace_chunks(content_id, chunk_records)
        return dict(chunk_id_pairs) if chunk_id_pairs else {}
