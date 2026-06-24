"""RAGRetriever — chunk-only semantic retrieval across content store."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Results from semantic retrieval (chunks only, no graph)."""

    content_results: list[dict] = field(default_factory=list)


class RAGRetriever:
    """Semantic retriever over chunked content (graph layer removed)."""

    def __init__(
        self,
        embedding_client,
        embedding_store,
    ) -> None:
        """Initialize RAGRetriever with embedding and content stores."""
        self._embedding_client = embedding_client
        self._embedding_store = embedding_store  # ContentStore (search, get_chunks_by_ids)

    async def retrieve(
        self,
        question: str,
        max_sources: int = 5,
        min_confidence: float = 0.0,
        **kwargs,  # noqa: ARG002 — ignore intent/retrieval_mode for backward compat
    ) -> RetrievalContext:
        """Retrieve content chunks via semantic search (chunk-only path)."""
        embedding = await self._embedding_client.embed(question)
        content_results = await self._embedding_store.search(
            query_embedding=embedding, limit=max_sources, query_text=question
        )
        return RetrievalContext(content_results=content_results)
