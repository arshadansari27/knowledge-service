"""OpenAI-compatible client for embeddings."""

from __future__ import annotations

import logging

import httpx

from knowledge_service.clients.base import BaseLLMClient

logger = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Raised when the LLM API returns an error or unexpected response."""


class EmbeddingClient(BaseLLMClient):
    """HTTP client wrapping the OpenAI-compatible embeddings API."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        # 120 s read budget. The shared Ollama instance hosting
        # ``nomic-embed-text`` runs alongside other models on a busy node;
        # 32-chunk batch requests routinely exceed 30 s during ingestion
        # bursts. Prod metrics 2026-04 showed ~225 ingestion jobs failing per
        # 14 d in the embedding phase from read timeouts at the previous
        # 30 s budget.
        super().__init__(base_url, model, api_key, read_timeout=120.0)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text."""
        vectors = await self._request([text])
        return vectors[0]

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        When batch_size is set, splits into sub-batches to avoid overwhelming
        the embedding endpoint. Default (None) sends all texts in one request.
        """
        if not texts:
            return []
        if batch_size is None or batch_size >= len(texts):
            return await self._request(texts)
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(await self._request(batch))
        return results

    async def _request(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                "/v1/embeddings",
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(f"LLM API returned {exc.response.status_code}") from exc
        except httpx.TimeoutException as exc:
            raise LLMClientError(f"LLM API request timed out: {exc}") from exc
        data = response.json()
        return [item["embedding"] for item in data["data"]]
