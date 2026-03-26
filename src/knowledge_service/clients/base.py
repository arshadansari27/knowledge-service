"""Shared base class for OpenAI-compatible LLM HTTP clients."""

from __future__ import annotations

import httpx


class BaseLLMClient:
    """Base for HTTP clients wrapping OpenAI-compatible APIs.

    Handles URL normalization (strip trailing /, remove /v1 suffix),
    auth header, and timeout configuration.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        read_timeout: float = 30.0,
    ) -> None:
        self._model = model
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self._client = httpx.AsyncClient(
            base_url=url,
            headers=headers,
            timeout=httpx.Timeout(connect=5.0, read=read_timeout, write=10.0, pool=5.0),
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if not self._client.is_closed:
            await self._client.aclose()
