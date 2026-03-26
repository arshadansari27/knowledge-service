"""Tests for BaseLLMClient shared initialization."""

import asyncio

from knowledge_service.clients.base import BaseLLMClient


def test_base_strips_trailing_slash():
    client = BaseLLMClient(base_url="http://localhost:11434/", model="test", api_key="")
    assert str(client._client.base_url).rstrip("/") == "http://localhost:11434"
    asyncio.run(client._client.aclose())


def test_base_strips_v1_suffix():
    client = BaseLLMClient(base_url="http://localhost:11434/v1", model="test", api_key="")
    assert "/v1/v1" not in str(client._client.base_url)
    asyncio.run(client._client.aclose())


def test_base_sets_auth_header():
    client = BaseLLMClient(base_url="http://localhost:11434", model="test", api_key="sk-test")
    assert client._client.headers.get("Authorization") == "Bearer sk-test"
    asyncio.run(client._client.aclose())


def test_base_no_auth_header_when_empty():
    client = BaseLLMClient(base_url="http://localhost:11434", model="test", api_key="")
    assert "Authorization" not in client._client.headers
    asyncio.run(client._client.aclose())
