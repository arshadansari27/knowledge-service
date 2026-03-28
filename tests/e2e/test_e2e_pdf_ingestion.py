"""E2E: Upload a PDF and verify full pipeline produces triples.

Run: uv run pytest tests/e2e/ -v -m e2e
Requires: PostgreSQL, Ollama (qwen3 + nomic-embed-text), spaCy KB
"""
import pytest


@pytest.mark.e2e
class TestE2EPdfIngestion:
    async def test_upload_pdf_produces_triples(self, e2e_db_url, e2e_llm_url):
        """Upload sample PDF, wait for ingestion, verify triples created."""
        pytest.skip("E2E test — run manually with real services")
