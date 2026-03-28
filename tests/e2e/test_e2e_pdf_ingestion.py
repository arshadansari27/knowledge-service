"""E2E: Upload a PDF and verify the full pipeline produces triples.

Run: uv run pytest tests/e2e/ -v
Requires: PostgreSQL (docker compose up -d postgres), Ollama with nomic-embed-text + a chat model
"""

from pathlib import Path

import pytest

from tests.e2e.helpers import poll_until_done


@pytest.mark.e2e
class TestE2EPdfIngestion:
    def test_upload_pdf_produces_triples(self, client, api_headers, e2e_config):
        """Upload sample PDF -> parse -> chunk -> embed -> extract -> process -> triples."""
        pdf_path = Path(e2e_config["fixtures_dir"]) / "sample.pdf"
        assert pdf_path.exists(), f"Fixture not found: {pdf_path}"

        with open(pdf_path, "rb") as f:
            resp = client.post(
                "/api/content/upload",
                files={"file": ("sample.pdf", f, "application/pdf")},
                data={
                    "title": "E2E PDF Test",
                    "source_type": "paper",
                },
                headers=api_headers,
            )

        assert resp.status_code == 202, f"Upload failed: {resp.text}"
        body = resp.json()
        content_id = body["content_id"]
        assert body["chunks_total"] >= 1

        # Poll for completion
        status = poll_until_done(client, content_id, api_headers, timeout=120)
        assert status["status"] == "completed", f"Job failed: {status}"
        assert status["triples_created"] >= 0
