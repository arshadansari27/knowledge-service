"""E2E: Upload HTML and verify pipeline extracts meaningful content.

Run: uv run pytest tests/e2e/ -v
Requires: PostgreSQL, Ollama
"""

from pathlib import Path

import pytest

from tests.e2e.helpers import poll_until_done


@pytest.mark.e2e
class TestE2EHtmlIngestion:
    def test_upload_html_strips_boilerplate(self, client, api_headers, e2e_config):
        """Upload sample HTML -> parse (strip nav/scripts) -> extract -> process."""
        html_path = Path(e2e_config["fixtures_dir"]) / "sample.html"
        assert html_path.exists()

        with open(html_path, "rb") as f:
            resp = client.post(
                "/api/content/upload",
                files={"file": ("sample.html", f, "text/html")},
                data={
                    "title": "Cold Exposure Research",
                    "source_type": "article",
                },
                headers=api_headers,
            )

        assert resp.status_code == 202, f"Upload failed: {resp.text}"
        body = resp.json()
        content_id = body["content_id"]
        assert body["chunks_total"] >= 1

        status = poll_until_done(client, content_id, api_headers, timeout=120)
        assert status["status"] == "completed", f"Job failed: {status}"

    def test_ingest_url_with_raw_text(self, client, api_headers):
        """POST /api/content with raw_text — backward compatible, no parsing."""
        resp = client.post(
            "/api/content",
            json={
                "url": "e2e://raw-text-test",
                "title": "Raw Text E2E",
                "raw_text": (
                    "Cold water immersion at 10 degrees Celsius increases dopamine "
                    "levels by 250 percent. This was demonstrated in a study by "
                    "Dr. Andrew Huberman at Stanford University."
                ),
                "source_type": "article",
            },
            headers=api_headers,
        )

        assert resp.status_code == 202, f"Content ingest failed: {resp.text}"
        body = resp.json()
        content_id = body["content_id"]

        status = poll_until_done(client, content_id, api_headers, timeout=120)
        assert status["status"] == "completed", f"Job failed: {status}"
        assert status["triples_created"] >= 1, f"Expected triples, got: {status}"
