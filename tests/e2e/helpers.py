"""Shared helpers for E2E tests."""

import time


def poll_until_done(client, content_id: str, headers: dict, timeout: int = 120) -> dict:
    """Poll /api/content/{id}/status until the job completes or times out."""
    elapsed = 0
    interval = 3

    while elapsed < timeout:
        resp = client.get(f"/api/content/{content_id}/status", headers=headers)
        if resp.status_code == 404:
            time.sleep(interval)
            elapsed += interval
            continue

        status = resp.json()
        if status["status"] in ("completed", "failed"):
            return status

        time.sleep(interval)
        elapsed += interval

    raise TimeoutError(f"Ingestion not completed within {timeout}s for content {content_id}")
