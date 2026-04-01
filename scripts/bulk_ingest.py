#!/usr/bin/env python3
"""Bulk ingest files and URLs into the knowledge service.

Usage:
    uv run python scripts/bulk_ingest.py ./documents/
    uv run python scripts/bulk_ingest.py --urls urls.txt
    uv run python scripts/bulk_ingest.py ./documents/ --urls urls.txt --tags health,research
"""

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md", ".json", ".csv"}

CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
    ".md": "text/plain",
    ".json": "application/json",
    ".csv": "text/csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk ingest files and URLs into the knowledge service."
    )
    parser.add_argument("path", nargs="?", help="Directory to scan recursively for files")
    parser.add_argument("--urls", metavar="FILE", help="Text file with one URL per line")
    parser.add_argument(
        "--server",
        default=os.getenv("KNOWLEDGE_URL", "http://localhost:8000"),
        help="Target server URL (default: KNOWLEDGE_URL env or http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("KNOWLEDGE_API_KEY", ""),
        help="API key (default: KNOWLEDGE_API_KEY env)",
    )
    parser.add_argument("--tags", default="", help="Comma-separated tags applied to all items")
    parser.add_argument("--domains", default="", help="Comma-separated domain hints")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be ingested, don't do it"
    )
    parser.add_argument(
        "--poll-timeout", type=int, default=300, help="Seconds to wait per job (default: 300)"
    )
    args = parser.parse_args()
    if not args.path and not args.urls:
        parser.error("At least one of PATH or --urls is required")
    return args


def discover_files(directory: str) -> list[Path]:
    """Recursively find supported files, sorted by name."""
    root = Path(directory)
    if not root.is_dir():
        print(f"ERROR: {directory} is not a directory")
        sys.exit(1)
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return files


def load_urls(filepath: str) -> list[str]:
    """Load URLs from a text file, skipping blank lines and comments."""
    path = Path(filepath)
    if not path.is_file():
        print(f"ERROR: {filepath} is not a file")
        sys.exit(1)
    urls = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def poll_job(client: httpx.Client, content_id: str, timeout: int) -> dict:
    """Poll job status until completed, failed, or timeout."""
    elapsed = 0
    while elapsed < timeout:
        resp = client.get(f"/api/content/{content_id}/status")
        if resp.status_code == 200:
            status = resp.json()
            if status["status"] in ("completed", "failed"):
                return status
        print(".", end="", flush=True)
        time.sleep(5)
        elapsed += 5
    return {"status": "timeout"}


def ingest_file(
    client: httpx.Client, filepath: Path, tags: str, domains: str, poll_timeout: int
) -> tuple[str, str]:
    """Upload a file. Returns (status, detail)."""
    ext = filepath.suffix.lower()
    content_type = CONTENT_TYPES.get(ext, "application/octet-stream")

    data: dict[str, str] = {}
    if tags:
        data["tags"] = tags
    if domains:
        data["domains"] = domains

    with open(filepath, "rb") as f:
        resp = client.post(
            "/api/content/upload",
            files={"file": (filepath.name, f, content_type)},
            data=data,
        )

    if resp.status_code == 409:
        return "SKIP", "already ingesting"
    if resp.status_code != 202:
        return "FAIL", f"HTTP {resp.status_code}: {resp.text[:100]}"

    content_id = resp.json()["content_id"]
    result = poll_job(client, content_id, poll_timeout)
    print(" ", end="")

    if result["status"] == "completed":
        return "OK", f"triples={result.get('triples_created', 0)}"
    elif result["status"] == "timeout":
        return "FAIL", "poll timeout"
    else:
        return "FAIL", result.get("error", "unknown error")


def ingest_url(
    client: httpx.Client, url: str, tags: str, domains: str, poll_timeout: int
) -> tuple[str, str]:
    """Submit a URL for ingestion. Returns (status, detail)."""
    body: dict = {"url": url}
    if tags:
        body["tags"] = [t.strip() for t in tags.split(",")]
    if domains:
        body["domains"] = [d.strip() for d in domains.split(",")]

    resp = client.post("/api/content", json=body)

    if resp.status_code == 409:
        return "SKIP", "already ingesting"
    if resp.status_code != 202:
        return "FAIL", f"HTTP {resp.status_code}: {resp.text[:100]}"

    content_id = resp.json()["content_id"]
    result = poll_job(client, content_id, poll_timeout)
    print(" ", end="")

    if result["status"] == "completed":
        return "OK", f"triples={result.get('triples_created', 0)}"
    elif result["status"] == "timeout":
        return "FAIL", "poll timeout"
    else:
        return "FAIL", result.get("error", "unknown error")


def main():
    args = parse_args()

    # Build item list
    files: list[Path] = []
    urls: list[str] = []
    if args.path:
        files = discover_files(args.path)
    if args.urls:
        urls = load_urls(args.urls)

    total = len(files) + len(urls)
    if total == 0:
        print("Nothing to ingest.")
        sys.exit(0)

    print(f"\nBulk ingest: {total} items ({len(files)} files, {len(urls)} URLs)")
    print(f"Target: {args.server}")
    if args.dry_run:
        print("\n[DRY RUN]\n")
        for i, f in enumerate(files, 1):
            print(f"  [{i}/{total}] {f}")
        for i, u in enumerate(urls, len(files) + 1):
            print(f"  [{i}/{total}] {u}")
        sys.exit(0)

    print()

    client = httpx.Client(
        base_url=args.server,
        timeout=60,
        headers={"X-API-Key": args.api_key} if args.api_key else {},
    )

    passed = 0
    failed = 0
    skipped = 0
    start_total = time.time()

    # Process files
    for i, filepath in enumerate(files, 1):
        label = str(filepath)
        print(f"  [{i}/{total}] {label} ", end="", flush=True)
        start = time.time()
        try:
            status, detail = ingest_file(
                client, filepath, args.tags, args.domains, args.poll_timeout
            )
        except httpx.ConnectError:
            print("FAIL  connection refused")
            print(f"\nERROR: Cannot connect to {args.server}")
            sys.exit(1)
        elapsed = time.time() - start
        print(f"{status}  {detail}  {elapsed:.0f}s")
        if status == "OK":
            passed += 1
        elif status == "SKIP":
            skipped += 1
        else:
            failed += 1

    # Process URLs
    for i, url in enumerate(urls, len(files) + 1):
        print(f"  [{i}/{total}] {url} ", end="", flush=True)
        start = time.time()
        try:
            status, detail = ingest_url(client, url, args.tags, args.domains, args.poll_timeout)
        except httpx.ConnectError:
            print("FAIL  connection refused")
            print(f"\nERROR: Cannot connect to {args.server}")
            sys.exit(1)
        elapsed = time.time() - start
        print(f"{status}  {detail}  {elapsed:.0f}s")
        if status == "OK":
            passed += 1
        elif status == "SKIP":
            skipped += 1
        else:
            failed += 1

    total_time = time.time() - start_total
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    print(
        f"\nResults: {passed} passed, {failed} failed, {skipped} skipped (total: {minutes}m {seconds}s)"
    )

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
