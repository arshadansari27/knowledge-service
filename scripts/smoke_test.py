#!/usr/bin/env python3
"""Production smoke test for knowledge-service.

Verifies the full pipeline works: health check, content ingestion (raw text,
HTML upload, PDF upload), knowledge graph query, and RAG question answering.

Usage:
    KNOWLEDGE_URL=https://knowledge.hikmahtech.in KNOWLEDGE_API_KEY=your-key \
        uv run python scripts/smoke_test.py
"""

import os
import sys
import time
from pathlib import Path

import httpx

BASE_URL = os.getenv("KNOWLEDGE_URL", "https://knowledge.hikmahtech.in")
API_KEY = os.getenv("KNOWLEDGE_API_KEY", "")
FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"

client = httpx.Client(base_url=BASE_URL, timeout=60, headers={"X-API-Key": API_KEY})

results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def poll_job(content_id: str, timeout: int = 180) -> dict:
    elapsed = 0
    while elapsed < timeout:
        resp = client.get(f"/api/content/{content_id}/status")
        if resp.status_code == 200:
            status = resp.json()
            if status["status"] in ("completed", "failed"):
                return status
        time.sleep(5)
        elapsed += 5
    return {"status": "timeout"}


def main():
    print(f"\nSmoke test against {BASE_URL}\n")

    # 1. Health check
    print("1. Health check")
    resp = client.get("/health")
    check(
        "GET /health",
        resp.status_code == 200 and resp.json().get("status") == "ok",
        resp.text[:100],
    )

    # 2. Get baseline stats
    print("\n2. Baseline stats")
    resp = client.get("/api/admin/stats/counts")
    baseline = resp.json() if resp.status_code == 200 else {}
    baseline_triples = baseline.get("triples", 0)
    check("GET /api/admin/stats/counts", resp.status_code == 200, f"triples={baseline_triples}")

    # 3. Ingest raw text (health domain)
    print("\n3. Ingest raw text (health domain)")
    resp = client.post(
        "/api/content",
        json={
            "url": f"smoke-test://health-{int(time.time())}",
            "title": "Vitamin D3 and Immune Function",
            "raw_text": (
                "Vitamin D3 (cholecalciferol) at a dose of 4000 IU daily improves immune "
                "function and reduces the risk of respiratory infections. A meta-analysis of "
                "25 randomized controlled trials with a combined sample size of 11,321 "
                "participants found that vitamin D supplementation reduces acute respiratory "
                "tract infections by 12%. Vitamin D3 is metabolized by the liver into "
                "25-hydroxyvitamin D, which is the biomarker measured in blood tests. "
                "Magnesium is required for vitamin D activation — vitamin D depletes magnesium. "
                "Vitamin D3 absorbs better with dietary fat."
            ),
            "source_type": "article",
        },
    )
    check("POST /api/content (health)", resp.status_code == 202, resp.text[:100])
    if resp.status_code == 202:
        cid = resp.json()["content_id"]
        status = poll_job(cid)
        check(
            "Ingestion completed",
            status["status"] == "completed",
            f"triples={status.get('triples_created', 0)}",
        )

    # 4. Ingest raw text (technology domain)
    print("\n4. Ingest raw text (technology domain)")
    resp = client.post(
        "/api/content",
        json={
            "url": f"smoke-test://tech-{int(time.time())}",
            "title": "PostgreSQL vs MySQL for Vector Search",
            "raw_text": (
                "PostgreSQL 16 with the pgvector extension provides native vector similarity "
                "search using HNSW indexes. PostgreSQL integrates with Python via asyncpg and "
                "psycopg2. pgvector is compatible with PostgreSQL 12 through 17. MySQL does "
                "not natively support vector search — it requires external tools like Milvus. "
                "PostgreSQL is licensed as PostgreSQL License (permissive open source). "
                "For AI applications, PostgreSQL with pgvector performs better than MySQL "
                "for hybrid search combining BM25 full-text and vector similarity."
            ),
            "source_type": "article",
        },
    )
    check("POST /api/content (tech)", resp.status_code == 202, resp.text[:100])
    if resp.status_code == 202:
        cid = resp.json()["content_id"]
        status = poll_job(cid)
        check(
            "Ingestion completed",
            status["status"] == "completed",
            f"triples={status.get('triples_created', 0)}",
        )

    # 5. Upload HTML file
    print("\n5. Upload HTML file")
    html_path = FIXTURES_DIR / "sample.html"
    if html_path.exists():
        with open(html_path, "rb") as f:
            resp = client.post(
                "/api/content/upload",
                files={"file": ("sample.html", f, "text/html")},
                data={"title": "Smoke Test HTML", "source_type": "article"},
            )
        check("POST /api/content/upload (HTML)", resp.status_code == 202, resp.text[:100])
        if resp.status_code == 202:
            cid = resp.json()["content_id"]
            status = poll_job(cid)
            check(
                "HTML ingestion completed",
                status["status"] == "completed",
                f"triples={status.get('triples_created', 0)}",
            )
    else:
        check("HTML fixture exists", False, f"Not found: {html_path}")

    # 6. Upload PDF file
    print("\n6. Upload PDF file")
    pdf_path = FIXTURES_DIR / "sample.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            resp = client.post(
                "/api/content/upload",
                files={"file": ("sample.pdf", f, "application/pdf")},
                data={"title": "Smoke Test PDF", "source_type": "paper"},
            )
        check("POST /api/content/upload (PDF)", resp.status_code == 202, resp.text[:100])
        if resp.status_code == 202:
            cid = resp.json()["content_id"]
            status = poll_job(cid)
            check(
                "PDF ingestion completed",
                status["status"] == "completed",
                f"triples={status.get('triples_created', 0)}",
            )
    else:
        check("PDF fixture exists", False, f"Not found: {pdf_path}")

    # 7. Check stats increased
    print("\n7. Verify stats increased")
    resp = client.get("/api/admin/stats/counts")
    if resp.status_code == 200:
        final = resp.json()
        final_triples = final.get("triples", 0)
        check(
            "Triples increased",
            final_triples > baseline_triples,
            f"{baseline_triples} -> {final_triples}",
        )
    else:
        check("Stats endpoint", False, resp.text[:100])

    # 8. Ask a question
    print("\n8. Ask a question (RAG)")
    resp = client.post(
        "/api/ask",
        json={
            "question": "What is the recommended dose of Vitamin D3?",
            "max_sources": 3,
        },
    )
    if resp.status_code == 200:
        answer = resp.json()
        check("POST /api/ask", True, f"answer length={len(answer.get('answer', ''))}")
    else:
        check("POST /api/ask", False, f"status={resp.status_code}")

    # Summary
    passed = sum(1 for _, p, _ in results if p)
    failed = sum(1 for _, p, _ in results if not p)
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} checks")
    print(f"{'=' * 50}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: Set KNOWLEDGE_API_KEY environment variable")
        sys.exit(1)
    main()
